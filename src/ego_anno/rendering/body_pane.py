"""Upper-body skeleton pane: a stick figure reconstructed from the head camera and the hands.

The result carries the head-camera pose and the 3D hand joints, nothing else of the
body. This pane infers a plausible upper body from them, posture-agnostic:

* the head IS the camera: the figure's head joint is pinned to the camera centre and
  drawn as the camera wireframe box with its RGB triad;
* the wrists are pinned to the annotated hand wrists (21-joint index 0); two-bone arm IK
  places the elbows, scapular protraction closes short reaches, and an arm that cannot
  reach shows a dashed reach line instead of a stretched bone;
* the head pitch is split into a trunk lean (the smallest lean that lets the arms reach,
  with a 12 % baseline), lower-cervical flexion and a head nod about C1;
* the pelvis hangs a fixed trunk length below the shoulders along world-up. Every bone
  has a constant anthropometric length (stature 1.75 m) and the trunk heading, lean,
  protraction and elbows are low-passed in time.

World-up is passed in by the caller (the follow-view pane's estimate or the user's
``--world-up``), so both 3D panes stand on the same gravity. No depth correction is
applied: the product JSON is authoritative. The observer is a single static
front-oblique camera solved once for the whole clip. numpy + OpenCV only.

Coordinates: the *product world* is the result's world frame (first camera frame).
The *display world* is right-handed with z = world-up, y = the clip's mean horizontal
gaze and the median head height at z = HEAD_HEIGHT_M (an assumed EgoDex median; the
result carries no floor), so the grid's Z labels are nominal heights.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# --- anthropometry, fractions of stature (Drillis & Contini / Winter) -----------
STATURE_M = 1.75
UPPER_ARM = 0.186 * STATURE_M
FOREARM = 0.146 * STATURE_M
ARM = UPPER_ARM + FOREARM
SHOULDER_HALF = 0.106 * STATURE_M
HIP_HALF = 0.048 * STATURE_M
NECK_LEN = 0.118 * STATURE_M          # C7 -> C1, upright
SPINE_LEN = 0.290 * STATURE_M         # hip centre -> shoulder centre, upright
PROTRACT_MAX = 0.070 * STATURE_M      # scapular protraction toward a far target
HEAD_PIVOT = np.asarray((0.0, 0.075, -0.085))  # C1 in camera coords (x right, y down, z gaze)
SPINE = (0.11, 0.13, 0.13, 0.13)      # pelvis -> shoulder centre segments, lumbar first
LEAN_MAX = math.radians(15.0)         # thoracic lean cap
NECK_MAX = math.radians(45.0)         # lower-cervical flexion cap
REACH = 0.9995 * ARM                  # farther than this the arm is drawn straight
HEAD_HEIGHT_M = 1.12                  # EgoDex median head height (display z of the median head)
REFERENCE_FPS = 30.0                  # smoothing sigmas below are in frames at this rate

UPZ = np.asarray((0.0, 0.0, 1.0))

HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
              (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))

BONES = (("head", "neck_top"), ("neck_top", "c7"), ("c7", "sh_c"), ("sh_c", "spine0"),
         ("spine0", "spine1"), ("spine1", "spine2"), ("spine2", "pelvis"),
         ("sh_c", "shoulder_left"), ("sh_c", "shoulder_right"),
         ("shoulder_left", "elbow_left"), ("elbow_left", "wrist_left"),
         ("shoulder_right", "elbow_right"), ("elbow_right", "wrist_right"),
         ("pelvis", "hip_left"), ("pelvis", "hip_right"))
SIDES = ("left", "right")             # renderer side 0 = 手0 = left, 1 = 手1 = right


def _bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


C_BONE = _bgr((77, 125, 214))
C_JOINT = _bgr((240, 110, 50))
C_CAM = _bgr((100, 112, 128))
C_GAZE = _bgr((235, 60, 30))
C_GRID = _bgr((44, 44, 44))
C_SPINE = (150, 150, 150)
C_LABEL = (200, 200, 200)
C_AXIS_LABEL = (210, 210, 210)
C_HAND = _bgr((88, 132, 220))
C_HAND_J = _bgr((230, 140, 120))
C_HAND_MESH = _bgr((40, 66, 120))
C_REACH = _bgr((90, 120, 170))
C_COUNTER = (238, 238, 238)
AX = (_bgr((225, 60, 45)), _bgr((60, 200, 80)), _bgr((65, 120, 240)))
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Reference pane: 580 x 914 px. Reserves and font sizes scale with the pane.
REF_PANE = (580.0, 914.0)
SS = 2                                # supersampling factor


# ---------------------------------------------------------------------------
# vector helpers
# ---------------------------------------------------------------------------
def nrm(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def smooth(x, sigma):
    """Zero-phase Gaussian low-pass along axis 0, edge padded (sigma in frames)."""
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, pad)


def two_bone_ik(root, target, l1, l2, pole):
    """(mid, end) of a two-bone chain; end reaches target when it is within REACH."""
    v = target - root
    d = float(np.linalg.norm(v))
    u = nrm(v, (0.0, 0.0, 1.0))
    d_c = float(np.clip(d, abs(l1 - l2) + 1e-4, 0.9995 * (l1 + l2)))
    a = (l1 * l1 - l2 * l2 + d_c * d_c) / (2.0 * d_c)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))
    n = nrm(pole - np.dot(pole, u) * u, (0.0, -1.0, 0.0))
    return root + a * u + h * n, root + d_c * u


def head_frame_metrics(R):
    """Gaze pitch below horizontal (rad) and the horizontal body-forward direction."""
    gaze, down = R[:, 2], R[:, 1]
    pitch = math.asin(float(np.clip(-gaze[2], -1, 1)))
    f = gaze - down
    f = np.asarray((f[0], f[1], 0.0))
    return pitch, nrm(f, (0.0, 1.0, 0.0))


def hand_axes(joints):
    """Wrist frame from the 21 joints: x toward the middle MCP, z the palm normal."""
    x = nrm(joints[9] - joints[0])
    z = nrm(np.cross(joints[5] - joints[0], joints[17] - joints[0]))
    return np.column_stack([x, np.cross(z, x), z])


def _fill_nearest(valid):
    """Index of the nearest valid frame for every frame (valid must have a True)."""
    idx = np.flatnonzero(valid)
    pos = np.arange(len(valid))
    right = idx[np.clip(np.searchsorted(idx, pos), 0, len(idx) - 1)]
    left = idx[np.clip(np.searchsorted(idx, pos, side="right") - 1, 0, len(idx) - 1)]
    return np.where(np.abs(right - pos) < np.abs(pos - left), right, left)


def display_frame(poses, pose_valid, up):
    """Product world -> display world: p_disp = p @ M.T + offset (z = up)."""
    up = nrm(up, (0.0, -1.0, 0.0))
    poses = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(pose_valid, dtype=bool)
    fw = poses[valid, :3, 2].mean(axis=0) if valid.any() else np.asarray((0.0, 0.0, 1.0))
    fw = fw - up * float(fw @ up)
    if np.linalg.norm(fw) < 1e-6:
        seed = np.asarray((0.0, 0.0, 1.0)) if abs(up[2]) < 0.9 else np.asarray((1.0, 0.0, 0.0))
        fw = seed - up * float(seed @ up)
    fw = nrm(fw)
    M = np.stack([np.cross(fw, up), fw, up])
    centre = np.median(poses[valid, :3, 3], axis=0) if valid.any() else np.zeros(3)
    offset = np.asarray((0.0, 0.0, HEAD_HEIGHT_M)) - M @ centre
    return M, offset


# ---------------------------------------------------------------------------
# body reconstruction (display world)
# ---------------------------------------------------------------------------
class UpperBody:
    """Per-frame upper-body joints from head poses and wrist targets.

    centres (n, 3) and rotations (n, 3, 3) are the camera poses in the display world;
    wrists maps "left"/"right" to (n, 3) annotated wrists with NaN rows where the hand
    is not annotated. A missing hand holds the arm's last pose (the first annotated
    pose before the first annotation); a side never annotated hangs relaxed.
    """

    def __init__(self, centres, rotations, wrists, fps=REFERENCE_FPS):
        k = float(fps) / REFERENCE_FPS if fps and np.isfinite(fps) and fps > 0 else 1.0
        self.n = n = len(centres)
        self.C = np.asarray(centres, dtype=np.float64)
        self.R = np.asarray(rotations, dtype=np.float64)
        self.wrist, self.valid = {}, {}
        for side in SIDES:
            w = np.asarray(wrists[side], dtype=np.float64)
            ok = np.isfinite(w).all(axis=1)
            self.valid[side] = ok
            if not ok.any():
                self.wrist[side] = None
                continue
            held, last = np.empty_like(w), w[np.flatnonzero(ok)[0]]
            for i in range(n):
                if ok[i]:
                    last = w[i]
                held[i] = last
            self.wrist[side] = held
        metrics = [head_frame_metrics(R) for R in self.R]
        self.pitch = np.asarray([m[0] for m in metrics])
        fwd = smooth(np.asarray([m[1] for m in metrics]), 8.0 * k)
        fwd[:, 2] = 0.0
        self.fwd = np.asarray([nrm(v, (0.0, 1.0, 0.0)) for v in fwd])
        lean0 = np.clip(0.12 * self.pitch, 0.0, LEAN_MAX)
        self.lean = np.clip(smooth(np.asarray([max(lean0[i], self._solve_lean(i)) for i in range(n)]),
                                   4.0 * k), 0.0, LEAN_MAX)
        self.pro = None
        raw = [self._frame(i)["pro"] for i in range(n)]
        radius = max(1, int(round(4 * k)))
        self.pro = {}
        for side in SIDES:
            r = np.asarray([x[side] for x in raw])
            dil = np.asarray([r[max(0, i - radius):i + radius + 1].max() for i in range(n)])
            self.pro[side] = np.minimum(smooth(dil, 3.0 * k), PROTRACT_MAX)
        self.J = [self._frame(i) for i in range(n)]
        self._smooth_elbows(1.5 * k)

    def _pose(self, i, lean):
        """Torso chain, built downward from the pinned head so the lean never moves it."""
        C, R, f = self.C[i], self.R[i], self.fwd[i]
        lat = nrm(np.cross(UPZ, f))
        H = C + R @ HEAD_PIVOT                                   # C1
        neck_flex = float(np.clip(0.5 * (self.pitch[i] - lean), 0.0, NECK_MAX))
        n_neck = math.cos(lean + neck_flex) * UPZ + math.sin(lean + neck_flex) * f
        c7 = H - NECK_LEN * n_neck
        d_tr = math.cos(lean) * UPZ + math.sin(lean) * f          # pelvis -> shoulder axis
        shoulder = c7 - 0.03 * d_tr
        # Spine segments of fixed length, lumbar more upright than thoracic, hung from
        # the shoulders: the pelvis is wherever the chain ends, so no bone stretches.
        total, cum, steps = sum(SPINE), 0.0, []
        for L in SPINE:
            cum += L
            a = lean * (0.45 + 0.55 * cum / total)
            steps.append((L / total) * SPINE_LEN * (math.cos(a) * UPZ + math.sin(a) * f))
        pelvis = shoulder - np.sum(steps, axis=0)
        # spine0 is the vertebra under the shoulders, spine2 the one above the pelvis
        spine = list(pelvis + np.cumsum(steps, axis=0)[:-1])[::-1]
        return {"head": C, "neck_top": H, "c7": c7, "sh_c": shoulder, "spine": spine,
                "pelvis": pelvis, "hip_left": pelvis + HIP_HALF * lat,
                "hip_right": pelvis - HIP_HALF * lat, "gap": {}, "pro": {"left": 0.0, "right": 0.0},
                "lean": lean}

    def _rest_shoulder(self, S, lat, d_tr, sgn):
        return S + SHOULDER_HALF * nrm(sgn * lat - 0.10 * d_tr)

    def _frame(self, i):
        f = self.fwd[i]
        lat = nrm(np.cross(UPZ, f))
        lean = self.lean[i]
        d_tr = math.cos(lean) * UPZ + math.sin(lean) * f
        j = self._pose(i, lean)
        S = j["sh_c"]
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            sh0 = self._rest_shoulder(S, lat, d_tr, sgn)
            W = None if self.wrist[side] is None else self.wrist[side][i]
            if W is None:
                d = nrm(0.3 * f - UPZ + 0.1 * sgn * lat)
                j["shoulder_" + side], j["elbow_" + side] = sh0, sh0 + UPPER_ARM * d
                j["wrist_" + side] = sh0 + ARM * d
                continue
            over = float(np.linalg.norm(W - sh0)) - 0.97 * ARM
            pro = (float(np.clip(over / 0.035, 0.0, 1.0)) * PROTRACT_MAX if self.pro is None
                   else float(self.pro[side][i]))
            j["pro"][side] = pro
            clav = sh0 - S
            sh = S + float(np.linalg.norm(clav)) * nrm(clav + pro * nrm(W - S))
            pole = nrm(sgn * 0.8 * lat - 0.7 * UPZ - 0.25 * f)
            E, Wr = two_bone_ik(sh, W, UPPER_ARM, FOREARM, pole)
            j["shoulder_" + side], j["elbow_" + side], j["wrist_" + side] = sh, E, Wr
            j["gap"][side] = float(np.linalg.norm(Wr - W))
        return j

    def _solve_lean(self, i):
        """Smallest trunk lean (<= LEAN_MAX) that lets both arms reach their wrists."""
        targets = [(sgn, self.wrist[side][i]) for side, sgn in (("left", 1.0), ("right", -1.0))
                   if self.wrist[side] is not None]
        if not targets:
            return math.radians(4.0)
        f = self.fwd[i]
        lat = nrm(np.cross(UPZ, f))
        for step in range(31):
            lean = LEAN_MAX * step / 30.0
            S = self._pose(i, lean)["sh_c"]
            d_tr = math.cos(lean) * UPZ + math.sin(lean) * f
            if all(np.linalg.norm(w - self._rest_shoulder(S, lat, d_tr, sgn)) <= 0.97 * ARM + PROTRACT_MAX
                   for sgn, w in targets):
                return lean
        return LEAN_MAX

    def _smooth_elbows(self, sigma):
        """Low-pass the elbow, then put it back on the feasible circle.

        Near full extension IK amplifies wrist jitter into elbow swings. The smoothed
        elbow is projected onto the circle of elbows that keep both bone lengths AND
        the wrist on its target, so smoothing never opens a wrist gap; only a target
        beyond REACH leaves the arm straight and short of it.
        """
        for side in SIDES:
            if self.wrist[side] is None:
                continue
            E = smooth(np.asarray([j["elbow_" + side] for j in self.J]), sigma)
            for i, j in enumerate(self.J):
                sh, W = j["shoulder_" + side], self.wrist[side][i]
                v = W - sh
                d = float(np.linalg.norm(v))
                if abs(UPPER_ARM - FOREARM) + 1e-4 < d <= REACH:
                    u = v / d
                    a = (UPPER_ARM ** 2 - FOREARM ** 2 + d * d) / (2.0 * d)
                    h = math.sqrt(max(0.0, UPPER_ARM ** 2 - a * a))
                    c = sh + a * u
                    radial = E[i] - c
                    radial = nrm(radial - u * float(radial @ u), j["elbow_" + side] - c)
                    e = c + h * radial
                else:
                    e = sh + UPPER_ARM * nrm(E[i] - sh)
                j["elbow_" + side] = e
                j["wrist_" + side] = e + FOREARM * nrm(W - e)
                j["gap"][side] = float(np.linalg.norm(j["wrist_" + side] - W))

    def named(self, i):
        j = self.J[i]
        out = {k: v for k, v in j.items() if k not in ("spine", "gap", "pro", "lean")}
        for s, p in enumerate(j["spine"]):
            out["spine%d" % s] = p
        return out


# ---------------------------------------------------------------------------
# static observer and raster
# ---------------------------------------------------------------------------
class _View:
    def __init__(self, w, h, eye, target, fov_deg):
        fz = nrm(target - eye)
        rx = nrm(np.cross(fz, UPZ))
        uy = np.cross(rx, fz)
        self.Rv = np.stack([rx, uy, fz])
        self.t = -self.Rv @ eye
        self.eye = np.asarray(eye, dtype=np.float64)
        self.w, self.h = w, h
        self.f = 0.5 * h / math.tan(0.5 * math.radians(fov_deg))
        self.shift = np.zeros(2)

    def project(self, pts):
        c = np.asarray(pts, dtype=np.float64).reshape(-1, 3) @ self.Rv.T + self.t
        z = np.maximum(c[:, 2], 1e-6)
        return np.stack([0.5 * self.w + self.shift[0] + self.f * c[:, 0] / z,
                         0.5 * self.h + self.shift[1] - self.f * c[:, 1] / z], 1), c[:, 2]

    def matrix(self):
        """(4, 4) view matrix with the principal-point shift folded into row 3."""
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = self.Rv, self.t
        out[3, :2] = self.shift
        return out


def _ip(p):
    return (int(round(float(p[0]))), int(round(float(p[1]))))


class _Raster:
    """Draws on a supersampled canvas; text is placed in final-pane pixels times SS."""

    def __init__(self, width, height, box_lo, box_hi, centre, facing, azim_deg=40.0, elev_deg=22.0):
        self.W, self.H = int(width), int(height)
        self.s = min(self.W / REF_PANE[0], self.H / REF_PANE[1])
        ss = SS
        self.lo, self.hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
        f = nrm(np.asarray((facing[0], facing[1], 0.0)), (0.0, 1.0, 0.0))
        right = -nrm(np.cross(UPZ, f))
        azim, elev = math.radians(azim_deg), math.radians(elev_deg)
        d = math.cos(elev) * (math.cos(azim) * f + math.sin(azim) * right) + math.sin(elev) * UPZ
        lo = self.lo - np.asarray((0.05, 0.05, 0.0))
        hi = self.hi + np.asarray((0.0, 0.0, 0.15))
        corners = np.asarray([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        x0, x1 = 50 * self.s * ss, (self.W - 20 * self.s) * ss
        y0, y1 = 120 * self.s * ss, (self.H - 10 * self.s) * ss
        c = np.asarray(centre, float)
        dist = 2.0
        for _ in range(200):
            view = _View(self.W * ss, self.H * ss, c + dist * d, c, 30.0)
            uv = view.project(corners)[0]
            if np.ptp(uv[:, 0]) < (x1 - x0) and np.ptp(uv[:, 1]) < (y1 - y0):
                break
            dist *= 1.03
        view.shift = np.asarray((0.5 * (x0 + x1) - 0.5 * (uv[:, 0].min() + uv[:, 0].max()),
                                 0.5 * (y0 + y1) - 0.5 * (uv[:, 1].min() + uv[:, 1].max())))
        self.view = view
        self.bg = self._background()

    # -- primitives ------------------------------------------------------------
    def P(self, pts):
        return self.view.project(pts)[0]

    def th(self, px):
        return max(1, int(round(px * self.s * SS)))

    def seg(self, img, a, b, col, th):
        uv = self.P(np.asarray([a, b]))
        cv2.line(img, _ip(uv[0]), _ip(uv[1]), col, th, cv2.LINE_AA)

    def text(self, img, u, v, s, col, size, bold=False, anchor="la"):
        """Hershey text; `size` is the reference-pane pixel size of the approved labels."""
        scale = size * self.s * SS / 30.0
        thick = max(1, int(round((2.0 if bold else 1.0) * self.s * SS * (size / 14.0 if bold else 0.75))))
        (tw, tht), _ = cv2.getTextSize(s, FONT, scale, thick)
        x = u - {"l": 0.0, "m": 0.5 * tw, "r": float(tw)}[anchor[0]]
        y = v + {"a": float(tht), "m": 0.5 * tht, "s": 0.0}[anchor[1]]
        cv2.putText(img, s, _ip((x, y)), FONT, scale, col, thick, cv2.LINE_AA)

    # -- static background ----------------------------------------------------------
    def _background(self):
        ss = SS
        img = np.zeros((self.H * ss, self.W * ss, 3), np.uint8)
        lo, hi = self.lo, self.hi
        eye = self.view.eye
        xb = hi[0] if eye[0] < 0.5 * (lo[0] + hi[0]) else lo[0]      # far x wall
        yb = lo[1] if eye[1] > 0.5 * (lo[1] + hi[1]) else hi[1]      # far y wall
        xn, yn = lo[0] + hi[0] - xb, lo[1] + hi[1] - yb
        xs = np.arange(math.ceil(lo[0] / 0.25) * 0.25, hi[0] + 1e-9, 0.25)
        ys = np.arange(math.ceil(lo[1] / 0.25) * 0.25, hi[1] + 1e-9, 0.25)
        z0 = lo[2]
        zs = np.arange(z0, hi[2] + 1e-9, 0.2)
        th = max(1, ss // 2)
        for x in xs:
            self.seg(img, (x, lo[1], z0), (x, hi[1], z0), C_GRID, th)
            self.seg(img, (x, yb, z0), (x, yb, hi[2]), C_GRID, th)
        for y in ys:
            self.seg(img, (lo[0], y, z0), (hi[0], y, z0), C_GRID, th)
            self.seg(img, (xb, y, z0), (xb, y, hi[2]), C_GRID, th)
        for z in zs:
            self.seg(img, (lo[0], yb, z), (hi[0], yb, z), C_GRID, th)
            self.seg(img, (xb, lo[1], z), (xb, hi[1], z), C_GRID, th)
        sp = self.th(1.0)
        self.seg(img, (lo[0], yn, z0), (hi[0], yn, z0), C_SPINE, sp)        # X axis
        self.seg(img, (xn, lo[1], z0), (xn, hi[1], z0), C_SPINE, sp)        # Y axis
        zx, zy = xn, yb
        self.seg(img, (zx, zy, z0), (zx, zy, hi[2]), C_SPINE, sp)           # Z axis
        tick = 0.035
        sx, sy = np.sign(yn - yb), np.sign(xn - xb)
        for x in xs[::2]:
            self.seg(img, (x, yn, z0), (x, yn + sx * tick, z0), C_SPINE, sp)
            uv = self.P([(x, yn + sx * 3.2 * tick, z0)])[0]
            self.text(img, uv[0], uv[1], "%.2f" % x, C_LABEL, 10, anchor="mm")
        for y in ys[::2]:
            self.seg(img, (xn, y, z0), (xn + sy * tick, y, z0), C_SPINE, sp)
            uv = self.P([(xn + sy * 3.2 * tick, y, z0)])[0]
            self.text(img, uv[0], uv[1], "%.2f" % y, C_LABEL, 10, anchor="mm")
        for z in zs:
            uv = self.P([(zx, zy, z)])[0]
            cv2.line(img, _ip(uv), _ip(uv - (10 * self.s * ss, 0)), C_SPINE, sp, cv2.LINE_AA)
            self.text(img, uv[0] - 14 * self.s * ss, uv[1], "%.2f" % z, C_LABEL, 10, anchor="rm")
        uv = self.P([(0.5 * (lo[0] + hi[0]), yn + sx * 9 * tick, z0)])[0]
        self.text(img, uv[0], uv[1], "X (m)", C_AXIS_LABEL, 13, anchor="mm")
        uv = self.P([(xn + sy * 9 * tick, 0.5 * (lo[1] + hi[1]), z0)])[0]
        self.text(img, uv[0], uv[1], "Y (m)", C_AXIS_LABEL, 13, anchor="mm")
        uv = self.P([(zx, zy, hi[2] + 0.08)])[0]
        self.text(img, uv[0], uv[1], "Z (m)", C_AXIS_LABEL, 13, anchor="mm")
        origin = np.asarray((zx + 0.12 * np.sign(xb - xn), zy + 0.12 * np.sign(yn - yb), z0))
        self.triad(img, origin, np.eye(3), 0.18, "", world=True)
        return img

    # -- dynamic objects ----------------------------------------------------------
    def triad(self, img, o, R, L, prefix, world=False):
        uv0 = self.P(o[None])[0]
        for k, name in enumerate("XYZ"):
            uv1 = self.P((o + L * R[:, k])[None])[0]
            if world:
                cv2.arrowedLine(img, _ip(uv0), _ip(uv1), AX[k], self.th(1.5), cv2.LINE_AA, tipLength=0.06)
            else:
                cv2.line(img, _ip(uv0), _ip(uv1), AX[k], self.th(1.6), cv2.LINE_AA)
            d = nrm(uv1 - uv0, (1.0, 0.0)) * 7 * self.s * SS
            self.text(img, uv1[0] + d[0], uv1[1] + d[1], prefix + name, AX[k], 10 if world else 9,
                      bold=not world, anchor="mm")

    def camera(self, img, C, R, s=0.06):
        hw, hh, dz = s, 0.75 * s, 1.3 * s
        back = np.asarray([[-hw, -hh, -0.5 * dz], [hw, -hh, -0.5 * dz], [hw, hh, -0.5 * dz], [-hw, hh, -0.5 * dz]])
        front = back * np.asarray((0.55, 0.55, 1.0)) + np.asarray((0.0, 0.0, dz))
        uv = self.P(C + np.vstack([back, front]) @ R.T)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)):
            cv2.line(img, _ip(uv[a]), _ip(uv[b]), C_CAM, self.th(1.7), cv2.LINE_AA)
        a, b = self.P(np.asarray([C, C + 0.20 * R[:, 2]]))
        cv2.arrowedLine(img, _ip(a), _ip(b), C_GAZE, self.th(2.2), cv2.LINE_AA, tipLength=0.3)
        self.triad(img, C, R, 0.08, "C-")

    def body(self, img, named):
        for a, b in BONES:
            self.seg(img, named[a], named[b], C_BONE, self.th(1.9))
        r = self.th(2.6)
        for k, p in named.items():
            if k != "head":
                cv2.circle(img, _ip(self.P(p[None])[0]), r, C_JOINT, -1, cv2.LINE_AA)

    def dashed(self, img, a, b, n=6):
        uv = self.P(np.asarray([a + (b - a) * k / (2 * n) for k in range(2 * n + 1)]))
        for k in range(0, 2 * n, 2):
            cv2.line(img, _ip(uv[k]), _ip(uv[k + 1]), C_REACH, self.th(1.0), cv2.LINE_AA)

    def hand(self, img, joints, verts=None, faces=None, prefix=""):
        if verts is not None and faces is not None:
            uv = self.P(verts)
            tri = np.rint(uv[np.asarray(faces)]).astype(np.int32)
            layer = img.copy()
            for t in tri:                       # one colour: painter order is irrelevant
                cv2.fillConvexPoly(layer, t, C_HAND_MESH, cv2.LINE_AA)
            cv2.addWeighted(layer, 0.75, img, 0.25, 0, img)
        uj = self.P(joints)
        for a, b in HAND_EDGES:
            cv2.line(img, _ip(uj[a]), _ip(uj[b]), C_HAND, self.th(1.3), cv2.LINE_AA)
        r = self.th(1.5)
        for p in uj:
            cv2.circle(img, _ip(p), r, C_HAND_J, -1, cv2.LINE_AA)
        self.triad(img, joints[0], hand_axes(joints), 0.07, prefix)

    def finish(self, img):
        return cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# public pane
# ---------------------------------------------------------------------------
class BodyPane:
    """Upper-body skeleton pane for a whole clip, drawn one frame at a time.

    poses (n, 4, 4) camera-to-world in the product world, pose_valid (n,), joints a
    pair of (n, 21, 3) arrays (left, right; NaN where absent), up the world-up vector in
    the product world (the follow-view pane's), size (width, height) in pixels. The
    body, the grid box and the observer are solved here once for the clip.
    """

    def __init__(self, poses, pose_valid, joints, up, fps, size):
        poses = np.asarray(poses, dtype=np.float64)
        valid = np.asarray(pose_valid, dtype=bool) & np.isfinite(poses).all(axis=(1, 2))
        self.n = len(poses)
        self.pose_valid = valid
        self.up = nrm(up, (0.0, -1.0, 0.0))
        self.M, self.offset = display_frame(poses, valid, self.up)
        self.width, self.height = int(size[0]), int(size[1])
        self.body = None
        if not valid.any():
            self.raster = _Raster(self.width, self.height, (-0.45, -0.45, 0.4), (0.45, 0.45, 1.4),
                                  (0.0, 0.0, 0.9), (0.0, 1.0, 0.0))
            return
        fill = _fill_nearest(valid)
        centres = self.p(poses[fill, :3, 3])
        rotations = np.einsum("ij,njk->nik", self.M, poses[fill, :3, :3])
        wrists = {}
        for side, name in enumerate(SIDES):
            w = np.asarray(joints[side], dtype=np.float64).reshape(self.n, 21, 3)[:, 0]
            ok = np.isfinite(w).all(axis=1) & valid
            wrists[name] = np.where(ok[:, None], self.p(np.nan_to_num(w)), np.nan)
        self.body = UpperBody(centres, rotations, wrists, fps)
        named = [self.body.named(i) for i in range(self.n)]
        pts = np.asarray([p for frame in named for p in frame.values()])
        ctr = np.median(pts, axis=0)
        cx, cy = round(ctr[0] * 4) / 4, round(ctr[1] * 4) / 4
        facing = nrm(self.body.fwd.mean(axis=0), (0.0, 1.0, 0.0))
        # Grid box: 0.9 m square around the upper body, widened (0.05 m steps) only as
        # far as the annotated hands reach, so a hand beyond the arm stays in view.
        hand_pts = [self.p(np.asarray(rows, dtype=np.float64).reshape(-1, 3)) for rows in joints]
        hand_pts = np.vstack([h[np.isfinite(h).all(axis=1)] for h in hand_pts] + [pts])
        hb = max(0.45, math.ceil((np.abs(hand_pts[:, :2] - (cx, cy)).max() + 0.05) / 0.05) * 0.05)
        zbot = math.floor((min(min(j["pelvis"][2] for j in self.body.J), hand_pts[:, 2].min()) - 0.15)
                          / 0.2) * 0.2
        ztop = math.ceil((max(j["head"][2] for j in self.body.J) + 0.25) / 0.2) * 0.2
        self.raster = _Raster(self.width, self.height, (cx - hb, cy - hb, zbot), (cx + hb, cy + hb, ztop),
                              (cx, cy, 0.5 * (zbot + ztop)), facing)

    def p(self, pts):
        """Product-world points -> display world."""
        return np.asarray(pts, dtype=np.float64) @ self.M.T + self.offset

    @property
    def view_matrix(self):
        return self.raster.view.matrix()

    def render(self, index, joints=(None, None), meshes=(None, None), faces=(None, None)):
        """BGR uint8 (height, width, 3) pane for frame `index` (0-based).

        joints: this frame's (21, 3) product-world joints per side (None/NaN: no hand);
        meshes/faces: the main renderer's MANO surface for that hand, when it built one.
        """
        ras = self.raster
        img = ras.bg.copy()
        s = ras.s
        if self.body is None or not self.pose_valid[index]:
            out = ras.finish(img)
            cv2.putText(out, "camera pose unavailable", _ip((18 * s, 100 * s)), FONT, 0.6 * s,
                        (0, 215, 255), max(1, int(round(2 * s))), cv2.LINE_AA)
            self._counter(out, index)
            return out
        named = self.body.named(index)
        ras.body(img, named)
        ras.camera(img, self.body.C[index], self.body.R[index])
        for side, name in enumerate(SIDES):
            J = joints[side]
            if J is None or np.asarray(J).shape != (21, 3) or not np.isfinite(J).all():
                continue
            Jd = self.p(J)
            if self.body.J[index]["gap"].get(name, 0.0) > 0.005:     # out of reach: show it
                ras.dashed(img, named["wrist_" + name], Jd[0])
            verts = self.p(meshes[side]) if meshes[side] is not None else None
            ras.hand(img, Jd, verts, faces[side], "L-" if side == 0 else "R-")
        out = ras.finish(img)
        self._counter(out, index)
        return out

    def _counter(self, img, index):
        s = self.raster.s
        cv2.putText(img, "frame %d" % (index + 1), _ip((18 * s, 84 * s)), FONT, 0.7 * s,
                    C_COUNTER, max(1, int(round(2 * s))), cv2.LINE_AA)
