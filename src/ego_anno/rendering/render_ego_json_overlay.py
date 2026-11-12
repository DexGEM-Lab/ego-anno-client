#!/usr/bin/env python3
"""Render the annotation overlay video from annotation JSON and the source video.

The default three-pane layout shows the hand overlay, the side-rear follow view
and the upper-body skeleton (body_pane.py); --layout two-pane keeps the first two.

Annotation v3 and Cosmos e2e JSON are supported. Hands with finite beta use
bundled MANO assets; hands without beta show their actual 21-joint skeleton.
No model assets are needed for skeleton-only inputs. Source projection requires
calibration; world skeletons and the camera trajectory do not.

Usage:
    python render_ego_json_overlay.py annotation.json --video source.mp4 -o overlay.mp4
    python render_ego_json_overlay.py annotation.json --video https://example.com/source.mp4 -o overlay.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional but recommended dependency
    Image = ImageDraw = ImageFont = None

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
# RRD constants are RGB; OpenCV rasterizes them as BGR.
HAND_COLORS = ((90, 225, 70), (40, 205, 255))  # left green, right yellow
KEYPOINT_COLOR = (40, 40, 255)                 # RRD KEYPOINT_RGBA red
SKIN_COLOR = (160, 205, 240)                   # RRD SKIN_RGBA [240, 205, 160]
CAMERA_COLOR = (22, 25, 28)                   # near-black cinema camera (BGR)
TRAIL_COLOR = (255, 180, 60)                   # RRD TRAIL_RGBA [60, 180, 255]
# Observer geometry relative to the smoothed horizontal heading: behind it,
# swung to one side by _EYE_AZIMUTH, raised _EYE_ELEVATION above the
# horizontal plane through the anchor (both in degrees, measured against the
# estimated world-up, never against the ego camera's own axes).
_EYE_AZIMUTH = 35.0
_EYE_ELEVATION = 32.0
# Standoff band and the 5 cm / 15 deg deadbands that turn head jitter into
# exactly zero observer motion.
_EYE_DISTANCE_RANGE = (0.7, 1.6)
_ANCHOR_DEADBAND_M = 0.05
_HEADING_DEADBAND_DEG = 15.0
_OBSERVER_MAX_SPEED = 0.6
# Assumed head pitch (deg, forward-down) when the clip's yaw spread cannot
# pin world-up: a typical measured value for head-mounted tabletop recordings.
_FALLBACK_HEAD_PITCH_DEG = 22.0
_FLOOR_DROP_M = 0.75
GRID_COLOR = (91, 89, 83)                     # sampled bright grid core, BGR
# Rerun's spatial viewport is a dark blue-green shaded floor rather than the
# neutral gray used by the old overlay.  Corners are sampled from the target
# view (stored here as BGR because OpenCV owns the raster).
BG = (24, 52, 24)
_SPATIAL_BG_CORNERS = np.asarray((
    ((16, 45, 30), (29, 44, 16)),
    ((20, 56, 32), (34, 55, 22)),
), dtype=np.float64)
AXIS_COLORS = {(0, 5): (50, 70, 235), (0, 6): (50, 220, 90), (0, 7): (235, 100, 55)}
OUTLINE = (3, 3, 3)

# MANO's 16 kinematic joints mapped to the JSON/OpenPose-style 21-joint order.
MANO16_TO_JSON21 = np.asarray((0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3), dtype=np.int64)
MANO_PARENTS = np.asarray((-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14), dtype=np.int64)
# Distal MANO joint -> (tip vertex, JSON tip joint).
DISTAL_TIPS = {3: (320, 8), 6: (443, 12), 9: (672, 20), 12: (555, 16), 15: (744, 4)}
FINGER_CHAINS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15))
PALM_MCP = np.asarray((1, 4, 7, 10, 13), dtype=np.int64)
# Parent of every JSON21 joint (wrist 0, then four joints per finger).
JSON21_PARENTS = np.asarray((-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19), dtype=np.int64)
SHAPE_BOUND = 3.0            # |beta| limit: PCA standard deviations of the hand shape space
SHAPE_PRIOR_MM2 = 1.0        # penalty per beta^2, in mm^2 of bone-length residual
SHAPE_MIN_FRAMES = 3         # fewer finite hand frames fall back to the mean shape


def _normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-10:
        return np.asarray(fallback, dtype=np.float64)
    return vector / norm


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Smallest 3D rotation that sends source direction to target direction."""
    a = _normalize(source, np.asarray([1.0, 0.0, 0.0]))
    b = _normalize(target, a)
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        axis_seed = np.asarray([1.0, 0.0, 0.0]) if abs(a[0]) < 0.8 else np.asarray([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(a, axis_seed), np.asarray([0.0, 0.0, 1.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    vx = np.asarray(((0.0, -cross[2], cross[1]), (cross[2], 0.0, -cross[0]), (-cross[1], cross[0], 0.0)))
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - cosine) / (sine * sine))


def _palm_rotation(rest_joints: np.ndarray, target_joints: np.ndarray) -> np.ndarray:
    rest = rest_joints[PALM_MCP] - rest_joints[0]
    target16 = target_joints[MANO16_TO_JSON21]
    target = target16[PALM_MCP] - target16[0]
    rest = np.asarray([_normalize(row, row) for row in rest])
    target = np.asarray([_normalize(row, row) for row in target])
    covariance = rest.T @ target
    u, _, vt = np.linalg.svd(covariance)
    q = u @ vt  # row-vector map: rest @ q ~= target
    if np.linalg.det(q) < 0.0:
        u[:, -1] *= -1.0
        q = u @ vt
    return q.T  # column-vector map


class _LegacyArrayStub:
    """Local placeholder for legacy chumpy wrappers in the hand pickles."""


class _CscMatrixStub:
    """numpy-only stand-in for the scipy ``csc_matrix`` stored in the hand pickles.

    The render extra ships numpy and OpenCV only. The standard hand assets keep
    ``J_regressor`` as a compressed-sparse-column matrix, so unpickling it with
    the real class would make scipy a hidden hard dependency of ``--mano-dir``.
    This stub accepts the pickled state and rebuilds the same dense array.
    """

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})

    def toarray(self) -> np.ndarray:
        shape = tuple(int(v) for v in (self.__dict__.get("_shape") or self.__dict__["shape"]))
        data = np.asarray(self.__dict__["data"], dtype=np.float64)
        indices = np.asarray(self.__dict__["indices"], dtype=np.int64)
        indptr = np.asarray(self.__dict__["indptr"], dtype=np.int64)
        dense = np.zeros(shape, dtype=np.float64)
        for column in range(shape[1]):
            start, end = indptr[column], indptr[column + 1]
            dense[indices[start:end], column] = data[start:end]
        return dense


class _HandModelUnpickler(pickle.Unpickler):
    """Load only the numpy/scipy globals needed by the standard hand assets."""

    _ALLOWED = {
        ("__builtin__", "set"),
        ("builtins", "set"),
        ("chumpy.ch", "Ch"),
        ("chumpy.reordering", "Select"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.random", "RandomState"),
        ("numpy.random", "MT19937"),
        ("scipy.sparse", "csc_matrix"),
        ("scipy.sparse.csc", "csc_matrix"),
        ("scipy.sparse._csc", "csc_matrix"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"unsupported pickle global in hand asset: {module}.{name}")
        if (module, name) in (("chumpy.ch", "Ch"), ("chumpy.reordering", "Select")):
            return _LegacyArrayStub
        if module.startswith("scipy.sparse"):
            return _CscMatrixStub
        return super().find_class(module, name)


def _rest_joints21_linear(model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Rest-pose JSON21 joints as base + dirs @ beta (metres).

    The 16 kinematic joints come from the regressor; the five tips are the
    vertices ``reconstruct`` also uses, so the fitted lengths are exactly the
    ones the posed surface will span.
    """
    template, shapedirs, regressor = model["v_template"], model["shapedirs"], model["J_regressor"]
    base = np.zeros((21, 3), dtype=np.float64)
    dirs = np.zeros((21, 3, shapedirs.shape[2]), dtype=np.float64)
    base[MANO16_TO_JSON21] = regressor @ template
    dirs[MANO16_TO_JSON21] = np.einsum("jv,vck->jck", regressor, shapedirs)
    for tip_vertex, tip_joint in DISTAL_TIPS.values():
        base[tip_joint] = template[tip_vertex]
        dirs[tip_joint] = shapedirs[tip_vertex]
    return base, dirs


def _bone_lengths21(joints: np.ndarray) -> np.ndarray:
    """Lengths of the 20 JSON21 bones (child joints 1..20), last axis of ``joints``."""
    joints = np.asarray(joints, dtype=np.float64)
    children = np.arange(1, 21)
    return np.linalg.norm(joints[..., children, :] - joints[..., JSON21_PARENTS[children], :], axis=-1)


def fit_shape_to_bone_lengths(model: dict[str, np.ndarray], target_lengths: np.ndarray,
                              prior: float = SHAPE_PRIOR_MM2, bound: float = SHAPE_BOUND,
                              iterations: int = 40) -> np.ndarray:
    """Least-squares beta whose rest-pose bone lengths match ``target_lengths`` (metres).

    Bone lengths are the pose-invariant part of a joint prediction, so this is
    the shape fit available when a result carries joints but no shape. Twenty
    lengths constrain ten components only weakly in the high-index directions:
    without the prior those run to the bound to remove sub-0.1 mm residual, so
    the Gauss-Newton solve is damped by ``prior`` mm^2 per beta^2 and projected
    onto |beta| <= bound.
    """
    base, dirs = _rest_joints21_linear(model)
    children = np.arange(1, 21)
    parents = JSON21_PARENTS[children]
    bone_base = (base[children] - base[parents]) * 1000.0
    bone_dirs = (dirs[children] - dirs[parents]) * 1000.0
    target = np.asarray(target_lengths, dtype=np.float64).reshape(20) * 1000.0
    count = bone_dirs.shape[2]
    beta = np.zeros(count, dtype=np.float64)
    for _ in range(iterations):
        bones = bone_base + bone_dirs @ beta
        lengths = np.maximum(np.linalg.norm(bones, axis=1), 1e-9)
        jacobian = np.einsum("bc,bck->bk", bones, bone_dirs) / lengths[:, None]
        hessian = jacobian.T @ jacobian + prior * np.eye(count)
        gradient = jacobian.T @ (lengths - target) + prior * beta
        updated = np.clip(beta - np.linalg.solve(hessian, gradient), -bound, bound)
        converged = np.max(np.abs(updated - beta)) < 1e-9
        beta = updated
        if converged:
            break
    return beta


def _load_standard_hand_model(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"missing bundled hand-model asset: {path}")
    with path.open("rb") as stream:
        raw = _HandModelUnpickler(stream, encoding="latin1").load()
    wrapped = raw["shapedirs"]
    shapedirs = np.asarray(wrapped.a.x, dtype=np.float64).reshape(-1)[wrapped.idxs].reshape(wrapped.preferred_shape)
    regressor = raw["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    return {
        "v_template": np.asarray(raw["v_template"], dtype=np.float64),
        "shapedirs": shapedirs,
        "J_regressor": np.asarray(regressor, dtype=np.float64),
        "weights": np.asarray(raw["weights"], dtype=np.float64),
        "faces": np.asarray(raw["f"], dtype=np.int32),
    }


class HandSurfaceAssets:
    """Runtime standard-hand assets matching the RRD renderer's loader."""

    def __init__(self, asset_dir: Path):
        standard = (asset_dir / "HAND_LEFT.pkl", asset_dir / "HAND_RIGHT.pkl")
        legacy = (asset_dir / "MANO_LEFT.pkl", asset_dir / "MANO_RIGHT.pkl", asset_dir / "mano_runtime.npz")
        if all(path.is_file() for path in standard):
            self.models = {
                "left": _load_standard_hand_model(standard[0]),
                "right": _load_standard_hand_model(standard[1]),
            }
        elif all(path.is_file() for path in legacy):
            archive = np.load(legacy[2])
            self.models = {
                side: {
                    "v_template": np.asarray(archive[f"{side}_v_template"], dtype=np.float64),
                    "shapedirs": np.asarray(archive[f"{side}_shapedirs"], dtype=np.float64),
                    "J_regressor": np.asarray(archive[f"{side}_J_regressor"], dtype=np.float64),
                    "weights": np.asarray(archive[f"{side}_weights"], dtype=np.float64),
                    "faces": np.asarray(archive[f"{side}_f"], dtype=np.int32),
                }
                for side in ("left", "right")
            }
        else:
            raise SystemExit(f"missing standard hand assets in {asset_dir}: expected HAND_LEFT.pkl and HAND_RIGHT.pkl")
        self._shape_cache: dict[tuple[str, tuple[float, ...]], tuple[np.ndarray, np.ndarray]] = {}

    def shaped(self, side: str, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        beta = np.asarray(beta, dtype=np.float64).reshape(10)
        key = (side, tuple(float(v) for v in np.round(beta, 8)))
        cached = self._shape_cache.get(key)
        if cached is not None:
            return cached
        model = self.models[side]
        vertices = model["v_template"] + np.tensordot(model["shapedirs"], beta, axes=([2], [0]))
        joints = model["J_regressor"] @ vertices
        result = (np.asarray(vertices, dtype=np.float64), np.asarray(joints, dtype=np.float64))
        self._shape_cache[key] = result
        return result

    def reconstruct(self, side: str, beta: np.ndarray, target21: np.ndarray) -> np.ndarray | None:
        target21 = np.asarray(target21, dtype=np.float64)
        if target21.shape != (21, 3) or not np.isfinite(target21).all():
            return None
        vertices, rest_joints = self.shaped(side, beta)
        target16 = target21[MANO16_TO_JSON21]
        rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 16, axis=0)
        rotations[0] = _palm_rotation(rest_joints, target21)
        for chain in FINGER_CHAINS:
            for joint in chain:
                parent = int(MANO_PARENTS[joint])
                if joint in DISTAL_TIPS:
                    tip_vertex, target_tip = DISTAL_TIPS[joint]
                    rest_out = vertices[tip_vertex] - rest_joints[joint]
                    target_out = target21[target_tip] - target16[joint]
                else:
                    child = joint + 1
                    rest_out = rest_joints[child] - rest_joints[joint]
                    target_out = target16[child] - target16[joint]
                predicted = rotations[parent] @ rest_out
                rotations[joint] = _rotation_between(predicted, target_out) @ rotations[parent]
        weights = self.models[side]["weights"]
        deformed = np.zeros_like(vertices)
        for joint in range(16):
            posed = (vertices - rest_joints[joint]) @ rotations[joint].T + target16[joint]
            deformed += weights[:, joint:joint + 1] * posed
        if not np.isfinite(deformed).all():
            return None
        return deformed

    def faces(self, side: str) -> np.ndarray:
        return self.models[side]["faces"]

    @staticmethod
    def present(asset_dir: Path) -> bool:
        standard = (asset_dir / "HAND_LEFT.pkl", asset_dir / "HAND_RIGHT.pkl")
        legacy = (asset_dir / "MANO_LEFT.pkl", asset_dir / "MANO_RIGHT.pkl", asset_dir / "mano_runtime.npz")
        return all(path.is_file() for path in standard) or all(path.is_file() for path in legacy)

    def fit_shape(self, side: str, joints: np.ndarray) -> np.ndarray:
        """One clip-wide beta for ``joints`` [T,21,3] (any rigid frame; NaN rows ignored).

        The median bone length over the finite frames is matched; with fewer
        than SHAPE_MIN_FRAMES such frames the mean shape (beta = 0) is used.
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(-1, 21, 3)
        finite = np.isfinite(joints).all(axis=(1, 2))
        if finite.sum() < SHAPE_MIN_FRAMES:
            return np.zeros(self.models[side]["shapedirs"].shape[2], dtype=np.float64)
        lengths = np.median(_bone_lengths21(joints[finite]), axis=0)
        return fit_shape_to_bone_lengths(self.models[side], lengths)


def _camera_coordinates(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    center = np.asarray(pose[:3, 3], dtype=np.float64)
    return (rotation.T @ (np.asarray(points_world, dtype=np.float64) - center).T).T


def _project_camera(points_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    if valid.any():
        homogeneous = (np.asarray(intrinsic, dtype=np.float64) @ points[valid].T).T
        pixels[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, depth, valid


def _anchored_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit target = (source-source_wrist) @ A + target_wrist, anchoring wrist exactly."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    if valid.sum() < 5 or not valid[0]:
        return None
    x = source[valid] - source[0]
    y = target[valid] - target[0]
    try:
        matrix, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 2 or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 20.0:
        return None
    return matrix, target[0]


def _apply_anchored_affine(points: np.ndarray, source_wrist: np.ndarray, fit: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    matrix, target_wrist = fit
    return (np.asarray(points, dtype=np.float64) - source_wrist) @ matrix + target_wrist


def _view_coordinates(points: np.ndarray, view: dict[str, np.ndarray | float]) -> np.ndarray:
    rel = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(view["eye"], dtype=np.float64).reshape(1, 3)
    return np.column_stack((
        rel @ np.asarray(view["right"], dtype=np.float64),
        rel @ np.asarray(view["up"], dtype=np.float64),
        rel @ np.asarray(view["forward"], dtype=np.float64),
    ))


def _project_view(points: np.ndarray, view: dict[str, np.ndarray | float], dims: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width, height = dims
    xyz = _view_coordinates(points, view)
    depth = xyz[:, 2]
    valid = np.isfinite(xyz).all(axis=1) & (depth > 1e-5)
    pixels = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    focal = float(view["focal"])
    if valid.any():
        pixels[valid, 0] = float(view["cx"]) + focal * xyz[valid, 0] / depth[valid]
        pixels[valid, 1] = float(view["cy"]) - focal * xyz[valid, 1] / depth[valid]
    return pixels, depth, valid, xyz


def _draw_mesh_pixels(canvas: np.ndarray, pixels: np.ndarray, view_xyz: np.ndarray, faces: np.ndarray,
                      color: tuple[int, int, int], alpha: float = 0.55, edges: bool = True) -> int:
    """Draw a depth-ordered mesh body.

    `alpha=1` gives an opaque surface. `edges=True` restores the previous
    pipeline-style dark triangle-edge overlay used by the source pane.
    """
    height, width = canvas.shape[:2]
    pixels = np.asarray(pixels, dtype=np.float64)
    xyz = np.asarray(view_xyz, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    finite_vertices = np.isfinite(pixels).all(axis=1) & np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-5)
    face_valid = finite_vertices[faces].all(axis=1)
    selected = faces[face_valid]
    if not len(selected):
        return 0
    tri_px = pixels[selected]
    inside = (
        (tri_px[:, :, 0].max(axis=1) >= -width)
        & (tri_px[:, :, 0].min(axis=1) <= width * 2)
        & (tri_px[:, :, 1].max(axis=1) >= -height)
        & (tri_px[:, :, 1].min(axis=1) <= height * 2)
    )
    selected = selected[inside]
    if not len(selected):
        return 0
    triangle_depth = xyz[selected, 2].mean(axis=1)
    order = np.argsort(triangle_depth)[::-1]
    selected = selected[order]
    normals = np.cross(xyz[selected[:, 1]] - xyz[selected[:, 0]], xyz[selected[:, 2]] - xyz[selected[:, 0]])
    normal_norm = np.linalg.norm(normals, axis=1)
    facing = np.abs(normals[:, 2]) / np.maximum(normal_norm, 1e-8)
    shades = 0.58 + 0.42 * facing
    mesh_layer = canvas.copy()
    edge_layer = canvas.copy()
    base = np.asarray(color, dtype=np.float64)
    drawn = 0
    for face, shade in zip(selected, shades):
        triangle = np.rint(pixels[face]).astype(np.int32)
        area = abs(float(cv2.contourArea(triangle)))
        if area < 0.15:
            continue
        shaded = tuple(int(np.clip(channel * shade, 0, 255)) for channel in base)
        cv2.fillConvexPoly(mesh_layer, triangle, shaded, cv2.LINE_AA)
        if edges:
            cv2.polylines(edge_layer, [triangle], True, (3, 3, 3), 1, cv2.LINE_AA)
        drawn += 1
    if drawn:
        if alpha >= 0.999:
            canvas[:] = mesh_layer
        else:
            cv2.addWeighted(mesh_layer, alpha, canvas, 1.0 - alpha, 0.0, canvas)
        if edges:
            cv2.addWeighted(edge_layer, 0.75, canvas, 0.25, 0.0, canvas)
    return drawn


def _draw_source_skeleton(canvas: np.ndarray, joints_px: np.ndarray, color: tuple[int, int, int]) -> None:
    height, width = canvas.shape[:2]
    for a, b in HAND_EDGES:
        pa, pb = joints_px[a], joints_px[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        xa, ya = int(round(float(pa[0]))), int(round(float(pa[1])))
        xb, yb = int(round(float(pb[0]))), int(round(float(pb[1])))
        if -80 <= xa <= width + 80 and -80 <= xb <= width + 80 and -80 <= ya <= height + 80 and -80 <= yb <= height + 80:
            cv2.line(canvas, (xa, ya), (xb, yb), OUTLINE, 5, cv2.LINE_AA)
            cv2.line(canvas, (xa, ya), (xb, yb), color, 3, cv2.LINE_AA)
    for point in joints_px:
        if not np.isfinite(point).all():
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(canvas, (x, y), 4, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)


def _draw_view_line(canvas: np.ndarray, world_points: np.ndarray, view: dict[str, np.ndarray | float],
                    color: tuple[int, int, int], thickness: int = 1) -> None:
    pixels, _, valid, _ = _project_view(world_points, view, (canvas.shape[1], canvas.shape[0]))
    for index in range(len(pixels) - 1):
        if not (valid[index] and valid[index + 1]):
            continue
        a = tuple(np.rint(pixels[index]).astype(int))
        b = tuple(np.rint(pixels[index + 1]).astype(int))
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)


def _draw_world_skeleton(canvas: np.ndarray, joints_world: np.ndarray, view: dict[str, np.ndarray | float],
                         color: tuple[int, int, int]) -> None:
    pixels, _, valid, _ = _project_view(joints_world, view, (canvas.shape[1], canvas.shape[0]))
    for a, b in HAND_EDGES:
        if not (valid[a] and valid[b]):
            continue
        pa = tuple(np.rint(pixels[a]).astype(int))
        pb = tuple(np.rint(pixels[b]).astype(int))
        cv2.line(canvas, pa, pb, OUTLINE, 5, cv2.LINE_AA)
        cv2.line(canvas, pa, pb, color, 3, cv2.LINE_AA)
    # RRD uses red keypoint dots independently of the left/right skeleton hue.
    for index, point in enumerate(pixels):
        if valid[index]:
            p = tuple(np.rint(point).astype(int))
            cv2.circle(canvas, p, 4, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 2, KEYPOINT_COLOR, -1, cv2.LINE_AA)


def _build_cinema_camera(s: float = 0.07) -> tuple[np.ndarray, np.ndarray]:
    """Return the detailed RDF cinema-camera mesh used by the RRD spatial view."""
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def add_box(cx: float, cy: float, cz: float, w: float, h: float, d: float) -> None:
        base = len(verts)
        for dx, dy, dz in ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                           (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)):
            verts.append([cx + dx * w / 2, cy + dy * h / 2, cz + dz * d / 2])
        for face in ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                     (0, 4, 5), (0, 5, 1), (2, 6, 7), (2, 7, 3),
                     (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)):
            faces.append([base + index for index in face])

    def add_cylinder(cx: float, cy: float, cz: float, radius: float, length: float,
                     n: int = 16, axis: str = "z") -> None:
        base = len(verts)
        for index in range(n):
            angle = 2 * np.pi * index / n
            if axis == "z":
                verts.extend(([cx + radius * np.cos(angle), cy + radius * np.sin(angle), cz],
                              [cx + radius * np.cos(angle), cy + radius * np.sin(angle), cz + length]))
            else:
                verts.extend(([cx + radius * np.cos(angle), cy, cz + radius * np.sin(angle)],
                              [cx + radius * np.cos(angle), cy + length, cz + radius * np.sin(angle)]))
        for index in range(n):
            nxt = (index + 1) % n
            faces.extend(([base + 2 * index, base + 2 * nxt, base + 2 * nxt + 1],
                          [base + 2 * index, base + 2 * nxt + 1, base + 2 * index + 1]))

    add_box(0, 0, 0, s * 1.4, s * 0.9, s * 0.8)
    add_box(0, -s * 0.55, -s * 0.1, s * 0.8, s * 0.15, s * 0.5)
    add_box(-s * 0.35, -s * 0.48, -s * 0.1, s * 0.08, s * 0.12, s * 0.3)
    add_box(s * 0.35, -s * 0.48, -s * 0.1, s * 0.08, s * 0.12, s * 0.3)
    add_cylinder(0, 0, s * 0.4, s * 0.45, s * 0.35, n=20)
    add_cylinder(0, 0, s * 0.4 + s * 0.35, s * 0.48, s * 0.06, n=20)
    add_box(0, -s * 0.15, -s * 0.5, s * 0.5, s * 0.35, s * 0.2)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _camera_marker_points(pose: np.ndarray, scale: float = 0.032) -> tuple[np.ndarray, list[tuple[int, int]]]:
    center = np.asarray(pose[:3, 3], dtype=np.float64)
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    x_axis, down_axis, forward = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    front = center + forward * scale
    half_w, half_h = scale * 0.60, scale * 0.42
    corners = np.asarray((
        front + x_axis * half_w + down_axis * half_h,
        front - x_axis * half_w + down_axis * half_h,
        front - x_axis * half_w - down_axis * half_h,
        front + x_axis * half_w - down_axis * half_h,
    ))
    points = np.vstack((center, corners, center + x_axis * scale, center + down_axis * scale, center + forward * scale * 1.35))
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1), (0, 5), (0, 6), (0, 7)]
    return points, edges


def _draw_camera_marker(canvas: np.ndarray, pose: np.ndarray, view: dict[str, np.ndarray | float]) -> None:
    points, edges = _camera_marker_points(pose)
    pixels, _, valid, _ = _project_view(points, view, (canvas.shape[1], canvas.shape[0]))
    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        color = AXIS_COLORS.get((a, b), CAMERA_COLOR)
        pa, pb = tuple(np.rint(pixels[a]).astype(int)), tuple(np.rint(pixels[b]).astype(int))
        # Keep the camera marker subordinate to the dark body, while making
        # the three short RGB direction arrows legible at the camera origin.
        width = 3 if (a, b) in AXIS_COLORS else 2
        cv2.arrowedLine(canvas, pa, pb, color, width, cv2.LINE_AA, tipLength=0.28)


def _trajectory_upto(centers: np.ndarray, frame_index: int) -> np.ndarray:
    """Camera path from the clip start up to and including the current frame.

    The RRD spatial view logs the trail as a static strip of every valid pose,
    but a per-frame raster must accumulate: drawing the whole clip at frame 0
    would show the camera walking along a path it has not taken yet, and would
    also make the final-frame picture of a short clip identical to its opening
    frame. Points after `frame_index` are dropped, not skipped, so the polyline
    still terminates exactly at the current camera body.
    """
    end = int(np.clip(frame_index + 1, 0, len(centers)))
    past = np.asarray(centers[:end], dtype=np.float64)
    return past[np.isfinite(past).all(axis=1)]


def _look_at_view(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> dict[str, np.ndarray]:
    """Build an orthonormal right-handed basis that looks from `eye` at `target`.

    `up_hint` is projected onto the plane perpendicular to the view direction, so
    the basis stays orthonormal and (right, up, forward) is a genuine camera
    frame rather than a permuted copy of the source camera axes. A view that is
    not rigid -- axes permuted, or the eye moved along a non-forward axis while
    the basis is held fixed -- is an oblique projection: it shears the scene and
    can invert the on-screen left/right order of the subjects, which is exactly
    how a left hand ends up drawn to the right of a right hand.
    """
    forward = _normalize(np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64), np.asarray([0.0, 0.0, 1.0]))
    hint = np.asarray(up_hint, dtype=np.float64)
    hint = hint - forward * float(np.dot(hint, forward))
    if float(np.linalg.norm(hint)) < 1e-6:
        fallback = np.asarray([0.0, 0.0, 1.0]) if abs(float(forward[2])) < 0.9 else np.asarray([1.0, 0.0, 0.0])
        hint = fallback - forward * float(np.dot(fallback, forward))
    up = _normalize(hint, np.asarray([0.0, 0.0, 1.0]))
    # right = cross(forward, up) is the screen-right axis for this renderer: the
    # projector maps +x to `right` and flips the `up` component into pixel-y
    # (camera convention, y down). With up = -down_cam that yields camera +x at
    # the camera origin, so the world pane agrees with the video pane instead of
    # being mirrored left/right. The reverse order, cross(up, forward), produces
    # camera -x and inverts every left/right relationship on screen.
    right = _normalize(np.cross(forward, up), np.asarray([1.0, 0.0, 0.0]))
    up = _normalize(np.cross(right, forward), up)
    return {"right": right, "up": up, "forward": forward}


def _clip_scene_points(poses: np.ndarray, pose_valid: np.ndarray, joints: list[np.ndarray],
                       cam_verts: np.ndarray) -> np.ndarray:
    """Every world point the fixed observer must keep in frame, for the whole clip.

    Union of the camera-body hull of each valid pose and all finite hand joints.
    Computed once so the observer can be placed from the full traverse instead
    of re-aiming per frame.
    """
    chunks: list[np.ndarray] = []
    for index in np.flatnonzero(pose_valid):
        pose = np.asarray(poses[index], dtype=np.float64)
        if not np.isfinite(pose).all():
            continue
        chunks.append((pose[:3, :3] @ cam_verts.T).T + pose[:3, 3])
    for side in range(len(joints)):
        rows = np.asarray(joints[side], dtype=np.float64).reshape(-1, 21, 3)
        chunks.append(rows[np.isfinite(rows).all(axis=(1, 2))].reshape(-1, 3))
    chunks = [chunk for chunk in chunks if len(chunk)]
    return np.vstack(chunks) if chunks else np.zeros((0, 3), dtype=np.float64)


def _frame_extent(points: np.ndarray,
                  view: dict[str, np.ndarray | float]) -> tuple[float, float, float, float] | None:
    """Projected bounds of `points` in normalised pinhole coordinates."""
    if not len(points):
        return None
    xyz = _view_coordinates(points, view)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-5)
    if not valid.any():
        return None
    nx = xyz[valid, 0] / xyz[valid, 2]
    ny = xyz[valid, 1] / xyz[valid, 2]
    return float(nx.min()), float(nx.max()), float(ny.min()), float(ny.max())


def _fit_focal(points: np.ndarray, view: dict[str, np.ndarray | float],
               width: int, height: int, margin: float,
               cx0: float | None = None, cy0: float | None = None) -> float:
    """Largest focal length that keeps every point `margin` px inside the pane.

    The principal point (cx0, cy0) is NOT the pane centre -- the spatial view
    sits slightly low (cy = 0.54h) so the floor grid reads -- and the scene is
    not centred on it either when the observer lags a moving camera. Each
    point therefore gets its own budget: the pixel distance from the principal
    point to the edge on its side, divided by its normalised coordinate.
    """
    extent = _frame_extent(points, view)
    if extent is None:
        return float(min(width, height))
    x_min, x_max, y_min, y_max = extent
    cx0 = width * 0.5 if cx0 is None else float(cx0)
    cy0 = height * 0.54 if cy0 is None else float(cy0)
    budgets = [float(min(width, height))]
    for coordinate, room in ((x_max, width - margin - cx0), (-x_min, cx0 - margin),
                             (-y_min, height - margin - cy0), (y_max, cy0 - margin)):
        # Pixel-y is flipped: a positive normalised y goes UP the pane, toward
        # the top edge whose room is cy0 - margin.
        if coordinate > 1e-9:
            budgets.append(max(room, 1.0) / coordinate)
    return float(min(budgets))


def _gaussian_smooth(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 with edge-value padding.

    Offline rendering has the whole clip, so a symmetric kernel is used: no lag,
    no overshoot, and a clip shorter than the kernel collapses to a near-constant
    track, which is what a static scene should produce.
    """
    values = np.asarray(values, dtype=np.float64)
    if sigma_frames <= 0.0 or len(values) < 2:
        return values.copy()
    radius = max(1, int(math.ceil(3.0 * sigma_frames)))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_frames) ** 2)
    taps /= taps.sum()
    flat = values.reshape(len(values), -1)
    padded = np.pad(flat, ((radius, radius), (0, 0)), mode="edge")
    out = np.empty_like(flat)
    for column in range(flat.shape[1]):
        out[:, column] = np.convolve(padded[:, column], taps, mode="valid")
    return out.reshape(values.shape)


def _rolling_extreme(values: np.ndarray, radius: int, maximum: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    radius = max(0, int(radius))
    padded = np.pad(values, (radius, radius), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    return windows.max(axis=1) if maximum else windows.min(axis=1)


def _deadband_follow(values: np.ndarray, deadband: float) -> np.ndarray:
    """Backlash follower: the output only moves once the input has left a ball
    of radius `deadband` around it, and then only by the overshoot.

    Jitter whose amplitude stays inside the band therefore produces exactly
    zero output motion, while a sustained move is followed at the input's own
    velocity, lagging by one band radius.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    if not len(values):
        return out
    current = values[0].copy()
    for index, value in enumerate(values):
        delta = value - current
        distance = float(np.linalg.norm(delta))
        if distance > deadband:
            current = value - delta * (deadband / distance)
        out[index] = current
    return out


def _slew_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    if not len(values):
        return out
    current = values[0].copy()
    for index, value in enumerate(values):
        delta = value - current
        distance = float(np.linalg.norm(delta))
        if distance > max_step:
            delta *= max_step / distance
        current = current + delta
        out[index] = current
    return out


def _fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Hold the nearest valid sample across invalid frames (axis 0)."""
    values = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    indices = np.arange(len(values))
    good = indices[valid]
    nearest = good[np.clip(np.searchsorted(good, indices), 0, len(good) - 1)]
    previous = good[np.clip(np.searchsorted(good, indices, side="right") - 1, 0, len(good) - 1)]
    pick = np.where(np.abs(nearest - indices) <= np.abs(indices - previous), nearest, previous)
    return values[pick]


def _estimate_world_up(poses: np.ndarray, pose_valid: np.ndarray,
                       world_up: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, object]]:
    """World-up (gravity) direction in the product world frame.

    The product world is the first camera frame and carries no gravity. A
    head-mounted camera keeps its right axis (+x) close to horizontal, so
    world-up is the direction most orthogonal to every right axis in the clip:
    the smallest-eigenvalue eigenvector of sum(r r^T). That is well-conditioned
    only when the heading varies; with little yaw the two smallest eigenvalues
    are close and the up vector is ambiguous within the plane perpendicular to
    the mean right axis. The fallback then takes the camera's -down axis in
    that plane and tilts it by the assumed head pitch so that the camera looks
    forward-down at a work surface.
    """
    rotations = np.asarray(poses, dtype=np.float64)[np.asarray(pose_valid, dtype=bool)][:, :3, :3]
    rotations = rotations[np.isfinite(rotations).all(axis=(1, 2))]
    if not len(rotations):
        rotations = np.eye(3, dtype=np.float64)[None]
    rights = rotations[:, :, 0]
    downs = rotations[:, :, 1]
    forwards = rotations[:, :, 2]
    mean_right = _normalize(rights.mean(axis=0), np.asarray([1.0, 0.0, 0.0]))
    mean_down = _normalize(downs.mean(axis=0), np.asarray([0.0, 1.0, 0.0]))
    mean_forward = _normalize(forwards.mean(axis=0), np.asarray([0.0, 0.0, 1.0]))
    if world_up is not None:
        up = _normalize(world_up, np.asarray([0.0, -1.0, 0.0]))
        return up, {"branch": "given", "eigen_ratio": float("nan"), "eigenvalues": [float("nan")] * 3}

    scatter = rights.T @ rights
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # The ratio floor is relative to the largest eigenvalue (~ frame count):
    # a single pose or a perfectly rigid clip has two eigenvalues at round-off,
    # and their quotient is noise, not conditioning.
    smallest = max(float(eigenvalues[0]), 1e-4 * max(float(eigenvalues[2]), 1e-12))
    ratio = float(eigenvalues[1]) / smallest
    # lambda_1 / lambda_0 >= 4 means the second-smallest spread is at least twice
    # the residual: the yaw variation has to exceed the roll noise before the
    # eigenvector is trusted. When roll dominates instead, the right axes sweep
    # around the camera forward and the eigenvector collapses onto that forward
    # axis, i.e. a head pitch near 90 deg; a head camera at a work surface pitches
    # 20-60 deg, so a candidate implying more than 75 deg is rejected too.
    candidate = _normalize(eigenvectors[:, 0], -mean_down)
    if float(np.dot(candidate, mean_down)) > 0.0:
        candidate = -candidate
    implied_pitch = math.degrees(math.asin(min(1.0, abs(float(np.dot(candidate, mean_forward))))))
    if ratio >= 4.0 and implied_pitch <= 75.0:
        up = candidate
        branch = "eigen"
    else:
        base = -mean_down - mean_right * float(np.dot(-mean_down, mean_right))
        base = _normalize(base, -mean_down)
        pitch = math.radians(_FALLBACK_HEAD_PITCH_DEG)
        candidates = []
        for sign in (1.0, -1.0):
            rotated = base * math.cos(sign * pitch) + np.cross(mean_right, base) * math.sin(sign * pitch) \
                + mean_right * float(np.dot(mean_right, base)) * (1.0 - math.cos(sign * pitch))
            candidates.append(_normalize(rotated, base))
        # Forward-down: the camera forward has a negative component along up.
        up = min(candidates, key=lambda candidate: float(np.dot(candidate, mean_forward)))
        branch = "fallback"
    return up, {"branch": branch, "eigen_ratio": ratio, "eigenvalues": [float(v) for v in eigenvalues]}


def _horizontal_basis(up: np.ndarray, forward_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two horizontal unit axes (h1 along the hint's horizontal part, h2 = up x h1)."""
    h1 = np.asarray(forward_hint, dtype=np.float64) - up * float(np.dot(forward_hint, up))
    if float(np.linalg.norm(h1)) < 1e-6:
        seed = np.asarray([1.0, 0.0, 0.0]) if abs(float(up[0])) < 0.9 else np.asarray([0.0, 0.0, 1.0])
        h1 = seed - up * float(np.dot(seed, up))
    h1 = _normalize(h1, np.asarray([0.0, 0.0, 1.0]))
    h2 = _normalize(np.cross(up, h1), np.asarray([1.0, 0.0, 0.0]))
    return h1, h2


def _rotate_about(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis, np.asarray([0.0, 0.0, 1.0]))
    return (vector * math.cos(angle) + np.cross(axis, vector) * math.sin(angle)
            + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle)))


def _floor_plane(points: np.ndarray, up: np.ndarray, forward_hint: np.ndarray) -> dict[str, np.ndarray]:
    """Level floor: perpendicular to `up`, _FLOOR_DROP_M below the lowest scene point."""
    h1, h2 = _horizontal_basis(up, forward_hint)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        points = np.zeros((1, 3), dtype=np.float64)
    heights = points @ up
    centre = np.median(points, axis=0)
    origin = centre - up * float(np.dot(centre, up)) + up * (float(heights.min()) - _FLOOR_DROP_M)
    return {"origin": origin, "h1": h1, "h2": h2, "up": np.asarray(up, dtype=np.float64)}


def _follow_view_track(poses: np.ndarray, pose_valid: np.ndarray, joints: list[np.ndarray],
                       cam_verts: np.ndarray, width: int, height: int, fps: float = 30.0,
                       world_up: np.ndarray | None = None,
                       margin: float = 0.06) -> tuple[list[dict[str, np.ndarray | float] | None], dict[str, object]]:
    """One observer view per frame that follows the camera and hands slowly.

    Everything is solved offline from the whole clip:
    - world-up comes from _estimate_world_up and is the up hint of EVERY frame,
      so the observer never rolls with the ego camera and the floor stays level;
    - the look-at target is the camera-weighted centroid of camera centre and
      hand joints, Gaussian-smoothed (sigma 1.5 s), passed through a 5 cm
      backlash so jitter yields no motion, slew-limited and re-smoothed;
    - the heading is the camera forward projected on the horizontal plane,
      unwrapped, smoothed with sigma 3 s and a 15 deg backlash; the eye sits
      behind that heading, _EYE_AZIMUTH to the side that keeps the hands in
      their video-pane left/right order, _EYE_ELEVATION above horizontal;
    - the standoff follows the smoothed scene extent within the fixed band and
      the focal is only shortened (smoothly) when the scene would leave the
      frame at the nominal field of view.
    Returns per-frame views (None where the pose is invalid) and a report dict.
    """
    poses = np.asarray(poses, dtype=np.float64)
    pose_valid = np.asarray(pose_valid, dtype=bool) & np.isfinite(poses).all(axis=(1, 2))
    count = len(poses)
    fps = float(fps) if np.isfinite(fps) and fps > 0 else 30.0
    up, gravity = _estimate_world_up(poses, pose_valid, world_up)

    centres = poses[:, :3, 3]
    forwards = poses[:, :3, 2]
    hand_means = np.full((count, len(joints), 3), np.nan, dtype=np.float64)
    hand_valid = np.zeros((count, len(joints)), dtype=bool)
    for side in range(len(joints)):
        rows = np.asarray(joints[side], dtype=np.float64).reshape(count, -1, 3)
        hand_valid[:, side] = np.isfinite(rows).all(axis=(1, 2))
        hand_means[hand_valid[:, side], side] = rows[hand_valid[:, side]].mean(axis=1)

    # Anchor: camera weight 2, each hand weight 1. A hand that drops out of
    # the annotation keeps its last camera-relative offset instead of leaving
    # the centroid, so visibility flicker cannot yank the observer.
    anchor = np.zeros((count, 3), dtype=np.float64)
    offsets = np.zeros((count, len(joints), 3), dtype=np.float64)
    for side in range(len(joints)):
        seen = hand_valid[:, side] & pose_valid
        if not seen.any():
            continue
        offsets[seen, side] = hand_means[seen, side] - centres[seen]
        offsets[:, side] = _fill_invalid(offsets[:, side], seen)
    hands_present = [side for side in range(len(joints)) if (hand_valid[:, side] & pose_valid).any()]
    for index in range(count):
        if not pose_valid[index]:
            continue
        weighted = [(2.0, centres[index])] + [(1.0, centres[index] + offsets[index, side]) for side in hands_present]
        total = sum(weight for weight, _ in weighted)
        anchor[index] = sum(weight * point for weight, point in weighted) / total
    anchor = _fill_invalid(anchor, pose_valid)
    target = _gaussian_smooth(anchor, 1.5 * fps)
    target = _deadband_follow(target, _ANCHOR_DEADBAND_M)
    target = _slew_limit(target, _OBSERVER_MAX_SPEED / fps)
    target = _gaussian_smooth(target, 0.5 * fps)

    # Heading: horizontal camera forward, unwrapped, heavily smoothed.
    h1, h2 = _horizontal_basis(up, _normalize(forwards[pose_valid].mean(axis=0) if pose_valid.any()
                                              else np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, 1.0])))
    forward_filled = _fill_invalid(forwards, pose_valid)
    heading = np.unwrap(np.arctan2(forward_filled @ h2, forward_filled @ h1))
    heading = _gaussian_smooth(heading, 3.0 * fps)
    heading = _deadband_follow(heading[:, None], math.radians(_HEADING_DEADBAND_DEG))[:, 0]
    heading = _gaussian_smooth(heading, 0.5 * fps)

    # Scene extent around the SMOOTHED target (the point the observer looks
    # at), max over +-1.5 s, then smoothed. Measuring it around the raw anchor
    # would ignore the lag of the target behind a walking camera, leaving the
    # focal guard to zoom out hard instead of the standoff growing gently.
    extent = np.zeros(count, dtype=np.float64)
    scene_per_frame: list[np.ndarray] = []
    for index in range(count):
        chunk: list[np.ndarray] = []
        if pose_valid[index]:
            chunk.append((poses[index, :3, :3] @ cam_verts.T).T + centres[index])
            for side in range(len(joints)):
                if hand_valid[index, side]:
                    chunk.append(np.asarray(joints[side][index], dtype=np.float64).reshape(-1, 3))
        points = np.vstack(chunk) if chunk else np.zeros((0, 3), dtype=np.float64)
        scene_per_frame.append(points)
        if len(points):
            extent[index] = float(np.linalg.norm(points - target[index], axis=1).max())
    extent = _fill_invalid(extent, pose_valid)
    extent = _rolling_extreme(extent, int(round(1.5 * fps)), maximum=True)
    extent = _gaussian_smooth(extent, 1.0 * fps)
    # Nominal field of view: focal = short side, so half the usable frame
    # (minus margin) spans 0.5 - margin in normalised coordinates.
    # The extent is a bounding radius; the scene is mostly elongated along the
    # view axis (hands in front of the camera), so 0.75 of the sphere-fit
    # distance frames it with room to spare. The focal guard below widens the
    # field of view smoothly on the rare frames where that is too tight.
    focal_nominal = float(min(width, height))
    usable = 0.5 - margin
    distance = np.clip(0.75 * extent / usable, *_EYE_DISTANCE_RANGE)
    distance = _gaussian_smooth(distance, 1.0 * fps)

    elevation = math.radians(_EYE_ELEVATION)

    def eye_for(index: int, azimuth_sign: float) -> np.ndarray:
        back = -(h1 * math.cos(heading[index]) + h2 * math.sin(heading[index]))
        side_dir = _rotate_about(back, up, azimuth_sign * math.radians(_EYE_AZIMUTH))
        return target[index] + distance[index] * (side_dir * math.cos(elevation) + up * math.sin(elevation))

    def views_for(azimuth_sign: float) -> list[dict[str, np.ndarray | float] | None]:
        result: list[dict[str, np.ndarray | float] | None] = []
        for index in range(count):
            if not pose_valid[index]:
                result.append(None)
                continue
            eye = eye_for(index, azimuth_sign)
            basis = _look_at_view(eye, target[index], up)
            result.append({"eye": eye, "right": basis["right"], "up": basis["up"], "forward": basis["forward"],
                           "focal": focal_nominal, "cx": width * 0.5, "cy": height * 0.54})
        return result

    # Side choice: the observer swings to whichever side keeps the anatomical
    # left hand to the screen-left of the right hand by the widest margin, i.e.
    # the hands neither mirror nor occlude each other. Camera-left (-x) is the
    # established default and wins ties. One side for the whole clip.
    def separation(views: list[dict[str, np.ndarray | float] | None]) -> float:
        gaps: list[float] = []
        for index in range(count):
            view = views[index]
            if view is None or len(joints) < 2 or not hand_valid[index].all():
                continue
            pixels, _, valid, _ = _project_view(hand_means[index], view, (width, height))
            if valid.all():
                gaps.append(float(pixels[1, 0] - pixels[0, 0]))
        return float(np.mean(gaps)) if gaps else 0.0

    candidates = {-1.0: views_for(-1.0), 1.0: views_for(1.0)}
    scores = {sign: separation(views) for sign, views in candidates.items()}
    side = -1.0 if scores[-1.0] >= scores[1.0] - 1e-9 else 1.0
    views = candidates[side]

    # Framing guard: if the nominal focal would clip the scene at some frame,
    # shorten the focal there (rolling-min over +-1.5 s, then smoothed) so the
    # widening is gradual rather than a zoom snap.
    px_margin = margin * min(width, height)
    required = np.full(count, focal_nominal, dtype=np.float64)
    for index in range(count):
        view = views[index]
        if view is None or not len(scene_per_frame[index]):
            continue
        required[index] = min(focal_nominal, _fit_focal(scene_per_frame[index], view, width, height, px_margin,
                                                       width * 0.5, height * 0.54))
    required = _fill_invalid(required, pose_valid)
    focal = _rolling_extreme(required, int(round(1.5 * fps)), maximum=False)
    focal = _gaussian_smooth(focal, 0.5 * fps)
    focal = np.minimum(focal, required)
    for index in range(count):
        if views[index] is not None:
            views[index]["focal"] = float(focal[index])

    scene_points = np.vstack([points for points in scene_per_frame if len(points)]) if any(
        len(points) for points in scene_per_frame) else np.zeros((0, 3), dtype=np.float64)
    report = {
        "up": up,
        "gravity": gravity,
        "floor": _floor_plane(scene_points, up, h1),
        "side": side,
        "anchor": anchor,
        "target": target,
        "heading": heading,
        "distance": distance,
        "focal": focal,
        "eye": np.asarray([view["eye"] if view is not None else np.full(3, np.nan) for view in views]),
    }
    return views, report


def _single_pose_view(pose: np.ndarray, frame_joints: list[np.ndarray], cam_verts: np.ndarray,
                      width: int, height: int) -> tuple[dict[str, np.ndarray | float], dict[str, object]]:
    """Follow view for a one-frame clip (stills and tests)."""
    joints = [np.asarray(hand, dtype=np.float64).reshape(1, -1, 3) for hand in frame_joints]
    views, report = _follow_view_track(np.asarray(pose, dtype=np.float64)[None], np.asarray([True]),
                                       joints, cam_verts, width, height)
    return views[0], report


def _spatial_background(width: int, height: int) -> np.ndarray:
    """Build the sampled dark blue-green spatial-view background in BGR."""
    x = np.linspace(0.0, 1.0, max(width, 1), dtype=np.float64)[None, :, None]
    y = np.linspace(0.0, 1.0, max(height, 1), dtype=np.float64)[:, None, None]
    top = (1.0 - x) * _SPATIAL_BG_CORNERS[0, 0] + x * _SPATIAL_BG_CORNERS[0, 1]
    bottom = (1.0 - x) * _SPATIAL_BG_CORNERS[1, 0] + x * _SPATIAL_BG_CORNERS[1, 1]
    return np.rint((1.0 - y) * top + y * bottom).astype(np.uint8)


def _draw_floor_grid(canvas: np.ndarray, floor: dict[str, np.ndarray],
                     view: dict[str, np.ndarray | float]) -> None:
    """Rasterize the RRD-style faint ground grid on the level floor plane.

    The product world is the first camera frame, which is tilted by whatever
    the head was doing at frame 1, so a grid on a world-axis plane leans. The
    grid lives instead on the plane perpendicular to the estimated world-up,
    _FLOOR_DROP_M below the lowest camera/hand point, with its lines aligned to
    the clip's mean heading.
    """
    origin = np.asarray(floor["origin"], dtype=np.float64)
    h1 = np.asarray(floor["h1"], dtype=np.float64)
    h2 = np.asarray(floor["h2"], dtype=np.float64)
    grid_lines: list[np.ndarray] = []
    for offset in np.arange(-2.0, 2.25, 0.5):
        grid_lines.append(np.asarray((origin + h1 * offset - h2 * 2.0, origin + h1 * offset + h2 * 2.0)))
        grid_lines.append(np.asarray((origin + h2 * offset - h1 * 2.0, origin + h2 * offset + h1 * 2.0)))
    # The target view uses a crisp, one-pixel neutral grid; antialiasing keeps
    # diagonals smooth while retaining the sampled bright line core.
    alpha = 0.92
    for points in grid_lines:
        pixels, _, valid, _ = _project_view(points, view, (canvas.shape[1], canvas.shape[0]))
        if not valid.all():
            continue
        layer = canvas.copy()
        a = tuple(np.rint(pixels[0]).astype(int))
        b = tuple(np.rint(pixels[1]).astype(int))
        cv2.line(layer, a, b, GRID_COLOR, 1, cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, canvas, 1.0 - alpha, 0.0, canvas)


def _resize_to_contain(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = min(width / max(source_w, 1), height / max(source_h, 1))
    resized_w, resized_h = max(1, int(round(source_w * scale))), max(1, int(round(source_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), (18, 20, 23), dtype=np.uint8)
    x0, y0 = (width - resized_w) // 2, (height - resized_h) // 2
    canvas[y0:y0 + resized_h, x0:x0 + resized_w] = resized
    return canvas


def _text_font(size: int):
    if ImageFont is None:
        return None
    candidates = (
        # Full-coverage CJK+Latin font first: the bundled fonts and
        # DroidSansFallbackFull are CJK-only subsets whose Latin glyphs render
        # as tofu boxes in mixed strings; rendered labels are English-only.
        Path.home() / ".fonts" / "NotoSansCJKsc-Regular.ttf",
        Path(__file__).resolve().parent / "fonts" / "NotoSansCJK-Regular.ttc",
        Path(__file__).resolve().parent / "fonts" / "NotoSansCJK-Regular.otf",
        Path(__file__).resolve().parent / "fonts" / "NotoSansCJK-Regular.ttf",
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return None


def _put_text_mixed(canvas: np.ndarray, text: str, origin: tuple[int, int], size: int,
                    color: tuple[int, int, int], thickness: int = 2) -> None:
    """Draw Chinese-safe text; fall back to OpenCV only for ASCII-only strings."""
    if Image is None or ImageDraw is None or not any(ord(ch) > 127 for ch in text):
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size / 32.0, color, thickness, cv2.LINE_AA)
        return
    font = _text_font(size)
    if font is None:
        return
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    x, y = origin
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]), stroke_width=max(1, thickness - 1), stroke_fill=(0, 0, 0))
    canvas[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _wrap_text(text: str, font: int, scale: float, thickness: int, max_width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if current and width > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_subtitle(canvas: np.ndarray, structured: str, narrative: str, frame_index: int, frame_count: int,
                   band_height: int, label_lines: list[str] | None = None, label_notice: str | None = None) -> None:
    """Draw the AR label block and the natural-language description in the bottom strip.

    Labels live only in this band; the pose/video panes are never covered.
    ``label_notice`` is the explicit marker for absent/unlabeled input — the
    renderer never invents label text from other fields.
    """
    del structured, frame_index, frame_count
    height, width = canvas.shape[:2]
    y0 = height - band_height
    layer = canvas.copy()
    cv2.rectangle(layer, (0, y0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(layer, 0.72, canvas, 0.28, 0.0, canvas)
    y = y0 + 42
    if label_notice is not None:
        cv2.putText(canvas, label_notice, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (80, 200, 255), 1, cv2.LINE_AA)
        y += 27
    elif label_lines:
        for line in label_lines:
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (120, 240, 160), 1, cv2.LINE_AA)
            y += 27
    lines = _wrap_text(narrative, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1, width - 36)
    remaining = max(0, (height - 6 - y)) // 27
    for line in lines[:min(3, remaining)]:
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 248, 248), 1, cv2.LINE_AA)
        y += 27


def _label_lines_for_frame(segments: list, time_seconds: float, prefix: str, max_width: int) -> list[str]:
    """Wrapped AR label lines active at a presentation time (half-open), capped at two."""
    lines: list[str] = []
    for start, end, text in segments:
        if start <= time_seconds < end:
            lines.extend(_wrap_text(prefix + text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1, max_width))
    if len(lines) > 2:
        lines = lines[:2]
        lines[-1] = lines[-1].rstrip(" .") + " …"
    return lines


def _subtitle_for_frame(rows: list[dict], frame_index: int, cursor: int, *, cosmos: bool = False) -> tuple[str, str, int]:
    if cosmos:
        detail = (
            "I0 has no hand prediction; action row 0 begins at the next video frame."
            if frame_index == 0 else
            "Predicted 3D joints; hand visibility and object labels are not provided."
        )
        return "", detail, cursor
    frame_number = frame_index + 1
    while cursor < len(rows) and frame_number > int(rows[cursor]["起始帧"][1]):
        cursor += 1
    if cursor >= len(rows):
        return "", "", cursor
    row = rows[cursor]
    start, end = (int(value) for value in row["起始帧"])
    if not (start <= frame_number <= end):
        return "", "", cursor

    narrative = str(row.get("总体描述") or "").strip()
    return "", narrative, cursor


def _labels_module():
    """The sibling labels module, both as a package import and as a script."""
    if __package__:
        from . import labels
        return labels
    try:
        import labels
        return labels
    except ModuleNotFoundError as exc:
        if exc.name != "labels":
            raise
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("labels", Path(__file__).with_name("labels.py"))
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _load_labels(path: Path):
    """Load and validate a frozen label sidecar; hard errors, never repairs."""
    labels = _labels_module()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read labels sidecar {path}: {exc}") from exc
    try:
        return labels.validate_sidecar(document)
    except ValueError as exc:
        raise SystemExit(f"labels sidecar is invalid: {exc}") from exc


class _LabelClock:
    """Map decoded frames to label seconds from container presentation times.

    policy ``pts`` uses each frame's presentation time directly (zeroed at the
    first rendered frame of the whole video); ``cfr_verified`` proves every
    consecutive delta equals the first within 1.5 ms and then maps by frame
    index. Both hard-fail when PTS is missing or non-monotonic: there is no
    silent fallback that could mislabel frames under a trusted identity.
    """

    def __init__(self, policy: str):
        if policy not in ("pts", "cfr_verified"):
            raise ValueError("timing policy must be pts or cfr_verified")
        self.policy = policy
        self.first = None
        self.last = None
        self.delta = None
        self.index = -1

    def tick(self, capture, frame_index: int) -> float:
        # Frame 0 legitimately has PTS 0; a backend without PTS support reports
        # 0 for every frame and is caught by the monotonicity check below.
        raw = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if self.last is not None and raw <= self.last + 1e-6:
            raise SystemExit(f"container does not provide usable monotonic frame PTS at frame "
                             f"{frame_index + 1}; label mapping requires monotonic frame PTS")
        self.index += 1
        if self.first is None:
            self.first = raw
        elif self.policy == "cfr_verified":
            delta = raw - self.last
            if self.delta is None:
                self.delta = delta
            elif abs(delta - self.delta) > 0.0015:
                raise SystemExit(f"video is not constant-frame-rate at frame {frame_index + 1}; "
                                 "timing_policy=cfr_verified does not apply")
        self.last = raw
        if self.policy == "cfr_verified" and self.delta is not None:
            return self.index * self.delta
        return raw - self.first


def _load_document(path: Path, camera_path: Path | None = None,
                   fit_shape=None) -> tuple[dict, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load a DAG or Cosmos document; ``fit_shape(side, joints)`` adds Cosmos shapes when given."""
    document = json.loads(path.read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema in ("cosmos-e2e-result/v1", "cosmos-e2e-result/v2"):
        if __package__:
            from .cosmos_e2e import CosmosE2EDocument, CosmosE2EHeadDocument
        else:
            try:
                from cosmos_e2e import CosmosE2EDocument, CosmosE2EHeadDocument
            except ModuleNotFoundError as exc:
                if exc.name != "cosmos_e2e":
                    raise
                # importlib file loading does not put this directory on sys.path.
                from importlib.util import module_from_spec, spec_from_file_location
                spec = spec_from_file_location("cosmos_e2e", Path(__file__).with_name("cosmos_e2e.py"))
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                CosmosE2EDocument = module.CosmosE2EDocument
                CosmosE2EHeadDocument = module.CosmosE2EHeadDocument
        camera = json.loads(camera_path.read_text(encoding="utf-8")) if camera_path is not None else None
        converter = CosmosE2EDocument if schema == "cosmos-e2e-result/v1" else CosmosE2EHeadDocument
        document = converter.convert(document, camera, fit_shape=fit_shape)
    elif camera_path is not None:
        raise SystemExit("--camera is only supported for Cosmos e2e JSON")
    metadata = document["Metadata"]
    frames = document["逐帧轨迹"]
    frame_count = int(metadata["帧数"])
    if len(frames) != frame_count:
        raise SystemExit(f"逐帧轨迹 length {len(frames)} != Metadata.帧数 {frame_count}")
    intrinsic = np.asarray(metadata["相机内参"], dtype=np.float64) if metadata["相机内参"] is not None else np.full((3, 3), np.nan)
    poses = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    joints = [np.full((frame_count, 21, 3), np.nan), np.full((frame_count, 21, 3), np.nan)]
    joints_px = [np.full((frame_count, 21, 2), np.nan), np.full((frame_count, 21, 2), np.nan)]
    betas = [np.full((frame_count, 10), np.nan), np.full((frame_count, 10), np.nan)]
    valids = [np.zeros(frame_count, dtype=bool), np.zeros(frame_count, dtype=bool)]
    pose_valid = np.zeros(frame_count, dtype=bool)
    for index, entry in enumerate(frames):
        frame = entry[f"帧{index + 1}"]
        pose = frame["相机世界位姿"]
        if pose is not None:
            poses[index] = np.asarray(pose, dtype=np.float64)
            pose_valid[index] = np.isfinite(poses[index]).all()
        for side, key, flag in ((0, "手0", "左手是否存在"), (1, "手1", "右手是否存在")):
            valids[side][index] = bool(frame[flag])
            hand = frame[key]
            if hand is None:
                continue
            if hand.get("手部21点坐标") is not None:
                joints[side][index] = np.asarray(hand["手部21点坐标"], dtype=np.float64)
            if hand.get("手部21点投影坐标") is not None:
                joints_px[side][index] = np.asarray(hand["手部21点投影坐标"], dtype=np.float64)
            if hand.get("beta参数") is not None:
                betas[side][index] = np.asarray(hand["beta参数"], dtype=np.float64)
    return document, intrinsic, poses, joints, joints_px, betas, pose_valid


def _has_beta(beta: object) -> bool:
    value = np.asarray(beta, dtype=np.float64)
    return value.shape == (10,) and bool(np.isfinite(value).all())


def _load_assets_if_needed(asset_dir: Path, joints: list[np.ndarray], betas: list[np.ndarray]) -> HandSurfaceAssets | None:
    for side in range(2):
        hand_valid = np.isfinite(joints[side]).all(axis=(1, 2))
        beta_valid = np.isfinite(betas[side]).all(axis=1)
        if np.any(hand_valid & beta_valid):
            return HandSurfaceAssets(asset_dir)
    return None


def _is_cosmos(document: dict) -> bool:
    return document["Metadata"].get("source_schema") in ("cosmos-e2e-result/v1", "cosmos-e2e-result/v2")


def _output_fps(document: dict, requested: float | None, legacy_fps: float | None = None) -> float:
    if _is_cosmos(document):
        fps = float(document["Metadata"]["fps"])
        if requested is not None and not math.isclose(float(requested), fps, rel_tol=0.0, abs_tol=1e-6):
            raise SystemExit(f"--fps must equal Cosmos metadata fps ({fps:g})")
    else:
        if requested is None and legacy_fps is None:
            raise SystemExit("--fps is required for legacy DAG JSON")
        fps = float(requested if requested is not None else legacy_fps)
    if not np.isfinite(fps) or fps <= 0:
        raise SystemExit("--fps must be finite and positive")
    return fps


def _validate_source_video(document: dict, capture) -> None:
    """Cosmos indices refer to the exact processed input, including I0."""
    if not _is_cosmos(document):
        return
    metadata = document["Metadata"]
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if count != int(metadata["帧数"]):
        raise SystemExit(f"Cosmos processed video frame count {count} != {metadata['帧数']} (including I0)")
    expected_fps = float(metadata["fps"])
    if not np.isfinite(fps) or not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=0.01):
        raise SystemExit(f"Cosmos requires the processed {expected_fps:g}fps video, got {fps:g}fps")
    expected_size = metadata.get("source_size_wh")
    actual_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    if expected_size is not None and actual_size != list(expected_size):
        raise SystemExit(f"Cosmos processed video dimensions {actual_size} != calibration/input size {expected_size}; no implicit resizing")


def _source_skeleton_pixels(joints_world: np.ndarray, direct_pixels: np.ndarray, pose: np.ndarray,
                            intrinsic: np.ndarray, pose_valid: bool) -> np.ndarray:
    pixels = np.asarray(direct_pixels, dtype=np.float64).copy()
    if pose_valid:
        camera = _camera_coordinates(joints_world, pose)
        front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
        if not np.isfinite(pixels).all() and np.isfinite(intrinsic).all():
            pixels, _, _ = _project_camera(camera, intrinsic)
        pixels[~front] = np.nan
    return pixels


def _draw_projection_notice(canvas: np.ndarray, intrinsic: np.ndarray) -> None:
    if not np.isfinite(intrinsic).all():
        cv2.putText(canvas, "source projection unavailable: no camera calibration", (16, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)


def _render_source_hand(source: np.ndarray, side: int, joints_world: np.ndarray, direct_pixels: np.ndarray,
                        beta: np.ndarray, pose: np.ndarray, intrinsic: np.ndarray, pose_valid: bool,
                        assets: HandSurfaceAssets | None) -> tuple[np.ndarray | None, int]:
    if not np.isfinite(joints_world).all():
        return None, 0
    side_name = "left" if side == 0 else "right"
    mesh = assets.reconstruct(side_name, beta, joints_world) if assets is not None and _has_beta(beta) else None
    drawn = 0
    if mesh is not None and pose_valid and np.isfinite(intrinsic).all():
        mesh_camera = _camera_coordinates(mesh, pose)
        mesh_pixels, _, _ = _project_camera(mesh_camera, intrinsic)
        projected_joints, _, _ = _project_camera(_camera_coordinates(joints_world, pose), intrinsic)
        if np.isfinite(direct_pixels).all():
            fit = _anchored_affine(projected_joints, direct_pixels)
            if fit is not None:
                mesh_pixels = _apply_anchored_affine(mesh_pixels, projected_joints[0], fit)
                warped_joints = _apply_anchored_affine(projected_joints, projected_joints[0], fit)
                joint_residual = direct_pixels[MANO16_TO_JSON21] - warped_joints[MANO16_TO_JSON21]
                mesh_pixels += assets.models[side_name]["weights"] @ joint_residual
        drawn = _draw_mesh_pixels(source, mesh_pixels, mesh_camera, assets.faces(side_name), HAND_COLORS[side])
    skeleton_pixels = _source_skeleton_pixels(joints_world, direct_pixels, pose, intrinsic, pose_valid)
    _draw_source_skeleton(source, skeleton_pixels, HAND_COLORS[side])
    return mesh, drawn


def _render_world(size: tuple[int, int], pose: np.ndarray, centers: np.ndarray, frame_index: int,
                  frame_joints: list[np.ndarray], meshes: list[np.ndarray | None], assets: HandSurfaceAssets | None,
                  frame_betas: list[np.ndarray] | None = None, intrinsic: np.ndarray | None = None,
                  source_size: tuple[int, int] | None = None,
                  view: dict[str, np.ndarray | float] | None = None,
                  floor: dict[str, np.ndarray] | None = None) -> np.ndarray:
    width, height = size
    canvas = _spatial_background(width, height)
    # Keep the reference cinema-camera proportions, but make the object small
    # enough to read as a spatial-view anchor rather than a dominant prop.
    cam_verts, cam_faces = _build_cinema_camera(s=0.085)
    cam_world = (np.asarray(pose[:3, :3], dtype=np.float64) @ cam_verts.T).T + pose[:3, 3]
    if view is None or floor is None:
        # Single-frame callers (tests, stills) solve the view from this one
        # pose; the render loop passes the clip-wide smoothed track instead.
        single_view, report = _single_pose_view(pose, frame_joints, cam_verts, width, height)
        view = single_view if view is None else view
        floor = report["floor"] if floor is None else floor
    _draw_floor_grid(canvas, floor, view)

    skin_color = SKIN_COLOR
    for side in range(2):
        joints_world = frame_joints[side]
        if np.asarray(joints_world).shape != (21, 3) or not np.isfinite(joints_world).all():
            continue
        mesh = meshes[side]
        if mesh is not None:
            pixels, _, _, xyz = _project_view(mesh, view, (width, height))
            _draw_mesh_pixels(canvas, pixels, xyz, assets.faces("left" if side == 0 else "right"), skin_color, alpha=1.0, edges=False)
        elif frame_betas is None or not _has_beta(frame_betas[side]):
            _draw_world_skeleton(canvas, joints_world, view, HAND_COLORS[side])

    # Painter order matches the spatial view: the camera is the anchor and its
    # small RGB axis arrows sit on top of the dark body.
    pixels, _, _, xyz = _project_view(cam_world, view, (width, height))
    _draw_mesh_pixels(canvas, pixels, xyz, cam_faces, CAMERA_COLOR, alpha=1.0, edges=False)
    # The trail is drawn AFTER the opaque camera body and before the axis arrows.
    # Drawing it first buried it completely: the path spans ~0.036 m against a
    # 0.193 m camera prop, so in these near-static clips it projects entirely
    # inside the body silhouette (measured: 0 visible trail pixels at every
    # frame, 130 painted then fully overpainted at the midpoint). In the RRD
    # reference the blue strip likewise rides over the black body, which is a
    # painter-order property rather than a scale one -- no geometric
    # exaggeration is applied here.
    trajectory = _trajectory_upto(centers, frame_index)
    if len(trajectory) >= 2:
        _draw_view_line(canvas, trajectory, view, TRAIL_COLOR, 2)
    _draw_camera_marker(canvas, pose, view)
    return canvas


# Three-pane layout: overlay | follow view | upper-body skeleton, approved at 1920 px
# as 760 / 580 / 580 columns over one full-width caption strip.
_THREE_PANE_COLUMNS = (760, 580, 580)
_THREE_PANE_TITLES = ("Hand joints overlay", "Camera and hands: side-rear follow view", "Upper-body skeleton")
_THREE_PANE_REF = (1920, 914)


def _caption_band(output_height: int) -> int:
    return min(220, max(165, int(round(output_height * 0.145))))


def _three_pane_layout(width: int, height: int) -> dict[str, object]:
    """Pane rectangles (x, y, w, h): equal height, the columns fill the width."""
    band = _caption_band(height)
    pane_height = height - band
    total = float(sum(_THREE_PANE_COLUMNS))
    first = int(round(width * _THREE_PANE_COLUMNS[0] / total))
    second = int(round((width - first) * _THREE_PANE_COLUMNS[1] / (total - _THREE_PANE_COLUMNS[0])))
    widths = (first, second, width - first - second)
    xs = (0, first, first + second)
    return {"band": band, "pane_height": pane_height,
            "panes": [(x, 0, w, pane_height) for x, w in zip(xs, widths)]}


def _body_pane_module():
    """The sibling body_pane module, both as a package import and as a script."""
    if __package__:
        from . import body_pane
        return body_pane
    try:
        import body_pane
    except ModuleNotFoundError as exc:
        if exc.name != "body_pane":
            raise
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("body_pane", Path(__file__).with_name("body_pane.py"))
        body_pane = module_from_spec(spec)
        spec.loader.exec_module(body_pane)
    return body_pane


def _overlay_crop(joints_px: list[np.ndarray], source_size: tuple[int, int], pane_size: tuple[int, int],
                  start: int, stop: int) -> tuple[int, int, int, int]:
    """Static source window (x, y, w, h) with the pane's aspect, centred on the hands.

    The largest window of the pane's aspect that fits the source, positioned once for
    the clip at the middle of the annotated hand pixel range (frame centre without
    hand pixels), so the overlay fills its pane without letterboxing or panning.
    """
    source_w, source_h = source_size
    aspect = pane_size[0] / max(pane_size[1], 1)
    if source_w / max(source_h, 1) > aspect:
        crop_h, crop_w = source_h, min(source_w, max(1, int(round(source_h * aspect))))
    else:
        crop_w, crop_h = source_w, min(source_h, max(1, int(round(source_w / aspect))))
    pixels = np.concatenate([np.asarray(side[start:stop], dtype=np.float64).reshape(-1, 2) for side in joints_px])
    pixels = pixels[np.isfinite(pixels).all(axis=1)]
    inside = pixels[(pixels[:, 0] >= 0) & (pixels[:, 0] < source_w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < source_h)]
    if len(inside):
        centre = 0.5 * (inside.min(axis=0) + inside.max(axis=0))
    else:
        centre = np.asarray((source_w * 0.5, source_h * 0.5))
    x = int(np.clip(round(centre[0] - crop_w / 2), 0, source_w - crop_w))
    y = int(np.clip(round(centre[1] - crop_h / 2), 0, source_h - crop_h))
    return x, y, crop_w, crop_h


def _draw_pane_title(canvas: np.ndarray, x: int, text: str, scale: float) -> None:
    """Pane title in the SDK title style: white Hershey simplex over a black outline."""
    thickness = max(1, int(round(2 * scale)))
    origin = (x + int(round(16 * scale)), int(round(40 * scale)))
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.78 * scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.78 * scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _download_video_address(address: str) -> tuple[Path, Path | None]:
    parsed = urlparse(address)
    if parsed.scheme not in ("http", "https"):
        return Path(address).expanduser(), None

    temp_dir = Path(tempfile.mkdtemp(prefix="ego-render-video-"))
    suffix = Path(parsed.path).suffix or ".mp4"
    destination = temp_dir / f"source{suffix}"
    request = Request(address, headers={"User-Agent": "ego-json-renderer/1.0"})
    received = 0
    limit = 8 * 1024 * 1024 * 1024
    try:
        with urlopen(request, timeout=60) as response, destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > limit:
                    raise RuntimeError("remote video exceeds the 8 GiB limit")
                output.write(chunk)
        if received == 0:
            raise RuntimeError("remote video download is empty")
        return destination, temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotation_json", type=Path, help="v3 annotation or Cosmos e2e JSON")
    parser.add_argument("--camera", type=Path, default=None, help="Cosmos calibration JSON for the exact processed video")
    parser.add_argument("--video", required=True, help="source video (Cosmos: exact processed input): path or HTTP(S) URL")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output MP4")
    parser.add_argument("--mano-dir", type=Path, default=Path(__file__).resolve().parent / "mano", help="directory containing bundled hand-model assets")
    parser.add_argument("--fps", type=float, default=None, help="output FPS; required for DAG, must equal metadata for Cosmos")
    parser.add_argument("--width", type=int, default=None, help="output width; defaults to source width")
    parser.add_argument("--height", type=int, default=None, help="output height; defaults to source height")
    parser.add_argument("--start-frame", type=int, default=1, help="first 1-based frame to render")
    parser.add_argument("--max-frames", type=int, default=None, help="optional number of frames to render")
    parser.add_argument("--labels", type=Path, default=None,
                        help="AR label sidecar (ego-anno-ar-labels/v1) to draw time-synced in the caption band")
    parser.add_argument("--timing-policy", choices=("pts", "cfr_verified"), default="pts",
                        help="label time mapping: container frame PTS (default) or verified-CFR index mapping")
    parser.add_argument("--world-up", type=str, default=None,
                        help="optional gravity up vector 'x,y,z' in the result's world frame; estimated when omitted")
    parser.add_argument("--layout", choices=("three-pane", "two-pane"), default="three-pane",
                        help="three-pane (default, 1920x1080): overlay | follow view | upper-body skeleton; "
                             "two-pane: overlay | follow view at the source size")
    args = parser.parse_args()
    three_pane = args.layout == "three-pane"
    world_up = None
    if args.world_up is not None:
        try:
            world_up = np.asarray([float(value) for value in args.world_up.split(",")], dtype=np.float64)
        except ValueError:
            raise SystemExit("--world-up must be three comma-separated numbers")
        if world_up.shape != (3,) or not np.isfinite(world_up).all() or np.linalg.norm(world_up) < 1e-9:
            raise SystemExit("--world-up must be a finite non-zero 'x,y,z' vector")

    # Cosmos results carry joints but no shape: with hand assets on hand the
    # converter fits one beta per side from the clip, otherwise it stays None
    # and both panes draw the measured skeletons.
    assets = HandSurfaceAssets(args.mano_dir) if HandSurfaceAssets.present(args.mano_dir) else None
    document, intrinsic, poses, joints, joints_px, betas, pose_valid = _load_document(
        args.annotation_json, args.camera, fit_shape=assets.fit_shape if assets is not None else None)
    frame_count = int(document["Metadata"]["帧数"])
    subtitles = document.get("字幕", [])
    sidecar = _load_labels(args.labels) if args.labels is not None else None
    if sidecar is None:
        label_segments = None
        label_notice = "AR labels: not provided"
        label_prefix = "AR: "
    else:
        label_segments = sidecar["segments"]
        label_notice = None if label_segments else "AR labels: no labeled intervals"
        label_prefix = "AR: " if sidecar["target_identity_sha256"] else "AR (unverified): "
    if assets is None:
        assets = _load_assets_if_needed(args.mano_dir, joints, betas)

    try:
        video_path, downloaded_video_dir = _download_video_address(args.video)
    except Exception as exc:
        raise SystemExit(f"cannot download video: {exc}") from exc
    if not video_path.is_file():
        if downloaded_video_dir is not None:
            shutil.rmtree(downloaded_video_dir, ignore_errors=True)
        raise SystemExit(f"video does not exist: {video_path}")

    capture = None
    source_width = source_height = None
    fps = _output_fps(document, args.fps)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        if downloaded_video_dir is not None:
            shutil.rmtree(downloaded_video_dir, ignore_errors=True)
        raise SystemExit(f"cannot open video: {video_path}")
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _validate_source_video(document, capture)

    if three_pane:
        output_width = int(args.width or 1920)
        output_height = int(args.height or 1080)
    else:
        output_width = int(args.width or source_width or 1920)
        output_height = int(args.height or source_height or 1080)
    output_width -= output_width % 2
    output_height -= output_height % 2
    band_height = _caption_band(output_height)
    pane_height = output_height - band_height
    pane_width = output_width // 2
    if three_pane:
        layout = _three_pane_layout(output_width, output_height)
        overlay_rect, world_rect, body_rect = layout["panes"]
        world_size = (world_rect[2], pane_height)
        title_scale = min(pane_height / _THREE_PANE_REF[1], output_width / _THREE_PANE_REF[0])
    else:
        world_size = (pane_width, pane_height)

    start = max(0, int(args.start_frame) - 1)
    stop = frame_count if args.max_frames is None else min(frame_count, start + max(0, int(args.max_frames)))
    if start >= stop:
        raise SystemExit("empty render range")
    if capture is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
    if not writer.isOpened():
        raise SystemExit(f"cannot open writer: {args.output}")

    centers = np.where(pose_valid[:, None], poses[:, :3, 3], np.nan)
    # The observer track is solved offline from every valid camera pose and
    # every hand joint, so per-frame rendering only looks its view up.
    world_cam_verts, _ = _build_cinema_camera(s=0.085)
    views, track = _follow_view_track(poses, pose_valid, joints, world_cam_verts, world_size[0], world_size[1],
                                      fps=fps, world_up=world_up)
    if three_pane:
        # Same up vector as the follow view: both 3D panes stand on one gravity.
        body_pane = _body_pane_module().BodyPane(poses, pose_valid, joints, track["up"], fps,
                                                 (body_rect[2], pane_height))
        crop = _overlay_crop(joints_px, (source_width, source_height), (overlay_rect[2], pane_height),
                             start, stop)
    gravity = track["gravity"]
    print("world-up: branch={} eigen_ratio={:.2f} up=[{:.4f}, {:.4f}, {:.4f}] observer_side={}".format(
        gravity["branch"], float(gravity["eigen_ratio"]), *[float(v) for v in track["up"]],
        "camera-left" if track["side"] < 0 else "camera-right"))
    cursor = 0
    while cursor < len(subtitles) and int(subtitles[cursor]["起始帧"][1]) < start + 1:
        cursor += 1
    # Labels are indexed by the full video's presentation timeline: decoding
    # starts at frame 0 whenever labels are present, even for a subrange render.
    if label_segments is not None and start > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    clock = _LabelClock(args.timing_policy) if label_segments is not None else None
    source_mesh_faces = 0
    world_mesh_faces = 0
    try:
        for frame_index in range(0 if clock is not None else start, stop):
            ok, source = capture.read()
            if not ok:
                if _is_cosmos(document):
                    raise SystemExit(f"cannot decode Cosmos processed video frame {frame_index}")
                source = np.full((source_height or output_height, source_width or output_width, 3), 35, dtype=np.uint8)
            if clock is not None:
                label_time = clock.tick(capture, frame_index)
                if frame_index < start:
                    continue
                label_lines = _label_lines_for_frame(label_segments, label_time, label_prefix, output_width - 36)
            else:
                label_lines = None

            frame_joints: list[np.ndarray] = [joints[0][frame_index], joints[1][frame_index]]
            meshes: list[np.ndarray | None] = [None, None]
            for side in range(2):
                meshes[side], drawn = _render_source_hand(
                    source, side, frame_joints[side], joints_px[side][frame_index],
                    betas[side][frame_index], poses[frame_index], intrinsic, bool(pose_valid[frame_index]), assets,
                )
                source_mesh_faces += drawn
            _draw_projection_notice(source, intrinsic)

            if pose_valid[frame_index]:
                world = _render_world(world_size, poses[frame_index], centers, frame_index, frame_joints, meshes, assets,
                                      [betas[0][frame_index], betas[1][frame_index]], intrinsic,
                                      (source_width, source_height), view=views[frame_index],
                                      floor=track["floor"])
                for side, side_name in ((0, "left"), (1, "right")):
                    if meshes[side] is not None:
                        world_mesh_faces += len(assets.faces(side_name))
            else:
                world = np.full((world_size[1], world_size[0], 3), BG, dtype=np.uint8)
                cv2.putText(world, "camera pose unavailable", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

            structured, narrative, cursor = _subtitle_for_frame(
                subtitles, frame_index, cursor, cosmos=_is_cosmos(document)
            )
            canvas = np.full((output_height, output_width, 3), 20, dtype=np.uint8)
            if three_pane:
                x, y, w, h = crop
                canvas[:pane_height, :overlay_rect[2]] = cv2.resize(
                    source[y:y + h, x:x + w], (overlay_rect[2], pane_height), interpolation=cv2.INTER_AREA)
                canvas[:pane_height, world_rect[0]:world_rect[0] + world_rect[2]] = world
                faces = [assets.faces(name) if meshes[side] is not None else None
                         for side, name in ((0, "left"), (1, "right"))]
                canvas[:pane_height, body_rect[0]:body_rect[0] + body_rect[2]] = body_pane.render(
                    frame_index, frame_joints, meshes, faces)
                for rect in (world_rect, body_rect):
                    canvas[:pane_height, max(0, rect[0] - 1):rect[0] + 1] = 60
                for rect, title in zip(layout["panes"], _THREE_PANE_TITLES):
                    _draw_pane_title(canvas, rect[0], title, title_scale)
            else:
                canvas[:pane_height, :pane_width] = _resize_to_contain(source, pane_width, pane_height)
                canvas[:pane_height, pane_width:pane_width * 2] = world
                title_size = max(30, int(round(pane_height * 0.043)))
                left_title = "Hand joints overlay"
                right_title = "Camera and hands: side-rear follow view"
                _put_text_mixed(canvas, left_title, (24, 64), title_size, (255, 255, 255), 2)
                _put_text_mixed(canvas, right_title, (pane_width + 24, 64), title_size, (255, 255, 255), 2)
            _draw_subtitle(canvas, structured, narrative, frame_index, frame_count, band_height,
                           label_lines=label_lines, label_notice=label_notice)
            writer.write(canvas)
    finally:
        writer.release()
        if capture is not None:
            capture.release()
        if downloaded_video_dir is not None:
            shutil.rmtree(downloaded_video_dir, ignore_errors=True)

    rendered = stop - start
    print(f"wrote {args.output} ({rendered} frames, source frames {start + 1}-{stop}, {output_width}x{output_height} @ {fps:g}fps)")
    print(f"mesh triangles drawn: source={source_mesh_faces}, world_candidates={world_mesh_faces}")
    if sidecar is not None:
        print(json.dumps({"labeling_sha256": sidecar["content_sha256"],
                          "render_recipe_version": 1}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
