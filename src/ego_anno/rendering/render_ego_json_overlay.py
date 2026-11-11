#!/usr/bin/env python3
"""Render a two-pane overlay from annotation JSON and the source video.

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
HAND_COLORS = ((76, 220, 92), (42, 205, 255))  # BGR: left green, right yellow
BG = (26, 29, 33)
HEAD_COLOR = (60, 80, 255)
OUTLINE = (3, 3, 3)
TRAJ_COLOR = (90, 105, 200)

# MANO's 16 kinematic joints mapped to the JSON/OpenPose-style 21-joint order.
MANO16_TO_JSON21 = np.asarray((0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3), dtype=np.int64)
MANO_PARENTS = np.asarray((-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14), dtype=np.int64)
# Distal MANO joint -> (tip vertex, JSON tip joint).
DISTAL_TIPS = {3: (320, 8), 6: (443, 12), 9: (672, 20), 12: (555, 16), 15: (744, 4)}
FINGER_CHAINS = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15))
PALM_MCP = np.asarray((1, 4, 7, 10, 13), dtype=np.int64)


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


class HandSurfaceAssets:
    """Numpy runtime representation extracted from the bundled left/right PKLs."""

    def __init__(self, asset_dir: Path):
        runtime_path = asset_dir / "mano_runtime.npz"
        for required in (asset_dir / "MANO_LEFT.pkl", asset_dir / "MANO_RIGHT.pkl", runtime_path):
            if not required.is_file():
                raise SystemExit(f"missing bundled hand-model asset: {required}")
        archive = np.load(runtime_path)
        self.models: dict[str, dict[str, np.ndarray]] = {}
        for side in ("left", "right"):
            self.models[side] = {
                "v_template": np.asarray(archive[f"{side}_v_template"], dtype=np.float64),
                "shapedirs": np.asarray(archive[f"{side}_shapedirs"], dtype=np.float64),
                "J_regressor": np.asarray(archive[f"{side}_J_regressor"], dtype=np.float64),
                "weights": np.asarray(archive[f"{side}_weights"], dtype=np.float64),
                "faces": np.asarray(archive[f"{side}_f"], dtype=np.int32),
            }
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
    for index, point in enumerate(pixels):
        if valid[index]:
            p = tuple(np.rint(point).astype(int))
            cv2.circle(canvas, p, 4, OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 2, color, -1, cv2.LINE_AA)


def _camera_marker_points(pose: np.ndarray, scale: float = 0.026) -> tuple[np.ndarray, list[tuple[int, int]]]:
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
    axis_colors = {(0, 5): (70, 100, 255), (0, 6): (70, 230, 90), (0, 7): (255, 210, 80)}
    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        color = axis_colors.get((a, b), HEAD_COLOR)
        pa, pb = tuple(np.rint(pixels[a]).astype(int)), tuple(np.rint(pixels[b]).astype(int))
        cv2.line(canvas, pa, pb, OUTLINE, 4, cv2.LINE_AA)
        cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)


def _rear_view(pose: np.ndarray, frame_joints: list[np.ndarray], width: int, height: int) -> dict[str, np.ndarray | float]:
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    center = np.asarray(pose[:3, 3], dtype=np.float64)
    right = _normalize(rotation[:, 0], np.asarray([1.0, 0.0, 0.0]))
    down = _normalize(rotation[:, 1], np.asarray([0.0, 1.0, 0.0]))
    forward = _normalize(rotation[:, 2], np.asarray([0.0, 0.0, 1.0]))
    valid_hands = [np.asarray(j, dtype=np.float64) for j in frame_joints if np.asarray(j).shape == (21, 3) and np.isfinite(j).all()]
    if valid_hands:
        merged = np.concatenate(valid_hands, axis=0)
        hand_depth = (_camera_coordinates(merged, pose))[:, 2]
        median_depth = float(np.median(hand_depth[hand_depth > 1e-4])) if (hand_depth > 1e-4).any() else 0.5
    else:
        merged = np.empty((0, 3), dtype=np.float64)
        median_depth = 0.5
    observer_distance = float(np.clip(0.66 * median_depth, 0.26, 0.44))
    # Slightly above the camera axis. In camera coordinates +y is image down,
    # so moving by -down raises the observer.
    eye = center - forward * observer_distance - down * (0.20 * observer_distance)
    view: dict[str, np.ndarray | float] = {
        "eye": eye,
        "right": right,
        "up": -down,
        "forward": forward,
        "focal": float(min(width, height) * 0.9),
        "cx": width * 0.5,
        "cy": height * 0.54,
        "light": _normalize(-forward - down, -forward),
    }
    marker, _ = _camera_marker_points(pose)
    required = np.vstack((marker, merged)) if len(merged) else marker
    xyz = _view_coordinates(required, view)
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 1e-5)
    if valid.any():
        nx = xyz[valid, 0] / xyz[valid, 2]
        ny = xyz[valid, 1] / xyz[valid, 2]
        x_min, x_max = float(nx.min()), float(nx.max())
        y_min, y_max = float(ny.min()), float(ny.max())
        range_x = max(x_max - x_min, 1e-4)
        range_y = max(y_max - y_min, 1e-4)
        focal = min(0.58 * width / range_x, 0.50 * height / range_y, 2.8 * min(width, height))
        focal = max(0.20 * min(width, height), focal)
        mean_x = 0.5 * (x_min + x_max)
        mean_y = 0.5 * (y_min + y_max)
        view["focal"] = float(focal)
        view["cx"] = float(width * 0.5 - focal * mean_x)
        view["cy"] = float(height * 0.54 + focal * mean_y)
    return view


def _draw_camera_grid(canvas: np.ndarray, pose: np.ndarray, joints: list[np.ndarray], view: dict[str, np.ndarray | float]) -> None:
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    center = np.asarray(pose[:3, 3], dtype=np.float64)
    right, down, forward = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    valid_hands = [np.asarray(j, dtype=np.float64) for j in joints if np.asarray(j).shape == (21, 3) and np.isfinite(j).all()]
    if valid_hands:
        camera_hand = _camera_coordinates(np.concatenate(valid_hands, axis=0), pose)
        floor_down = float(np.percentile(camera_hand[:, 1], 95) + 0.09)
        z_end = float(max(0.8, np.percentile(camera_hand[:, 2], 98) + 0.18))
        x_half = float(max(0.45, np.percentile(np.abs(camera_hand[:, 0]), 98) + 0.15))
    else:
        floor_down, z_end, x_half = 0.35, 0.8, 0.45
    for x in np.linspace(-x_half, x_half, 9):
        points = np.asarray((center + right * x + down * floor_down, center + right * x + down * floor_down + forward * z_end))
        _draw_view_line(canvas, points, view, (52, 61, 71), 1)
    for z in np.linspace(0.0, z_end, 9):
        points = np.asarray((center - right * x_half + down * floor_down + forward * z, center + right * x_half + down * floor_down + forward * z))
        _draw_view_line(canvas, points, view, (52, 61, 71), 1)


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
        # as tofu boxes in mixed strings (e.g. "手部姿态overlay").
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


def _draw_subtitle(canvas: np.ndarray, structured: str, narrative: str, frame_index: int, frame_count: int, band_height: int) -> None:
    height, width = canvas.shape[:2]
    y0 = height - band_height
    layer = canvas.copy()
    cv2.rectangle(layer, (0, y0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(layer, 0.72, canvas, 0.28, 0.0, canvas)
    cv2.putText(canvas, f"frame {frame_index + 1:06d}/{frame_count}", (18, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (190, 215, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, structured, (18, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (112, 235, 255), 1, cv2.LINE_AA)
    lines = _wrap_text(narrative, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1, width - 36)
    y = y0 + 88
    for line in lines[:3]:
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 248, 248), 1, cv2.LINE_AA)
        y += 27


def _subtitle_for_frame(rows: list[dict], frame_index: int, cursor: int, *, cosmos: bool = False) -> tuple[str, str, int]:
    if cosmos:
        detail = (
            "I0 has no hand prediction; action row 0 begins at the next video frame."
            if frame_index == 0 else
            "Predicted 3D joints; hand visibility and object labels are not provided."
        )
        return "Cosmos | green=left, yellow=right | skeleton only (no beta)", detail, cursor
    frame_number = frame_index + 1
    while cursor < len(rows) and frame_number > int(rows[cursor]["起始帧"][1]):
        cursor += 1
    if cursor >= len(rows):
        return "left hand: none, right hand: none", "", cursor
    row = rows[cursor]
    start, end = (int(value) for value in row["起始帧"])
    if not (start <= frame_number <= end):
        return "left hand: none, right hand: none", "", cursor

    def object_text(value: object) -> str:
        text = str(value).strip() if value is not None else ""
        return text if text else "none"

    structured = f"left hand: {object_text(row.get('左手物体'))}, right hand: {object_text(row.get('右手物体'))}"
    narrative = str(row.get("总体描述") or "").strip()
    return structured, narrative, cursor


def _load_document(path: Path, camera_path: Path | None = None) -> tuple[dict, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
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
        document = converter.convert(document, camera)
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
                  frame_betas: list[np.ndarray] | None = None) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), BG, dtype=np.uint8)
    view = _rear_view(pose, frame_joints, width, height)
    _draw_camera_grid(canvas, pose, frame_joints, view)
    start = max(0, frame_index - 90)
    trajectory = centers[start:frame_index + 1]
    trajectory = trajectory[np.isfinite(trajectory).all(axis=1)]
    if len(trajectory) >= 2:
        _draw_view_line(canvas, trajectory, view, TRAJ_COLOR, 2)

    skin_color = (160, 205, 240)  # BGR: bright, pale skin tone
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

    # Draw the camera last so it covers hand pixels at every 2D overlap.
    # Painter order gives the requested hierarchy without slow depth rasterization.
    _draw_camera_marker(canvas, pose, view)
    cv2.putText(canvas, "rear overhead view: camera-relative depth order", (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_index + 1:06d} | camera right/down/forward coordinates | units: meters", (16, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 232, 240), 1, cv2.LINE_AA)
    skeleton_visible = any(
        meshes[side] is None and np.isfinite(frame_joints[side]).all()
        and (frame_betas is None or not _has_beta(frame_betas[side]))
        for side in range(2)
    )
    legend = (
        "red=camera | blue=past path | green/yellow=joints"
        if skeleton_visible else "red=camera | blue=past path | skin=hands"
    )
    cv2.putText(canvas, legend, (16, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (205, 215, 226), 1, cv2.LINE_AA)
    return canvas


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
    args = parser.parse_args()

    document, intrinsic, poses, joints, joints_px, betas, pose_valid = _load_document(args.annotation_json, args.camera)
    frame_count = int(document["Metadata"]["帧数"])
    subtitles = document.get("字幕", [])
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

    output_width = int(args.width or source_width or 1920)
    output_height = int(args.height or source_height or 1080)
    output_width -= output_width % 2
    output_height -= output_height % 2
    band_height = min(220, max(165, int(round(output_height * 0.145))))
    pane_height = output_height - band_height
    pane_width = output_width // 2

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
    cursor = 0
    while cursor < len(subtitles) and int(subtitles[cursor]["起始帧"][1]) < start + 1:
        cursor += 1
    source_mesh_faces = 0
    world_mesh_faces = 0
    try:
        for frame_index in range(start, stop):
            ok, source = capture.read()
            if not ok:
                if _is_cosmos(document):
                    raise SystemExit(f"cannot decode Cosmos processed video frame {frame_index}")
                source = np.full((source_height or output_height, source_width or output_width, 3), 35, dtype=np.uint8)

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
                world = _render_world((pane_width, pane_height), poses[frame_index], centers, frame_index, frame_joints, meshes, assets,
                                      [betas[0][frame_index], betas[1][frame_index]])
                for side, side_name in ((0, "left"), (1, "right")):
                    if meshes[side] is not None:
                        world_mesh_faces += len(assets.faces(side_name))
            else:
                world = np.full((pane_height, pane_width, 3), BG, dtype=np.uint8)
                cv2.putText(world, "camera pose unavailable", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2, cv2.LINE_AA)

            structured, narrative, cursor = _subtitle_for_frame(
                subtitles, frame_index, cursor, cosmos=_is_cosmos(document)
            )
            canvas = np.full((output_height, output_width, 3), 20, dtype=np.uint8)
            canvas[:pane_height, :pane_width] = _resize_to_contain(source, pane_width, pane_height)
            canvas[:pane_height, pane_width:pane_width * 2] = world
            title_size = max(30, int(round(pane_height * 0.043)))
            left_title = "Hand joints overlay" if _is_cosmos(document) else "手部姿态overlay"
            right_title = "Camera and hands in 3D" if _is_cosmos(document) else "相机-手 世界坐标位姿"
            _put_text_mixed(canvas, left_title, (24, 64), title_size, (255, 255, 255), 2)
            _put_text_mixed(canvas, right_title, (pane_width + 24, 64), title_size, (255, 255, 255), 2)
            _draw_subtitle(canvas, structured, narrative, frame_index, frame_count, band_height)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
