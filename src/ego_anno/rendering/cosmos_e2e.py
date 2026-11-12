"""Convert physical CosmosHandPose147 predictions to the existing render view.

This does not normalize actions, estimate intrinsics, or classify visibility.
The world origin is camera I0; action row k supplies the camera delta
I_k -> I_(k+1) and hands at I_(k+1), never hands at I0.

Cosmos predicts joints, not a hand shape. Both converters accept an optional
``fit_shape(side, joints_camera)`` callable (the renderer passes
``HandSurfaceAssets.fit_shape`` when hand assets are available) that returns
one shape vector for the whole clip; it is written to every frame's
``beta参数`` so the surface path renders. Without it ``beta参数`` stays None.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

ShapeFitter = Callable[[str, np.ndarray], np.ndarray]


def _clip_shape(fit_shape: ShapeFitter | None, side: str, joints_camera: np.ndarray) -> list[float] | None:
    """Fit one finite 10-vector for ``side`` from the clip's camera-frame joints, or None."""
    if fit_shape is None:
        return None
    beta = np.asarray(fit_shape(side, np.asarray(joints_camera, dtype=np.float64)), dtype=np.float64)
    if beta.shape != (10,) or not np.isfinite(beta).all():
        raise ValueError(f"hand shape fit for the {side} hand must return 10 finite values")
    return beta.tolist()


class CosmosE2EDocument:
    SCHEMA = "cosmos-e2e-result/v1"
    TIP_JOINTS = (4, 8, 12, 16, 20)
    OTHER_JOINTS = (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)
    # Renderer hand0 is left; Cosmos serializes the right hand first.
    HAND_BLOCKS = (("left", 33, 42, 102), ("right", 9, 18, 57))

    @staticmethod
    def _rotations(values: np.ndarray) -> np.ndarray:
        """Frozen evaluation convention: columns + cross product, SVD onto SO(3).

        Do not reshape six channels as a row-major 3x2 matrix or use a
        Gram-Schmidt decoder: those change the predicted wrist geometry.
        """
        first, second = values[:, :3], values[:, 3:]
        first_norm = np.linalg.norm(first, axis=-1)
        orthogonal = second - (
            np.sum(first * second, axis=-1, keepdims=True)
            / np.maximum(first_norm[:, None] ** 2, 1e-30)
        ) * first
        if np.any(first_norm < 1e-8) or np.any(np.linalg.norm(orthogonal, axis=-1) < 1e-8):
            raise ValueError("Cosmos rot6d columns must be nonzero and noncollinear")
        matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
        u, _, vt = np.linalg.svd(matrix)
        u[:, :, 2] *= np.sign(np.linalg.det(u @ vt))[:, None]
        return u @ vt

    @staticmethod
    def _camera(camera: dict | None) -> tuple[np.ndarray | None, list[int] | None]:
        if camera is None:
            return None, None
        if not isinstance(camera, dict):
            raise ValueError("camera must contain source_size_wh and source_K")
        size = camera.get("source_size_wh")
        if (
            not isinstance(size, list) or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
        ):
            raise ValueError("camera.source_size_wh must contain two positive integer dimensions")
        intrinsic = np.asarray(camera.get("source_K"), dtype=np.float64)
        if (
            intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-9)
            or not np.isclose(intrinsic[0, 1], 0, rtol=0, atol=1e-9)
            or not np.isclose(intrinsic[1, 0], 0, rtol=0, atol=1e-9)
        ):
            raise ValueError("camera.source_K must be a finite zero-skew pinhole matrix")
        return intrinsic, list(size)

    @staticmethod
    def _project(points: np.ndarray, intrinsic: np.ndarray | None) -> list | None:
        if intrinsic is None:
            return None
        positive = points[:, 2] > 1e-9
        projected = points @ intrinsic.T
        pixels = np.full((21, 2), np.nan)
        pixels[positive] = projected[positive, :2] / projected[positive, 2:3]
        return [row.tolist() if np.isfinite(row).all() else [None, None] for row in pixels]

    @classmethod
    def convert(cls, document: dict, camera: dict | None = None,
                fit_shape: ShapeFitter | None = None) -> dict[str, Any]:
        if document.get("schema") != cls.SCHEMA:
            raise ValueError(f"expected {cls.SCHEMA}")
        actions = np.asarray(document.get("action"), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 147 or len(actions) < 1:
            raise ValueError("Cosmos action must be a nonempty physical [N-1,147] array")
        if not np.isfinite(actions).all():
            raise ValueError("Cosmos physical actions contain non-finite values")
        count = len(actions)
        if (
            type(document.get("num_frames")) is not int
            or document["num_frames"] != count + 1
            or type(document.get("num_actions")) is not int
            or document["num_actions"] != count
            or document.get("action_dim") != 147
        ):
            raise ValueError("Cosmos num_frames/num_actions/action_dim disagree with the action array")
        fps = document.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cosmos fps must be finite and positive")
        destination_times = document.get("frame_alignment", {}).get("destination_timestamps_seconds")
        if destination_times is not None:
            times = np.asarray(destination_times, dtype=np.float64)
            if times.shape != (count,) or not np.allclose(
                times, np.arange(1, count + 1) / fps, rtol=0, atol=1e-6
            ):
                raise ValueError("Cosmos destination timestamps must align action k to frame k+1")
        intrinsic, source_size = cls._camera(camera)

        deltas = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
        deltas[:, :3, :3] = cls._rotations(actions[:, 3:9])
        deltas[:, :3, 3] = actions[:, :3]
        poses = np.repeat(np.eye(4, dtype=np.float64)[None], count + 1, axis=0)
        for index, delta in enumerate(deltas):
            poses[index + 1] = poses[index] @ delta
        hands: list[list[dict | None]] = [[None], [None]]
        for side_index, (side, wrist, tips, rest) in enumerate(cls.HAND_BLOCKS):
            wrist_rotation = cls._rotations(actions[:, wrist + 3:wrist + 9])
            wrist_translation = actions[:, wrist:wrist + 3]
            local = np.zeros((count, 21, 3), dtype=np.float64)
            local[:, cls.TIP_JOINTS, :] = actions[:, tips:tips + 15].reshape(count, 5, 3)
            local[:, cls.OTHER_JOINTS, :] = actions[:, rest:rest + 45].reshape(count, 15, 3)
            camera_joints = np.einsum("tij,tnj->tni", wrist_rotation, local) + wrist_translation[:, None]
            world_joints = (
                np.einsum("tij,tnj->tni", poses[1:, :3, :3], camera_joints)
                + poses[1:, None, :3, 3]
            )
            wrist_poses = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
            wrist_poses[:, :3, :3] = wrist_rotation
            wrist_poses[:, :3, 3] = wrist_translation
            world_wrists = poses[1:] @ wrist_poses
            beta = _clip_shape(fit_shape, side, camera_joints)
            for index in range(count):
                hands[side_index].append({
                    "左或右": side,
                    "beta参数": beta,
                    "手部21点坐标": world_joints[index].tolist(),
                    "手部21点投影坐标": cls._project(camera_joints[index], intrinsic),
                    "整手姿态": world_wrists[index].tolist(),
                })
        frames = []
        for index, pose in enumerate(poses):
            frames.append({f"帧{index + 1}": {
                "相机世界位姿": pose.tolist(),
                "左手是否存在": index > 0,
                "右手是否存在": index > 0,
                "左手是否可见": None,
                "右手是否可见": None,
                "手0": hands[0][index],
                "手1": hands[1][index],
            }})
        return {
            "Metadata": {
                "schema": "ego.annotation.product_json.v3",
                "source_schema": cls.SCHEMA,
                "帧数": count + 1,
                "相机内参": None if intrinsic is None else intrinsic.tolist(),
                "完成状态": "完成",
                "fps": fps,
                "source_size_wh": source_size,
                "hand_presence": "Predicted joints only; visibility is not provided by this model.",
                "world_origin": "Camera I0; trajectory integrated from predicted camera deltas.",
            },
            "逐帧轨迹": frames,
            "字幕": [],
        }


class CosmosE2EHeadDocument:
    """Convert a ``cosmos-e2e-result/v2`` (hand+head, 156-D action) result.

    The v2 result carries the final head trajectory explicitly: ``c2w`` is
    already BA-refined where the service selected it, in metres, with frame 0
    the identity. This converter uses that trajectory verbatim and never
    integrates ``action[:, 147:156]``: that head anchor is relative to the
    first frame of the window owning each row (see ``frame_alignment``), so
    integrating it across window cuts of a long video would be wrong.
    Hands come from ``joints_camera`` (row k holds the joints in the camera
    frame of video frame k+1); there is no hand prediction for frame 0.
    """
    SCHEMA = "cosmos-e2e-result/v2"

    @classmethod
    def convert(cls, document: dict, camera: dict | None = None,
                fit_shape: ShapeFitter | None = None) -> dict[str, Any]:
        if document.get("schema") != cls.SCHEMA:
            raise ValueError(f"expected {cls.SCHEMA}")
        count = document.get("num_frames")
        if type(count) is not int or count < 2:
            raise ValueError("Cosmos v2 num_frames must be an integer of at least 2")
        if (type(document.get("num_actions")) is not int or document["num_actions"] != count - 1
                or document.get("action_dim") != 156):
            raise ValueError("Cosmos v2 num_actions/action_dim disagree with num_frames")
        actions = np.asarray(document.get("action"), dtype=np.float64)
        if actions.shape != (count - 1, 156) or not np.isfinite(actions).all():
            raise ValueError("Cosmos v2 action must be a finite physical [N-1,156] array")
        fps = document.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cosmos fps must be finite and positive")
        destination_times = document.get("frame_alignment", {}).get("destination_timestamps_seconds")
        if destination_times is not None:
            times = np.asarray(destination_times, dtype=np.float64)
            if times.shape != (count - 1,) or not np.allclose(
                times, np.arange(1, count) / fps, rtol=0, atol=1e-6
            ):
                raise ValueError("Cosmos destination timestamps must align action k to frame k+1")
        layout = document.get("joints_layout") or {}
        side_order = layout.get("side_order")
        if side_order != ["right", "left"]:
            raise ValueError("Cosmos v2 joints_layout.side_order must be ['right', 'left']")
        joints_camera = np.asarray(document.get("joints_camera"), dtype=np.float64)
        if joints_camera.shape != (count - 1, 2, 21, 3) or not np.isfinite(joints_camera).all():
            raise ValueError("Cosmos v2 joints_camera must be a finite [N-1,2,21,3] array")
        poses = np.asarray(document.get("c2w"), dtype=np.float64)
        if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
            raise ValueError("Cosmos v2 c2w must be a finite [N,4,4] array")
        if not np.allclose(poses[0], np.eye(4), rtol=0, atol=1e-6):
            raise ValueError("Cosmos v2 c2w frame 0 must be the identity")
        if not np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-9):
            raise ValueError("Cosmos v2 c2w must be homogeneous SE(3) matrices")
        rotations = poses[:, :3, :3]
        if (not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), rtol=0, atol=1e-4)
                or not np.allclose(np.linalg.det(rotations), 1.0, rtol=0, atol=1e-4)):
            raise ValueError("Cosmos v2 c2w rotations must be proper")
        intrinsic, source_size = CosmosE2EDocument._camera(camera)

        hands: list[list[dict | None]] = [[None], [None]]
        for side_index, side in enumerate(("left", "right")):
            camera_joints = joints_camera[:, side_order.index(side)]
            world_joints = (
                np.einsum("tij,tnj->tni", rotations[1:], camera_joints)
                + poses[1:, None, :3, 3]
            )
            beta = _clip_shape(fit_shape, side, camera_joints)
            for index in range(count - 1):
                hands[side_index].append({
                    "左或右": side,
                    "beta参数": beta,
                    "手部21点坐标": world_joints[index].tolist(),
                    "手部21点投影坐标": CosmosE2EDocument._project(camera_joints[index], intrinsic),
                    "整手姿态": None,
                })
        frames = []
        for index in range(count):
            frames.append({f"帧{index + 1}": {
                "相机世界位姿": poses[index].tolist(),
                "左手是否存在": index > 0,
                "右手是否存在": index > 0,
                "左手是否可见": None,
                "右手是否可见": None,
                "手0": hands[0][index],
                "手1": hands[1][index],
            }})
        return {
            "Metadata": {
                "schema": "ego.annotation.product_json.v3",
                "source_schema": cls.SCHEMA,
                "帧数": count,
                "相机内参": None if intrinsic is None else intrinsic.tolist(),
                "完成状态": "完成",
                "fps": fps,
                "source_size_wh": source_size,
                "hand_presence": "Predicted joints only; visibility is not provided by this model.",
                "world_origin": "Camera I0; explicit c2w from the v2 result "
                                "(BA-refined where the service selected it).",
            },
            "逐帧轨迹": frames,
            "字幕": [],
        }
