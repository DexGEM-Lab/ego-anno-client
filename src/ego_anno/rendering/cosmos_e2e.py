"""Convert physical CosmosHandPose147 predictions to the existing render view.

This does not normalize actions, estimate intrinsics, classify visibility, or
fit a MANO surface. The world origin is camera I0; action row k supplies the
camera delta I_k -> I_(k+1) and hands at I_(k+1), never hands at I0.
"""
from __future__ import annotations

from typing import Any

import numpy as np


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
    def convert(cls, document: dict, camera: dict | None = None) -> dict[str, Any]:
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
            for index in range(count):
                hands[side_index].append({
                    "左或右": side,
                    "beta参数": None,
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
