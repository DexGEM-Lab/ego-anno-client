"""Public offline rendering tests; optional real-media cases use no model assets."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from ego_anno import rendering
from ego_anno.transport import ClientError


def physical_result(count=3):
    """Finite physical hand147 rows with valid camera and wrist rot6d columns."""
    rows = []
    for frame in range(count - 1):
        row = [0.0] * 147
        row[0] = 0.005
        row[3:9] = [1, 0, 0, 0, 1, 0]
        for wrist, tips, rest, x in ((9, 18, 57, 0.10), (33, 42, 102, -0.10)):
            row[wrist:wrist + 3] = [x, 0.01 * frame, 0.7]
            row[wrist + 3:wrist + 9] = [1, 0, 0, 0, 1, 0]
            for finger in range(5):
                row[tips + finger * 3:tips + finger * 3 + 3] = [(finger - 2) * 0.018, -0.09, 0.01]
            for index in range(15):
                row[rest + index * 3:rest + index * 3 + 3] = [(index // 3 - 2) * 0.018,
                                                                           -0.02 * (index % 3 + 1), 0.0]
        rows.append(row)
    return {"schema": "cosmos-e2e-result/v1", "num_frames": count,
            "num_actions": count - 1, "action_dim": 147, "fps": 15, "action": rows,
            "frame_alignment": {"destination_timestamps_seconds": [i / 15 for i in range(1, count)]}}


def physical_result_v2(count=3):
    """Finite cosmos-e2e-result/v2 with explicit c2w and right-first camera joints."""
    rows = []
    for frame in range(count - 1):
        row = [0.0] * 156
        row[3:9] = [1, 0, 0, 0, 1, 0]
        # The head anchor is window-first-frame-relative; the converter must
        # never integrate it. Large finite values make any integration visible.
        row[147:156] = [100.0, 200.0, 300.0, 1, 0, 0, 0, 1, 0]
        rows.append(row)
    joints = []
    for frame in range(count - 1):
        hands = []
        for side in range(2):  # the v2 layout serializes the right hand first
            base = 0.05 if side == 0 else -0.05
            hands.append([[base + 0.001 * joint, -0.002 * joint, 0.5 + 0.001 * joint]
                          for joint in range(21)])
        joints.append(hands)
    poses = []
    for frame in range(count):
        poses.append([[1.0, 0.0, 0.0, 0.01 * frame], [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return {"schema": "cosmos-e2e-result/v2", "num_frames": count,
            "num_actions": count - 1, "action_dim": 156, "fps": 15, "action": rows,
            "joints_camera": joints,
            "joints_layout": {"side_order": ["right", "left"],
                              "joint_convention": "wrist0-thumb1_4-index5_8-middle9_12-ring13_16-little17_20",
                              "coordinate_frame": "camera_absolute",
                              "row_k": "joints in the camera frame of video frame k+1"},
            "c2w": poses,
            "frame_alignment": {"destination_timestamps_seconds": [i / 15 for i in range(1, count)]}}


class BundleCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "saved-inputs"
        self.inputs.mkdir()
        self.result_path = self.root / "annotation.json"
        self.output = self.root / "preview.mp4"
        self.result = physical_result()
        self.submission = {"schema": "ego-anno-e2e-input/v1", "pipeline": "e2e",
                           "num_frames": 3, "fps": 15, "width": 320, "height": 240,
                           "idempotency_key": "3107cf0b-5648-4f1c-9166-a63b4d0c632d",
                           "annotation_job_id": None, "prompt": None}
        self.save_json(self.result_path, self.result)
        self.save_json(self.inputs / "submission.json", self.submission)
        (self.inputs / "video.mp4").write_bytes(b"not a real video in validation-only tests")

    def save_json(self, path, value):
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def render(self, **kwargs):
        return rendering.render_result(self.result_path, self.inputs, self.output, **kwargs)

    def rejected(self, message):
        self.output.write_bytes(b"existing destination")
        with self.assertRaisesRegex(ClientError, message):
            self.render()
        self.assertEqual(self.output.read_bytes(), b"existing destination")
        self.assertFalse(list(self.root.glob(".ego-anno-render-*")))


class RenderValidationTests(BundleCase):
    def test_only_cosmos_schemas_are_accepted(self):
        for result in ({"schema": "ego.annotation.product_json.v3"}, {}, {"schema": "cosmos-e2e-result/v3"}):
            with self.subTest(result=result):
                self.save_json(self.result_path, result)
                self.rejected("only accepts cosmos-e2e-result/v1 or cosmos-e2e-result/v2")

    def test_submission_schema_and_pipeline_are_required(self):
        for field, value in (("schema", "other/v1"), ("pipeline", "dag")):
            with self.subTest(field=field):
                self.save_json(self.inputs / "submission.json", dict(self.submission, **{field: value}))
                self.rejected("schema ego-anno-e2e-input/v1 and pipeline e2e")

    def test_submission_count_must_match_result(self):
        self.submission["num_frames"] = 5
        self.save_json(self.inputs / "submission.json", self.submission)
        self.rejected("frame count disagrees")

    def test_submission_fps_is_validated_without_resampling(self):
        for value in (True, 0, 30, "15"):
            with self.subTest(value=value):
                self.save_json(self.inputs / "submission.json", dict(self.submission, fps=value))
                self.rejected("fps must be 15")
        self.result["fps"] = 30
        self.save_json(self.result_path, self.result)
        self.save_json(self.inputs / "submission.json", self.submission)
        self.rejected("fps disagrees")

    def test_submission_dimensions_are_validated(self):
        for value in (True, 0, -1, 320.0):
            with self.subTest(value=value):
                self.save_json(self.inputs / "submission.json", dict(self.submission, width=value))
                self.rejected("width and height must be positive integers")

    def test_recorded_result_resolution_must_match(self):
        self.result["source_size_wh"] = [640, 480]
        self.save_json(self.result_path, self.result)
        self.rejected("processed resolution disagrees with result")

    def test_calibration_resolution_must_match(self):
        self.save_json(self.inputs / "camera.json", {"source_size_wh": [640, 480]})
        self.rejected("camera.json dimensions disagree")

    def test_missing_video_is_reported(self):
        (self.inputs / "video.mp4").unlink()
        self.rejected("video does not exist")

    def test_invalid_json_is_a_client_error(self):
        self.result_path.write_text("not JSON", encoding="utf-8")
        self.rejected("cannot read result")
        self.save_json(self.result_path, [])
        self.rejected("must be a JSON object")

    def test_output_cannot_overwrite_any_protected_path(self):
        protected = [self.result_path] + [self.inputs / name for name in
                                        ("video.mp4", "camera.json", "submission.json")]
        for path in protected:
            with self.subTest(path=path):
                before = path.read_bytes() if path.exists() else None
                with self.assertRaisesRegex(ClientError, "must not overwrite"):
                    rendering.render_result(self.result_path, self.inputs, path)
                self.assertEqual(path.read_bytes() if path.exists() else None, before)

    def test_symlink_and_hardlink_collisions_are_rejected(self):
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                output = self.root / (kind + ".mp4")
                if kind == "symlink":
                    output.symlink_to(self.inputs / "video.mp4")
                else:
                    os.link(str(self.inputs / "video.mp4"), str(output))
                with self.assertRaisesRegex(ClientError, "must not overwrite"):
                    rendering.render_result(self.result_path, self.inputs, output)

    def test_output_dimensions_are_not_silently_rounded(self):
        for kwargs in ({"width": 319}, {"height": 239}, {"width": True}, {"height": 100}, {"width": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ClientError, "render (width|height|dimensions)"):
                self.render(**kwargs)

    def test_missing_ffmpeg_has_a_clear_error(self):
        with mock.patch.object(rendering.shutil, "which", return_value=None):
            self.rejected("requires system ffmpeg")

    def test_importing_rendering_does_not_import_optional_dependencies(self):
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'cv2', 'PIL'):
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import ego_anno.rendering
"""
        environment = dict(os.environ)
        environment.pop("EGO_ANNO_API_KEY", None)
        environment["PYTHONPATH"] = str(Path(rendering.__file__).resolve().parents[2])
        completed = subprocess.run([sys.executable, "-c", code], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_subprocess_environment_omits_api_keys(self):
        with mock.patch.dict(os.environ, {"EGO_ANNO_API_KEY": "secret", "OTHER_API_KEY": "other",
                                         "EGO_ANNO_API_HOST": "private", "SAFE_TEST_VARIABLE": "keep"}):
            environment = rendering._subprocess_environment()
        self.assertNotIn("EGO_ANNO_API_KEY", environment)
        self.assertNotIn("OTHER_API_KEY", environment)
        self.assertNotIn("EGO_ANNO_API_HOST", environment)
        self.assertEqual(environment["SAFE_TEST_VARIABLE"], "keep")

    def test_vendor_failures_do_not_expose_tracebacks(self):
        failure = subprocess.CompletedProcess([], 1, "", "Traceback (most recent call last):\nValueError: invalid action\n")
        with mock.patch.object(rendering.subprocess, "run", return_value=failure):
            with self.assertRaises(ClientError) as caught:
                rendering._run(["renderer"], {}, "Cosmos renderer")
        self.assertEqual(str(caught.exception), "Cosmos renderer failed: ValueError: invalid action")

    def test_provenance_hashes_and_no_shipped_model_assets(self):
        package = Path(rendering.__file__).parent
        provenance = json.loads((package / "provenance.json").read_text(encoding="utf-8"))
        self.assertTrue(provenance["source_commit"].startswith("2ba840b"))
        self.assertEqual({entry["vendored_path"] for entry in provenance["files"]},
                         {"body_pane.py", "cosmos_e2e.py", "labels.py", "render_ego_json_overlay.py"})
        for entry in provenance["files"]:
            self.assertEqual(hashlib.sha256((package / entry["vendored_path"]).read_bytes()).hexdigest(),
                             entry["sha256"])
        self.assertFalse((package / "mano").exists())
        self.assertFalse((package / "fonts").exists())
        self.assertFalse([path for path in package.rglob("*") if path.suffix in (".pkl", ".npz", ".ttf", ".mp4")])


class RealOfflineRenderTests(BundleCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("real rendering requires system ffmpeg and ffprobe")
        try:
            importlib.import_module("numpy")
            cls.cv2 = importlib.import_module("cv2")
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest("real rendering requires ego-anno[render]: {}".format(exc))

    def setUp(self):
        super().setUp()
        self.make_video()

    def make_video(self, count=3, fps=15, size="320x240"):
        completed = subprocess.run([shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                                    "-f", "lavfi", "-i", "color=c=navy:s={}:r={}".format(size, fps),
                                    "-frames:v", str(count), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                    str(self.inputs / "video.mp4")],
                                   env=rendering._subprocess_environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if "Unknown encoder" in completed.stderr:
            self.skipTest("system ffmpeg lacks the required libx264 encoder")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def probe_output(self):
        completed = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
                                    "-of", "json", str(self.output)], env=rendering._subprocess_environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        stream = json.loads(completed.stdout)["streams"][0]
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        self.assertEqual(int(stream["nb_read_frames"]), 3)
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        self.assertAlmostEqual(numerator / denominator, 15)
        payload = self.output.read_bytes()
        self.assertLess(payload.index(b"moov"), payload.index(b"mdat"), "MP4 must enable faststart")
        return stream

    def test_real_render_without_calibration_or_assets_or_api_key(self):
        original_video = (self.inputs / "video.mp4").read_bytes()
        original_result = self.result_path.read_bytes()
        with mock.patch.dict(os.environ, {"EGO_ANNO_API_KEY": ""}), \
                mock.patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")):
            destination = self.render()
        self.assertEqual(destination, self.output)
        stream = self.probe_output()
        self.assertEqual([stream["width"], stream["height"]], [1920, 1080])
        self.assertEqual((self.inputs / "video.mp4").read_bytes(), original_video)
        self.assertEqual(self.result_path.read_bytes(), original_result)
        self.assertFalse(list(self.root.glob(".ego-anno-render-*")))
        # Inspect the vendored adapter: I0 has no predicted hand, no fabricated
        # shape exists, and absent calibration remains absent.
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        with mock.patch.object(renderer, "HandSurfaceAssets", side_effect=AssertionError("assets forbidden")):
            document, intrinsic, poses, joints, projected, betas, valid = renderer._load_document(self.result_path)
            self.assertIsNone(renderer._load_assets_if_needed(self.root / "nonexistent-assets", joints, betas))
        self.assertIsNone(document["逐帧轨迹"][0]["帧1"]["手0"])
        self.assertIsNone(document["Metadata"]["相机内参"])
        self.assertIsNone(document["逐帧轨迹"][1]["帧2"]["手0"]["beta参数"])

    def test_real_render_v2_bundle_uses_the_explicit_trajectory(self):
        self.save_json(self.result_path, physical_result_v2())
        destination = self.render()
        self.assertEqual(destination, self.output)
        stream = self.probe_output()
        self.assertEqual([stream["width"], stream["height"]], [1920, 1080])

    def test_real_render_two_pane_keeps_the_source_size(self):
        self.render(layout="two-pane")
        stream = self.probe_output()
        self.assertEqual([stream["width"], stream["height"]], [320, 240])

    def test_real_render_with_processed_calibration_and_layout_override(self):
        camera = {"source_size_wh": [320, 240], "source_K": [[240, 0, 159.5], [0, 240, 119.5], [0, 0, 1]]}
        self.save_json(self.inputs / "camera.json", camera)
        self.output.write_bytes(b"old output replaced only after success")
        self.render(width=640, height=360)
        stream = self.probe_output()
        self.assertEqual([stream["width"], stream["height"]], [640, 360])
        self.assertEqual(json.loads((self.inputs / "camera.json").read_text(encoding="utf-8")), camera)

    def test_real_video_count_mismatch_preserves_destination(self):
        self.make_video(count=2)
        self.rejected("frame count")

    def test_real_video_fps_mismatch_preserves_destination(self):
        self.make_video(fps=30)
        self.rejected("processed 15fps")

    def test_real_video_size_mismatch_preserves_destination_without_calibration(self):
        self.make_video(size="640x480")
        self.rejected("no implicit resizing")

    def test_real_invalid_calibration_is_not_replaced_with_invented_k(self):
        self.save_json(self.inputs / "camera.json", {"source_size_wh": [320, 240],
                                                     "source_K": [[0, 0, 160], [0, 240, 120], [0, 0, 1]]})
        self.rejected("zero-skew pinhole matrix")

    def test_invalid_physical_action_is_reported_without_traceback(self):
        self.result["action"][0][3:9] = [0] * 6
        self.save_json(self.result_path, self.result)
        self.rejected("rot6d columns")

    def test_invalid_v2_c2w_is_reported_without_traceback(self):
        result = physical_result_v2()
        result["c2w"][0][0][0] = 2.0  # frame 0 of the final trajectory must be the identity
        self.save_json(self.result_path, result)
        self.rejected("identity")

    def test_transcode_failure_preserves_destination_and_cleans_temporary_files(self):
        real_run = rendering._run

        def fail_encoding(command, environment, label):
            if label == "ffmpeg H.264 encoding":
                raise ClientError("simulated ffmpeg encoding failure")
            return real_run(command, environment, label)

        with mock.patch.object(rendering, "_run", side_effect=fail_encoding):
            self.rejected("encoding failure")


def dag_document(count=3, width=640, height=480):
    """Synthetic ego.annotation.product_json.v3 with two measured hands."""
    focal = 480.0
    cx, cy = width / 2.0, height / 2.0

    def hand(side, shift):
        joints, projected = [], []
        for index in range(21):
            x = shift + 0.02 * (index % 5 - 2)
            y = 0.02 * (index // 5 - 2)
            z = 0.5
            joints.append([x, y, z])
            projected.append([cx + focal * x / z, cy + focal * y / z])
        return {"左或右": side, "beta参数": [0.0] * 10, "手部21点坐标": joints,
                "手部21点投影坐标": projected,
                "整手姿态": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]}

    frames = []
    for index in range(count):
        frames.append({"帧{}".format(index + 1): {
            "左手是否存在": True, "右手是否存在": True,
            "左手是否可见": True, "右手是否可见": True,
            "手0": hand("left", -0.05), "手1": hand("right", 0.05),
            "相机世界位姿": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        }})
    return {"Metadata": {"schema": "ego.annotation.product_json.v3", "帧数": count,
                         "相机内参": [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
                         "完成状态": "完成"},
            "逐帧轨迹": frames, "字幕": []}


# BGR values the renderer paints measured joints with: left green, right yellow.
HAND_COLORS = ((76, 220, 92), (42, 205, 255))


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "numpy (render extra) not installed")
class CosmosHeadDocumentTests(unittest.TestCase):
    """Unit tests for the cosmos-e2e-result/v2 converter (no ffmpeg needed)."""

    def setUp(self):
        import numpy as np
        self.np = np
        from ego_anno.rendering.cosmos_e2e import CosmosE2EHeadDocument
        self.converter = CosmosE2EHeadDocument

    def test_explicit_c2w_and_right_first_layout(self):
        result = physical_result_v2(3)
        document = self.converter.convert(result)
        self.assertEqual(document["Metadata"]["source_schema"], "cosmos-e2e-result/v2")
        self.assertEqual(document["Metadata"]["帧数"], 3)
        frames = document["逐帧轨迹"]
        for index, frame in enumerate(frames):
            body = frame["帧{}".format(index + 1)]
            self.assertEqual(body["相机世界位姿"], result["c2w"][index])
        self.assertIsNone(frames[0]["帧1"]["手0"])
        self.assertFalse(frames[0]["帧1"]["左手是否存在"])
        # Renderer hand0 is left; the v2 layout serializes the right hand first.
        left = self.np.array(result["joints_camera"][0][1])
        right = self.np.array(result["joints_camera"][0][0])
        hand = frames[1]["帧2"]["手0"]
        self.assertEqual(hand["左或右"], "left")
        # World joints are the camera-frame joints transformed by c2w[frame].
        self.assertTrue(self.np.allclose(self.np.array(hand["手部21点坐标"]), left + [0.01, 0, 0]))
        self.assertTrue(self.np.allclose(self.np.array(frames[1]["帧2"]["手1"]["手部21点坐标"]),
                                         right + [0.01, 0, 0]))

    def test_action_camera_and_anchor_channels_are_never_integrated(self):
        result = physical_result_v2(4)
        for row in result["action"]:
            row[0:3] = [5.0, 6.0, 7.0]  # camera delta; v1 would integrate this
            row[147:156] = [1000.0, 0.0, 0.0, 1, 0, 0, 0, 1, 0]
        document = self.converter.convert(result)
        for index, frame in enumerate(document["逐帧轨迹"]):
            self.assertEqual(frame["帧{}".format(index + 1)]["相机世界位姿"], result["c2w"][index])

    def test_projection_uses_camera_frame_joints(self):
        camera = {"source_size_wh": [320, 240],
                  "source_K": [[240, 0, 159.5], [0, 240, 119.5], [0, 0, 1]]}
        result = physical_result_v2(2)
        document = self.converter.convert(result, camera)
        hand = document["逐帧轨迹"][1]["帧2"]["手0"]
        joint = result["joints_camera"][0][1][0]  # left wrist, camera frame of frame 2
        expected = [240 * joint[0] / joint[2] + 159.5, 240 * joint[1] / joint[2] + 119.5]
        self.assertTrue(self.np.allclose(hand["手部21点投影坐标"][0], expected))
        self.assertEqual(document["Metadata"]["相机内参"], camera["source_K"])
        without = self.converter.convert(physical_result_v2(2))
        self.assertIsNone(without["逐帧轨迹"][1]["帧2"]["手0"]["手部21点投影坐标"])
        self.assertIsNone(without["Metadata"]["相机内参"])

    def test_shape_fitter_output_fills_every_predicted_frame(self):
        """A clip-wide shape is written to beta参数 wherever a hand is predicted."""
        from ego_anno.rendering.cosmos_e2e import CosmosE2EDocument
        seen = {}

        def fit_shape(side, joints):
            seen[side] = self.np.asarray(joints)
            return self.np.full(10, 0.5 if side == "left" else -0.25)

        for converter, result in ((self.converter, physical_result_v2(4)), (CosmosE2EDocument, physical_result(4))):
            with self.subTest(schema=result["schema"]):
                seen.clear()
                document = converter.convert(result, fit_shape=fit_shape)
                self.assertEqual(set(seen), {"left", "right"})
                for side in ("left", "right"):
                    self.assertEqual(seen[side].shape, (3, 21, 3))
                    self.assertTrue(self.np.isfinite(seen[side]).all())
                frames = document["逐帧轨迹"]
                self.assertIsNone(frames[0]["帧1"]["手0"])
                for index in range(1, 4):
                    frame = frames[index]["帧{}".format(index + 1)]
                    self.assertEqual(frame["手0"]["beta参数"], [0.5] * 10)
                    self.assertEqual(frame["手1"]["beta参数"], [-0.25] * 10)
                    self.assertTrue(all(self.np.isfinite(frame[key]["beta参数"]).all() for key in ("手0", "手1")))
                json.dumps(document, allow_nan=False)
        # Without a fitter nothing is invented.
        plain = self.converter.convert(physical_result_v2(3))
        self.assertIsNone(plain["逐帧轨迹"][1]["帧2"]["手0"]["beta参数"])
        with self.assertRaisesRegex(ValueError, "10 finite values"):
            self.converter.convert(physical_result_v2(3), fit_shape=lambda side, joints: self.np.full(10, float("nan")))
        with self.assertRaisesRegex(ValueError, "10 finite values"):
            self.converter.convert(physical_result_v2(3), fit_shape=lambda side, joints: self.np.zeros(9))

    def test_malformed_v2_results_are_rejected(self):
        cases = []
        bad = physical_result_v2(); bad["schema"] = "cosmos-e2e-result/v1"
        cases.append((bad, "expected cosmos-e2e-result/v2"))
        bad = physical_result_v2(); bad["action"] = [row[:147] for row in bad["action"]]
        cases.append((bad, r"physical \[N-1,156\]"))
        bad = physical_result_v2(); bad["num_actions"] = 5
        cases.append((bad, "num_actions/action_dim"))
        bad = physical_result_v2(); bad["action_dim"] = 147
        cases.append((bad, "num_actions/action_dim"))
        bad = physical_result_v2(); bad["joints_layout"]["side_order"] = ["left", "right"]
        cases.append((bad, "side_order"))
        bad = physical_result_v2(); bad["joints_camera"][0][0][0][0] = float("nan")
        cases.append((bad, "joints_camera"))
        bad = physical_result_v2(); bad["c2w"][0][0][3] = 0.5
        cases.append((bad, "identity"))
        bad = physical_result_v2(); bad["c2w"][1][3] = [0, 0, 0, 2]
        cases.append((bad, "homogeneous"))
        bad = physical_result_v2(); bad["c2w"][1][0][0] = 2.0
        cases.append((bad, "proper"))
        bad = physical_result_v2(); bad["fps"] = 30
        cases.append((bad, "destination timestamps"))
        for result, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.converter.convert(result)


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "numpy (render extra) not installed")
class HandShapeFitTests(unittest.TestCase):
    """The clip-wide shape fit on a synthetic hand model (no licensed assets)."""

    def setUp(self):
        import numpy as np
        self.np = np
        self.renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        rng = np.random.default_rng(7)
        # 778 vertices like MANO so the tip vertex indices reconstruct uses exist;
        # each kinematic joint regresses a few nearby vertices.
        vertices = rng.uniform(-0.05, 0.05, (778, 3))
        regressor = np.zeros((16, 778))
        for joint in range(16):
            picked = rng.choice(778, 4, replace=False)
            regressor[joint, picked] = 0.25
        self.model = {
            "v_template": vertices,
            "shapedirs": rng.normal(0.0, 0.004, (778, 3, 10)),
            "J_regressor": regressor,
            "weights": np.eye(16)[rng.integers(0, 16, 778)],
            "faces": np.zeros((0, 3), dtype=np.int32),
        }
        self.assets = self.renderer.HandSurfaceAssets.__new__(self.renderer.HandSurfaceAssets)
        self.assets.models = {"left": self.model, "right": self.model}
        self.assets._shape_cache = {}

    def rest_joints(self, beta):
        base, dirs = self.renderer._rest_joints21_linear(self.model)
        return base + dirs @ beta

    def test_rest_joints_are_the_reconstruct_targets(self):
        """The fitted skeleton is the one reconstruct spans: regressor joints plus tip vertices."""
        np = self.np
        beta = np.linspace(-1.0, 1.0, 10)
        vertices, joints16 = self.assets.shaped("left", beta)
        rest = self.rest_joints(beta)
        self.assertTrue(np.allclose(rest[self.renderer.MANO16_TO_JSON21], joints16))
        for tip_vertex, tip_joint in self.renderer.DISTAL_TIPS.values():
            self.assertTrue(np.allclose(rest[tip_joint], vertices[tip_vertex]))

    def test_bone_length_fit_recovers_the_shape_from_posed_frames(self):
        np = self.np
        rng = np.random.default_rng(3)
        truth = rng.uniform(-2.0, 2.0, 10)
        rest = self.rest_joints(truth)
        frames = []
        for _ in range(12):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0, np.pi)
            k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
            frames.append(rest @ rotation.T + rng.normal(size=3))
        frames.append(np.full((21, 3), np.nan))  # an absent frame is ignored
        fitted = self.assets.fit_shape("right", np.asarray(frames))
        self.assertEqual(fitted.shape, (10,))
        self.assertTrue(np.isfinite(fitted).all())
        self.assertLess(np.abs(fitted - truth).max(), 0.05, (fitted, truth))
        fitted_lengths = self.renderer._bone_lengths21(self.rest_joints(fitted))
        # The damping prior leaves a sub-millimetre bias on the bone lengths.
        self.assertLess(np.abs(fitted_lengths - self.renderer._bone_lengths21(rest)).max(), 5e-4)

    def test_fit_is_bounded_and_falls_back_to_the_mean_shape(self):
        np = self.np
        rest = self.rest_joints(np.zeros(10))
        self.assertTrue(np.array_equal(self.assets.fit_shape("left", np.stack([rest, rest])), np.zeros(10)))
        self.assertTrue(np.array_equal(self.assets.fit_shape("left", np.full((5, 21, 3), np.nan)), np.zeros(10)))
        huge = self.renderer.fit_shape_to_bone_lengths(self.model, 50.0 * self.renderer._bone_lengths21(rest))
        self.assertTrue(np.isfinite(huge).all())
        self.assertLessEqual(np.abs(huge).max(), self.renderer.SHAPE_BOUND)
        self.assertGreater(np.abs(huge).max(), 1.0)

    def test_cosmos_document_carries_the_fitted_shape_only_with_assets(self):
        np = self.np
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(physical_result_v2(5), allow_nan=False), encoding="utf-8")
            _, _, _, joints, _, betas, _ = self.renderer._load_document(path)
            self.assertTrue(np.isnan(betas[0]).all() and np.isnan(betas[1]).all())
            self.assertIsNone(self.renderer._load_assets_if_needed(Path(temporary) / "absent", joints, betas))
            _, _, _, joints, _, betas, _ = self.renderer._load_document(path, fit_shape=self.assets.fit_shape)
            for side in range(2):
                self.assertTrue(np.isnan(betas[side][0]).all())
                self.assertTrue(np.isfinite(betas[side][1:]).all())
                self.assertTrue((betas[side][1:] == betas[side][1]).all())
            self.assertFalse(self.renderer.HandSurfaceAssets.present(Path(temporary)))


class DagResultTests(unittest.TestCase):
    """Document handling for the default (DAG) pipeline result."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.result = self.root / "annotation.json"
        self.video = self.root / "clip.mp4"
        self.output = self.root / "preview.mp4"
        self.document = dag_document()
        self.write()

    def write(self):
        self.result.write_text(json.dumps(self.document, ensure_ascii=False), encoding="utf-8")

    def rejected(self, message):
        self.video.write_bytes(b"placeholder")
        with self.assertRaisesRegex(ClientError, message):
            rendering.render_video_result(self.result, self.video, self.output)

    def test_dag_schema_lives_in_metadata(self):
        self.assertEqual(rendering._result_schema(self.document),
                         "ego.annotation.product_json.v3")
        self.assertEqual(rendering._result_schema({"schema": "cosmos-e2e-result/v1"}),
                         "cosmos-e2e-result/v1")
        self.assertIsNone(rendering._result_schema({"Metadata": {}}))
        self.assertEqual(rendering.SUPPORTED_SCHEMAS,
                         ("cosmos-e2e-result/v1", "cosmos-e2e-result/v2", "ego.annotation.product_json.v3"))

    def test_unsupported_schema_names_every_documented_result(self):
        self.document = {"schema": "ego.annotation.product_json.v4"}
        self.write()
        self.rejected("accepts cosmos-e2e-result/v1 or cosmos-e2e-result/v2 or ego.annotation.product_json.v3")

    def test_metadata_must_carry_the_dag_schema(self):
        # A top-level schema selects the reader; Metadata.schema is the DAG's
        # own contract marker and must still name v3.
        self.document["schema"] = "ego.annotation.product_json.v3"
        self.document["Metadata"]["schema"] = "ego.annotation.product_json.v2"
        self.write()
        self.rejected("Metadata.schema must be ego.annotation.product_json.v3")

    def test_frame_count_must_match_the_trajectory(self):
        self.document["Metadata"]["帧数"] = 4
        self.write()
        self.rejected("逐帧轨迹 does not hold Metadata.帧数 entries")

    def test_frame_keys_must_be_one_based(self):
        self.document["逐帧轨迹"][1] = {"frame2": self.document["逐帧轨迹"][1]["帧2"]}
        self.write()
        self.rejected("entry 2 is not keyed 帧2")

    def test_missing_video_is_reported(self):
        with self.assertRaisesRegex(ClientError, "source video does not exist"):
            rendering.render_video_result(self.result, self.video, self.output)

    def test_camera_applies_only_to_cosmos(self):
        self.video.write_bytes(b"placeholder")
        camera = self.root / "camera.json"
        camera.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ClientError, "--camera applies to Cosmos e2e results"):
            rendering.render_video_result(self.result, self.video, self.output, camera=str(camera))

    def test_mano_directory_is_validated_before_rendering(self):
        self.video.write_bytes(b"placeholder")
        empty = self.root / "assets"
        empty.mkdir()
        with self.assertRaisesRegex(ClientError, "does not exist"):
            rendering.render_video_result(self.result, self.video, self.output,
                                          mano_dir=str(self.root / "absent"))
        with self.assertRaisesRegex(ClientError, "missing MANO_LEFT.pkl"):
            rendering.render_video_result(self.result, self.video, self.output,
                                          mano_dir=str(empty))

    def test_beta_is_elided_without_touching_the_result_on_disk(self):
        before = self.result.read_bytes()
        prepared = rendering._without_mano_shapes(self.document)
        self.assertEqual(self.result.read_bytes(), before)
        for frame in prepared["逐帧轨迹"]:
            body = next(iter(frame.values()))
            self.assertIsNone(body["手0"]["beta参数"])
            self.assertIsNone(body["手1"]["beta参数"])
            self.assertEqual(body["手0"]["左或右"], "left")
            self.assertEqual(len(body["手0"]["手部21点坐标"]), 21)
            self.assertEqual(len(body["手0"]["手部21点投影坐标"]), 21)
            self.assertEqual(body["相机世界位姿"][0], [1.0, 0.0, 0.0, 0.0])
        # The original document object is not mutated either.
        self.assertEqual(self.document["逐帧轨迹"][0]["帧1"]["手0"]["beta参数"], [0.0] * 10)
        self.assertEqual(prepared["Metadata"], self.document["Metadata"])

    def test_beta_elision_is_idempotent_and_tolerates_absent_hands(self):
        prepared = rendering._without_mano_shapes(self.document)
        again = rendering._without_mano_shapes(prepared)
        self.assertEqual(again, prepared)
        absent = dag_document(count=1)
        absent["逐帧轨迹"][0]["帧1"]["手0"] = None
        prepared = rendering._without_mano_shapes(absent)
        self.assertIsNone(prepared["逐帧轨迹"][0]["帧1"]["手0"])

    def test_output_must_not_overwrite_the_result_or_the_video(self):
        for protected in (self.result, self.video):
            protected.write_bytes(b"keep")
            with self.assertRaisesRegex(ClientError, "must not overwrite"):
                rendering.render_video_result(self.result, self.video, protected)
            self.assertEqual(protected.read_bytes(), b"keep")

    def test_side_rear_follow_view_frames_camera_and_hands_across_motion(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        hand = np.asarray(self.document["逐帧轨迹"][0]["帧1"]["手0"]["手部21点坐标"], dtype=float)
        for translation in (0.0, 0.8, -0.6):
            pose = np.eye(4, dtype=float)
            pose[:3, 3] = [translation, 0.0, 0.0]
            cam_verts, _ = renderer._build_cinema_camera(s=0.085)
            view, report = renderer._single_pose_view(pose, [hand + pose[:3, 3], hand + pose[:3, 3]],
                                                      cam_verts, 640, 400)
            marker, _ = renderer._camera_marker_points(pose)
            points = np.vstack((marker, hand + pose[:3, 3], hand + pose[:3, 3]))
            pixels, _, valid, _ = renderer._project_view(points, view, (640, 400))
            self.assertTrue(valid.all(), "camera and both hands must be in front of the view")
            self.assertTrue(np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < 640)))
            self.assertTrue(np.all((pixels[:, 1] >= 0) & (pixels[:, 1] < 400)))
            # Side-above-behind: measured against the estimated world-up and
            # the camera's horizontal heading, not against the camera axes.
            up = report["up"]
            heading = pose[:3, 2] - up * float(np.dot(pose[:3, 2], up))
            heading /= np.linalg.norm(heading)
            off = view["eye"] - report["target"][0]
            rear = float(np.dot(off, heading))
            vertical = float(np.dot(off, up))
            lateral = float(np.linalg.norm(off - heading * rear - up * vertical))
            self.assertLess(rear, 0.0, "observer must sit behind the heading")
            self.assertGreater(vertical, 0.0, "observer must sit above the anchor")
            self.assertGreater(lateral, 0.0)
            self.assertAlmostEqual(np.degrees(np.arctan2(lateral, abs(rear))), renderer._EYE_AZIMUTH, places=6)
            elevation = np.degrees(np.arctan2(vertical, np.hypot(rear, lateral)))
            self.assertAlmostEqual(elevation, renderer._EYE_ELEVATION, places=6)
            self.assertGreater(elevation, 30.0)
            self.assertLess(elevation, 35.0)

    def test_view_basis_is_a_rigid_look_at_frame(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        hand = np.asarray(self.document["逐帧轨迹"][0]["帧1"]["手0"]["手部21点坐标"], dtype=float)
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [0.12, -0.05, 0.02]
        cam_verts, _ = renderer._build_cinema_camera(s=0.085)
        view, report = renderer._single_pose_view(pose, [hand + pose[:3, 3]], cam_verts, 640, 400)
        basis = np.stack([view["right"], view["up"], view["forward"]])
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-9, err_msg="basis must be orthonormal")
        # The renderer projects like a pinhole camera (pixel-y is flipped), so the
        # rigid frame is (right, up, forward) with right = forward x up. This is
        # the invariant that keeps camera +x on the screen-right; det is -1 by
        # that convention, which is why the orthonormality check above (not det)
        # is what proves the view is a rotation, not a shear.
        np.testing.assert_allclose(np.cross(view["forward"], view["up"]), view["right"], atol=1e-9)
        # The look direction points from the eye at the smoothed anchor.
        direction = renderer._normalize(report["target"][0] - view["eye"], np.zeros(3))
        self.assertGreater(float(np.dot(direction, view["forward"])), 0.999)
        # Zero roll: the view up lies in the vertical plane spanned by world-up
        # and the view direction, i.e. its component along `right` vanishes.
        up = report["up"]
        self.assertAlmostEqual(float(np.dot(up, view["right"])), 0.0, places=9)
        self.assertGreater(float(np.dot(view["up"], up)), 0.8)
        # A single frame cannot pin world-up from yaw variation, so the fallback
        # tilts the camera's -down by the assumed head pitch.
        self.assertEqual(report["gravity"]["branch"], "fallback")
        pitch = np.degrees(np.arcsin(np.clip(-float(np.dot(pose[:3, 2], up)), -1.0, 1.0)))
        self.assertAlmostEqual(pitch, renderer._FALLBACK_HEAD_PITCH_DEG, places=6)
        # The observer looks downward (forward has a negative world-up component).
        self.assertLess(float(np.dot(view["forward"], up)), 0.0)
        # "The horizon is level": a vertical line through the look-at target
        # (on the optical axis, where perspective convergence vanishes) projects
        # exactly to screen-vertical.
        target = report["target"][0]
        probe = np.vstack((target - 0.1 * up, target + 0.1 * up))
        pixels, _, valid, _ = renderer._project_view(probe, view, (640, 400))
        self.assertTrue(valid.all())
        delta = pixels[1] - pixels[0]
        roll = np.degrees(np.arctan2(delta[0], abs(delta[1])))
        self.assertLess(abs(roll), 1e-6, "world-up must stay screen-vertical")

    def test_world_pane_keeps_the_video_pane_left_right_order(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [0.03, 0.01, 0.0]
        # Asymmetric scene: the anatomical LEFT hand is displaced toward camera
        # -x (camera-left on the video frame) and forward of the right hand, just
        # like a real two-hand insert task. Left must land left on screen.
        left = np.array([[0.05, 0.20, 0.62]] * 21, dtype=float)
        right = np.array([[0.16, 0.13, 0.60]] * 21, dtype=float)
        intrinsic = np.asarray(self.document["Metadata"]["相机内参"], dtype=float)
        cam_verts, _ = renderer._build_cinema_camera(s=0.085)
        view, _ = renderer._single_pose_view(pose, [left + pose[:3, 3], right + pose[:3, 3]], cam_verts, 640, 400)
        pixels, _, valid, _ = renderer._project_view(
            np.vstack((left + pose[:3, 3], right + pose[:3, 3])), view, (640, 400))
        self.assertTrue(valid.all())
        left_x = float(pixels[:21, 0].mean())
        right_x = float(pixels[21:, 0].mean())
        self.assertLess(left_x, right_x, "world pane must not mirror the hands")
        # Sanity: the same 3D scene through the real intrinsics really does put
        # the left hand on the camera-left, i.e. the fixture is anatomically
        # consistent and the pane is what has to agree with it.
        video, _, video_valid = renderer._project_camera(
            renderer._camera_coordinates(np.vstack((left + pose[:3, 3], right + pose[:3, 3])), pose), intrinsic)
        self.assertTrue(video_valid.all())
        self.assertLess(float(video[:21, 0].mean()), float(video[21:, 0].mean()))
        # And the world pane agrees with the video pane about which side is which.
        self.assertEqual(bool(left_x < right_x), bool(video[:21, 0].mean() < video[21:, 0].mean()))

    def test_world_projection_matches_the_video_pane_through_the_intrinsics(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        intrinsic = np.asarray(self.document["Metadata"]["相机内参"], dtype=float)
        pose = np.eye(4, dtype=float)
        for frame in self.document["逐帧轨迹"]:
            body = frame[list(frame)[0]]
            world_pose = np.asarray(body["相机世界位姿"], dtype=float)
            for key in ("手0", "手1"):
                joints = np.asarray(body[key]["手部21点坐标"], dtype=float)
                stored = np.asarray(body[key]["手部21点投影坐标"], dtype=float)
                pixels, _, valid = renderer._project_camera(
                    renderer._camera_coordinates(joints, world_pose), intrinsic)
                self.assertTrue(valid.all())
                # 3D joints re-projected through the stored intrinsics must agree
                # with the recorded 2D overlay pixels to sub-pixel precision.
                np.testing.assert_allclose(pixels, stored, atol=1e-6)
        # The view must not distort that agreement: the camera-relative lateral
        # ordering of every joint pair survives the world projection.
        cam_verts, _ = renderer._build_cinema_camera(s=0.085)
        view, _ = renderer._single_pose_view(
            pose, [np.asarray(self.document["逐帧轨迹"][0]["帧1"]["手0"]["手部21点坐标"], dtype=float)],
            cam_verts, 640, 400)
        self.assertAlmostEqual(float(np.linalg.norm(view["forward"])), 1.0, places=9)

    def test_spatial_scene_contains_small_camera_rgb_axes_and_full_trail(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        vertices, faces = renderer._build_cinema_camera()
        self.assertGreater(len(vertices), 80)
        self.assertGreater(len(faces), 100)
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [0.3, 0.0, 0.0]
        world = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
        np.testing.assert_allclose(world - vertices, np.tile([0.3, 0.0, 0.0], (len(vertices), 1)))
        self.assertFalse(hasattr(renderer, "_camera_frustum_points"))
        self.assertNotIn("_draw_frustum", renderer.__dict__)
        centers = np.asarray([[-0.8, 0, 0], [0.3, 0, 0], [1.2, 0, 0]], dtype=float)
        hand = np.asarray(self.document["逐帧轨迹"][0]["帧1"]["手0"]["手部21点坐标"], dtype=float) + pose[:3, 3]
        rendered = renderer._render_world((640, 400), pose, centers, 2, [hand, hand], [None, None], None,
                                          [np.full(10, np.nan), np.full(10, np.nan)],
                                          np.asarray(self.document["Metadata"]["相机内参"], dtype=float), (640, 480))
        self.assertEqual(rendered.shape, (400, 640, 3))
        # Axis arrows + blue full-trail pixels are painted. The view is now a
        # three-quarter rear view, so all three camera axes are visible rather
        # than one foreshortening to a dot as in the old strict profile.
        self.assertGreater(int(((rendered[:, :, 1] > 120) & (rendered[:, :, 1] > rendered[:, :, 2] * 1.5)).sum()), 20)
        self.assertGreater(int(((rendered[:, :, 0] > 80) & (rendered[:, :, 0] > rendered[:, :, 2] * 1.3) & (rendered[:, :, 0] > rendered[:, :, 1] * 1.3)).sum()), 20)
        self.assertGreater(int(((rendered[:, :, 2] > 150) & (rendered[:, :, 2] > rendered[:, :, 0] * 1.8) & (rendered[:, :, 2] > rendered[:, :, 1] * 1.8)).sum()), 20)
        # The camera body is intentionally near-black, not the old red prop.
        self.assertGreater(int(((rendered[:, :, 0] < 45) & (rendered[:, :, 1] < 45) & (rendered[:, :, 2] < 45)).sum()), 20)

    def test_trail_accumulates_to_the_current_frame_and_survives_the_camera_body(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        intrinsic = np.asarray(self.document["Metadata"]["相机内参"], dtype=float)
        hand = np.asarray(self.document["逐帧轨迹"][0]["帧1"]["手0"]["手部21点坐标"], dtype=float)
        # A realistic head-wander path: ~3 cm of total travel, i.e. far smaller
        # than the 19 cm camera prop, which is exactly the regime where a trail
        # drawn *before* the opaque body is completely overpainted.
        steps = np.asarray([
            [0.000, 0.000, 0.000], [0.006, -0.002, 0.004], [0.011, -0.004, 0.009],
            [0.009, -0.007, 0.015], [0.004, -0.009, 0.020], [0.001, -0.010, 0.026],
        ], dtype=float)
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = steps[-1]
        trail_bgr = np.asarray(renderer.TRAIL_COLOR, dtype=int)

        def trail_pixels(frame_index: int) -> int:
            world = renderer._render_world((640, 400), pose, steps, frame_index, [hand, hand], [None, None], None,
                                           [np.full(10, np.nan), np.full(10, np.nan)], intrinsic, (640, 480))
            raster = world.astype(int)
            # Tight threshold: the blue axis arrow's antialiased rim lands near
            # BGR [194, 125, 65], which is inside a loose 70 window. At 40 the
            # match is attributable to the trail itself.
            return int((np.abs(raster - trail_bgr).max(axis=2) < 40).sum())

        # The path is painted over the opaque camera body at every frame once it
        # has length -- this is the regression that hid it entirely before.
        for frame_index in range(1, len(steps)):
            self.assertGreater(trail_pixels(frame_index), 0,
                               "trail must remain visible on top of the camera body at frame {}".format(frame_index))
        # Accumulation, not a static strip: a later frame carries more of the path.
        self.assertGreater(trail_pixels(5), trail_pixels(1))
        # And nothing is drawn for a single-point path, nor beyond the clip.
        self.assertEqual(trail_pixels(0), 0)
        np.testing.assert_array_equal(renderer._trajectory_upto(steps, 2), steps[:3])
        self.assertEqual(len(renderer._trajectory_upto(steps, 99)), len(steps))
        gaps = np.asarray([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [0.002, 0.0, 0.004]], dtype=float)
        np.testing.assert_array_equal(renderer._trajectory_upto(gaps, 2), np.asarray([[0.0, 0.0, 0.0], [0.002, 0.0, 0.004]]))

    @staticmethod
    def synthetic_clip(count, yaw, pitch, roll, centres, true_up):
        """Camera poses (x right, y down, z forward) and two hands from Euler tracks.

        `true_up` is the gravity direction the clip is built around; the
        renderer never sees it unless a test passes it in explicitly.
        """
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        true_up = np.asarray(true_up, dtype=float) / np.linalg.norm(true_up)
        seed = np.array([0.0, 0.0, 1.0]) if abs(true_up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        forward0 = seed - true_up * float(np.dot(seed, true_up))
        forward0 /= np.linalg.norm(forward0)
        poses = np.zeros((count, 4, 4), dtype=float)
        hands = [np.zeros((count, 21, 3), dtype=float) for _ in range(2)]
        offsets = (np.array([-0.10, 0.15, 0.45]), np.array([0.10, 0.15, 0.45]))
        for index in range(count):
            forward = renderer._rotate_about(forward0, true_up, float(yaw[index]))
            right = np.cross(forward, true_up)
            right /= np.linalg.norm(right)
            # pitch: nose down about the right axis; roll: about the forward axis.
            forward = renderer._rotate_about(forward, right, -float(pitch[index]))
            down = np.cross(forward, right)
            right = renderer._rotate_about(right, forward, float(roll[index]))
            down = renderer._rotate_about(down, forward, float(roll[index]))
            rotation = np.column_stack((right, down, forward))
            poses[index, :3, :3] = rotation
            poses[index, :3, 3] = centres[index]
            poses[index, 3, 3] = 1.0
            for side in range(2):
                base = rotation @ offsets[side] + centres[index]
                hands[side][index] = base + np.arange(21)[:, None] * np.array([0.002, 0.001, 0.0])
        return poses, hands

    def track_metrics(self, poses, hands, world_up=None, fps=30.0):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        cam_verts, _ = renderer._build_cinema_camera(s=0.085)
        views, report = renderer._follow_view_track(poses, np.ones(len(poses), dtype=bool), hands, cam_verts,
                                                    640, 400, fps=fps, world_up=world_up)
        up = report["up"]
        roll = []
        up_angle = []
        margins = []
        for index, view in enumerate(views):
            self.assertIsNotNone(view)
            # Roll is the component of world-up along screen-right: zero means
            # the view up and world-up share a vertical plane.
            roll.append(np.degrees(np.arcsin(np.clip(float(np.dot(up, view["right"])), -1.0, 1.0))))
            # View up equals world-up projected perpendicular to the view axis.
            projected = up - view["forward"] * float(np.dot(up, view["forward"]))
            projected /= np.linalg.norm(projected)
            up_angle.append(np.degrees(np.arccos(np.clip(float(np.dot(projected, view["up"])), -1.0, 1.0))))
            body = (poses[index, :3, :3] @ cam_verts.T).T + poses[index, :3, 3]
            pixels, _, valid, _ = renderer._project_view(np.vstack((body, hands[0][index], hands[1][index])),
                                                         view, (640, 400))
            self.assertTrue(valid.all())
            margins.append(min(pixels[:, 0].min(), pixels[:, 1].min(),
                               639 - pixels[:, 0].max(), 399 - pixels[:, 1].max()))
        eye = report["eye"]
        steps = np.linalg.norm(np.diff(eye, axis=0), axis=1)
        speed = steps * fps
        accel = np.abs(np.diff(speed)) * fps if len(speed) > 1 else np.zeros(1)
        return {"views": views, "report": report, "up": up, "roll": np.asarray(roll),
                "up_angle": np.asarray(up_angle), "margin": float(min(margins)),
                "eye_path": float(steps.sum()), "max_speed": float(speed.max()), "max_accel": float(accel.max())}

    def test_observer_up_is_world_up_under_camera_roll(self):
        import numpy as np
        count = 120
        t = np.linspace(0.0, 1.0, count)
        true_up = np.array([0.0, -1.0, 0.0])
        poses, hands = self.synthetic_clip(
            count, yaw=np.radians(15.0) * np.sin(2.0 * np.pi * t), pitch=np.full(count, np.radians(20.0)),
            roll=np.radians(30.0) * np.sin(3.0 * np.pi * t),
            centres=np.outer(t, [0.5, 0.0, 0.3]), true_up=true_up)
        for world_up in (None, true_up):
            with self.subTest(world_up="estimated" if world_up is None else "given"):
                metrics = self.track_metrics(poses, hands, world_up=world_up)
                # The observer's up is the estimated world-up on every frame.
                self.assertLess(float(metrics["up_angle"].max()), 0.5)
                self.assertLess(float(np.abs(metrics["roll"]).max()), 1e-6)
                self.assertGreater(metrics["margin"], 0.0)
                estimate_error = np.degrees(np.arccos(np.clip(float(np.dot(metrics["up"], true_up)), -1.0, 1.0)))
                if world_up is None:
                    # Heavy roll breaks the "right axis is horizontal" premise of
                    # the eigen estimate; the estimator must notice and fall back
                    # to the pitch-tilted camera down axis, which stays near truth.
                    self.assertEqual(metrics["report"]["gravity"]["branch"], "fallback")
                    self.assertLess(estimate_error, 10.0)
                else:
                    self.assertLess(estimate_error, 1e-9)
        # The camera really did roll: its right axis left the horizontal by ~30 deg.
        tilt = np.degrees(np.arcsin(np.abs(poses[:, :3, 0] @ true_up)))
        self.assertGreater(float(tilt.max()), 28.0)

    def test_jitter_produces_no_observer_motion(self):
        import numpy as np
        count = 90
        rng = np.random.default_rng(7)
        poses, hands = self.synthetic_clip(
            count, yaw=np.radians(rng.uniform(-2.0, 2.0, count)), pitch=np.radians(20.0 + rng.uniform(-2.0, 2.0, count)),
            roll=np.radians(rng.uniform(-2.0, 2.0, count)), centres=rng.uniform(-0.01, 0.01, (count, 3)),
            true_up=[0.0, -1.0, 0.0])
        metrics = self.track_metrics(poses, hands)
        self.assertLessEqual(metrics["eye_path"], 0.005, "jitter must not move the observer")
        self.assertLess(float(np.abs(metrics["roll"]).max()), 1e-6)
        self.assertGreater(metrics["margin"], 0.0)

    def test_walk_and_turn_is_followed(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        fps = 30.0
        count = int(10 * fps)
        t = np.linspace(0.0, 10.0, count)
        true_up = np.array([0.0, -1.0, 0.0])
        walk = np.clip(t / 3.0, 0.0, 1.0)                  # 0-3 s: walk 2 m forward
        turn = np.clip((t - 3.0) / 3.0, 0.0, 1.0)          # 3-6 s: turn 90 deg, then hold
        yaw = np.radians(90.0) * turn
        poses, hands = self.synthetic_clip(count, yaw=yaw, pitch=np.full(count, np.radians(20.0)),
                                           roll=np.zeros(count), centres=np.outer(walk, [0.0, 0.0, 2.0]),
                                           true_up=true_up)
        metrics = self.track_metrics(poses, hands, fps=fps)
        self.assertEqual(metrics["report"]["gravity"]["branch"], "eigen")
        self.assertLess(np.degrees(np.arccos(float(np.dot(metrics["up"], true_up)))), 1.0)
        # Followed: the observer travelled with the walk, within the speed and
        # acceleration bounds and without any roll.
        self.assertGreater(metrics["eye_path"], 1.0)
        self.assertLessEqual(metrics["max_speed"], renderer._OBSERVER_MAX_SPEED + 1e-6)
        self.assertLess(metrics["max_accel"], 3.0)
        self.assertLess(float(np.abs(metrics["roll"]).max()), 1e-6)
        self.assertGreater(metrics["margin"], 0.0)
        # Ends behind the new heading: the eye's horizontal offset from the
        # target opposes the final camera forward, and the smoothed heading has
        # swung through more than half of the 90 deg turn.
        final = metrics["views"][-1]
        target = metrics["report"]["target"][-1]
        offset = final["eye"] - target
        horizontal = offset - true_up * float(np.dot(offset, true_up))
        final_forward = poses[-1, :3, 2] - true_up * float(np.dot(poses[-1, :3, 2], true_up))
        self.assertLess(float(np.dot(horizontal, final_forward)), 0.0)
        heading = metrics["report"]["heading"]
        self.assertGreater(float(np.degrees(heading[-1] - heading[0])), 45.0)
        # Smooth: no per-frame reversals in the heading track.
        self.assertTrue(np.all(np.diff(heading) >= -1e-9))

    def test_floor_is_level_and_below_the_scene(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        count = 60
        t = np.linspace(0.0, 1.0, count)
        true_up = np.array([0.0, -1.0, 0.0])
        poses, hands = self.synthetic_clip(count, yaw=np.radians(20.0) * np.sin(2.0 * np.pi * t),
                                           pitch=np.full(count, np.radians(25.0)), roll=np.zeros(count),
                                           centres=np.outer(t, [0.2, 0.0, 0.2]), true_up=true_up)
        metrics = self.track_metrics(poses, hands)
        floor = metrics["report"]["floor"]
        up = metrics["up"]
        np.testing.assert_allclose(floor["up"], up)
        self.assertAlmostEqual(float(np.dot(floor["h1"], up)), 0.0, places=9)
        self.assertAlmostEqual(float(np.dot(floor["h2"], up)), 0.0, places=9)
        self.assertAlmostEqual(float(np.dot(floor["h1"], floor["h2"])), 0.0, places=9)
        lowest = min(float((poses[:, :3, 3] @ up).min()), float((hands[0].reshape(-1, 3) @ up).min()),
                     float((hands[1].reshape(-1, 3) @ up).min()))
        self.assertLess(float(np.dot(floor["origin"], up)), lowest)
        self.assertAlmostEqual(float(np.dot(floor["origin"], up)), lowest - renderer._FLOOR_DROP_M, places=2)
        # Rendering with the track draws the grid without error and keeps the
        # established dark blue-green style.
        view = metrics["views"][count // 2]
        world = renderer._render_world((640, 400), poses[count // 2], poses[:, :3, 3], count // 2,
                                       [hands[0][count // 2], hands[1][count // 2]], [None, None], None,
                                       [np.full(10, np.nan), np.full(10, np.nan)], None, (640, 480),
                                       view=view, floor=floor)
        self.assertEqual(world.shape, (400, 640, 3))
        self.assertGreater(int((np.abs(world.astype(int) - np.asarray(renderer.GRID_COLOR)).max(axis=2) < 25).sum()), 200)

    def test_follow_track_keeps_the_fixture_scene_in_frame(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        frames = self.document["逐帧轨迹"]
        poses = np.asarray([frame[list(frame)[0]]["相机世界位姿"] for frame in frames], dtype=float)
        sides = [[], []]
        for frame in frames:
            body = frame[list(frame)[0]]
            for index, key in enumerate(("手0", "手1")):
                sides[index].append(np.asarray(body[key]["手部21点坐标"], dtype=float))
        joints = [np.asarray(side, dtype=float) for side in sides]
        pose_valid = np.isfinite(poses).all(axis=(1, 2))
        cam_verts, _ = renderer._build_cinema_camera(s=0.085)
        width, height = 640, 400
        views, report = renderer._follow_view_track(poses, pose_valid, joints, cam_verts, width, height, fps=15.0)
        self.assertEqual(len(views), len(frames))
        self.assertIn(report["gravity"]["branch"], ("eigen", "fallback"))
        # The same input yields the same track: nothing is random or stateful.
        again, _ = renderer._follow_view_track(poses, pose_valid, joints, cam_verts, width, height, fps=15.0)
        for first, second in zip(views, again):
            np.testing.assert_array_equal(first["eye"], second["eye"])
            self.assertEqual(first["focal"], second["focal"])
        worst = None
        for frame_index in np.flatnonzero(pose_valid):
            pose = poses[frame_index]
            body = (pose[:3, :3] @ cam_verts.T).T + pose[:3, 3]
            chunk = [body]
            for side in range(2):
                row = joints[side][frame_index]
                if np.isfinite(row).all():
                    chunk.append(row)
            pixels, _, valid, _ = renderer._project_view(np.vstack(chunk), views[frame_index], (width, height))
            self.assertTrue(valid.all(), "scene must sit in front of the eye")
            margin = min(pixels[:, 0].min(), pixels[:, 1].min(),
                         (width - 1) - pixels[:, 0].max(), (height - 1) - pixels[:, 1].max())
            worst = margin if worst is None else min(worst, margin)
        self.assertGreater(worst, 0.04 * min(width, height), "worst-case margin {:.1f}px".format(worst))

    def test_render_labels_are_english(self):
        source = Path(importlib.import_module("ego_anno.rendering.render_ego_json_overlay").__file__).read_text(encoding="utf-8")
        self.assertIn("Hand joints overlay", source)
        self.assertIn("Camera and hands: side-rear follow view", source)
        self.assertNotIn("fixed side-rear view: camera and hands", source)
        self.assertNotIn("left hand: ", source)
        self.assertNotIn("right hand: ", source)
        self.assertNotIn("手部姿态overlay", source)
        self.assertNotIn("相机-手 世界坐标位姿", source)

    def test_command_line_requires_exactly_one_input_form(self):
        from ego_anno import cli
        base = ["render", "--result", "r.json", "-o", "out.mp4"]
        with self.assertRaises(SystemExit):
            cli.main(base)
        with self.assertRaises(SystemExit):
            cli.main(base + ["--inputs", "bundle", "--video", "clip.mp4"])
        with self.assertRaises(SystemExit):
            cli.main(base + ["--inputs", "bundle", "--fps", "15"])
        with self.assertRaises(SystemExit):
            cli.main(base + ["--inputs", "bundle", "--world-up", "0,-1,0"])

    def test_mano_dir_is_accepted_with_a_saved_bundle(self):
        from ego_anno import cli
        with mock.patch.object(rendering, "render_result", return_value=Path("out.mp4")) as render_result:
            self.assertEqual(cli.main(["render", "--result", "r.json", "--inputs", "bundle",
                                       "--mano-dir", "assets", "-o", "out.mp4"]), 0)
        render_result.assert_called_once_with("r.json", "bundle", "out.mp4", None, None, mano_dir="assets",
                                              layout="three-pane", labels=None)
        with mock.patch.object(rendering, "render_result", return_value=Path("out.mp4")) as render_result:
            self.assertEqual(cli.main(["render", "--result", "r.json", "--inputs", "bundle", "-o", "out.mp4"]), 0)
        render_result.assert_called_once_with("r.json", "bundle", "out.mp4", None, None, mano_dir=None,
                                              layout="three-pane", labels=None)
        self.video.write_bytes(b"placeholder")
        with self.assertRaisesRegex(ClientError, "does not exist"):
            rendering.render_result(self.result, self.root, self.output, mano_dir=str(self.root / "absent"))
        self.assertFalse(self.output.exists())

    def test_world_up_is_validated_and_forwarded_to_the_renderer(self):
        self.assertEqual(rendering._world_up_argument(None), [])
        self.assertEqual(rendering._world_up_argument("0,-1,0"), ["--world-up", "0.0,-1.0,0.0"])
        self.assertEqual(rendering._world_up_argument((0.1, -0.9, -0.4)), ["--world-up", "0.1,-0.9,-0.4"])
        for bad in ("0,-1", "a,b,c", "0,0,0", "nan,1,0", (1.0, 2.0), "1,2,3,4"):
            with self.subTest(bad=bad), self.assertRaisesRegex(ClientError, "--world-up"):
                rendering._world_up_argument(bad)
        self.video.write_bytes(b"placeholder")
        with self.assertRaisesRegex(ClientError, "--world-up"):
            rendering.render_video_result(self.result, self.video, self.output, world_up="0,0")
        self.assertFalse(self.output.exists())


class DagRenderCase(unittest.TestCase):
    """Shared real-media harness: a decodable clip plus the DAG document."""

    @classmethod
    def setUpClass(cls):
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise unittest.SkipTest("real rendering requires system ffmpeg and ffprobe")
        try:
            importlib.import_module("numpy")
            cls.cv2 = importlib.import_module("cv2")
        except (ImportError, OSError) as exc:
            raise unittest.SkipTest("real rendering requires ego-anno[render]: {}".format(exc))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.video = self.root / "clip.mp4"
        self.result = self.root / "annotation.json"
        self.output = self.root / "preview.mp4"
        self.make_video()
        self.save_json(self.result, dag_document())

    def save_json(self, path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def make_video(self, count=3, fps=15, size="640x480"):
        completed = subprocess.run(
            [shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=navy:s={}:r={}".format(size, fps),
             "-frames:v", str(count), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(self.video)],
            env=rendering._subprocess_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if "Unknown encoder" in completed.stderr:
            self.skipTest("system ffmpeg lacks the required libx264 encoder")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def probe(self, path):
        completed = subprocess.run(
            [shutil.which("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
             "-of", "json", str(path)], env=rendering._subprocess_environment(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)["streams"][0]

    def panes(self, path, frame_index=0, layout="three-pane"):
        """(overlay, follow view[, skeleton]) pane crops of one decoded frame."""
        capture = self.cv2.VideoCapture(str(path))
        try:
            capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        self.assertTrue(ok, "cannot decode rendered frame {}".format(frame_index))
        height, width = frame.shape[:2]
        if layout == "three-pane":
            renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
            return tuple(frame[y:y + h, x:x + w]
                         for x, y, w, h in renderer._three_pane_layout(width, height)["panes"])
        band = min(220, max(165, int(round(height * 0.145))))
        pane_height, pane_width = height - band, width // 2
        return frame[:pane_height, :pane_width], frame[:pane_height, pane_width:pane_width * 2]

    def body_joint_pixels(self, pane):
        """Pixels of the skeleton pane's orange body joints."""
        numpy = __import__("numpy")
        return int((abs(pane.astype(int) - numpy.array((50, 110, 240))).sum(axis=2) < 90).sum())

    def joint_pixels(self, pane):
        total = 0
        for color in HAND_COLORS:
            delta = abs(pane.astype(int) - __import__("numpy").array(color, int)).sum(axis=2)
            total += int((delta < 60).sum())
        return total


class RealDagRenderTests(DagRenderCase):
    def test_render_without_assets_draws_measured_joints_in_both_panes(self):
        original_result = self.result.read_bytes()
        destination = rendering.render_video_result(self.result, self.video, self.output)
        self.assertEqual(destination, self.output)
        stream = self.probe(self.output)
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        self.assertEqual(int(stream["nb_read_frames"]), 3)
        self.assertEqual([stream["width"], stream["height"]], [1920, 1080])
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        self.assertAlmostEqual(numerator / denominator, 15)
        payload = self.output.read_bytes()
        self.assertLess(payload.index(b"moov"), payload.index(b"mdat"), "MP4 must enable faststart")
        self.assertEqual(self.result.read_bytes(), original_result)
        self.assertFalse(list(self.root.glob(".ego-anno-render-*")))
        for frame_index in range(3):
            source_pane, world_pane, body_pane = self.panes(self.output, frame_index)
            self.assertGreater(self.joint_pixels(source_pane), 0, "no joints over the source frame")
            self.assertGreater(self.joint_pixels(world_pane), 0, "no joints in the 3D pane")
            self.assertGreater(self.body_joint_pixels(body_pane), 0, "no body joints in the skeleton pane")

    def test_two_pane_layout_keeps_the_source_size_and_both_panes(self):
        rendering.render_video_result(self.result, self.video, self.output, layout="two-pane")
        stream = self.probe(self.output)
        self.assertEqual([stream["width"], stream["height"]], [640, 480])
        source_pane, world_pane = self.panes(self.output, 1, layout="two-pane")
        self.assertGreater(self.joint_pixels(source_pane), 0)
        self.assertGreater(self.joint_pixels(world_pane), 0)

    def test_unknown_layout_is_rejected(self):
        with self.assertRaisesRegex(ClientError, "--layout must be one of"):
            rendering.render_video_result(self.result, self.video, self.output, layout="four-pane")
        self.assertFalse(self.output.exists())

    def test_layout_override_is_honoured(self):
        rendering.render_video_result(self.result, self.video, self.output, width=1280, height=960)
        stream = self.probe(self.output)
        self.assertEqual([stream["width"], stream["height"]], [1280, 960])

    def test_explicit_world_up_renders(self):
        rendering.render_video_result(self.result, self.video, self.output, world_up="0,-1,0")
        self.probe(self.output)

    def test_frame_count_mismatch_preserves_destination(self):
        self.make_video(count=2)
        self.output.write_bytes(b"existing destination")
        with self.assertRaisesRegex(ClientError, "source video has 2 frames but the result has 3"):
            rendering.render_video_result(self.result, self.video, self.output)
        self.assertEqual(self.output.read_bytes(), b"existing destination")

    def test_disagreeing_fps_flag_is_rejected(self):
        with self.assertRaisesRegex(ClientError, "reports 15 fps but --fps is 30"):
            rendering.render_video_result(self.result, self.video, self.output, fps=30)

    def test_cosmos_result_renders_from_the_exact_video(self):
        document = physical_result()
        document["action"] = document["action"][:2]
        self.save_json(self.result, document)
        rendering.render_video_result(self.result, self.video, self.output)
        self.assertEqual(int(self.probe(self.output)["nb_read_frames"]), 3)

    def test_cosmos_v2_result_renders_from_the_exact_video(self):
        self.save_json(self.result, physical_result_v2())
        rendering.render_video_result(self.result, self.video, self.output)
        self.assertEqual(int(self.probe(self.output)["nb_read_frames"]), 3)

    def test_cosmos_rejects_an_fps_override(self):
        self.save_json(self.result, physical_result())
        with self.assertRaisesRegex(ClientError, "--fps is fixed"):
            rendering.render_video_result(self.result, self.video, self.output, fps=15)

    @unittest.skipUnless(os.environ.get("EGO_ANNO_MANO_DIR"),
                         "set EGO_ANNO_MANO_DIR to a licensed hand-model asset directory")
    def test_licensed_assets_render_surfaces(self):
        rendering.render_video_result(self.result, self.video, self.output,
                                      mano_dir=os.environ["EGO_ANNO_MANO_DIR"])
        source_pane = self.panes(self.output, 0)[0]
        self.assertGreater(self.joint_pixels(source_pane), 0)


def labels_sidecar(video_path, segments, target="c" * 64):
    """A frozen ego-anno-ar-labels/v1 sidecar bound to the given video bytes."""
    from ego_anno.rendering import labels as labels_module
    normalized = labels_module._segments(segments, "segments")
    return {"schema": "ego-anno-ar-labels/v1",
            "content_sha256": labels_module.content_sha256(normalized),
            "canonical_input_id": "test-clip",
            "input_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            "timebase": {"clock": "uploaded_video_seconds", "origin": "first_frame_pts",
                         "half_open": True, "fps": 15, "duration_sec": 5.0,
                         "record_offset_sec": 0.0},
            "target_identity_sha256": target,
            "segments": segments}


class LabelRenderTests(DagRenderCase):
    """Time-synced AR label overlay on real rendered frames."""

    def setUp(self):
        super().setUp()
        self.save_json(self.result, dag_document(count=75))
        self.make_video(count=75)
        self.labels_path = self.root / "labels.json"
        self.renderer_module = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")

    def write_labels(self, segments, **overrides):
        sidecar = labels_sidecar(self.video, segments, **overrides)
        self.save_json(self.labels_path, sidecar)
        return sidecar

    def band(self, frame_index):
        """The caption band crop of one rendered frame."""
        capture = self.cv2.VideoCapture(str(self.output))
        try:
            capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        self.assertTrue(ok, "cannot decode rendered frame {}".format(frame_index))
        height = frame.shape[0]
        return frame[height - self.renderer_module._caption_band(height):]

    def band_bright(self, frame_index, rows=60):
        """Bright-pixel count in the top rows of the caption band of one frame."""
        region = self.band(frame_index)[:rows]
        return int((region.astype(int).sum(axis=2) > 200).sum())

    def test_labeled_render_syncs_segments_to_frame_pts(self):
        self.write_labels([
            {"start_sec": 0.5, "end_sec": 1.0, "subtask": "pick up the cup"},
            {"start_sec": 2.0, "end_sec": 3.0,
             "subtask": "carry the cup across the room with a long descriptive sentence to force wrapping"},
            {"start_sec": 1 / 15, "end_sec": 2 / 15, "subtask": "edge"},
        ], target=None)
        rendering.render_video_result(self.result, self.video, self.output,
                                      layout="two-pane", labels=str(self.labels_path))
        stream = self.probe(self.output)
        self.assertEqual(int(stream["nb_read_frames"]), 75)
        for frame_index, expect in ((0, 0), (9, 1), (22, 0), (37, 1), (60, 0), (1, 1), (2, 0)):
            bright = self.band_bright(frame_index)
            if expect:
                self.assertGreater(bright, 0, "frame {} lost its label".format(frame_index + 1))
            else:
                self.assertEqual(bright, 0, "frame {} must stay in the gap".format(frame_index + 1))

    def test_sidecar_bound_to_other_video_is_rejected(self):
        sidecar = self.write_labels([{"start_sec": 0.0, "end_sec": 1.0, "subtask": "a"}])
        sidecar["input_sha256"] = "0" * 64
        self.save_json(self.labels_path, sidecar)
        self.output.write_bytes(b"existing destination")
        with self.assertRaisesRegex(ClientError, "does not match the video bytes"):
            rendering.render_video_result(self.result, self.video, self.output,
                                          labels=str(self.labels_path))
        self.assertEqual(self.output.read_bytes(), b"existing destination")

    def test_tampered_sidecar_is_rejected(self):
        sidecar = self.write_labels([{"start_sec": 0.0, "end_sec": 1.0, "subtask": "a"}])
        sidecar["content_sha256"] = "b" * 64
        self.save_json(self.labels_path, sidecar)
        with self.assertRaisesRegex(ClientError, "content_sha256 does not match"):
            rendering.render_video_result(self.result, self.video, self.output,
                                          labels=str(self.labels_path))
        self.assertFalse(self.output.exists())

    def test_missing_labels_draws_explicit_notice_and_never_fabricates(self):
        original_result = self.result.read_bytes()
        rendering.render_video_result(self.result, self.video, self.output, layout="two-pane")
        self.assertGreater(self.band_bright(0), 0, "the no-labels notice must be visible")
        self.assertGreater(self.band_bright(40), 0, "the no-labels notice is persistent")
        self.assertEqual(self.result.read_bytes(), original_result)

    def test_saved_bundle_render_with_labels(self):
        inputs = self.root / "bundle"
        inputs.mkdir()
        self.save_json(inputs / "submission.json",
                       {"schema": "ego-anno-e2e-input/v1", "pipeline": "e2e", "num_frames": 75,
                        "fps": 15, "width": 640, "height": 480,
                        "idempotency_key": "3107cf0b-5648-4f1c-9166-a63b4d0c632d",
                        "annotation_job_id": None, "prompt": None})
        shutil.copyfile(self.video, inputs / "video.mp4")
        v2 = physical_result_v2(75)
        self.save_json(self.result, v2)
        self.write_labels([{"start_sec": 0.5, "end_sec": 1.0, "subtask": "pick up the cup"}])
        rendering.render_result(self.result, inputs, self.output, layout="two-pane",
                                labels=str(self.labels_path))
        self.assertEqual(int(self.probe(self.output)["nb_read_frames"]), 75)
        import numpy as np
        self.assertFalse(np.array_equal(self.band(9), self.band(40)),
                         "the labeled frame's band must differ from the gap frame's")

    def test_cli_forwards_labels(self):
        from ego_anno import cli
        with mock.patch.object(rendering, "render_video_result", return_value=Path("out.mp4")) as render:
            self.assertEqual(cli.main(["render", "--result", "r.json", "--video", "clip.mp4",
                                       "--labels", "labels.json", "-o", "out.mp4"]), 0)
        self.assertEqual(render.call_args[1]["labels"], "labels.json")
        with mock.patch.object(rendering, "render_result", return_value=Path("out.mp4")) as render:
            self.assertEqual(cli.main(["render", "--result", "r.json", "--inputs", "bundle",
                                       "--labels", "labels.json", "-o", "out.mp4"]), 0)
        self.assertEqual(render.call_args[1]["labels"], "labels.json")


@unittest.skipUnless(os.environ.get("EGO_ANNO_REAL_RESULT_DIR"),
                     "set EGO_ANNO_REAL_RESULT_DIR to a directory of real saved artifacts")
class RealSavedArtifactTests(DagRenderCase):
    """Render a real job artifact and diff it against the server-rendered MP4.

    The directory must hold a DAG result (``annotation-result.json``) and the
    exact source video it was produced from (``rgb.mp4``); when the
    server-rendered product (``annotation-rendered.mp4``) is present too, the
    source pane is compared against it pixel-wise. This is the only check that
    ties the wrapper to the shipped server product instead of to a fixture
    invented here, so it is opt-in and never part of the default suite.
    """

    SAMPLES = (0, 1, 60, 120, 180, 240)
    SKIN = (160, 205, 240)

    def setUp(self):
        super().setUp()
        self.root = Path(os.environ["EGO_ANNO_REAL_RESULT_DIR"]).expanduser().resolve()
        self.result = self.root / "annotation-result.json"
        self.video = self.root / "rgb.mp4"
        self.server = self.root / "annotation-rendered.mp4"
        self.output = Path(self.temporary.name) / "real-preview.mp4"
        if not self.result.is_file() or not self.video.is_file():
            self.skipTest("real artifacts not present in {}".format(self.root))

    def require_server_product(self):
        if not self.server.is_file():
            self.skipTest("server-rendered product not present for comparison")

    def mask(self, pane, color, tolerance=60):
        import numpy as np
        return np.abs(pane.astype(int) - np.array(color)).sum(axis=2) < tolerance

    def coverage(self, path):
        """Fraction of painted hand pixels that land on the server product.

        Two lossy encodes never land on identical pixels, so the server mask is
        grown by 2 px: the claim under test is that the client draws its joints
        on the server's hands, not that both codecs agree bit for bit.
        """
        measured = []
        for frame_index in self.SAMPLES:
            client_pane, client_world = self.panes(path, frame_index, layout="two-pane")
            self.assertGreater(self.joint_pixels(client_world), 0, "3D pane lost the hands")
            server_pane, _ = self.panes(self.server, frame_index, layout="two-pane")
            self.assertEqual(client_pane.shape, server_pane.shape)
            for color in HAND_COLORS:
                mine = self.mask(client_pane, color)
                self.assertGreater(int(mine.sum()), 0, "source pane lost a hand")
                theirs = self.cv2.dilate(self.mask(server_pane, color).astype("uint8"),
                                         __import__("numpy").ones((5, 5), "uint8"))
                measured.append(float((mine & theirs.astype(bool)).sum()) / float(mine.sum()))
        return measured

    def test_real_artifact_renders_the_documented_frame_count(self):
        rendering.render_video_result(self.result, self.video, self.output, layout="two-pane")
        stream = self.probe(self.output)
        self.assertEqual(stream["codec_name"], "h264")
        document = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(int(stream["nb_read_frames"]), document["Metadata"]["帧数"])
        self.assertEqual([int(stream["width"]), int(stream["height"])], [1920, 1456])

    def test_skeleton_render_lands_on_the_server_product(self):
        self.require_server_product()
        rendering.render_video_result(self.result, self.video, self.output, layout="two-pane")
        measured = self.coverage(self.output)
        self.assertEqual(len(measured), 2 * len(self.SAMPLES), measured)
        self.assertGreater(min(measured), 0.9, measured)

    @unittest.skipUnless(os.environ.get("EGO_ANNO_MANO_DIR"),
                         "set EGO_ANNO_MANO_DIR to a licensed hand-model asset directory")
    def test_licensed_assets_reproduce_the_server_product(self):
        self.require_server_product()
        rendering.render_video_result(self.result, self.video, self.output,
                                      mano_dir=os.environ["EGO_ANNO_MANO_DIR"], layout="two-pane")
        self.assertGreater(min(self.coverage(self.output)), 0.9)
        _, world_pane = self.panes(self.output, 120, layout="two-pane")
        surfaces = int(self.mask(world_pane, self.SKIN, tolerance=120).sum())
        self.assertGreater(surfaces, 20000, "no MANO surfaces drawn in the 3D pane")


def walking_reach_clip(count=60):
    """Head poses and hands in the product world (first camera frame) plus world-up.

    In a gravity frame (z up) the head yaws +-20 deg at a 25 deg forward-down pitch and
    drifts; the left hand stays within reach except for an annotation gap, the right
    hand moves out of reach for the last third of the clip.
    """
    import numpy as np
    pitch = np.radians(25.0)
    base = np.column_stack(((1.0, 0.0, 0.0),
                            (0.0, -np.sin(pitch), -np.cos(pitch)),
                            (0.0, np.cos(pitch), -np.sin(pitch))))
    pattern = np.asarray([[0.012 * (k % 5 - 2), 0.02 * (k // 5), 0.004 * (k % 3)] for k in range(21)])
    gravity_poses, hands = [], [np.full((count, 21, 3), np.nan), np.full((count, 21, 3), np.nan)]
    for i in range(count):
        yaw = np.radians(20.0) * np.sin(2 * np.pi * i / count)
        rz = np.asarray(((np.cos(yaw), -np.sin(yaw), 0.0), (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0)))
        pose = np.eye(4)
        pose[:3, :3] = rz @ base
        pose[:3, 3] = (0.002 * i, 0.001 * i, 1.6 + 0.01 * np.sin(i / 7.0))
        gravity_poses.append(pose)
        R, C = pose[:3, :3], pose[:3, 3]
        left = C + R @ np.asarray((-0.12 + 0.05 * np.sin(i / 5.0), 0.30, 0.40))
        reach_right = 0.45 if i < 2 * count // 3 else 1.30
        right = C + R @ np.asarray((0.14, 0.30, reach_right))
        if not 20 <= i < 26:
            hands[0][i] = left + pattern @ R.T
        hands[1][i] = right + pattern @ R.T
    first_inverse = np.linalg.inv(gravity_poses[0])
    poses = np.asarray([first_inverse @ pose for pose in gravity_poses])
    joints = [(rows.reshape(-1, 3) @ first_inverse[:3, :3].T + first_inverse[:3, 3]).reshape(count, 21, 3)
              for rows in hands]
    up = first_inverse[:3, :3] @ np.asarray((0.0, 0.0, 1.0))
    return poses, joints, up


class _CapturingWriter:
    """cv2.VideoWriter stand-in that keeps every raw canvas (no codec in the loop)."""
    frames = []

    def __init__(self, *args, **kwargs):
        type(self).frames = []

    def isOpened(self):
        return True

    def write(self, frame):
        type(self).frames.append(frame.copy())

    def release(self):
        pass


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("cv2") is not None,
                     "numpy and cv2 (render extra) not installed")
class BodyPaneTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.body_pane = importlib.import_module("ego_anno.rendering.body_pane")
        self.poses, self.joints, self.up = walking_reach_clip()
        self.pane = self.body_pane.BodyPane(self.poses, np.ones(len(self.poses), bool), self.joints,
                                            self.up, 30.0, (580, 914))

    def test_three_pane_rectangles_are_equal_height_and_fill_the_width(self):
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        self.assertEqual([w for _, _, w, _ in renderer._three_pane_layout(1920, 1080)["panes"]], [760, 580, 580])
        for width, height in ((1920, 1080), (1280, 720), (3840, 2160), (640, 480), (1000, 1000)):
            layout = renderer._three_pane_layout(width, height)
            panes = layout["panes"]
            self.assertEqual(len(panes), 3)
            self.assertEqual({h for _, _, _, h in panes}, {height - layout["band"]})
            self.assertEqual({y for _, y, _, _ in panes}, {0})
            self.assertEqual(panes[0][0], 0)
            for left, right in zip(panes, panes[1:]):
                self.assertEqual(left[0] + left[2], right[0])
            self.assertEqual(panes[-1][0] + panes[-1][2], width)

    def test_observer_is_identical_on_every_frame(self):
        view = self.pane.view_matrix.copy()
        background = self.pane.raster.bg.copy()
        for index in range(self.pane.n):
            frame = self.pane.render(index, [self.joints[0][index], self.joints[1][index]])
            self.assertEqual(frame.shape, (914, 580, 3))
            self.assertTrue(self.np.array_equal(self.pane.view_matrix, view), index)
        self.assertTrue(self.np.array_equal(self.pane.raster.bg, background))
        # level observer: its right axis has no world-up component (no roll)
        self.assertAlmostEqual(float(view[0, 2]), 0.0, places=12)

    def test_bone_lengths_are_constant(self):
        np = self.np
        lengths = np.asarray([[np.linalg.norm(named[a] - named[b]) for a, b in self.body_pane.BONES]
                              for named in (self.pane.body.named(i) for i in range(self.pane.n))])
        self.assertLess(float((lengths.max(axis=0) - lengths.min(axis=0)).max()), 1e-4)
        arm = self.body_pane.UPPER_ARM + self.body_pane.FOREARM
        self.assertLessEqual(float(lengths[:, 9:13].reshape(-1, 2, 2).sum(axis=2).max()), arm + 1e-9)

    def test_wrists_coincide_with_annotated_wrists_when_reachable(self):
        np = self.np
        body = self.pane.body
        reachable = unreachable = 0
        for side, name in enumerate(self.body_pane.SIDES):
            for index in range(body.n):
                wrist = self.joints[side][index][0]
                if not np.isfinite(wrist).all():
                    continue
                target = self.pane.p(wrist)
                joint = body.J[index]
                distance = float(np.linalg.norm(target - joint["shoulder_" + name]))
                gap = float(np.linalg.norm(joint["wrist_" + name] - target))
                if distance <= self.body_pane.REACH:
                    reachable += 1
                    self.assertLess(gap, 1e-6, (name, index))
                else:
                    unreachable += 1
                    self.assertGreater(gap, 0.0)
        self.assertGreater(reachable, 60)
        self.assertGreater(unreachable, 5, "fixture must exercise the out-of-reach dashed line")

    def test_held_arm_and_pinned_head(self):
        np = self.np
        body = self.pane.body
        self.assertTrue(np.allclose(body.J[22]["wrist_left"], body.J[19]["wrist_left"]))
        for index in range(body.n):
            self.assertTrue(np.allclose(body.J[index]["head"], self.pane.p(self.poses[index, :3, 3])))

    def test_display_world_up_is_the_given_up(self):
        np = self.np
        self.assertTrue(np.allclose(self.pane.M[2], self.up / np.linalg.norm(self.up)))
        self.assertTrue(np.allclose(self.pane.M @ self.pane.M.T, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(self.pane.M)), 1.0)


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("cv2") is not None,
                     "numpy and cv2 (render extra) not installed")
class InProcessRendererTests(unittest.TestCase):
    """Run the renderer's main() in-process with a capturing writer: raw canvases, no codec."""

    def setUp(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("fixture video requires system ffmpeg")
        import numpy as np
        self.np = np
        self.cv2 = importlib.import_module("cv2")
        self.renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        poses, joints, _ = walking_reach_clip(count=8)
        document = dag_document(count=8)
        for index, entry in enumerate(document["逐帧轨迹"]):
            frame = entry["帧{}".format(index + 1)]
            frame["相机世界位姿"] = poses[index].tolist()
            for key in ("手0", "手1"):          # measured skeletons: no MANO assets needed
                frame[key]["beta参数"] = None
        self.result = self.root / "annotation.json"
        self.result.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        self.video = self.root / "clip.mp4"
        completed = subprocess.run(
            [shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
             "-i", "testsrc2=s=640x480:r=15", "-frames:v", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(self.video)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if "Unknown encoder" in completed.stderr:
            self.skipTest("system ffmpeg lacks libx264")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def run_main(self, module, *extra):
        argv = ["render_ego_json_overlay.py", str(self.result), "--video", str(self.video),
                "-o", str(self.root / "unused.mp4"), "--fps", "15", *extra]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(self.cv2, "VideoWriter", _CapturingWriter), \
                mock.patch("sys.stdout", new_callable=__import__("io").StringIO):
            self.assertEqual(module.main(), 0)
        return list(_CapturingWriter.frames)

    def test_default_canvas_is_three_panes_at_1920x1080(self):
        frames = self.run_main(self.renderer)
        self.assertEqual(len(frames), 8)
        frame = frames[3]
        self.assertEqual(frame.shape, (1080, 1920, 3))
        panes = self.renderer._three_pane_layout(1920, 1080)["panes"]
        for x, _, _, h in panes[1:]:                       # separators between panes
            self.assertTrue((frame[60:h, x - 1:x + 1] == 60).all())
        overlay, world, body = (frame[y:y + h, x:x + w] for x, y, w, h in panes)
        self.assertGreater(int((abs(body.astype(int) - (50, 110, 240)).sum(axis=2) < 90).sum()), 20)
        self.assertEqual(int(body[-5:, 5:-5].max()), 0, "skeleton pane background is black")
        self.assertGreater(float(world[..., 1].mean()), float(world[..., 0].mean()), "follow pane is green")

    def test_skeleton_up_is_the_follow_view_up(self):
        np = self.np
        captured = {}
        follow = self.renderer._follow_view_track
        module = self.renderer._body_pane_module()

        def track(*args, **kwargs):
            views, report = follow(*args, **kwargs)
            captured["follow"] = report["up"]
            return views, report

        class Pane(module.BodyPane):
            def __init__(self, poses, pose_valid, joints, up, fps, size):
                captured["body"] = np.asarray(up)
                super().__init__(poses, pose_valid, joints, up, fps, size)
                captured["display_up"] = self.M[2]

        for extra in ((), ("--world-up", "0.1,-0.9,-0.4")):
            captured.clear()
            with mock.patch.object(self.renderer, "_follow_view_track", side_effect=track), \
                    mock.patch.object(module, "BodyPane", Pane):
                self.run_main(self.renderer, *extra)
            self.assertTrue(np.array_equal(captured["body"], captured["follow"]), extra)
            self.assertTrue(np.allclose(captured["display_up"], captured["follow"]), extra)
            if extra:
                given = np.asarray((0.1, -0.9, -0.4))
                self.assertTrue(np.allclose(captured["follow"], given / np.linalg.norm(given)))

    def test_two_pane_canvas_is_unchanged_from_e55ff01(self):
        """Pixel-compare --layout two-pane against the renderer as of commit e55ff01."""
        repository = Path(__file__).resolve().parents[1]
        git = shutil.which("git")
        if git is None:
            self.skipTest("git is required to load the e55ff01 renderer")
        completed = subprocess.run([git, "-C", str(repository), "show",
                                    "e55ff01:src/ego_anno/rendering/render_ego_json_overlay.py"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            self.skipTest("commit e55ff01 is not available in this checkout")
        old_path = self.root / "render_ego_json_overlay_e55ff01.py"
        old_path.write_bytes(completed.stdout)
        spec = importlib.util.spec_from_file_location("render_ego_json_overlay_e55ff01", old_path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        before = self.run_main(old)
        after = self.run_main(self.renderer, "--layout", "two-pane")
        self.assertEqual(len(before), len(after))
        band = self.renderer._caption_band(480)
        for index, (a, b) in enumerate(zip(before, after)):
            self.assertEqual(a.shape, (480, 640, 3))
            # The panes are pixel-identical; the caption band deliberately
            # gained the explicit "AR labels: not provided" notice.
            self.assertTrue(self.np.array_equal(a[:a.shape[0] - band], b[:b.shape[0] - band]),
                            "frame {} pane region differs".format(index))
            self.assertFalse(self.np.array_equal(a[-band:], b[-band:]),
                             "frame {} band should carry the no-labels notice".format(index))


if __name__ == "__main__":
    unittest.main()


class HandAssetDependencyTests(unittest.TestCase):
    def test_sparse_regressor_unpickles_without_scipy(self):
        """The render extra is numpy+OpenCV only; a pickled scipy csc_matrix must
        load through the numpy stand-in and rebuild the same dense matrix."""
        import pickle
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")

        class FakeCsc:  # pickles like scipy: module scipy.sparse._csc, class csc_matrix
            pass
        FakeCsc.__module__ = "scipy.sparse._csc"
        FakeCsc.__qualname__ = FakeCsc.__name__ = "csc_matrix"
        dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        value = FakeCsc()
        value.__dict__.update({"data": [1.0, 3.0, 2.0], "indices": [0, 1, 0],
                               "indptr": [0, 1, 2, 3], "_shape": (2, 3)})
        import sys, types
        module = types.ModuleType("scipy.sparse._csc"); module.csc_matrix = FakeCsc
        saved = {k: sys.modules.get(k) for k in ("scipy", "scipy.sparse", "scipy.sparse._csc")}
        sys.modules["scipy.sparse._csc"] = module
        try:
            blob = pickle.dumps({"J_regressor": value}, protocol=2)
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        import io
        loaded = renderer._HandModelUnpickler(io.BytesIO(blob), encoding="latin1").load()["J_regressor"]
        np.testing.assert_array_equal(loaded.toarray(), dense)
