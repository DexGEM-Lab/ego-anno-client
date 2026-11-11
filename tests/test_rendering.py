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
                         {"cosmos_e2e.py", "render_ego_json_overlay.py"})
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
        self.assertEqual([stream["width"], stream["height"]], [320, 240])
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
            view = renderer._rear_view(pose, [hand + pose[:3, 3], hand + pose[:3, 3]], 640, 400)
            marker, _ = renderer._camera_marker_points(pose)
            points = np.vstack((marker, hand + pose[:3, 3], hand + pose[:3, 3]))
            pixels, _, valid, _ = renderer._project_view(points, view, (640, 400))
            self.assertTrue(valid.all(), "camera and both hands must be in front of the view")
            self.assertTrue(np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < 640)))
            self.assertTrue(np.all((pixels[:, 1] >= 0) & (pixels[:, 1] < 400)))
            # The lateral offset is in the camera-right basis, with no world-up spin.
            self.assertGreater(float(np.dot(view["eye"] - pose[:3, 3], pose[:3, 0])), 0.0)

    def test_spatial_scene_contains_camera_mesh_frustum_and_full_trail(self):
        import numpy as np
        renderer = importlib.import_module("ego_anno.rendering.render_ego_json_overlay")
        vertices, faces = renderer._build_cinema_camera()
        self.assertGreater(len(vertices), 80)
        self.assertGreater(len(faces), 100)
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [0.3, 0.0, 0.0]
        world = (pose[:3, :3] @ vertices.T).T + pose[:3, 3]
        np.testing.assert_allclose(world - vertices, np.tile([0.3, 0.0, 0.0], (len(vertices), 1)))
        frustum = renderer._camera_frustum_points(pose, self.document["Metadata"]["相机内参"], (640, 480))
        self.assertEqual(frustum.shape, (6, 3))
        self.assertTrue(np.isfinite(frustum).all())
        centers = np.asarray([[-0.8, 0, 0], [0.3, 0, 0], [1.2, 0, 0]], dtype=float)
        hand = np.asarray(self.document["逐帧轨迹"][0]["帧1"]["手0"]["手部21点坐标"], dtype=float) + pose[:3, 3]
        rendered = renderer._render_world((640, 400), pose, centers, 2, [hand, hand], [None, None], None,
                                          [np.full(10, np.nan), np.full(10, np.nan)],
                                          np.asarray(self.document["Metadata"]["相机内参"], dtype=float), (640, 480))
        self.assertEqual(rendered.shape, (400, 640, 3))
        # Red cinema camera/frustum pixels and blue full-trail pixels are painted.
        self.assertGreater(int(((rendered[:, :, 2] > 120) & (rendered[:, :, 2] > rendered[:, :, 1] * 1.5)).sum()), 20)
        self.assertGreater(int(((rendered[:, :, 0] > 80) & (rendered[:, :, 2] > 80)).sum()), 20)

    def test_render_labels_are_english(self):
        source = Path(importlib.import_module("ego_anno.rendering.render_ego_json_overlay").__file__).read_text(encoding="utf-8")
        self.assertIn("Hand joints overlay", source)
        self.assertIn("Camera and hands: side-rear follow view", source)
        self.assertIn("side-rear follow view: camera and hands", source)
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

    def panes(self, path, frame_index=0):
        capture = self.cv2.VideoCapture(str(path))
        try:
            capture.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
        finally:
            capture.release()
        self.assertTrue(ok, "cannot decode rendered frame {}".format(frame_index))
        height, width = frame.shape[:2]
        band = min(220, max(165, int(round(height * 0.145))))
        pane_height, pane_width = height - band, width // 2
        return frame[:pane_height, :pane_width], frame[:pane_height, pane_width:pane_width * 2]

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
        self.assertEqual([stream["width"], stream["height"]], [640, 480])
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        self.assertAlmostEqual(numerator / denominator, 15)
        payload = self.output.read_bytes()
        self.assertLess(payload.index(b"moov"), payload.index(b"mdat"), "MP4 must enable faststart")
        self.assertEqual(self.result.read_bytes(), original_result)
        self.assertFalse(list(self.root.glob(".ego-anno-render-*")))
        for frame_index in range(3):
            source_pane, world_pane = self.panes(self.output, frame_index)
            self.assertGreater(self.joint_pixels(source_pane), 0, "no joints over the source frame")
            self.assertGreater(self.joint_pixels(world_pane), 0, "no joints in the 3D pane")

    def test_layout_override_is_honoured(self):
        rendering.render_video_result(self.result, self.video, self.output, width=1280, height=960)
        stream = self.probe(self.output)
        self.assertEqual([stream["width"], stream["height"]], [1280, 960])

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
        source_pane, _ = self.panes(self.output, 0)
        self.assertGreater(self.joint_pixels(source_pane), 0)


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
            client_pane, client_world = self.panes(path, frame_index)
            self.assertGreater(self.joint_pixels(client_world), 0, "3D pane lost the hands")
            server_pane, _ = self.panes(self.server, frame_index)
            self.assertEqual(client_pane.shape, server_pane.shape)
            for color in HAND_COLORS:
                mine = self.mask(client_pane, color)
                self.assertGreater(int(mine.sum()), 0, "source pane lost a hand")
                theirs = self.cv2.dilate(self.mask(server_pane, color).astype("uint8"),
                                         __import__("numpy").ones((5, 5), "uint8"))
                measured.append(float((mine & theirs.astype(bool)).sum()) / float(mine.sum()))
        return measured

    def test_real_artifact_renders_the_documented_frame_count(self):
        rendering.render_video_result(self.result, self.video, self.output)
        stream = self.probe(self.output)
        self.assertEqual(stream["codec_name"], "h264")
        document = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(int(stream["nb_read_frames"]), document["Metadata"]["帧数"])
        self.assertEqual([int(stream["width"]), int(stream["height"])], [1920, 1456])

    def test_skeleton_render_lands_on_the_server_product(self):
        self.require_server_product()
        rendering.render_video_result(self.result, self.video, self.output)
        measured = self.coverage(self.output)
        self.assertEqual(len(measured), 2 * len(self.SAMPLES), measured)
        self.assertGreater(min(measured), 0.9, measured)

    @unittest.skipUnless(os.environ.get("EGO_ANNO_MANO_DIR"),
                         "set EGO_ANNO_MANO_DIR to a licensed hand-model asset directory")
    def test_licensed_assets_reproduce_the_server_product(self):
        self.require_server_product()
        rendering.render_video_result(self.result, self.video, self.output,
                                      mano_dir=os.environ["EGO_ANNO_MANO_DIR"])
        self.assertGreater(min(self.coverage(self.output)), 0.9)
        _, world_pane = self.panes(self.output, 120)
        surfaces = int(self.mask(world_pane, self.SKIN, tolerance=120).sum())
        self.assertGreater(surfaces, 20000, "no MANO surfaces drawn in the 3D pane")


if __name__ == "__main__":
    unittest.main()
