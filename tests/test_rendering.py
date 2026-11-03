"""Public offline rendering tests; optional real-media cases use no model assets."""
import hashlib
import importlib
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
    def test_only_cosmos_schema_is_accepted(self):
        for result in ({"schema": "ego.annotation.product_json.v3"}, {}, {"schema": "cosmos-e2e-result/v2"}):
            with self.subTest(result=result):
                self.save_json(self.result_path, result)
                self.rejected("only accepts cosmos-e2e-result/v1")

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

    def test_transcode_failure_preserves_destination_and_cleans_temporary_files(self):
        real_run = rendering._run

        def fail_encoding(command, environment, label):
            if label == "ffmpeg H.264 encoding":
                raise ClientError("simulated ffmpeg encoding failure")
            return real_run(command, environment, label)

        with mock.patch.object(rendering, "_run", side_effect=fail_encoding):
            self.rejected("encoding failure")


if __name__ == "__main__":
    unittest.main()
