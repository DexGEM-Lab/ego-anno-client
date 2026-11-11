import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from ego_anno import EgoAnnoClient, ClientError
from ego_anno import client
from test_ego_anno import Connection, Response


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class SavedInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "original.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            "color=c=blue:s=640x640:r=15:d=2.2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(self.source),
        ], check=True)
        self.key = "test-sensitive-key-not-for-persistence"
        self.job = "12345678-1234-4234-8234-123456789abc"
        self.idem = "11111111-2222-4333-8444-555555555555"
        self.camera = {"source_size_wh": [640, 640], "source_K": [
            [300, 0, 319.5], [0, 300, 319.5], [0, 0, 1],
        ]}

    def test_success_survives_temp_cleanup_and_is_ready_for_offline_renderer(self):
        destination = self.root / "render-inputs"
        response = Connection(Response(202, json.dumps({"annotation_job_id": self.job}).encode()))
        with mock.patch.object(client, "connect", return_value=response), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            job = EgoAnnoClient(self.key).submit_e2e(
                str(self.source), camera=self.camera, prompt="wave", idempotency_key=self.idem,
                save_inputs=str(destination),
            )
        self.assertEqual(job, self.job)
        meta = json.loads((destination / "submission.json").read_text())
        self.assertEqual(meta["annotation_job_id"], self.job)
        self.assertEqual(meta["idempotency_key"], self.idem)
        self.assertEqual(meta["num_frames"], 33)
        self.assertEqual(meta["fps"], 15)
        self.assertEqual(json.loads((destination / "camera.json").read_text()), self.camera)
        self.assertNotIn(self.key, (destination / "submission.json").read_text())
        # The original TemporaryDirectory is gone: the durable video still decodes
        # and has the timeline expected by the downloaded 32-row action output.
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json",
            str(destination / "video.mp4"),
        ], check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual((stream["nb_read_frames"], stream["width"], stream["height"]), ("33", 640, 640))
        self.assertEqual(stream["r_frame_rate"], "15/1")

    def test_failed_submission_preserves_inputs_and_retry_identity_without_false_job_id(self):
        destination = self.root / "failed-inputs"
        response = Connection(Response(401, b'{"error":{"code":"invalid_api_key","message":"invalid"}}'))
        with mock.patch.object(client, "connect", return_value=response), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaises(ClientError):
                EgoAnnoClient(self.key).submit_e2e(
                    str(self.source), idempotency_key=self.idem, save_inputs=str(destination),
                )
        meta = json.loads((destination / "submission.json").read_text())
        self.assertIsNone(meta["annotation_job_id"])
        self.assertEqual(meta["idempotency_key"], self.idem)
        self.assertTrue((destination / "video.mp4").is_file())
        self.assertFalse((destination / "camera.json").exists())
        self.assertNotIn(self.key, json.dumps(meta))
        # A second run must not silently replace video/calibration from the first.
        with mock.patch.object(client, "connect", side_effect=AssertionError("must not send")):
            with self.assertRaises(ClientError):
                EgoAnnoClient(self.key).submit_e2e(str(self.source), save_inputs=str(destination))
        self.assertEqual(json.loads((destination / "submission.json").read_text()), meta)


class OfflineCliTests(unittest.TestCase):
    def test_render_errors_offline_without_asking_for_credentials(self):
        env = dict(os.environ)
        env.pop("EGO_ANNO_API_KEY", None)
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([
                sys.executable, "-m", "ego_anno", "render", "--result", directory + "/missing.json",
                "--inputs", directory, "-o", directory + "/preview.mp4",
            ], env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 1)
        self.assertIn("Error:", process.stderr)
        self.assertNotIn("API key:", process.stderr + process.stdout)
        self.assertNotIn("Traceback", process.stderr)

    def test_dag_does_not_silently_accept_save_inputs(self):
        process = subprocess.run([
            sys.executable, "-m", "ego_anno", "submit", "--video", "x.mp4", "--save-inputs", "saved",
        ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 2)


if __name__ == "__main__":
    unittest.main()
