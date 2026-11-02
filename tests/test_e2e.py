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

import ego_anno
from ego_anno import client, e2e

from test_ego_anno import Connection, Response

ACCEPTED = b'{"annotation_job_id":"876c2588-abf5-4cb5-bca7-3da815d0c3b2"}'


def wire(conn):
    return b"".join(w[1] for w in conn.writes if w[0] == "send")


def field_names(body):
    return [part.split(b'"')[1] for part in body.split(b'Content-Disposition: form-data; name=')[1:]]


class SourceRuleTests(unittest.TestCase):
    def ok(self, fps=30.0, duration=10.0, width=1920, height=1080):
        e2e.check_source(fps, duration, width, height)

    def rejected(self, message, **kwargs):
        with self.assertRaisesRegex(ego_anno.ClientError, message):
            self.ok(**kwargs)

    def test_fps_boundaries(self):
        self.rejected("15 to 30 fps", fps=14.99)
        self.ok(fps=15.0)
        self.ok(fps=30.0)
        self.ok(fps=30000 / 1001)
        self.rejected("15 to 30 fps", fps=30.01)
        self.rejected("frame rate", fps=None)

    def test_duration_boundaries(self):
        self.ok(duration=20.0)
        self.rejected("at most 20 s", duration=20.01)
        self.rejected("duration", duration=None)

    def test_short_side_boundaries(self):
        for width, height in ((639, 1280), (1280, 639), (1441, 2560), (2560, 1441)):
            self.rejected("short side of 640 to 1440", width=width, height=height)
        for width, height in ((640, 1280), (1280, 640), (1440, 2560), (2560, 1440)):
            self.ok(width=width, height=height)

    def test_frame_count_trimming(self):
        self.assertEqual(e2e.frame_count(241), 241)
        self.assertEqual(e2e.frame_count(240), 237)
        self.assertEqual(e2e.frame_count(300), 241)
        self.assertEqual(e2e.frame_count(150), 149)
        self.assertEqual(e2e.frame_count(20), 17)
        self.assertEqual(e2e.frame_count(17), 17)
        for available in (16, 1, 0):
            with self.assertRaisesRegex(ego_anno.ClientError, "at least 17 frames"):
                e2e.frame_count(available)

    def test_slot_count_uses_timestamps(self):
        thirty = [round(i / 30, 6) for i in range(540)]  # 18 s at 30 fps
        self.assertEqual(e2e.slot_count(thirty, thirty[-1] + 1 / 30), 270)
        twenty_four = [round(i / 24, 6) for i in range(240)]  # 10 s at 24 fps
        self.assertEqual(e2e.slot_count(twenty_four, twenty_four[-1] + 1 / 24), 150)
        offset = [0.5 + t for t in twenty_four]
        self.assertEqual(e2e.slot_count(offset, offset[-1] + 1 / 24), 150)

    def test_nearest_frame_selection(self):
        twenty_four = [round(i / 24, 6) for i in range(240)]
        # Slot k at k/15 s; 24/15 = 1.6 source frames per slot; ties go to the earlier frame.
        self.assertEqual(e2e.nearest_indices(twenty_four, 6), [0, 2, 3, 5, 6, 8])
        thirty = [round(i / 30, 6) for i in range(60)]
        self.assertEqual(e2e.nearest_indices(thirty, 4), [0, 2, 4, 6])
        fifteen = [1.0 + round(i / 15, 6) for i in range(20)]
        self.assertEqual(e2e.nearest_indices(fifteen, 17), list(range(17)))
        # Variable frame rate: a gap picks the nearest real frame by timestamp.
        vfr = [0.0, 0.03, 0.2, 0.21, 0.27]
        # Slots 0, .067, .133, .2, .267 s -> frames at 0, .03, .2, .2, .27 s.
        self.assertEqual(e2e.nearest_indices(vfr, 5), [0, 1, 2, 2, 4])

    def test_output_geometry(self):
        self.assertEqual(e2e.output_geometry(1920, 1080), (1920, 1080, False))
        self.assertEqual(e2e.output_geometry(2560, 1440), (1920, 1080, True))
        self.assertEqual(e2e.output_geometry(1440, 2560), (1080, 1920, True))
        self.assertEqual(e2e.output_geometry(1600, 1200), (1440, 1080, True))
        self.assertEqual(e2e.output_geometry(1081, 1921), (1080, 1920, True))
        self.assertEqual(e2e.output_geometry(641, 853), (640, 852, False))
        with self.assertRaisesRegex(ego_anno.ClientError, "long side of at most 2560"):
            e2e.output_geometry(3840, 1080)

    def test_prompt_rules(self):
        self.assertIsNone(e2e.check_prompt(None))
        self.assertEqual(e2e.check_prompt("x" * 1000), "x" * 1000)
        for prompt in ("", "   ", "x" * 1001, "a\x00b"):
            with self.assertRaisesRegex(ego_anno.ClientError, "prompt"):
                e2e.check_prompt(prompt)


class CameraTests(unittest.TestCase):
    CAMERA = {"source_size_wh": [2560, 1440], "source_K": [[1800, 0, 1279.5], [0, 1700, 719.5], [0, 0, 1]]}

    def test_rescale_pixel_center_convention(self):
        camera = e2e.load_camera(self.CAMERA)
        field = e2e.rescale_camera(camera, 2560, 1440, e2e.output_geometry(2560, 1440))
        self.assertEqual(field, {"source_size_wh": [1920, 1080],
                                 "source_K": [[1350.0, 0.0, 959.5], [0.0, 1275.0, 539.5], [0.0, 0.0, 1.0]]})
        off_center = e2e.load_camera({"source_size_wh": [1600, 1200],
                                      "source_K": [[1000, 0, 100], [0, 1000, 50], [0, 0, 1]]})
        field = e2e.rescale_camera(off_center, 1600, 1200, e2e.output_geometry(1600, 1200))
        sx, sy = 1440 / 1600, 1080 / 1200
        expected = [[1000 * sx, 0.0, (100 + 0.5) * sx - 0.5], [0.0, 1000 * sy, (50 + 0.5) * sy - 0.5],
                    [0.0, 0.0, 1.0]]
        self.assertEqual(field["source_size_wh"], [1440, 1080])
        for row, want in zip(field["source_K"], expected):
            for value, target in zip(row, want):
                self.assertAlmostEqual(value, target, places=12)

    def test_unscaled_and_odd_crop_keep_k(self):
        camera = e2e.load_camera({"source_size_wh": [641, 853], "source_K": [[500, 0, 320], [0, 500, 426], [0, 0, 1]]})
        field = e2e.rescale_camera(camera, 641, 853, e2e.output_geometry(641, 853))
        self.assertEqual(field, {"source_size_wh": [640, 852],
                                 "source_K": [[500.0, 0.0, 320.0], [0.0, 500.0, 426.0], [0.0, 0.0, 1.0]]})

    def test_size_must_match_displayed_video(self):
        camera = e2e.load_camera(self.CAMERA)
        with self.assertRaisesRegex(ego_anno.ClientError, "does not match the video's displayed size 1440x2560"):
            e2e.rescale_camera(camera, 1440, 2560, e2e.output_geometry(1440, 2560))

    def test_camera_validation(self):
        bad = [
            {"source_size_wh": [1, 1]},
            {"source_size_wh": [1, 1], "source_K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "extra": 1},
            {"source_size_wh": [0, 1], "source_K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"source_size_wh": [1.5, 1], "source_K": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"source_size_wh": [1, 1], "source_K": [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]},
            {"source_size_wh": [1, 1], "source_K": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            {"source_size_wh": [1, 1], "source_K": [[1, 0, 0], [0, 1, 0], [0, 0, 2]]},
            {"source_size_wh": [1, 1], "source_K": [[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]]},
            {"source_size_wh": [1, 1], "source_K": [[1, 0, 0], [0, 1, 0]]},
            [],
        ]
        for payload in bad:
            with self.assertRaises(ego_anno.ClientError, msg=payload):
                e2e.load_camera(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            path.write_text(json.dumps(self.CAMERA))
            self.assertEqual(e2e.load_camera(str(path))[0], [2560, 1440])
            path.write_text("{not json")
            with self.assertRaisesRegex(ego_anno.ClientError, "cannot read camera JSON"):
                e2e.load_camera(str(path))


class SubmitTests(unittest.TestCase):
    def submit_e2e(self, connections, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "clip.mov"
            source.write_bytes(b"source")

            def preprocess(video, workdir, camera):
                self.assertEqual(video, str(source))
                path = os.path.join(workdir, "clip.mp4")
                Path(path).write_bytes(b"processed video")
                field = None if camera is None else e2e.rescale_camera(
                    camera, 2560, 1440, e2e.output_geometry(2560, 1440))
                return e2e.Prepared(path, 149, 1920, 1080, field)

            with mock.patch.object(client.e2e, "preprocess", side_effect=preprocess), \
                 mock.patch.object(client, "connect", side_effect=connections), \
                 mock.patch.object(ego_anno.transport.time, "sleep"), \
                 mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
                return ego_anno.EgoAnnoClient("key").submit_e2e(str(source), **kwargs)

    def test_fields_with_prompt_and_camera(self):
        conn = Connection(Response(202, ACCEPTED))
        self.submit_e2e([conn], prompt="pick up the cup", camera=CameraTests.CAMERA)
        body = wire(conn)
        self.assertEqual(field_names(body), [b"input_mode", b"pipeline", b"prompt", b"camera", b"video"])
        self.assertIn(b'name="pipeline"\r\n\r\ne2e\r\n', body)
        self.assertIn(b'name="input_mode"\r\n\r\nmono\r\n', body)
        self.assertIn(b'name="prompt"\r\n\r\npick up the cup\r\n', body)
        camera = body.split(b'name="camera"\r\n\r\n')[1].split(b"\r\n")[0]
        self.assertEqual(json.loads(camera), {"source_size_wh": [1920, 1080],
                                              "source_K": [[1350.0, 0.0, 959.5], [0.0, 1275.0, 539.5],
                                                           [0.0, 0.0, 1.0]]})
        self.assertIn(b'filename="clip.mp4"\r\nContent-Type: video/mp4\r\n\r\nprocessed video\r\n', body)
        self.assertNotIn(b"hand_refinement_method", body)
        self.assertNotIn(b"render", body)

    def test_without_prompt_or_camera_sends_no_fields(self):
        conn = Connection(Response(202, ACCEPTED))
        self.submit_e2e([conn])
        self.assertEqual(field_names(wire(conn)), [b"input_mode", b"pipeline", b"video"])

    def test_retry_reuses_key_and_processed_bytes(self):
        error = b'{"error":{"code":"busy","message":"later"}}'
        connections = [Connection(Response(503, error)), Connection(Response(202, ACCEPTED))]
        key = "11111111-2222-4333-8444-555555555555"
        self.submit_e2e(connections, prompt="wave", idempotency_key=key)
        keys = [[w[2] for w in c.writes if w[:2] == ("header", "Idempotency-Key")] for c in connections]
        self.assertEqual(keys, [[key], [key]])
        self.assertEqual(wire(connections[0]), wire(connections[1]))

    def test_invalid_inputs_fail_before_preprocessing(self):
        api = ego_anno.EgoAnnoClient("key")
        with mock.patch.object(client.e2e, "preprocess") as preprocess:
            with self.assertRaises(ValueError):
                api.submit_e2e("clip.mp4", idempotency_key="not-a-uuid")
            with self.assertRaisesRegex(ego_anno.ClientError, "prompt"):
                api.submit_e2e("clip.mp4", prompt=" ")
            with self.assertRaisesRegex(ego_anno.ClientError, "camera"):
                api.submit_e2e("clip.mp4", camera={"source_size_wh": [1, 1]})
        preprocess.assert_not_called()

    def test_dag_mono_body_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video")
            conn = Connection(Response(202, ACCEPTED))
            with mock.patch.object(client, "connect", return_value=conn), \
                 mock.patch.object(client.e2e, "preprocess") as preprocess, \
                 mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
                ego_anno.EgoAnnoClient("key").submit_mono(str(video))
            preprocess.assert_not_called()
            body = wire(conn)
            self.assertEqual(field_names(body), [b"input_mode", b"video"])
            self.assertIn(b'name="input_mode"\r\n\r\nmono\r\n', body)
            self.assertIn(b'filename="video.mp4"\r\nContent-Type: video/mp4\r\n\r\nvideo\r\n', body)
            request = [w for w in conn.writes if w[0] == "request"]
            self.assertEqual(request, [("request", "POST", "/api/v2/annotations")])

    def test_missing_ffmpeg_is_a_clear_error(self):
        with mock.patch.object(e2e.shutil, "which", side_effect=lambda name: None if name == "ffmpeg" else "/x"):
            with self.assertRaisesRegex(ego_anno.ClientError, "ffmpeg not found on PATH. Install FFmpeg"):
                e2e.preprocess("clip.mp4", ".")


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = dict(os.environ, EGO_ANNO_API_KEY="key")
        return subprocess.run([sys.executable, "-m", "ego_anno", "submit"] + list(args), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def test_e2e_usage_errors(self):
        for args, message in (
            (["--pipeline", "e2e", "--video", "v.mp4", "--method", "baseline"], "cannot be combined"),
            (["--pipeline", "e2e", "--left", "l.mp4", "--right", "r.mp4", "--calibration", "c.json"],
             "requires --video"),
            (["--pipeline", "e2e", "--video", "v.mp4", "--left", "l.mp4"], "cannot be combined"),
            (["--video", "v.mp4", "--prompt", "wave"], "require --pipeline e2e"),
            (["--pipeline", "dag", "--video", "v.mp4", "--camera", "c.json"], "require --pipeline e2e"),
            (["--pipeline", "e2e", "--video", "v.mp4", "--render"], "unrecognized arguments"),
            (["--pipeline", "other", "--video", "v.mp4"], "invalid choice"),
        ):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 2, args)
            self.assertIn(message, result.stderr)

    def test_cli_dispatch(self):
        from ego_anno import cli
        with mock.patch.object(cli.EgoAnnoClient, "submit_e2e") as submit_e2e, \
             mock.patch.object(cli.EgoAnnoClient, "submit_mono") as submit_mono, \
             mock.patch.dict(os.environ, {"EGO_ANNO_API_KEY": "key"}):
            self.assertEqual(cli.main(["submit", "--video", "v.mp4"]), 0)
            self.assertEqual(cli.main(["submit", "--video", "v.mp4", "--pipeline", "e2e", "--prompt", "wave",
                                       "--camera", "c.json", "--wait"]), 0)
        submit_mono.assert_called_once_with("v.mp4", idempotency_key=None, wait=False)
        submit_e2e.assert_called_once_with("v.mp4", prompt="wave", camera="c.json", idempotency_key=None,
                                           wait=True)

    def test_cli_reports_preprocessing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli("--pipeline", "e2e", "--video", str(Path(directory) / "missing.mp4"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("Error: ", result.stderr)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe not installed")
class FfmpegTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = temp.name
        self.out = os.path.join(self.directory, "out")
        os.mkdir(self.out)
        stderr = mock.patch("sys.stderr", new_callable=io.StringIO)
        stderr.start()
        self.addCleanup(stderr.stop)

    def make(self, name, rate, seconds, size, extra=()):
        path = os.path.join(self.directory, name)
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i",
                        "testsrc=size={}:rate={}".format(size, rate), "-t", str(seconds), "-c:v", "libx264",
                        "-preset", "ultrafast"] + list(extra) + [path], check=True)
        return path

    def test_resample_and_verify(self):
        # 4:4:4 allows odd dimensions; the output drops the last column/row.
        video = self.make("clip.mp4", 24, 2, "853x641", ["-pix_fmt", "yuv444p"])
        prepared = e2e.preprocess(video, self.out)
        self.assertEqual(prepared[1:], (29, 852, 640, None))
        probe = e2e._probe_json("ffprobe", ["-count_frames", "-show_streams", prepared.path])["streams"]
        self.assertEqual([(s["codec_type"], s["pix_fmt"], s["avg_frame_rate"], s["nb_read_frames"]) for s in probe],
                         [("video", "yuv420p", "15/1", "29")])
        with self.assertRaisesRegex(ego_anno.ClientError, "refusing to overwrite"):
            e2e.preprocess(video, self.out)

    def test_rotation_uses_displayed_size(self):
        coded = self.make("coded.mp4", 30, 1.2, "1280x720")
        video = os.path.join(self.directory, "rotated.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", coded, "-c", "copy",
                        "-metadata:s:v:0", "rotate=90", video], check=True)
        camera = e2e.load_camera({"source_size_wh": [720, 1280],
                                  "source_K": [[900, 0, 359.5], [0, 900, 639.5], [0, 0, 1]]})
        prepared = e2e.preprocess(video, self.out, camera)
        self.assertEqual((prepared.num_frames, prepared.width, prepared.height), (17, 720, 1280))
        self.assertEqual(prepared.camera["source_size_wh"], [720, 1280])
        stream = e2e._probe_json("ffprobe", ["-show_streams", prepared.path])["streams"][0]
        self.assertEqual((stream["width"], stream["height"]), (720, 1280))
        self.assertFalse(stream.get("side_data_list"))

    def test_rejects_low_fps(self):
        video = self.make("slow.mp4", 10, 2, "854x640")
        with self.assertRaisesRegex(ego_anno.ClientError, "15 to 30 fps"):
            e2e.preprocess(video, self.out)


if __name__ == "__main__":
    unittest.main()
