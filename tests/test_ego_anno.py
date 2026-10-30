import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ego_anno
from ego_anno import client, transport


class Response:
    def __init__(self, status, body=b"", length=None):
        self.status = status
        self.body = io.BytesIO(body)
        self.length = length

    def read(self, n=-1):
        return self.body.read(n)

    def getheader(self, name):
        return self.length if name == "Content-Length" else None


class Connection:
    def __init__(self, response=None, cert=b"certificate"):
        self.response = response
        self.cert = cert
        self.sock = self
        self.writes = []
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def getpeercert(self, binary_form=False):
        return self.cert

    def putrequest(self, method, path):
        self.writes.append(("request", method, path))

    def putheader(self, name, value):
        self.writes.append(("header", name, value))

    def endheaders(self):
        self.writes.append(("end",))

    def send(self, data):
        self.writes.append(("send", data))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class ClientTests(unittest.TestCase):
    def test_pin_mismatch_sends_nothing(self):
        conn = Connection()
        with mock.patch.object(transport.http.client, "HTTPSConnection", return_value=conn), \
             mock.patch.dict(os.environ, {"EGO_ANNO_PIN_SHA256": "0" * 64}):
            with self.assertRaisesRegex(ego_anno.PinError, "server certificate does not match the pinned fingerprint"):
                transport.connect()
        self.assertEqual(conn.writes, [])
        self.assertTrue(conn.closed)

    def test_pin_match_then_authorization(self):
        conn = Connection()
        pin = hashlib.sha256(conn.cert).hexdigest().upper()
        with mock.patch.object(transport.http.client, "HTTPSConnection", return_value=conn), \
             mock.patch.dict(os.environ, {"EGO_ANNO_PIN_SHA256": pin}):
            verified = transport.connect()
            transport.headers(verified, "GET", "/test", "secret")
        self.assertEqual(conn.writes[1], ("header", "Authorization", "Bearer secret"))

    def test_multipart_content_length_and_bounded_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            # Three blocks plus a tail exposes attempts to read the entire file.
            video.write_bytes(b"x" * (3 * transport.CHUNK_SIZE + 19))
            calib = Path(directory) / "calibration.json"
            calib.write_text("{}")
            for fields, files, expected in [
                ([("input_mode", "mono")], [("video", video, "video/mp4")], [b'name="video"', b'\r\nmono\r\n']),
                ([("input_mode", "stereo"), ("hand_refinement_method", "baseline")],
                 [("left_video", video, "video/mp4"), ("right_video", video, "video/mp4"),
                  ("calibration", calib, "application/json")],
                 [b'name="left_video"', b'name="right_video"', b'name="calibration"', b'name="hand_refinement_method"', b'\r\nstereo\r\n'])]:
                body = transport.Multipart(fields, files)
                try:
                    conn = Connection()
                    body.send(conn)
                    chunks = [item[1] for item in conn.writes if item[0] == "send"]
                    self.assertEqual(sum(map(len, chunks)), body.length)
                    self.assertLessEqual(max(map(len, chunks)), transport.CHUNK_SIZE)
                    wire = b"".join(chunks)
                    for field in expected:
                        self.assertIn(field, wire)
                    self.assertTrue(wire.endswith(("--" + body.boundary + "--\r\n").encode()))
                finally:
                    body.close()

    def test_retry_429_and_5xx_with_same_key_and_body(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video bytes")
            error = json.dumps({"error": {"code": "busy", "message": "try again"}}).encode()
            connections = [Connection(Response(status, error)) for status in (429, 503)]
            connections.append(Connection(Response(202, b'{"annotation_job_id":"job"}')))
            with mock.patch.object(client, "connect", side_effect=connections), mock.patch.object(transport.time, "sleep"), \
                 mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
                ego_anno.EgoAnnoClient("api-key").submit_mono(
                    str(video), idempotency_key="876c2588-abf5-4cb5-bca7-3da815d0c3b2")
            keys = [[w[2] for w in c.writes if w[:2] == ("header", "Idempotency-Key")] for c in connections]
            self.assertEqual(keys, [["876c2588-abf5-4cb5-bca7-3da815d0c3b2"]] * 3)
            bodies = [b"".join(w[1] for w in c.writes if w[0] == "send") for c in connections]
            self.assertEqual(bodies[0], bodies[1])
            self.assertEqual(bodies[1], bodies[2])

    def test_4xx_no_retry_and_error_code(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video")
            conn = Connection(Response(409, b'{"error":{"code":"idempotency_conflict","message":"conflict"}}'))
            with mock.patch.object(client, "connect", return_value=conn) as connect, \
                 mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaisesRegex(ego_anno.ClientError, "HTTP 409: idempotency_conflict: conflict"):
                    ego_anno.EgoAnnoClient("key").submit_mono(str(video))
            self.assertEqual(connect.call_count, 1)

    def test_early_4xx_during_upload_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video")
            conn = Connection(Response(409, b'{"error":{"code":"insufficient_annotation_credits","message":"no credits"}}'))
            conn.send = mock.Mock(side_effect=BrokenPipeError("server closed upload"))
            with mock.patch.object(client, "connect", return_value=conn) as connect, \
                 mock.patch("sys.stdout", new_callable=io.StringIO):
                with self.assertRaisesRegex(ego_anno.ClientError, "HTTP 409: insufficient_annotation_credits: no credits"):
                    ego_anno.EgoAnnoClient("key").submit_mono(str(video))
            self.assertEqual(connect.call_count, 1)

    def test_stereo_submit_uses_backend_method_field(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.mp4"
            video.write_bytes(b"video")
            calibration = Path(directory) / "calibration.json"
            calibration.write_text("{}")
            conn = Connection(Response(202, b'{"annotation_job_id":"job"}'))
            with mock.patch.object(client, "connect", return_value=conn), \
                 mock.patch("sys.stdout", new_callable=io.StringIO), mock.patch("sys.stderr", new_callable=io.StringIO):
                ego_anno.EgoAnnoClient("key").submit_stereo(
                    str(video), str(video), str(calibration), method="stereo_sjcf", render=True)
            body = b"".join(w[1] for w in conn.writes if w[0] == "send")
            for name in (b'input_mode', b'left_video', b'right_video', b'calibration', b'hand_refinement_method', b'render'):
                self.assertIn(b'name="' + name + b'"', body)
            self.assertNotIn(b'name="method"', body)

    def test_wait_success_and_terminal_failure(self):
        job = "876c2588-abf5-4cb5-bca7-3da815d0c3b2"
        api = ego_anno.EgoAnnoClient("key")
        with mock.patch.object(client, "get_json", side_effect=[{"status": "queued"}, {"status": "succeeded", "result_available": True}]), \
             mock.patch.object(client.time, "sleep"), mock.patch("sys.stdout", new_callable=io.StringIO):
            api.wait(job, 1)
        with mock.patch.object(client, "get_json", return_value={"status": "failed", "failure_code": "failure"}), \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            with self.assertRaisesRegex(ego_anno.ClientError, "status=failed failure_code=failure"):
                api.wait(job, 1)

    def test_download_atomic_replace_and_cleanup_on_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            path.write_text("old")
            bad = Connection(Response(200, b"partial", "12"))
            good = Connection(Response(200, b"new content", "11"))
            real_replace = os.replace

            def replace(source, target):
                self.assertEqual(path.read_text(), "old")
                self.assertEqual(Path(source).read_bytes(), b"new content")
                real_replace(source, target)

            with mock.patch.object(client, "connect", side_effect=[bad, good]), \
                 mock.patch.object(transport.time, "sleep"), mock.patch.object(client.os, "replace", side_effect=replace), \
                 mock.patch("sys.stderr", new_callable=io.StringIO), mock.patch("sys.stdout", new_callable=io.StringIO):
                ego_anno.EgoAnnoClient("key").download("876c2588-abf5-4cb5-bca7-3da815d0c3b2", str(path))
            self.assertEqual(path.read_bytes(), b"new content")
            self.assertEqual(list(Path(directory).iterdir()), [path])


class EntryPointTests(unittest.TestCase):
    def test_python_m_ego_anno_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "ego_anno", "--help"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: ego-anno", result.stdout)
        self.assertNotIn("_ping", result.stdout)

    def test_console_entry_point_exists(self):
        executable = Path(sys.executable).with_name("ego-anno")
        self.assertTrue(executable.is_file(), "ego-anno is not installed beside " + sys.executable)
        result = subprocess.run(
            [str(executable), "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: ego-anno", result.stdout)


if __name__ == "__main__":
    unittest.main()
