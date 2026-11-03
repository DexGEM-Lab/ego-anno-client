"""Python client for submitting and retrieving ego-anno annotations."""
import http.client
import json
import mimetypes
import os
import shutil
import tempfile
import time
import uuid

from . import e2e
from .transport import CHUNK_SIZE, ClientError, Multipart, check_response, connect, headers, with_retries


class EgoAnnoClient:
    """API-key-bound client; read the key from EGO_ANNO_API_KEY when omitted."""

    def __init__(self, api_key=None):
        self.api_key = api_key if api_key is not None else os.environ.get("EGO_ANNO_API_KEY", "")
        if not self.api_key:
            raise ClientError("API key is required")
        if any(char in self.api_key for char in "\r\n"):
            raise ClientError("API key contains an invalid character")

    def submit_mono(self, video, idempotency_key=None, wait=False):
        """Submit one monocular video to the DAG pipeline and return its annotation job ID."""
        files = [("video", video, mimetypes.guess_type(video)[0] or "video/mp4")]
        return self._submit([("input_mode", "mono")], files, idempotency_key, wait)

    def submit_e2e(self, video, prompt=None, camera=None, idempotency_key=None, wait=False,
                   save_inputs=None):
        """Preprocess a video locally and submit it to the e2e pipeline; return the job ID.

        ``camera`` is an optional path to (or mapping of)
        ``{"source_size_wh": [w, h], "source_K": 3x3}`` for the source video's
        displayed size; it is rescaled with the video. Requires ffmpeg/ffprobe.
        ``save_inputs`` preserves the exact upload and processed calibration in
        a new directory for offline rendering; no credentials are saved.
        """
        idempotency_key = parse_key(idempotency_key) or str(uuid.uuid4())
        if save_inputs is not None and os.path.lexists(save_inputs):
            raise ClientError("--save-inputs directory already exists: {}".format(save_inputs))
        prompt = e2e.check_prompt(prompt)
        source_camera = e2e.load_camera(camera)
        with tempfile.TemporaryDirectory(prefix="ego-anno-e2e-") as workdir:
            prepared = e2e.preprocess(video, workdir, source_camera)
            saved = None
            if save_inputs is not None:
                saved = _save_e2e_inputs(prepared, save_inputs, idempotency_key, prompt)
            fields = [("input_mode", "mono"), ("pipeline", "e2e")]
            if prompt is not None:
                fields.append(("prompt", prompt))
            if prepared.camera is not None:
                fields.append(("camera", json.dumps(prepared.camera, separators=(",", ":"))))
            job_id = self._submit(fields, [("video", prepared.path, "video/mp4")], idempotency_key, False)
            if saved is not None:
                saved["annotation_job_id"] = job_id
                _write_json(os.path.join(save_inputs, "submission.json"), saved)
                print("Saved rendering inputs: {}".format(os.path.abspath(save_inputs)), flush=True)
        if wait:
            self.wait(job_id, 10)
        return job_id

    def submit_stereo(self, left, right, calibration, method=None,
                      idempotency_key=None, wait=False):
        """Submit synchronized stereo videos and calibration; return the job ID."""
        files = [
            ("left_video", left, mimetypes.guess_type(left)[0] or "video/mp4"),
            ("right_video", right, mimetypes.guess_type(right)[0] or "video/mp4"),
            ("calibration", calibration, "application/json"),
        ]
        fields = [("input_mode", "stereo")]
        if method:
            fields.append(("hand_refinement_method", method))
        return self._submit(fields, files, idempotency_key, wait)

    def _submit(self, fields, files, idempotency_key, wait_for_result):
        idem = parse_key(idempotency_key) or str(uuid.uuid4())
        body = Multipart(fields, files)
        # Preserve the key even if a network failure prevents receiving the response.
        print("Idempotency-Key: {}".format(idem), flush=True)
        try:
            def attempt():
                conn = connect()
                try:
                    headers(conn, "POST", "/api/v2/annotations", self.api_key, {
                        "Content-Type": "multipart/form-data; boundary=" + body.boundary,
                        "Content-Length": body.length, "Idempotency-Key": idem})
                    try:
                        body.send(conn)
                    except (OSError, http.client.HTTPException) as upload_error:
                        # Admission may reject headers before reading a large body. If
                        # that closes the upload, preserve the server's 4xx response.
                        try:
                            response = conn.getresponse()
                        except (OSError, http.client.HTTPException):
                            raise upload_error
                        check_response(response, 202)
                        raise ClientError("connection ended before upload completed")
                    response = conn.getresponse()
                    check_response(response, 202)
                    return json.loads(response.read())
                finally:
                    conn.close()
            result = with_retries(attempt)
        finally:
            body.close()
        job_id = result["annotation_job_id"]
        print("annotation_job_id: {}".format(job_id), flush=True)
        if wait_for_result:
            self.wait(job_id, 10)
        return job_id

    def status(self, job_id):
        """Return the current job status document."""
        return get_json(job_path(job_id), self.api_key)

    def wait(self, job_id, interval=10):
        """Print status updates until a result is available or the job fails."""
        path = job_path(job_id)
        while True:
            data = get_json(path, self.api_key)
            print(json.dumps(data), flush=True)
            state = data.get("status")
            if state == "succeeded" and data.get("result_available") is True:
                return
            if state not in ("queued", "running"):
                raise ClientError("job ended with status={} failure_code={} result_available={}".format(
                    state, data.get("failure_code"), data.get("result_available")))
            time.sleep(interval)

    def download(self, job_id, destination="annotation.json"):
        """Atomically download an annotation result."""
        path = job_path(job_id)
        download(path + "/result", destination, self.api_key)


def _write_json(path, document):
    fd, temporary = tempfile.mkstemp(prefix=".ego-anno-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(document, out, ensure_ascii=False, allow_nan=False, indent=2)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_e2e_inputs(prepared, destination, idempotency_key, prompt):
    destination = os.path.abspath(destination)
    os.makedirs(destination, mode=0o700, exist_ok=False)
    metadata = {
        "schema": "ego-anno-e2e-input/v1",
        "pipeline": "e2e",
        "num_frames": prepared.num_frames,
        "fps": e2e.TARGET_FPS,
        "width": prepared.width,
        "height": prepared.height,
        "idempotency_key": idempotency_key,
        "annotation_job_id": None,
        "prompt": prompt,
    }
    try:
        shutil.copyfile(prepared.path, os.path.join(destination, "video.mp4"))
        if prepared.camera is not None:
            _write_json(os.path.join(destination, "camera.json"), prepared.camera)
        _write_json(os.path.join(destination, "submission.json"), metadata)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return metadata


def get_json(path, key):
    def attempt():
        conn = connect()
        try:
            headers(conn, "GET", path, key)
            response = conn.getresponse()
            check_response(response, 200)
            return json.loads(response.read())
        finally:
            conn.close()
    return with_retries(attempt)


def parse_key(idempotency_key):
    """Normalize an optional idempotency key; invalid UUIDs raise ValueError."""
    return str(uuid.UUID(idempotency_key)) if idempotency_key else None


def job_path(job_id):
    return "/api/v1/annotation-jobs/" + str(uuid.UUID(job_id))


def download(path, destination, key):
    def attempt():
        conn = connect()
        temp = None
        try:
            headers(conn, "GET", path, key)
            response = conn.getresponse()
            check_response(response, 200)
            expected = response.getheader("Content-Length")
            fd, temp = tempfile.mkstemp(prefix=".ego-anno-", dir=os.path.dirname(os.path.abspath(destination)))
            total = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
            if expected is not None and total != int(expected):
                raise http.client.IncompleteRead(b"", int(expected) - total)
            os.replace(temp, destination)
            temp = None
        finally:
            conn.close()
            if temp is not None:
                os.unlink(temp)
    with_retries(attempt)
    print("Downloaded {}".format(destination))
