"""Python client for submitting and retrieving ego-anno annotations."""
import http.client
import json
import mimetypes
import os
import tempfile
import time
import uuid

from .transport import CHUNK_SIZE, ClientError, Multipart, check_response, connect, headers, with_retries


class EgoAnnoClient:
    """API-key-bound client; read the key from EGO_ANNO_API_KEY when omitted."""

    def __init__(self, api_key=None):
        self.api_key = api_key if api_key is not None else os.environ.get("EGO_ANNO_API_KEY", "")
        if not self.api_key:
            raise ClientError("API key is required")
        if any(char in self.api_key for char in "\r\n"):
            raise ClientError("API key contains an invalid character")

    def submit_mono(self, video, render=False, idempotency_key=None, wait=False):
        """Submit one monocular video and return its annotation job ID."""
        files = [("video", video, mimetypes.guess_type(video)[0] or "video/mp4")]
        return self._submit("mono", files, render, None, idempotency_key, wait)

    def submit_stereo(self, left, right, calibration, method=None, render=False,
                      idempotency_key=None, wait=False):
        """Submit synchronized stereo videos and calibration; return the job ID."""
        files = [
            ("left_video", left, mimetypes.guess_type(left)[0] or "video/mp4"),
            ("right_video", right, mimetypes.guess_type(right)[0] or "video/mp4"),
            ("calibration", calibration, "application/json"),
        ]
        return self._submit("stereo", files, render, method, idempotency_key, wait)

    def _submit(self, input_mode, files, render, method, idempotency_key, wait_for_result):
        idem = str(uuid.UUID(idempotency_key)) if idempotency_key else str(uuid.uuid4())
        fields = [("input_mode", input_mode)]
        if render:
            fields.append(("render", "true"))
        if method:
            fields.append(("hand_refinement_method", method))
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

    def download(self, job_id, destination="annotation.json", rendered=None):
        """Atomically download a result and, when requested, its rendered video."""
        path = job_path(job_id)
        download(path + "/result", destination, self.api_key)
        if rendered:
            download(path + "/rendered", rendered, self.api_key)


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
