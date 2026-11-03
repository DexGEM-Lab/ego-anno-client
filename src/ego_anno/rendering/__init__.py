"""Offline Cosmos rendering; optional numerical dependencies are loaded on demand.

The rendering algorithms are unmodified snapshots described by provenance.json.
Only this wrapper handles the public input bundle, subprocesses, and publication.
"""
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from ..transport import ClientError

__all__ = ["render_result"]


def _read_object(path, label):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClientError("cannot read {} {}: {}".format(label, path, exc)) from None
    if not isinstance(value, dict):
        raise ClientError("{} must be a JSON object: {}".format(label, path))
    return value


def _positive_integer(value):
    return type(value) is int and value > 0


def _finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _validate_submission(submission, result, camera):
    if submission.get("schema") != "ego-anno-e2e-input/v1" or submission.get("pipeline") != "e2e":
        raise ClientError("submission.json must use schema ego-anno-e2e-input/v1 and pipeline e2e")
    count = submission.get("num_frames")
    if not _positive_integer(count) or count < 2:
        raise ClientError("submission.json num_frames must be an integer of at least 2")
    if type(result.get("num_frames")) is not int or count != result["num_frames"]:
        raise ClientError("submission.json frame count disagrees with result num_frames")
    fps = submission.get("fps")
    if not _finite_positive(fps) or not math.isclose(fps, 15.0, rel_tol=0, abs_tol=1e-6):
        raise ClientError("submission.json fps must be 15 (the exact processed input rate)")
    if not _finite_positive(result.get("fps")) or not math.isclose(fps, result["fps"], rel_tol=0, abs_tol=1e-6):
        raise ClientError("submission.json fps disagrees with result fps")
    size = [submission.get("width"), submission.get("height")]
    if not all(_positive_integer(value) for value in size):
        raise ClientError("submission.json width and height must be positive integers")
    # Cosmos usually omits source resolution. Compare it only when recorded;
    # neither output layout dimensions nor an inferred intrinsic matrix is K.
    if result.get("source_size_wh") is not None and result["source_size_wh"] != size:
        raise ClientError("submission.json processed resolution disagrees with result source_size_wh")
    if camera is not None and camera.get("source_size_wh") != size:
        raise ClientError("camera.json dimensions disagree with submission.json processed resolution")
    return size


def _refuse_collision(destination, protected):
    for path in protected:
        if destination == path or (destination.exists() and path.exists() and destination.samefile(path)):
            raise ClientError("output must not overwrite result or saved input: {}".format(path))


def _load_video_dependency():
    try:
        importlib.import_module("numpy")
        return importlib.import_module("cv2")
    except (ImportError, OSError) as exc:
        raise ClientError("offline rendering requires the render extra; install this client "
                          "with [render] using the public README instructions ({})".format(exc)) from None


def _subprocess_environment():
    # Render is entirely offline; do not expose client credentials to either
    # child, including when a caller selected an overridden API endpoint.
    return {key: value for key, value in os.environ.items()
            if not key.upper().startswith("EGO_ANNO_") and "API_KEY" not in key.upper()}


def _run(command, environment, label):
    try:
        completed = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
    except OSError as exc:
        raise ClientError("cannot start {}: {}".format(label, exc)) from None
    if completed.returncode:
        lines = completed.stderr.strip().splitlines()
        detail = lines[-1] if lines else "exit code {}".format(completed.returncode)
        # In particular, malformed physical arrays must not surface a Python
        # traceback through the public CLI's ordinary ClientError handler.
        raise ClientError("{} failed: {}".format(label, detail))


def render_result(result, inputs, output, width=None, height=None):
    """Render a saved e2e bundle and atomically publish a browser-playable MP4.

    ``inputs`` is the directory written by ``submit --save-inputs``. Its video
    must be the exact uploaded bytes; no frame resampling or calibration/shape
    estimation occurs. Returns the destination ``Path``. No API key or network
    connection is used, and an existing destination survives every failure.
    """
    try:
        result_path = Path(result).expanduser().resolve()
        inputs_path = Path(inputs).expanduser().resolve()
        destination = Path(output).expanduser().resolve()
        video_path = inputs_path / "video.mp4"
        camera_path = inputs_path / "camera.json"
        submission_path = inputs_path / "submission.json"
        protected = [result_path, video_path.resolve(), camera_path.resolve(), submission_path.resolve()]
        _refuse_collision(destination, protected)
        document = _read_object(result_path, "result")
        if document.get("schema") != "cosmos-e2e-result/v1":
            raise ClientError("offline rendering only accepts cosmos-e2e-result/v1 (DAG rendering is not supported)")
        submission = _read_object(submission_path, "submission metadata")
        camera = _read_object(camera_path, "calibration") if camera_path.exists() else None
        expected_size = _validate_submission(submission, document, camera)
        if not video_path.is_file():
            raise ClientError("saved processed input video does not exist: {}".format(video_path))
        for name, value in (("width", width), ("height", height)):
            if value is not None and (not _positive_integer(value) or value % 2):
                raise ClientError("render {} must be a positive even integer".format(name))
        output_size = [width if width is not None else expected_size[0],
                       height if height is not None else expected_size[1]]
        if output_size[0] < 4 or output_size[1] < 168 or any(value % 2 for value in output_size):
            raise ClientError("render dimensions must be even, at least 4 pixels wide and 168 pixels high")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ClientError("offline rendering requires system ffmpeg with the libx264 encoder on PATH")
        cv2 = _load_video_dependency()
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ClientError("cannot open saved processed input video: {}".format(video_path))
            actual_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            if actual_size != expected_size:
                raise ClientError("processed video dimensions {} != submission.json {}; no implicit resizing".format(
                    actual_size, expected_size))
        finally:
            capture.release()
        environment = _subprocess_environment()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Same filesystem as destination: only the final replace can publish.
        with tempfile.TemporaryDirectory(prefix=".ego-anno-render-", dir=str(destination.parent)) as temporary:
            temporary = Path(temporary)
            rendered = temporary / "rendered.mp4"
            encoded = temporary / "encoded.mp4"
            command = [sys.executable, str(Path(__file__).with_name("render_ego_json_overlay.py")),
                       str(result_path), "--video", str(video_path), "--output", str(rendered),
                       "--width", str(output_size[0]), "--height", str(output_size[1])]
            if camera is not None:
                command.extend(["--camera", str(camera_path)])
            _run(command, environment, "Cosmos renderer")
            _run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                  "-i", str(rendered), "-map", "0:v:0", "-an", "-c:v", "libx264",
                  "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
                 environment, "ffmpeg H.264 encoding")
            if not encoded.is_file() or encoded.stat().st_size == 0:
                raise ClientError("ffmpeg did not produce an MP4; output was not replaced")
            os.replace(str(encoded), str(destination))
        return destination
    except ClientError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise ClientError("cannot render offline: {}".format(exc)) from None
