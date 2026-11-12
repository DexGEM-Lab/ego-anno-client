"""Offline rendering of annotation results; optional numerical dependencies load on demand.

Two anchored entry points publish the same browser-playable MP4:

``render_result``
    a saved e2e bundle (``--save-inputs``: video, submission.json, optional
    camera.json) with the processed-video identity proven from the bundle.
``render_video_result``
    any supported result JSON plus the exact video that job consumed, so a
    result downloaded from the web UI or the API can be rendered locally.

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

__all__ = ["render_result", "render_video_result"]

DAG_SCHEMA = "ego.annotation.product_json.v3"
COSMOS_SCHEMA = "cosmos-e2e-result/v1"
COSMOS_HEAD_SCHEMA = "cosmos-e2e-result/v2"
COSMOS_SCHEMAS = (COSMOS_SCHEMA, COSMOS_HEAD_SCHEMA)
SUPPORTED_SCHEMAS = COSMOS_SCHEMAS + (DAG_SCHEMA,)

# The DAG writes one 手0/手1 entry per frame; MANO shape parameters live here.
HAND_KEYS = ("手0", "手1")
SHAPE_KEY = "beta参数"


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


def _check_output_dimensions(width, height, fallback):
    """Return the even (width, height) pair to encode; never round silently."""
    for name, value in (("width", width), ("height", height)):
        if value is not None and (not _positive_integer(value) or value % 2):
            raise ClientError("render {} must be a positive even integer".format(name))
    output_size = [width if width is not None else fallback[0],
                   height if height is not None else fallback[1]]
    if output_size[0] < 4 or output_size[1] < 168 or any(value % 2 for value in output_size):
        raise ClientError("render dimensions must be even, at least 4 pixels wide and 168 pixels high")
    return output_size


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


def _encode_renderer_output(result_path, video_path, camera_path, destination, width, height, extra):
    """Run the vendored renderer, re-encode its MP4 to H.264, then publish atomically."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ClientError("offline rendering requires system ffmpeg with the libx264 encoder on PATH")
    environment = _subprocess_environment()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem as destination: only the final replace can publish.
    with tempfile.TemporaryDirectory(prefix=".ego-anno-render-", dir=str(destination.parent)) as temporary:
        temporary = Path(temporary)
        rendered = temporary / "rendered.mp4"
        encoded = temporary / "encoded.mp4"
        command = [sys.executable, str(Path(__file__).with_name("render_ego_json_overlay.py")),
                   str(result_path), "--video", str(video_path), "--output", str(rendered),
                   "--width", str(width), "--height", str(height)]
        if camera_path is not None:
            command.extend(["--camera", str(camera_path)])
        command.extend(extra)
        _run(command, environment, "annotation renderer")
        _run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
              "-i", str(rendered), "-map", "0:v:0", "-an", "-c:v", "libx264",
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
             environment, "ffmpeg H.264 encoding")
        if not encoded.is_file() or encoded.stat().st_size == 0:
            raise ClientError("ffmpeg did not produce an MP4; output was not replaced")
        os.replace(str(encoded), str(destination))


def render_result(result, inputs, output, width=None, height=None, mano_dir=None):
    """Render a saved e2e bundle and atomically publish a browser-playable MP4.

    ``inputs`` is the directory written by ``submit --save-inputs``. Its video
    must be the exact uploaded bytes; no frame resampling or calibration
    estimation occurs. With ``mano_dir`` (the licensed hand model, not
    distributed with this client) one hand shape per side is fitted to the
    clip's predicted bone lengths and MANO surfaces are drawn on the predicted
    joints; without it both panes draw the predicted 21-joint skeletons.
    Returns the destination ``Path``. No API key or network connection is
    used, and an existing destination survives every failure.
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
        extra = [] if mano_dir is None else ["--mano-dir", str(_check_mano_dir(mano_dir))]
        document = _read_object(result_path, "result")
        if document.get("schema") not in COSMOS_SCHEMAS:
            raise ClientError("offline rendering of a saved bundle only accepts {}; ".format(
                " or ".join(COSMOS_SCHEMAS)) + "render a DAG result with --video FILE")
        submission = _read_object(submission_path, "submission metadata")
        camera = _read_object(camera_path, "calibration") if camera_path.exists() else None
        expected_size = _validate_submission(submission, document, camera)
        if not video_path.is_file():
            raise ClientError("saved processed input video does not exist: {}".format(video_path))
        output_size = _check_output_dimensions(width, height, expected_size)
        if shutil.which("ffmpeg") is None:
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
        _encode_renderer_output(result_path, video_path,
                                camera_path if camera is not None else None,
                                destination, output_size[0], output_size[1], extra)
        return destination
    except ClientError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise ClientError("cannot render offline: {}".format(exc)) from None


def _result_schema(document):
    """Return the result schema; the DAG carries it under ``Metadata.schema``."""
    schema = document.get("schema")
    if isinstance(schema, str) and schema:
        return schema
    metadata = document.get("Metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("schema"), str):
        return metadata["schema"]
    return None


def _without_mano_shapes(document):
    """Copy a DAG result with the MANO shape parameters removed.

    The public client does not distribute the licensed MANO model, so a render
    without assets must draw the measured 21-joint skeletons the result does
    carry instead of fitted surfaces. Only ``手0``/``手1`` ``beta参数`` is
    cleared; every other field reaches the renderer unchanged, and the input
    JSON on disk is never modified.
    """
    prepared = dict(document)
    frames = document.get("逐帧轨迹")
    if not isinstance(frames, list):
        return prepared
    prepared_frames = []
    for entry in frames:
        if not isinstance(entry, dict) or len(entry) != 1:
            prepared_frames.append(entry)
            continue
        key = next(iter(entry))
        frame = entry[key]
        if isinstance(frame, dict):
            frame = dict(frame)
            for hand_key in HAND_KEYS:
                hand = frame.get(hand_key)
                if isinstance(hand, dict) and hand.get(SHAPE_KEY) is not None:
                    hand = dict(hand)
                    hand[SHAPE_KEY] = None
                    frame[hand_key] = hand
        prepared_frames.append({key: frame})
    prepared["逐帧轨迹"] = prepared_frames
    return prepared


def _write_document(path, document):
    with open(str(path), "w", encoding="utf-8") as out:
        json.dump(document, out, ensure_ascii=False, allow_nan=False)


def _check_mano_dir(mano_dir):
    root = Path(mano_dir).expanduser().resolve()
    if not root.is_dir():
        raise ClientError("hand-model asset directory does not exist: {}".format(root))
    standard = (root / "HAND_LEFT.pkl", root / "HAND_RIGHT.pkl")
    legacy = (root / "MANO_LEFT.pkl", root / "MANO_RIGHT.pkl", root / "mano_runtime.npz")
    if all(path.is_file() for path in standard) or all(path.is_file() for path in legacy):
        return root
    if not standard[0].is_file() and not legacy[0].is_file():
        raise ClientError("hand-model asset directory is missing MANO_LEFT.pkl or HAND_LEFT.pkl: {}".format(root))
    missing = next(path.name for path in (*standard, *legacy) if not path.is_file())
    raise ClientError("hand-model asset directory is missing {}: {}".format(missing, root))


def _validate_dag_document(document):
    """Validate the DAG result itself and return its frame count."""
    metadata = document.get("Metadata")
    if not isinstance(metadata, dict):
        raise ClientError("DAG result has no Metadata object")
    if metadata.get("schema") != DAG_SCHEMA:
        raise ClientError("DAG result Metadata.schema must be {}".format(DAG_SCHEMA))
    count = metadata.get("帧数")
    if not _positive_integer(count):
        raise ClientError("DAG result Metadata.帧数 must be a positive integer")
    frames = document.get("逐帧轨迹")
    if not isinstance(frames, list) or len(frames) != count:
        raise ClientError("DAG result 逐帧轨迹 does not hold Metadata.帧数 entries")
    for index in range(count):
        entry = frames[index]
        if not isinstance(entry, dict) or "帧{}".format(index + 1) not in entry:
            raise ClientError("DAG result entry {} is not keyed 帧{}".format(index + 1, index + 1))
    return count


def _validate_dag_video(cv2, capture, count, requested_fps):
    """Prove the video is the job's exact input, then return the render fps.

    The DAG decodes at native resolution and never truncates or resamples, so
    the only video the result's pixel projections describe is the exact
    submitted file: equal frame count, equal frame rate, same grid. Anything
    else fails here instead of producing a quietly misaligned overlay.
    """
    if not capture.isOpened():
        raise ClientError("cannot decode the source video")
    video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_count != count:
        raise ClientError("source video has {} frames but the result has {}; render the exact "
                          "submitted video (the DAG neither resamples nor trims frames)".format(
                              video_count, count))
    probed = float(capture.get(cv2.CAP_PROP_FPS))
    if requested_fps is not None:
        if not _finite_positive(requested_fps):
            raise ClientError("--fps must be a finite positive number")
        if _finite_positive(probed) and not math.isclose(probed, float(requested_fps), rel_tol=0.01):
            raise ClientError("source video reports {:g} fps but --fps is {:g}".format(
                probed, float(requested_fps)))
        return float(requested_fps)
    if not _finite_positive(probed):
        raise ClientError("cannot read the source video frame rate; pass --fps")
    return probed


def _world_up_argument(world_up):
    """Renderer flag for an explicit gravity-up vector, or nothing."""
    if world_up is None:
        return []
    if isinstance(world_up, str):
        world_up = world_up.split(",")
    try:
        values = [float(value) for value in world_up]
    except (TypeError, ValueError):
        raise ClientError("--world-up must be three comma-separated numbers x,y,z") from None
    if len(values) != 3 or not all(math.isfinite(value) for value in values) or not any(values):
        raise ClientError("--world-up must be a finite non-zero x,y,z vector")
    return ["--world-up", ",".join("{!r}".format(value) for value in values)]


def render_video_result(result, video, output, width=None, height=None, camera=None, fps=None,
                        mano_dir=None, world_up=None):
    """Render a downloaded result against the exact video the job consumed.

    Supported results are the DAG ``ego.annotation.product_json.v3`` document
    and the Cosmos ``cosmos-e2e-result/v1``/``cosmos-e2e-result/v2`` documents,
    both as served by
    ``GET /api/v1/annotation-jobs/{id}/result``. ``video`` must be the file the
    job was submitted with; nothing is resampled or calibrated implicitly.

    MANO surfaces need the licensed hand model, which this client does not
    distribute: without ``mano_dir`` the render draws the measured 21-joint
    skeletons the result carries (and, for a DAG result, prints that
    limitation). A Cosmos result carries no shape, so with ``mano_dir`` one
    shape per side is fitted to the clip's median predicted bone lengths and
    the surfaces are posed on the predicted joints. ``world_up`` is an
    optional gravity-up direction ``(x, y, z)`` in the result's world frame
    (the first camera frame); when omitted the 3D pane estimates it from the
    camera trajectory. Returns the destination ``Path``; no API key or network
    connection is used, and an existing destination survives every failure.
    """
    temporary = None
    try:
        result_path = Path(result).expanduser().resolve()
        video_path = Path(video).expanduser().resolve()
        destination = Path(output).expanduser().resolve()
        camera_path = Path(camera).expanduser().resolve() if camera is not None else None
        world_up_flag = _world_up_argument(world_up)
        protected = [result_path, video_path]
        if camera_path is not None:
            protected.append(camera_path)
        _refuse_collision(destination, protected)
        document = _read_object(result_path, "result")
        schema = _result_schema(document)
        if schema not in SUPPORTED_SCHEMAS:
            raise ClientError("offline rendering accepts {}; got {!r}".format(
                " or ".join(SUPPORTED_SCHEMAS), schema))
        if not video_path.is_file():
            raise ClientError("source video does not exist: {}".format(video_path))
        if camera_path is not None and not camera_path.is_file():
            raise ClientError("calibration does not exist: {}".format(camera_path))
        assets = _check_mano_dir(mano_dir) if mano_dir is not None else None
        count = None
        if schema == DAG_SCHEMA:
            if camera_path is not None:
                raise ClientError("--camera applies to Cosmos e2e results; DAG results "
                                  "carry their own 相机内参")
            count = _validate_dag_document(document)
        cv2 = _load_video_dependency()
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ClientError("cannot decode the source video: {}".format(video_path))
            source_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]

            extra = []
            if schema == DAG_SCHEMA:
                extra = ["--fps", "{:g}".format(_validate_dag_video(cv2, capture, count, fps))]
            elif fps is not None:
                raise ClientError("--fps is fixed by the cosmos-e2e-result metadata; omit it")
            output_size = _check_output_dimensions(width, height, source_size)
        finally:
            capture.release()
        if assets is not None:
            extra.extend(["--mano-dir", str(assets)])
        extra.extend(world_up_flag)
        prepared = result_path
        if schema == DAG_SCHEMA and assets is None:
            print("notice: the licensed MANO hand model is not distributed with this client; "
                  "rendering the measured 21-joint skeletons. Pass --mano-dir to render surfaces.",
                  flush=True)
            temporary = tempfile.TemporaryDirectory(prefix="ego-anno-render-input-")
            prepared = Path(temporary.name) / "annotation.json"
            _write_document(prepared, _without_mano_shapes(document))
        _encode_renderer_output(prepared, video_path, camera_path if schema in COSMOS_SCHEMAS else None,
                                destination, output_size[0], output_size[1], extra)
        return destination
    except ClientError:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise ClientError("cannot render offline: {}".format(exc)) from None
    finally:
        if temporary is not None:
            temporary.cleanup()
