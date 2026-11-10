"""Local preprocessing for ``pipeline=e2e`` (Cosmos3 hand147) submissions.

The service accepts only constant 15 fps H.264 clips of 17..241 frames with
(N - 1) % 4 == 0 and a short side of 640..1080 pixels. This module turns a
15..30 fps source of at most 20 s into such a clip with the system ``ffprobe``
and ``ffmpeg`` binaries (standard library subprocesses only):

1. Probe the source: displayed size (rotation metadata applied), average frame
   rate, duration, and every decoded frame's ``best_effort_timestamp_time``.
2. Reject > 30 fps, < 15 fps (both within 1e-3), > 20 s, and short sides
   outside 640..1440.
3. Resample to 15 fps by timestamp: output slot k (time t0 + k/15, kept while
   it is before the source end, last timestamp + 1/fps) shows the source frame
   whose timestamp is nearest (the earlier one on a tie). Keep the first
   min(241, slots) slots, trim to the largest N with (N - 1) % 4 == 0, and
   reject N < 17.
4. Decode only the chosen source frames with the ffmpeg video filter
   ``select=<balanced sum of not(n-index) terms>,<geometry>,format=yuv420p``
   (``-fps_mode passthrough`` or legacy ``-vsync passthrough``, raw yuv420p to a pipe; ``n`` is the decode-order
   frame index that ``ffprobe`` reported). ``<geometry>`` is
   ``scale=W:H:flags=lanczos`` when the short side exceeds 1080 (short side
   becomes 1080, long side the nearest even value keeping the aspect ratio),
   ``crop=W:H:0:0`` when a kept side is odd (drops the last column/row so
   yuv420p dimensions are even; intrinsics are unchanged), else nothing.
   Each decoded frame is written to the encoder as many times as slots use it.
5. Encode the raw frames at exactly 15 fps: libx264 CRF 12, yuv420p, no audio,
   ``+faststart``, bit-exact flags, then re-probe the output and require
   exactly N frames, 15/1 fps, the expected size, and ``moov`` before ``mdat``.
"""
import bisect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from collections import namedtuple
from fractions import Fraction

from .transport import ClientError, filename_header

TARGET_FPS = 15
MIN_FPS = 15
MAX_FPS = 30
FPS_TOLERANCE = 1e-3
MAX_DURATION_SECONDS = 20.0
MIN_FRAMES = 17
MAX_FRAMES = 241
MIN_SHORT_SIDE = 640
MAX_KEPT_SHORT_SIDE = 1080
MAX_SHORT_SIDE = 1440
MAX_LONG_SIDE = 2560
CRF = 12
MAX_PROMPT_CHARS = 1000
# ffprobe prints timestamps rounded to microseconds; time differences below
# this many seconds are treated as equal (ties, slot exactly at the end).
TIMESTAMP_EPSILON = 1e-5

Source = namedtuple("Source", "width height fps duration")
Geometry = namedtuple("Geometry", "width height scaled")
Prepared = namedtuple("Prepared", "path num_frames width height camera")


def check_prompt(prompt):
    """Return the optional task description or raise ClientError."""
    if prompt is None:
        return None
    if (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS
            or "\x00" in prompt):
        raise ClientError("prompt must be non-blank text of at most {} characters".format(MAX_PROMPT_CHARS))
    return prompt


def check_source(fps, duration, width, height):
    """Apply the client admission rules to the probed source (displayed size)."""
    if not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
        raise ClientError("cannot determine the video frame rate")
    if fps > MAX_FPS + FPS_TOLERANCE or fps < MIN_FPS - FPS_TOLERANCE:
        raise ClientError("e2e accepts 15 to 30 fps; the video is {:.3f} fps".format(fps))
    if not (isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0):
        raise ClientError("cannot determine the video duration")
    if duration > MAX_DURATION_SECONDS:
        raise ClientError("e2e accepts videos of at most 20 s; the video is {:.3f} s".format(duration))
    if not MIN_SHORT_SIDE <= min(width, height) <= MAX_SHORT_SIDE:
        raise ClientError("e2e requires a short side of 640 to 1440 pixels; the video is {}x{}".format(
            width, height))


def output_geometry(width, height):
    """Return the processed size for a displayed source size."""
    short, long_ = min(width, height), max(width, height)
    if short > MAX_KEPT_SHORT_SIDE:
        # Even long side nearest to long * 1080 / short (exact rational, halves round up).
        new_long = 2 * math.floor(Fraction(long_ * MAX_KEPT_SHORT_SIDE, 2 * short) + Fraction(1, 2))
        size = (MAX_KEPT_SHORT_SIDE, new_long) if width == short else (new_long, MAX_KEPT_SHORT_SIDE)
        geometry = Geometry(size[0], size[1], True)
    else:
        geometry = Geometry(width - width % 2, height - height % 2, False)
    if max(geometry.width, geometry.height) > MAX_LONG_SIDE:
        raise ClientError("e2e requires a processed long side of at most {} pixels; the video is {}x{}".format(
            MAX_LONG_SIDE, width, height))
    return geometry


def slot_count(timestamps, end):
    """Number of 15 fps slots t0 + k/15 strictly before ``end`` (seconds)."""
    return max(0, math.ceil((end - TIMESTAMP_EPSILON - timestamps[0]) * TARGET_FPS))


def frame_count(available):
    """First min(241, available) slots trimmed to (N - 1) % 4 == 0; N >= 17."""
    count = min(MAX_FRAMES, available)
    if count >= 1:
        count -= (count - 1) % 4
    if count < MIN_FRAMES:
        raise ClientError("e2e needs at least {} frames at 15 fps (about 1.07 s); the video yields {}".format(
            MIN_FRAMES, available))
    return count


def nearest_indices(timestamps, count):
    """Source frame index nearest to each slot t0 + k/15 (earlier frame on a tie)."""
    t0 = timestamps[0]
    last = len(timestamps) - 1
    indices = []
    for k in range(count):
        target = t0 + k / TARGET_FPS
        j = min(bisect.bisect_left(timestamps, target), last)
        if j > 0 and target - timestamps[j - 1] <= timestamps[j] - target + TIMESTAMP_EPSILON:
            j -= 1
        indices.append(j)
    return indices


def load_camera(camera):
    """Validate a camera mapping or JSON file path; return ([w, h], K) or None."""
    if camera is None:
        return None
    if isinstance(camera, (str, bytes, os.PathLike)):
        try:
            with open(camera, "rb") as handle:
                payload = json.loads(handle.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise ClientError("cannot read camera JSON {}: {}".format(camera, exc))
    else:
        payload = camera
    if not isinstance(payload, dict) or set(payload) != {"source_size_wh", "source_K"}:
        raise ClientError("camera must contain exactly source_size_wh and source_K")
    size = payload["source_size_wh"]
    if (not isinstance(size, list) or len(size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size)):
        raise ClientError("camera.source_size_wh must be [width, height] positive integers")
    matrix = payload["source_K"]
    if (not isinstance(matrix, list) or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)):
        raise ClientError("camera.source_K must be a 3x3 matrix")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for row in matrix for v in row):
        raise ClientError("camera.source_K entries must be finite numbers")
    k = [[float(v) for v in row] for row in matrix]
    if k[0][1] != 0 or k[1][0] != 0 or k[2] != [0.0, 0.0, 1.0] or k[0][0] <= 0 or k[1][1] <= 0:
        raise ClientError("camera.source_K must be [[fx,0,cx],[0,fy,cy],[0,0,1]] with fx, fy > 0")
    return list(size), k


def rescale_camera(camera, source_width, source_height, geometry):
    """Return the camera field for the processed video (pixel-center convention)."""
    size, k = camera
    if size != [source_width, source_height]:
        raise ClientError("camera source_size_wh {}x{} does not match the video's displayed size {}x{}".format(
            size[0], size[1], source_width, source_height))
    fx, cx, fy, cy = k[0][0], k[0][2], k[1][1], k[1][2]
    if geometry.scaled:
        sx = geometry.width / source_width
        sy = geometry.height / source_height
        fx, cx = fx * sx, (cx + 0.5) * sx - 0.5
        fy, cy = fy * sy, (cy + 0.5) * sy - 0.5
    # A right/bottom crop keeps the origin, so K is unchanged.
    return {"source_size_wh": [geometry.width, geometry.height],
            "source_K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]}


def video_filter(unique_indices, source_width, source_height, geometry):
    terms = ["not(n-{})".format(i) for i in unique_indices]
    if not terms:
        raise ClientError("frame selection is empty")
    # AVExpr recursively parses sums: a flat 241-term expression overflows
    # newer FFmpeg's parser. A balanced tree keeps exactly the same membership.
    while len(terms) > 1:
        terms = ["({}+{})".format(terms[i], terms[i + 1]) if i + 1 < len(terms)
                 else terms[i] for i in range(0, len(terms), 2)]
    selected = terms[0]
    steps = ["select=" + selected]
    if geometry.scaled:
        steps.append("scale={}:{}:flags=lanczos".format(geometry.width, geometry.height))
    elif (geometry.width, geometry.height) != (source_width, source_height):
        steps.append("crop={}:{}:0:0".format(geometry.width, geometry.height))
    steps.append("format=yuv420p")
    return ",".join(steps)


def require_tools():
    found = {name: shutil.which(name) for name in ("ffprobe", "ffmpeg")}
    missing = [name for name, path in found.items() if not path]
    if missing:
        raise ClientError("--pipeline e2e needs FFmpeg; {} not found on PATH. Install FFmpeg "
                          "(ffmpeg and ffprobe) and retry".format(" and ".join(missing)))
    return found["ffprobe"], found["ffmpeg"]


def _input(path):
    # The file: protocol keeps names beginning with '-' or containing ':' literal.
    return "file:" + os.path.abspath(path)


def _tail(data):
    text = data.decode("utf-8", "replace").strip() if isinstance(data, bytes) else str(data)
    return text[-2000:] or "no diagnostic output"


def _probe_json(ffprobe, args):
    try:
        result = subprocess.run([ffprobe, "-v", "error"] + args + ["-of", "json"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise ClientError("cannot run ffprobe: {}".format(exc))
    if result.returncode != 0:
        raise ClientError("ffprobe failed: {}".format(_tail(result.stderr)))
    try:
        return json.loads(result.stdout.decode("utf-8"))
    except ValueError:
        raise ClientError("ffprobe returned unparseable output")


def _rate(value):
    try:
        rate = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return float(rate) if rate > 0 else None


def _seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _rotation(stream):
    for item in stream.get("side_data_list") or []:
        if "rotation" in item:
            return float(item["rotation"])
    return float((stream.get("tags") or {}).get("rotate", 0) or 0)


def probe_source(ffprobe, video):
    """Return the source's displayed size, average fps and duration."""
    data = _probe_json(ffprobe, ["-select_streams", "v:0", "-show_streams", "-show_format", _input(video)])
    streams = data.get("streams") or []
    if not streams:
        raise ClientError("no video stream found in {}".format(video))
    stream = streams[0]
    try:
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, TypeError, ValueError):
        raise ClientError("cannot determine the video size")
    rotation = _rotation(stream)
    if rotation % 90:
        raise ClientError("unsupported video rotation {} degrees".format(rotation))
    if rotation % 180:
        width, height = height, width
    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    duration = _seconds(stream.get("duration")) or _seconds((data.get("format") or {}).get("duration"))
    return Source(width, height, fps, duration)


def probe_timestamps(ffprobe, video):
    """Decode the video stream and return its frame timestamps in decode output order."""
    data = _probe_json(ffprobe, ["-select_streams", "v:0", "-show_entries",
                                 "frame=best_effort_timestamp_time", _input(video)])
    try:
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in data.get("frames") or []]
    except (KeyError, TypeError, ValueError):
        raise ClientError("the video has frames without timestamps")
    if not timestamps:
        raise ClientError("no decodable video frames found in {}".format(video))
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ClientError("the video has non-increasing frame timestamps")
    return timestamps


def _wait(process, log, name):
    if process.wait() != 0:
        log.seek(0)
        raise ClientError("{} failed: {}".format(name, _tail(log.read())))


def passthrough_args(ffmpeg):
    """Inspect capabilities instead of guessing a vendor build's version."""
    try:
        result = subprocess.run([ffmpeg, "-hide_banner", "-h", "full"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise ClientError("cannot inspect ffmpeg: {}".format(exc))
    if result.returncode:
        raise ClientError("ffmpeg capability check failed: {}".format(_tail(result.stderr)))
    options = result.stdout.decode("utf-8", "replace")
    if any(line.startswith(("-fps_mode ", "-fps_mode[")) for line in options.splitlines()):
        return ["-fps_mode", "passthrough"]
    if any(line.startswith("-vsync ") for line in options.splitlines()):
        return ["-vsync", "passthrough"]
    raise ClientError("ffmpeg has neither fps_mode nor vsync passthrough support")


def transcode(ffmpeg, video, output, indices, source_width, source_height, geometry):
    """Write the slot frames ``indices`` (source frame numbers) as a 15 fps H.264 MP4."""
    unique = sorted(set(indices))
    repeats = [indices.count(i) for i in unique]
    frame_bytes = geometry.width * geometry.height * 3 // 2
    decode = [ffmpeg, "-nostdin", "-v", "error", "-i", _input(video), "-map", "0:v:0", "-an", "-sn", "-dn",
              "-vf", video_filter(unique, source_width, source_height, geometry)
              ] + passthrough_args(ffmpeg) + ["-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    encode = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
              "-s", "{}x{}".format(geometry.width, geometry.height), "-framerate", str(TARGET_FPS),
              "-i", "pipe:0", "-map", "0:v:0", "-an", "-c:v", "libx264", "-crf", str(CRF),
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-map_metadata", "-1",
              "-fflags", "+bitexact", "-flags:v", "+bitexact", "-y", _input(output)]
    with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
        processes = []
        try:
            decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=decode_log)
            processes.append(decoder)
            encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=encode_log)
            processes.append(encoder)
            try:
                for repeat in repeats:
                    frame = decoder.stdout.read(frame_bytes)
                    if len(frame) != frame_bytes:
                        _wait(decoder, decode_log, "ffmpeg decode")
                        raise ClientError("ffmpeg decoded fewer frames than ffprobe reported")
                    for _ in range(repeat):
                        encoder.stdin.write(frame)
                if decoder.stdout.read(1):
                    raise ClientError("ffmpeg decoded more frames than ffprobe reported")
                encoder.stdin.close()
            except BrokenPipeError:
                _wait(encoder, encode_log, "ffmpeg encode")
                raise ClientError("ffmpeg encode stopped early")
            _wait(decoder, decode_log, "ffmpeg decode")
            _wait(encoder, encode_log, "ffmpeg encode")
        except OSError as exc:
            raise ClientError("cannot run ffmpeg: {}".format(exc))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass


def _top_level_boxes(path):
    boxes = []
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        offset = 0
        while offset + 8 <= size:
            handle.seek(offset)
            header = handle.read(16)
            length = int.from_bytes(header[:4], "big")
            if length == 1 and len(header) == 16:
                length = int.from_bytes(header[8:16], "big")
            elif length == 0:
                length = size - offset
            boxes.append(header[4:8])
            if length < 8:
                break
            offset += length
    return boxes


def verify_output(ffprobe, output, num_frames, geometry):
    data = _probe_json(ffprobe, ["-count_frames", "-show_streams", _input(output)])
    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    found = {
        "streams": len(streams),
        "codec": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "size": (stream.get("width"), stream.get("height")),
        "avg_frame_rate": _rate(stream.get("avg_frame_rate")),
        "r_frame_rate": _rate(stream.get("r_frame_rate")),
        "frames": stream.get("nb_read_frames"),
    }
    expected = {
        "streams": 1, "codec": "h264", "pix_fmt": "yuv420p", "size": (geometry.width, geometry.height),
        "avg_frame_rate": float(TARGET_FPS), "r_frame_rate": float(TARGET_FPS), "frames": str(num_frames),
    }
    if found != expected or stream.get("codec_type") != "video":
        raise ClientError("preprocessed video failed verification: expected {}, found {}".format(expected, found))
    boxes = _top_level_boxes(output)
    if b"moov" not in boxes or b"mdat" not in boxes or boxes.index(b"moov") > boxes.index(b"mdat"):
        raise ClientError("preprocessed video is not faststart (moov must precede mdat)")


def preprocess(video, workdir, camera=None):
    """Create the e2e upload in the empty directory ``workdir``.

    ``camera`` comes from :func:`load_camera`. The upload keeps the source
    file's stem with an ``.mp4`` suffix.
    """
    ffprobe, ffmpeg = require_tools()
    if not os.path.isfile(video):
        raise ClientError("video not found: {}".format(video))
    output = os.path.join(workdir, filename_header((Path(video).stem or "video") + ".mp4"))
    if os.path.exists(output):
        raise ClientError("refusing to overwrite existing file {}".format(output))
    source = probe_source(ffprobe, video)
    check_source(source.fps, source.duration, source.width, source.height)
    geometry = output_geometry(source.width, source.height)
    camera_field = None
    if camera is not None:
        camera_field = rescale_camera(camera, source.width, source.height, geometry)
    timestamps = probe_timestamps(ffprobe, video)
    end = timestamps[-1] + 1 / source.fps
    num_frames = frame_count(slot_count(timestamps, end))
    indices = nearest_indices(timestamps, num_frames)
    print("e2e preprocessing: {}x{} {:.3f} fps {:.3f} s -> {} frames at 15 fps {}x{}".format(
        source.width, source.height, source.fps, source.duration, num_frames,
        geometry.width, geometry.height), file=sys.stderr, flush=True)
    transcode(ffmpeg, video, output, indices, source.width, source.height, geometry)
    verify_output(ffprobe, output, num_frames, geometry)
    return Prepared(output, num_frames, geometry.width, geometry.height, camera_field)
