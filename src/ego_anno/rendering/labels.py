"""AR label overlay input for time-synced offline rendering.

Two documents live here. The frozen render sidecar (what the renderer
consumes; web freezes the field set)::

    {"schema": "ego-anno-ar-labels/v1",
     "content_sha256": "<sha256 of the canonical segments list>",
     "canonical_input_id": "<canonical job/clip id>",
     "input_sha256": "<sha256 of the rendered video file bytes>",
     "timebase": {"clock": "uploaded_video_seconds", "origin": "first_frame_pts",
                  "half_open": true, "fps": 15.0},
     "target_identity_sha256": "<pinned AR runtime identity>" | null,
     "segments": [{"start_sec": 1.2, "end_sec": 3.4, "subtask": "..."}, ...]}

and the minimal binding artifact ``{"schema", "clock", "labels"}`` whose
whole-document :func:`identity` the AR binding layer records. Segment seconds
are 0-based on the timeline of the video actually being rendered (the
uploaded clip for both pipelines: the processed 15 fps upload for e2e, the
original upload for DAG), intervals are half-open [start_sec, end_sec), gaps
are legal, and overlapping intervals render as parallel lines. The renderer
never invents labels: absent input produces an explicit notice, never text
derived from other fields or models, and ``target_identity_sha256`` is null
when the AR runtime identity is not verified.
"""
import hashlib
import json
import math

SCHEMA = "ego-anno-ar-labels/v1"
CLOCK = "clip_local_seconds"
SEGMENT_KEYS = ("start_sec", "end_sec", "subtask")
TIMEBASE = {"clock": "uploaded_video_seconds", "origin": "first_frame_pts", "half_open": True}
RENDER_IDENTITY_SCHEMA = "anneal-labeled-render/v1"
RENDER_RECIPE_VERSION = 1
# Frozen with the backend: the recipe this renderer implements and the SDK
# version string the render identity pins. The render worker imports both
# from here so there is exactly one source.
RENDERER_RECIPE_ID = "egodex-render/0.6.0-joint-labels/v1"
RENDERER_VERSION = "0.6.1"


def _canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(payload):
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _segments(raw, field):
    """Normalize a raw segment list to sorted [(start, end, subtask)]."""
    if not isinstance(raw, list):
        raise ValueError("{} must be a list of segments".format(field))
    segments = []
    for index, segment in enumerate(raw):
        if not isinstance(segment, dict) or set(segment) != set(SEGMENT_KEYS):
            raise ValueError("{}[{}] must contain exactly start_sec, end_sec and subtask".format(field, index))
        start, end, text = segment["start_sec"], segment["end_sec"], segment["subtask"]
        for name, value in (("start_sec", start), ("end_sec", end)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("{}[{}].{} must be a finite number".format(field, index, name))
        if not 0 <= start < end:
            raise ValueError("{}[{}] must satisfy 0 <= start_sec < end_sec".format(field, index))
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError("{}[{}].subtask must be non-blank text".format(field, index))
        segments.append((float(start), float(end), " ".join(text.split())))
    segments.sort(key=lambda item: (item[0], item[1], item[2]))
    return segments


def _sha256_field(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("{} must be 64 lowercase hex characters".format(name))
    return value


def validate(document):
    """Validate the minimal binding document; return sorted segment triples."""
    if not isinstance(document, dict):
        raise ValueError("labels document must be a JSON object")
    if document.get("schema") != SCHEMA:
        raise ValueError("labels schema must be {}".format(SCHEMA))
    if document.get("clock") != CLOCK:
        raise ValueError("labels clock must be {} (0-based seconds of the rendered video)".format(CLOCK))
    return _segments(document.get("labels"), "labels")


def identity(document):
    """sha256 of the canonical minimal document, as the AR bindings record it."""
    segments = validate(document)
    canonical = {"schema": SCHEMA, "clock": CLOCK,
                 "labels": [dict(zip(SEGMENT_KEYS, segment)) for segment in segments]}
    return _sha256(canonical)


def content_sha256(segments):
    """sha256 of the canonical segments list (float start/end, stable order).

    This exact formula is frozen with the web store: normalize start/end to
    float, sort by (start_sec, end_sec, subtask), then hash the compact JSON
    of the segment objects. Nothing else may be hashed.
    """
    return _sha256([dict(zip(SEGMENT_KEYS, segment)) for segment in segments])


def validate_timebase(timebase):
    """Validate the frozen timebase object; return it normalized.

    The object carries exactly six fields: the three clock constants plus
    ``fps``, ``duration_sec`` and ``record_offset_sec`` (the label clock's
    rate, extent and offset on the uploaded video). Both sides of the
    render identity hash exactly this object.
    """
    if not isinstance(timebase, dict):
        raise ValueError("timebase must be an object")
    if set(timebase) != set(TIMEBASE) | {"fps", "duration_sec", "record_offset_sec"}:
        raise ValueError("timebase must contain exactly clock, origin, half_open, "
                         "fps, duration_sec and record_offset_sec")
    for key, value in TIMEBASE.items():
        if timebase.get(key) != value:
            raise ValueError("timebase.{} must be {!r}".format(key, value))
    normalized = {"clock": timebase["clock"], "origin": timebase["origin"], "half_open": True}
    for key in ("fps", "duration_sec", "record_offset_sec"):
        value = timebase.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("timebase.{} must be a finite number".format(key))
        if key == "record_offset_sec":
            if value < 0:
                raise ValueError("timebase.record_offset_sec must be >= 0")
        elif value <= 0:
            raise ValueError("timebase.{} must be positive".format(key))
        normalized[key] = float(value)
    return normalized


def validate_sidecar(document):
    """Validate the frozen label sidecar; return a dict of its checked fields.

    The declared content_sha256 must match the canonical segments payload: a
    misassembled or tampered sidecar is a hard render error, never a render
    with approximate labels.
    """
    if not isinstance(document, dict):
        raise ValueError("labels sidecar must be a JSON object")
    if document.get("schema") != SCHEMA:
        raise ValueError("labels sidecar schema must be {}".format(SCHEMA))
    segments = _segments(document.get("segments"), "segments")
    declared = _sha256_field(document.get("content_sha256"), "content_sha256")
    if content_sha256(segments) != declared:
        raise ValueError("labels sidecar content_sha256 does not match its segments")
    canonical_input_id = document.get("canonical_input_id")
    if not isinstance(canonical_input_id, str) or not canonical_input_id.strip():
        raise ValueError("canonical_input_id must be non-blank text")
    input_sha256 = _sha256_field(document.get("input_sha256"), "input_sha256")
    timebase = validate_timebase(document.get("timebase"))
    target = document.get("target_identity_sha256")
    if target is not None:
        target = _sha256_field(target, "target_identity_sha256")
    # The served envelope may carry additive metadata (job_id,
    # labeling_sha256, quality_state, quality_reason); only the required
    # subset is validated, additive fields are tolerated.
    if "labeling_sha256" in document and document["labeling_sha256"] != declared:
        raise ValueError("envelope labeling_sha256 does not match its segments")
    return {"segments": segments, "content_sha256": declared,
            "canonical_input_id": canonical_input_id.strip(), "input_sha256": input_sha256,
            "timebase": timebase, "target_identity_sha256": target}


def render_identity(job_id, dm_result_sha256, labeling_sha256, sidecar):
    """sha256 of the canonical render-identity object (the frozen formula).

    ``labeling_sha256`` is the sidecar's content_sha256, or None for an
    explicitly unlabeled render; ``sidecar`` is the validated sidecar dict
    (or None). No fields beyond the frozen set are ever added.
    """
    if sidecar is None:
        if labeling_sha256 is not None:
            raise ValueError("an unlabeled render has no labeling_sha256")
        canonical_input_id = input_sha256 = timebase = target = None
    else:
        if labeling_sha256 != sidecar["content_sha256"]:
            raise ValueError("labeling_sha256 must be the sidecar content_sha256")
        canonical_input_id = sidecar["canonical_input_id"]
        input_sha256 = sidecar["input_sha256"]
        timebase = sidecar["timebase"]
        target = sidecar["target_identity_sha256"]
    payload = {"schema": RENDER_IDENTITY_SCHEMA, "job_id": job_id,
               "dm_result_sha256": _sha256_field(dm_result_sha256, "dm_result_sha256"),
               "labeling_sha256": labeling_sha256, "canonical_input_id": canonical_input_id,
               "input_sha256": input_sha256, "timebase": timebase,
               "target_identity_sha256": target, "render_recipe_version": RENDER_RECIPE_VERSION}
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be non-blank text")
    return _sha256(payload)


def active(segments, time_seconds):
    """Subtask texts whose half-open interval contains the presentation time."""
    return [text for start, end, text in segments if start <= time_seconds < end]
