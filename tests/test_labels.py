"""Contract tests for the ego-anno-ar-labels/v1 sidecar and render identity."""
import hashlib
import json
import unittest

from ego_anno.rendering import labels


SEGMENTS = [
    {"start_sec": 2.0, "end_sec": 3.5, "subtask": "carry the cup"},
    {"start_sec": 0.0, "end_sec": 1.0, "subtask": "pick up the cup"},
]


def sidecar(**overrides):
    document = {
        "schema": "ego-anno-ar-labels/v1",
        "content_sha256": labels.content_sha256(labels._segments(SEGMENTS, "segments")),
        "canonical_input_id": "clip-024a",
        "input_sha256": "a" * 64,
        "timebase": {"clock": "uploaded_video_seconds", "origin": "first_frame_pts",
                     "half_open": True, "fps": 15},
        "target_identity_sha256": None,
        "segments": SEGMENTS,
    }
    document.update(overrides)
    return document


class MinimalDocumentTests(unittest.TestCase):
    def test_validate_sorts_and_normalizes(self):
        doc = {"schema": "ego-anno-ar-labels/v1", "clock": "clip_local_seconds",
               "labels": [dict(SEGMENTS[1], start_sec=0), SEGMENTS[0]]}
        segments = labels.validate(doc)
        self.assertEqual([(s, e) for s, e, _ in segments], [(0.0, 1.0), (2.0, 3.5)])
        self.assertIs(type(segments[0][0]), float)
        self.assertIs(type(segments[0][2]), str)

    def test_validate_rejects_malformed(self):
        base = {"schema": "ego-anno-ar-labels/v1", "clock": "clip_local_seconds", "labels": []}
        for document, message in (
            ([], "JSON object"),
            (dict(base, schema="other/v1"), "schema"),
            (dict(base, clock="recording_seconds"), "clock"),
            (dict(base, labels={}), "list"),
            (dict(base, labels=[{"start_sec": 0, "end_sec": 1}]), "exactly"),
            (dict(base, labels=[{"start_sec": 1, "end_sec": 1, "subtask": "x"}]), "start_sec < end_sec"),
            (dict(base, labels=[{"start_sec": -1, "end_sec": 1, "subtask": "x"}]), "start_sec < end_sec"),
            (dict(base, labels=[{"start_sec": float("nan"), "end_sec": 1, "subtask": "x"}]), "finite"),
            (dict(base, labels=[{"start_sec": 0, "end_sec": 1, "subtask": "  "}]), "non-blank"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                labels.validate(document)

    def test_identity_is_order_independent_and_content_sensitive(self):
        doc = {"schema": "ego-anno-ar-labels/v1", "clock": "clip_local_seconds",
               "labels": list(reversed(SEGMENTS))}
        reference = labels.identity(doc)
        shuffled_keys = json.loads(json.dumps(doc), object_pairs_hook=dict)
        self.assertEqual(labels.identity(shuffled_keys), reference)
        changed = dict(doc, labels=[dict(SEGMENTS[0], subtask="carry the mug"), SEGMENTS[1]])
        self.assertNotEqual(labels.identity(changed), reference)
        # The reference value pinned by the AR binding layer for this payload.
        self.assertEqual(reference, labels.identity({"schema": "ego-anno-ar-labels/v1",
                                                     "clock": "clip_local_seconds",
                                                     "labels": SEGMENTS}))

    def test_active_uses_half_open_intervals(self):
        segments = labels.validate({"schema": "ego-anno-ar-labels/v1", "clock": "clip_local_seconds",
                                    "labels": SEGMENTS})
        self.assertEqual(labels.active(segments, 0.0), ["pick up the cup"])
        self.assertEqual(labels.active(segments, 0.999), ["pick up the cup"])
        self.assertEqual(labels.active(segments, 1.0), [])
        self.assertEqual(labels.active(segments, 1.5), [])
        self.assertEqual(labels.active(segments, 2.0), ["carry the cup"])
        self.assertEqual(labels.active(segments, 3.5), [])

    def test_active_supports_parallel_labels(self):
        doc = {"schema": "ego-anno-ar-labels/v1", "clock": "clip_local_seconds",
               "labels": [{"start_sec": 0, "end_sec": 2, "subtask": "a"},
                          {"start_sec": 1, "end_sec": 3, "subtask": "b"}]}
        segments = labels.validate(doc)
        self.assertEqual(labels.active(segments, 1.5), ["a", "b"])


class SidecarTests(unittest.TestCase):
    def test_valid_sidecar_roundtrip(self):
        checked = labels.validate_sidecar(sidecar())
        self.assertEqual(checked["content_sha256"], sidecar()["content_sha256"])
        self.assertEqual(checked["canonical_input_id"], "clip-024a")
        self.assertIsNone(checked["target_identity_sha256"])
        self.assertEqual(checked["timebase"]["fps"], 15.0)
        self.assertEqual([(s, e) for s, e, _ in checked["segments"]], [(0.0, 1.0), (2.0, 3.5)])

    def test_content_sha256_matches_the_frozen_formula(self):
        segments = labels._segments(SEGMENTS, "segments")
        canonical = json.dumps([{"start_sec": s, "end_sec": e, "subtask": t} for s, e, t in segments],
                               sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(labels.content_sha256(segments), hashlib.sha256(canonical).hexdigest())

    def test_content_mismatch_is_rejected(self):
        document = sidecar(content_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "content_sha256 does not match"):
            labels.validate_sidecar(document)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            labels.validate_sidecar(sidecar(extra_field=1))

    def test_timebase_is_validated(self):
        with self.assertRaisesRegex(ValueError, "timebase.clock"):
            labels.validate_sidecar(sidecar(timebase={"clock": "recording_seconds",
                                                      "origin": "first_frame_pts", "half_open": True, "fps": 15}))
        with self.assertRaisesRegex(ValueError, "timebase.half_open"):
            labels.validate_sidecar(sidecar(timebase={"clock": "uploaded_video_seconds",
                                                      "origin": "first_frame_pts", "half_open": False, "fps": 15}))
        with self.assertRaisesRegex(ValueError, "timebase.fps"):
            labels.validate_sidecar(sidecar(timebase={"clock": "uploaded_video_seconds",
                                                      "origin": "first_frame_pts", "half_open": True, "fps": 0}))

    def test_target_is_null_or_hex(self):
        checked = labels.validate_sidecar(sidecar(target_identity_sha256="c" * 64))
        self.assertEqual(checked["target_identity_sha256"], "c" * 64)
        with self.assertRaisesRegex(ValueError, "target_identity_sha256"):
            labels.validate_sidecar(sidecar(target_identity_sha256="0" * 63))


class RenderIdentityTests(unittest.TestCase):
    def test_frozen_formula_and_field_set(self):
        checked = labels.validate_sidecar(sidecar())
        identity = labels.render_identity("9ddb3966-6e40-4c45-81a7-b6c35f18edf9",
                                          "d" * 64, checked["content_sha256"], checked)
        preimage = {"schema": "anneal-labeled-render/v1",
                    "job_id": "9ddb3966-6e40-4c45-81a7-b6c35f18edf9",
                    "dm_result_sha256": "d" * 64,
                    "labeling_sha256": checked["content_sha256"],
                    "canonical_input_id": "clip-024a",
                    "input_sha256": "a" * 64,
                    "timebase": checked["timebase"],
                    "target_identity_sha256": None,
                    "render_recipe_version": 1}
        canonical = json.dumps(preimage, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(identity, hashlib.sha256(canonical).hexdigest())

    def test_unlabeled_render_has_null_labeling_fields(self):
        identity = labels.render_identity("job-1", "d" * 64, None, None)
        preimage = {"schema": "anneal-labeled-render/v1", "job_id": "job-1",
                    "dm_result_sha256": "d" * 64, "labeling_sha256": None,
                    "canonical_input_id": None, "input_sha256": None, "timebase": None,
                    "target_identity_sha256": None, "render_recipe_version": 1}
        canonical = json.dumps(preimage, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(identity, hashlib.sha256(canonical).hexdigest())

    def test_labeling_sha_must_be_the_sidecar_content_sha(self):
        checked = labels.validate_sidecar(sidecar())
        with self.assertRaisesRegex(ValueError, "labeling_sha256"):
            labels.render_identity("job-1", "d" * 64, "e" * 64, checked)
        with self.assertRaisesRegex(ValueError, "no labeling_sha256"):
            labels.render_identity("job-1", "d" * 64, "e" * 64, None)

    def test_identity_depends_on_every_input(self):
        checked = labels.validate_sidecar(sidecar())
        base = labels.render_identity("job-1", "d" * 64, checked["content_sha256"], checked)
        self.assertNotEqual(labels.render_identity("job-2", "d" * 64, checked["content_sha256"], checked), base)
        self.assertNotEqual(labels.render_identity("job-1", "e" * 64, checked["content_sha256"], checked), base)
        verified = labels.validate_sidecar(sidecar(target_identity_sha256="c" * 64))
        self.assertNotEqual(labels.render_identity("job-1", "d" * 64, verified["content_sha256"], verified), base)


if __name__ == "__main__":
    unittest.main()
