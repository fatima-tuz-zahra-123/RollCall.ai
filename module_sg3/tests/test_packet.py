"""Incoming four-field SG2 packet compatibility; model-free boundary tests."""
import json
import unittest

import numpy as np

from module_sg3.api import RecognitionService
from module_sg3.mock_sg4 import consume_result


class StubRecognizer:
    method = "sface"
    score_type = "cosine_similarity"
    threshold = .363

    def __init__(self, score=.8):
        self.score, self.calls, self.last_image = score, 0, None

    def match(self, image, row=None):
        self.calls += 1
        self.last_image = image
        return "S001", self.score


def packet(quality_pass=True):
    return {"frame_id": 42, "face_id": "face-2", "quality_pass": quality_pass,
            "aligned_face": np.full((112, 112, 3), 128, np.uint8) if quality_pass else None}


class IncomingPacketTests(unittest.TestCase):
    def test_minimal_four_field_packet_recognizes_without_inventing_geometry(self):
        rec, incoming = StubRecognizer(), packet()
        out = RecognitionService(rec).recognize_face(incoming)
        self.assertEqual(out["status"], "RECOGNIZED")
        self.assertEqual(out["student_id"], "S001")
        self.assertEqual(out["frame_id"], 42)
        self.assertEqual(out["face_id"], "face-2")
        self.assertEqual(out["detection_id"], out["face_id"])
        self.assertIsNone(out["bbox"])
        self.assertIsNone(out["timestamp"])
        self.assertIsNone(out["quality"])
        self.assertEqual(out["quality_metrics"], {})
        self.assertIs(rec.last_image, incoming["aligned_face"])
        self.assertEqual(rec.calls, 1)
        json.dumps(out, allow_nan=False)
        self.assertIsNone(consume_result(out)["bbox_xyxy"])

    def test_quality_false_is_rejection_and_never_unknown_or_inference(self):
        rec = StubRecognizer()
        service = RecognitionService(rec)
        for incoming in (packet(False), {"frame_id": 42, "face_id": 0, "quality_pass": False}):
            out = service.recognize_face(incoming)
            self.assertEqual(out["status"], "QUALITY_REJECTED")
            self.assertIsNone(out["student_id"])
            self.assertIsNone(out["score"])
            self.assertEqual(out["face_id"], incoming["face_id"])
        self.assertEqual(rec.calls, 0)

    def test_unknown_decision_preserves_packet_ids(self):
        out = RecognitionService(StubRecognizer(score=.1)).recognize_face(packet())
        self.assertEqual((out["status"], out["student_id"]), ("UNKNOWN", "UNKNOWN"))
        self.assertEqual(out["face_id"], "face-2")
        self.assertEqual(out["detection_id"], "face-2")

    def test_existing_result_shape_keeps_working_and_exposes_face_id_alias(self):
        incoming = packet()
        legacy = {"accept": True, "face": incoming["aligned_face"], "detection_id": 3,
                  "frame_id": 42, "reason": None, "quality": .9, "metrics": {"blur": 123.0}}
        out = RecognitionService(StubRecognizer()).recognize_face(legacy)
        self.assertEqual(out["status"], "RECOGNIZED")
        self.assertEqual((out["detection_id"], out["face_id"]), (3, 3))

    def test_consistent_duplicate_aliases_are_accepted(self):
        incoming = packet()
        incoming.update(accept=True, face=incoming["aligned_face"].copy(), detection_id=incoming["face_id"])
        self.assertEqual(RecognitionService(StubRecognizer()).recognize_face(incoming)["status"], "RECOGNIZED")

    def test_contradictory_aliases_and_crossed_partial_packets_are_errors(self):
        original = packet()
        cases = [{**original, "accept": False},
                 {**original, "face": np.zeros((112, 112, 3), np.uint8)},
                 {**original, "face": None},
                 {**original, "detection_id": "different-face"},
                 {"frame_id": 42, "face_id": 2, "accept": True, "aligned_face": original["aligned_face"]},
                 {"frame_id": 42, "face_id": 2, "quality_pass": True, "face": original["aligned_face"]},
                 {**original, "quality_pass": "true"},
                 {**original, "accept": 1},
                 {**original, "face_id": []}]
        rec = StubRecognizer()
        service = RecognitionService(rec)
        for incoming in cases:
            with self.subTest(keys=list(incoming)):
                result = service.recognize_face(incoming)
                self.assertEqual(result["status"], "ERROR")
                self.assertEqual(result["reason"], "invalid_input")
                json.dumps(result, allow_nan=False)
        self.assertEqual(rec.calls, 0)

    def test_bad_aligned_pixels_are_errors_without_calling_model(self):
        rec = StubRecognizer()
        for image in (None, [], np.zeros((112, 112), np.uint8),
                      np.zeros((112, 112, 3), np.float32), np.zeros((224, 224, 3), np.uint8)):
            incoming = {**packet(), "aligned_face": image}
            self.assertEqual(RecognitionService(rec).recognize_face(incoming)["status"], "ERROR")
        self.assertEqual(rec.calls, 0)

    def test_optional_metadata_can_add_geometry_but_cannot_change_face_id(self):
        rec = StubRecognizer()
        service = RecognitionService(rec)
        out = service.recognize_face(packet(), {"bbox": [10, 20, 30, 40], "timestamp": 1.25})
        self.assertEqual(out["bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(out["timestamp"], 1.25)
        for metadata in ({"face_id": "other"}, {"detection_id": "other"},
                         {"face_id": "face-2", "detection_id": "other"}):
            self.assertEqual(service.recognize_face(packet(), metadata)["status"], "ERROR")
        self.assertEqual(rec.calls, 1)

    def test_empty_or_malformed_input_retains_error_contract(self):
        rec = StubRecognizer()
        for incoming in (None, {}, [], {"frame_id": 42, "face_id": 2}):
            out = RecognitionService(rec).recognize_face(incoming)
            self.assertEqual(out["status"], "ERROR")
            self.assertIsNone(out["student_id"])
            self.assertIsNone(out["face_id"])
        self.assertEqual(rec.calls, 0)


if __name__ == "__main__":
    unittest.main()
