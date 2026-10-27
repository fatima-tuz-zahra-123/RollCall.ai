"""Boundary tests use synthetic arrays, not evidence of recognition accuracy."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from module_sg2 import sg2_quality
from module_sg3.api import RecognitionService
from module_sg3.mock_sg4 import consume_frame, consume_result
from module_sg3.pipeline import YuNetSG2Pipeline

ROOT = Path(__file__).resolve().parents[2]


class StubRecognizer:
    method = "sface"
    score_type = "cosine_similarity"
    threshold = 0.363

    def __init__(self, score=0.8, sid="S001", fail=False):
        self.score, self.sid, self.fail, self.calls = score, sid, fail, 0

    def match(self, face, row=None):
        self.calls += 1
        if self.fail:
            raise cv2.error("synthetic inference failure")
        return self.sid, self.score


def accepted_result():
    return {"accept": True, "reason": None, "face": np.full((112, 112, 3), 128, np.uint8),
            "quality": 0.8, "metrics": {"blur": 150.0, "yaw": None}}


def pipeline_with_fake_detector(rows, max_side=640):
    class FakeDetector:
        def setInputSize(self, size):
            self.size = size

        def detect(self, image):
            self.input_shape = image.shape
            return 1, rows

    pipeline = YuNetSG2Pipeline.__new__(YuNetSG2Pipeline)
    pipeline.sg2_cfg = sg2_quality.load_config(ROOT / "module_sg2/config/baseline_v1.json")
    pipeline.max_side = max_side
    pipeline.det = FakeDetector()
    return pipeline


class RecognitionAPITests(unittest.TestCase):
    def test_recognized_result_preserves_metadata_and_full_score(self):
        rec = StubRecognizer(score=0.612345678901)
        result = accepted_result()
        result.update(frame_id=12, detection_id=3, timestamp=1.25, bbox=[20, 30, 45, 60])
        out = RecognitionService(rec).recognize_face(result)
        self.assertEqual(out["status"], "RECOGNIZED")
        self.assertEqual(out["student_id"], "S001")
        self.assertEqual(out["score"], rec.score)
        self.assertEqual(out["bbox"], [20.0, 30.0, 45.0, 60.0])
        self.assertEqual(out["frame_id"], 12)
        self.assertEqual(out["timestamp"], 1.25)
        self.assertEqual(out["detection_id"], 3)
        self.assertGreaterEqual(out["recognition_ms"], 0)
        json.dumps(out, allow_nan=False)
        self.assertEqual(rec.calls, 1)

    def test_unknown_retains_candidate_without_assigning_it(self):
        out = RecognitionService(StubRecognizer(score=0.2)).recognize_face(accepted_result())
        self.assertEqual((out["status"], out["student_id"], out["best_match"]),
                         ("UNKNOWN", "UNKNOWN", "S001"))
        self.assertEqual(out["reason"], "below_threshold")

    def test_quality_rejection_never_calls_recognition(self):
        rec = StubRecognizer()
        for reason in ("too_small", "pose", "blurry", "illumination", "no_landmarks"):
            with self.subTest(reason=reason):
                out = RecognitionService(rec).recognize_face(
                    {"accept": False, "reason": reason, "face": None, "quality": 0.0, "metrics": {}})
                self.assertEqual(out["status"], "QUALITY_REJECTED")
                self.assertEqual(out["reason"], reason)
                self.assertIsNone(out["student_id"])
                self.assertIsNone(out["score"])
        self.assertEqual(rec.calls, 0)

    def test_malformed_input_never_calls_recognition(self):
        rec = StubRecognizer()
        bad = [None, [], {}, {**accepted_result(), "accept": "true"},
               {**accepted_result(), "face": None},
               {**accepted_result(), "face": np.zeros((112, 112), np.uint8)},
               {**accepted_result(), "face": np.zeros((112, 112, 3), np.float32)},
               {**accepted_result(), "face": np.zeros((224, 224, 3), np.uint8)},
               {**accepted_result(), "quality": float("nan")},
               {**accepted_result(), "metrics": {"blur": float("inf")}},
               {**accepted_result(), "reason": "contradictory_rejection"}]
        for item in bad:
            with self.subTest(item_type=type(item).__name__):
                out = RecognitionService(rec).recognize_face(item)
                self.assertEqual(out["status"], "ERROR")
                json.dumps(out, allow_nan=False)
        self.assertEqual(rec.calls, 0)

    def test_metadata_validation_and_override(self):
        rec = StubRecognizer()
        for metadata in ({"timestamp": float("nan")}, {"timestamp": -1},
                         {"frame_id": []}, {"detection_id": True},
                         {"bbox": [0, 0, 0, 20]}, {"bbox": [0, 0, 20]}, []):
            out = RecognitionService(rec).recognize_face(accepted_result(), metadata)
            self.assertEqual(out["status"], "ERROR")
            json.dumps(out, allow_nan=False)
        self.assertEqual(rec.calls, 0)
        result = accepted_result()
        result["frame_id"] = 1
        out = RecognitionService(rec).recognize_face(result, {"frame_id": 2})
        self.assertEqual(out["frame_id"], 2)

    def test_empty_gallery_and_inference_failure_are_errors(self):
        for rec, reason in [(StubRecognizer(sid=None), "empty_or_invalid_gallery"),
                            (StubRecognizer(fail=True), "recognition_failure"),
                            (StubRecognizer(score=float("nan")), "recognition_failure")]:
            out = RecognitionService(rec).recognize_face(accepted_result())
            self.assertEqual(out["status"], "ERROR")
            self.assertEqual(out["reason"], reason)
            self.assertIsNone(out["student_id"])
            self.assertIsNone(out["score"])

    def test_threshold_comparison_directions_and_boundary(self):
        rec = StubRecognizer(score=0.363)
        self.assertEqual(RecognitionService(rec).recognize_face(accepted_result())["status"], "RECOGNIZED")
        rec.method, rec.score_type, rec.threshold = "lbph", "lbph_distance", 80.0
        rec.score = 80.0
        self.assertEqual(RecognitionService(rec).recognize_face(accepted_result())["status"], "RECOGNIZED")
        rec.score = 81.0
        self.assertEqual(RecognitionService(rec).recognize_face(accepted_result())["status"], "UNKNOWN")

    def test_preprocessing_failure_is_not_quality_rejection(self):
        rec = StubRecognizer()
        out = RecognitionService(rec).recognize_face(
            {"accept": False, "reason": "preprocessing_failure", "pipeline_error": True})
        self.assertEqual((out["status"], out["reason"]), ("ERROR", "preprocessing_failure"))
        self.assertEqual(rec.calls, 0)


class ActualSG2BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.frame = self.rng.integers(30, 225, (480, 640, 3), dtype=np.uint8)
        self.landmarks = (sg2_quality.TEMPLATE_112 * 1.5 + [200, 100]).tolist()
        self.box = [210, 120, 160, 180]
        self.row = np.array([*self.box, *np.array(self.landmarks).ravel(), .95], np.float32)

    def test_real_sg2_output_is_consumed_with_original_coordinates(self):
        pipeline = pipeline_with_fake_detector(np.array([self.row]))
        out = pipeline.process(self.frame, frame_id=23, timestamp=3.5)
        self.assertEqual(len(out["results"]), 1)
        sg2 = out["results"][0]
        self.assertTrue(sg2["accept"])
        self.assertEqual(sg2["face"].shape, (112, 112, 3))
        self.assertEqual(sg2["face"].dtype, np.uint8)
        self.assertEqual(sg2["bbox"], self.box)
        evidence = RecognitionService(StubRecognizer()).recognize_face(sg2)
        downstream = consume_result(evidence)
        self.assertEqual(downstream["bbox_xyxy"], [210, 120, 370, 300])
        self.assertEqual((downstream["frame_id"], downstream["timestamp"]), (23, 3.5))
        self.assertFalse(downstream["tracking"])

    def test_actual_sg2_pose_blur_and_dark_rejections(self):
        turned = self.row.copy()
        turned[8] += 80  # nose off eye-axis midpoint: actual SG2 yaw gate
        cases = [(self.frame, turned, "pose"),
                 (np.full_like(self.frame, 128), self.row, "blurry"),
                 (np.full_like(self.frame, 5), self.row, "illumination")]
        rec = StubRecognizer()
        for frame, row, reason in cases:
            with self.subTest(reason=reason):
                sg2 = pipeline_with_fake_detector(np.array([row])).process(frame)["results"][0]
                self.assertFalse(sg2["accept"])
                self.assertEqual(sg2["reason"], reason)
                self.assertEqual(RecognitionService(rec).recognize_face(sg2)["status"], "QUALITY_REJECTED")
        self.assertEqual(rec.calls, 0)

    def test_multiple_faces_and_empty_frame_detections(self):
        pipeline = pipeline_with_fake_detector(np.array([self.row, self.row]))
        out = pipeline.process(self.frame)
        self.assertEqual([r["face_index"] for r in out["results"]], [0, 1])
        self.assertEqual([r["detection_id"] for r in out["results"]], [0, 1])
        self.assertEqual(pipeline_with_fake_detector(None).process(self.frame)["results"], [])

    def test_coordinate_restoration_accounts_for_resize_rounding(self):
        row = np.array([33.3, 20, 66.6, 40, 40, 30, 80, 30, 60, 40, 45, 50, 75, 50, .9])
        pipeline = pipeline_with_fake_detector(np.array([row]), max_side=333)
        out = pipeline.detect(np.zeros((601, 1000, 3), np.uint8))[0]
        np.testing.assert_allclose(out["box"], [100, 60.1, 200, 120.2])
        self.assertEqual(pipeline.det.size, (333, 200))
        np.testing.assert_allclose(np.array(out["landmarks"])[0], [40 / .333, 30 / (200/601)])

    def test_bad_frame_raises_and_sg2_failure_is_per_face(self):
        pipeline = pipeline_with_fake_detector(np.array([self.row]))
        with self.assertRaises(ValueError):
            pipeline.process(None)
        with patch("module_sg3.pipeline.sg2_quality.process_frame", side_effect=RuntimeError("test")):
            out = pipeline.process(self.frame)
        self.assertTrue(out["results"][0]["pipeline_error"])
        self.assertEqual(RecognitionService(StubRecognizer()).recognize_face(out["results"][0])["status"], "ERROR")


class MockConsumerTests(unittest.TestCase):
    def test_empty_and_unknown(self):
        self.assertEqual(consume_frame([]), [])
        unknown = RecognitionService(StubRecognizer(score=.1)).recognize_face(accepted_result())
        self.assertEqual(consume_result(unknown)["id"], "UNKNOWN")

    def test_rejects_inconsistent_or_malformed_evidence(self):
        out = RecognitionService(StubRecognizer()).recognize_face(accepted_result())
        for bad in ({**out, "status": "tracking"}, {**out, "score": float("nan")},
                    {**out, "student_id": "UNKNOWN"}, {**out, "score": .01},
                    {"schema_version": "sg3.v1"}):
            with self.assertRaises(ValueError):
                consume_result(bad)


if __name__ == "__main__":
    unittest.main()
