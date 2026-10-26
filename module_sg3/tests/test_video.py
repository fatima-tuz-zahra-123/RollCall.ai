"""Model-free video contract tests with an actual encode/decode round trip."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from module_sg3.video_demo import run_video


class FakePipeline:
    def process(self, frame, frame_id=None, timestamp=None):
        base = {"frame_id": frame_id, "timestamp": timestamp, "bbox": [2, 3, 20, 21]}
        return {
            "results": [{**base, "accept": True, "detection_id": 0},
                        {**base, "accept": False, "detection_id": 1, "reason": "blurry"}],
            "detection_ms": 1.0,
            "preprocessing_ms": 2.0,
        }


class FakeService:
    def recognize_face(self, face):
        return {
            "status": "RECOGNIZED" if face["accept"] else "QUALITY_REJECTED",
            "student_id": "STUDENT_001" if face["accept"] else None,
            "score": 0.5 if face["accept"] else None,
            "score_type": "cosine_similarity",
            "reason": face.get("reason"),
            "bbox": face["bbox"],
            "frame_id": face["frame_id"],
            "timestamp": face["timestamp"],
            "detection_id": face["detection_id"],
            "recognition_ms": 0.2 if face["accept"] else 0.0,
        }


class VideoDemoTests(unittest.TestCase):
    def setUp(self):
        # Native OpenCV cannot write the redirected Windows sandbox temp folder.
        temp_root = Path(__file__).resolve().parents[2] / "private_runs" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.avi"
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
        if not writer.isOpened():
            writer.release()
            self.skipTest("MJPG video writer is unavailable in this OpenCV build")
        for index in range(20):
            frame = np.full((64, 96, 3), index * 10, np.uint8)
            writer.write(frame)
        writer.release()

    def run_demo(self, **kwargs):
        return run_video(self.source, self.root / "output", cfg={}, pipeline=FakePipeline(),
                         service=FakeService(), **kwargs)

    def test_sampling_metadata_statuses_and_real_saved_video(self):
        summary = self.run_demo(interval=0.5, start=0.2, duration=1.1, max_frames=10)
        self.assertEqual(summary["sampling"]["sampled_frames"], 3)
        self.assertEqual(summary["preview"]["decoded_frames"], 3)
        self.assertEqual(summary["preview"]["duration_seconds"], 1.5)
        self.assertEqual(summary["counts"]["detected_faces"], 6)
        self.assertEqual(summary["counts"]["sg2_accepted_faces"], 3)
        self.assertEqual(summary["counts"]["statuses"]["QUALITY_REJECTED"], 3)
        self.assertEqual(summary["counts"]["quality_rejection_reasons"], {"blurry": 3})
        self.assertFalse(summary["tracking"])
        self.assertFalse(summary["attendance_records"])
        self.assertIsNone(summary["accuracy"])
        saved = self.root / "output" / "frame_results.jsonl"
        records = [json.loads(line) for line in saved.read_text().splitlines()]
        self.assertEqual([r["frame_id"] for r in records], [2, 7, 12])
        self.assertEqual([r["timestamp"] for r in records], [0.2, 0.7, 1.2])
        self.assertEqual(records[1]["results"][0]["bbox"], [2, 3, 20, 21])
        self.assertEqual(records[1]["results"][0]["frame_id"], 7)
        self.assertEqual(summary["timing"]["stages"]["detection"]["total_ms"], 3.0)

    def test_frame_limit_and_existing_outputs(self):
        summary = self.run_demo(interval=0.1, max_frames=2)
        self.assertEqual(summary["sampling"]["sampled_frames"], 2)
        with self.assertRaises(FileExistsError):
            self.run_demo(interval=0.1, max_frames=2)
        summary = self.run_demo(interval=0.1, max_frames=1, overwrite=True)
        self.assertEqual(summary["preview"]["decoded_frames"], 1)

    def test_invalid_window_and_empty_start(self):
        for args in ({"interval": 0}, {"interval": float("nan")}, {"start": -1},
                     {"duration": 0}, {"max_frames": 0}, {"start": 3}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.run_demo(**args)
        self.assertFalse((self.root / "output").exists())

    def test_input_output_alias_cannot_destroy_original(self):
        aliased = self.root / "annotated_samples.avi"
        aliased.write_bytes(self.source.read_bytes())
        original = aliased.read_bytes()
        with self.assertRaises(ValueError):
            run_video(aliased, self.root, cfg={}, pipeline=FakePipeline(), service=FakeService(), overwrite=True)
        self.assertEqual(aliased.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
