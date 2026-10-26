import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

SG2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SG2_DIR))

from sg2_quality import TEMPLATE_112, assess_face, load_config, process_frame, to_json


class SG2QualityContractTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.config = load_config(SG2_DIR / "config" / "baseline_v1.json")
		rng = np.random.default_rng(7)
		cls.frame = rng.integers(20, 235, (480, 640, 3), dtype=np.uint8)
		cls.landmarks = ((TEMPLATE_112 - 56) * 1.5 + [320, 240]).tolist()
		cls.face = {
			"box": [230, 150, 180, 180],
			"det_conf": 0.95,
			"landmarks": cls.landmarks,
		}

	def test_accepted_face_matches_sg3_contract(self):
		result = assess_face(self.frame, self.face, self.config)

		self.assertTrue(result["accept"])
		self.assertIsNone(result["reason"])
		self.assertEqual(result["face"].shape, (112, 112, 3))
		self.assertEqual(result["face"].dtype, np.uint8)
		self.assertGreaterEqual(result["quality"], 0.0)
		self.assertLessEqual(result["quality"], 1.0)

	def test_process_frame_preserves_order_and_json_removes_images(self):
		output = process_frame(self.frame, [self.face, {**self.face, "det_conf": 0.1}], self.config, frame_id=4)

		self.assertEqual([r["face_index"] for r in output["results"]], [0, 1])
		self.assertTrue(output["results"][0]["accept"])
		self.assertEqual(output["results"][1]["reason"], "low_det_conf")
		serializable = to_json(output)
		self.assertNotIn("face", serializable["results"][0])
		self.assertEqual(serializable["frame_id"], 4)

	def test_rejects_are_explainable_and_do_not_raise(self):
		cases = [
			({**self.face, "box": [230, 150, 20, 20]}, "too_small"),
			({**self.face, "landmarks": None}, "no_landmarks"),
			({**self.face, "det_conf": 0.1}, "low_det_conf"),
			({"box": "invalid"}, "bad_input"),
		]

		for face, reason in cases:
			with self.subTest(reason=reason):
				result = assess_face(self.frame, face, self.config)
				self.assertFalse(result["accept"])
				self.assertEqual(result["reason"], reason)
				self.assertIsNone(result["face"])

	def test_empty_input_produces_empty_output(self):
		self.assertEqual(process_frame(self.frame, [], self.config, frame_id=8), {
			"frame_id": 8,
			"results": [],
		})


if __name__ == "__main__":
	unittest.main()
