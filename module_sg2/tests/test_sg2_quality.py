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

	def test_sg3_handoff_keeps_only_accepted_faces_with_ids(self):
		import tempfile
		from sg2_quality import save_sg3_npz, sg3_items
		faces = [self.face, {**self.face, "det_conf": 0.1}, self.face]
		out = process_frame(self.frame, faces, self.config, frame_id=9)
		items = sg3_items(out, faces)
		self.assertEqual([it["face_id"] for it in items], ["9_0", "9_2"])
		self.assertEqual(items[0]["box"], [230.0, 150.0, 180.0, 180.0])
		with tempfile.TemporaryDirectory() as tmp:
			n = save_sg3_npz(Path(tmp) / "f.npz", out, faces)
			d = np.load(Path(tmp) / "f.npz", allow_pickle=False)
			self.assertEqual(n, 2)
			self.assertEqual(d["faces"].shape, (2, 112, 112, 3))
			self.assertEqual(d["faces"].dtype, np.uint8)
			self.assertEqual(list(d["face_ids"]), ["9_0", "9_2"])
			self.assertEqual(int(d["frame_id"]), 9)
			# a frame with no accepted face still gives a valid, empty file
			save_sg3_npz(Path(tmp) / "e.npz", process_frame(self.frame, [], self.config, 10))
			self.assertEqual(np.load(Path(tmp) / "e.npz")["faces"].shape, (0, 112, 112, 3))

	def test_empty_input_produces_empty_output(self):
		self.assertEqual(process_frame(self.frame, [], self.config, frame_id=8), {
			"frame_id": 8,
			"results": [],
		})


YUNET = SG2_DIR / "models" / "face_detection_yunet_2023mar.onnx"
SAMPLE = SG2_DIR / "samples" / "example_input_frame.jpg"
EXAMPLE_IN = SG2_DIR.parent / "interfaces" / "examples" / "sg2_example_input.json"


@unittest.skipUnless(YUNET.is_file(), "run scripts/download_assets.py --models first")
class LandmarkFallbackTests(unittest.TestCase):
	"""SG-1 (YOLOv12n-face) sends boxes only: SG-2 must find landmarks itself and still align."""

	@classmethod
	def setUpClass(cls):
		import json
		cls.config = load_config(SG2_DIR / "config" / "sg1_yolov12_v1.json")
		cls.frame = cv2.imread(str(SAMPLE))
		cls.face = json.loads(EXAMPLE_IN.read_text())["faces"][0]  # real face with YuNet landmarks

	def test_box_only_face_is_accepted_via_fallback(self):
		box_only = {**self.face, "landmarks": None}
		result = assess_face(self.frame, box_only, self.config)

		self.assertTrue(result["accept"], result["reason"])
		self.assertEqual(result["metrics"]["lmk_source"], "yunet_roi")
		self.assertEqual(result["face"].shape, (112, 112, 3))
		# recovered landmarks agree with the full-frame ones to within 15% of the eye distance
		ref = np.float32(self.face["landmarks"])
		err = np.linalg.norm(np.float32(result["landmarks"]) - ref, axis=1).mean()
		self.assertLess(err / np.linalg.norm(ref[1] - ref[0]), 0.15)

	def test_sg1_landmarks_take_priority_over_fallback(self):
		result = assess_face(self.frame, self.face, self.config)
		self.assertEqual(result["metrics"]["lmk_source"], "sg1")

	def test_box_without_a_face_is_rejected_not_raised(self):
		h, w = self.frame.shape[:2]
		empty = {"box": [0, 0, 40, 40], "det_conf": 0.9, "landmarks": None}
		flat = np.full_like(self.frame, 128)
		result = assess_face(flat, empty, self.config)
		self.assertFalse(result["accept"])
		self.assertEqual(result["reason"], "no_landmarks")

	def test_baseline_config_still_rejects_box_only_faces(self):
		baseline = load_config(SG2_DIR / "config" / "baseline_v1.json")
		result = assess_face(self.frame, {**self.face, "landmarks": None}, baseline)
		self.assertEqual(result["reason"], "no_landmarks")


if __name__ == "__main__":
	unittest.main()
