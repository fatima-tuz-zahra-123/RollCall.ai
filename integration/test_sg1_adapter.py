"""SG-1 -> SG-2 interface tests (no torch needed).   python -m unittest integration/test_sg1_adapter.py -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "module_sg2"))

from sg1_adapter import xyxy_to_v1_faces  # noqa: E402
from sg2_quality import load_config, process_frame  # noqa: E402


class SG1AdapterTests(unittest.TestCase):
    def test_xyxy_becomes_xywh_with_no_landmarks(self):
        faces = xyxy_to_v1_faces([[10, 20, 50, 80]], [0.9])
        self.assertEqual(faces, [{"box": [10.0, 20.0, 40.0, 60.0], "det_conf": 0.9, "landmarks": None}])

    def test_boxes_are_clipped_to_frame_and_degenerate_ones_dropped(self):
        faces = xyxy_to_v1_faces([[-5, -5, 30, 30], [700, 10, 720, 40]], [0.8, 0.7], frame_shape=(480, 640, 3))
        self.assertEqual(len(faces), 1)  # second box lies fully right of a 640-wide frame
        self.assertEqual(faces[0]["box"], [0.0, 0.0, 30.0, 30.0])

    def test_keypoints_are_passed_through_when_present(self):
        kpts = np.arange(10, dtype=float).reshape(1, 5, 2)
        faces = xyxy_to_v1_faces([[0, 0, 10, 10]], [0.5], keypoints=kpts)
        self.assertEqual(faces[0]["landmarks"], kpts[0].tolist())

    def test_empty_detection_gives_empty_list(self):
        self.assertEqual(xyxy_to_v1_faces(np.zeros((0, 4)), np.zeros(0)), [])

    def test_adapter_output_is_accepted_by_sg2_without_raising(self):
        frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = xyxy_to_v1_faces([[100, 100, 200, 220], [5, 5, 15, 15]], [0.9, 0.3], frame_shape=frame.shape)
        cfg = load_config(HERE.parent / "module_sg2" / "config" / "baseline_v1.json")
        out = process_frame(frame, faces, cfg, frame_id=3)
        self.assertEqual([r["face_index"] for r in out["results"]], [0, 1])
        self.assertEqual(out["results"][1]["reason"], "low_det_conf")


if __name__ == "__main__":
    unittest.main()
