"""SG-1 -> SG-2 adapter: turns SG-1's YOLOv12n-face output into the V1 `faces` list SG-2 consumes.

SG-1 (module_sg1/run.py) wraps Ultralytics YOLO. A Results object gives, per face:
    boxes.xyxy  [x1, y1, x2, y2]  absolute pixels in the ORIGINAL frame (Ultralytics undoes the letterbox)
    boxes.conf  float in [0, 1]
    keypoints   None for this checkpoint (DetectionModel / Detect head, no pose head)
V1 (interfaces/sg2_v1_interface.md) wants:
    {"box": [x, y, w, h], "det_conf": float, "landmarks": [[x, y] x 5] or None}

`xyxy_to_v1_faces` is pure numpy so it is unit-testable without torch. Once SG-1 agrees, this file can move
into module_sg1/ so SG-1 emits V1 directly.
"""
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO / "module_sg1" / "weights" / "yolov12n-face.pt"


def xyxy_to_v1_faces(xyxy, conf, keypoints=None, frame_shape=None):
    """[N,4] xyxy + [N] conf (+ optional [N,5,2] keypoints) -> V1 faces, boxes clipped to the frame."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    conf = np.asarray(conf, dtype=float).reshape(-1)
    faces = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        if frame_shape is not None:
            h, w = frame_shape[:2]
            x1, x2 = np.clip([x1, x2], 0, w)
            y1, y2 = np.clip([y1, y2], 0, h)
        if x2 <= x1 or y2 <= y1:
            continue  # degenerate after clipping: never forward it
        lmk = None
        if keypoints is not None and len(keypoints) > i and np.shape(keypoints[i]) == (5, 2):
            lmk = np.asarray(keypoints[i], dtype=float).tolist()
        faces.append({"box": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                      "det_conf": float(np.clip(conf[i], 0.0, 1.0)),
                      "landmarks": lmk})
    return faces


class SG1Detector:
    """Thin wrapper around SG-1's model with the same settings as `run.py predict`."""

    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.25, imgsz=1280, device=None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, frame_bgr):
        r = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, max_det=1000,
                               device=self.device, verbose=False)[0]
        kpts = None
        if getattr(r, "keypoints", None) is not None and r.keypoints is not None:
            kpts = r.keypoints.xy.cpu().numpy()
        return xyxy_to_v1_faces(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), kpts, frame_bgr.shape)
