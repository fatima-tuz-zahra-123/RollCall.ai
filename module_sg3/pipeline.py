"""Temporary all-faces YuNet SG1 substitute and actual, unchanged SG2 adapter."""
from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from module_sg2 import sg2_quality
from .api import validate_metadata
from .sg3_core import model_path


class YuNetSG2Pipeline:
    """Detect all faces, restore full-frame coordinates, and invoke SG2 baseline.

    This is a temporary detector adapter, not a claim of SG1 integration.
    Detection/preprocessing timing includes resize and metadata conversion.
    """

    def __init__(self, cfg, sg2_config_path=None, max_side=None):
        self.sg2_config_path = Path(sg2_config_path) if sg2_config_path else (
            Path(__file__).resolve().parents[1] / "module_sg2" / "config" / "baseline_v1.json")
        self.sg2_cfg = sg2_quality.load_config(self.sg2_config_path)
        if self.sg2_cfg.get("output_size") != 112:
            raise ValueError("SG3 requires SG2 output_size=112")
        if self.sg2_cfg.get("align") != "similarity5":
            raise ValueError("SG3 prototype requires SG2 similarity5 alignment")
        detector_cfg = cfg["detector"]
        self.max_side = int(detector_cfg["max_side"] if max_side is None else max_side)
        if self.max_side < 32:
            raise ValueError("max_side must be at least 32")
        self.det = cv2.FaceDetectorYN.create(
            model_path(cfg, detector_cfg["model"]), "", (320, 320),
            float(detector_cfg["score_threshold"]), float(detector_cfg["nms_threshold"]), 5000)

    @staticmethod
    def _validate_frame(frame):
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
                or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1):
            raise ValueError("frame must be a nonempty HxWx3 uint8 BGR array")

    def detect(self, frame):
        """Return all SG2 detection dicts in original-frame xywh coordinates.

        YuNet row[4:14] has the exact order consumed by OpenCV alignCrop and
        SG2's identical TEMPLATE_112; no eye/mouth swap is applied.
        """
        self._validate_frame(frame)
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        image = frame if scale == 1.0 else cv2.resize(frame, (resized_width, resized_height))
        # Actual x/y ratios account for integer resize rounding independently.
        sx, sy = resized_width / width, resized_height / height
        self.det.setInputSize((resized_width, resized_height))
        _, rows = self.det.detect(image)
        if rows is None:
            return []
        detections = []
        for row in rows:
            row = np.asarray(row, dtype=np.float64).reshape(-1)
            if len(row) != 15 or not np.isfinite(row).all():
                raise ValueError("YuNet returned a malformed detection")
            box = [float(row[0] / sx), float(row[1] / sy),
                   float(row[2] / sx), float(row[3] / sy)]
            if box[2] <= 0 or box[3] <= 0 or not 0 <= row[14] <= 1:
                raise ValueError("YuNet returned invalid geometry or confidence")
            landmarks = row[4:14].reshape(5, 2) / [sx, sy]
            detections.append({"box": box, "det_conf": float(row[14]),
                               "landmarks": landmarks.tolist()})
        return detections

    def process(self, frame, frame_id=None, timestamp=None):
        """One SG2 result per face, augmented with source association fields.

        Invalid frame/detector output raises ValueError (frame-level failure).
        An SG2 exception becomes pipeline_error on that face, which the SG3 API
        reports as ERROR. A normal SG2 gate rejection remains QUALITY_REJECTED.
        """
        metadata = validate_metadata({"frame_id": frame_id, "timestamp": timestamp})
        start = time.perf_counter()
        detections = self.detect(frame)
        detection_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        results = []
        # Isolate SG2 exceptions per face without changing subgroup ownership.
        for index, detection in enumerate(detections):
            try:
                result = sg2_quality.process_frame(frame, [detection], self.sg2_cfg,
                                                   frame_id=frame_id)["results"][0]
                if not isinstance(result, dict) or not isinstance(result.get("accept"), (bool, np.bool_)):
                    raise ValueError("SG2 returned a malformed result")
            except Exception:
                result = {"accept": False, "reason": "preprocessing_failure", "quality": 0.0,
                          "metrics": {}, "face": None, "pipeline_error": True}
            result.update(face_index=index, frame_id=metadata["frame_id"], detection_id=index,
                          timestamp=metadata["timestamp"], bbox=list(detection["box"]))
            results.append(result)
        return {"frame_id": metadata["frame_id"], "timestamp": metadata["timestamp"],
                "detector": "temporary_sg3_yunet", "detections": detections,
                "results": results, "detection_ms": detection_ms,
                "preprocessing_ms": (time.perf_counter() - start) * 1000.0}
