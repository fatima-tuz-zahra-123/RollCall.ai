"""Strict SG2 -> SG3 callable boundary; recognition is frame-level evidence only."""
from __future__ import annotations

from collections.abc import Mapping
import math
import numbers
import time

import numpy as np

STATUSES = ("RECOGNIZED", "UNKNOWN", "QUALITY_REJECTED", "ERROR")
METADATA_FIELDS = ("frame_id", "detection_id", "timestamp", "bbox")


def finite_number(value):
    """Return a builtin float; reject booleans, strings, infinity and NaN."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError("expected a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def json_value(value):
    """Copy JSON-shaped metadata, rejecting unsupported/nonfinite values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return finite_number(value)
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("metadata keys must be strings")
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    raise ValueError("metadata must contain JSON-compatible values")


def validate_metadata(metadata):
    """Validate association fields; face_id aliases the frame-local detection_id."""
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    out = {key: None for key in METADATA_FIELDS}
    for key in ("frame_id", "detection_id"):
        value = metadata.get(key)
        if key == "detection_id" and "face_id" in metadata:
            if key in metadata and value != metadata["face_id"]:
                raise ValueError("face_id and detection_id disagree")
            value = metadata["face_id"]
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, numbers.Integral)):
                raise ValueError(f"{key} must be a string or integer")
            value = int(value) if isinstance(value, numbers.Integral) else value
        out[key] = value
    if metadata.get("timestamp") is not None:
        out["timestamp"] = finite_number(metadata["timestamp"])
        if out["timestamp"] < 0:
            raise ValueError("timestamp must be nonnegative seconds")
    box = metadata.get("bbox")
    if box is not None:
        if not isinstance(box, (list, tuple, np.ndarray)) or np.shape(box) != (4,):
            raise ValueError("bbox must be [x,y,width,height]")
        out["bbox"] = [finite_number(v) for v in box]
        if out["bbox"][2] <= 0 or out["bbox"][3] <= 0:
            raise ValueError("bbox width and height must be positive")
    return out


def _normalize_face_packet(packet):
    """Accept incoming SG2 packets and the repository's older SG2 result shape.

    Duplicate aliases must agree. A positive quality decision requires one
    complete quality/image pair; crossed partial payloads are malformed.
    """
    gates = [packet[key] for key in ("accept", "quality_pass") if key in packet]
    if not gates or not all(isinstance(value, (bool, np.bool_)) for value in gates):
        raise ValueError("accept/quality_pass must be boolean")
    if any(bool(value) != bool(gates[0]) for value in gates[1:]):
        raise ValueError("accept and quality_pass disagree")
    if "face" in packet and "aligned_face" in packet:
        face, aligned_face = packet["face"], packet["aligned_face"]
        same = face is aligned_face or (
            isinstance(face, np.ndarray) and isinstance(aligned_face, np.ndarray)
            and face.shape == aligned_face.shape and face.dtype == aligned_face.dtype
            and np.array_equal(face, aligned_face))
        if not same:
            raise ValueError("face and aligned_face disagree")
    accepted = bool(gates[0])
    if accepted and not (("accept" in packet and "face" in packet)
                         or ("quality_pass" in packet and "aligned_face" in packet)):
        raise ValueError("accepted packet needs a complete quality/image pair")
    return accepted, packet.get("aligned_face", packet.get("face"))


class RecognitionService:
    """Wrap an existing sg3_core SFace/LBPH recognizer, without changing it.

    The recognizer must already be enrolled/loaded. Scores are raw similarity or
    distance measurements, never probabilities. Invalid input returns ERROR.
    """

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.method = recognizer.method
        self.score_type = recognizer.score_type
        self.threshold = finite_number(recognizer.threshold)
        if self.method not in ("sface", "lbph"):
            raise ValueError("unsupported recognition method")
        expected = "cosine_similarity" if self.method == "sface" else "lbph_distance"
        if self.score_type != expected:
            raise ValueError("score_type does not match recognition method")
        if self.method == "sface" and not -1 <= self.threshold <= 1:
            raise ValueError("cosine threshold must be in [-1,1]")
        if self.method == "lbph" and self.threshold < 0:
            raise ValueError("LBPH threshold must be nonnegative")

    def recognize_face(self, sg2_result, metadata=None):
        """Consume one SG2 packet/result and return a JSON-safe SG3 V1 result.

        Incoming packet aliases: quality_pass, aligned_face, face_id. The older
        accept/face/detection_id shape remains supported. No pixel arrays or
        embeddings leave this API; absent geometry/time is never invented.
        """
        start = time.perf_counter()
        out = {
            "schema_version": "sg3.v1", "status": "ERROR", "method": self.method,
            "score_type": self.score_type, "threshold": self.threshold,
            "student_id": None, "best_match": None, "score": None,
            **{key: None for key in METADATA_FIELDS},
            "face_id": None,
            "quality": None, "quality_metrics": {}, "reason": "invalid_input",
            "recognition_ms": 0.0,
        }

        def finish():
            out["face_id"] = out["detection_id"]
            out["recognition_ms"] = (time.perf_counter() - start) * 1000.0
            return out

        if not isinstance(sg2_result, Mapping):
            return finish()
        try:
            accepted, face = _normalize_face_packet(sg2_result)
            association_fields = (*METADATA_FIELDS, "face_id")
            merged = {key: sg2_result[key] for key in association_fields if key in sg2_result}
            original_metadata = validate_metadata(merged)
            if metadata is not None:
                if not isinstance(metadata, Mapping):
                    raise ValueError("metadata must be a mapping")
                override = validate_metadata(metadata)
                if ("face_id" in sg2_result and any(key in metadata for key in ("face_id", "detection_id"))
                        and override["detection_id"] != original_metadata["detection_id"]):
                    raise ValueError("metadata must preserve the incoming face_id")
                merged.update({key: metadata[key] for key in association_fields if key in metadata})
            out.update(validate_metadata(merged))
            quality = sg2_result.get("quality")
            if quality is not None:
                quality = finite_number(quality)
                if not 0 <= quality <= 1:
                    raise ValueError("quality must be in [0,1]")
            metrics = sg2_result.get("metrics", {})
            if not isinstance(metrics, Mapping):
                raise ValueError("metrics must be a mapping")
            out["quality"] = quality
            out["quality_metrics"] = json_value(metrics)
            if sg2_result.get("pipeline_error"):
                out["reason"] = "preprocessing_failure"
                return finish()
            if not accepted:
                reason = sg2_result.get("reason")
                if reason is not None and not isinstance(reason, str):
                    raise ValueError("reason must be a string or null")
                out.update(status="QUALITY_REJECTED", reason=reason or "quality_rejected")
                return finish()
            if sg2_result.get("reason") is not None:
                raise ValueError("accepted SG2 output must have a null reason")
            if not isinstance(face, np.ndarray) or face.shape != (112, 112, 3) or face.dtype != np.uint8:
                raise ValueError("accepted face must be 112x112x3 uint8 BGR")
        except (ValueError, TypeError, OverflowError, RecursionError):
            # Only previously validated, JSON-safe fields have been copied.
            out["reason"] = "invalid_input"
            return finish()

        try:
            sid, score = self.recognizer.match(face, None)
            if not isinstance(sid, str) or not sid or sid == "UNKNOWN":
                out["reason"] = "empty_or_invalid_gallery"
                return finish()
            score = finite_number(score)
            if self.method == "sface":
                if not -1.000001 <= score <= 1.000001:
                    raise ValueError("invalid cosine score")
                score = min(1.0, max(-1.0, score))  # floating-point roundoff only
            elif score < 0:
                raise ValueError("invalid LBPH distance")
            known = score >= self.threshold if self.method == "sface" else score <= self.threshold
            out.update(status="RECOGNIZED" if known else "UNKNOWN",
                       student_id=sid if known else "UNKNOWN", best_match=sid,
                       score=score, reason=None if known else (
                           "below_threshold" if self.method == "sface" else "above_threshold"))
        except Exception:
            # Model/OpenCV errors are recognition failures, never UNKNOWN evidence.
            out.update(status="ERROR", reason="recognition_failure")
        return finish()
