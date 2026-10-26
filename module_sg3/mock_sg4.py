"""Strict downstream contract consumer. This mock does NOT track or attend."""
from __future__ import annotations

from collections.abc import Mapping

from .api import STATUSES, finite_number, json_value, validate_metadata


def consume_result(result):
    """Validate a SG3 V1 result and return an xyxy identity-evidence record.

    This is an integration example only. It never imports or claims to run SG4's
    LFW stand-in, which has no compatible SG3 ingestion/tracking interface.
    """
    if not isinstance(result, Mapping) or result.get("schema_version") != "sg3.v1":
        raise ValueError("expected a sg3.v1 mapping")
    required = {"status", "method", "score_type", "threshold", "student_id", "best_match",
                "score", "frame_id", "detection_id", "timestamp", "bbox", "quality",
                "quality_metrics", "reason", "recognition_ms"}
    if not required.issubset(result):
        raise ValueError("SG3 result is missing required fields")
    json_value(dict(result))  # disallow arrays, NaN and Infinity everywhere
    metadata = validate_metadata(result)
    status = result["status"]
    if status not in STATUSES:
        raise ValueError("invalid recognition status")
    method, score_type = result["method"], result["score_type"]
    expected_score_type = {"sface": "cosine_similarity", "lbph": "lbph_distance"}
    if method not in expected_score_type or score_type != expected_score_type[method]:
        raise ValueError("invalid method or score type")
    threshold = finite_number(result["threshold"])
    if (method == "sface" and not -1 <= threshold <= 1) or (method == "lbph" and threshold < 0):
        raise ValueError("invalid threshold")
    if finite_number(result["recognition_ms"]) < 0:
        raise ValueError("negative recognition_ms")
    if result["quality"] is not None and not 0 <= finite_number(result["quality"]) <= 1:
        raise ValueError("invalid quality")
    if not isinstance(result["quality_metrics"], Mapping):
        raise ValueError("invalid quality_metrics")
    sid, candidate, score = result["student_id"], result["best_match"], result["score"]
    if status in ("RECOGNIZED", "UNKNOWN"):
        score = finite_number(score)
        if not isinstance(candidate, str) or not candidate or candidate == "UNKNOWN":
            raise ValueError("identity evidence needs a best match")
        if method == "sface" and not -1 <= score <= 1 or method == "lbph" and score < 0:
            raise ValueError("invalid score range")
        accepted = score >= threshold if method == "sface" else score <= threshold
        if status == "RECOGNIZED" and (sid != candidate or not accepted):
            raise ValueError("inconsistent recognized identity")
        if status == "UNKNOWN" and (sid != "UNKNOWN" or accepted):
            raise ValueError("inconsistent UNKNOWN identity")
    elif any(value is not None for value in (sid, candidate, score)):
        raise ValueError("rejected/error results must not assign identity evidence")
    box = metadata["bbox"]
    return {**metadata, "bbox_xyxy": None if box is None else [box[0], box[1], box[0]+box[2], box[1]+box[3]],
            "id": sid, "status": status, "score": score, "score_type": score_type,
            "quality": result["quality"], "tracking": False}


def consume_frame(results):
    """Empty results -> empty records. Validation failures raise ValueError."""
    if not isinstance(results, (list, tuple)):
        raise ValueError("results must be a list")
    return [consume_result(result) for result in results]
