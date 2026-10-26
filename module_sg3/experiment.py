"""Reproducible local enrollment, reload verification, and SG3 experiments.

Both methods consume the exact same actual SG2-accepted crops. Independent
evaluation and enrollment replay are reported separately. Undefined rates are
JSON null, never zero. The threshold sweep is descriptive, not calibration.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from .api import RecognitionService
from .download_models import sha256
from .mock_sg4 import consume_result
from .pipeline import YuNetSG2Pipeline
from .sg3_core import load_config, make_recognizer, read_image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def statistics(values):
    values = list(values)
    if not values:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95))}


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows, threshold, method):
    """Rates condition on scored faces; quality failures remain separate."""
    scored = [r for r in rows if r["status"] in ("RECOGNIZED", "UNKNOWN")]
    known = [r for r in scored if r["role"] != "evaluation_unknown"]
    unknown = [r for r in scored if r["role"] == "evaluation_unknown"]

    def prediction(row):
        accept = row["score"] >= threshold if method == "sface" else row["score"] <= threshold
        return row["best_match"] if accept else "UNKNOWN"

    ta = sum(prediction(r) == r["true_id"] for r in known)
    fr = sum(prediction(r) == "UNKNOWN" for r in known)
    misid = len(known) - ta - fr
    fa = sum(prediction(r) != "UNKNOWN" for r in unknown)
    tr = len(unknown) - fa
    return {"threshold": threshold, "input_samples": len(rows), "scored_known": len(known),
            "scored_unknown": len(unknown), "correct_identifications": ta, "false_rejections": fr,
            "misidentifications": misid, "false_acceptances": fa, "unknown_rejections": tr,
            "conditional_identification_accuracy": rate(ta, len(known)),
            "conditional_FAR": rate(fa, len(unknown)), "conditional_FRR": rate(fr, len(known)),
            "conditional_misidentification_rate": rate(misid, len(known)),
            "conditional_unknown_rejection_rate": rate(tr, len(unknown)),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "pipeline_acceptance_rate": rate(len(scored), len(rows))}


def select_subject(packet):
    """Largest subject only when dominant; never guess labels among peers."""
    results = sorted(packet["results"], key=lambda r: r["bbox"][2] * r["bbox"][3], reverse=True)
    if not results:
        return {"accept": False, "reason": "no_face_detected", "quality": 0,
                "metrics": {}, "face": None, "pipeline_error": True}
    if len(results) > 1:
        area = lambda r: r["bbox"][2] * r["bbox"][3]
        if area(results[0]) < 1.5 * area(results[1]):
            return {"accept": False, "reason": "ambiguous_subject", "quality": 0,
                    "metrics": {}, "face": None, "pipeline_error": True}
    return results[0]


def prepare_faces(manifest_path, pipeline):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sg3.dataset.v1":
        raise ValueError("Unsupported dataset manifest")
    prepared, timings = [], {"detection": [], "preprocessing": []}
    for item in manifest["items"]:
        if item["split"] not in ("enrollment", "evaluation_known", "evaluation_unknown"):
            continue
        image_path = (manifest_path.parent / item["image_path"]).resolve()
        if not image_path.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("Manifest image path escapes prepared directory")
        try:
            frame = read_image(str(image_path))
            packet = pipeline.process(frame, frame_id=item["sample_id"])
            face = select_subject(packet)
            timings["detection"].append(packet["detection_ms"])
            timings["preprocessing"].append(packet["preprocessing_ms"])
        except (ValueError, OSError, cv2.error):
            face = {"accept": False, "reason": "input_or_detector_failure", "quality": 0,
                    "metrics": {}, "face": None, "pipeline_error": True}
        prepared.append({"item": item, "sg2": face})
    return manifest, prepared, timings


class TimedRecognizer:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.last_embedding_ms = None
        self.last_matching_ms = None

    def __getattr__(self, name):
        return getattr(self.recognizer, name)

    def match(self, face, row=None):
        start = time.perf_counter()
        if self.recognizer.method == "sface":
            embedding = self.recognizer.embed(face, row)
            self.last_embedding_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            result = self.recognizer.match_embedding(embedding)
        else:
            self.last_embedding_ms = None
            result = self.recognizer.match(face, row)
        self.last_matching_ms = (time.perf_counter() - start) * 1000
        return result


def enroll_prepared(prepared, cfg, methods, manifest_path, pipeline):
    samples = [(r["item"]["student_id"], r["sg2"]["face"], None) for r in prepared
               if r["item"]["split"] == "enrollment" and r["sg2"].get("accept")]
    if not samples:
        raise ValueError("No accepted enrollment images; inspect quality gates and source images")
    recognizers, report = {}, {}
    provenance = {"schema_version": "sg3.gallery.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": sha256(manifest_path), "sg2_config_sha256": sha256(pipeline.sg2_config_path),
                  "sg2_config": pipeline.sg2_cfg, "config": cfg,
                  "models": {p.name: sha256(p) for p in Path(cfg["models_dir"]).glob("*.onnx")},
                  "sample_ids": [r["item"]["sample_id"] for r in prepared
                                 if r["item"]["split"] == "enrollment" and r["sg2"].get("accept")],
                  "student_ids": sorted({s[0] for s in samples}), "methods": {}}
    for method in methods:
        rec = make_recognizer(method, cfg)
        start = time.perf_counter()
        rec.fit(samples)
        fit_ms = (time.perf_counter() - start) * 1000
        path = rec.save()
        reference = rec.match(samples[0][1])
        reloaded = make_recognizer(method, cfg)
        reloaded.load()
        repeated = reloaded.match(samples[0][1])
        if reference[0] != repeated[0] or not np.isclose(reference[1], repeated[1], atol=1e-5):
            raise RuntimeError(f"{method} persistence verification failed")
        report[method] = {"students": len({s[0] for s in samples}), "templates": len(samples),
                          "fit_ms": fit_ms, "gallery_bytes": Path(path).stat().st_size,
                          "gallery_reload_verified": True, "gallery_sha256": sha256(path)}
        provenance["methods"][method] = report[method]
        recognizers[method] = reloaded
    write_json(Path(cfg["gallery_dir"]) / "provenance.private.json", provenance)
    return recognizers, report


def evaluate_prepared(prepared, rec):
    timed = TimedRecognizer(rec)
    service = RecognitionService(timed)
    rows = []
    for sample in prepared:
        item, face = sample["item"], sample["sg2"]
        timed.last_embedding_ms = timed.last_matching_ms = None
        result = service.recognize_face(face, {"frame_id": item["sample_id"]})
        consume_result(result)  # Verify the documented mock downstream contract.
        rows.append({**result, "sample_id": item["sample_id"], "true_id": item["student_id"],
                     "role": "enrollment_replay" if item["split"] == "enrollment" else item["split"],
                     "pose_hint": item.get("pose_hint"), "embedding_ms": timed.last_embedding_ms,
                     "matching_ms": timed.last_matching_ms,
                     "preprocessing_reason": face.get("reason")})
    return rows


def stress_demo(prepared, manifest_path, pipeline, rec):
    candidates = [r for r in prepared if r["item"]["split"] == "enrollment" and r["sg2"].get("accept")]
    if not candidates:
        return []
    selected = candidates[0]
    image = read_image(str(Path(manifest_path).parent / selected["item"]["image_path"]))
    # Downscale once and detect again so the recorded boxes remain meaningful.
    scale = min(1.0, 1280 / max(image.shape[:2]))
    image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    original = select_subject(pipeline.process(image))
    if not original.get("accept"):
        return []
    occluded = image.copy()
    x, y, w, h = [round(v) for v in original["bbox"]]
    cv2.rectangle(occluded, (max(0, x), max(0, y + h // 2)),
                  (min(image.shape[1] - 1, x + w), min(image.shape[0] - 1, y + h)), (0, 0, 0), -1)
    variants = {"gaussian_blur_sigma8": cv2.GaussianBlur(image, (0, 0), 8),
                "darken_0.2": (image.astype(np.float32) * .2).astype(np.uint8),
                "lower_face_occlusion": occluded}
    service, rows = RecognitionService(rec), []
    for label, variant in variants.items():
        result = service.recognize_face(select_subject(pipeline.process(variant)))
        rows.append({"variant": label, "status": result["status"], "reason": result["reason"],
                     "score": result["score"], "correct_identity": result["student_id"] == selected["item"]["student_id"],
                     "preprocessing_metrics": result["quality_metrics"],
                     "note": "Synthetic transformation of enrollment image; diagnostic only, not evaluation"})
    return rows


def plot_sweeps(report, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(destination).parent / ".mplcache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(report["methods"]), figsize=(6 * len(report["methods"]), 4), squeeze=False)
    for ax, (method, data) in zip(axes[0], report["methods"].items()):
        sweep = data["threshold_sweep"]
        ax.plot([s["threshold"] for s in sweep],
                [s["evaluation"]["conditional_FAR"] for s in sweep], "o-", label="Held-out identity FAR")
        ax.plot([s["threshold"] for s in sweep],
                [s["enrollment_replay"]["conditional_identification_accuracy"] for s in sweep],
                "s--", label="Enrollment replay recall (training)")
        ax.axvline(data["default_threshold"], color="gray", linestyle=":")
        ax.set(title=method.upper(), xlabel="Raw score threshold", ylabel="Rate", ylim=(-.05, 1.05))
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle("Exploratory thresholds" + ("; no independent known-student test set"
                  if not report["known_evaluation_available"] else "; independent sessions held out"))
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run_experiment(manifest_path, cfg, output_dir, *, methods=("sface", "lbph"), max_side=1280, sg2_config_path=None):
    start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Experimental enrollment must never replace the active classroom gallery.
    cfg = copy.deepcopy(cfg)
    cfg["gallery_dir"] = str(output_dir.resolve() / "gallery")
    pipeline = YuNetSG2Pipeline(cfg, sg2_config_path=sg2_config_path, max_side=max_side)
    prep_start = time.perf_counter()
    manifest, prepared, timings = prepare_faces(manifest_path, pipeline)
    prep_ms = (time.perf_counter() - prep_start) * 1000
    recognizers, enrollment = enroll_prepared(prepared, cfg, methods, manifest_path, pipeline)
    enrolled_ids = {r["item"]["student_id"] for r in prepared
                    if r["item"]["split"] == "enrollment" and r["sg2"].get("accept")}
    if any(r["item"]["split"] == "evaluation_known" and r["item"]["student_id"] not in enrolled_ids
           for r in prepared):
        raise ValueError("A known-evaluation identity has no accepted enrollment samples; revise enrollment before evaluating")
    report = {"schema_version": "sg3.experiment.v1", "generated_utc": datetime.now(timezone.utc).isoformat(), "dataset_audit": manifest["audit"],
              "protocol": "Whole held-out identities for UNKNOWN; distinct sessions required for known evaluation. Enrollment replay is training data, not accuracy evidence.",
              "known_evaluation_available": bool(manifest["audit"]["session_independent_known_evaluation_available"]),
              "threshold_policy": "Existing defaults retained. Sweeps describe this small sample; no tuned or calibrated production threshold claimed.",
              "preprocessing": {"detector": "temporary_sg3_yunet", "max_side": max_side,
                                "sg2_config_sha256": sha256(pipeline.sg2_config_path), "same_crops_for_both_methods": True,
                                "subject_selection": "largest face, at least 1.5x runner-up area; otherwise ambiguous/error",
                                "preparation_ms_including_decode": prep_ms,
                                "detection": statistics(timings["detection"]), "sg2": statistics(timings["preprocessing"]),
                                "sample_status_counts": dict(Counter("accepted" if r["sg2"].get("accept") else r["sg2"].get("reason") for r in prepared))},
              "methods": {}, "limitations": [
                 "Withheld students represent a small open-set test, not all unknown populations.",
                 "No labeled CCTV identities; CCTV accuracy cannot be measured.",
                 "CPU Windows measurements only; Jetson, full SG4 and attendance untested."]}
    if not report["known_evaluation_available"]:
        report["limitations"].insert(0, "No session-independent known-student samples; identification accuracy and FRR cannot be estimated.")
    for method, rec in recognizers.items():
        rec.match(next(r["sg2"]["face"] for r in prepared if r["sg2"].get("accept")))
        rows = evaluate_prepared(prepared, rec)
        write_json(output_dir / f"{method}_records.private.json", rows)
        evaluation = [r for r in rows if r["role"] != "enrollment_replay"]
        replay = [r for r in rows if r["role"] == "enrollment_replay"]
        scored = [r for r in rows if r["status"] in ("RECOGNIZED", "UNKNOWN")]
        report["methods"][method] = {"score_type": rec.score_type, "default_threshold": rec.threshold,
            "enrollment": enrollment[method], "evaluation": metrics(evaluation, rec.threshold, method),
            "enrollment_replay": metrics(replay, rec.threshold, method),
            "timing": {"recognition": statistics(r["recognition_ms"] for r in scored),
                       "embedding": statistics(r["embedding_ms"] for r in scored if r["embedding_ms"] is not None),
                       "matching_including_lbph_preprocessing": statistics(r["matching_ms"] for r in scored),
                       "note": "Warm model, sequential CPU wall time. Face loading/detection/SG2 excluded from recognition timing."},
            "threshold_sweep": [{"threshold": t, "evaluation": metrics(evaluation, t, method),
                                 "enrollment_replay": metrics(replay, t, method)} for t in cfg[method]["sweep"]],
            "enrollment_pose_replay": {p: metrics([r for r in replay if r["pose_hint"] == p], rec.threshold, method)
                                       for p in sorted({r["pose_hint"] for r in replay if r["pose_hint"]})},
            "synthetic_stress": stress_demo(prepared, manifest_path, pipeline, rec)}
    report["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                         "processor": platform.processor(), "opencv_threads": cv2.getNumThreads(),
                         "packages": {n: importlib.metadata.version(n) for n in ("opencv-contrib-python", "numpy", "Pillow", "pillow-heif", "matplotlib")}}
    report["total_wall_seconds"] = time.perf_counter() - start
    report["processed_samples_per_wall_second"] = len(prepared) / report["total_wall_seconds"]
    write_json(output_dir / "aggregate_results.json", report)
    plot_sweeps(report, output_dir / "thresholds.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", default="module_sg3/private_runs/experiment")
    parser.add_argument("--methods", nargs="+", choices=("sface", "lbph"), default=("sface", "lbph"))
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--sg2-config")
    args = parser.parse_args()
    report = run_experiment(args.manifest, load_config(args.config), args.output_dir,
                            methods=args.methods, max_side=args.max_side, sg2_config_path=args.sg2_config)
    print(json.dumps({method: {"enrollment": data["enrollment"], "evaluation": data["evaluation"],
                               "enrollment_replay": data["enrollment_replay"], "timing": data["timing"]}
                      for method, data in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
