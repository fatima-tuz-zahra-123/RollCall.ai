"""Enroll an explicitly enrollment-only manifest without inventing evaluation.

Run from the repository root::

    python -m module_sg3.enroll_dataset --manifest private_data/prepared/manifest.json \
        --gallery-dir module_sg3/gallery_class --output-dir module_sg3/private_runs/class_enrollment

All images, student lookup tables and templates stay local. Every SG2-accepted
enrollment photograph produces a separate SFace template. Outputs are prepared
in sibling staging directories, verified after reload, and then renamed into
place. Existing nonempty destinations require --replace and are backed up.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import uuid

import cv2
import numpy as np

from .download_models import sha256
from .experiment import prepare_faces, statistics, write_json
from .pipeline import YuNetSG2Pipeline
from .sg3_core import load_config, make_recognizer


def validate_enrollment_manifest(manifest):
    """Refuse evaluation manifests even if they happen to contain enrollment rows."""
    if manifest.get("schema_version") != "sg3.dataset.v1":
        raise ValueError("Unsupported dataset manifest")
    if manifest.get("mode") != "enrollment_only":
        raise ValueError("Enrollment requires an explicit manifest mode='enrollment_only'")
    if not isinstance(manifest.get("items"), list) or not isinstance(manifest.get("students"), dict):
        raise ValueError("Manifest items/students must be a list and dictionary")
    sample_ids = set()
    for item in manifest["items"]:
        sid, role = item.get("student_id"), item.get("split")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("Manifest sample IDs must be nonempty and unique")
        sample_ids.add(sample_id)
        if role in ("evaluation_known", "evaluation_unknown"):
            raise ValueError("Evaluation records must never enter enrollment-only processing")
        if role not in ("enrollment", "excluded", "unlabeled"):
            raise ValueError("Unexpected record role in enrollment-only manifest")
        if item.get("error") and role != "excluded":
            raise ValueError("Undecodable records must be explicitly excluded")
        if sid is not None:
            if not isinstance(sid, str) or not sid or sid == "UNKNOWN" or sid not in manifest["students"]:
                raise ValueError("Every labeled record must have a declared, valid student identifier")
            if role not in ("enrollment", "excluded"):
                raise ValueError("Every usable labeled record must be enrollment")
        elif role == "enrollment":
            raise ValueError("Unlabeled images cannot be enrolled")
        if role == "enrollment" and not isinstance(item.get("image_path"), str):
            raise ValueError("Enrollment images require a prepared relative image_path")
    for sid, student in manifest["students"].items():
        if not isinstance(sid, str) or not sid or sid == "UNKNOWN" or not isinstance(student, dict):
            raise ValueError("Invalid manifest student roster")
    return manifest


def _check_destination(destination, manifest_path, replace):
    destination = Path(destination).expanduser().resolve()
    prepared_root = manifest_path.parent
    repository_root = Path(__file__).resolve().parents[1]
    if (destination == repository_root or destination == Path(__file__).resolve().parent
            or destination.parent == destination or prepared_root.is_relative_to(destination)
            or destination.is_relative_to(prepared_root)):
        raise ValueError("Output destination must be separate from the prepared data and repository root")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Destination must be a directory: {destination}")
        if any(destination.iterdir()) and not replace:
            raise FileExistsError(f"Destination is nonempty: {destination}; use a new directory or explicit --replace")
    return destination


def _new_stage(destination, token):
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.staging-{token}")
    stage.mkdir()
    return stage


def _publish(stage, destination, token, replace):
    """Rename within one parent; retain every previous destination as a backup."""
    if stage.parent.resolve() != destination.parent.resolve():
        raise ValueError("Staging and final destination must be sibling directories")
    backup = None
    if destination.exists():
        if any(destination.iterdir()) and not replace:
            raise FileExistsError("Destination changed during enrollment; publication refused")
        backup = destination.with_name(f"{destination.name}.backup-{token}")
        destination.rename(backup)
    try:
        stage.rename(destination)
    except Exception:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
    return backup


def _excluded_reason(item):
    return item.get("exclusion_reason") or (f"decode_error:{item['error']}" if item.get("error") else "excluded")


def _sample_decision(sample):
    item, face = sample["item"], sample["sg2"]
    accepted = bool(face.get("accept"))
    status = "ACCEPTED_FOR_ENROLLMENT" if accepted else ("ERROR" if face.get("pipeline_error") else "QUALITY_REJECTED")
    return {"sample_id": item["sample_id"], "student_id": item["student_id"],
            "source": item.get("source"), "source_sha256": item.get("sha256"),
            "image_path": item.get("image_path"), "capture_session": item.get("capture_session"),
            "pose_hint": item.get("pose_hint"), "status": status,
            "accepted": accepted, "reason": face.get("reason"), "quality": face.get("quality"),
            "quality_metrics": face.get("metrics", {}), "bbox": face.get("bbox")}


def _fit_and_verify(method, samples, cfg, factory):
    rec = factory(method, cfg)
    embedding_ms = []
    original_embed = getattr(rec, "embed", None)
    if method == "sface":
        def timed_embed(image, row=None):
            start = time.perf_counter()
            vector = original_embed(image, row)
            embedding_ms.append((time.perf_counter() - start) * 1000)
            return vector
        rec.embed = timed_embed
    start = time.perf_counter()
    try:
        rec.fit(samples)
    finally:
        if method == "sface":
            rec.embed = original_embed
    fit_ms = (time.perf_counter() - start) * 1000
    path = Path(rec.save())
    reload_start = time.perf_counter()
    loaded = factory(method, cfg)
    loaded.load()
    queries = 0
    if method == "sface":
        if set(rec.gallery) != set(loaded.gallery):
            raise RuntimeError("SFace student keys changed after gallery reload")
        for sid, vectors in rec.gallery.items():
            restored = loaded.gallery[sid]
            if not np.array_equal(vectors, restored):
                raise RuntimeError("SFace template values changed after gallery reload")
            if (vectors.ndim != 2 or vectors.shape[1] != 128 or vectors.dtype != np.float32
                    or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)):
                raise RuntimeError("SFace templates must be finite normalized float32 128-D vectors")
            for vector in vectors:
                before = rec.match_embedding(vector)
                after = loaded.match_embedding(vector)
                if before[0] != after[0] or not np.isclose(before[1], after[1], atol=1e-6, rtol=0):
                    raise RuntimeError("SFace query result changed after gallery reload")
                queries += 1
        if queries != len(samples):
            raise RuntimeError("SFace template count does not equal accepted image count")
    else:
        if rec.labels != loaded.labels:
            raise RuntimeError("LBPH student labels changed after reload")
        for _, image, row in samples:
            before, after = rec.match(image, row), loaded.match(image, row)
            if before[0] != after[0] or not np.isclose(before[1], after[1], atol=1e-5, rtol=0):
                raise RuntimeError("LBPH query result changed after gallery reload")
            queries += 1
    files = [path]
    if method == "lbph":
        files.append(path.parent / "lbph_labels.json")
    report = {"students": len({s[0] for s in samples}), "templates": len(samples),
              "fit_ms": fit_ms, "embedding": statistics(embedding_ms),
              "reload_verification_ms": (time.perf_counter() - reload_start) * 1000,
              "reload_verified_queries": queries, "gallery_reload_verified": True,
              "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}}
    return loaded, report, embedding_ms


def enroll_dataset(manifest_path, cfg, output_dir="module_sg3/private_runs/class_enrollment", *,
                   gallery_dir=None, methods=("sface",), sg2_config_path=None, max_side=1280,
                   replace=False, require_all_students=False, pipeline=None,
                   recognizer_factory=None):
    """Enroll accepted labeled images, verify persistence, then publish directories.

    This creates no evaluation split, UNKNOWN holdout, accuracy or threshold sweep.
    Failure preserves prior galleries and diagnostic staging artifacts. A caller
    can explicitly require coverage of every declared student before publication.
    """
    start = time.perf_counter()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = validate_enrollment_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(m not in ("sface", "lbph") for m in methods):
        raise ValueError("methods must contain distinct sface/lbph choices")
    gallery_dir = _check_destination(gallery_dir or cfg["gallery_dir"], manifest_path, replace)
    output_dir = _check_destination(output_dir, manifest_path, replace)
    if gallery_dir.is_relative_to(output_dir) or output_dir.is_relative_to(gallery_dir):
        raise ValueError("Gallery and report destinations must be separate directories")
    if pipeline is None:
        pipeline = YuNetSG2Pipeline(cfg, sg2_config_path=sg2_config_path, max_side=max_side)
    factory = recognizer_factory or make_recognizer
    prep_start = time.perf_counter()
    checked_manifest, prepared, preprocessing_times = prepare_faces(manifest_path, pipeline)
    # Detect any rewrite of the manifest between validation and image loading.
    if checked_manifest != manifest:
        raise ValueError("Manifest changed during enrollment preparation")
    preparation_ms = (time.perf_counter() - prep_start) * 1000
    decisions = [_sample_decision(sample) for sample in prepared]
    accepted = [sample for sample in prepared if sample["sg2"].get("accept")]
    samples = [(sample["item"]["student_id"], sample["sg2"]["face"], None) for sample in accepted]
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
    gallery_stage = _new_stage(gallery_dir, token)
    output_stage = _new_stage(output_dir, token)
    write_json(output_stage / "image_decisions.private.json", decisions)
    by_student = defaultdict(list)
    for row in decisions:
        by_student[row["student_id"]].append(row)
    roster, lookup = {}, {}
    for sid, student in sorted(manifest["students"].items()):
        source_items = [r for r in manifest["items"] if r.get("student_id") == sid]
        rows = by_student[sid]
        count = sum(r["accepted"] for r in rows)
        roster[sid] = {"student_id": sid, "source_label": student.get("source_label"),
                       "labeled_images": len(source_items), "candidate_images": len(rows),
                       "accepted_images": count, "template_count": count,
                       "has_templates": bool(count), "rejected_images": len(rows) - count,
                       "rejection_reasons": dict(Counter(r["reason"] for r in rows if not r["accepted"])),
                       "excluded_images": sum(r.get("split") == "excluded" for r in source_items),
                       "excluded_reasons": dict(Counter(_excluded_reason(r) for r in source_items if r.get("split") == "excluded")),
                       "accepted_pose_hints": dict(Counter(r["pose_hint"] or "unspecified" for r in rows if r["accepted"]))}
        lookup[sid] = {"source_label": student.get("source_label")}
    write_json(output_stage / "roster.private.json", roster)
    missing = [sid for sid, row in roster.items() if not row["has_templates"]]
    if not samples:
        raise ValueError(f"No SG2-accepted enrollment images; diagnostic decisions preserved in {output_stage}")
    if require_all_students and missing:
        raise ValueError(f"{len(missing)} students have no accepted enrollment images; diagnostic roster preserved in {output_stage}")
    staged_cfg = dict(cfg, gallery_dir=str(gallery_stage))
    template_rows, counters = [], Counter()
    for sample in accepted:
        item = sample["item"]
        sid = item["student_id"]
        template_rows.append({"student_id": sid, "gallery_row": counters[sid],
                              "sample_id": item["sample_id"], "source": item.get("source"),
                              "source_sha256": item.get("sha256"), "image_path": item.get("image_path"),
                              "capture_session": item.get("capture_session"), "pose_hint": item.get("pose_hint"),
                              "quality": sample["sg2"].get("quality"), "quality_metrics": sample["sg2"].get("metrics", {})})
        counters[sid] += 1
    model_names = {cfg["detector"]["model"]}
    if "sface" in methods:
        model_names.add(cfg["sface"]["model"])
    model_hashes = {name: sha256(Path(cfg["models_dir"]) / name) for name in sorted(model_names)}
    method_reports, per_sample_embedding = {}, {}
    for method in methods:
        _, method_reports[method], embedding_times = _fit_and_verify(method, samples, staged_cfg, factory)
        if method == "sface":
            per_sample_embedding = {sample["item"]["sample_id"]: value for sample, value in zip(accepted, embedding_times)}
    for row in decisions:
        row["embedding_ms"] = per_sample_embedding.get(row["sample_id"])
    write_json(output_stage / "image_decisions.private.json", decisions)
    provenance = {"schema_version": "sg3.gallery.v1", "mode": "enrollment_only",
                  "generated_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": sha256(manifest_path),
                  "manifest_source_root": manifest.get("source_root"),
                  "sg2_config_sha256": sha256(pipeline.sg2_config_path), "sg2_config": pipeline.sg2_cfg,
                  "config": dict(cfg, gallery_dir=str(gallery_dir)), "models": model_hashes,
                  "sample_ids": [r["sample_id"] for r in template_rows],
                  "student_ids": sorted(counters), "template_rows": template_rows,
                  "methods": method_reports,
                  "note": "Enrollment templates and source linkage are private biometric data; reload checks are persistence tests, not recognition accuracy."}
    for filename, value in (("provenance.private.json", provenance), ("student_lookup.private.json", lookup),
                            ("roster.private.json", roster)):
        write_json(gallery_stage / filename, value)
    pose_reports = {}
    for pose in sorted({r["pose_hint"] or "unspecified" for r in decisions}):
        rows = [r for r in decisions if (r["pose_hint"] or "unspecified") == pose]
        pose_reports[pose] = {"candidate_images": len(rows), "accepted_images": sum(r["accepted"] for r in rows),
                              "rejection_reasons": dict(Counter(r["reason"] for r in rows if not r["accepted"])),
                              "embedding": statistics(r["embedding_ms"] for r in rows if r["embedding_ms"] is not None)}
    excluded = [r for r in manifest["items"] if r.get("split") == "excluded"]
    summary = {"schema_version": "sg3.enrollment.v1", "mode": "enrollment_only",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "counts": {"labeled_students": len(roster), "students_with_templates": len(counters),
                          "students_without_templates": len(missing), "candidate_images": len(decisions),
                          "accepted_images": len(samples), "templates_per_method": len(samples),
                          "rejected_images": len(decisions) - len(samples), "excluded_images": len(excluded),
                          "corrupt_or_undecodable_images": sum(bool(r.get("error")) for r in manifest["items"]),
                          "unlabeled_images": sum(r.get("student_id") is None for r in manifest["items"])},
               "status_counts": dict(Counter(r["status"] for r in decisions)),
               "rejection_reasons": dict(Counter(r["reason"] for r in decisions if not r["accepted"])),
               "excluded_reasons": dict(Counter(_excluded_reason(r) for r in excluded)),
               "capture_view_hints": pose_reports,
               "preprocessing": {"detector": "temporary_sg3_yunet", "max_side": max_side,
                                 "sg2_config_sha256": provenance["sg2_config_sha256"],
                                 "sg2_gates_modified": False, "preparation_ms_including_decode": preparation_ms,
                                 "detection": statistics(preprocessing_times["detection"]),
                                 "sg2": statistics(preprocessing_times["preprocessing"])},
               "methods": method_reports,
               "manifest_sha256": provenance["manifest_sha256"],
               "model_sha256": model_hashes,
               "wall_seconds_before_publication": time.perf_counter() - start,
               "runtime": {"python": platform.python_version(), "opencv": cv2.__version__,
                           "numpy": np.__version__, "platform": platform.platform(),
                           "opencv_threads": cv2.getNumThreads()},
               "note": "Enrollment only. No held-out identities, evaluation, accuracy, false-accept/reject estimates, or threshold calibration were performed.",
               "publication": {"gallery_directory": gallery_dir.name, "report_directory": output_dir.name,
                               "previous_destinations_preserved_as_backups": True}}
    write_json(output_stage / "enrollment_summary.json", summary)
    # Verify all metadata is complete/JSON-safe before publishing any directory.
    for stage in (gallery_stage, output_stage):
        for path in stage.glob("*.json"):
            json.loads(path.read_text(encoding="utf-8"))
    gallery_backup = _publish(gallery_stage, gallery_dir, token, replace)
    try:
        output_backup = _publish(output_stage, output_dir, token, replace)
    except Exception:
        # Restore the prior gallery without deleting either new or previous data.
        gallery_dir.rename(gallery_stage)
        if gallery_backup is not None:
            gallery_backup.rename(gallery_dir)
        raise
    summary["publication"].update(gallery_backup=gallery_backup.name if gallery_backup else None,
                                  report_backup=output_backup.name if output_backup else None)
    write_json(output_dir / "enrollment_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", default="module_sg3/private_runs/class_enrollment")
    parser.add_argument("--gallery-dir", help="default: gallery_dir from the selected configuration")
    parser.add_argument("--methods", nargs="+", choices=("sface", "lbph"), default=("sface",))
    parser.add_argument("--sg2-config")
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--replace", action="store_true", help="back up and replace existing output/gallery directories")
    parser.add_argument("--require-all-students", action="store_true", help="refuse publication if any declared student has zero accepted templates")
    args = parser.parse_args(argv)
    try:
        report = enroll_dataset(args.manifest, load_config(args.config), args.output_dir,
                                gallery_dir=args.gallery_dir, methods=args.methods,
                                sg2_config_path=args.sg2_config, max_side=args.max_side,
                                replace=args.replace, require_all_students=args.require_all_students)
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        parser.exit(1, f"enroll_dataset: {exc}\n")
    print(json.dumps({"mode": report["mode"], "counts": report["counts"],
                      "methods": report["methods"], "publication": report["publication"]}, indent=2))


if __name__ == "__main__":
    main()
