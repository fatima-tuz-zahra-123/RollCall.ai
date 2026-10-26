"""Prepare a private, reproducible SG3 manifest without modifying source images.

Run from the repository root, for example::

    python -m module_sg3.prepare_dataset --source "../private_data/Class Dataset" \
        --output module_sg3/private_data/prepared --unknown-students 2

All paths in manifest items are relative to source_root or manifest.parent.
The manifest and converted photographs contain private biometric data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
SCHEMA_VERSION = "sg3.dataset.v1"


def _stable_id(prefix: str, value: str, length: int) -> str:
    normalized = unicodedata.normalize("NFC", value).casefold()
    return prefix + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def _capture_session(exif: Image.Exif) -> str:
    """One camera/day is one conservative session; missing EXIF is one group."""
    date = exif.get(36867) or exif.get(306)
    try:
        date = exif.get_ifd(34665).get(36867) or date
    except (KeyError, TypeError, ValueError, SyntaxError):
        pass
    if not date:
        return "unknown-session"
    day = str(date)[:10].replace(":", "-")
    camera = f"{exif.get(271, '')}|{exif.get(272, '')}"
    return f"{day}:camera-{hashlib.sha256(camera.encode()).hexdigest()[:8]}"


def _dhash(image: Image.Image) -> str:
    small = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = small.load()
    values = [pixels[x, y] for y in range(8) for x in range(9)]
    number = 0
    for y in range(8):
        for x in range(8):
            number = (number << 1) | int(values[y * 9 + x + 1] > values[y * 9 + x])
    return f"{number:016x}"


def _decode(path: Path):
    if path.suffix.lower() in {".heic", ".heif"}:
        import pillow_heif

        pillow_heif.register_heif_opener()
    with Image.open(path) as original:
        original.load()  # Fully decode so a readable header cannot hide corruption.
        session = _capture_session(original.getexif())
        image = ImageOps.exif_transpose(original).convert("RGB")
    return image, session


def _duplicate_groups(items: list[dict], threshold: int | None) -> list[list[dict]]:
    """Conservative transitive grouping by bytes, decoded pixels, or dHash."""
    valid = [r for r in items if not r.get("error")]
    parent = list(range(len(valid)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    for i, a in enumerate(valid):
        for j in range(i):
            b = valid[j]
            same = a["sha256"] == b["sha256"] or a["pixel_sha256"] == b["pixel_sha256"]
            near = threshold is not None and (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count() <= threshold
            if same or near:
                union(i, j)
    groups = defaultdict(list)
    for i, row in enumerate(valid):
        groups[find(i)].append(row)
    return list(groups.values())


def assign_splits(items: list[dict], unknown_students: int = 2, near_duplicate_hamming: int = 5,
                  *, mode: str = "evaluation", near_duplicate_action: str = "exclude"):
    """Assign whole identities to UNKNOWN; known tests require another session.

    Evaluation conservatively excludes duplicate/near-duplicate candidates.
    With near_duplicate_action='report', only exact duplicates are excluded and
    perceptual candidates are flagged. Conflicting labels in an excluded group
    exclude the entire group. Unlabeled images never imply UNKNOWN. Operational
    enrollment includes every labeled session and withholds no identities.
    """
    if unknown_students < 0:
        raise ValueError("unknown_students must be nonnegative")
    if not 0 <= near_duplicate_hamming <= 64:
        raise ValueError("near_duplicate_hamming must be in [0, 64]")
    if mode not in ("evaluation", "enrollment_only"):
        raise ValueError("mode must be evaluation or enrollment_only")
    if near_duplicate_action not in ("exclude", "report"):
        raise ValueError("near_duplicate_action must be exclude or report")
    if mode == "enrollment_only":
        unknown_students = 0
    for row in items:
        row["split"] = "excluded" if row.get("error") else "unassigned"
        row.pop("duplicate_of", None)
        row.pop("exclusion_reason", None)
        row.pop("near_duplicate_group", None)
    candidate_groups = _duplicate_groups(items, near_duplicate_hamming)
    duplicate_groups = sum(len(group) > 1 for group in candidate_groups)
    for group in candidate_groups:
        if len(group) > 1:
            group_id = min(r["sample_id"] for r in group)
            for row in group:
                row["near_duplicate_group"] = group_id
    # Full-photo dHash can group different students sharing a background. For
    # operational enrollment it flags review candidates; only exact duplicates
    # are removed. Evaluation keeps its existing conservative exclusion policy.
    groups = candidate_groups if near_duplicate_action == "exclude" else _duplicate_groups(items, None)
    conflicts = 0
    for group in groups:
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r["source"])
        canonical = group[0]
        if len({r["student_id"] for r in group}) > 1:
            conflicts += 1
            for row in group:
                row.update(split="excluded", exclusion_reason="duplicate_label_conflict")
        else:
            for row in group[1:]:
                row.update(split="excluded", exclusion_reason="duplicate_or_near_duplicate", duplicate_of=canonical["sample_id"])
    by_student = defaultdict(list)
    for row in items:
        if row["split"] == "excluded":
            continue
        if row["student_id"] is None:
            row["split"] = "unlabeled"
        else:
            by_student[row["student_id"]].append(row)
    # Sorted pseudonyms stay stable as photographs change. Retain at least one
    # enrolled identity, and require >=2 usable unique photos for unknown IDs.
    eligible = sorted(sid for sid, rows in by_student.items() if len(rows) >= 2)
    count = min(unknown_students, len(eligible), max(0, len(by_student) - 1))
    withheld = set(eligible[-count:]) if count else set()
    for sid, rows in by_student.items():
        if sid in withheld:
            for row in rows:
                row["split"] = "evaluation_unknown"
            continue
        # Missing EXIF is deliberately treated as an unknown group. It cannot
        # prove separation from any dated capture and stays in enrollment.
        sessions = sorted({row["capture_session"] for row in rows})
        if mode == "enrollment_only" or "unknown-session" in sessions:
            enrollment_sessions = set(sessions)
        else:
            enrollment_sessions = {sessions[0]}
        for row in rows:
            row["split"] = "enrollment" if row["capture_session"] in enrollment_sessions else "evaluation_known"
    return {
        "requested_unknown_students": unknown_students,
        "withheld_student_ids": sorted(withheld),
        "duplicate_candidate_groups": duplicate_groups,
        "duplicate_label_conflict_groups": conflicts,
        "near_duplicate_hamming_threshold": near_duplicate_hamming,
        "near_duplicate_action": near_duplicate_action,
        "mode": mode,
    }


def prepare_dataset(source: str | Path, output: str | Path, unknown_students: int = 2,
                    near_duplicate_hamming: int = 5, *, mode: str = "evaluation",
                    near_duplicate_action: str | None = None, first_image_front: bool = False) -> dict:
    if unknown_students < 0 or not 0 <= near_duplicate_hamming <= 64:
        raise ValueError("Invalid unknown student count or near-duplicate threshold")
    if mode not in ("evaluation", "enrollment_only"):
        raise ValueError("mode must be evaluation or enrollment_only")
    near_duplicate_action = near_duplicate_action or ("report" if mode == "enrollment_only" else "exclude")
    if near_duplicate_action not in ("report", "exclude"):
        raise ValueError("near_duplicate_action must be report or exclude")
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_dir():
        raise ValueError("Source must be an existing directory")
    if output == source or output.is_relative_to(source):
        raise ValueError("Output must be outside the source directory")
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=lambda p: p.relative_to(source).as_posix().casefold())
    items, labels = [], {}
    for path in files:
        relative = path.relative_to(source)
        label = relative.parts[0] if len(relative.parts) > 1 else None
        sid = _stable_id("S", label, 8) if label else None
        if label:
            if sid in labels and labels[sid] != label:
                raise ValueError("Student pseudonym collision; labels need manual review")
            labels[sid] = label
        sample_id = _stable_id("I", relative.as_posix(), 12)
        row = {"sample_id": sample_id, "student_id": sid, "source": relative.as_posix(),
               "sha256": None, "capture_session": "unknown-session", "pose_hint": None,
               "width": None, "height": None}
        if path.stem.casefold() in {"front", "left", "right", "top"}:
            row["pose_hint"] = path.stem.casefold()
        try:
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            image, session = _decode(path)
            row.update(width=image.width, height=image.height, capture_session=session,
                       pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(), dhash=_dhash(image))
            converted = image_dir / f"{sample_id}.jpg"
            image.save(converted, format="JPEG", quality=95, subsampling=0)
            row["image_path"] = converted.relative_to(output).as_posix()
        except Exception as exc:
            # The manifest records the type only: native decoder messages can
            # contain personal source paths and must not leak through summaries.
            row["error"] = type(exc).__name__
        items.append(row)
    if first_image_front:
        by_label = defaultdict(list)
        for item in items:
            if item["student_id"] is not None:
                by_label[item["student_id"]].append(item)
        for rows in by_label.values():
            rows.sort(key=lambda r: r["source"].casefold())
            for index, item in enumerate(rows):
                item["capture_order"] = index
                if item["pose_hint"] is None:
                    item["pose_hint"] = "front" if index == 0 else "side_unspecified"
                    item["pose_hint_source"] = "user_declared_first_filename_frontal_rest_side_views"
    policy = assign_splits(items, unknown_students, near_duplicate_hamming,
                           mode=mode, near_duplicate_action=near_duplicate_action)
    students = {}
    for sid, original in sorted(labels.items()):
        rows = [r for r in items if r["student_id"] == sid]
        students[sid] = {"source_label": original, "image_count": len(rows),
                         "usable_images": sum(r["split"] != "excluded" for r in rows),
                         "split_counts": dict(Counter(r["split"] for r in rows)),
                         "capture_sessions": sorted({r["capture_session"] for r in rows if not r.get("error")})}
    counts = Counter(r["split"] for r in items)
    decoded = [r for r in items if not r.get("error")]
    original_hashes = [r["sha256"] for r in items if r["sha256"] is not None]
    audit = {"total_images": len(items), "labeled_students": len(students),
             "labeled_images": sum(r["student_id"] is not None for r in items),
             "unlabeled_images": sum(r["student_id"] is None for r in items),
             "corrupt_or_undecodable": sum(bool(r.get("error")) for r in items),
             "split_counts": dict(counts),
             "resolution_counts": dict(Counter(f"{r['width']}x{r['height']}" for r in items if not r.get("error"))),
             "session_independent_known_evaluation_available": bool(counts["evaluation_known"]),
             "capture_sessions": sorted({r["capture_session"] for r in decoded}),
             "exact_duplicate_extra_files": len(original_hashes) - len(set(original_hashes)),
             "near_duplicate_candidate_groups": policy["duplicate_candidate_groups"],
             "all_images_decoded": not any(r.get("error") for r in items)}
    manifest = {"schema_version": SCHEMA_VERSION, "mode": mode, "source_root": str(source), "items": items,
                "students": students, "audit": audit,
                "split_policy": {**policy, "identity_assignment": "existing folder labels only; top-level images remain unlabeled",
                                 "unknown_selection": "last sorted stable pseudonyms with >=2 usable unique images; keep >=1 enrolled identity",
                                 "session_definition": "EXIF capture date plus camera; missing EXIF groups conservatively stay in enrollment",
                                 "known_selection": ("all labeled sessions enrolled; no evaluation or withheld identities" if mode == "enrollment_only" else
                                                     "earliest session enrollment; later distinct sessions evaluation_known; no within-session evaluation"),
                                 "first_image_front": first_image_front,
                                 "deduplication": ("exact SHA256/pixel-hash duplicates excluded; dHash candidates flagged only" if near_duplicate_action == "report" else
                                                   "transitive SHA256/pixel-hash/dHash grouping; one canonical same-label image; conflicting labels excluded"),
                                 "converted_images": "EXIF orientation applied; RGB JPEG quality 95, no identity EXIF; consumers decode BGR with OpenCV"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unknown-students", type=int, default=2)
    parser.add_argument("--near-duplicate-hamming", type=int, default=5)
    parser.add_argument("--mode", choices=("evaluation", "enrollment_only"), default="evaluation",
                        help="enrollment_only includes all labeled students/sessions with no UNKNOWN holdout")
    parser.add_argument("--near-duplicate-action", choices=("report", "exclude"),
                        help="default: report for enrollment_only; exclude for evaluation")
    parser.add_argument("--first-image-front", action="store_true",
                        help="user declares first sorted filename in each student folder frontal; remaining views unspecified side")
    args = parser.parse_args()
    manifest = prepare_dataset(args.source, args.output, args.unknown_students, args.near_duplicate_hamming,
                               mode=args.mode, near_duplicate_action=args.near_duplicate_action,
                               first_image_front=args.first_image_front)
    print(json.dumps(manifest["audit"], indent=2))


if __name__ == "__main__":
    main()
