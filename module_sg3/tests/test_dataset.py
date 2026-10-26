"""Dataset split integrity tests. Synthetic images require no models or faces."""
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from module_sg3.prepare_dataset import assign_splits, prepare_dataset


def row(sample, sid, session, dhash):
    return {"sample_id": sample, "student_id": sid, "capture_session": session,
            "source": f"{sid}/{sample}.jpg", "sha256": hashlib.sha256(sample.encode()).hexdigest(),
            "pixel_sha256": hashlib.sha256((sample + "pixels").encode()).hexdigest(), "dhash": dhash}


def test_same_session_never_becomes_known_evaluation():
    items = [row("a", "S1", "day1", "0000000000000000"),
             row("b", "S1", "day1", "ffffffffffffffff")]
    assign_splits(items, unknown_students=0)
    assert {r["split"] for r in items} == {"enrollment"}


def test_known_evaluation_requires_distinct_session_and_missing_exif_is_conservative():
    items = [row("a", "S1", "day1", "0000000000000000"),
             row("b", "S1", "day2", "ffffffffffffffff"),
             row("c", "S2", "day1", "aaaaaaaaaaaaaaaa"),
             row("d", "S2", "unknown-session", "5555555555555555")]
    assign_splits(items, unknown_students=0)
    assert [r["split"] for r in items] == ["enrollment", "evaluation_known", "enrollment", "enrollment"]


def test_unknown_id_is_disjoint_from_gallery_and_unlabeled_is_not_unknown():
    items = [row("a", "S1", "day1", "0000000000000000"),
             row("b", "S1", "day1", "ffffffffffffffff"),
             row("c", "S2", "day1", "aaaaaaaaaaaaaaaa"),
             row("d", "S2", "day1", "5555555555555555"),
             row("e", None, "day1", "00000000ffffffff")]
    policy = assign_splits(items, unknown_students=1)
    enrolled = {r["student_id"] for r in items if r["split"] == "enrollment"}
    unknown = {r["student_id"] for r in items if r["split"] == "evaluation_unknown"}
    assert enrolled == {"S1"} and unknown == {"S2"}
    assert policy["withheld_student_ids"] == ["S2"]
    assert items[-1]["split"] == "unlabeled"


def test_near_duplicate_cannot_cross_sessions_and_label_conflicts_are_excluded():
    items = [row("a", "S1", "day1", "0000000000000000"),
             row("b", "S1", "day2", "0000000000000001"),
             row("c", "S2", "day1", "ffffffffffffffff"),
             row("d", "S3", "day2", "fffffffffffffffe")]
    policy = assign_splits(items, unknown_students=0)
    assert items[0]["split"] == "enrollment"
    assert items[1]["split"] == "excluded" and items[1]["duplicate_of"] == "a"
    assert items[2]["split"] == items[3]["split"] == "excluded"
    assert policy["duplicate_label_conflict_groups"] == 1


def test_prepare_is_reproducible_and_preserves_sources(tmp_path):
    source, output = tmp_path / "source", tmp_path / "prepared"
    (source / "Student Label").mkdir(parents=True)
    image = Image.new("RGB", (24, 32), (100, 120, 140))
    exif = Image.Exif()
    exif[306] = "2026:10:15 10:00:00"
    exif[274] = 6
    original = source / "Student Label" / "front.jpg"
    image.save(original, exif=exif)
    (source / "broken.jpg").write_bytes(b"this is not an image")
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    first = prepare_dataset(source, output, unknown_students=0)
    second = prepare_dataset(source, output, unknown_students=0)
    assert first == second
    valid = next(r for r in first["items"] if not r.get("error"))
    assert valid["pose_hint"] == "front" and (valid["width"], valid["height"]) == (32, 24)
    assert valid["student_id"].startswith("S") and "Student" not in valid["student_id"]
    assert valid["split"] == "enrollment"
    assert (output / valid["image_path"]).is_file()
    assert first["audit"]["corrupt_or_undecodable"] == 1
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    public = json.loads((output / "audit_summary.json").read_text())
    assert "Student Label" not in json.dumps(public) and str(source) not in json.dumps(public)


def test_reject_output_inside_source_and_invalid_counts(tmp_path):
    with pytest.raises(ValueError):
        prepare_dataset(tmp_path, tmp_path / "converted")
    with pytest.raises(ValueError):
        assign_splits([], unknown_students=-1)


def test_empty_source_is_explicit(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    result = prepare_dataset(source, tmp_path / "out")
    assert result["items"] == []
    assert result["audit"]["total_images"] == 0
    assert result["audit"]["session_independent_known_evaluation_available"] is False


def test_operating_gallery_enrolls_all_students_and_sessions():
    items = [row("a", "S1", "day1", "0000000000000000"),
             row("b", "S1", "day2", "ffffffffffffffff"),
             row("c", "S2", "day1", "aaaaaaaaaaaaaaaa"),
             row("d", "S2", "day2", "5555555555555555")]
    policy = assign_splits(items, unknown_students=2, mode="enrollment_only")
    assert {r["split"] for r in items} == {"enrollment"}
    assert policy["withheld_student_ids"] == []


def test_report_only_similarity_preserves_labels_but_exact_conflicts_fail_closed():
    items = [row("a", "S1", "day1", "0000000000000000"),
             row("b", "S2", "day1", "0000000000000001"),
             row("c", "S3", "day1", "ffffffffffffffff"),
             row("d", "S4", "day1", "fffffffffffffffe")]
    items[3]["sha256"] = items[2]["sha256"]
    policy = assign_splits(items, mode="enrollment_only", near_duplicate_action="report")
    assert items[0]["split"] == items[1]["split"] == "enrollment"
    assert items[0]["near_duplicate_group"] == items[1]["near_duplicate_group"]
    assert items[2]["split"] == items[3]["split"] == "excluded"
    assert policy["duplicate_label_conflict_groups"] == 1


def test_first_frontal_order_is_user_metadata_not_a_visual_inference(tmp_path):
    source = tmp_path / "source"
    (source / "Student").mkdir(parents=True)
    for name, color in (("IMG_0002.jpg", (120, 130, 140)), ("IMG_0001.jpg", (90, 100, 110))):
        Image.new("RGB", (24, 32), color).save(source / "Student" / name)
    result = prepare_dataset(source, tmp_path / "prepared", mode="enrollment_only", first_image_front=True)
    assert [r["pose_hint"] for r in result["items"]] == ["front", "side_unspecified"]
    assert result["mode"] == "enrollment_only"
    assert result["split_policy"]["near_duplicate_action"] == "report"
    assert result["split_policy"]["withheld_student_ids"] == []
