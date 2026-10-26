"""Enrollment-only integrity and publication tests; no biometric images/models."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from module_sg3.enroll_dataset import enroll_dataset, validate_enrollment_manifest
from module_sg3.sg3_core import LBPHRecognizer, SFaceRecognizer


class StubPipeline:
    def __init__(self, root):
        self.sg2_config_path = root / "sg2.json"
        self.sg2_config_path.write_text('{"output_size":112,"gates":{"unchanged":true}}')
        self.sg2_cfg = json.loads(self.sg2_config_path.read_text())

    def process(self, frame, frame_id=None):
        value = int(frame[0, 0, 0])
        accept = value != 0
        return {"results": [{"accept": accept, "reason": None if accept else "pose",
                             "face": np.full((112, 112, 3), value, np.uint8) if accept else None,
                             "quality": .9 if accept else 0, "metrics": {"face_px": 90.0},
                             "bbox": [1, 2, 90, 100]}],
                "detection_ms": 1.0, "preprocessing_ms": 2.0}


def factory(method, cfg):
    if method == "lbph":
        return LBPHRecognizer(cfg)
    # Exercise the real multi-template fit/NPZ/load/match functions without ONNX.
    rec = SFaceRecognizer.__new__(SFaceRecognizer)
    rec.cfg, rec.threshold, rec.gallery = cfg, .363, {}

    def embed(image, row=None):
        vector = np.random.default_rng(int(image[0, 0, 0])).normal(size=128).astype(np.float32)
        return vector / np.linalg.norm(vector)
    rec.embed = embed
    return rec


@pytest.fixture
def setup_enrollment(tmp_path):
    prepared = tmp_path / "prepared"
    (prepared / "images").mkdir(parents=True)
    students = {sid: {"source_label": "Private " + sid} for sid in ("S1", "S2", "S3")}
    records = []
    for index, (sid, value, pose) in enumerate((("S1", 50, "front"), ("S1", 80, "left"),
                                               ("S1", 0, "right"), ("S2", 120, "front"),
                                               ("S3", 0, "front"))):
        path = prepared / "images" / f"I{index}.png"
        cv2.imencode(".png", np.full((120, 140, 3), value, np.uint8))[1].tofile(path)
        records.append({"sample_id": f"I{index}", "student_id": sid, "split": "enrollment",
                        "image_path": path.relative_to(prepared).as_posix(),
                        "source": f"Private {sid}/{pose}.jpg", "sha256": str(index),
                        "pose_hint": pose, "capture_session": "same-day"})
    records.append({"sample_id": "broken", "student_id": "S3", "split": "excluded",
                    "source": "Private S3/broken.jpg", "error": "UnidentifiedImageError"})
    manifest = {"schema_version": "sg3.dataset.v1", "mode": "enrollment_only", "items": records,
                "students": students, "audit": {}}
    manifest_path = prepared / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    models = tmp_path / "models"
    models.mkdir()
    for name in ("detector.onnx", "recognition.onnx"):
        (models / name).write_bytes(name.encode())
    cfg = {"gallery_dir": str(tmp_path / "gallery"), "models_dir": str(models),
           "detector": {"model": "detector.onnx"}, "sface": {"model": "recognition.onnx", "threshold": .363},
           "lbph": {"face_size": 112, "threshold": 80, "radius": 1, "neighbors": 8, "grid_x": 8, "grid_y": 8}}
    return manifest_path, cfg, tmp_path / "reports", StubPipeline(tmp_path)


def enroll(setup, **kwargs):
    manifest, cfg, output, pipeline = setup
    return enroll_dataset(manifest, cfg, output, pipeline=pipeline,
                          recognizer_factory=kwargs.pop("recognizer_factory", factory), **kwargs)


def test_all_accepted_images_are_separate_normalized_linked_templates(setup_enrollment):
    manifest, cfg, output, _ = setup_enrollment
    report = enroll(setup_enrollment)
    assert report["counts"] == {"labeled_students": 3, "students_with_templates": 2,
                                "students_without_templates": 1, "candidate_images": 5,
                                "accepted_images": 3, "templates_per_method": 3,
                                "rejected_images": 2, "excluded_images": 1,
                                "corrupt_or_undecodable_images": 1, "unlabeled_images": 0}
    assert report["methods"]["sface"]["reload_verified_queries"] == 3
    assert "accuracy" not in report
    assert "Private S1" not in json.dumps(report)
    with np.load(Path(cfg["gallery_dir"]) / "sface_gallery.npz", allow_pickle=False) as gallery:
        assert gallery["S1"].shape == (2, 128)
        assert gallery["S2"].shape == (1, 128)
        assert np.allclose(np.linalg.norm(gallery["S1"], axis=1), 1)
    provenance = json.loads((Path(cfg["gallery_dir"]) / "provenance.private.json").read_text())
    assert [(r["student_id"], r["gallery_row"], r["sample_id"]) for r in provenance["template_rows"]] == [
        ("S1", 0, "I0"), ("S1", 1, "I1"), ("S2", 0, "I3")]
    roster = json.loads((output / "roster.private.json").read_text())
    assert roster["S3"]["template_count"] == 0
    assert roster["S3"]["excluded_images"] == 1
    decisions = json.loads((output / "image_decisions.private.json").read_text())
    assert [r["status"] for r in decisions] == ["ACCEPTED_FOR_ENROLLMENT", "ACCEPTED_FOR_ENROLLMENT",
                                                "QUALITY_REJECTED", "ACCEPTED_FOR_ENROLLMENT", "QUALITY_REJECTED"]
    assert all("face" not in decision for decision in decisions)


@pytest.mark.parametrize("mode,role", [(None, "enrollment"), ("evaluation", "enrollment"),
                                       ("enrollment_only", "evaluation_known"),
                                       ("enrollment_only", "evaluation_unknown")])
def test_reject_evaluation_and_implicit_modes(setup_enrollment, mode, role):
    path, cfg, output, _ = setup_enrollment
    manifest = json.loads(path.read_text())
    manifest["mode"], manifest["items"][0]["split"] = mode, role
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        enroll(setup_enrollment)
    assert not Path(cfg["gallery_dir"]).exists()
    assert not output.exists()


def test_optional_coverage_gate_preserves_diagnostics_without_publication(setup_enrollment):
    _, cfg, output, _ = setup_enrollment
    with pytest.raises(ValueError, match="students have no accepted"):
        enroll(setup_enrollment, require_all_students=True)
    assert not Path(cfg["gallery_dir"]).exists()
    assert not output.exists()
    diagnostics = list(output.parent.glob(".reports.staging-*/roster.private.json"))
    assert len(diagnostics) == 1
    assert json.loads(diagnostics[0].read_text())["S3"]["template_count"] == 0


def test_nonempty_gallery_requires_replace_and_keeps_timestamped_backup(setup_enrollment):
    _, cfg, output, _ = setup_enrollment
    gallery = Path(cfg["gallery_dir"])
    gallery.mkdir()
    (gallery / "original.txt").write_text("old gallery preserved")
    with pytest.raises(FileExistsError):
        enroll(setup_enrollment)
    report = enroll(setup_enrollment, replace=True)
    backup = gallery.parent / report["publication"]["gallery_backup"]
    assert (backup / "original.txt").read_text() == "old gallery preserved"
    assert (gallery / "sface_gallery.npz").exists()
    assert (output / "enrollment_summary.json").exists()


def test_failed_template_reload_never_replaces_existing_gallery(setup_enrollment):
    _, cfg, output, _ = setup_enrollment
    gallery = Path(cfg["gallery_dir"])
    gallery.mkdir()
    sentinel = gallery / "original.txt"
    sentinel.write_text("preserve")
    calls = []

    def corrupt_factory(method, config):
        rec = factory(method, config)
        calls.append(rec)
        if len(calls) == 2:
            original_load = rec.load
            def corrupt_load():
                original_load()
                rec.gallery["S1"][0, 0] += .01
            rec.load = corrupt_load
        return rec
    with pytest.raises(RuntimeError, match="template values changed"):
        enroll(setup_enrollment, replace=True, recognizer_factory=corrupt_factory)
    assert sentinel.read_text() == "preserve"
    assert not (gallery / "sface_gallery.npz").exists()
    assert not output.exists()


def test_optional_lbph_uses_identical_samples_and_verifies_every_query(setup_enrollment):
    report = enroll(setup_enrollment, methods=("sface", "lbph"))
    assert set(report["methods"]) == {"sface", "lbph"}
    assert report["methods"]["lbph"]["templates"] == 3
    assert report["methods"]["lbph"]["reload_verified_queries"] == 3


def test_no_accepted_faces_cannot_publish_empty_gallery(setup_enrollment):
    path, cfg, output, _ = setup_enrollment
    manifest = json.loads(path.read_text())
    for item in manifest["items"]:
        if item["split"] == "enrollment":
            cv2.imencode(".png", np.zeros((120, 140, 3), np.uint8))[1].tofile(path.parent / item["image_path"])
    with pytest.raises(ValueError, match="No SG2-accepted"):
        enroll(setup_enrollment)
    assert not Path(cfg["gallery_dir"]).exists()
    assert not output.exists()


def test_manifest_requires_unique_samples_and_labeled_enrollment(setup_enrollment):
    manifest = json.loads(setup_enrollment[0].read_text())
    manifest["items"][1]["sample_id"] = manifest["items"][0]["sample_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_enrollment_manifest(manifest)
    manifest = json.loads(setup_enrollment[0].read_text())
    manifest["items"][0]["student_id"] = None
    with pytest.raises(ValueError, match="Unlabeled"):
        validate_enrollment_manifest(manifest)
