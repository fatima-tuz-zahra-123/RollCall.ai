"""Protocol tests: missing evidence must not become a successful UNKNOWN test."""
import copy
import json
import unittest
from unittest.mock import Mock, patch

import numpy as np

from module_sg3.api import RecognitionService
from module_sg3.experiment import (
    evaluate_prepared, metrics, rate, run_experiment, select_subject, statistics,
)


def scored(role, truth, candidate, score, status="RECOGNIZED"):
    return {"role": role, "true_id": truth, "best_match": candidate,
            "score": score, "status": status}


class MetricProtocolTests(unittest.TestCase):
    def test_undefined_denominators_are_null_and_serialize_strictly(self):
        empty = metrics([], .363, "sface")
        for name in ("conditional_identification_accuracy", "conditional_FAR", "conditional_FRR",
                     "conditional_misidentification_rate", "conditional_unknown_rejection_rate",
                     "pipeline_acceptance_rate"):
            self.assertIsNone(empty[name])
        self.assertIsNone(rate(0, 0))
        self.assertEqual(statistics([]), {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None})
        json.dumps(empty, allow_nan=False)

    def test_rejection_and_error_do_not_become_unknown_rejection_successes(self):
        failures = [{"status": "QUALITY_REJECTED", "role": "evaluation_unknown"},
                    {"status": "ERROR", "role": "evaluation_unknown"},
                    {"status": "QUALITY_REJECTED", "role": "evaluation_known"}]
        out = metrics(failures, .363, "sface")
        self.assertEqual((out["input_samples"], out["scored_known"], out["scored_unknown"]), (3, 0, 0))
        self.assertIsNone(out["conditional_FAR"])
        self.assertIsNone(out["conditional_unknown_rejection_rate"])
        self.assertEqual(out["unknown_rejections"], 0)
        self.assertEqual(out["pipeline_acceptance_rate"], 0)
        self.assertEqual(out["status_counts"], {"QUALITY_REJECTED": 2, "ERROR": 1})

    def test_conditional_counts_keep_wrong_identity_distinct_from_rejection(self):
        rows = [scored("evaluation_known", "S1", "S1", .8),
                scored("evaluation_known", "S1", "S1", .2, "UNKNOWN"),
                scored("evaluation_known", "S1", "S2", .8),
                scored("evaluation_unknown", "U1", "S2", .8),
                scored("evaluation_unknown", "U2", "S2", .2, "UNKNOWN"),
                {"status": "QUALITY_REJECTED", "role": "evaluation_known"},
                {"status": "QUALITY_REJECTED", "role": "evaluation_unknown"},
                {"status": "ERROR", "role": "evaluation_unknown"}]
        out = metrics(rows, .5, "sface")
        self.assertEqual(out["correct_identifications"], 1)
        self.assertEqual(out["false_rejections"], 1)
        self.assertEqual(out["misidentifications"], 1)
        self.assertEqual(out["false_acceptances"], 1)
        self.assertEqual(out["unknown_rejections"], 1)
        self.assertAlmostEqual(out["conditional_identification_accuracy"], 1/3)
        self.assertAlmostEqual(out["conditional_FRR"], 1/3)
        self.assertAlmostEqual(out["conditional_misidentification_rate"], 1/3)
        self.assertEqual(out["conditional_FAR"], .5)
        self.assertEqual(out["pipeline_acceptance_rate"], 5/8)

    def test_unknown_only_evaluation_cannot_claim_known_accuracy_or_frr(self):
        out = metrics([scored("evaluation_unknown", "U1", "S1", .1, "UNKNOWN")], .363, "sface")
        self.assertEqual(out["conditional_FAR"], 0)
        self.assertEqual(out["conditional_unknown_rejection_rate"], 1)
        self.assertIsNone(out["conditional_identification_accuracy"])
        self.assertIsNone(out["conditional_FRR"])

    def test_sweep_reclassifies_scores_without_reusing_default_decisions_or_mutating_rows(self):
        rows = [scored("evaluation_known", "S1", "S1", .4, "UNKNOWN"),
                scored("evaluation_unknown", "U1", "S1", .35, "UNKNOWN")]
        before = copy.deepcopy(rows)
        strict = metrics(rows, .5, "sface")
        loose = metrics(rows, .3, "sface")
        self.assertEqual((strict["false_rejections"], strict["false_acceptances"]), (1, 0))
        self.assertEqual((loose["correct_identifications"], loose["false_acceptances"]), (1, 1))
        self.assertEqual(rows, before)

    def test_lbph_sweep_uses_lower_distance_and_inclusive_threshold(self):
        rows = [scored("evaluation_known", "S1", "S1", 60),
                scored("evaluation_unknown", "U1", "S1", 80)]
        low, middle, high = (metrics(rows, t, "lbph") for t in (50, 60, 80))
        self.assertEqual(low["false_rejections"], 1)
        self.assertEqual(middle["correct_identifications"], 1)
        self.assertEqual(middle["false_acceptances"], 0)
        self.assertEqual(high["false_acceptances"], 1)

    def test_experiment_keeps_replay_out_of_heldout_threshold_reports(self):
        # Exercise the actual orchestration/report grouping; only I/O/models are
        # mocked. Optimistic training replay must not create known-test accuracy.
        records = [{**scored("enrollment_replay", "S1", "S1", 1.0),
                    "recognition_ms": 1.0, "embedding_ms": .9, "matching_ms": .1, "pose_hint": None},
                   {**scored("evaluation_unknown", "U1", "S1", .4),
                    "recognition_ms": 1.0, "embedding_ms": .9, "matching_ms": .1, "pose_hint": None}]
        prepared = [{"item": {"student_id": "S1", "split": "enrollment"},
                     "sg2": {"accept": True, "face": np.zeros((112,112,3), np.uint8)}}]
        pipeline = Mock(sg2_config_path="unused-config")
        recognizer = Mock(method="sface", score_type="cosine_similarity", threshold=.363)
        cfg = {"sface": {"sweep": [.3, .5]}}
        original_cfg = copy.deepcopy(cfg)
        manifest = {"audit": {"session_independent_known_evaluation_available": False}}
        with patch("module_sg3.experiment.YuNetSG2Pipeline", return_value=pipeline), \
             patch("module_sg3.experiment.prepare_faces", return_value=(manifest, prepared, {"detection": [], "preprocessing": []})), \
             patch("module_sg3.experiment.enroll_prepared", return_value=({"sface": recognizer}, {"sface": {}})) as enrollment_mock, \
             patch("module_sg3.experiment.evaluate_prepared", return_value=records), \
             patch("module_sg3.experiment.stress_demo", return_value=[]), \
             patch("module_sg3.experiment.sha256", return_value="0" * 64), \
             patch("module_sg3.experiment.write_json"), \
             patch("module_sg3.experiment.plot_sweeps"), \
             patch("module_sg3.experiment.Path.mkdir"), \
             patch("module_sg3.experiment.importlib.metadata.version", return_value="test"):
            report = run_experiment("unused-manifest", cfg, "unused-output", methods=("sface",))
            prepared.append({"item": {"student_id": "S2", "split": "evaluation_known"},
                             "sg2": {"accept": True, "face": np.zeros((112,112,3), np.uint8)}})
            with self.assertRaisesRegex(ValueError, "no accepted enrollment"):
                run_experiment("unused-manifest", cfg, "unused-output", methods=("sface",))
        method = report["methods"]["sface"]
        self.assertEqual(method["evaluation"]["input_samples"], 1)
        self.assertEqual(method["evaluation"]["scored_known"], 0)
        self.assertIsNone(method["evaluation"]["conditional_identification_accuracy"])
        self.assertEqual(method["enrollment_replay"]["conditional_identification_accuracy"], 1)
        self.assertEqual([s["evaluation"]["conditional_FAR"] for s in method["threshold_sweep"]], [1, 0])
        self.assertEqual([s["evaluation"]["scored_known"] for s in method["threshold_sweep"]], [0, 0])
        self.assertEqual(cfg, original_cfg)
        self.assertEqual(recognizer.threshold, .363)
        experimental_cfg = enrollment_mock.call_args.args[1]
        self.assertTrue(experimental_cfg["gallery_dir"].endswith("unused-output\\gallery") or
                        experimental_cfg["gallery_dir"].endswith("unused-output/gallery"))
        self.assertIsNot(experimental_cfg, cfg)


class SubjectSelectionTests(unittest.TestCase):
    @staticmethod
    def face(width, height, accepted=True):
        return {"bbox": [10, 20, width, height], "accept": accepted,
                "reason": None if accepted else "blurry", "quality": .8,
                "metrics": {}, "face": np.zeros((112, 112, 3), np.uint8) if accepted else None}

    def test_no_detection_is_an_error_not_unknown_identity(self):
        selected = select_subject({"results": []})
        self.assertEqual(selected["reason"], "no_face_detected")
        rec = Mock(method="sface", score_type="cosine_similarity", threshold=.363)
        out = RecognitionService(rec).recognize_face(selected)
        self.assertEqual(out["status"], "ERROR")
        self.assertIsNone(out["student_id"])
        rec.match.assert_not_called()

    def test_peers_are_ambiguous_even_if_one_face_passed_quality(self):
        small, large = self.face(90, 100), self.face(100, 100, False)
        selected = select_subject({"results": [small, large]})
        self.assertEqual(selected["reason"], "ambiguous_subject")
        self.assertTrue(selected["pipeline_error"])

    def test_dominant_subject_uses_area_and_preserves_rejection(self):
        # A smaller accepted bystander must not inherit the folder's identity.
        bystander, dominant = self.face(80, 100), self.face(160, 100, False)
        packet = {"results": [bystander, dominant]}
        self.assertIs(select_subject(packet), dominant)
        self.assertIs(packet["results"][0], bystander)  # does not reorder caller's packet
        self.assertFalse(select_subject(packet)["accept"])

    def test_exact_dominance_boundary_and_single_subject(self):
        small, large = self.face(100, 100), self.face(150, 100)
        self.assertIs(select_subject({"results": [small, large]}), large)
        self.assertIs(select_subject({"results": [small]}), small)


class EvaluationAPITests(unittest.TestCase):
    def test_evaluation_records_separate_gate_unknown_and_failure(self):
        class Recognizer:
            method, score_type, threshold = "sface", "cosine_similarity", .363

            def __init__(self):
                self.calls = 0

            def embed(self, image, row=None):
                self.calls += 1
                if self.calls == 2:
                    raise ValueError("synthetic model failure")
                return np.zeros(128)

            def match_embedding(self, embedding):
                return "S1", .1

        rec = Recognizer()
        base = {"accept": True, "reason": None, "quality": .8, "metrics": {},
                "face": np.zeros((112, 112, 3), np.uint8)}
        rejected = {"accept": False, "reason": "blurry", "face": None, "quality": 0, "metrics": {}}
        prepared = [{"item": {"sample_id": f"I{i}", "student_id": "U1", "split": "evaluation_unknown"},
                     "sg2": face} for i, face in enumerate((base, rejected, base))]
        rows = evaluate_prepared(prepared, rec)
        self.assertEqual([r["status"] for r in rows], ["UNKNOWN", "QUALITY_REJECTED", "ERROR"])
        self.assertEqual(rec.calls, 2)
        self.assertIsNotNone(rows[0]["embedding_ms"])
        self.assertIsNone(rows[1]["embedding_ms"])
        self.assertIsNone(rows[2]["embedding_ms"])
        self.assertEqual(rows[1]["preprocessing_reason"], "blurry")
        out = metrics(rows, .363, "sface")
        self.assertEqual((out["scored_unknown"], out["unknown_rejections"]), (1, 1))
        self.assertEqual(out["pipeline_acceptance_rate"], 1/3)
        json.dumps(rows, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
