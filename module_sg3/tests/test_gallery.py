"""Model-free tests for persisted template validation and actual cosine matching."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import warnings

import numpy as np

from module_sg3.sg3_core import LBPHRecognizer, SFaceRecognizer


def unit(index=0, sign=1.0):
    vector = np.zeros(128, np.float32)
    vector[index] = sign
    return vector


class SFaceGalleryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "sface_gallery.npz"
        self.rec = SFaceRecognizer.__new__(SFaceRecognizer)
        self.rec.cfg = {"gallery_dir": self.temporary.name}
        self.rec.gallery = {}
        self.rec.threshold = 0.363

    def test_valid_multiple_templates_roundtrip_and_maximum_match(self):
        # Query matches the second template, not a person's template centroid.
        self.rec.gallery = {"S001": np.stack([unit(0), unit(1)]), "S002": unit(2)[None]}
        self.rec.save()
        self.rec.gallery = {}
        self.rec.load()
        self.assertEqual(self.rec.gallery["S001"].shape, (2, 128))
        self.assertEqual(self.rec.match_embedding(unit(1)), ("S001", 1.0))
        self.assertEqual(self.rec.match_embedding(unit(2)), ("S002", 1.0))

    def test_separate_matching_does_not_reextract_embedding(self):
        self.rec.gallery = {"S001": unit()[None]}
        self.rec.embed = Mock(side_effect=AssertionError("must not run extraction"))
        self.assertEqual(self.rec.match_embedding(unit()), ("S001", 1.0))
        self.rec.embed.assert_not_called()

    def test_image_match_extracts_once_and_uses_same_matching_rule(self):
        self.rec.gallery = {"S001": unit()[None], "S002": unit(1)[None]}
        self.rec.embed = Mock(return_value=unit(1))
        image, row = object(), object()
        self.assertEqual(self.rec.match(image, row), ("S002", 1.0))
        self.rec.embed.assert_called_once_with(image, row)

    def test_exact_negative_one_cosine_still_has_a_best_identity(self):
        self.rec.gallery = {"S001": unit()[None]}
        self.assertEqual(self.rec.match_embedding(unit(sign=-1)), ("S001", -1.0))

    def test_empty_fit_save_and_match_fail_explicitly(self):
        self.rec.embed = Mock()
        with self.assertRaises(ValueError):
            self.rec.fit([])
        self.rec.embed.assert_not_called()
        with self.assertRaises(ValueError):
            self.rec.save()
        with self.assertRaises(ValueError):
            self.rec.match_embedding(unit())

    def test_fit_keeps_multiple_templates_per_student(self):
        self.rec.embed = Mock(side_effect=[unit(), unit(1), unit(2)])
        self.rec.fit([("S001", object(), None), ("S001", object(), None), ("S002", object(), None)])
        self.assertEqual(self.rec.embed.call_count, 3)
        self.assertEqual(self.rec.gallery["S001"].shape, (2, 128))
        np.testing.assert_array_equal(self.rec.gallery["S002"], unit(2)[None])

    def test_empty_or_malformed_gallery_is_rejected(self):
        cases = [{}, {"S001": np.empty((0, 128), np.float32)},
                 {"S001": np.zeros((1, 128), np.float32)},
                 {"S001": np.ones((1, 128), np.float32)},
                 {"S001": unit()}, {"S001": np.zeros((1, 127), np.float32)},
                 {"S001": np.full((1, 128), np.nan)},
                 {"S001": np.full((1, 128), np.inf)},
                 {"UNKNOWN": unit()[None]}, {"": unit()[None]},
                 {"S001": np.array([[object()] * 128], dtype=object)}]
        for gallery in cases:
            with self.subTest(keys=list(gallery)):
                np.savez_compressed(self.path, **gallery)
                with self.assertRaises((ValueError, TypeError)):
                    self.rec.load()

    def test_failed_load_does_not_replace_previous_valid_gallery(self):
        previous = {"S001": unit()[None]}
        self.rec.gallery = previous
        np.savez(self.path, S002=np.zeros((1, 128), np.float32))
        with self.assertRaises(ValueError):
            self.rec.load()
        self.assertIs(self.rec.gallery, previous)

    def test_missing_and_corrupt_npz_fail_without_pickle(self):
        with self.assertRaises(FileNotFoundError):
            self.rec.load()
        self.path.write_bytes(b"This is not a NumPy archive")
        with self.assertRaises(ValueError):
            self.rec.load()

    def test_invalid_queries_are_rejected_before_matching(self):
        self.rec.gallery = {"S001": unit()[None]}
        cases = [np.zeros(128), np.ones(128), unit()[None], np.zeros(127),
                 np.full(128, np.nan), np.full(128, np.inf), "invalid"]
        for query in cases:
            with self.subTest(shape=np.shape(query)):
                with self.assertRaises(ValueError):
                    self.rec.match_embedding(query)

    def test_embedding_validation_and_normalization(self):
        image = np.zeros((112, 112, 3), np.uint8)
        self.rec.net = Mock()
        self.rec.net.feature.return_value = unit()[None] * 3.0
        np.testing.assert_array_equal(self.rec.embed(image), unit())
        for bad in (np.zeros((1, 128)), np.zeros((1, 127)),
                    np.full((1, 128), np.nan), np.full((1, 128), 1e30, np.float32)):
            self.rec.net.feature.return_value = bad
            with self.subTest(shape=bad.shape), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with self.assertRaises(ValueError):
                    self.rec.embed(image)


class LBPHEmptyEnrollmentTests(unittest.TestCase):
    def test_empty_fit_is_rejected_without_invoking_opencv_training(self):
        rec = LBPHRecognizer.__new__(LBPHRecognizer)
        rec.model = Mock()
        with self.assertRaises(ValueError):
            rec.fit([])
        rec.model.train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
