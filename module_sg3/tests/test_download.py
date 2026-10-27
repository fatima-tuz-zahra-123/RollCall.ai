"""Offline model-download tests: integrity verification and atomic replacement."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from module_sg3 import download_models


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.target = self.directory / "test_model.onnx"
        self.part = self.directory / "test_model.onnx.part"
        self.payload = b"synthetic public model test payload"
        self.entry = {"url": "https://example.invalid/model.onnx", "size": len(self.payload),
                      "sha256": hashlib.sha256(self.payload).hexdigest()}
        self.manifest = {self.target.name: self.entry}
        self.cfg = {"models_dir": self.temporary.name}
        self.manifest_patch = patch.object(download_models.json, "loads", return_value=self.manifest)
        self.manifest_patch.start()
        self.addCleanup(self.manifest_patch.stop)

    def test_verified_existing_model_is_not_downloaded_or_rewritten(self):
        self.target.write_bytes(self.payload)
        before = self.target.stat().st_mtime_ns
        with patch.object(download_models.urllib.request, "urlopen") as urlopen:
            download_models.download(self.cfg)
        urlopen.assert_not_called()
        self.assertEqual(self.target.stat().st_mtime_ns, before)
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_valid_payload_replaces_bad_existing_model_after_verification(self):
        self.target.write_bytes(b"old invalid model")
        with patch.object(download_models.urllib.request, "urlopen", return_value=io.BytesIO(self.payload)) as urlopen:
            download_models.download(self.cfg)
        urlopen.assert_called_once_with(self.entry["url"], timeout=60)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.part.exists())

    def test_hash_failure_never_overwrites_existing_model(self):
        previous = b"preserved existing model"
        self.target.write_bytes(previous)
        wrong_hash_same_size = bytes([self.payload[0] ^ 1]) + self.payload[1:]
        with patch.object(download_models.urllib.request, "urlopen", return_value=io.BytesIO(wrong_hash_same_size)):
            with self.assertRaisesRegex(ValueError, "integrity"):
                download_models.download(self.cfg)
        self.assertEqual(self.target.read_bytes(), previous)
        self.assertFalse(self.part.exists())

    def test_wrong_size_creates_no_model_and_cleans_partial_file(self):
        with patch.object(download_models.urllib.request, "urlopen", return_value=io.BytesIO(self.payload[:-1])):
            with self.assertRaisesRegex(ValueError, "integrity"):
                download_models.download(self.cfg)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.part.exists())

    def test_network_failure_preserves_target_and_cleans_partial_file(self):
        self.target.write_bytes(b"existing model")
        self.part.write_bytes(b"stale interrupted download")
        with patch.object(download_models.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline test")):
            with self.assertRaises(urllib.error.URLError):
                download_models.download(self.cfg)
        self.assertEqual(self.target.read_bytes(), b"existing model")
        self.assertFalse(self.part.exists())

    def test_interrupted_read_preserves_target_and_cleans_partial_file(self):
        class InterruptedResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("synthetic interrupted response")
                return super().read(4)

        self.target.write_bytes(b"existing model")
        with patch.object(download_models.urllib.request, "urlopen", return_value=InterruptedResponse(self.payload)):
            with self.assertRaises(OSError):
                download_models.download(self.cfg)
        self.assertEqual(self.target.read_bytes(), b"existing model")
        self.assertFalse(self.part.exists())


if __name__ == "__main__":
    unittest.main()
