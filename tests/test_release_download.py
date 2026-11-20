"""Reject damaged release assets without replacing local model files."""

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_release_weights import download_asset


class ReleaseDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"
        self.asset = {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}

    def test_download_verifies_bytes(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"abc")):
            download_asset("https://github.com/example", self.path, self.asset)
        self.assertEqual(self.path.read_bytes(), b"abc")

    def test_corrupt_download_is_removed(self):
        with (
            patch("urllib.request.urlopen", return_value=io.BytesIO(b"bad")),
            self.assertRaises(ValueError),
        ):
            download_asset("https://github.com/example", self.path, self.asset)
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_existing_weights_are_preserved(self):
        self.path.write_bytes(b"user")
        with self.assertRaises(FileExistsError):
            download_asset("https://github.com/example", self.path, self.asset)
        self.assertEqual(self.path.read_bytes(), b"user")
