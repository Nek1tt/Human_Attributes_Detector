from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TrainingDataTests(unittest.TestCase):
    def test_public_download_restores_data_missing_from_an_existing_dvc_remote(self):
        from silhouette_detector.peta_data import ensure_data

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "dataset"
            remote = base / "remote"
            remote.mkdir()
            Path(str(root) + ".dvc").write_text("outs: []\n", encoding="utf-8")
            config = SimpleNamespace(root=root, remote=remote)

            def download(settings):
                settings.root.mkdir()
                (settings.root / "records.json").write_text('[{"split": "train"}]')

            def pull(*arguments):
                if not (root / "records.json").exists():
                    raise subprocess.CalledProcessError(1, ["dvc", *map(str, arguments)])

            with (
                patch("silhouette_detector.peta_data.download_peta", side_effect=download),
                patch("silhouette_detector.peta_data.dvc_command", side_effect=pull),
            ):
                restored = ensure_data(config)
            self.assertEqual(restored, root)
            self.assertEqual(
                json.loads((restored / "records.json").read_text()), [{"split": "train"}]
            )

    def test_hydra_overrides_nested_settings(self):
        from silhouette_detector.commands import load_config

        config = load_config(["training.epochs=2", "inference.confidence_threshold=0.8"])
        self.assertEqual(config.training.epochs, 2)
        self.assertEqual(config.inference.confidence_threshold, 0.8)


if __name__ == "__main__":
    unittest.main()
