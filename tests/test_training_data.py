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
        from silhouette_detector.market_data import ensure_data

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
                patch("silhouette_detector.market_data.download_data", side_effect=download),
                patch("silhouette_detector.market_data.dvc_command", side_effect=pull),
            ):
                restored = ensure_data(config)
            self.assertEqual(restored, root)
            self.assertEqual(
                json.loads((restored / "records.json").read_text()), [{"split": "train"}]
            )

    def test_mask_uses_length_before_padding(self) -> None:
        import torch

        from silhouette_detector.training.data import collate_embeddings

        short = (torch.ones(1, 2, 4), {"пол": 0}, "a")
        long = (torch.ones(3, 2, 4), {"пол": 1}, "b")
        embeddings, _, mask, _ = collate_embeddings([short, long])
        self.assertEqual(embeddings.shape, (2, 3, 2, 4))
        self.assertEqual(mask.tolist(), [[True, False, False], [True, True, True]])


if __name__ == "__main__":
    unittest.main()
