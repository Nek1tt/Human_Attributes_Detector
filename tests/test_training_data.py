from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TrainingDataTests(unittest.TestCase):
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
