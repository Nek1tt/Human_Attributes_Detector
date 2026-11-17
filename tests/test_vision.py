"""Check attention compatibility and architecture checkpoint selection."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from silhouette_detector.vision_model import EfficientAttention


class VisionTests(unittest.TestCase):
    def test_sdpa_matches_scaled_attention(self):
        torch.manual_seed(7)
        original = SimpleNamespace(
            q_proj=nn.Linear(32, 32),
            k_proj=nn.Linear(32, 32),
            v_proj=nn.Linear(32, 32),
            out_proj=nn.Linear(32, 32),
            num_heads=4,
        )
        layer = EfficientAttention(original)
        values = torch.randn(2, 9, 32)
        query, key, value = [
            projection(values).reshape(2, 9, 4, 8).transpose(1, 2)
            for projection in [original.q_proj, original.k_proj, original.v_proj]
        ]
        weights = (query @ key.transpose(-2, -1) / 8**0.5).softmax(dim=-1)
        expected = original.out_proj((weights @ value).transpose(1, 2).reshape(2, 9, 32))
        actual, _ = layer(values)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
