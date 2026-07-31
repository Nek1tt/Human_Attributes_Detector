from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TransformerTests(unittest.TestCase):
    def test_shapes_and_padding_mask(self) -> None:
        import torch

        from silhouette_detector.attributes.transformer import VisionAttrTransformer

        model = VisionAttrTransformer(
            input_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_layers=2,
            attr_sizes={"пол": 3, "возраст": 6},
            drop_path_rate=0,
        ).eval()
        embeddings = torch.randn(2, 3, 4, 8)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        with torch.inference_mode():
            output = model(embeddings, mask)
        self.assertEqual(output["пол"].shape, (2, 3))
        self.assertEqual(output["возраст"].shape, (2, 6))

    def test_wrong_mask_shape_is_rejected(self) -> None:
        import torch

        from silhouette_detector.attributes.transformer import VisionAttrTransformer

        model = VisionAttrTransformer(
            input_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            attr_sizes={"пол": 3},
        )
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 3, 8), torch.ones(1, 3, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
