"""Checks for data separation, missing labels, and frozen pretrained features."""

import importlib.util
import json
import unittest
from pathlib import Path

import torch

from silhouette_detector.market_data import encode_annotations
from silhouette_detector.market_model import LABELS, MarketModel


class MarketTests(unittest.TestCase):
    def test_missing_color_is_excluded_from_training(self):
        annotation = {
            name: 1
            for name in [
                "gender",
                "age",
                "hair",
                "up",
                "down",
                "clothes",
                "hat",
                "backpack",
                "bag",
                "handbag",
            ]
        }
        for attribute, prefix in [("top_color", "up"), ("bottom_color", "down")]:
            annotation.update({prefix + color: 1 for color in LABELS[attribute]})
        annotation["downblue"] = 2
        labels = encode_annotations(annotation)
        self.assertEqual(labels[list(LABELS).index("top_color")], -100)
        self.assertEqual(
            labels[list(LABELS).index("bottom_color")], LABELS["bottom_color"].index("blue")
        )

    def test_frozen_backbone_does_not_update_batchnorm(self):
        model = MarketModel(freeze_backbone=True).train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.classifier.training)
        before = model.backbone.bn1.running_mean.clone()
        model(torch.randn(2, 3, 64, 32)).sum().backward()
        self.assertTrue(torch.equal(before, model.backbone.bn1.running_mean))
        self.assertIsNone(model.backbone.conv1.weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)

    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Install the train extra")
    def test_hydra_overrides_nested_settings(self):
        from silhouette_detector.commands import load_config

        config = load_config(["training.epochs=2", "inference.confidence_threshold=0.8"])
        self.assertEqual(config.training.epochs, 2)
        self.assertEqual(config.inference.confidence_threshold, 0.8)

    @unittest.skipUnless(Path("data/market/records.json").exists(), "Run had download first")
    def test_real_dataset_has_no_identity_leakage(self):
        records = json.loads(Path("data/market/records.json").read_text(encoding="utf-8"))
        groups = {
            split: {row["group_id"] for row in records if row["split"] == split}
            for split in ["train", "validation", "test"]
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(len(groups["train"]) + len(groups["validation"]), 751)
        self.assertEqual(len(groups["test"]), 750)


if __name__ == "__main__":
    unittest.main()
