"""Check ambiguous annotations, output dimensions, and person separation."""

import json
import unittest
from pathlib import Path

import torch

from silhouette_detector.attributes.labels import ATTRIBUTE_KEYS
from silhouette_detector.market_model import build_model, decode
from silhouette_detector.peta_data import PETA_LABELS, encode_peta


class PetaTests(unittest.TestCase):
    def test_nine_original_attributes_have_44_outputs(self):
        self.assertEqual(tuple(PETA_LABELS), ATTRIBUTE_KEYS)
        model = build_model({"label_schema": "peta", "pretrained": False})
        self.assertEqual(model.classifier.out_features, 44)
        prediction = decode(torch.zeros(1, 44), labels=model.labels)[0]
        self.assertEqual(tuple(prediction), ATTRIBUTE_KEYS)
        with self.assertRaises(ValueError):
            decode(torch.zeros(1, 39), labels=model.labels)

    def test_ambiguous_color_and_missing_clothing_are_ignored(self):
        encoded = dict(
            zip(
                PETA_LABELS,
                encode_peta(
                    {
                        "personalFemale",
                        "personalLess30",
                        "upperBodyBlack",
                        "upperBodyWhite",
                        "lowerBodyJeans",
                        "lowerBodyTrousers",
                        "carryingMessengerBag",
                    }
                ),
                strict=True,
            )
        )
        self.assertEqual(encoded["gender"], 1)
        self.assertEqual(encoded["age"], 1)
        self.assertEqual(encoded["top_color"], -100)
        self.assertEqual(encoded["top_type"], -100)
        self.assertEqual(encoded["bottom_type"], 0)
        self.assertEqual(encoded["handbag"], 1)
        self.assertEqual(encoded["backpack"], 0)

    @unittest.skipUnless(Path("data/peta/records.json").exists(), "Run had download first")
    def test_real_data_has_disjoint_person_ids(self):
        records = json.loads(Path("data/peta/records.json").read_text(encoding="utf-8"))
        groups = {
            split: {row["group_id"] for row in records if row["split"] == split}
            for split in ["train", "validation", "test"]
        }
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertTrue(all(len(row["labels"]) == 9 for row in records))
        self.assertTrue(all(not row["group_id"].startswith("i-LID/") for row in records))


if __name__ == "__main__":
    unittest.main()
