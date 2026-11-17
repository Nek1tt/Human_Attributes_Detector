"""Guard the exact classifier topology specified in the project proposal."""

import ast
import gc
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torchmetrics.classification import MulticlassF1Score

from silhouette_detector.attributes.labels import ATTRIBUTE_LABELS_RU, RU_TO_API
from silhouette_detector.attributes.resnet import ResNetEnsemble
from silhouette_detector.attributes.transformer import VisionAttrTransformer
from silhouette_detector.original_features import unknown_targets
from silhouette_detector.original_models import OriginalClassifier, transfer_common_classes
from silhouette_detector.peta_data import ORIGINAL_PETA_LABELS


class OriginalArchitectureTests(unittest.TestCase):
    def test_old_output_rows_follow_class_names_instead_of_class_indices(self):
        model = torch.nn.Module()
        model.heads = torch.nn.ModuleDict()
        state = {}
        names = {api: ru for ru, api in RU_TO_API.items()}
        for attribute, choices in ORIGINAL_PETA_LABELS.items():
            model.heads[attribute] = torch.nn.Sequential(
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Identity(),
                torch.nn.Linear(2, len(choices)),
            )
            count = len(ATTRIBUTE_LABELS_RU[names[attribute]])
            state[f"heads.{attribute}.3.weight"] = (
                torch.arange(count).float().unsqueeze(1).repeat(1, 2)
            )
            state[f"heads.{attribute}.3.bias"] = torch.arange(count).float()
        transfer_common_classes(model, state, ORIGINAL_PETA_LABELS, transformer=True)
        self.assertEqual(model.heads["backpack"][3].bias.tolist(), [1.0, 0.0, 2.0])
        self.assertEqual(float(model.heads["top_color"][3].bias[0]), 7.0)
        self.assertEqual(float(model.heads["age"][3].bias[-1]), 5.0)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and Path("models/original-minicpm-visual/visual.safetensors").exists()
        and Path("data/peta/records.json").exists(),
        "Original visual weights, PETA and CUDA are required",
    )
    def test_visual_tokens_match_the_original_minicpm_function_exactly(self):
        from PIL import Image

        from silhouette_detector.original_visual import (
            SOURCE_HASHES,
            OriginalVisualEncoder,
            file_hash,
        )

        directory = Path("models/original-minicpm-visual")
        source_path = directory / "modeling_minicpmo.py"
        self.assertEqual(file_hash(source_path), SOURCE_HASHES[source_path.name])
        encoder = OriginalVisualEncoder(directory)
        source = ast.parse(source_path.read_text(encoding="utf-8"))
        method = next(
            node
            for group in source.body
            if isinstance(group, ast.ClassDef) and group.name == "MiniCPMO"
            for node in group.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_vllm_embedding"
        )
        namespace = {"torch": torch, "math": math}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace
        )
        # The ignored text embedding is independent of the returned visual tokens.
        reference = SimpleNamespace(
            vpm=encoder.vpm,
            resampler=encoder.resampler,
            config=SimpleNamespace(vision_batch_size=16),
            training=False,
            llm=SimpleNamespace(
                model=SimpleNamespace(
                    embed_tokens=torch.nn.Embedding(1, 3584, device="cuda", dtype=torch.bfloat16)
                ),
                config=SimpleNamespace(),
            ),
        )
        paths = list(Path("data/peta/PETA dataset/CAVIAR4REID/archive").glob("*.jpg"))[:2]
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        prepared = encoder.prepare(images).to("cuda")
        prepared["input_ids"] = torch.zeros(len(images), 1, dtype=torch.long, device="cuda")
        prepared["image_bound"] = [[] for _ in images]
        with torch.inference_mode():
            actual = encoder.encode_prepared(prepared)
            _, expected = namespace["get_vllm_embedding"](reference, prepared)
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_transformer_keeps_all_original_layers(self):
        model = OriginalClassifier("original_transformer", ORIGINAL_PETA_LABELS, pretrained=False)
        self.assertIsInstance(model.model, VisionAttrTransformer)
        self.assertEqual(tuple(model.model.input_proj.weight.shape), (768, 3584))
        self.assertEqual(len(model.model.layers), 6)
        self.assertTrue(all(layer.self_attn.num_heads == 12 for layer in model.model.layers))
        self.assertEqual(len(model.model.heads), 9)
        self.assertEqual(model.model.heads["gender"][0].out_features, 384)

    def test_resnet_keeps_nine_independent_resnet50_networks(self):
        model = OriginalClassifier("original_resnet50", ORIGINAL_PETA_LABELS, pretrained=False)
        self.assertIsInstance(model.model, ResNetEnsemble)
        for index, choices in enumerate(ORIGINAL_PETA_LABELS.values()):
            backbone = getattr(model.model, f"resnet_{index}")
            self.assertEqual([len(backbone[stage]) for stage in [4, 5, 6, 7]], [3, 4, 6, 3])
            head = getattr(model.model, f"fc_{index}")
            self.assertEqual(
                [head[layer].out_features for layer in [0, 3, 6]], [1024, 512, len(choices)]
            )
        self.assertNotEqual(
            model.model.resnet_0[0].weight.data_ptr(), model.model.resnet_1[0].weight.data_ptr()
        )
        del model
        gc.collect()

    def test_missing_categories_become_the_explicit_unknown_class(self):
        record = {"labels": [-100] * 9}
        converted = unknown_targets([record])[0]["labels"]
        self.assertEqual(converted, [len(choices) - 1 for choices in ORIGINAL_PETA_LABELS.values()])
        self.assertEqual(sum(len(choices) for choices in ORIGINAL_PETA_LABELS.values()), 53)

    def test_macro_f1_excludes_classes_absent_from_both_truth_and_predictions(self):
        metric = MulticlassF1Score(num_classes=3, average="macro")
        result = metric(torch.tensor([0, 1]), torch.tensor([0, 1]))
        self.assertAlmostEqual(float(result), 1.0)


if __name__ == "__main__":
    unittest.main()
