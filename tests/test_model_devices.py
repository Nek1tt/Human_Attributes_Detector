"""Smoke-test the published checkpoints when model files are available."""

import gc
import unittest
from pathlib import Path

import torch
from PIL import Image

from silhouette_detector.attributes.labels import ATTRIBUTE_KEYS
from silhouette_detector.attributes.peta import PetaBackend


class RealModelDeviceTests(unittest.TestCase):
    def check_prediction(self, family, device):
        checkpoint = Path(f"models/original_{family}_peta.pt")
        if not checkpoint.is_file():
            self.skipTest(f"Restore {checkpoint} with DVC")
        if (
            family == "transformer"
            and not Path("models/original-minicpm-visual/visual.safetensors").is_file()
        ):
            self.skipTest("Run had original-download")
        backend = PetaBackend(checkpoint, device)
        try:
            prediction = backend.predict(Image.new("RGB", (96, 192)))
            self.assertEqual(tuple(prediction), ATTRIBUTE_KEYS)
            self.assertEqual(next(backend.model.parameters()).device.type, device)
        finally:
            del backend
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    def test_resnet_cpu(self):
        self.check_prediction("resnet", "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_resnet_cuda(self):
        self.check_prediction("resnet", "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
    def test_transformer_cuda(self):
        self.check_prediction("transformer", "cuda")
