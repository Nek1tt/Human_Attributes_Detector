"""Integration tests for real ResNet and Transformer checkpoints on CPU/CUDA.

The tests auto-detect the checkpoint names used by this repository. Set
``HAD_TEST_RESNET_CHECKPOINT`` or ``HAD_TEST_TRANSFORMER_CHECKPOINT`` to use
another path. CUDA tests are skipped in CPU-only environments unless
``HAD_REQUIRE_CUDA_TESTS=1`` is set.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

from PIL import Image


PYTORCH_STACK_AVAILABLE = all(
    importlib.util.find_spec(module) is not None for module in ("torch", "torchvision")
)
if PYTORCH_STACK_AVAILABLE:
    import torch

    from silhouette_detector.attributes.labels import ATTRIBUTE_KEYS, ATTRIBUTE_SIZES_RU
    from silhouette_detector.attributes.minicpm import _load_transformer_checkpoint
    from silhouette_detector.attributes.resnet import ResNetBackend
    from silhouette_detector.attributes.transformer import VisionAttrTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def _checkpoint_from_environment_or_models(
    environment_name: str,
    candidates: tuple[str, ...],
) -> Path | None:
    configured = os.environ.get(environment_name)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    for candidate in candidates:
        path = MODELS_DIR / candidate
        if path.is_file():
            return path.resolve()
    return None


def _require_cuda() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    message = (
        "PyTorch CUDA is unavailable. Install a CUDA-enabled PyTorch build and verify "
        "that the NVIDIA driver is visible."
    )
    if os.environ.get("HAD_REQUIRE_CUDA_TESTS") == "1":
        raise AssertionError(message)
    raise unittest.SkipTest(message)


def _build_transformer(
    checkpoint: Path,
) -> tuple[VisionAttrTransformer, int, dict[str, int]]:
    state, config = _load_transformer_checkpoint(checkpoint)
    input_dim = int(state["input_proj.weight"].shape[1])
    attr_sizes = dict(config.get("attr_sizes", ATTRIBUTE_SIZES_RU))
    model = VisionAttrTransformer(
        input_dim=input_dim,
        hidden_dim=int(config.get("hidden_dim", 768)),
        num_heads=int(config.get("num_heads", 12)),
        num_layers=int(config.get("num_layers", 6)),
        attr_sizes=attr_sizes,
        drop_path_rate=float(config.get("drop_path_rate", 0.1)),
    )
    model.load_state_dict(state, strict=True)
    return model, input_dim, attr_sizes


def _run_transformer(checkpoint: Path, device: str) -> tuple[int, int, int]:
    model, input_dim, attr_sizes = _build_transformer(checkpoint)
    model.to(device).eval()
    embeddings = torch.randn(1, 2, 2, input_dim, device=device)
    mask = torch.tensor([[True, True]], dtype=torch.bool, device=device)

    with torch.inference_mode():
        outputs = model(embeddings, mask)

    if set(outputs) != set(attr_sizes):
        raise AssertionError(f"unexpected Transformer output heads: {sorted(outputs)}")
    for attribute, size in attr_sizes.items():
        output = outputs[attribute]
        if output.shape != (1, size):
            raise AssertionError(
                f"unexpected shape for {attribute!r}: {tuple(output.shape)}, expected {(1, size)}"
            )
        if output.device.type != torch.device(device).type:
            raise AssertionError(f"{attribute!r} output is on {output.device}, expected {device}")
        if not torch.isfinite(output).all():
            raise AssertionError(f"{attribute!r} output contains NaN or infinity")

    parameter_device = next(model.parameters()).device
    if parameter_device.type != torch.device(device).type:
        raise AssertionError(f"Transformer is on {parameter_device}, expected {device}")
    allocated_bytes = torch.cuda.memory_allocated() if parameter_device.type == "cuda" else 0
    return input_dim, len(outputs), allocated_bytes


class RealModelDeviceTests(unittest.TestCase):
    """Smoke-test the real project checkpoints on their supported devices."""

    @classmethod
    def setUpClass(cls) -> None:
        if PYTORCH_STACK_AVAILABLE:
            return
        message = "PyTorch and torchvision are not installed"
        if os.environ.get("HAD_REQUIRE_CUDA_TESTS") == "1":
            raise AssertionError(message)
        raise unittest.SkipTest(message)

    def test_resnet_checkpoint_runs_on_cuda(self) -> None:
        """Load the real ResNet weights on CUDA and execute one prediction."""
        device = _require_cuda()
        checkpoint = _checkpoint_from_environment_or_models(
            "HAD_TEST_RESNET_CHECKPOINT",
            ("resnet_ens_11.19_e60_s0.782.pt", "resnet_attributes.pt"),
        )
        if checkpoint is None:
            self.skipTest(
                "ResNet checkpoint not found; set HAD_TEST_RESNET_CHECKPOINT or place it in models/"
            )
        self.assertTrue(checkpoint.is_file(), f"ResNet checkpoint not found: {checkpoint}")

        torch.cuda.empty_cache()
        backend = ResNetBackend(checkpoint, device=device)
        self.assertEqual(next(backend.model.parameters()).device.type, "cuda")

        image = Image.new("RGB", (256, 256), color=(96, 128, 160))
        prediction = backend.predict(image)

        self.assertEqual(set(prediction), set(ATTRIBUTE_KEYS))
        self.assertTrue(all(isinstance(value, str) and value for value in prediction.values()))
        self.assertGreater(torch.cuda.memory_allocated(), 0)
        print(
            f"ResNet CUDA: device={device}, attributes={len(prediction)}, "
            f"allocated_mb={torch.cuda.memory_allocated() / 1024**2:.1f}"
        )

    def test_transformer_checkpoint_runs_on_cpu_and_cuda(self) -> None:
        """Strictly load the real Transformer checkpoint and run it on CPU and CUDA."""
        checkpoint = _checkpoint_from_environment_or_models(
            "HAD_TEST_TRANSFORMER_CHECKPOINT",
            (
                "MiniCPM-2.6int4 weights.pt",
                "MiniCPM-o 2.6int4 weights.pt",
                "best_checkpoint.pt",
                "vision_attr_transformer.pt",
            ),
        )
        if checkpoint is None:
            self.skipTest(
                "Transformer checkpoint not found; set HAD_TEST_TRANSFORMER_CHECKPOINT "
                "or place it in models/"
            )
        self.assertTrue(checkpoint.is_file(), f"Transformer checkpoint not found: {checkpoint}")

        input_dim, output_heads, _ = _run_transformer(checkpoint, "cpu")
        print(f"Transformer CPU: input_dim={input_dim}, output_heads={output_heads}")

        device = _require_cuda()
        torch.cuda.empty_cache()
        input_dim, output_heads, allocated_bytes = _run_transformer(checkpoint, device)
        self.assertGreater(allocated_bytes, 0)
        print(
            f"Transformer CUDA: device={device}, input_dim={input_dim}, "
            f"output_heads={output_heads}, "
            f"allocated_mb={allocated_bytes / 1024**2:.1f}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
