"""Offline-first MiniCPM embedding backend plus the trained Transformer head."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from PIL import Image

from ..device import resolve_torch_device
from .base import AttributeBackend
from .labels import ATTRIBUTE_LABELS_RU, ATTRIBUTE_SIZES_RU, RU_TO_API
from .transformer import VisionAttrTransformer


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(model_dir: Path, manifest: Path) -> None:
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if not metadata.get("resolved_revision"):
        raise RuntimeError("MiniCPM model manifest has no resolved revision")
    expected = metadata.get("python_files_sha256")
    if not isinstance(expected, Mapping):
        raise RuntimeError("MiniCPM model manifest has no Python file hashes")
    actual_files = {str(path.relative_to(model_dir)): path for path in model_dir.rglob("*.py")}
    if set(expected) != set(actual_files):
        raise RuntimeError("MiniCPM Python files differ from the verified model manifest")
    for relative_path, expected_hash in expected.items():
        if _file_sha256(actual_files[relative_path]) != expected_hash:
            raise RuntimeError(f"MiniCPM model code hash mismatch: {relative_path}")


def _move_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _is_gptq_snapshot(model_dir: Path) -> bool:
    """Detect GPTQ from the snapshot metadata instead of a mutable directory name."""

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiniCPM config not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MiniCPM config is not valid JSON: {config_path}") from exc
    if not isinstance(config, Mapping):
        raise RuntimeError(f"MiniCPM config must be a JSON object: {config_path}")

    quantization_config = config.get("quantization_config")
    return isinstance(quantization_config, Mapping) and (
        str(quantization_config.get("quant_method", "")).lower() == "gptq"
    )


class MiniCPMEmbeddingExtractor:
    """Loads MiniCPM once from a verified local snapshot and extracts visual tokens."""

    def __init__(
        self,
        model_dir: Path,
        device: str = "auto",
        *,
        allow_unverified_model_code: bool = False,
    ) -> None:
        model_dir = model_dir.resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"MiniCPM model directory not found: {model_dir}. Run had-download-minicpm first."
            )
        manifest = model_dir / "model-manifest.json"
        if not manifest.is_file() and not allow_unverified_model_code:
            raise RuntimeError(
                "MiniCPM snapshot has no model-manifest.json. Download it with "
                "had-download-minicpm or explicitly set HAD_ALLOW_UNVERIFIED_MODEL_CODE=true."
            )
        if manifest.is_file():
            _verify_manifest(model_dir, manifest)

        self.device = resolve_torch_device(device)
        is_gptq = _is_gptq_snapshot(model_dir)
        if is_gptq and self.device == "cpu":
            raise RuntimeError(
                "The legacy MiniCPM INT4 checkpoint requires CUDA. For CPU use a full-precision "
                "MiniCPM-o 2.6 snapshot or select the ResNet backend."
            )
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the minicpm extra to use this backend") from exc

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )

        if is_gptq:
            try:
                from auto_gptq import AutoGPTQForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "MiniCPM-o 2.6 INT4 requires the patched minicpmo AutoGPTQ fork. "
                    "Run scripts/setup_minicpm_int4_windows.ps1."
                ) from exc

            self._quantized_model = AutoGPTQForCausalLM.from_quantized(
                str(model_dir),
                torch_dtype=torch.bfloat16,
                device=self.device,
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                disable_exllama=True,
                disable_exllamav2=True,
            )
            self.model = self._quantized_model.model
        else:
            self.model = AutoModel.from_pretrained(
                str(model_dir),
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)

        self.model.eval()
        if not hasattr(self.model, "get_vllm_embedding"):
            raise RuntimeError("This MiniCPM snapshot does not expose get_vllm_embedding")
        self._lock = Lock()

    def extract(self, image: Image.Image) -> torch.Tensor:
        processed = self.processor(text=" ", images=[image.convert("RGB")], return_tensors="pt")
        model_input = {key: _move_to_device(value, self.device) for key, value in processed.items()}
        model_input.setdefault("input_ids", torch.tensor([[220]], device=self.device))
        model_input.setdefault("image_bound", [[[0, 1]]])
        with self._lock, torch.inference_mode():
            _, hidden_states = self.model.get_vllm_embedding(model_input)
        embeddings = hidden_states[0].float()
        while embeddings.ndim > 3 and embeddings.shape[0] == 1:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.ndim != 3:
            raise RuntimeError(
                f"Unexpected MiniCPM embedding shape {tuple(embeddings.shape)}; expected [N, P, D]"
            )
        return embeddings


def _load_transformer_checkpoint(path: Path) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Transformer checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Transformer checkpoint must be a mapping")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, Mapping) or "input_proj.weight" not in state:
        raise ValueError("Transformer checkpoint has no compatible model_state_dict")
    config = dict(payload.get("config", {}))
    return state, config


class MiniCPMTransformerBackend(AttributeBackend):
    def __init__(
        self,
        model_dir: Path,
        checkpoint: Path,
        device: str = "auto",
        *,
        allow_unverified_model_code: bool = False,
    ) -> None:
        self.extractor = MiniCPMEmbeddingExtractor(
            model_dir,
            device,
            allow_unverified_model_code=allow_unverified_model_code,
        )
        state, config = _load_transformer_checkpoint(checkpoint)
        input_dim = int(state["input_proj.weight"].shape[1])
        self.classifier = VisionAttrTransformer(
            input_dim=input_dim,
            hidden_dim=int(config.get("hidden_dim", 768)),
            num_heads=int(config.get("num_heads", 12)),
            num_layers=int(config.get("num_layers", 6)),
            attr_sizes=config.get("attr_sizes", ATTRIBUTE_SIZES_RU),
            drop_path_rate=float(config.get("drop_path_rate", 0.1)),
        )
        self.classifier.load_state_dict(state, strict=True)
        self.classifier.to(self.extractor.device).eval()

    @property
    def name(self) -> str:
        return "minicpm-transformer"

    def predict(self, image: Image.Image) -> dict[str, str]:
        embeddings = self.extractor.extract(image)
        if embeddings.shape[-1] != self.classifier.input_proj.in_features:
            raise RuntimeError(
                "MiniCPM embedding dimension does not match the trained Transformer checkpoint. "
                "A newer MiniCPM model is not a drop-in replacement; re-extract data and retrain."
            )
        batch = embeddings.unsqueeze(0)
        mask = torch.ones((1, embeddings.shape[0]), dtype=torch.bool, device=batch.device)
        with torch.inference_mode():
            logits = self.classifier(batch, mask)
        prediction = {
            RU_TO_API[attribute]: ATTRIBUTE_LABELS_RU[attribute][
                int(output.argmax(dim=1).item())
            ]
            for attribute, output in logits.items()
        }
        return self.normalize(prediction)
