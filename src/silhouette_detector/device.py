"""CPU/GPU selection helpers shared by PyTorch and ONNX Runtime."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def resolve_torch_device(requested: str, torch_module: Any | None = None) -> str:
    requested = requested.lower()
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed; install the cpu or gpu extra") from exc

    has_cuda = bool(torch_module.cuda.is_available())
    if requested == "auto":
        return "cuda:0" if has_cuda else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda:"):
        if not has_cuda:
            raise RuntimeError(f"{requested} was requested, but CUDA is not available in PyTorch")
        index = int(requested.split(":", 1)[1])
        if index < 0 or index >= torch_module.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is not available")
        return requested
    raise ValueError(f"Unsupported device: {requested}")


def select_onnx_providers(requested: str, available: Iterable[str]) -> list[str]:
    available_set = set(available)
    cpu = "CPUExecutionProvider"
    cuda = "CUDAExecutionProvider"
    if cpu not in available_set:
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
    if requested == "auto":
        return [cuda, cpu] if cuda in available_set else [cpu]
    if requested == "cpu":
        return [cpu]
    if requested == "cuda" or requested.startswith("cuda:"):
        if cuda not in available_set:
            raise RuntimeError(
                "CUDA was requested, but ONNX Runtime has no CUDAExecutionProvider; "
                "install onnxruntime-gpu and verify CUDA/cuDNN"
            )
        return [cuda, cpu]
    raise ValueError(f"Unsupported device: {requested}")
