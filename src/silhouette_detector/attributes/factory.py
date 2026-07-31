"""Construct one configured attribute backend."""

from __future__ import annotations

from ..settings import Settings
from .base import AttributeBackend, NullBackend


def create_attribute_backend(settings: Settings) -> AttributeBackend:
    if settings.attribute_backend == "none":
        return NullBackend()
    if settings.attribute_backend == "resnet":
        from .resnet import ResNetBackend

        return ResNetBackend(settings.resnet_checkpoint, settings.device)
    if settings.attribute_backend == "minicpm":
        from .minicpm import MiniCPMTransformerBackend

        return MiniCPMTransformerBackend(
            settings.minicpm_model_dir,
            settings.transformer_checkpoint,
            settings.device,
            allow_unverified_model_code=settings.allow_unverified_model_code,
        )
    raise ValueError(f"Unsupported attribute backend: {settings.attribute_backend}")
