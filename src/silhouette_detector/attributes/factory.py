"""Construct one configured attribute backend."""

from __future__ import annotations

from ..settings import Settings
from .base import AttributeBackend, NullBackend


def create_attribute_backend(settings: Settings) -> AttributeBackend:
    if settings.attribute_backend == "none":
        return NullBackend()
    if settings.attribute_backend == "peta":
        from .peta import PetaBackend

        return PetaBackend(settings.attribute_checkpoint, settings.device)
    raise ValueError(f"Unsupported attribute backend: {settings.attribute_backend}")
