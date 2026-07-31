"""Backend contract for human attribute classification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from PIL import Image

from .labels import ATTRIBUTE_KEYS, UNKNOWN_ATTRIBUTES


class AttributeBackend(ABC):
    """A model loaded once and reused for all tracks and video jobs."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image.Image) -> dict[str, str]:
        raise NotImplementedError

    def close(self) -> None:
        """Release optional backend resources."""
        return None

    @staticmethod
    def normalize(prediction: Mapping[str, str]) -> dict[str, str]:
        return {key: str(prediction.get(key, UNKNOWN_ATTRIBUTES[key])) for key in ATTRIBUTE_KEYS}


class NullBackend(AttributeBackend):
    @property
    def name(self) -> str:
        return "none"

    def predict(self, image: Image.Image) -> dict[str, str]:
        del image
        return dict(UNKNOWN_ATTRIBUTES)
