"""Legacy nine-head ResNet ensemble exposed through the common backend API."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from collections.abc import Mapping

import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ..device import resolve_torch_device
from .base import AttributeBackend
from .labels import ATTRIBUTE_LABELS_RU, RU_TO_API


class ResNetEnsemble(nn.Module):
    """Checkpoint-compatible form of the original nine independent ResNet50 models."""

    def __init__(self) -> None:
        super().__init__()
        for index, labels in enumerate(ATTRIBUTE_LABELS_RU.values()):
            backbone = models.resnet50(weights=None)
            setattr(self, f"resnet_{index}", nn.Sequential(*list(backbone.children())[:-1]))
            setattr(
                self,
                f"fc_{index}",
                nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.Dropout(1 / 3),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(1 / 3),
                    nn.Linear(512, len(labels)),
                ),
            )

    def forward(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index in range(len(ATTRIBUTE_LABELS_RU)):
            features = getattr(self, f"resnet_{index}")(tensor).flatten(1)
            outputs.append(getattr(self, f"fc_{index}")(features))
        return outputs


def _safe_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"ResNet checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("ResNet checkpoint does not contain a state_dict")
    return payload


class ResNetBackend(AttributeBackend):
    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        self.device = resolve_torch_device(device)
        self.model = ResNetEnsemble()
        self.model.load_state_dict(_safe_state_dict(checkpoint), strict=True)
        self.model.to(self.device).eval()
        self._lock = Lock()

    @property
    def name(self) -> str:
        return "resnet"

    @staticmethod
    def _preprocess(image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        shortest = min(image.size)
        scale = 224 / max(image.size)
        resized_shortest = max(224, round(shortest * scale))
        image = TF.resize(image, resized_shortest, interpolation=InterpolationMode.BICUBIC)
        image = TF.center_crop(image, [224, 224])
        tensor = TF.to_tensor(image)
        tensor = TF.normalize(tensor, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return tensor.unsqueeze(0)

    def predict(self, image: Image.Image) -> dict[str, str]:
        tensor = self._preprocess(image).to(self.device)
        with self._lock, torch.inference_mode():
            logits = self.model(tensor)
        prediction: dict[str, str] = {}
        for (attribute, labels), output in zip(ATTRIBUTE_LABELS_RU.items(), logits, strict=True):
            class_index = int(output.argmax(dim=1).item())
            prediction[RU_TO_API[attribute]] = labels[class_index]
        return self.normalize(prediction)
