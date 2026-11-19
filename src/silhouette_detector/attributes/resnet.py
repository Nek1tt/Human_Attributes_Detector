"""Nine independent ResNet50 classifiers."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torchvision import models

from .labels import ATTRIBUTE_LABELS_RU


class ResNetEnsemble(nn.Module):
    """Checkpoint-compatible form of the original nine independent ResNet50 models."""

    def __init__(self, attr_sizes: Mapping[str, int] | None = None) -> None:
        super().__init__()
        sizes = attr_sizes or {name: len(labels) for name, labels in ATTRIBUTE_LABELS_RU.items()}
        for index, size in enumerate(sizes.values()):
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
                    nn.Linear(512, size),
                ),
            )

    def forward(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for index in range(len(ATTRIBUTE_LABELS_RU)):
            features = getattr(self, f"resnet_{index}")(tensor).flatten(1)
            outputs.append(getattr(self, f"fc_{index}")(features))
        return outputs
