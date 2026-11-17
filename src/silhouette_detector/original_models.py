"""Adapters for the original classifiers; their layers remain unchanged."""

from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .attributes.labels import RU_TO_API
from .attributes.resnet import ResNetEnsemble
from .attributes.transformer import VisionAttrTransformer


def initialize_from_old(model, checkpoint, transformer=False):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("model_state_dict", payload)
    compatible = {}
    for name, value in state.items():
        if transformer and name.startswith("heads."):
            parts = name.split(".")
            parts[1] = RU_TO_API.get(parts[1], parts[1])
            name = ".".join(parts)
        final_layer = (
            name.startswith("heads.") and ".3." in name
            if transformer
            else name.startswith("fc_") and ".6." in name
        )
        if not final_layer:
            compatible[name] = value
    result = model.load_state_dict(compatible, strict=False)
    expected_missing = [
        name
        for name in model.state_dict()
        if (
            name.startswith("heads.") and ".3." in name
            if transformer
            else name.startswith("fc_") and ".6." in name
        )
    ]
    if set(result.missing_keys) != set(expected_missing) or result.unexpected_keys:
        raise ValueError("Old weights do not match the original architecture")


class OriginalClassifier(nn.Module):
    def __init__(
        self, architecture, labels, pretrained=True, initial_weights=None, visual_directory=None
    ):
        super().__init__()
        self.labels = labels
        self.architecture = architecture
        self.visual_directory = visual_directory
        sizes = {name: len(values) for name, values in labels.items()}
        if architecture == "original_resnet50":
            self.model = ResNetEnsemble(attr_sizes=sizes)
        elif architecture == "original_transformer":
            self.model = VisionAttrTransformer(
                input_dim=3584,
                hidden_dim=768,
                num_heads=12,
                num_layers=6,
                attr_sizes=sizes,
                drop_path_rate=0.1,
            )
        else:
            raise ValueError(f"Unknown original architecture: {architecture}")
        if pretrained and initial_weights:
            initialize_from_old(
                self.model, Path(initial_weights), architecture == "original_transformer"
            )
        elif pretrained and architecture == "original_resnet50":
            reference = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            backbone = nn.Sequential(*list(reference.children())[:-1])
            for index in range(len(labels)):
                getattr(self.model, f"resnet_{index}").load_state_dict(backbone.state_dict())

    def parameter_groups(self, learning_rate, backbone_learning_rate):
        heads, body = [], []
        for name, parameter in self.model.named_parameters():
            (heads if name.startswith(("heads.", "fc_")) else body).append(parameter)
        return [
            {"params": heads, "lr": learning_rate},
            {"params": body, "lr": backbone_learning_rate},
        ]

    def forward(self, inputs):
        output = self.model(inputs)
        return torch.cat(list(output.values()) if isinstance(output, dict) else output, dim=1)
