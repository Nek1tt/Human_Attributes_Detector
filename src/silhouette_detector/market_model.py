"""One ResNet18 backbone and twelve attribute classifiers."""

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

LABELS = {
    "gender": ["male", "female"],
    "age": ["young", "teenager", "adult", "old"],
    "hair": ["short", "long"],
    "sleeve": ["long", "short"],
    "lower_length": ["long", "short"],
    "lower_type": ["dress", "pants"],
    "hat": ["no", "yes"],
    "backpack": ["no", "yes"],
    "bag": ["no", "yes"],
    "handbag": ["no", "yes"],
    "top_color": ["black", "white", "red", "purple", "yellow", "gray", "blue", "green"],
    "bottom_color": [
        "black",
        "white",
        "pink",
        "purple",
        "yellow",
        "gray",
        "blue",
        "green",
        "brown",
    ],
}


def image_transform(config, training=False):
    operations = [transforms.Resize((config["height"], config["width"]))]
    if training:
        operations.append(transforms.RandomHorizontalFlip(config["horizontal_flip"]))
    operations.extend([transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    return transforms.Compose(operations)


class MarketModel(nn.Module):
    def __init__(self, pretrained=False, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        dimension = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(dimension, sum(len(labels) for labels in LABELS.values()))
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images):
        return self.classifier(self.backbone(images))


def load_model(checkpoint: Path, device="cpu"):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = MarketModel()
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload["preprocess"]


def decode(logits, threshold=0.0):
    predictions = [{} for _ in range(len(logits))]
    offset = 0
    for attribute, labels in LABELS.items():
        probabilities = logits[:, offset : offset + len(labels)].softmax(dim=1)
        confidence, indices = probabilities.max(dim=1)
        for prediction, index, score in zip(
            predictions, indices.tolist(), confidence.tolist(), strict=True
        ):
            prediction[attribute] = labels[index] if score >= threshold else "unknown"
        offset += len(labels)
    return predictions


def predict_image(checkpoint, image, device="cpu", threshold=0.0):
    model, preprocess = load_model(Path(checkpoint), device)
    with Image.open(image) as source:
        tensor = image_transform(preprocess)(source.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        return decode(model(tensor.to(device)).cpu(), threshold)[0]
