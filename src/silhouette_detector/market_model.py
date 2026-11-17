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
    def __init__(self, pretrained=False, freeze_backbone=False, labels=None):
        super().__init__()
        self.labels = LABELS if labels is None else labels
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        dimension = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(dimension, sum(len(values) for values in self.labels.values()))
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


def build_model(config, pretrained=None):
    options = dict(config)
    options.pop("output", None)
    schema = options.pop("label_schema", "market")
    if schema in {"peta", "peta_original"}:
        from .peta_data import ORIGINAL_PETA_LABELS, PETA_LABELS

        options["labels"] = ORIGINAL_PETA_LABELS if schema == "peta_original" else PETA_LABELS
    elif schema != "market":
        raise ValueError(f"Unknown label schema: {schema}")
    architecture = options.pop("architecture", "resnet18")
    if pretrained is not None:
        options["pretrained"] = pretrained
    if architecture in {"original_resnet50", "original_transformer"}:
        from .original_models import OriginalClassifier

        return OriginalClassifier(architecture=architecture, **options)
    if architecture == "minicpm_vision":
        from .vision_model import VisionAttributeTransformer

        return VisionAttributeTransformer(**options)
    return MarketModel(**options)


def load_model(checkpoint: Path, device="cpu", vision_directory=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = dict(payload.get("model_config", {}))
    if vision_directory is not None:
        config[
            (
                "visual_directory"
                if config.get("architecture") == "original_transformer"
                else "directory"
            )
        ] = str(vision_directory)
    model = build_model(config, pretrained=False)
    if payload.get("labels", model.labels) != model.labels:
        raise ValueError("Checkpoint labels do not match the model configuration")
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload["preprocess"]


def decode(logits, threshold=0.0, labels=None):
    schema = LABELS if labels is None else labels
    if logits.shape[1] != sum(len(values) for values in schema.values()):
        raise ValueError("Logit dimension does not match the attribute labels")
    predictions = [{} for _ in range(len(logits))]
    offset = 0
    for attribute, values in schema.items():
        probabilities = logits[:, offset : offset + len(values)].softmax(dim=1)
        confidence, indices = probabilities.max(dim=1)
        for prediction, index, score in zip(
            predictions, indices.tolist(), confidence.tolist(), strict=True
        ):
            prediction[attribute] = values[index] if score >= threshold else "unknown"
        offset += len(values)
    return predictions


def predict_image(checkpoint, image, device="cpu", threshold=0.0):
    model, preprocess = load_model(Path(checkpoint), device)
    with Image.open(image) as source:
        if getattr(model, "architecture", None) == "original_transformer":
            from .original_visual import OriginalVisualEncoder

            encoder = OriginalVisualEncoder(model.visual_directory, device)
            tensor = encoder.extract([source.convert("RGB")])[0].unsqueeze(0)
        else:
            tensor = image_transform(preprocess)(source.convert("RGB")).unsqueeze(0)
    with (
        torch.inference_mode(),
        torch.autocast(
            torch.device(device).type, dtype=torch.bfloat16, enabled=str(device).startswith("cuda")
        ),
    ):
        return decode(model(tensor.to(device)).cpu(), threshold, model.labels)[0]
