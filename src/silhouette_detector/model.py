"""Attribute model factory, checkpoint loading and image prediction."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def image_transform(config, training=False):
    operations = [transforms.Resize((config["height"], config["width"]))]
    if training:
        operations.append(transforms.RandomHorizontalFlip(config["horizontal_flip"]))
    operations.extend([transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    return transforms.Compose(operations)


def build_model(config, pretrained=None):
    from .original_models import OriginalClassifier
    from .peta_data import ORIGINAL_PETA_LABELS

    options = dict(config)
    options.pop("output", None)
    schema = options.pop("label_schema", "peta_original")
    if schema != "peta_original":
        raise ValueError(f"Unsupported label schema: {schema}. Use the final PETA checkpoint.")
    if pretrained is not None:
        options["pretrained"] = pretrained
    return OriginalClassifier(labels=ORIGINAL_PETA_LABELS, **options)


def load_model(checkpoint: Path, device="cpu", vision_directory=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = dict(payload.get("model_config", {}))
    if vision_directory is not None and config.get("architecture") == "original_transformer":
        config["visual_directory"] = str(vision_directory)
    model = build_model(config, pretrained=False)
    if payload.get("labels", model.labels) != model.labels:
        raise ValueError("Checkpoint labels do not match the model configuration")
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload["preprocess"]


def decode(logits, threshold=0.0, *, labels):
    if logits.shape[1] != sum(len(values) for values in labels.values()):
        raise ValueError("Logit dimension does not match the attribute labels")
    predictions = [{} for _ in range(len(logits))]
    offset = 0
    for attribute, values in labels.items():
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
        return decode(model(tensor.to(device)).cpu(), threshold, labels=model.labels)[0]
