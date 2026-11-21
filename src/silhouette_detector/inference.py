"""Predict attributes of cropped people without importing training code."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import decode, load_model, predict_image
from .peta_data import PetaImageDataset, ensure_data


def ensure_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Run python scripts/download_release_weights.py or train the model: {checkpoint}"
        )
    return checkpoint


def infer(config):
    checkpoint = ensure_checkpoint(config.inference.checkpoint)
    output = Path(config.inference.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if config.inference.image:
        prediction = predict_image(
            checkpoint,
            config.inference.image,
            config.inference.device,
            config.inference.confidence_threshold,
        )
        output.write_text(
            json.dumps({"image": config.inference.image, "attributes": prediction}) + "\n",
            encoding="utf-8",
        )
        print(output)
        return
    root = ensure_data(config.data)
    model, preprocess = load_model(checkpoint, config.inference.device)
    if getattr(model, "architecture", None) == "original_transformer":
        from .original_features import OriginalEmbeddingDataset, ensure_embeddings

        dataset = OriginalEmbeddingDataset(
            ensure_embeddings(config), config.inference.split, config.inference.limit
        )
    else:
        dataset = PetaImageDataset(root, config.inference.split, preprocess, config.inference.limit)
    loader = DataLoader(dataset, batch_size=config.inference.batch_size)
    offset = 0
    with output.open("w", encoding="utf-8") as destination, torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(config.inference.device)).cpu()
            predictions = decode(logits, config.inference.confidence_threshold, labels=model.labels)
            for prediction in predictions:
                record = dataset.records[offset]
                destination.write(
                    json.dumps({"image": record["image"], "attributes": prediction}) + "\n"
                )
                offset += 1
    print(f"Saved {offset} predictions to {output}")
