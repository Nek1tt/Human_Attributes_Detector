"""Predict attributes of cropped people without importing training code."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .market_data import MarketDataset, dvc_command, ensure_data
from .market_model import decode, load_model, predict_image


def ensure_checkpoint(checkpoint):
    checkpoint = Path(checkpoint)
    pointer = Path(str(checkpoint) + ".dvc")
    if pointer.exists():
        dvc_command("pull", pointer)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train the model first or restore the checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    directory = payload.get("model_config", {}).get("directory")
    if directory and not Path(directory).exists():
        dvc_command("pull", str(directory) + ".dvc")
    return checkpoint


def infer(config):
    root = ensure_data(config.data)
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
    model, preprocess = load_model(checkpoint, config.inference.device)
    dataset = MarketDataset(root, config.inference.split, preprocess, config.inference.limit)
    loader = DataLoader(dataset, batch_size=config.inference.batch_size)
    offset = 0
    with output.open("w", encoding="utf-8") as destination, torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(config.inference.device)).cpu()
            predictions = decode(logits, config.inference.confidence_threshold)
            for prediction in predictions:
                record = dataset.records[offset]
                destination.write(
                    json.dumps({"image": record["image"], "attributes": prediction}) + "\n"
                )
                offset += 1
    print(f"Saved {offset} predictions to {output}")
