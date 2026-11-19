"""Evaluate every attribute on a held-out split and save its confusion matrix."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .inference import ensure_checkpoint
from .model import load_model
from .peta_data import PetaImageDataset, ensure_data


def evaluate(config):
    root = ensure_data(config.data)
    model, preprocess = load_model(
        ensure_checkpoint(config.inference.checkpoint), config.inference.device
    )
    if getattr(model, "architecture", None) == "original_transformer":
        from .original_features import OriginalEmbeddingDataset, ensure_embeddings

        dataset = OriginalEmbeddingDataset(
            ensure_embeddings(config), config.inference.split, config.inference.limit
        )
    else:
        dataset = PetaImageDataset(root, config.inference.split, preprocess, config.inference.limit)
    loader = DataLoader(dataset, batch_size=config.inference.batch_size)
    matrices = {
        name: torch.zeros(len(labels), len(labels), dtype=torch.int64)
        for name, labels in model.labels.items()
    }
    if len(dataset.records[0]["labels"]) != len(model.labels):
        raise ValueError("Dataset and checkpoint have different attribute schemas")
    with torch.inference_mode():
        for images, targets in loader:
            with torch.autocast(
                torch.device(config.inference.device).type,
                dtype=torch.bfloat16,
                enabled=config.inference.device.startswith("cuda"),
            ):
                logits = model(images.to(config.inference.device)).cpu()
            offset = 0
            for index, (name, labels) in enumerate(model.labels.items()):
                dimension = len(labels)
                predicted = logits[:, offset : offset + dimension].argmax(dim=1)
                target = targets[:, index]
                valid = target != -100
                matrices[name] += torch.bincount(
                    target[valid] * dimension + predicted[valid], minlength=dimension**2
                ).reshape(dimension, dimension)
                offset += dimension
    report = {
        "checkpoint": str(config.inference.checkpoint),
        "split": config.inference.split,
        "images": len(dataset),
        "attributes": {},
    }
    total_correct = 0
    total_support = 0
    for name, matrix in matrices.items():
        true_positive = matrix.diag().double()
        support = matrix.sum(dim=1)
        precision = true_positive / matrix.sum(dim=0).clamp_min(1)
        recall = true_positive / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        report["attributes"][name] = {
            "accuracy": float(true_positive.sum() / support.sum().clamp_min(1)),
            "macro_f1": float(f1[(matrix.sum(dim=0) + matrix.sum(dim=1)) > 0].mean()),
            "classes": [
                {
                    "label": label,
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, label in enumerate(model.labels[name])
            ],
            "confusion_matrix": matrix.tolist(),
        }
        total_correct += int(true_positive.sum())
        total_support += int(support.sum())
    report["accuracy"] = total_correct / max(1, total_support)
    report["macro_f1"] = sum(row["macro_f1"] for row in report["attributes"].values()) / len(
        matrices
    )
    output = Path(config.inference.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Evaluation: {output}; accuracy={report['accuracy']:.4f}; "
        f"macro_f1={report['macro_f1']:.4f}"
    )
