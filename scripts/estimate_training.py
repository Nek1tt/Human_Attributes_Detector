"""Measure real training batches without saving models or changing existing weights."""

import gc
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

import click
import torch
from torch.nn import functional
from torch.utils.data import DataLoader

from silhouette_detector.commands import load_config
from silhouette_detector.market_data import MarketDataset
from silhouette_detector.market_model import build_model


def batch_loss(model, images, targets):
    output = model(images)
    offset = 0
    losses = []
    for index, labels in enumerate(model.labels.values()):
        dimension = len(labels)
        if (targets[:, index] != -100).any():
            losses.append(
                functional.cross_entropy(
                    output[:, offset : offset + dimension], targets[:, index], ignore_index=-100
                )
            )
        offset += dimension
    return torch.stack(losses).mean()


def measure(model, loader, optimizer, mixed, steps):
    training = optimizer is not None
    model.train(training)
    iterator = iter(loader)
    times = []
    for index in range(steps + 3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        images, targets = next(iterator)
        images, targets = images.cuda(), targets.cuda()
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed),
        ):
            loss = batch_loss(model, images, targets)
        if training:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        if index >= 3:
            times.append(time.perf_counter() - start)
    return {"median_seconds": statistics.median(times), "mean_seconds": statistics.mean(times)}


@click.command()
@click.option("--steps", default=8, type=click.IntRange(min=2))
@click.option("--epochs", default=5, type=click.IntRange(min=1))
@click.option("--output", default="var/training-estimate.json", type=click.Path(path_type=Path))
def main(steps, epochs, output):
    if not torch.cuda.is_available():
        raise RuntimeError("Run in the CUDA environment")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.hub.set_dir(str(Path(".cache/torch/hub").resolve()))
    report = {
        "gpu": torch.cuda.get_device_name(),
        "free_disk_bytes": shutil.disk_usage(".").free,
        "epochs_per_stage": epochs,
        "measured_batches": steps,
        "stages": {},
    }
    for name, overrides, batch_size, mixed in [
        ("resnet18", [], 64, False),
        ("minicpm_head", ["model=minicpm_vision", "preprocess=vision"], 32, True),
        (
            "minicpm_finetuned",
            ["model=minicpm_vision", "preprocess=vision", "model.unfreeze_layers=2"],
            32,
            True,
        ),
    ]:
        config = load_config(["data=peta", *overrides])
        model = build_model(config.model).cuda()
        train_set = MarketDataset(config.data.root, "train", config.preprocess)
        validation_set = MarketDataset(config.data.root, "validation", config.preprocess)
        options = {
            "batch_size": batch_size,
            "num_workers": 2,
            "pin_memory": True,
            "persistent_workers": True,
        }
        train_loader = DataLoader(train_set, shuffle=True, **options)
        validation_loader = DataLoader(validation_set, **options)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.0001
        )
        torch.cuda.reset_peak_memory_stats()
        training = measure(model, train_loader, optimizer, mixed, steps)
        validation = measure(model, validation_loader, None, mixed, steps)
        seconds_per_epoch = (
            math.ceil(len(train_set) / batch_size) * training["mean_seconds"]
            + math.ceil(len(validation_set) / batch_size) * validation["mean_seconds"]
        )
        report["stages"][name] = {
            "batch_size": batch_size,
            "train_images": len(train_set),
            "validation_images": len(validation_set),
            "train_batch": training,
            "validation_batch": validation,
            "seconds_per_epoch": seconds_per_epoch,
            "estimated_stage_seconds": seconds_per_epoch * epochs,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "weights_bytes": sum(
                value.numel() * value.element_size() for value in model.state_dict().values()
            ),
        }
        print(name, json.dumps(report["stages"][name]), flush=True)
        del model, optimizer, train_loader, validation_loader
        gc.collect()
        torch.cuda.empty_cache()
    report["estimated_training_seconds"] = sum(
        stage["estimated_stage_seconds"] for stage in report["stages"].values()
    )
    weights = report["stages"]["minicpm_head"]["weights_bytes"]
    baseline = report["stages"]["resnet18"]["weights_bytes"]
    report["additional_disk_peak_bytes"] = 7 * weights + 3 * baseline + 100 * 1024**2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
