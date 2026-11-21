"""Train attribute classifiers with Lightning and log the experiment to MLflow."""

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import lightning as lightning
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from lightning.pytorch.plugins.io import CheckpointIO
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

from .model import build_model
from .peta_data import ORIGINAL_PETA_LABELS, PetaImageDataset, ensure_data


class LocalCheckpointIO(CheckpointIO):
    """Write large checkpoints to disk without an extra in-memory archive."""

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(destination)

    def load_checkpoint(self, path, map_location=None):
        return torch.load(path, map_location=map_location, weights_only=True, mmap=True)

    def remove_checkpoint(self, path):
        Path(path).unlink(missing_ok=True)


class EpochSummary(Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_index):
        if (batch_index + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "epoch": trainer.current_epoch + 1,
                        "batch": batch_index + 1,
                        "train_batches": trainer.num_training_batches,
                    }
                ),
                flush=True,
            )

    def on_train_epoch_end(self, trainer, module):
        values = {name: float(value) for name, value in trainer.callback_metrics.items()}
        values["epoch"] = trainer.current_epoch + 1
        print(json.dumps(values), flush=True)


class AttributeTraining(lightning.LightningModule):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.network = build_model(config.model)
        self.labels = self.network.labels
        if config.training.get("initial_checkpoint"):
            payload = torch.load(
                config.training.initial_checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            self.network.load_state_dict(payload["model_state_dict"], strict=True)
        self.learning_rate = config.training.learning_rate
        self.backbone_learning_rate = config.training.get(
            "backbone_learning_rate", self.learning_rate
        )
        self.weight_decay = config.training.weight_decay
        self.loss_reduction = config.training.get("loss_reduction", "mean")
        self.weighted_loss = class_weights is not None
        if class_weights is not None:
            for index, weights in enumerate(class_weights):
                self.register_buffer(f"loss_weights_{index}", weights)
        self.f1_scores = nn.ModuleDict(
            {
                name: MulticlassF1Score(num_classes=len(labels), average="macro", ignore_index=-100)
                for name, labels in self.labels.items()
            }
        )

    def step(self, batch, stage):
        images, targets = batch
        logits = self.network(images)
        losses = []
        correct = logits.new_zeros((), dtype=torch.float32)
        count = logits.new_zeros((), dtype=torch.float32)
        offset = 0
        for index, (name, labels) in enumerate(self.labels.items()):
            output = logits[:, offset : offset + len(labels)]
            target = targets[:, index]
            valid = target != -100
            if valid.any():
                weights = getattr(self, f"loss_weights_{index}") if self.weighted_loss else None
                losses.append(
                    functional.cross_entropy(output, target, weight=weights, ignore_index=-100)
                )
            correct += ((output.argmax(dim=1) == target) & valid).sum()
            count += valid.sum()
            if stage == "validation":
                self.f1_scores[name].update(output, target)
            offset += len(labels)
        loss = (
            torch.stack(losses).sum()
            if self.loss_reduction == "sum"
            else torch.stack(losses).mean()
        )
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, batch_size=len(images))
        self.log(
            f"{stage}_accuracy",
            correct / count.clamp_min(1),
            on_step=False,
            on_epoch=True,
            batch_size=len(images),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "validation")

    def on_validation_epoch_end(self):
        self.log(
            "validation_macro_f1",
            torch.stack([metric.compute() for metric in self.f1_scores.values()]).mean(),
        )
        for metric in self.f1_scores.values():
            metric.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.network.parameter_groups(self.learning_rate, self.backbone_learning_rate),
            weight_decay=self.weight_decay,
        )


def save_plots(csv_path, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    with Path(csv_path).open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for name, metrics in {
        "loss": ["train_loss", "validation_loss"],
        "accuracy": ["train_accuracy", "validation_accuracy"],
        "macro_f1": ["validation_macro_f1"],
    }.items():
        figure, axis = pyplot.subplots(figsize=(6, 4))
        for metric in metrics:
            points = [
                (int(float(row["epoch"])) + 1, float(row[metric]))
                for row in rows
                if row.get(metric) and row.get("epoch")
            ]
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=metric,
            )
        axis.set(xlabel="Epoch", ylabel=name)
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / f"{name}.png", dpi=140)
        pyplot.close(figure)


def train(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.training.threads)
    torch.hub.set_dir(str(Path(".cache/torch/hub").resolve()))
    lightning.seed_everything(config.training.seed, workers=True)
    root = ensure_data(config.data)
    if config.model.get("architecture") == "original_transformer":
        from .original_features import OriginalEmbeddingDataset, ensure_embeddings

        embedding_root = ensure_embeddings(config)
        train_set = OriginalEmbeddingDataset(embedding_root, "train")
        validation_set = OriginalEmbeddingDataset(embedding_root, "validation")
    else:
        train_set = PetaImageDataset(root, "train", config.preprocess)
        validation_set = PetaImageDataset(root, "validation", config.preprocess)
    loader_options = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.accelerator == "gpu",
        "persistent_workers": config.training.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    mlflow_logger = MLFlowLogger(
        experiment_name=config.logging.experiment,
        run_name=config.logging.run_name,
        tracking_uri=config.logging.tracking_uri,
        tags={"git_commit": git_commit, "git_dirty": str(git_dirty)},
    )
    mlflow_logger.log_hyperparams({"git_commit": git_commit, "git_dirty": str(git_dirty)})
    csv_logger = CSVLogger(config.logging.plots_dir, name="training")
    checkpoint = ModelCheckpoint(
        dirpath=Path("var/checkpoints") / config.logging.run_name,
        filename="best",
        monitor=config.training.get("monitor", "validation_macro_f1"),
        mode="max",
        save_top_k=1,
        save_weights_only=True,
    )
    class_weights = None
    if config.training.get("balanced_loss", False):
        class_weights = []
        for index, choices in enumerate(ORIGINAL_PETA_LABELS.values()):
            targets = torch.tensor([record["labels"][index] for record in train_set.records])
            counts = torch.bincount(targets[targets >= 0], minlength=len(choices)).float()
            weights = (counts.sum() / (len(choices) * counts.clamp_min(1))).sqrt()
            class_weights.append(weights / weights.mean())
    module = AttributeTraining(config, class_weights)
    if len(train_set.records[0]["labels"]) != len(module.labels):
        raise ValueError("Training data and model have different attribute schemas")
    weights_bytes = sum(
        value.numel() * value.element_size() for value in module.network.state_dict().values()
    )
    required_bytes = 4 * weights_bytes + 4 * 1024**3
    if shutil.disk_usage(".").free < required_bytes:
        raise RuntimeError(f"Training needs at least {required_bytes / 1024**3:.1f} GiB free")
    trainer = lightning.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        deterministic=config.training.deterministic,
        precision=config.training.get("precision", "32-true"),
        gradient_clip_val=config.training.gradient_clip_val,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        logger=[mlflow_logger, csv_logger],
        callbacks=[
            checkpoint,
            EpochSummary(),
            *(
                [
                    EarlyStopping(
                        monitor=config.training.monitor,
                        mode="max",
                        patience=config.training.patience,
                    )
                ]
                if config.training.get("patience") is not None
                else []
            ),
        ],
        plugins=[LocalCheckpointIO()],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        log_every_n_steps=1,
        accumulate_grad_batches=config.training.get("accumulate_grad_batches", 1),
    )
    trainer.fit(module, train_loader, validation_loader)
    best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True, mmap=True)
    module.load_state_dict(best["state_dict"])
    del best
    output = Path(config.training.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    module.zero_grad(set_to_none=True)
    torch.save(
        {
            "model_state_dict": module.network.cpu().state_dict(),
            "preprocess": OmegaConf.to_container(config.preprocess, resolve=True),
            "labels": module.labels,
            "model_config": OmegaConf.to_container(config.model, resolve=True),
            "git_commit": git_commit,
        },
        output,
    )
    plots = Path(csv_logger.log_dir)
    save_plots(plots / "metrics.csv", plots)
    OmegaConf.save(config, plots / "config.yaml")
    (plots / "code-version.txt").write_text(
        f"git_commit: {git_commit}\ngit_dirty: {git_dirty}\n", encoding="utf-8"
    )
    mlflow_logger.experiment.log_artifacts(mlflow_logger.run_id, str(plots))
    Path(checkpoint.best_model_path).unlink(missing_ok=True)
    print(f"Model: {output}; plots: {plots}; MLflow run: {mlflow_logger.run_id}")
