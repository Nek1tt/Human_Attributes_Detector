"""Train attribute classifiers with Lightning and log the experiment to MLflow."""

import csv
import subprocess
from pathlib import Path

import lightning as lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassF1Score

from .market_data import MarketDataset, dvc_command, ensure_data
from .market_model import LABELS, MarketModel


class AttributeTraining(lightning.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.network = MarketModel(**config.model)
        self.learning_rate = config.training.learning_rate
        self.weight_decay = config.training.weight_decay
        self.f1_scores = nn.ModuleDict(
            {
                name: MulticlassF1Score(num_classes=len(labels), average="macro", ignore_index=-100)
                for name, labels in LABELS.items()
            }
        )

    def step(self, batch, stage):
        images, targets = batch
        logits = self.network(images)
        losses = []
        correct = logits.new_zeros(())
        count = logits.new_zeros(())
        offset = 0
        for index, (name, labels) in enumerate(LABELS.items()):
            output = logits[:, offset : offset + len(labels)]
            target = targets[:, index]
            valid = target != -100
            if valid.any():
                losses.append(functional.cross_entropy(output, target, ignore_index=-100))
            correct += ((output.argmax(dim=1) == target) & valid).sum()
            count += valid.sum()
            if stage == "validation":
                self.f1_scores[name].update(output, target)
            offset += len(labels)
        loss = torch.stack(losses).mean()
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
            filter(lambda parameter: parameter.requires_grad, self.network.parameters()),
            lr=self.learning_rate,
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
    torch.set_num_threads(config.training.threads)
    torch.hub.set_dir(str(Path(".cache/torch/hub").resolve()))
    lightning.seed_everything(config.training.seed, workers=True)
    root = ensure_data(config.data)
    train_set = MarketDataset(root, "train", config.preprocess)
    validation_set = MarketDataset(root, "validation", config.preprocess)
    loader_options = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
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
        dirpath="var/checkpoints",
        filename="market-best",
        monitor="validation_macro_f1",
        mode="max",
        save_top_k=1,
    )
    module = AttributeTraining(config)
    trainer = lightning.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        deterministic=config.training.deterministic,
        gradient_clip_val=config.training.gradient_clip_val,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        logger=[mlflow_logger, csv_logger],
        callbacks=[checkpoint],
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(module, train_loader, validation_loader)
    best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)
    module.load_state_dict(best["state_dict"])
    output = Path(config.training.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": module.network.cpu().state_dict(),
            "preprocess": OmegaConf.to_container(config.preprocess, resolve=True),
            "labels": LABELS,
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
    dvc_command("add", output)
    dvc_command("push", str(output) + ".dvc")
    print(f"Model: {output}; plots: {plots}; MLflow run: {mlflow_logger.run_id}")
