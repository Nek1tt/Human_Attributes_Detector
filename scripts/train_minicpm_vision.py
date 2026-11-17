"""Train the Transformer head, then fine-tune the last MiniCPM vision blocks."""

import subprocess
import sys

import click


def run(operation, *overrides):
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "silhouette_detector.commands",
            operation,
            "model=minicpm_vision",
            "preprocess=vision",
            "training=vision",
            *overrides,
        ],
        check=True,
    )


@click.command()
@click.option("--head-epochs", default=5, type=click.IntRange(min=1))
@click.option("--finetune-epochs", default=5, type=click.IntRange(min=1))
@click.option("--batch-size", default=32, type=click.IntRange(min=1))
def main(head_epochs, finetune_epochs, batch_size):
    run("vision-download")
    run(
        "train",
        f"training.epochs={head_epochs}",
        f"training.batch_size={batch_size}",
        "logging.run_name=minicpm-vision-head",
        "logging.plots_dir=plots/minicpm-head",
    )
    run(
        "train",
        f"training.epochs={finetune_epochs}",
        f"training.batch_size={batch_size}",
        "model.unfreeze_layers=2",
        "training.learning_rate=0.0001",
        "training.initial_checkpoint=models/minicpm_vision_head.pt",
        "training.output=models/minicpm_vision_finetuned.pt",
        "logging.run_name=minicpm-vision-finetuned",
        "logging.plots_dir=plots/minicpm-finetuned",
    )
    print("MiniCPM vision and Transformer training completed", flush=True)


if __name__ == "__main__":
    main()
