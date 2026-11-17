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
@click.option("--dataset", default="peta", type=click.Choice(["peta", "market"]))
def main(head_epochs, finetune_epochs, batch_size, dataset):
    head_path = f"models/{dataset}_vision_head.pt"
    final_path = f"models/{dataset}_vision_finetuned.pt"
    run("vision-download")
    run(
        "train",
        f"data={dataset}",
        f"training.epochs={head_epochs}",
        f"training.batch_size={batch_size}",
        f"training.output={head_path}",
        f"logging.run_name={dataset}-vision-head",
        f"logging.plots_dir=plots/{dataset}-head",
    )
    run(
        "train",
        f"data={dataset}",
        f"training.epochs={finetune_epochs}",
        f"training.batch_size={batch_size}",
        "model.unfreeze_layers=2",
        "training.learning_rate=0.0001",
        f"training.initial_checkpoint={head_path}",
        f"training.output={final_path}",
        f"logging.run_name={dataset}-vision-finetuned",
        f"logging.plots_dir=plots/{dataset}-finetuned",
    )
    print("MiniCPM vision and Transformer training completed", flush=True)


if __name__ == "__main__":
    main()
