"""Project commands; configuration overrides use Hydra syntax."""

import sys
from pathlib import Path

import click
from hydra import compose, initialize_config_dir


def load_config(overrides):
    directory = Path.cwd() / "configs"
    with initialize_config_dir(config_dir=str(directory.resolve()), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


@click.command()
@click.argument(
    "operation",
    type=click.Choice(
        [
            "download",
            "vision-download",
            "original-download",
            "original-extract",
            "train",
            "infer",
            "evaluate",
            "export",
            "package",
            "tensorrt",
            "triton",
        ]
    ),
)
@click.argument("overrides", nargs=-1)
def main(operation, overrides):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    config = load_config(overrides)
    if operation == "original-download":
        from .market_data import dvc_command
        from .original_visual import download_original_visual

        directory = download_original_visual(config.embeddings.visual_directory)
        dvc_command("add", directory)
        dvc_command("push", str(directory) + ".dvc")
    elif operation == "original-extract":
        from .original_features import extract_original_features

        extract_original_features(config)
    elif operation == "vision-download":
        from .market_data import dvc_command
        from .vision_store import download_vision

        directory = download_vision(config.model.directory)
        dvc_command("add", directory)
        dvc_command("push", str(directory) + ".dvc")
    elif operation == "download":
        from .market_data import download_data

        download_data(config.data)
    elif operation == "train":
        from .market_training import train

        train(config)
    elif operation == "infer":
        from .market_inference import infer

        infer(config)
    elif operation == "evaluate":
        from .evaluation import evaluate

        evaluate(config)
    else:
        from .production import export_onnx, package_mlflow, package_triton, to_tensorrt

        actions = {
            "export": export_onnx,
            "package": package_mlflow,
            "tensorrt": to_tensorrt,
            "triton": package_triton,
        }
        actions[operation](config)


if __name__ == "__main__":
    main()
