"""Run checks, real-data training, inference, export, and MLflow packaging."""

import base64
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click


def run(*arguments):
    print(" ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], check=True)


@click.command()
@click.option("--full-training", is_flag=True, help="Train all batches for five epochs")
def main(full_training):
    run("-m", "pre_commit", "run", "--all-files")
    run("-m", "unittest", "discover", "-s", "tests", "-v")
    overrides = (
        []
        if full_training
        else [
            "training.epochs=2",
            "training.limit_train_batches=2",
            "training.limit_val_batches=2",
            "training.accelerator=cpu",
            "training.batch_size=1",
            "training.num_workers=0",
            "training.output=models/smoke_attributes.pt",
            "logging.plots_dir=var/smoke-plots",
            "logging.run_name=smoke",
        ]
    )
    run("-m", "silhouette_detector.commands", "train", *overrides)
    checkpoint = "models/original_resnet_peta.pt" if full_training else "models/smoke_attributes.pt"
    serving_path = "var/check-serving-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run(
        "-m",
        "silhouette_detector.commands",
        "infer",
        f"inference.checkpoint={checkpoint}",
        "inference.limit=3",
    )
    run(
        "-m",
        "silhouette_detector.commands",
        "export",
        f"inference.checkpoint={checkpoint}",
        "production.onnx=var/check-model.onnx",
    )
    run(
        "-m",
        "silhouette_detector.commands",
        "package",
        f"inference.checkpoint={checkpoint}",
        "production.mlflow_model=" + serving_path,
    )
    import mlflow.pyfunc

    from silhouette_detector.market_model import predict_image

    image = next(Path("data/peta/PETA dataset").rglob("*.jpg"))
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    packaged = mlflow.pyfunc.load_model(serving_path).predict({"image": [encoded]})
    if packaged != [predict_image(checkpoint, image)]:
        raise ValueError("Packaged and local predictions differ")
    print("Checks completed")


if __name__ == "__main__":
    main()
