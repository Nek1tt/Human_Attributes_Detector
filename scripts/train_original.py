"""Run experiments with the two unchanged architectures from the project proposal."""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import click
import torch
import yaml

from silhouette_detector.market_data import dvc_command
from silhouette_detector.original_visual import file_hash


def run(operation, *overrides):
    subprocess.run(
        [sys.executable, "-u", "-m", "silhouette_detector.commands", operation, *overrides],
        check=True,
    )


def import_weights(source, destination):
    destination = Path(destination)
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=True, mmap=True)
        if payload["source_sha256"] != file_hash(source):
            raise ValueError(f"Different initial checkpoint already exists: {destination}")
        return str(destination)
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("model_state_dict", payload)
    torch.save(
        {
            "model_state_dict": state,
            "source_filename": source.name,
            "source_sha256": file_hash(source),
        },
        destination,
    )
    dvc_command("add", destination)
    dvc_command("push", str(destination) + ".dvc")
    return str(destination)


def publish_best(selected, candidates):
    """Keep one final weight file per architecture and preserve all experiment logs."""
    destination = Path(f"models/original_{selected['family']}_peta.pt")
    source = Path(selected["checkpoint"])
    if source != destination:
        shutil.copyfile(source, destination)
        dvc_command("add", destination)
        dvc_command("push", str(destination) + ".dvc")
    keep_hash = yaml.safe_load(Path(str(destination) + ".dvc").read_text())["outs"][0]["md5"]
    root = Path.cwd().resolve()
    for row in candidates:
        if row.get("checkpoint") and Path(row["checkpoint"]) != destination:
            checkpoint = Path(row["checkpoint"])
            if not checkpoint.name.startswith("original-") or checkpoint.parent != Path("models"):
                raise ValueError("Refusing to remove a checkpoint outside this experiment")
            pointer = Path(str(checkpoint) + ".dvc")
            digest = yaml.safe_load(pointer.read_text())["outs"][0]["md5"]
            checkpoint.unlink()
            pointer.unlink()
            referenced = {
                item["md5"]
                for path in Path("models").glob("*.dvc")
                for item in yaml.safe_load(path.read_text())["outs"]
            }
            if digest != keep_hash and digest not in referenced:
                for directory in [".dvc/cache/files/md5", ".dvc-storage/files/md5"]:
                    cached = (root / directory / digest[:2] / digest[2:]).resolve()
                    if not cached.is_relative_to(root):
                        raise ValueError("Unsafe cache cleanup path")
                    if cached.exists():
                        cached.chmod(0o666)
                        cached.unlink()
        row["checkpoint"] = str(destination) if row["name"] == selected["name"] else None
        marker = Path(f"plots/{row['name']}/completed.json")
        marker.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return str(destination)


@click.command()
@click.option("--resnet-weights", type=click.Path(exists=True, path_type=Path))
@click.option("--transformer-weights", type=click.Path(exists=True, path_type=Path))
@click.option("--epochs", default=4, type=click.IntRange(min=1))
@click.option("--only", default="all", type=click.Choice(["all", "resnet", "transformer"]))
@click.option("--skip-extraction", is_flag=True)
@click.option("--restart", is_flag=True, help="Repeat completed experiments")
@click.option("--resnet-batch-size", default=8, type=click.IntRange(min=1))
@click.option("--transformer-batch-size", default=32, type=click.IntRange(min=1))
def main(
    resnet_weights,
    transformer_weights,
    epochs,
    only,
    skip_extraction,
    restart,
    resnet_batch_size,
    transformer_batch_size,
):
    initial = {}
    for name, source in [("resnet", resnet_weights), ("transformer", transformer_weights)]:
        imported = Path(f"models/original_{name}_init.pt")
        if source:
            initial[name] = import_weights(source, imported)
        else:
            pointer = Path(str(imported) + ".dvc")
            if pointer.exists() and not imported.exists():
                digest = yaml.safe_load(pointer.read_text())["outs"][0]["md5"]
                objects = [
                    Path(directory) / digest[:2] / digest[2:]
                    for directory in [".dvc/cache/files/md5", ".dvc-storage/files/md5"]
                ]
                if any(path.exists() for path in objects):
                    dvc_command("pull", pointer)
            initial[name] = str(imported) if imported.exists() else None
            if initial[name] is None:
                print(f"No old {name} weights: using public initialization", flush=True)
    if only in {"all", "transformer"} and not skip_extraction:
        run("original-download")
        run("original-extract")
    results = []
    for family in ["resnet", "transformer"] if only == "all" else [only]:
        batch_size = resnet_batch_size if family == "resnet" else transformer_batch_size
        architecture = "original_resnet50" if family == "resnet" else "original_transformer"
        for experiment, body_rate, head_rate, balanced in [
            ("standard", 0.00001 if family == "resnet" else 0.00005, 0.0003, False),
            ("balanced", 0.000003 if family == "resnet" else 0.00001, 0.0001, True),
        ]:
            name = f"original-{family}-{experiment}"
            output = f"models/{name}.pt"
            directory = f"plots/{name}"
            marker = Path(directory) / "completed.json"
            if not marker.exists() or restart:
                if shutil.disk_usage(".").free < 8 * 1024**3:
                    raise RuntimeError("Less than 8 GiB free before the next experiment")
                run(
                    "train",
                    f"model={architecture}",
                    "data=peta",
                    "preprocess=original_resnet",
                    "training=original",
                    f"training.epochs={epochs}",
                    f"training.batch_size={batch_size}",
                    f"model.initial_weights={initial[family] or 'null'}",
                    f"training.backbone_learning_rate={body_rate}",
                    f"training.learning_rate={head_rate}",
                    f"training.balanced_loss={str(balanced).lower()}",
                    f"training.output={output}",
                    f"logging.run_name={name}",
                    f"logging.plots_dir={directory}",
                )
                csv_path = sorted(Path(directory).glob("training/version_*/metrics.csv"))[-1]
                with csv_path.open(encoding="utf-8") as stream:
                    rows = [row for row in csv.DictReader(stream) if row.get("validation_accuracy")]
                best = max(rows, key=lambda row: float(row["validation_accuracy"]))
                marker.write_text(
                    json.dumps(
                        {
                            "name": name,
                            "family": family,
                            "checkpoint": output,
                            "validation_accuracy": float(best["validation_accuracy"]),
                            "validation_macro_f1": float(best["validation_macro_f1"]),
                            "epoch": int(float(best["epoch"])) + 1,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            results.append(json.loads(marker.read_text()))
        candidates = [row for row in results if row["family"] == family]
        selected = max(candidates, key=lambda row: row["validation_accuracy"])
        selected["checkpoint"] = publish_best(selected, candidates)
        run(
            "evaluate",
            f"inference.checkpoint={selected['checkpoint']}",
            "inference.device=cuda",
            f"inference.output=plots/original-{family}-test.json",
            "inference.batch_size=8" if family == "resnet" else "inference.batch_size=32",
        )
    Path("plots/original-experiments.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("Original architecture experiments completed", flush=True)


if __name__ == "__main__":
    main()
