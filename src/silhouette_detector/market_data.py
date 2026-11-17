"""Download Market-1501, convert annotations, and keep the dataset in DVC."""

import hashlib
import json
import random
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .market_model import LABELS, image_transform


def download_file(url, destination, checksum=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    if checksum and hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Checksum mismatch: {destination}; remove the file and retry")


def encode_annotations(annotation):
    field_names = {
        "sleeve": "up",
        "lower_length": "down",
        "lower_type": "clothes",
    }
    encoded = []
    for attribute in LABELS:
        if attribute in {"top_color", "bottom_color"}:
            prefix = "up" if attribute == "top_color" else "down"
            matches = [
                i
                for i, color in enumerate(LABELS[attribute])
                if int(annotation[prefix + color]) == 2
            ]
            if len(matches) > 1:
                raise ValueError(f"Multiple colors for {attribute}: {matches}")
            encoded.append(matches[0] if matches else -100)
        else:
            encoded.append(int(annotation[field_names.get(attribute, attribute)]) - 1)
    return encoded


def prepare_records(root, config):
    from scipy.io import loadmat

    annotations = loadmat(root / "market_attribute.mat", simplify_cells=True)["market_attribute"]
    image_root = root / "Market-1501-v15.09.15"
    train_ids = sorted(int(value) for value in annotations["train"]["image_index"])
    random.Random(config.seed).shuffle(train_ids)
    count = max(1, round(len(train_ids) * config.validation_fraction))
    validation_ids = set(train_ids[:count])
    records = []
    for source_split, folder in [("train", "bounding_box_train"), ("test", "bounding_box_test")]:
        table = annotations[source_split]
        positions = {int(identity): i for i, identity in enumerate(table["image_index"])}
        for image in sorted((image_root / folder).glob("*.jpg")):
            identity = int(image.name.split("_")[0])
            if identity not in positions:
                continue
            position = positions[identity]
            annotation = {name: values[position] for name, values in table.items()}
            split = (
                "validation"
                if source_split == "train" and identity in validation_ids
                else source_split
            )
            records.append(
                {
                    "image": image.relative_to(root).as_posix(),
                    "group_id": identity,
                    "split": split,
                    "labels": encode_annotations(annotation),
                }
            )
    if not records:
        raise ValueError("The archive contains no Market-1501 images")
    (root / "records.json").write_text(json.dumps(records), encoding="utf-8")


def dvc_command(*arguments):
    import sys

    subprocess.run([sys.executable, "-m", "dvc", *map(str, arguments)], check=True)


def download_data(config):
    if config.get("schema", "market") == "peta":
        from .peta_data import download_peta

        return download_peta(config)
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    archive = Path("var/downloads/market.tar.gz")
    download_file(config.image_url, archive, config.image_sha256)
    download_file(config.annotation_url, root / "market_attribute.mat", config.annotation_sha256)
    marker = root / ".extracted"
    if not marker.exists():
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
        marker.touch()
    prepare_records(root, config)
    if not Path(".dvc").exists():
        dvc_command("init")
    dvc_command("remote", "add", "--force", "--default", "local", config.remote)
    dvc_command("add", root)
    dvc_command("push", str(root) + ".dvc")
    return root


def ensure_data(config):
    root = Path(config.root)
    pointer = Path(str(root) + ".dvc")
    if not pointer.exists() or not Path(config.remote).exists():
        download_data(config)
    dvc_command("pull", pointer)
    return root


class MarketDataset(Dataset):
    def __init__(self, root, split, preprocess, limit=None):
        self.root = Path(root)
        records = json.loads((self.root / "records.json").read_text(encoding="utf-8"))
        self.records = [record for record in records if record["split"] == split]
        if limit is not None:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"No records in split {split}")
        self.transform = image_transform(preprocess, training=split == "train")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch

        record = self.records[index]
        with Image.open(self.root / record["image"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(record["labels"], dtype=torch.long)
