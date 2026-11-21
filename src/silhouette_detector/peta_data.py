"""Convert the original PETA annotations to nine categorical attributes."""

import hashlib
import json
import random
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .model import image_transform

COLORS = [
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
]
PETA_LABELS = {
    "gender": ["male", "female"],
    "age": ["less15", "16-30", "31-45", "46-60", "over60"],
    "headwear": ["no", "yes"],
    "top_color": COLORS,
    "bottom_color": COLORS,
    "top_type": ["jacket", "sweater", "suit", "tshirt", "other"],
    "bottom_type": ["trousers", "shorts", "skirt", "capri"],
    "backpack": ["no", "yes"],
    "handbag": ["no", "yes"],
}

ORIGINAL_PETA_LABELS = {name: [*choices, "unknown"] for name, choices in PETA_LABELS.items()}


def one_class(tags, groups):
    matches = [index for index, names in enumerate(groups) if tags.intersection(names)]
    return matches[0] if len(matches) == 1 else -100


def encode_peta(tags):
    """Exclude absent or conflicting categorical labels from loss and metrics."""
    tags = set(tags)
    return [
        one_class(tags, [{"personalMale"}, {"personalFemale"}]),
        one_class(
            tags,
            [
                {name}
                for name in [
                    "personalLess15",
                    "personalLess30",
                    "personalLess45",
                    "personalLess60",
                    "personalLarger60",
                ]
            ],
        ),
        int(bool(tags & {"accessoryHat", "accessoryKerchief"})),
        one_class(tags, [{"upperBody" + color.title()} for color in COLORS]),
        one_class(tags, [{"lowerBody" + color.title()} for color in COLORS]),
        one_class(tags, [{"upperBody" + kind.title()} for kind in PETA_LABELS["top_type"]]),
        one_class(
            tags,
            [
                {"lowerBodyTrousers", "lowerBodyJeans", "lowerBodySuits"},
                {"lowerBodyShorts", "lowerBodyHotPants"},
                {"lowerBodyShortSkirt", "lowerBodyLongSkirt"},
                {"lowerBodyCapri"},
            ],
        ),
        int("carryingBackpack" in tags),
        int(
            bool(
                tags
                & {
                    "carryingMessengerBag",
                    "carryingPlasticBags",
                    "carryingLuggageCase",
                    "carryingSuitcase",
                }
            )
        ),
    ]


def person_id(filename):
    return int(Path(filename).stem.split("_")[0])


def prepare_peta_records(root, config):
    records = []
    for annotation_file in sorted((root / "PETA dataset").glob("*/archive/Label.txt")):
        subset = annotation_file.parents[1].name
        if subset in config.excluded_subsets:
            continue
        annotations = {}
        for line in annotation_file.read_text(encoding="utf-8-sig").splitlines():
            fields = line.split()
            if fields:
                identity = person_id(fields[0])
                tags = set(fields[1:])
                if identity in annotations and tags != annotations[identity]:
                    raise ValueError(f"Conflicting annotations: {subset}/{identity}")
                annotations[identity] = tags
        identities = sorted(annotations)
        random.Random(f"{config.seed}:{subset}").shuffle(identities)
        validation_count = max(1, round(len(identities) * config.validation_fraction))
        test_count = max(1, round(len(identities) * config.test_fraction))
        validation_ids = set(identities[:validation_count])
        test_ids = set(identities[validation_count : validation_count + test_count])
        for image in sorted(annotation_file.parent.iterdir()):
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            identity = person_id(image.name)
            if identity not in annotations:
                raise ValueError(f"Missing annotation: {image}")
            split = (
                "validation"
                if identity in validation_ids
                else "test" if identity in test_ids else "train"
            )
            records.append(
                {
                    "image": image.relative_to(root).as_posix(),
                    "group_id": f"{subset}/{identity}",
                    "split": split,
                    "labels": encode_peta(annotations[identity]),
                }
            )
    if not records:
        raise ValueError("No annotated PETA images found")
    (root / "records.json").write_text(json.dumps(records), encoding="utf-8")
    return records


def download_peta(config):
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = Path("var/downloads/peta.zip")
    download_file(config.image_url, archive_path, config.image_sha256)
    if not (root / ".extracted").exists():
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if not destination.is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                if (
                    len(Path(member.filename).parts) > 1
                    and Path(member.filename).parts[1] in config.excluded_subsets
                ):
                    continue
                archive.extract(member, root)
        (root / ".extracted").touch()
    prepare_peta_records(root, config)
    if not Path(".dvc").exists():
        dvc_command("init")
    dvc_command("remote", "add", "--force", "--local", "local", config.remote)
    dvc_command("add", root)
    dvc_command("push", "-r", "local", str(root) + ".dvc")
    return root


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


def dvc_command(*arguments):
    import sys

    subprocess.run([sys.executable, "-m", "dvc", *map(str, arguments)], check=True)


def ensure_data(config):
    root = Path(config.root)
    pointer = Path(str(root) + ".dvc")
    if not pointer.exists() or not Path(config.remote).exists():
        download_peta(config)
    try:
        dvc_command("pull", "-r", "local", pointer)
    except subprocess.CalledProcessError:
        download_peta(config)
        dvc_command("pull", "-r", "local", pointer)
    return root


def unknown_targets(records):
    for record in records:
        record["labels"] = [
            target if target >= 0 else len(choices) - 1
            for target, choices in zip(record["labels"], ORIGINAL_PETA_LABELS.values(), strict=True)
        ]
    return records


class PetaImageDataset(Dataset):
    def __init__(self, root, split, preprocess, limit=None):
        self.root = Path(root)
        records = json.loads((self.root / "records.json").read_text(encoding="utf-8"))
        self.records = [record for record in records if record["split"] == split]
        if limit is not None:
            self.records = self.records[:limit]
        self.records = unknown_targets(self.records)
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
