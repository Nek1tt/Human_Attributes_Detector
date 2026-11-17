"""Extract MiniCPM embeddings from a manifest without machine-specific paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import click
import torch
from PIL import Image

from ..attributes.labels import ATTRIBUTE_LABELS_RU
from ..attributes.minicpm import MiniCPMEmbeddingExtractor


def _encoded_labels(raw: dict[str, str]) -> dict[str, int]:
    encoded = {}
    for attribute, choices in ATTRIBUTE_LABELS_RU.items():
        value = raw.get(attribute, "не определен")
        if value not in choices:
            raise ValueError(f"Unknown label {value!r} for {attribute}")
        encoded[attribute] = choices.index(value)
    return encoded


def extract_manifest(
    manifest: Path,
    output_dir: Path,
    model_dir: Path,
    device: str,
    allow_unverified_model_code: bool,
) -> None:
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Dataset manifest must be a JSON list")
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = MiniCPMEmbeddingExtractor(
        model_dir,
        device,
        allow_unverified_model_code=allow_unverified_model_code,
    )
    for record in records:
        image_path = Path(record["image"]).expanduser().resolve()
        group_id = str(record.get("group_id") or image_path.parent.name)
        embedding = extractor.extract(Image.open(image_path).convert("RGB")).cpu()
        destination = output_dir / f"{uuid5(NAMESPACE_URL, str(image_path))}.pt"
        torch.save(
            {
                "embedding": embedding,
                "labels": _encoded_labels(record["labels"]),
                "group_id": group_id,
                "source": str(image_path),
            },
            destination,
        )


def parse_args() -> SimpleNamespace:
    @click.command()
    @click.argument("manifest", type=click.Path(path_type=Path))
    @click.argument("output_dir", type=click.Path(path_type=Path))
    @click.option("--model-dir", type=click.Path(path_type=Path), required=True)
    @click.option("--device", default="auto")
    @click.option("--allow-unverified-model-code", is_flag=True)
    def parse(**values):
        return SimpleNamespace(**values)

    result = parse.main(standalone_mode=False)
    if isinstance(result, int):
        raise SystemExit(result)
    return result


def main() -> None:
    args = parse_args()
    extract_manifest(
        args.manifest,
        args.output_dir,
        args.model_dir,
        args.device,
        args.allow_unverified_model_code,
    )


if __name__ == "__main__":
    main()
