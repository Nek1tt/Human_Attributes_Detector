"""Extract the original 64 x 3584 MiniCPM tokens once and store them in DVC."""

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .market_data import MarketDataset, dvc_command, ensure_data
from .original_visual import OriginalVisualEncoder, file_hash
from .peta_data import ORIGINAL_PETA_LABELS


def unknown_targets(records):
    for record in records:
        record["labels"] = [
            target if target >= 0 else len(choices) - 1
            for target, choices in zip(record["labels"], ORIGINAL_PETA_LABELS.values(), strict=True)
        ]
    return records


class OriginalImageDataset(MarketDataset):
    def __init__(self, root, split, preprocess, limit=None):
        super().__init__(root, split, preprocess, limit)
        self.records = unknown_targets(self.records)


def extract_original_features(config):
    image_root = ensure_data(config.data)
    root = Path(config.embeddings.root)
    root.mkdir(parents=True, exist_ok=True)
    records_path = image_root / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    source_hash = file_hash(records_path)
    if metadata and metadata["records_sha256"] != source_hash:
        raise ValueError("PETA records changed; use a new embedding directory")
    if metadata.get("complete"):
        return root
    expected_bytes = len(records) * 64 * 3584 * 2
    available = shutil.disk_usage(root).free
    if not metadata and available < 3 * expected_bytes + config.embeddings.reserve_gib * 1024**3:
        raise RuntimeError(
            "Insufficient space for embeddings, DVC copies and the configured reserve"
        )
    data_path = root / "tokens.npy"
    data = np.lib.format.open_memmap(
        data_path,
        mode="r+" if data_path.exists() else "w+",
        dtype=np.uint16,
        shape=(len(records), 64, 3584),
    )
    completed = metadata.get("completed", 0)
    encoder = OriginalVisualEncoder(config.embeddings.visual_directory, config.embeddings.device)
    start = time.perf_counter()
    initial_completed = completed
    for offset in range(completed, len(records), config.embeddings.batch_size):
        batch = records[offset : offset + config.embeddings.batch_size]
        images = []
        for record in batch:
            with Image.open(image_root / record["image"]) as image:
                images.append(image.convert("RGB"))
        tokens = encoder.extract(images)
        for index, tensor in enumerate(tokens):
            if tensor.shape != (1, 64, 3584):
                raise ValueError(
                    f"Expected one crop with 64 x 3584 tokens: {batch[index]['image']}"
                )
            data[offset + index] = tensor[0].cpu().contiguous().view(torch.uint16).numpy()
        completed = offset + len(batch)
        data.flush()
        metadata = {
            "records_sha256": source_hash,
            "completed": completed,
            "complete": False,
            "shape": [len(records), 64, 3584],
            "dtype": "bfloat16",
            "visual_manifest_sha256": file_hash(
                Path(config.embeddings.visual_directory) / "visual-manifest.json"
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if completed % 256 < config.embeddings.batch_size or completed == len(records):
            elapsed = time.perf_counter() - start
            rate = (completed - initial_completed) / elapsed
            print(
                f"Original MiniCPM embeddings: {completed}/{len(records)}, {rate:.1f} images/s, "
                f"remaining {(len(records)-completed)/rate/60:.1f} min",
                flush=True,
            )
        if shutil.disk_usage(root).free < config.embeddings.reserve_gib * 1024**3:
            raise RuntimeError(
                "Stopped extraction at the disk reserve; completed rows are preserved"
            )
    del data
    (root / "records.json").write_text(json.dumps(records) + "\n", encoding="utf-8")
    metadata["complete"] = True
    metadata["tokens_sha256"] = file_hash(data_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    dvc_command("add", root)
    dvc_command("push", str(root) + ".dvc")
    return root


def ensure_embeddings(config):
    root = Path(config.embeddings.root)
    pointer = Path(str(root) + ".dvc")
    if pointer.exists():
        try:
            dvc_command("pull", pointer)
        except subprocess.CalledProcessError:
            print(
                "Cached embeddings are unavailable; rebuilding from the public PETA images",
                flush=True,
            )
    if not (Path(config.embeddings.visual_directory) / "visual.safetensors").exists():
        from .original_visual import download_original_visual

        download_original_visual(config.embeddings.visual_directory)
    if not (root / "metadata.json").exists():
        return extract_original_features(config)
    if not json.loads((root / "metadata.json").read_text())["complete"]:
        return extract_original_features(config)
    return root


class OriginalEmbeddingDataset(Dataset):
    def __init__(self, root, split, limit=None):
        self.root = Path(root)
        records = json.loads((self.root / "records.json").read_text(encoding="utf-8"))
        self.records = unknown_targets(
            [
                {**record, "embedding_index": index}
                for index, record in enumerate(records)
                if record["split"] == split
            ]
        )
        if limit is not None:
            self.records = self.records[:limit]
        self.data = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.data is None:
            self.data = np.load(self.root / "tokens.npy", mmap_mode="r")
        record = self.records[index]
        values = torch.from_numpy(np.array(self.data[record["embedding_index"]], copy=True))
        return values.view(torch.bfloat16).float().unsqueeze(0), torch.tensor(
            record["labels"], dtype=torch.long
        )

    def __getstate__(self):
        return {**self.__dict__, "data": None}
