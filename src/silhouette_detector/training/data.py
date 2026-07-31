"""Safe embedding dataset and a correct variable-length collator."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

# This RNG provides a reproducible dataset split; it is not used for security.
random.Random(seed).shuffle(group_ids)  # noqa: S311

class EmbeddingDataset(Dataset):
    def __init__(self, directory: Path) -> None:
        self.files = sorted(directory.glob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt embeddings found in {directory}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        payload = torch.load(self.files[index], map_location="cpu", weights_only=True)
        embedding = payload["embedding"].float()
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 3:
            raise ValueError(f"{self.files[index]} must contain embedding [N, P, D]")
        return embedding, dict(payload["labels"]), str(payload["group_id"])


def collate_embeddings(batch):
    embeddings, labels, groups = zip(*batch, strict=True)
    patches = {embedding.shape[1] for embedding in embeddings}
    dimensions = {embedding.shape[2] for embedding in embeddings}
    if len(patches) != 1 or len(dimensions) != 1:
        raise ValueError("All embeddings in a batch must share P and D dimensions")
    max_images = max(embedding.shape[0] for embedding in embeddings)
    padded = []
    masks = []
    for embedding in embeddings:
        valid_images = embedding.shape[0]
        pad_images = max_images - valid_images
        if pad_images:
            zeros = torch.zeros(
                pad_images, embedding.shape[1], embedding.shape[2], dtype=embedding.dtype
            )
            embedding = torch.cat([embedding, zeros], dim=0)
        padded.append(embedding)
        mask = torch.zeros(max_images, dtype=torch.bool)
        mask[:valid_images] = True
        masks.append(mask)
    return torch.stack(padded), list(labels), torch.stack(masks), list(groups)


def grouped_split(dataset: EmbeddingDataset, validation_fraction: float = 0.2, seed: int = 42):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        _, _, group_id = dataset[index]
        groups[group_id].append(index)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("At least two group_id values are required for a leakage-free split")
    random.Random(seed).shuffle(group_ids)
    validation_count = min(len(group_ids) - 1, max(1, round(len(group_ids) * validation_fraction)))
    validation_groups = set(group_ids[:validation_count])
    train_indices = [
        index
        for group in group_ids
        if group not in validation_groups
        for index in groups[group]
    ]
    validation_indices = [
        index
        for group in group_ids
        if group in validation_groups
        for index in groups[group]
    ]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)
