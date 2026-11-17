"""Train the Transformer head with a group-aware train/validation split."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..attributes.labels import ATTRIBUTE_SIZES_RU
from ..attributes.transformer import VisionAttrTransformer
from ..device import resolve_torch_device
from .data import EmbeddingDataset, collate_embeddings, grouped_split


def _epoch(model, loader, device: str, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for embeddings, labels, mask, _ in loader:
        embeddings, mask = embeddings.to(device), mask.to(device)
        outputs = model(embeddings, mask)
        loss = sum(
            F.cross_entropy(
                logits,
                torch.tensor([label[attribute] for label in labels], device=device),
            )
            for attribute, logits in outputs.items()
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.detach())
        for attribute, logits in outputs.items():
            target = torch.tensor([label[attribute] for label in labels], device=device)
            correct += int((logits.argmax(1) == target).sum())
            count += target.numel()
    return total_loss / max(1, len(loader)), correct / max(1, count)


def train(args) -> None:
    torch.manual_seed(args.seed)
    device = resolve_torch_device(args.device)
    dataset = EmbeddingDataset(args.embeddings)
    train_set, validation_set = grouped_split(dataset, args.validation_fraction, args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_embeddings
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embeddings
    )
    first_embedding, _, _ = dataset[0]
    model = VisionAttrTransformer(
        input_dim=first_embedding.shape[-1],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        attr_sizes=ATTRIBUTE_SIZES_RU,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    history = []
    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_accuracy = _epoch(model, train_loader, device, optimizer)
        with torch.inference_mode():
            validation_loss, validation_accuracy = _epoch(model, validation_loader, device)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(metrics)
        print(json.dumps(metrics))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "schema_version": 1,
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "input_dim": first_embedding.shape[-1],
                        "hidden_dim": args.hidden_dim,
                        "num_heads": args.num_heads,
                        "num_layers": args.num_layers,
                        "attr_sizes": ATTRIBUTE_SIZES_RU,
                        "drop_path_rate": args.drop_path_rate,
                    },
                    "metrics": metrics,
                },
                args.output,
            )
    args.output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> SimpleNamespace:
    @click.command()
    @click.argument("embeddings", type=click.Path(path_type=Path))
    @click.argument("output", type=click.Path(path_type=Path))
    @click.option("--device", default="auto")
    @click.option("--epochs", type=int, default=25)
    @click.option("--batch-size", type=int, default=16)
    @click.option("--hidden-dim", type=int, default=768)
    @click.option("--num-heads", type=int, default=12)
    @click.option("--num-layers", type=int, default=6)
    @click.option("--drop-path-rate", type=float, default=0.1)
    @click.option("--learning-rate", type=float, default=5e-05)
    @click.option("--validation-fraction", type=float, default=0.2)
    @click.option("--seed", type=int, default=42)
    def parse(**values):
        return SimpleNamespace(**values)

    result = parse.main(standalone_mode=False)
    if isinstance(result, int):
        raise SystemExit(result)
    return result


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
