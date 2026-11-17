"""The unchanged VPM and Resampler used by MiniCPM-o 2.6 INT4."""

import hashlib
import importlib.util
import json
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

import torch

from .model_store import PINNED_MINICPM_REPO, PINNED_MINICPM_REVISION, _patch_resampler_typing_list
from .vision_store import range_request

SOURCE_HASHES = {
    "config.json": "6b126bd590987009358195a6c9658b8876099b6f3c214cb5fd0e8379584443d6",
    "modeling_navit_siglip.py": "f0a3e2fb52824ae06e30973c9e92abfdb441d0585674fa0cfc779bca4820c223",
    "resampler.py": "1b304c316358b142c60ee546fe46a05c5c2f2fa6ba295662445db1093ea73e7b",
    "image_processing_minicpmv.py": (
        "be645337bd25811ae1f86e3aaa21fa5b555719870f5612004b4a67357148fe1f"
    ),
    "modeling_minicpmo.py": "706c968bc8f8d43652c555c7088b3869a35b4f973491227ed1481a5896160783",
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_original_visual(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{PINNED_MINICPM_REPO}/resolve/{PINNED_MINICPM_REVISION}/"
    for name in [*SOURCE_HASHES, "preprocessor_config.json"]:
        path = directory / name
        if not path.exists():
            with urllib.request.urlopen(base + name, timeout=120) as response:
                path.write_bytes(response.read())
        if name == "resampler.py":
            _patch_resampler_typing_list(directory)
        if name in SOURCE_HASHES and file_hash(path) != SOURCE_HASHES[name]:
            raise ValueError(f"Original MiniCPM code hash mismatch: {name}")
    weights = directory / "visual.safetensors"
    if not weights.exists():
        if shutil.disk_usage(directory).free < 5 * 1024**3:
            raise RuntimeError("Need 5 GiB free before downloading the original visual modules")
        url = base + "model.safetensors"
        with range_request(url, 0, 7) as response:
            header_size = struct.unpack("<Q", response.read())[0]
        with range_request(url, 8, 7 + header_size) as response:
            header = json.load(response)
        selected = {}
        ranges = []
        cursor = 0
        for prefix in ["vpm.", "resampler."]:
            tensors = {name: value for name, value in header.items() if name.startswith(prefix)}
            start = min(value["data_offsets"][0] for value in tensors.values())
            end = max(value["data_offsets"][1] for value in tensors.values())
            total = sum(
                value["data_offsets"][1] - value["data_offsets"][0] for value in tensors.values()
            )
            if total != end - start:
                raise ValueError(f"Non-contiguous original weights: {prefix}")
            for name, value in tensors.items():
                selected[name] = {
                    **value,
                    "data_offsets": [offset - start + cursor for offset in value["data_offsets"]],
                }
            ranges.append((start, end))
            cursor += total
        encoded = json.dumps(selected, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        temporary = weights.with_suffix(".part")
        with temporary.open("wb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            for start, end in ranges:
                with range_request(url, 8 + header_size + start, 7 + header_size + end) as response:
                    copied = 0
                    while chunk := response.read(8 * 1024 * 1024):
                        output.write(chunk)
                        copied += len(chunk)
                    if copied != end - start:
                        raise ValueError("Incomplete original MiniCPM tensor download")
                print(f"Original MiniCPM: downloaded {copied / 1024**2:.1f} MiB", flush=True)
        temporary.replace(weights)
    manifest = {
        "repository": PINNED_MINICPM_REPO,
        "revision": PINNED_MINICPM_REVISION,
        "weights_sha256": file_hash(weights),
        "source_sha256": SOURCE_HASHES,
    }
    (directory / "visual-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def source_module(directory, name):
    path = Path(directory) / f"{name}.py"
    if file_hash(path) != SOURCE_HASHES[path.name]:
        raise ValueError(f"Original MiniCPM code hash mismatch: {path.name}")
    spec = importlib.util.spec_from_file_location(f"original_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OriginalVisualEncoder(torch.nn.Module):
    def __init__(self, directory, device="cuda"):
        super().__init__()
        from safetensors.torch import load_file

        directory = Path(directory)
        if file_hash(directory / "config.json") != SOURCE_HASHES["config.json"]:
            raise ValueError("Original MiniCPM config hash mismatch")
        self.config = json.loads((directory / "config.json").read_text())
        vision = source_module(directory, "modeling_navit_siglip")
        resampler = source_module(directory, "resampler")
        processing = source_module(directory, "image_processing_minicpmv")
        self.vpm = vision.SiglipVisionTransformer(
            vision.SiglipVisionConfig(**self.config["vision_config"])
        )
        self.resampler = resampler.Resampler(
            num_queries=64, embed_dim=3584, num_heads=28, kv_dim=1152, adaptive=True
        )
        self.processor = processing.MiniCPMVImageProcessor(
            **json.loads((directory / "preprocessor_config.json").read_text())
        )
        self.load_state_dict(load_file(directory / "visual.safetensors"), strict=True)
        self.device_name = device
        self.to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
        self.requires_grad_(False).eval()

    def prepare(self, images):
        return self.processor.preprocess(
            [[image.convert("RGB")] for image in images], return_tensors="pt"
        )

    def encode_prepared(self, data):
        pixels = [
            image.flatten(end_dim=1).permute(1, 0)
            for images in data["pixel_values"]
            for image in images
        ]
        sizes = torch.vstack(data["tgt_sizes"]).to(self.device_name, dtype=torch.int32)
        padded = torch.nn.utils.rnn.pad_sequence(pixels, batch_first=True, padding_value=0.0)
        batch, length, _ = padded.shape
        padded = (
            padded.permute(0, 2, 1)
            .reshape(batch, 3, -1, length)
            .to(self.device_name, dtype=self.vpm.embeddings.patch_embedding.weight.dtype)
        )
        patch_counts = sizes[:, 0] * sizes[:, 1]
        mask = torch.zeros(
            batch, 1, int(patch_counts.max()), dtype=torch.bool, device=self.device_name
        )
        for index, count in enumerate(patch_counts):
            mask[index, 0, :count] = True
        features = torch.cat(
            [
                self.vpm(
                    padded[start : start + 16],
                    patch_attention_mask=mask[start : start + 16],
                    tgt_sizes=sizes[start : start + 16],
                ).last_hidden_state
                for start in range(0, batch, 16)
            ]
        )
        tokens = self.resampler(features, sizes)
        counts = [len(images) for images in data["pixel_values"]]
        return list(tokens.split(counts))

    @torch.inference_mode()
    def extract(self, images):
        with torch.autocast(
            "cuda" if self.device_name.startswith("cuda") else "cpu", enabled=False
        ):
            return self.encode_prepared(self.prepare(images))
