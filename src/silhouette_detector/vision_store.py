"""Download only the pretrained vision tower from the official MiniCPM checkpoint."""

import hashlib
import json
import struct
import urllib.request
from pathlib import Path

REPOSITORY = "openbmb/MiniCPM-o-2_6"
REVISION = "cf0b69a6118cd3b2c90bb37f3f4366e2fba802ab"
SOURCE_HASHES = {
    "config.json": "917813ee571faf0447669b6b3549c8357c583141b5e4a8be1bcf9cc800a7c8cf",
    "modeling_navit_siglip.py": "f0a3e2fb52824ae06e30973c9e92abfdb441d0585674fa0cfc779bca4820c223",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def range_request(url, start, end):
    request = urllib.request.Request(
        url + f"?offset={start}", headers={"Range": f"bytes={start}-{end}"}
    )
    response = urllib.request.urlopen(request, timeout=120)
    if response.status != 206:
        response.close()
        raise RuntimeError("The model server did not support HTTP Range")
    return response


def download_vision(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/"
    for filename, checksum in SOURCE_HASHES.items():
        path = directory / filename
        if not path.exists():
            with urllib.request.urlopen(base + filename, timeout=120) as response:
                path.write_bytes(response.read())
        if sha256(path) != checksum:
            raise ValueError(f"MiniCPM source checksum mismatch: {filename}")
    weights = directory / "vision.safetensors"
    if not weights.exists():
        url = base + "model-00004-of-00004.safetensors"
        with range_request(url, 0, 7) as response:
            header_size = struct.unpack("<Q", response.read())[0]
        with range_request(url, 8, header_size + 7) as response:
            header = json.load(response)
        tensors = {
            name.removeprefix("vpm."): value
            for name, value in header.items()
            if name.startswith("vpm.")
        }
        start = min(value["data_offsets"][0] for value in tensors.values())
        end = max(value["data_offsets"][1] for value in tensors.values())
        if (
            sum(value["data_offsets"][1] - value["data_offsets"][0] for value in tensors.values())
            != end - start
        ):
            raise RuntimeError("Vision tensors are not contiguous in the source checkpoint")
        for value in tensors.values():
            value["data_offsets"] = [offset - start for offset in value["data_offsets"]]
        encoded = json.dumps(tensors, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        temporary = weights.with_suffix(".part")
        with range_request(url, 8 + header_size + start, 7 + header_size + end) as response:
            with temporary.open("wb") as destination:
                destination.write(struct.pack("<Q", len(encoded)))
                destination.write(encoded)
                downloaded = 0
                while chunk := response.read(8 * 1024 * 1024):
                    destination.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (64 * 1024 * 1024) == 0:
                        print(f"MiniCPM vision: {downloaded // (1024 * 1024)} MiB", flush=True)
                if downloaded != end - start:
                    raise RuntimeError("Incomplete MiniCPM vision download")
        temporary.replace(weights)
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "weights_sha256": sha256(weights),
        "source_sha256": SOURCE_HASHES,
    }
    (directory / "vision-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"MiniCPM vision weights: {weights}", flush=True)
    return directory


def verify_source(directory):
    directory = Path(directory)
    for filename, checksum in SOURCE_HASHES.items():
        if not (directory / filename).exists() or sha256(directory / filename) != checksum:
            raise ValueError(f"Run vision-download to restore the pinned MiniCPM file: {filename}")
