"""Download the pinned release classifiers and verify their SHA-256 identities."""

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def matches(path, asset):
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == asset["sha256"]


def download_asset(url, destination, asset):
    if destination.exists():
        if matches(destination, asset):
            print(f"Verified: {destination.name}")
            return
        raise FileExistsError(f"Preserving existing file with a different hash: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(  # noqa: S310 - URL comes from the pinned manifest.
            url, headers={"User-Agent": "HAD-release-downloader"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,  # noqa: S310
            tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output,
        ):
            temporary = Path(output.name)
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
        if not matches(temporary, asset):
            raise ValueError(f"Release asset size or SHA-256 mismatch: {destination.name}")
        # rename refuses to replace an existing file on Windows.
        temporary.rename(destination)
        print(f"Downloaded and verified: {destination.name}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    manifest = json.loads((ROOT / "docs/release-weights.json").read_text(encoding="utf-8"))
    base = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}/"
    for asset in manifest["assets"]:
        download_asset(base + asset["name"], ROOT / "models" / asset["name"], asset)


if __name__ == "__main__":
    main()
