"""Download MiniCPM once, resolve a mutable revision, and record local code hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_snapshot(
    repo_id: str, revision: str, destination: Path
) -> dict[str, object]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the minicpm extra before downloading MiniCPM") from exc
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Destination is not empty: {destination}. "
            "Use a new directory to avoid mixing revisions."
        )
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError("Hugging Face did not return an immutable model revision")
    snapshot_download(repo_id, revision=info.sha, local_dir=destination)
    code_hashes = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*.py"))
    }
    manifest: dict[str, object] = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "python_files_sha256": code_hashes,
    }
    (destination / "model-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="openbmb/MiniCPM-o-2_6-int4")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--destination", type=Path, default=Path("models/minicpm-o-2_6-int4"))
    args = parser.parse_args()
    manifest = download_snapshot(args.repo_id, args.revision, args.destination)
    print(f"Saved immutable revision {manifest['resolved_revision']} to {args.destination}")


if __name__ == "__main__":
    main()
