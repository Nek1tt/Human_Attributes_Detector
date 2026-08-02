"""Download MiniCPM once, resolve a mutable revision, and record local code hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

PINNED_MINICPM_REPO = "openbmb/MiniCPM-o-2_6-int4"
PINNED_MINICPM_REVISION = "f347c848dd57a5dfdf5b6e32eb257101a6a8a07f"
RESAMPLER_PATCH_ID = "resampler-typing-list-import"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_hashes(model_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(model_dir)): _sha256(path)
        for path in sorted(model_dir.rglob("*.py"))
    }


def _patch_resampler_typing_list(model_dir: Path) -> bool:
    """Apply the one-line remote-code fix without modifying an HF cache blob in place."""

    resampler = model_dir / "resampler.py"
    if not resampler.is_file():
        raise FileNotFoundError(f"MiniCPM resampler.py not found: {resampler}")
    text = resampler.read_text(encoding="utf-8")
    if re.search(r"(?m)^from typing import .*\bList\b", text):
        return False
    old_import = "from typing import Optional"
    if old_import not in text:
        raise RuntimeError(
            "Could not apply the MiniCPM resampler patch: expected typing import is absent"
        )
    patched = text.replace(old_import, "from typing import List, Optional", 1)
    temporary = resampler.with_name(f".{resampler.name}.patched.tmp")
    temporary.write_text(patched, encoding="utf-8", newline="")
    os.replace(temporary, resampler)
    return True


def write_snapshot_manifest(
    repo_id: str,
    requested_revision: str,
    resolved_revision: str,
    destination: Path,
    *,
    patch_resampler: bool = False,
) -> dict[str, object]:
    """Validate a local snapshot and write the manifest consumed by the runtime."""

    destination = destination.resolve()
    config_path = destination / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiniCPM config not found: {config_path}")
    patches: list[dict[str, object]] = []
    if patch_resampler:
        changed = _patch_resampler_typing_list(destination)
        patches.append(
            {
                "id": RESAMPLER_PATCH_ID,
                "changed_during_this_run": changed,
                "description": "Added the missing typing.List import required by Python 3.11.",
            }
        )
    code_hashes = _python_hashes(destination)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "repo_id": repo_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "created_at": datetime.now(UTC).isoformat(),
        "config_sha256": _sha256(config_path),
        "python_files_sha256": code_hashes,
        "patches": patches,
    }
    (destination / "model-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def download_snapshot(
    repo_id: str,
    revision: str,
    destination: Path,
    *,
    patch_resampler: bool = False,
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
    return write_snapshot_manifest(
        repo_id,
        revision,
        info.sha,
        destination,
        patch_resampler=patch_resampler,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=PINNED_MINICPM_REPO)
    parser.add_argument("--revision", default=PINNED_MINICPM_REVISION)
    parser.add_argument("--destination", type=Path, default=Path("models/minicpm-o-2_6-int4"))
    parser.add_argument(
        "--existing-snapshot",
        action="store_true",
        help="Do not download; finalize and hash an existing snapshot directory.",
    )
    parser.add_argument(
        "--patch-resampler-list-import",
        action="store_true",
        help="Apply the pinned one-line resampler.py compatibility patch before hashing.",
    )
    args = parser.parse_args()
    if args.existing_snapshot:
        if not args.destination.is_dir():
            raise FileNotFoundError(f"MiniCPM snapshot not found: {args.destination.resolve()}")
        manifest = write_snapshot_manifest(
            args.repo_id,
            args.revision,
            args.revision,
            args.destination,
            patch_resampler=args.patch_resampler_list_import,
        )
    else:
        manifest = download_snapshot(
            args.repo_id,
            args.revision,
            args.destination,
            patch_resampler=args.patch_resampler_list_import,
        )
    print(f"Saved immutable revision {manifest['resolved_revision']} to {args.destination}")


if __name__ == "__main__":
    main()
