"""Capture software, hardware, source, model, and Docker provenance in one report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "reproducibility" / "windows-cuda-baseline.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(  # noqa: S603 - argv is explicit and shell=False
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return {"available": False, "command": command[0]}
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _sanitize_remote(remote: str | None) -> str | None:
    if remote is None:
        return None
    return re.sub(r"^(https?://)[^/@]+@", r"\1", remote)


def _git_metadata(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"available": False, "path": str(path)}

    def git(*arguments: str) -> str | None:
        result = _run(["git", *arguments], cwd=path)
        return result.get("stdout") if result.get("returncode") == 0 else None

    diff = git("diff", "--binary", "HEAD") or ""
    modified_files = (git("diff", "--name-only", "HEAD") or "").splitlines()
    return {
        "available": True,
        "path": str(path.resolve()),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "remote_origin": _sanitize_remote(git("remote", "get-url", "origin")),
        "status": (git("status", "--short", "--branch") or "").splitlines(),
        "submodules": (git("submodule", "status", "--recursive") or "").splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "tracked_diff_present": bool(diff),
        "tracked_modified_files": modified_files,
    }


def _autogptq_metadata(path: Path) -> dict[str, Any]:
    metadata = _git_metadata(path)
    kernel_files = (
        path / "autogptq_extension" / "cuda_64" / "autogptq_cuda_kernel_64.cu",
        path / "autogptq_extension" / "cuda_256" / "autogptq_cuda_kernel_256.cu",
    )
    old_count = 0
    new_count = 0
    missing: list[str] = []
    for kernel in kernel_files:
        if not kernel.is_file():
            missing.append(str(kernel))
            continue
        text = kernel.read_text(encoding="utf-8")
        old_count += text.count("vec.type()")
        new_count += text.count("vec.scalar_type()")
    metadata["torch28_cuda_patch"] = {
        "old_expression_count": old_count,
        "new_expression_count": new_count,
        "missing_files": missing,
    }
    return metadata


def _installed_packages() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower().replace("_", "-")] = distribution.version
    return [
        {"name": name, "version": packages[name]}
        for name in sorted(packages)
    ]


def _locked_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        if "==" not in line:
            raise ValueError(f"Lock contains a non-exact requirement: {line}")
        name, version = line.split("==", 1)
        normalized = name.lower().replace("_", "-")
        if normalized in packages:
            raise ValueError(f"Lock contains duplicate package: {normalized}")
        packages[normalized] = version
    return packages


def _python_environment(output_dir: Path) -> dict[str, Any]:
    freeze = _run([sys.executable, "-m", "pip", "freeze", "--all"])
    freeze_text = freeze.get("stdout", "")
    (output_dir / "pip-freeze.txt").write_text(freeze_text + "\n", encoding="utf-8")

    portable_raw = _run(
        [sys.executable, "-m", "pip", "freeze", "--all", "--exclude-editable"]
    ).get("stdout", "")
    portable_lines = [
        line
        for line in portable_raw.splitlines()
        if not re.match(r"(?i)^(auto[-_]gptq|human[-_]attributes[-_]detector)(==|\s@)", line)
    ]
    portable = "\n".join(portable_lines)
    try:
        import torch

        pytorch_index = (
            "https://download.pytorch.org/whl/cu128"
            if torch.version.cuda == "12.8"
            else "https://download.pytorch.org/whl/cpu"
        )
    except ImportError:
        pytorch_index = None
    lock_header = [
        "# Candidate generated from a real environment; review and commit under "
        "requirements/locks/.",
        f"# Python {platform.python_version()} on {platform.platform()}",
        "# The editable project and patched AutoGPTQ checkout are intentionally recorded "
        "separately.",
    ]
    if pytorch_index:
        lock_header.append(f"--extra-index-url {pytorch_index}")
    lock_text = "\n".join(lock_header) + "\n" + portable + "\n"
    (output_dir / "requirements-lock-candidate.txt").write_text(
        lock_text, encoding="utf-8"
    )

    pip_check = _run([sys.executable, "-m", "pip", "check"])
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "pip": _version("pip"),
        "setuptools": _version("setuptools"),
        "wheel": _version("wheel"),
        "packages": _installed_packages(),
        "pip_freeze_returncode": freeze.get("returncode"),
        "pip_check": pip_check,
    }


def _pytorch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    data: dict[str, Any] = {
        "available": True,
        "torch": torch.__version__,
        "torchvision": _version("torchvision"),
        "torchaudio": _version("torchaudio"),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_vram_bytes": properties.total_memory,
                    "total_vram_mib": round(properties.total_memory / 1024**2, 3),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    data["devices"] = devices
    return data


def _nvidia_environment() -> dict[str, Any]:
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, str]] = []
    if query.get("returncode") == 0:
        for line in query.get("stdout", "").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                devices.append(
                    dict(
                        zip(
                            (
                                "index",
                                "name",
                                "driver_version",
                                "memory_total_mib",
                            ),
                            fields,
                            strict=True,
                        )
                    )
                )
    nvcc = _run(["nvcc", "--version"])
    nvcc_text = "\n".join((nvcc.get("stdout", ""), nvcc.get("stderr", "")))
    toolkit_match = re.search(r"release\s+([0-9.]+)", nvcc_text)
    compiler_match = re.search(r"\bV([0-9]+(?:\.[0-9]+)+)", nvcc_text)
    return {
        "nvidia_smi": query,
        "devices": devices,
        "nvcc": nvcc,
        "cuda_toolkit": toolkit_match.group(1) if toolkit_match else None,
        "nvcc_version": compiler_match.group(1) if compiler_match else None,
    }


def _minicpm_metadata(model_dir: Path | None) -> dict[str, Any] | None:
    if model_dir is None:
        return None
    model_dir = model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"MiniCPM snapshot not found: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiniCPM config not found: {config_path}")
    manifest_path = model_dir / "model-manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    revision = None
    if model_dir.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", model_dir.name):
        revision = model_dir.name
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config", {})
    python_hashes = {
        str(path.relative_to(model_dir)): _sha256(path)
        for path in sorted(model_dir.rglob("*.py"))
    }
    return {
        "path": str(model_dir),
        "inferred_revision": revision,
        "manifest": manifest,
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "config_sha256": _sha256(config_path),
        "model_type": config.get("model_type"),
        "quant_method": quantization.get("quant_method"),
        "python_files_sha256": python_hashes,
    }


def _transformer_metadata(checkpoint: Path | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Transformer checkpoint not found: {checkpoint}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to inspect the Transformer checkpoint") from exc
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("Transformer checkpoint must contain a mapping")
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict) or "input_proj.weight" not in state:
        raise RuntimeError("Transformer checkpoint has no compatible model_state_dict")
    config = dict(payload.get("config", {}))
    try:
        from silhouette_detector.attributes.labels import ATTRIBUTE_SIZES_RU
    except ImportError:
        ATTRIBUTE_SIZES_RU = {}
    effective = {
        "input_dim": int(state["input_proj.weight"].shape[1]),
        "hidden_dim": int(config.get("hidden_dim", 768)),
        "num_heads": int(config.get("num_heads", 12)),
        "num_layers": int(config.get("num_layers", 6)),
        "drop_path_rate": float(config.get("drop_path_rate", 0.1)),
        "attr_sizes": dict(config.get("attr_sizes", ATTRIBUTE_SIZES_RU)),
    }
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "embedded_config": config,
        "embedded_config_sha256": _json_sha256(config),
        "effective_architecture": effective,
        "state_tensor_count": sum(1 for value in state.values() if torch.is_tensor(value)),
    }


def _docker_bases() -> dict[str, str | None]:
    bases: dict[str, str | None] = {}
    for name in ("Dockerfile", "Dockerfile.gpu"):
        path = PROJECT_ROOT / name
        match = re.search(r"(?m)^FROM\s+([^\s]+)", path.read_text(encoding="utf-8"))
        bases[name] = match.group(1) if match else None
    return bases


def _record_check(
    checks: list[dict[str, Any]], name: str, expected: object, actual: object
) -> None:
    checks.append(
        {"name": name, "expected": expected, "actual": actual, "passed": expected == actual}
    )


def _baseline_checks(report: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    python = report["python"]
    pytorch = report["pytorch"]
    _record_check(checks, "platform", baseline["platform"]["os"], report["host"]["platform"])
    _record_check(checks, "python", baseline["platform"]["python"], python["version"])
    _record_check(
        checks,
        "python:implementation",
        baseline["platform"]["python_implementation"],
        python["implementation"],
    )
    _record_check(
        checks,
        "python:architecture",
        baseline["platform"]["architecture"],
        python["architecture"],
    )
    for name in ("torch", "torchvision", "torchaudio", "torch_cuda"):
        _record_check(checks, name, baseline["pytorch"][name], pytorch.get(name))
    _record_check(checks, "cudnn", baseline["pytorch"]["cudnn"], pytorch.get("cudnn"))
    _record_check(
        checks,
        "cuda_toolkit",
        baseline["pytorch"]["cuda_toolkit"],
        report["nvidia"].get("cuda_toolkit"),
    )
    _record_check(
        checks,
        "nvcc_version",
        baseline["pytorch"]["nvcc_version"],
        report["nvidia"].get("nvcc_version"),
    )
    first_device = (pytorch.get("devices") or [None])[0]
    _record_check(
        checks,
        "gpu:name",
        baseline["hardware"]["gpu"],
        first_device.get("name") if first_device else None,
    )
    _record_check(
        checks,
        "gpu:total_vram_mib",
        baseline["hardware"]["total_vram_mib"],
        first_device.get("total_vram_mib") if first_device else None,
    )
    _record_check(
        checks,
        "gpu:compute_capability",
        baseline["hardware"]["compute_capability"],
        first_device.get("compute_capability") if first_device else None,
    )
    nvidia_device = (report["nvidia"].get("devices") or [None])[0]
    _record_check(
        checks,
        "nvidia:driver",
        baseline["hardware"]["nvidia_driver"],
        nvidia_device.get("driver_version") if nvidia_device else None,
    )
    installed = {item["name"]: item["version"] for item in python["packages"]}
    for name, expected in baseline["validated_packages"].items():
        _record_check(checks, f"package:{name}", expected, installed.get(name))
    lock = baseline["locks"]["windows_cuda_py311"]
    lock_path = PROJECT_ROOT / lock["path"]
    locked_packages = _locked_packages(lock_path) if lock_path.is_file() else {}
    portable_installed = {
        name: version
        for name, version in installed.items()
        if name not in {"auto-gptq", "human-attributes-detector"}
    }
    _record_check(
        checks,
        "python:all_locked_packages",
        locked_packages,
        portable_installed,
    )

    minicpm = report.get("models", {}).get("minicpm")
    if minicpm:
        _record_check(
            checks,
            "minicpm:manifest_present",
            True,
            minicpm.get("manifest") is not None,
        )
        _record_check(
            checks,
            "minicpm:revision",
            baseline["minicpm"]["revision"],
            (minicpm.get("manifest") or {}).get("resolved_revision")
            or minicpm.get("inferred_revision"),
        )
        _record_check(
            checks,
            "minicpm:config_sha256",
            baseline["minicpm"]["config_sha256"],
            minicpm["config_sha256"],
        )
        _record_check(
            checks,
            "minicpm:python_files_sha256",
            baseline["minicpm"]["python_files_sha256_after_patch"],
            minicpm["python_files_sha256"],
        )

    transformer = report.get("models", {}).get("transformer")
    if transformer:
        _record_check(
            checks,
            "transformer:checkpoint_sha256",
            baseline["transformer"]["checkpoint_sha256"],
            transformer["sha256"],
        )
        _record_check(
            checks,
            "transformer:checkpoint_size_bytes",
            baseline["transformer"]["checkpoint_size_bytes"],
            transformer["size_bytes"],
        )
        architecture = transformer["effective_architecture"]
        for expected_name, actual_name in (
            ("input_dim", "input_dim"),
            ("hidden_dim", "hidden_dim"),
            ("num_layers", "num_layers"),
            ("num_attention_heads", "num_heads"),
        ):
            _record_check(
                checks,
                f"transformer:{expected_name}",
                baseline["transformer"][expected_name],
                architecture[actual_name],
            )
        _record_check(
            checks,
            "transformer:attribute_head_sizes_ru",
            baseline["transformer"]["attribute_head_sizes_ru"],
            architecture["attr_sizes"],
        )

    resnet = report.get("models", {}).get("resnet")
    if resnet:
        _record_check(
            checks,
            "resnet:checkpoint_sha256",
            baseline["resnet"]["checkpoint_sha256"],
            resnet["sha256"],
        )
        _record_check(
            checks,
            "resnet:checkpoint_size_bytes",
            baseline["resnet"]["checkpoint_size_bytes"],
            resnet["size_bytes"],
        )

    auto_source = report["source"]["autogptq"]
    _record_check(
        checks,
        "autogptq:commit",
        baseline["autogptq"]["commit"],
        auto_source.get("commit"),
    )
    _record_check(
        checks,
        "autogptq:remote",
        baseline["autogptq"]["repository"],
        auto_source.get("remote_origin"),
    )
    _record_check(
        checks,
        "autogptq:tracked_diff_sha256",
        baseline["autogptq"]["tracked_diff_sha256"],
        auto_source.get("tracked_diff_sha256"),
    )
    _record_check(
        checks,
        "autogptq:modified_files",
        sorted(baseline["autogptq"]["cuda_patch"]["modified_files"]),
        sorted(auto_source.get("tracked_modified_files", [])),
    )
    patch = auto_source["torch28_cuda_patch"]
    _record_check(checks, "autogptq:old_patch_expressions", 0, patch["old_expression_count"])
    _record_check(
        checks,
        "autogptq:new_patch_expressions",
        baseline["autogptq"]["cuda_patch"]["expected_replacements"],
        patch["new_expression_count"],
    )
    for dockerfile, baseline_name in (
        ("Dockerfile", "cpu_base"),
        ("Dockerfile.gpu", "gpu_base"),
    ):
        _record_check(
            checks,
            f"docker:{dockerfile}",
            baseline["docker"][baseline_name],
            report["docker"][dockerfile],
        )
    _record_check(
        checks,
        "lock:windows_cuda_py311_sha256",
        lock["sha256"],
        _sha256(lock_path) if lock_path.is_file() else None,
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--minicpm-model-dir", type=Path)
    parser.add_argument("--transformer-checkpoint", type=Path)
    parser.add_argument("--resnet-checkpoint", type=Path)
    parser.add_argument(
        "--autogptq-source", type=Path, default=PROJECT_ROOT / ".deps" / "AutoGPTQ-minicpmo"
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    source_files = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / "Dockerfile.gpu",
        PROJECT_ROOT / ".github" / "workflows" / "ci.yml",
        args.baseline.resolve(),
        *sorted((PROJECT_ROOT / "requirements").rglob("*.txt")),
        *sorted((PROJECT_ROOT / "scripts").glob("*.ps1")),
        *sorted((PROJECT_ROOT / "scripts").glob("*.py")),
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "baseline": str(args.baseline.resolve()),
        "host": {
            "os": os.name,
            "platform": platform.platform(),
        },
        "python": _python_environment(output_dir),
        "pytorch": _pytorch_environment(),
        "nvidia": _nvidia_environment(),
        "build_tools": {
            "git": _run(["git", "--version"]),
            "msvc": _run(["cl.exe"]),
        },
        "source": {
            "project": _git_metadata(PROJECT_ROOT),
            "autogptq": _autogptq_metadata(args.autogptq_source.expanduser().resolve()),
            "tracked_configuration_sha256": {
                (
                    str(path.relative_to(PROJECT_ROOT))
                    if path.is_relative_to(PROJECT_ROOT)
                    else str(path)
                ): _sha256(path)
                for path in source_files
                if path.is_file()
            },
        },
        "models": {
            "minicpm": _minicpm_metadata(args.minicpm_model_dir),
            "transformer": _transformer_metadata(args.transformer_checkpoint),
            "resnet": (
                {
                    "path": str(args.resnet_checkpoint.expanduser().resolve()),
                    "size_bytes": args.resnet_checkpoint.expanduser().resolve().stat().st_size,
                    "sha256": _sha256(args.resnet_checkpoint.expanduser().resolve()),
                }
                if args.resnet_checkpoint
                else None
            ),
        },
        "docker": _docker_bases(),
    }
    report["checks"] = _baseline_checks(report, baseline)
    failed = [check for check in report["checks"] if not check["passed"]]
    pip_check = report["python"]["pip_check"]
    report["status"] = "passed" if not failed and pip_check.get("returncode") == 0 else "failed"
    report["failed_checks"] = failed
    report_path = output_dir / "environment-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Environment report: {report_path}")
    print(f"Status: {report['status']}")
    print(f"Python packages captured: {len(report['python']['packages'])}")
    print(f"Failed baseline checks: {len(failed)}")
    for check in failed:
        print(f"  - {check['name']}: expected {check['expected']!r}, got {check['actual']!r}")
    if args.strict and report["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
