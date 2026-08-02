"""Profile the local MiniCPM + trained Transformer pipeline in one CUDA process.

The script is intentionally separate from the regular test suite because it loads
the real multi-gigabyte model. It never downloads model files and does not clear
the CUDA allocator between repeated inferences, so memory growth remains visible.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEBIBYTE = 1024**2
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


class BaselineCheckError(RuntimeError):
    """Raised when the pipeline runs but violates a baseline invariant."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_type(value: object) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _resolve_existing(path: Path, *, directory: bool, label: str) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    resolved = resolved.resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} not found: {resolved}")
    return resolved


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        return source.convert("RGB")


def _snapshot_metadata(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MiniCPM config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise BaselineCheckError("MiniCPM config.json must contain a JSON object")

    manifest_path = model_dir / "model-manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded_manifest, dict):
            manifest = loaded_manifest

    code_hashes = {
        str(path.relative_to(model_dir)): _sha256(path)
        for path in sorted(model_dir.rglob("*.py"))
    }
    quantization = config.get("quantization_config")
    quant_method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    inferred_revision = model_dir.name if model_dir.parent.name == "snapshots" else None
    return {
        "path": str(model_dir),
        "config_sha256": _sha256(config_path),
        "model_type": config.get("model_type"),
        "quant_method": quant_method,
        "revision": (manifest or {}).get("resolved_revision") or inferred_revision,
        "manifest_present": manifest_path.is_file(),
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "local_snapshot": True,
        "python_files_sha256": code_hashes,
    }


def _cuda_memory(torch: Any, device: Any) -> dict[str, float]:
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / MEBIBYTE, 3),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / MEBIBYTE, 3),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / MEBIBYTE, 3),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / MEBIBYTE, 3),
    }


def _backend_identities(backend: Any) -> dict[str, int | None]:
    return {
        "processor": id(backend.extractor.processor),
        "embedding_model": id(backend.extractor.model),
        "quantized_model": (
            id(backend.extractor._quantized_model)
            if hasattr(backend.extractor, "_quantized_model")
            else None
        ),
        "classifier": id(backend.classifier),
    }


def _validate_prediction(
    prediction: dict[str, str],
    attribute_keys: tuple[str, ...],
    api_to_labels: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    actual_keys = tuple(prediction)
    missing = [key for key in attribute_keys if key not in prediction]
    extra = [key for key in actual_keys if key not in attribute_keys]
    invalid_values = {
        key: value
        for key, value in prediction.items()
        if key not in api_to_labels or value not in api_to_labels[key]
    }
    passed = actual_keys == attribute_keys and not invalid_values
    return {
        "passed": passed,
        "expected_api_keys": list(attribute_keys),
        "actual_api_keys": list(actual_keys),
        "missing_api_keys": missing,
        "extra_api_keys": extra,
        "invalid_russian_values": invalid_values,
    }


def _iteration_summary(iterations: list[dict[str, Any]]) -> dict[str, int | float]:
    times = [float(item["elapsed_ms"]) for item in iterations]
    return {
        "count": len(times),
        "mean_ms": round(statistics.fmean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="One test image")
    parser.add_argument(
        "--minicpm-model-dir",
        required=True,
        type=Path,
        help="Local MiniCPM snapshot directory containing config.json",
    )
    parser.add_argument(
        "--transformer",
        required=True,
        type=Path,
        help="Trained VisionAttrTransformer checkpoint",
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument(
        "--repeats",
        default=5,
        type=_positive_int,
        help="Repeated full inferences after the first pass (default: 5)",
    )
    parser.add_argument(
        "--memory-growth-limit-mb",
        default=64.0,
        type=_non_negative_float,
        help="Maximum allowed live VRAM growth across repeats (default: 64 MiB)",
    )
    parser.add_argument(
        "--allow-unverified-model-code",
        action="store_true",
        help="Allow a patched snapshot without a matching model-manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory (default: var/autogptq-baseline-<timestamp>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "var" / f"autogptq-baseline-{timestamp}"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    report_path = output_dir / "report.json"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "started_at": _utc_now(),
        "stage": "initialization",
        "offline_mode": True,
        "cuda": {"oom_detected": False},
        "checks": {},
    }
    started_total = time.perf_counter()
    torch: Any | None = None

    # Force the diagnostic to use only the supplied snapshot, even if the caller's
    # shell has different Hugging Face settings.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    try:
        report["stage"] = "input_validation"
        input_path = _resolve_existing(args.input, directory=False, label="Input image")
        model_dir = _resolve_existing(
            args.minicpm_model_dir, directory=True, label="MiniCPM model"
        )
        checkpoint = _resolve_existing(
            args.transformer, directory=False, label="Transformer checkpoint"
        )
        image = _load_rgb_image(input_path)
        snapshot = _snapshot_metadata(model_dir)
        report["input"] = {
            "path": str(input_path),
            "size": list(image.size),
            "mode": image.mode,
        }
        report["models"] = {
            "minicpm": snapshot,
            "transformer": {
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": _sha256(checkpoint),
            },
        }
        if not snapshot["manifest_present"] and not args.allow_unverified_model_code:
            raise BaselineCheckError(
                "MiniCPM model-manifest.json is absent; pass "
                "--allow-unverified-model-code only for the explicitly patched snapshot"
            )

        report["stage"] = "cuda_preflight"
        import torch as imported_torch

        torch = imported_torch
        from silhouette_detector.attributes.labels import (
            ATTRIBUTE_KEYS,
            ATTRIBUTE_LABELS_RU,
            RU_TO_API,
        )
        from silhouette_detector.attributes.minicpm import MiniCPMTransformerBackend

        requested_device = args.device.lower()
        if not requested_device.startswith("cuda"):
            raise BaselineCheckError("This AutoGPTQ baseline requires --device cuda or cuda:N")
        if not torch.cuda.is_available():
            raise BaselineCheckError("PyTorch CUDA is unavailable; CPU fallback is forbidden")
        device = torch.device("cuda:0" if requested_device == "cuda" else requested_device)
        torch.cuda.set_device(device)
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        report["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": _package_version("transformers"),
            "auto_gptq": _package_version("auto-gptq"),
            "torchaudio": _package_version("torchaudio"),
            "torchvision": _package_version("torchvision"),
            "accelerate": _package_version("accelerate"),
            "librosa": _package_version("librosa"),
            "sentencepiece": _package_version("sentencepiece"),
            "soundfile": _package_version("soundfile"),
            "timm": _package_version("timm"),
            "vector_quantize_pytorch": _package_version("vector-quantize-pytorch"),
            "vocos": _package_version("vocos"),
        }
        report["cuda"].update(
            {
                "device": str(device),
                "device_index": device_index,
                "gpu": torch.cuda.get_device_name(device_index),
                "total_vram_mb": round(properties.total_memory / MEBIBYTE, 3),
            }
        )
        logging.info("CUDA ready: %s (%s)", device, report["cuda"]["gpu"])

        report["stage"] = "model_loading"
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        load_started = time.perf_counter()
        backend = MiniCPMTransformerBackend(
            model_dir,
            checkpoint,
            device=str(device),
            allow_unverified_model_code=args.allow_unverified_model_code,
        )
        torch.cuda.synchronize(device)
        load_elapsed_ms = (time.perf_counter() - load_started) * 1000
        load_memory = _cuda_memory(torch, device)
        initial_identities = _backend_identities(backend)

        input_dim = int(backend.classifier.input_proj.in_features)
        hidden_dim = int(backend.classifier.input_proj.out_features)
        classifier_device = str(next(backend.classifier.parameters()).device)
        head_sizes = {
            name: int(head[-1].out_features) for name, head in backend.classifier.heads.items()
        }
        expected_head_sizes = {
            name: len(ATTRIBUTE_LABELS_RU[name]) for name in ATTRIBUTE_LABELS_RU
        }
        checkpoint_compatible = head_sizes == expected_head_sizes
        device_compatible = (
            backend.extractor.device == str(device) and classifier_device == str(device)
        )
        components_loaded = (
            backend.extractor.processor is not None and backend.extractor.model is not None
        )
        report["checks"]["model_loading"] = {
            "passed": checkpoint_compatible and device_compatible and components_loaded,
            "elapsed_ms": round(load_elapsed_ms, 3),
            "memory": load_memory,
            "processor_loaded": backend.extractor.processor is not None,
            "processor_type": _object_type(backend.extractor.processor),
            "embedding_model_loaded": backend.extractor.model is not None,
            "embedding_model_type": _object_type(backend.extractor.model),
            "extractor_device": backend.extractor.device,
            "classifier_device": classifier_device,
            "transformer_strict_state_dict_load": True,
            "transformer_architecture": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "num_layers": len(backend.classifier.layers),
                "num_attention_heads": backend.classifier.layers[0].self_attn.num_heads,
                "attribute_head_sizes_ru": head_sizes,
            },
            "attribute_heads_match_labels": checkpoint_compatible,
            "devices_match_request": device_compatible,
        }
        if not checkpoint_compatible:
            raise BaselineCheckError(
                f"Transformer heads {head_sizes} do not match label sizes {expected_head_sizes}"
            )
        if not device_compatible:
            raise BaselineCheckError(
                f"Model device mismatch: extractor={backend.extractor.device}, "
                f"classifier={classifier_device}, requested={device}"
            )
        logging.info(
            "Models loaded once in %.1f ms; peak allocated VRAM %.1f MiB",
            load_elapsed_ms,
            load_memory["peak_allocated_mb"],
        )

        extraction_calls: list[dict[str, Any]] = []
        original_extract = backend.extractor.extract

        def tracked_extract(current_image: Image.Image) -> Any:
            torch.cuda.synchronize(device)
            extract_started = time.perf_counter()
            embeddings = original_extract(current_image)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - extract_started) * 1000
            embedding_check = {
                "call": len(extraction_calls) + 1,
                "elapsed_ms": round(elapsed_ms, 3),
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "device": str(embeddings.device),
                "finite": bool(torch.isfinite(embeddings).all().item()),
            }
            extraction_calls.append(embedding_check)
            dimension_matches = (
                embeddings.ndim == 3 and int(embeddings.shape[-1]) == input_dim
            )
            if len(extraction_calls) == 1:
                report["checks"]["embedding_extraction"] = {
                    "passed": dimension_matches and embedding_check["finite"],
                    **embedding_check,
                    "expected_layout": "[images_or_slices, patches, embedding_dim]",
                }
                report["checks"]["embedding_dimension"] = {
                    "passed": dimension_matches,
                    "embedding_dimension": int(embeddings.shape[-1]),
                    "transformer_input_proj_in_features": input_dim,
                }
            return embeddings

        backend.extractor.extract = tracked_extract
        api_to_labels = {
            RU_TO_API[russian_key]: tuple(labels)
            for russian_key, labels in ATTRIBUTE_LABELS_RU.items()
        }

        def run_inference(iteration: str) -> dict[str, Any]:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            memory_before = _cuda_memory(torch, device)
            inference_started = time.perf_counter()
            prediction = backend.predict(image)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - inference_started) * 1000
            memory_after = _cuda_memory(torch, device)
            gc.collect()
            torch.cuda.synchronize(device)
            memory_after_gc = _cuda_memory(torch, device)
            label_check = _validate_prediction(prediction, ATTRIBUTE_KEYS, api_to_labels)
            if not label_check["passed"]:
                raise BaselineCheckError(
                    f"Invalid API keys or Russian labels on iteration {iteration}: {label_check}"
                )
            return {
                "iteration": iteration,
                "elapsed_ms": round(elapsed_ms, 3),
                "minicpm_elapsed_ms": extraction_calls[-1]["elapsed_ms"],
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_after_gc": memory_after_gc,
                "prediction": prediction,
                "label_check": label_check,
            }

        report["stage"] = "first_full_inference"
        first = run_inference("first")
        first_embedding = extraction_calls[0]
        embedding_shape = first_embedding["shape"]
        dimension_compatible = len(embedding_shape) == 3 and embedding_shape[-1] == input_dim
        report["checks"]["embedding_extraction"] = {
            "passed": dimension_compatible and first_embedding["finite"],
            **first_embedding,
            "expected_layout": "[images_or_slices, patches, embedding_dim]",
        }
        report["checks"]["embedding_dimension"] = {
            "passed": dimension_compatible,
            "embedding_dimension": embedding_shape[-1] if embedding_shape else None,
            "transformer_input_proj_in_features": input_dim,
        }
        report["checks"]["first_full_inference"] = {"passed": True, **first}
        if not dimension_compatible:
            raise BaselineCheckError(
                f"MiniCPM embedding shape {embedding_shape} is incompatible with "
                f"Transformer input_proj.in_features={input_dim}"
            )
        if not first_embedding["finite"]:
            raise BaselineCheckError("MiniCPM embeddings contain NaN or infinity")
        logging.info(
            "First inference: %.1f ms; embeddings=%s; peak VRAM=%.1f MiB",
            first["elapsed_ms"],
            embedding_shape,
            first["memory_after"]["peak_allocated_mb"],
        )

        report["stage"] = "repeated_inference"
        repeated: list[dict[str, Any]] = []
        for index in range(1, args.repeats + 1):
            result = run_inference(f"repeat_{index}")
            repeated.append(result)
            logging.info(
                "Repeat %d/%d: %.1f ms; allocated after GC %.1f MiB; peak %.1f MiB",
                index,
                args.repeats,
                result["elapsed_ms"],
                result["memory_after_gc"]["allocated_mb"],
                result["memory_after"]["peak_allocated_mb"],
            )

        repeated_allocated = [item["memory_after_gc"]["allocated_mb"] for item in repeated]
        allocated_growth = repeated_allocated[-1] - repeated_allocated[0]
        allocated_spread = max(repeated_allocated) - min(repeated_allocated)
        memory_stable = (
            allocated_growth <= args.memory_growth_limit_mb
            and allocated_spread <= args.memory_growth_limit_mb
        )
        predictions_stable = all(
            item["prediction"] == first["prediction"] for item in repeated
        )
        final_identities = _backend_identities(backend)
        objects_stable = final_identities == initial_identities
        shapes_stable = all(item["shape"] == embedding_shape for item in extraction_calls)
        all_peaks = [
            load_memory["peak_allocated_mb"],
            first["memory_after"]["peak_allocated_mb"],
            *(item["memory_after"]["peak_allocated_mb"] for item in repeated),
        ]
        all_reserved_peaks = [
            load_memory["peak_reserved_mb"],
            first["memory_after"]["peak_reserved_mb"],
            *(item["memory_after"]["peak_reserved_mb"] for item in repeated),
        ]

        report["checks"]["repeated_inference"] = {
            "passed": memory_stable and predictions_stable and objects_stable and shapes_stable,
            "iterations": repeated,
            "timing_summary": _iteration_summary(repeated),
            "predictions_stable": predictions_stable,
            "embedding_shapes_stable": shapes_stable,
        }
        report["checks"]["single_model_load"] = {
            "passed": objects_stable,
            "backend_instances_created": 1,
            "expected_extract_calls": args.repeats + 1,
            "actual_extract_calls": len(extraction_calls),
            "object_identities_stable": objects_stable,
            "initial_object_ids": initial_identities,
            "final_object_ids": final_identities,
        }
        report["checks"]["cuda_memory"] = {
            "passed": memory_stable,
            "oom_detected": False,
            "memory_growth_limit_mb": args.memory_growth_limit_mb,
            "repeat_allocated_after_gc_mb": repeated_allocated,
            "allocated_growth_first_to_last_repeat_mb": round(allocated_growth, 3),
            "allocated_spread_across_repeats_mb": round(allocated_spread, 3),
            "maximum_peak_allocated_vram_mb": round(max(all_peaks), 3),
            "maximum_peak_reserved_vram_mb": round(max(all_reserved_peaks), 3),
            "note": "CUDA cache/reserved memory is reported but only live allocated memory "
            "is used for the leak check.",
        }
        if not predictions_stable:
            raise BaselineCheckError("Predictions changed between repeated eval-mode inferences")
        if not objects_stable:
            raise BaselineCheckError("Processor or model object was replaced between inferences")
        if not shapes_stable:
            raise BaselineCheckError("MiniCPM embedding shape changed between inferences")
        if not memory_stable:
            raise BaselineCheckError(
                f"Live CUDA memory grew by {allocated_growth:.1f} MiB with a "
                f"{allocated_spread:.1f} MiB spread; limit is {args.memory_growth_limit_mb:.1f} MiB"
            )

        report["status"] = "passed"
        report["stage"] = "completed"
        logging.info("MINICPM + TRANSFORMER BASELINE PASSED")
        return_code = 0
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        report["error"] = message
        oom_detected = "out of memory" in str(exc).lower()
        if torch is not None:
            out_of_memory_type = getattr(torch, "OutOfMemoryError", ())
            oom_detected = oom_detected or isinstance(exc, out_of_memory_type)
        report["cuda"]["oom_detected"] = oom_detected
        logging.exception("MINICPM + TRANSFORMER BASELINE FAILED at %s", report["stage"])
        return_code = 1
    finally:
        report["finished_at"] = _utc_now()
        report["total_elapsed_ms"] = round((time.perf_counter() - started_total) * 1000, 3)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logging.info("Report: %s", report_path)
        logging.info("Log: %s", log_path)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
