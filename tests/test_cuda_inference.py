"""Run real ResNet and MiniCPM+Transformer inference on CUDA and save artifacts.

This is an executable integration test, not a unittest test case. It deliberately
fails instead of falling back to CPU when CUDA or a required model is unavailable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _existing_path(raw: str | None, candidates: tuple[str, ...], label: str) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path = path.resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"{label} not found: {path}")
    for candidate in candidates:
        path = MODELS_DIR / candidate
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"{label} not found. Pass its path explicitly; searched: "
        + ", ".join(str(MODELS_DIR / name) for name in candidates)
    )


def _load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        with Image.open(path) as source:
            return source.convert("RGB")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required when --input is a video") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read the first frame from {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _timed_cuda_prediction(torch: Any, callback: Any) -> tuple[dict[str, str], float, float]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = callback()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return prediction, elapsed_ms, peak_mb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real ResNet and MiniCPM+Transformer inference strictly on CUDA."
    )
    parser.add_argument("--input", required=True, help="Image or video; first video frame is used")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--resnet", help="ResNet checkpoint path")
    parser.add_argument("--transformer", help="VisionAttrTransformer checkpoint path")
    parser.add_argument("--minicpm-model-dir", help="Local MiniCPM snapshot directory")
    parser.add_argument(
        "--allow-unverified-model-code",
        action="store_true",
        help="Allow a local MiniCPM directory without the generated verification manifest",
    )
    parser.add_argument("--output-dir", help="Artifact directory (default: var/cuda-tests/<time>)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "var" / "cuda-tests" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=False)
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
    report: dict[str, Any] = {"status": "failed", "models": {}}

    try:
        import torch

        from silhouette_detector.attributes.minicpm import MiniCPMTransformerBackend
        from silhouette_detector.attributes.resnet import ResNetBackend

        if not args.device.startswith("cuda"):
            raise ValueError("--device must be cuda or cuda:N; CPU fallback is forbidden")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch CUDA is unavailable. Install a CUDA build and verify nvidia-smi."
            )
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
        report["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": gpu_name,
        }
        logging.info("CUDA ready: %s (%s)", device, gpu_name)

        input_path = _existing_path(args.input, (), "Input image/video")
        image = _load_image(input_path)
        report["input"] = {"path": str(input_path), "size": list(image.size)}

        resnet_path = _existing_path(
            args.resnet,
            ("resnet_ens_11.19_e60_s0.782.pt", "resnet_attributes.pt"),
            "ResNet checkpoint",
        )
        logging.info("Loading ResNet checkpoint: %s", resnet_path)
        resnet = ResNetBackend(resnet_path, device=str(device))
        resnet_prediction, resnet_ms, resnet_peak = _timed_cuda_prediction(
            torch, lambda: resnet.predict(image)
        )
        report["models"]["resnet"] = {
            "checkpoint": str(resnet_path),
            "device": str(next(resnet.model.parameters()).device),
            "elapsed_ms": round(resnet_ms, 3),
            "peak_memory_mb": round(resnet_peak, 3),
            "prediction": resnet_prediction,
        }
        logging.info("ResNet result: %s", json.dumps(resnet_prediction, ensure_ascii=False))

        transformer_path = _existing_path(
            args.transformer,
            (
                "MiniCPM-2.6int4 weights.pt",
                "MiniCPM-o 2.6int4 weights.pt",
                "best_checkpoint.pt",
                "vision_attr_transformer.pt",
            ),
            "Transformer checkpoint",
        )
        minicpm_dir = _existing_path(
            args.minicpm_model_dir,
            ("minicpm-o-2_6-int4", "MiniCPM-o-2_6-int4", "minicpm-o-2_6"),
            "MiniCPM model directory",
        )
        if not minicpm_dir.is_dir():
            raise NotADirectoryError(f"MiniCPM model path is not a directory: {minicpm_dir}")
        logging.info("Loading MiniCPM snapshot: %s", minicpm_dir)
        logging.info("Loading Transformer checkpoint: %s", transformer_path)
        transformer = MiniCPMTransformerBackend(
            minicpm_dir,
            transformer_path,
            device=str(device),
            allow_unverified_model_code=args.allow_unverified_model_code,
        )
        transformer_prediction, transformer_ms, transformer_peak = _timed_cuda_prediction(
            torch, lambda: transformer.predict(image)
        )
        report["models"]["transformer"] = {
            "checkpoint": str(transformer_path),
            "embedding_model": str(minicpm_dir),
            "extractor_device": transformer.extractor.device,
            "classifier_device": str(next(transformer.classifier.parameters()).device),
            "elapsed_ms": round(transformer_ms, 3),
            "peak_memory_mb": round(transformer_peak, 3),
            "prediction": transformer_prediction,
        }
        logging.info(
            "Transformer result: %s",
            json.dumps(transformer_prediction, ensure_ascii=False),
        )
        report["status"] = "passed"
        logging.info("CUDA INFERENCE TEST PASSED")
        return_code = 0
    except Exception as exc:
        report["error"] = f"{exc.__class__.__name__}: {exc}"
        logging.exception("CUDA INFERENCE TEST FAILED")
        return_code = 1
    finally:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logging.info("Report: %s", report_path)
        logging.info("Log: %s", log_path)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
