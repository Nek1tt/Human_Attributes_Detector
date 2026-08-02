"""ONNX YOLO detector with explicit CPU/GPU provider selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from .device import select_onnx_providers, validate_onnx_session_providers
from .nms import class_aware_nms


@dataclass(frozen=True, slots=True)
class Detection:
    box: tuple[int, int, int, int]
    class_id: int
    score: float


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        confidence: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO ONNX model not found: {model_path}")

        # On Windows the CUDA/cuDNN DLLs bundled with the PyTorch wheel are not
        # necessarily visible to ONNX Runtime in a fresh process.  Loading
        # PyTorch before the ORT session is the officially supported bridge.
        if device.lower() != "cpu":
            try:
                __import__("torch")
            except ImportError:
                # ORT may still use a system CUDA/cuDNN installation.
                pass
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install the cpu or gpu extra to use ONNX detection") from exc
        providers = select_onnx_providers(device, ort.get_available_providers())
        if "CUDAExecutionProvider" in providers:
            preload_dlls = getattr(ort, "preload_dlls", None)
            if callable(preload_dlls):
                preload_dlls()
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_names = [output.name for output in self.session.get_outputs()]
        shape = model_input.shape
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else 576
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else 1024
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.providers = validate_onnx_session_providers(
            device, self.session.get_providers()
        )
        self._lock = Lock()

    @staticmethod
    def _letterbox(
        image: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        import cv2

        source_height, source_width = image.shape[:2]
        ratio = min(width / source_width, height / source_height)
        resized_width = round(source_width * ratio)
        resized_height = round(source_height * ratio)
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        delta_width = width - resized_width
        delta_height = height - resized_height
        left = delta_width // 2
        right = delta_width - left
        top = delta_height // 2
        bottom = delta_height - top
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return padded, ratio, (float(left), float(top))

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        padded, ratio, offset = self._letterbox(image, self.input_width, self.input_height)
        rgb = padded[:, :, ::-1]
        tensor = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor), ratio, offset

    def _decode(
        self,
        output: np.ndarray,
        ratio: float,
        offset: tuple[float, float],
        frame_shape: tuple[int, int],
    ) -> list[Detection]:
        predictions = np.squeeze(output)
        if predictions.ndim != 2:
            raise RuntimeError(f"Unsupported YOLO output shape: {output.shape}")
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] >= 5:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise RuntimeError(f"Unsupported YOLO output shape: {output.shape}")

        class_scores = predictions[:, 4:]
        scores = class_scores.max(axis=1)
        candidates = scores >= self.confidence
        if not candidates.any():
            return []
        boxes_xywh = predictions[candidates, :4]
        scores = scores[candidates]
        class_ids = class_scores[candidates].argmax(axis=1).astype(np.int64)
        boxes = np.empty_like(boxes_xywh, dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - offset[0]) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - offset[1]) / ratio
        frame_height, frame_width = frame_shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_height - 1)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
        if not len(boxes):
            return []
        keep = class_aware_nms(boxes, scores, class_ids, self.iou_threshold)
        return [
            Detection(
                tuple(int(value) for value in boxes[index].round()),
                int(class_ids[index]),
                float(scores[index]),
            )
            for index in keep
        ]

    def detect(self, image: np.ndarray) -> list[Detection]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape [H, W, 3]")
        tensor, ratio, offset = self._preprocess(image)
        with self._lock:
            outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._decode(outputs[0], ratio, offset, image.shape[:2])
