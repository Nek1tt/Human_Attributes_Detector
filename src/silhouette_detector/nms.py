"""Class-aware non-maximum suppression."""

from __future__ import annotations

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=np.int64)
    boxes = boxes.astype(np.float32, copy=False)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Suppress overlapping boxes only when they belong to the same class."""
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        indices = np.flatnonzero(class_ids == class_id)
        selected = nms(boxes[indices], scores[indices], iou_threshold)
        kept.extend(indices[selected].tolist())
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return np.asarray(kept, dtype=np.int64)
