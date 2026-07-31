"""Single supported tracker adapter (SFSORT)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import Detection


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    box: tuple[int, int, int, int]
    track_id: int
    class_id: int
    score: float


def default_sfsort_settings(width: int, height: int, fps: float) -> dict[str, float | bool | int]:
    return {
        "high_th": 0.3,
        "match_th_first": 0.8,
        "low_th": 0.1,
        "new_track_th": 0.3,
        "dynamic_tuning": True,
        "cth": 0.5,
        "high_th_m": 0.1,
        "match_th_first_m": 0.05,
        "match_th_second": 0.4,
        "new_track_th_m": 0.2,
        "marginal_timeout": max(1, round(0.7 * fps)),
        "central_timeout": max(1, round(fps)),
        "horizontal_margin": width // 10,
        "vertical_margin": height // 10,
        "frame_width": width,
        "frame_height": height,
    }


class SFSORTTracker:
    def __init__(self, width: int, height: int, fps: float) -> None:
        try:
            from SFSORT import SFSORT
        except ImportError as exc:
            raise RuntimeError(
                "SFSORT or one of its dependencies is missing. "
                "Install the project dependencies with: python -m pip install -e ."
            ) from exc
        self._tracker = SFSORT(default_sfsort_settings(width, height, fps))

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        if detections:
            boxes = np.asarray([detection.box for detection in detections], dtype=np.float32)
            scores = np.asarray([detection.score for detection in detections], dtype=np.float32)
            classes = np.asarray([detection.class_id for detection in detections], dtype=np.int64)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
            classes = np.empty(0, dtype=np.int64)
        tracks = self._tracker.update(boxes, scores, classes)
        return [
            TrackedDetection(
                tuple(int(value) for value in track[0]),
                int(track[1]),
                int(track[2]),
                float(track[3]),
            )
            for track in tracks
        ]
