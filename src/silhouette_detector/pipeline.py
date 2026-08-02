"""Video pipeline: YOLO -> SFSORT -> one attribute backend -> annotated MP4."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .attribute_service import AttributeService
from .attributes.labels import UNKNOWN_ATTRIBUTES
from .detection import YoloOnnxDetector
from .tracking import SFSORTTracker, TrackedDetection


@dataclass(slots=True)
class TrackMemory:
    frames_seen: int = 0
    submitted: bool = False
    attributes: dict[str, str] = field(default_factory=lambda: dict(UNKNOWN_ATTRIBUTES))


@dataclass(frozen=True, slots=True)
class FrameResult:
    track_id: int
    box: tuple[int, int, int, int]
    confidence: float
    attributes: dict[str, str]


class FramePipeline:
    def __init__(
        self,
        task_id: str,
        detector: YoloOnnxDetector,
        attribute_service: AttributeService,
        width: int,
        height: int,
        fps: float,
        min_track_frames: int,
        synchronous_attributes: bool = False,
    ) -> None:
        self.task_id = task_id
        self.detector = detector
        self.attributes = attribute_service
        self.tracker = SFSORTTracker(width, height, fps)
        self.min_track_frames = min_track_frames
        self.synchronous_attributes = synchronous_attributes
        self.memory: dict[int, TrackMemory] = {}

    @staticmethod
    def _crop(frame: np.ndarray, track: TrackedDetection) -> Image.Image | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = track.box
        x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width))))
        y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height))))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        rgb = frame[y1:y2, x1:x2, ::-1]
        return Image.fromarray(np.ascontiguousarray(rgb))

    def process(self, frame: np.ndarray) -> list[FrameResult]:
        people = [detection for detection in self.detector.detect(frame) if detection.class_id == 0]
        tracks = self.tracker.update(people)
        results: list[FrameResult] = []
        for track in tracks:
            memory = self.memory.setdefault(track.track_id, TrackMemory())
            memory.frames_seen += 1
            completed = self.attributes.get(self.task_id, track.track_id)
            if completed is not None:
                memory.attributes = completed
            if not memory.submitted and memory.frames_seen >= self.min_track_frames:
                crop = self._crop(frame, track)
                if crop is not None:
                    memory.submitted = self.attributes.submit(self.task_id, track.track_id, crop)
                    if memory.submitted and self.synchronous_attributes:
                        completed = self.attributes.wait(self.task_id, track.track_id)
                        if completed is not None:
                            memory.attributes = completed
            results.append(
                FrameResult(track.track_id, track.box, track.score, dict(memory.attributes))
            )
        return results


@lru_cache(maxsize=1)
def _annotation_font() -> ImageFont.FreeTypeFont:
    configured = os.getenv("HAD_FONT_PATH")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return ImageFont.truetype(str(candidate), size=16)
    raise RuntimeError(
        "No Cyrillic TrueType font found. Set HAD_FONT_PATH to a .ttf font with Cyrillic glyphs"
    )


def draw_results(frame: np.ndarray, results: list[FrameResult]) -> np.ndarray:
    import cv2

    output = frame.copy()
    text_items: list[tuple[tuple[int, int], str, tuple[int, int, int]]] = []
    for result in results:
        x1, y1, x2, y2 = result.box
        color = (37, 190, 90)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        lines = [f"ID {result.track_id} | {result.confidence:.2f}"]
        known = [
            f"{key}: {value}"
            for key, value in result.attributes.items()
            if value != "не определен"
        ]
        lines.extend(known[:5])
        top = max(18, y1 - 8 - 18 * len(lines))
        for index, line in enumerate(lines):
            position = (max(0, x1), top + index * 18)
            text_items.append((position, line, color))

    if not text_items:
        return output

    # OpenCV's built-in Hershey fonts do not contain Cyrillic glyphs.
    image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _annotation_font()
    for position, line, bgr_color in text_items:
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        draw.text(position, line, font=font, fill=rgb_color, stroke_width=2, stroke_fill="white")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class VideoProcessor:
    def __init__(
        self,
        detector: YoloOnnxDetector,
        attribute_service: AttributeService,
        *,
        target_fps: float,
        min_track_frames: int,
        max_video_seconds: int,
        max_frame_pixels: int,
        synchronous_attributes: bool = False,
    ) -> None:
        self.detector = detector
        self.attribute_service = attribute_service
        self.target_fps = target_fps
        self.min_track_frames = min_track_frames
        self.max_video_seconds = max_video_seconds
        self.max_frame_pixels = max_frame_pixels
        self.synchronous_attributes = synchronous_attributes

    def process(
        self,
        input_path: Path,
        output_path: Path,
        metadata_path: Path,
        task_id: str,
        progress: Callable[[float], None] | None = None,
    ) -> None:
        import cv2

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("The uploaded file is not a readable video")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_fps = source_fps if source_fps > 0 else self.target_fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / source_fps if frame_count > 0 else 0
        if width <= 0 or height <= 0 or width * height > self.max_frame_pixels:
            capture.release()
            raise ValueError("Video resolution is invalid or exceeds HAD_MAX_FRAME_PIXELS")
        if duration > self.max_video_seconds:
            capture.release()
            raise ValueError("Video duration exceeds HAD_MAX_VIDEO_SECONDS")

        output_fps = min(self.target_fps, source_fps)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not initialize the MP4 writer")
        pipeline = FramePipeline(
            task_id,
            self.detector,
            self.attribute_service,
            width,
            height,
            output_fps,
            self.min_track_frames,
            self.synchronous_attributes,
        )
        sample_interval = max(1, round(source_fps / output_fps))
        source_index = 0
        processed = 0
        try:
            with metadata_path.open("w", encoding="utf-8") as metadata:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if source_index / source_fps > self.max_video_seconds:
                        raise ValueError("Video duration exceeds HAD_MAX_VIDEO_SECONDS")
                    if source_index % sample_interval:
                        source_index += 1
                        continue
                    results = pipeline.process(frame)
                    writer.write(draw_results(frame, results))
                    metadata.write(
                        json.dumps(
                            {
                                "frame": source_index,
                                "time_seconds": source_index / source_fps,
                                "people": [asdict(result) for result in results],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    processed += 1
                    source_index += 1
                    if progress and frame_count > 0:
                        progress(min(0.99, source_index / frame_count))
            if processed == 0:
                raise ValueError("The video contains no readable frames")
            if progress:
                progress(1.0)
        finally:
            capture.release()
            writer.release()
            self.attribute_service.drop_task(task_id)
