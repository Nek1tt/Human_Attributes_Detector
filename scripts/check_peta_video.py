"""Run YOLO, tracking and the trained nine-attribute model on a local video."""

import json
from pathlib import Path

import click
import cv2

from silhouette_detector.attribute_service import AttributeService
from silhouette_detector.attributes.factory import create_attribute_backend
from silhouette_detector.detection import YoloOnnxDetector
from silhouette_detector.peta_data import PETA_LABELS
from silhouette_detector.pipeline import VideoProcessor
from silhouette_detector.settings import Settings


@click.command()
@click.option("--video", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--yolo", required=True, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--checkpoint",
    default="models/original_resnet_peta.pt",
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--device", default="cpu")
@click.option("--output", default="var/peta-video-check", type=click.Path(path_type=Path))
def main(video, yolo, checkpoint, device, output):
    settings = Settings(attribute_backend="peta", attribute_checkpoint=checkpoint, device=device)
    settings.validate()
    backend = create_attribute_backend(settings)
    if tuple(backend.model.labels) != tuple(PETA_LABELS):
        raise ValueError("Use a PETA checkpoint with nine attributes")
    service = AttributeService(backend)
    detector = YoloOnnxDetector(yolo, device="cpu")
    processor = VideoProcessor(
        detector,
        service,
        target_fps=5,
        min_track_frames=1,
        max_video_seconds=120,
        max_frame_pixels=8294400,
        synchronous_attributes=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    video_output = output / "result.mp4"
    metadata_output = output / "result.jsonl"
    try:
        processor.process(video, video_output, metadata_output, "peta-check")
    finally:
        service.close()
    rows = [json.loads(line) for line in metadata_output.read_text(encoding="utf-8").splitlines()]
    people = [person for row in rows for person in row["people"]]
    if not people:
        raise ValueError("No people detected; choose a video with visible full-body people")
    for person in people:
        attributes = person["attributes"]
        if set(attributes) != set(PETA_LABELS):
            raise ValueError("The video pipeline did not return all nine attributes")
    if not any(
        any(value != "не определен" for value in person["attributes"].values()) for person in people
    ):
        raise ValueError("No attribute prediction completed")
    capture = cv2.VideoCapture(str(video_output))
    readable, _ = capture.read()
    capture.release()
    if not readable:
        raise ValueError("Output video is not readable")
    print(
        f"Video: {video_output}; frames={len(rows)}; person_records={len(people)}; "
        "nine attributes verified"
    )


if __name__ == "__main__":
    main()
