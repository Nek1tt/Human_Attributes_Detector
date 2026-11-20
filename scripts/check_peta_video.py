"""Run YOLO, tracking and the trained nine-attribute model on a local video."""

import json
from pathlib import Path
from time import perf_counter

import click
import cv2
import torch

from silhouette_detector.attribute_service import AttributeService
from silhouette_detector.attributes.factory import create_attribute_backend
from silhouette_detector.detection import YoloOnnxDetector
from silhouette_detector.device import resolve_torch_device
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
    device = resolve_torch_device(device)
    cuda = device.startswith("cuda")
    if cuda:
        # The allocator must exist before resetting its counters in a fresh process.
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
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
        if cuda:
            torch.cuda.synchronize(device)
        start = perf_counter()
        processor.process(video, video_output, metadata_output, "peta-check")
        if cuda:
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
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
    metrics = {
        "video": video.name,
        "checkpoint": checkpoint.name,
        "classifier_device": device,
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "detector_device": "cpu",
        "target_fps": processor.target_fps,
        "processed_frames": len(rows),
        "person_records": len(people),
        "processing_seconds": elapsed,
        "peak_torch_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None
        ),
    }
    report = json.dumps(metrics, indent=2)
    (output / "benchmark.json").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(
        f"Video: {video_output}; frames={len(rows)}; person_records={len(people)}; "
        "nine attributes verified"
    )


if __name__ == "__main__":
    main()
