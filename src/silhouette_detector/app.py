"""Authenticated, bounded FastAPI job API."""

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from uuid import uuid4

from .attribute_service import AttributeService
from .attributes.factory import create_attribute_backend
from .detection import YoloOnnxDetector
from .pipeline import VideoProcessor
from .security import is_authorized, validated_video_suffix
from .settings import Settings

LOGGER = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class JobRecord:
    job_id: str
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    error: str | None = None


class JobQueueFullError(RuntimeError):
    pass


class JobManager:
    """Owns bounded video workers and lazily initializes expensive models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_runtime_dirs()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs, thread_name_prefix="video-job"
        )
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self._runtime_lock = Lock()
        self._processor: VideoProcessor | None = None
        self._attribute_service: AttributeService | None = None
        self._runtime_error: str | None = None

    def _ensure_runtime(self) -> VideoProcessor:
        if self._processor is not None:
            return self._processor
        with self._runtime_lock:
            if self._processor is not None:
                return self._processor
            try:
                detector = YoloOnnxDetector(self.settings.detector_model, self.settings.device)
                backend = create_attribute_backend(self.settings)
                self._attribute_service = AttributeService(backend)
                self._processor = VideoProcessor(
                    detector,
                    self._attribute_service,
                    target_fps=self.settings.target_fps,
                    min_track_frames=self.settings.min_track_frames,
                    max_video_seconds=self.settings.max_video_seconds,
                    max_frame_pixels=self.settings.max_frame_pixels,
                    synchronous_attributes=self.settings.synchronous_attributes,
                )
                self._runtime_error = None
            except Exception as exc:
                self._runtime_error = str(exc)
                raise
        return self._processor

    def submit(self, input_path: Path) -> JobRecord:
        job_id = uuid4().hex
        record = JobRecord(job_id)
        with self._lock:
            active_jobs = sum(
                job.state in {JobState.QUEUED, JobState.RUNNING} for job in self._jobs.values()
            )
            if active_jobs >= self.settings.max_queued_jobs:
                raise JobQueueFullError("The job queue is full")
            self._jobs[job_id] = record
        self._executor.submit(self._run, job_id, input_path)
        return JobRecord(**asdict(record))

    def _update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            record = self._jobs[job_id]
            for key, value in changes.items():
                setattr(record, key, value)

    def _run(self, job_id: str, input_path: Path) -> None:
        output_path = self.settings.output_dir / f"{job_id}.mp4"
        metadata_path = self.settings.output_dir / f"{job_id}.jsonl"
        self._update(job_id, state=JobState.RUNNING)
        try:
            processor = self._ensure_runtime()
            processor.process(
                input_path,
                output_path,
                metadata_path,
                job_id,
                progress=lambda value: self._update(job_id, progress=float(value)),
            )
            self._update(job_id, state=JobState.COMPLETED, progress=1.0)
        except Exception as exc:
            LOGGER.exception("Video job %s failed", job_id)
            output_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            self._update(job_id, state=JobState.FAILED, error=str(exc)[:500])
        finally:
            input_path.unlink(missing_ok=True)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return JobRecord(**asdict(record)) if record else None

    def result_path(self, job_id: str) -> Path:
        return self.settings.output_dir / f"{job_id}.mp4"

    def delete(self, job_id: str) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return False
            if record.state in {JobState.QUEUED, JobState.RUNNING}:
                raise RuntimeError("A running job cannot be deleted")
            del self._jobs[job_id]
        self.result_path(job_id).unlink(missing_ok=True)
        (self.settings.output_dir / f"{job_id}.jsonl").unlink(missing_ok=True)
        return True

    @property
    def runtime_status(self) -> dict[str, str | bool | None]:
        return {
            "initialized": self._processor is not None,
            "initialization_failed": self._runtime_error is not None,
            "backend": self.settings.attribute_backend,
            "device": self.settings.device,
            "synchronous_attributes": self.settings.synchronous_attributes,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        if self._attribute_service is not None:
            self._attribute_service.close()


def create_app(settings: Settings | None = None, manager: JobManager | None = None):
    try:
        from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Install the api extra to run the HTTP service") from exc

    settings = settings or Settings.from_env()
    manager = manager or JobManager(settings)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        manager.close()

    app = FastAPI(title="Human Attributes Detector", version="2.0.0", lifespan=lifespan)
    app.state.manager = manager

    async def authorize(
        request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")
    ) -> None:
        client_host = request.client.host if request.client else None
        if not is_authorized(
            settings.api_key,
            x_api_key,
            client_host,
            settings.allow_unauthenticated_local,
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "runtime": manager.runtime_status}

    @app.post("/api/v1/jobs", dependencies=[Depends(authorize)], status_code=202)
    async def create_job(file: UploadFile = File(...)) -> dict[str, object]:
        try:
            suffix = validated_video_suffix(file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        upload_path = settings.upload_dir / f"{uuid4().hex}{suffix}"
        total = 0
        try:
            with upload_path.open("xb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Uploaded video is too large")
                    output.write(chunk)
        except Exception:
            upload_path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        if total == 0:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="Uploaded video is empty")
        try:
            record = manager.submit(upload_path)
        except JobQueueFullError as exc:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {"job_id": record.job_id, "state": record.state, "progress": record.progress}

    @app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_job(job_id: str) -> dict[str, object]:
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return asdict(record)

    @app.get("/api/v1/jobs/{job_id}/result", dependencies=[Depends(authorize)])
    def get_result(job_id: str):
        record = manager.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.state != JobState.COMPLETED:
            raise HTTPException(status_code=409, detail=f"Job is {record.state}")
        path = manager.result_path(job_id)
        if not path.is_file():
            raise HTTPException(status_code=410, detail="Result file is no longer available")
        return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")

    @app.delete("/api/v1/jobs/{job_id}", dependencies=[Depends(authorize)], status_code=204)
    def delete_job(job_id: str) -> None:
        try:
            deleted = manager.delete(job_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Job not found")

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
