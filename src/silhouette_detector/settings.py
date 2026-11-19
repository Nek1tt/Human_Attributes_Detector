"""Typed runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    """Application configuration.

    Secrets and machine-specific paths are intentionally kept out of source control.
    """

    root: Path = field(default_factory=_project_root)
    detector_model: Path = field(default_factory=lambda: _project_root() / "models/yolo.onnx")
    attribute_backend: str = "peta"
    attribute_checkpoint: Path = field(
        default_factory=lambda: _project_root() / "models/original_resnet_peta.pt"
    )
    device: str = "auto"
    target_fps: float = 10.0
    min_track_frames: int = 3
    synchronous_attributes: bool = False
    max_upload_bytes: int = 250 * 1024 * 1024
    max_video_seconds: int = 600
    max_frame_pixels: int = 3840 * 2160
    max_concurrent_jobs: int = 1
    max_queued_jobs: int = 8
    api_key: str | None = None
    allow_unauthenticated_local: bool = True
    data_dir: Path = field(default_factory=lambda: _project_root() / "var")

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(os.getenv("HAD_ROOT", str(_project_root()))).expanduser().resolve()
        backend = os.getenv("HAD_ATTRIBUTE_BACKEND", "peta").strip().lower()

        def path_value(name: str, relative_default: str) -> Path:
            raw = os.getenv(name)
            if raw:
                value = Path(raw).expanduser()
                if not value.is_absolute():
                    value = root / value
            else:
                value = root / relative_default
            return value.resolve()

        settings = cls(
            root=root,
            detector_model=path_value("HAD_DETECTOR_MODEL", "models/yolo.onnx"),
            attribute_backend=backend,
            attribute_checkpoint=path_value(
                "HAD_ATTRIBUTE_CHECKPOINT", "models/original_resnet_peta.pt"
            ),
            device=os.getenv("HAD_DEVICE", "auto").strip().lower(),
            target_fps=float(os.getenv("HAD_TARGET_FPS", "10")),
            min_track_frames=_env_int("HAD_MIN_TRACK_FRAMES", 3, minimum=1),
            synchronous_attributes=_env_bool("HAD_SYNCHRONOUS_ATTRIBUTES", False),
            max_upload_bytes=_env_int("HAD_MAX_UPLOAD_BYTES", 250 * 1024 * 1024, minimum=1024),
            max_video_seconds=_env_int("HAD_MAX_VIDEO_SECONDS", 600, minimum=1),
            max_frame_pixels=_env_int("HAD_MAX_FRAME_PIXELS", 3840 * 2160, minimum=224 * 224),
            max_concurrent_jobs=_env_int("HAD_MAX_CONCURRENT_JOBS", 1, minimum=1),
            max_queued_jobs=_env_int("HAD_MAX_QUEUED_JOBS", 8, minimum=1),
            api_key=os.getenv("HAD_API_KEY") or None,
            allow_unauthenticated_local=_env_bool("HAD_ALLOW_UNAUTHENTICATED_LOCAL", True),
            data_dir=path_value("HAD_DATA_DIR", "var"),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.attribute_backend not in {"peta", "none"}:
            raise ValueError("HAD_ATTRIBUTE_BACKEND must be peta or none")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("HAD_DEVICE must be auto, cpu, cuda, or cuda:<index>")
        if self.target_fps <= 0:
            raise ValueError("HAD_TARGET_FPS must be positive")
        if self.api_key is not None and len(self.api_key) < 24:
            raise ValueError("HAD_API_KEY must contain at least 24 characters")

    def ensure_runtime_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
