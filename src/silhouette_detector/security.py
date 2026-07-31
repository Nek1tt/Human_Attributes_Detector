"""Small security primitives used by the HTTP boundary."""

from __future__ import annotations

import ipaddress
import secrets
from pathlib import Path

ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def validated_video_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_SUFFIXES))
        raise ValueError(f"Unsupported video extension. Allowed: {allowed}")
    return suffix


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def is_authorized(
    configured_key: str | None,
    supplied_key: str | None,
    client_host: str | None,
    allow_unauthenticated_local: bool,
) -> bool:
    if configured_key:
        return supplied_key is not None and secrets.compare_digest(configured_key, supplied_key)
    return allow_unauthenticated_local and is_loopback_host(client_host)
