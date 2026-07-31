"""One model instance and one bounded inference worker shared by all video jobs."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from PIL import Image

from .attributes.base import AttributeBackend

LOGGER = logging.getLogger(__name__)
TrackKey = tuple[str, int]


class AttributeService:
    def __init__(self, backend: AttributeBackend, max_pending: int = 32) -> None:
        self.backend = backend
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="attributes")
        self._futures: dict[TrackKey, Future[dict[str, str]]] = {}
        self._results: dict[TrackKey, dict[str, str]] = {}
        self._failed: set[TrackKey] = set()
        self._lock = Lock()

    def submit(self, task_id: str, track_id: int, image: Image.Image) -> bool:
        key = (task_id, track_id)
        with self._lock:
            if key in self._results or key in self._futures or key in self._failed:
                return False
            if len(self._futures) >= self.max_pending:
                return False
            self._futures[key] = self._executor.submit(self.backend.predict, image.copy())
            return True

    def get(self, task_id: str, track_id: int) -> dict[str, str] | None:
        key = (task_id, track_id)
        with self._lock:
            result = self._results.get(key)
            future = self._futures.get(key)
        if result is not None:
            return dict(result)
        if future is None or not future.done():
            return None
        try:
            result = self.backend.normalize(future.result())
        except Exception:
            LOGGER.exception("Attribute inference failed for task=%s track=%s", task_id, track_id)
            with self._lock:
                self._failed.add(key)
                self._futures.pop(key, None)
            return None
        with self._lock:
            self._results[key] = result
            self._futures.pop(key, None)
        return dict(result)

    def drop_task(self, task_id: str) -> None:
        with self._lock:
            keys = {key for key in self._futures | self._results if key[0] == task_id}
            for key in keys:
                future = self._futures.pop(key, None)
                if future is not None:
                    future.cancel()
                self._results.pop(key, None)
                self._failed.discard(key)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.backend.close()
