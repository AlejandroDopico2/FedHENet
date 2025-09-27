from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class CommunicationStats:
    published_bytes: int = 0
    received_bytes: int = 0


class MetricsRecorder:
    _instance: "MetricsRecorder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._start_time_s: float | None = None
        self._end_time_s: float | None = None
        self.comm = CommunicationStats()

    @classmethod
    def instance(cls) -> "MetricsRecorder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = MetricsRecorder()
            return cls._instance

    # Timing
    def start(self) -> None:
        self._start_time_s = time.time()

    def end(self) -> None:
        self._end_time_s = time.time()

    def elapsed_seconds(self) -> float:
        if self._start_time_s is None:
            return 0.0
        end = self._end_time_s if self._end_time_s is not None else time.time()
        return max(0.0, end - self._start_time_s)

    # Communication
    def add_published_bytes(self, size_bytes: int) -> None:
        self.comm.published_bytes += int(size_bytes)

    def add_received_bytes(self, size_bytes: int) -> None:
        self.comm.received_bytes += int(size_bytes)

    def snapshot(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds(),
            "published_bytes": self.comm.published_bytes,
            "received_bytes": self.comm.received_bytes,
        }


__all__ = ["MetricsRecorder"]
