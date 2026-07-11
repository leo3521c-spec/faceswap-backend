"""
Lightweight metrics collector with rolling-window averages.
Thread-safe for use across asyncio + thread-pool boundaries.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    frames_processed: int = 0
    frames_dropped: int = 0
    total_faces_detected: int = 0
    _latencies: deque = field(default_factory=lambda: deque(maxlen=120))
    _fps_times: deque = field(default_factory=lambda: deque(maxlen=120))
    _confidences: deque = field(default_factory=lambda: deque(maxlen=120))
    _face_counts: deque = field(default_factory=lambda: deque(maxlen=120))
    _start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_frame(
        self,
        latency_ms: float,
        face_count: int = 0,
        confidence: float = 0.0,
    ) -> None:
        with self._lock:
            self.frames_processed += 1
            self.total_faces_detected += face_count
            now = time.time()
            self._latencies.append(latency_ms)
            self._fps_times.append(now)
            self._confidences.append(confidence)
            self._face_counts.append(face_count)

    def record_drop(self) -> None:
        with self._lock:
            self.frames_dropped += 1

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def current_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        if elapsed <= 0:
            return 0.0
        return len(self._fps_times) / elapsed

    @property
    def avg_confidence(self) -> float:
        if not self._confidences:
            return 0.0
        return sum(self._confidences) / len(self._confidences)

    @property
    def avg_face_count(self) -> float:
        if not self._face_counts:
            return 0.0
        return sum(self._face_counts) / len(self._face_counts)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "total_faces_detected": self.total_faces_detected,
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "current_fps": round(self.current_fps, 2),
                "avg_confidence": round(self.avg_confidence, 4),
                "avg_face_count": round(self.avg_face_count, 2),
                "uptime_seconds": round(time.time() - self._start_time, 1),
            }


metrics = MetricsCollector()