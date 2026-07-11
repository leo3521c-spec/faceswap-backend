"""
Production-grade Latest Frame Queue + 3-thread pipeline.

Architecture
────────────
    ┌───────────┐        ┌──────────────┐        ┌──────────┐
    │  Capture  │──put──→│  FrameQueue  │──get──→│ Process  │
    │  (async)  │        │  max_size=1  │        │ (thread) │
    └───────────┘        │  latest-wins │        └────┬─────┘
                         └──────────────┘             │
                          drops stale                  │ put_threadsafe
                          frames auto                  ↓
                                              ┌──────────────┐
                                              │ OutputQueue  │
                                              │  (asyncio)   │
                                              └──────┬───────┘
                                                     │ get
                                                     ↓
                                              ┌──────────────┐
                                              │   Sender     │
                                              │   (async)    │
                                              └──────────────┘

• Capture task  — async, reads binary JPEGs from the WebSocket
• Processing    — real OS thread, blocks on queue.get(), runs GPU inference
• Sender task   — async, sends JSON metadata + binary JPEG back to client

The queue always keeps the NEWEST frame and drops older ones, so
end-to-end latency stays under 100 ms even under heavy load.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utils.logger import setup_logger

logger = setup_logger("frame_queue")


# ── Latest Frame Queue ───────────────────────────────────────


@dataclass
class QueuedFrame:
    """A frame waiting in the queue."""
    frame_id: int
    data: bytes
    put_time: float  # perf_counter() at put


class LatestFrameQueue:
    """
    Thread-safe, single-slot, latest-frame-wins queue.

    • Maximum queue size = 1
    • put() is non-blocking — if a frame is already waiting, it is
      replaced (the old one is counted as dropped)
    • get() blocks until a frame arrives or timeout
    • Tracks: dropped frames, queue size, avg wait time, frame ID
    """

    MAX_SIZE = 1

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame: Optional[QueuedFrame] = None

        # Metrics
        self._frame_counter = 0
        self._dropped_count = 0
        self._wait_times: deque = field(default_factory=lambda: deque(maxlen=120))
        self._last_frame_id = 0

    # ── Producer ────────────────────────────────────────────

    def put(self, data: bytes) -> int:
        """
        Store *data* as the latest frame. If a frame is already
        pending, it is dropped (replaced). Non-blocking, thread-safe.
        Returns the assigned frame_id.
        """
        with self._cond:
            self._frame_counter += 1
            frame_id = self._frame_counter

            if self._frame is not None:
                self._dropped_count += 1
                logger.debug(
                    "Dropped frame %d (replaced by %d)",
                    self._frame.frame_id,
                    frame_id,
                )

            self._frame = QueuedFrame(
                frame_id=frame_id,
                data=data,
                put_time=time.perf_counter(),
            )
            self._cond.notify_all()
            return frame_id

    # ── Consumer ────────────────────────────────────────────

    def get(self, timeout: float = 0.5) -> Optional[QueuedFrame]:
        """
        Block until a frame is available or *timeout* elapses.
        Returns the QueuedFrame, or None on timeout.
        Records wait time for metrics.
        """
        with self._cond:
            if self._frame is None:
                self._cond.wait(timeout=timeout)

            if self._frame is None:
                return None

            frame = self._frame
            self._frame = None

            wait_time = time.perf_counter() - frame.put_time
            self._wait_times.append(wait_time)
            self._last_frame_id = frame.frame_id
            return frame

    # ── Control ─────────────────────────────────────────────

    def clear(self) -> None:
        """Discard any pending frame."""
        with self._cond:
            self._frame = None
            self._cond.notify_all()

    # ── Metrics ─────────────────────────────────────────────

    @property
    def queue_size(self) -> int:
        """Current occupancy (0 or 1)."""
        with self._lock:
            return 1 if self._frame is not None else 0

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    @property
    def last_frame_id(self) -> int:
        with self._lock:
            return self._last_frame_id

    @property
    def avg_wait_time_ms(self) -> float:
        with self._lock:
            if not self._wait_times:
                return 0.0
            return (sum(self._wait_times) / len(self._wait_times)) * 1000

    def get_metrics(self) -> dict:
        return {
            "queue_size": self.queue_size,
            "max_size": self.MAX_SIZE,
            "dropped_frames": self.dropped_count,
            "last_frame_id": self.last_frame_id,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
        }


# ── Pipeline Metrics ─────────────────────────────────────────


@dataclass
class PipelineMetrics:
    """Aggregated metrics across the capture → process → send pipeline."""
    frames_captured: int = 0
    frames_processed: int = 0
    frames_sent: int = 0
    frames_dropped: int = 0
    _processing_times: deque = field(default_factory=lambda: deque(maxlen=120))
    _wait_times: deque = field(default_factory=lambda: deque(maxlen=120))
    _latencies: deque = field(default_factory=lambda: deque(maxlen=120))
    _send_times: deque = field(default_factory=lambda: deque(maxlen=120))
    _last_frame_id: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_capture(self, frame_id: int) -> None:
        with self._lock:
            self.frames_captured += 1

    def record_drop(self) -> None:
        with self._lock:
            self.frames_dropped += 1

    def record_processed(
        self,
        frame_id: int,
        processing_ms: float,
        wait_ms: float,
    ) -> None:
        with self._lock:
            self.frames_processed += 1
            self._processing_times.append(processing_ms)
            self._wait_times.append(wait_ms)
            self._last_frame_id = frame_id

    def record_sent(self, frame_id: int, latency_ms: float) -> None:
        with self._lock:
            self.frames_sent += 1
            self._latencies.append(latency_ms)
            self._send_times.append(time.time())

    @property
    def avg_processing_time_ms(self) -> float:
        with self._lock:
            if not self._processing_times:
                return 0.0
            return sum(self._processing_times) / len(self._processing_times)

    @property
    def avg_wait_time_ms(self) -> float:
        with self._lock:
            if not self._wait_times:
                return 0.0
            return sum(self._wait_times) / len(self._wait_times) * 1000

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    @property
    def current_fps(self) -> float:
        with self._lock:
            if len(self._send_times) < 2:
                return 0.0
            elapsed = self._send_times[-1] - self._send_times[0]
            if elapsed <= 0:
                return 0.0
            return len(self._send_times) / elapsed

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "frames_captured": self.frames_captured,
                "frames_processed": self.frames_processed,
                "frames_sent": self.frames_sent,
                "frames_dropped": self.frames_dropped,
                "last_frame_id": self._last_frame_id,
                "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
                "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "current_fps": round(self.current_fps, 2),
                "target_latency_ms": 100,
                "latency_ok": self.avg_latency_ms < 100,
            }


# ── Frame Pipeline ───────────────────────────────────────────


class FramePipeline:
    """
    Orchestrates the 3-thread face-swap pipeline:

        Capture (async) → LatestFrameQueue → Processing (thread) → OutputQueue → Sender (async)

    The processing thread runs real GPU inference off the event loop.
    Capture and sender are async tasks (WebSocket I/O requires the loop).
    """

    def __init__(
        self,
        process_fn: Callable[[bytes, Any], Any],
        source_face: Any,
        loop: asyncio.AbstractEventLoop,
        on_result: Callable[[Any, int, float, float], Any] | None = None,
    ) -> None:
        self._process_fn = process_fn
        self._source_face = source_face
        self._loop = loop
        self._on_result = on_result

        self.input_queue = LatestFrameQueue()
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.metrics = PipelineMetrics()

        self._stop_event = threading.Event()
        self._proc_thread: Optional[threading.Thread] = None

    # ── Lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        """Start the processing thread."""
        self._proc_thread = threading.Thread(
            target=self._processing_loop,
            name="face-swap-processing",
            daemon=True,
        )
        self._proc_thread.start()
        logger.info("Processing thread started")

    def stop(self) -> None:
        """Signal all threads to stop and clean up."""
        self._stop_event.set()
        self.input_queue.clear()
        if self._proc_thread and self._proc_thread.is_alive():
            self._proc_thread.join(timeout=2.0)
        logger.info("Processing thread stopped")

    # ── Capture (called from async task) ────────────────────

    def submit_frame(self, data: bytes) -> int:
        """
        Called by the capture task when a new frame arrives from the WS.
        Non-blocking, thread-safe. Returns the frame_id.
        """
        frame_id = self.input_queue.put(data)
        self.metrics.record_capture(frame_id)
        return frame_id

    # ── Processing (real OS thread) ─────────────────────────

    def _processing_loop(self) -> None:
        """Main loop of the processing thread."""
        while not self._stop_event.is_set():
            queued = self.input_queue.get(timeout=0.5)
            if queued is None:
                continue

            wait_ms = (time.perf_counter() - queued.put_time) * 1000

            try:
                result = self._process_fn(queued.data, self._source_face)
                processing_ms = getattr(result, "inference_time_ms", 0.0)

                self.metrics.record_processed(
                    queued.frame_id, processing_ms, wait_ms
                )

                # Thread-safe put into the async output queue
                self._loop.call_soon_threadsafe(
                    self._put_output, result, queued.frame_id, queued.put_time
                )
            except Exception as exc:
                logger.error(
                    "Processing error on frame %d: %s",
                    queued.frame_id,
                    exc,
                )

    def _put_output(self, result: Any, frame_id: int, put_time: float) -> None:
        """Called on the event loop via call_soon_threadsafe."""
        try:
            self.output_queue.put_nowait((result, frame_id, put_time))
        except asyncio.QueueFull:
            # Output queue is full — drop oldest by replacing
            logger.debug("Output queue full, dropping result for frame %d", frame_id)
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.output_queue.put_nowait((result, frame_id, put_time))
            except asyncio.QueueFull:
                pass

    # ── Sender (called from async task) ─────────────────────

    async def get_result(self) -> tuple[Any, int, float]:
        """
        Called by the sender task. Blocks until a result is available.
        Returns (result, frame_id, put_time).
        """
        return await self.output_queue.get()

    # ── Metrics ─────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Combined queue + pipeline metrics."""
        data = self.metrics.to_dict()
        data["queue"] = self.input_queue.get_metrics()
        return data


# ── Global pipeline reference (for /metrics endpoint) ────────

_active_pipeline: Optional[FramePipeline] = None


def set_active_pipeline(pipeline: FramePipeline | None) -> None:
    global _active_pipeline
    _active_pipeline = pipeline


def get_active_pipeline_metrics() -> dict:
    if _active_pipeline is None:
        return {
            "active": False,
            "message": "No active pipeline",
        }
    return {
        "active": True,
        **_active_pipeline.get_metrics(),
    }