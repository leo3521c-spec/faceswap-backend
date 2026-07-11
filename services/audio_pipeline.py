"""
Low-latency audio pipeline — 3-thread architecture for real-time voice changing.

    ┌───────────┐        ┌──────────────┐        ┌──────────┐
    │  Capture  │──put──→│  AudioQueue  │──get──→│ Process  │
    │  (async)  │        │  max_size=1  │        │ (thread) │
    └───────────┘        │  latest-wins │        └────┬─────┘
                         └──────────────┘             │ put_threadsafe
                          drops stale                  ↓
                          chunks auto          ┌──────────────┐
                                               │ OutputQueue  │
                                               │  (asyncio)   │
                                               └──────┬───────┘
                                                      │ get
                                                      ↓
                                               ┌──────────────┐
                                               │   Sender     │
                                               │   (async)    │
                                               └──────────────┘

• Capture task  — async, reads binary PCM from the WebSocket
• Processing    — real OS thread, runs the voice processing chain
• Sender task   — async, sends JSON metadata + binary PCM back to client

The queue always keeps the NEWEST chunk and drops older ones, so
end-to-end latency stays low even under heavy load.
Target end-to-end latency: < 50 ms.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from utils.logger import setup_logger

logger = setup_logger("audio_pipeline")


# ── Queued Chunk ──────────────────────────────────────────────


@dataclass
class QueuedChunk:
    chunk_id: int
    data: bytes
    sample_rate: int
    channels: int
    put_time: float


# ── Latest Chunk Queue ────────────────────────────────────────


class LatestChunkQueue:
    """Thread-safe, single-slot, latest-chunk-wins queue."""

    MAX_SIZE = 1

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._chunk: Optional[QueuedChunk] = None
        self._counter = 0
        self._dropped = 0

    def put(self, data: bytes, sample_rate: int, channels: int) -> int:
        with self._cond:
            self._counter += 1
            chunk_id = self._counter
            if self._chunk is not None:
                self._dropped += 1
            self._chunk = QueuedChunk(
                chunk_id=chunk_id,
                data=data,
                sample_rate=sample_rate,
                channels=channels,
                put_time=time.perf_counter(),
            )
            self._cond.notify_all()
            return chunk_id

    def get(self, timeout: float = 0.5) -> Optional[QueuedChunk]:
        with self._cond:
            if self._chunk is None:
                self._cond.wait(timeout=timeout)
            if self._chunk is None:
                return None
            chunk = self._chunk
            self._chunk = None
            return chunk

    def clear(self) -> None:
        with self._cond:
            self._chunk = None
            self._cond.notify_all()

    def get_metrics(self) -> dict:
        with self._lock:
            return {
                "queue_size": 1 if self._chunk else 0,
                "max_size": self.MAX_SIZE,
                "dropped_chunks": self._dropped,
                "last_chunk_id": self._counter,
            }


# ── Pipeline Metrics ──────────────────────────────────────────


@dataclass
class VoicePipelineMetrics:
    chunks_captured: int = 0
    chunks_processed: int = 0
    chunks_sent: int = 0
    chunks_dropped: int = 0
    _processing_times: deque = field(default_factory=lambda: deque(maxlen=120))
    _latencies: deque = field(default_factory=lambda: deque(maxlen=120))
    _send_times: deque = field(default_factory=lambda: deque(maxlen=120))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_capture(self) -> None:
        with self._lock:
            self.chunks_captured += 1

    def record_processed(self, processing_ms: float) -> None:
        with self._lock:
            self.chunks_processed += 1
            self._processing_times.append(processing_ms)

    def record_sent(self, latency_ms: float) -> None:
        with self._lock:
            self.chunks_sent += 1
            self._latencies.append(latency_ms)
            self._send_times.append(time.time())

    @property
    def avg_processing_ms(self) -> float:
        with self._lock:
            if not self._processing_times:
                return 0.0
            return sum(self._processing_times) / len(self._processing_times)

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)

    @property
    def current_chunks_per_sec(self) -> float:
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
                "chunks_captured": self.chunks_captured,
                "chunks_processed": self.chunks_processed,
                "chunks_sent": self.chunks_sent,
                "chunks_dropped": self.chunks_dropped,
                "avg_processing_time_ms": round(self.avg_processing_ms, 2),
                "avg_latency_ms": round(self.avg_latency_ms, 2),
                "current_chunks_per_sec": round(self.current_chunks_per_sec, 2),
                "target_latency_ms": 50,
                "latency_ok": self.avg_latency_ms < 50,
            }


# ── Voice Pipeline ────────────────────────────────────────────


class VoicePipeline:
    """3-thread audio pipeline: capture (async) -> process (thread) -> send (async)."""

    def __init__(
        self,
        process_fn: Callable,
        sample_rate: int,
        channels: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._process_fn = process_fn
        self._sample_rate = sample_rate
        self._channels = channels
        self._loop = loop

        self.input_queue = LatestChunkQueue()
        self.output_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self.metrics = VoicePipelineMetrics()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._processing_loop,
            name="voice-processing",
            daemon=True,
        )
        self._thread.start()
        logger.info("Voice processing thread started")

    def stop(self) -> None:
        self._stop_event.set()
        self.input_queue.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Voice processing thread stopped")

    def submit_chunk(self, data: bytes) -> int:
        """Called by the capture task when a new PCM chunk arrives."""
        chunk_id = self.input_queue.put(
            data, self._sample_rate, self._channels
        )
        self.metrics.record_capture()
        return chunk_id

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            queued = self.input_queue.get(timeout=0.5)
            if queued is None:
                continue
            try:
                result = self._process_fn(
                    queued.data,
                    queued.sample_rate,
                    queued.channels,
                    queued.chunk_id,
                )
                self.metrics.record_processed(result.processing_time_ms)
                self._loop.call_soon_threadsafe(
                    self._put_output, result, queued.put_time
                )
            except Exception as exc:
                logger.error(
                    "Audio processing error on chunk %d: %s",
                    queued.chunk_id, exc,
                )

    def _put_output(self, result: Any, put_time: float) -> None:
        try:
            self.output_queue.put_nowait((result, put_time))
        except asyncio.QueueFull:
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.output_queue.put_nowait((result, put_time))
            except asyncio.QueueFull:
                pass

    async def get_result(self) -> tuple:
        return await self.output_queue.get()

    def get_metrics(self) -> dict:
        data = self.metrics.to_dict()
        data["queue"] = self.input_queue.get_metrics()
        return data


# ── Global pipeline reference ─────────────────────────────────

_active_voice_pipeline: Optional[VoicePipeline] = None


def set_active_voice_pipeline(pipeline: VoicePipeline | None) -> None:
    global _active_voice_pipeline
    _active_voice_pipeline = pipeline


def get_active_voice_pipeline_metrics() -> dict:
    if _active_voice_pipeline is None:
        return {"active": False, "message": "No active voice pipeline"}
    return {"active": True, **_active_voice_pipeline.get_metrics()}