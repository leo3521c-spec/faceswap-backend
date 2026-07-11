"""
Acoustic echo cancellation for the voice pipeline.

Architecture (to be implemented):
  • WebRTC AEC3 or SpeexDSP adaptive filter
  • Requires a reference signal — the audio being played back to speakers
  • Tail length configurable (default 128 ms)
  • Runs in the processing thread; adds < 1 ms per chunk

The reference signal is fed via set_reference() from the output stage
so the canceller knows what to subtract from the microphone input.
"""
from __future__ import annotations

import threading

from utils.logger import setup_logger
from services.audio_chunk import AudioChunk

logger = setup_logger("echo_canceller")


class EchoCanceller:
    def __init__(self, enabled: bool = True, tail_length_ms: int = 128) -> None:
        self._enabled = enabled
        self._tail_length_ms = tail_length_ms
        self._reference: bytes | None = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        logger.info("Echo cancellation: %s", "ON" if enabled else "OFF")

    def set_tail_length(self, ms: int) -> None:
        self._tail_length_ms = ms

    def set_reference(self, reference_pcm: bytes) -> None:
        """Feed the played-back audio so the filter can cancel it."""
        with self._lock:
            self._reference = reference_pcm

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Cancel echo from chunk using the reference signal.

        TODO: implement adaptive filter (WebRTC AEC3 / SpeexDSP).
        Currently passes audio through unchanged.
        """
        return chunk

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "tail_length_ms": self._tail_length_ms,
            "has_reference": self._reference is not None,
        }


echo_canceller = EchoCanceller()