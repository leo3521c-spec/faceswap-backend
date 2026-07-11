"""
Microphone capture service for server-side audio input.

Architecture (to be implemented):
  • Uses sounddevice (PortAudio) for cross-platform capture
  • Captures 16-bit signed PCM at configurable sample rate
  • Outputs fixed-size chunks (e.g. 480 samples @ 24 kHz = 20 ms)
  • Runs capture in a dedicated daemon thread
  • Supports device selection by index or name

When the voice pipeline receives audio via WebSocket (client-side mic),
this service is not used. It is for server-side capture scenarios
(e.g. capturing system audio for OBS virtual cable output).
"""
from __future__ import annotations

import threading

from utils.logger import setup_logger

logger = setup_logger("microphone_capture")


class MicrophoneCapture:
    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_duration_ms: int = 20,
        channels: int = 1,
        device: int | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration_ms = chunk_duration_ms
        self._channels = channels
        self._device = device
        self._active = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def active(self) -> bool:
        return self._active

    @property
    def chunk_samples(self) -> int:
        return int(self._sample_rate * self._chunk_duration_ms / 1000)

    def start(self) -> dict:
        """Open the microphone device and begin capture.

        TODO: implement sounddevice.InputStream with callback.
        """
        self._active = True
        self._stop_event.clear()
        logger.info(
            "Microphone capture started (device=%s, rate=%d, chunk=%d samples)",
            self._device,
            self._sample_rate,
            self.chunk_samples,
        )
        return self.get_status()

    def stop(self) -> dict:
        """Stop capture and release the device."""
        self._stop_event.set()
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Microphone capture stopped")
        return self.get_status()

    def list_devices(self) -> list:
        """List available input devices.

        TODO: implement via sounddevice.query_devices().
        """
        return []

    def get_status(self) -> dict:
        return {
            "active": self._active,
            "device": self._device,
            "sample_rate": self._sample_rate,
            "channels": self._channels,
            "chunk_duration_ms": self._chunk_duration_ms,
            "chunk_samples": self.chunk_samples,
            "available_devices": self.list_devices(),
        }


microphone_capture = MicrophoneCapture()