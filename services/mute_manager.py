"""
Mute manager for the voice pipeline.

When muted, replaces PCM data with silence (zeros) while preserving
chunk metadata so downstream consumers still receive well-formed chunks.

Supports:
  • Toggle mode — mute stays until explicitly unmuted
  • Push-to-talk — unmuted only while held

Thread-safe; called from the processing thread.
"""
from __future__ import annotations

import threading

from utils.logger import setup_logger
from services.audio_chunk import AudioChunk

logger = setup_logger("mute_manager")


class MuteManager:
    def __init__(self, muted: bool = False) -> None:
        self._muted = muted
        self._lock = threading.Lock()

    def is_muted(self) -> bool:
        with self._lock:
            return self._muted

    def set_muted(self, muted: bool) -> bool:
        with self._lock:
            self._muted = muted
            logger.info("Mute: %s", "ON" if muted else "OFF")
            return self._muted

    def toggle(self) -> bool:
        with self._lock:
            self._muted = not self._muted
            logger.info("Mute toggled: %s", "ON" if self._muted else "OFF")
            return self._muted

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """If muted, replace audio with silence. Returns the chunk."""
        if not self.is_muted():
            return chunk
        chunk.pcm_data = bytes(len(chunk.pcm_data))  # zero-filled silence
        return chunk

    def get_status(self) -> dict:
        return {"muted": self.is_muted()}


mute_manager = MuteManager()