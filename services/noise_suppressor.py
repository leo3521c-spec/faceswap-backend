"""
Real-time noise suppression for the voice pipeline.

Architecture (to be implemented):
  • RNNoise (Mozilla) or DeepFilterNet for neural noise suppression
  • Processes 20 ms frames at 16/24/48 kHz
  • Aggressiveness levels 0-4 (0 = off, 4 = max suppression)
  • GPU acceleration optional via ONNX Runtime

Runs in the processing thread; target < 2 ms per 20 ms chunk.
"""
from __future__ import annotations

import threading

from utils.logger import setup_logger
from services.audio_chunk import AudioChunk

logger = setup_logger("noise_suppressor")


class NoiseSuppressor:
    def __init__(self, enabled: bool = True, aggressiveness: int = 2) -> None:
        self._enabled = enabled
        self._aggressiveness = aggressiveness
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        logger.info("Noise suppression: %s", "ON" if enabled else "OFF")

    def set_aggressiveness(self, level: int) -> bool:
        if not 0 <= level <= 4:
            return False
        with self._lock:
            self._aggressiveness = level
        return True

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Suppress background noise from the audio chunk.

        TODO: implement RNNoise / DeepFilterNet inference.
        Currently passes audio through unchanged.
        """
        return chunk

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "aggressiveness": self._aggressiveness,
        }


noise_suppressor = NoiseSuppressor()