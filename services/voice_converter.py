"""
Real-time voice conversion for the voice pipeline.

Architecture (to be implemented):
  • RVC (Retrieval-based Voice Conversion) or so-VITS-SVC
  • GPU-accelerated via ONNX Runtime (shares GPU manager)
  • Pitch shifting (+/-12 semitones) + timbre transfer
  • Voice model presets loadable at runtime
  • Target latency: < 30 ms per 20 ms chunk

Processing chain (when implemented):
  1. Extract F0 (fundamental frequency) via pyin/CREPE
  2. Shift F0 by requested semitones
  3. Extract content features (encoder)
  4. Synthesize with target voice decoder
  5. Crossfade with previous chunk to avoid edge artifacts
"""
from __future__ import annotations

import threading

from utils.logger import setup_logger
from services.audio_chunk import AudioChunk

logger = setup_logger("voice_converter")


class VoiceConverter:
    def __init__(
        self,
        enabled: bool = False,
        model_path: str = "",
        pitch_shift: int = 0,
    ) -> None:
        self._enabled = enabled
        self._model_path = model_path
        self._model_name = ""
        self._pitch_shift = pitch_shift
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def pitch_shift(self) -> int:
        return self._pitch_shift

    @property
    def model_name(self) -> str:
        return self._model_name

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        logger.info("Voice conversion: %s", "ON" if enabled else "OFF")

    def set_pitch(self, semitones: int) -> bool:
        if not -12 <= semitones <= 12:
            return False
        with self._lock:
            self._pitch_shift = semitones
        return True

    def load_model(self, model_path: str) -> bool:
        """Load a voice conversion model at runtime.

        TODO: implement model loading + ONNX session creation.
        """
        with self._lock:
            self._model_path = model_path
        logger.info("Model load requested: %s", model_path)
        return True

    def process(self, chunk: AudioChunk) -> AudioChunk:
        """Convert voice in the audio chunk.

        TODO: implement RVC/so-VITS-SVC inference pipeline.
        Currently passes audio through unchanged.
        """
        return chunk

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "model_path": self._model_path,
            "model_name": self._model_name,
            "pitch_shift": self._pitch_shift,
            "model_loaded": bool(self._model_path),
        }


voice_converter = VoiceConverter()