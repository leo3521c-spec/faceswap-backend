"""
Pitch Shifter voice effect plugin.

Simple, dependency-free pitch shifting via resampling. Each semitone
shifts the pitch by multiplying the sample rate ratio by 2^(n/12).

For higher quality, a future plugin could wrap the existing
voice_converter.py (which supports RVC models) — just drop a new
file in plugins/voice_effects/ and it auto-registers.
"""
import numpy as np

from plugins.base import VoiceEffectPlugin


class PitchShifterPlugin(VoiceEffectPlugin):
    name = "pitch_shifter"
    display_name = "Pitch Shifter"
    version = "1.0.0"
    description = "Real-time pitch shifting via linear-interpolation resampling (-12 to +12 semitones)"
    author = "FaceSwap AI"

    def initialize(self) -> bool:
        self._pitch_shift: int = self.config.get("pitch_shift", 0)
        self._initialized = True
        return True

    def process_audio(self, pcm_data: bytes, sample_rate: int, channels: int) -> bytes:
        if self._pitch_shift == 0:
            return pcm_data

        audio = np.frombuffer(pcm_data, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)

        factor = 2.0 ** (self._pitch_shift / 12.0)
        new_length = max(1, int(len(audio) / factor))

        if channels > 1:
            shifted = np.zeros((new_length, channels), dtype=np.int16)
            for ch in range(channels):
                shifted[:, ch] = np.interp(
                    np.linspace(0, len(audio) - 1, new_length),
                    np.arange(len(audio)),
                    audio[:, ch],
                ).astype(np.int16)
        else:
            shifted = np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio,
            ).astype(np.int16)

        return shifted.tobytes()

    def set_parameter(self, key: str, value) -> bool:
        if key == "pitch_shift":
            try:
                val = int(value)
                self._pitch_shift = max(-12, min(12, val))
                return True
            except (ValueError, TypeError):
                return False
        return False

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "pitch_shift": self._pitch_shift,
            "range": "-12 to +12 semitones",
        }


def create(settings=None):
    config = {}
    if settings and hasattr(settings, "voice_pitch_shift"):
        config["pitch_shift"] = settings.voice_pitch_shift
    return PitchShifterPlugin(config=config)