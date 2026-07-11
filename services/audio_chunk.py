"""
Audio data structures for the voice-changing pipeline.

Defines the chunk and result types that flow through:
    Capture → EchoCancel → NoiseSuppress → VoiceConvert → Mute → Output
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AudioChunk:
    """A single chunk of raw PCM audio flowing through the pipeline."""
    chunk_id: int
    pcm_data: bytes           # 16-bit signed LE PCM
    sample_rate: int          # e.g. 24000
    channels: int             # 1 = mono
    put_time: float = field(default_factory=time.perf_counter)

    @property
    def sample_count(self) -> int:
        """Number of audio frames (one per channel group)."""
        if self.channels == 0:
            return 0
        return len(self.pcm_data) // (2 * self.channels)

    @property
    def duration_ms(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return (self.sample_count / self.sample_rate) * 1000


@dataclass
class AudioResult:
    """Processed audio chunk with full pipeline telemetry."""
    pcm_data: bytes
    chunk_id: int
    processing_time_ms: float
    sample_rate: int
    channels: int
    muted: bool
    echo_cancelled: bool
    noise_suppressed: bool
    voice_converted: bool
    pitch_shift: int
    model_name: str

    def to_metadata(self) -> dict:
        return {
            "type": "audio_result",
            "chunk_id": self.chunk_id,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_ms": round(
                (len(self.pcm_data) // 2 // max(self.channels, 1))
                / max(self.sample_rate, 1) * 1000,
                2,
            ),
            "muted": self.muted,
            "echo_cancelled": self.echo_cancelled,
            "noise_suppressed": self.noise_suppressed,
            "voice_converted": self.voice_converted,
            "pitch_shift": self.pitch_shift,
            "model_name": self.model_name,
        }