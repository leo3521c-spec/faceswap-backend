"""
Voice processing chain — orchestrates the full audio pipeline.

Chain order (each stage is skip-able via its enabled flag):
  1. Echo Cancellation  — removes played-back audio from mic input
  2. Noise Suppression  — removes background noise
  3. Voice Conversion   — changes pitch + timbre
  4. Mute               — outputs silence if muted

Mirrors face_processor.process_frame() but for audio chunks.
Runs on the processing thread; must complete within chunk duration
(e.g. < 20 ms for a 20 ms chunk) to maintain real-time.
"""
from __future__ import annotations

import time

from services.audio_chunk import AudioChunk, AudioResult
from services.echo_canceller import echo_canceller
from services.noise_suppressor import noise_suppressor
from services.voice_converter import voice_converter
from services.mute_manager import mute_manager


def process_audio(
    pcm_data: bytes,
    sample_rate: int,
    channels: int,
    chunk_id: int,
) -> AudioResult:
    t_start = time.perf_counter()

    chunk = AudioChunk(
        chunk_id=chunk_id,
        pcm_data=pcm_data,
        sample_rate=sample_rate,
        channels=channels,
    )

    # 1 - Echo cancellation
    chunk = echo_canceller.process(chunk)

    # 2 - Noise suppression
    chunk = noise_suppressor.process(chunk)

    # 3 - Voice conversion
    chunk = voice_converter.process(chunk)

    # 4 - Mute (replaces with silence if muted)
    chunk = mute_manager.process(chunk)

    processing_ms = (time.perf_counter() - t_start) * 1000

    return AudioResult(
        pcm_data=chunk.pcm_data,
        chunk_id=chunk_id,
        processing_time_ms=processing_ms,
        sample_rate=sample_rate,
        channels=channels,
        muted=mute_manager.is_muted(),
        echo_cancelled=echo_canceller.enabled,
        noise_suppressed=noise_suppressor.enabled,
        voice_converted=voice_converter.enabled,
        pitch_shift=voice_converter.pitch_shift,
        model_name=voice_converter.model_name,
    )