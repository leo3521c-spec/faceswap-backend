"""
Abstract base class for platform integrations.

This is the plugin contract — any new platform (Zoom, Discord, Slack,
Teams, etc.) implements this interface and registers with the
PlatformManager. The AI engine never imports platform-specific code;
it only processes PCM bytes via the voice pipeline.

Audio flow:
    Platform SDK → adapter._on_incoming_audio() → manager.feed()
    → VoicePipeline → manager._route_result() → adapter.send_audio()
    → Platform SDK

Adding a new platform:
    1. class MyAdapter(PlatformAdapter): implement 5 abstract methods
    2. platform_manager.register(MyAdapter())
    3. Done — no changes to voice_processor, audio_pipeline, or main.py logic
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class PlatformAdapter(ABC):
    """Plugin interface — all platform integrations implement this."""

    platform: str = "base"
    display_name: str = "Base Platform"

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._connected = False
        self._streaming = False
        self._audio_handler: Optional[Callable] = None

    # ── Lifecycle ──────────────────────────────────────────

    @abstractmethod
    def connect(self, **kwargs) -> dict:
        """Authenticate / handshake with the platform."""
        ...

    @abstractmethod
    def disconnect(self) -> dict:
        """Cleanly disconnect from the platform."""
        ...

    @abstractmethod
    def start_stream(self) -> dict:
        """Begin capturing audio from and sending audio to the platform."""
        ...

    @abstractmethod
    def stop_stream(self) -> dict:
        """Stop audio I/O but keep the connection alive."""
        ...

    # ── Audio I/O ──────────────────────────────────────────

    @abstractmethod
    def send_audio(self, pcm_data: bytes) -> None:
        """Send processed PCM back to the platform (virtual mic / voice)."""
        ...

    # ── Status ─────────────────────────────────────────────

    @abstractmethod
    def get_status(self) -> dict:
        """Return platform-specific status + connection state."""
        ...

    # ── Internal: audio routing (set by PlatformManager) ───

    def set_audio_handler(self, handler: Callable) -> None:
        """Register the callback that feeds incoming audio to the pipeline."""
        self._audio_handler = handler

    def _on_incoming_audio(
        self, pcm_data: bytes, sample_rate: int, channels: int
    ) -> None:
        """Called by the platform SDK when audio arrives.

        Subclasses call this from their SDK's audio callback.
        """
        if self._audio_handler and self._streaming:
            self._audio_handler(self, pcm_data, sample_rate, channels)

    # ── Properties ─────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def streaming(self) -> bool:
        return self._streaming