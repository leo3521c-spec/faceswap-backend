"""
Plugin manager — registers platform adapters and routes audio between
the voice pipeline and active platforms.

The manager owns the VoicePipeline lifecycle and the result routing
loop. Adapters are completely decoupled from the AI engine — they
only know about PCM bytes and the PlatformAdapter contract.

Adding a new platform requires zero changes to:
    • voice_processor.py (the AI chain)
    • audio_pipeline.py (the 3-thread pipeline)
    • Any existing adapter
"""
from __future__ import annotations

import asyncio
import threading
from typing import Optional

from utils.logger import setup_logger
from services.audio_pipeline import (
    VoicePipeline,
    set_active_voice_pipeline,
    get_active_voice_pipeline_metrics,
)
from services.platform_base import PlatformAdapter

logger = setup_logger("platform_manager")


class PlatformManager:
    def __init__(self) -> None:
        self._adapters: dict[str, PlatformAdapter] = {}
        self._pipeline: Optional[VoicePipeline] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

    # ── Registration ───────────────────────────────────────

    def register(self, adapter: PlatformAdapter) -> None:
        """Register a platform adapter. Called once at startup."""
        self._adapters[adapter.platform] = adapter
        adapter.set_audio_handler(self._on_platform_audio)
        logger.info("Registered platform adapter: %s", adapter.platform)

    def get_adapter(self, platform: str) -> Optional[PlatformAdapter]:
        return self._adapters.get(platform)

    def list_platforms(self) -> list[dict]:
        return [
            {
                "platform": a.platform,
                "display_name": a.display_name,
                "connected": a.connected,
                "streaming": a.streaming,
                "status": a.get_status(),
            }
            for a in self._adapters.values()
        ]

    # ── Pipeline lifecycle ─────────────────────────────────

    def start_pipeline(
        self,
        sample_rate: int,
        channels: int,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Create and start the voice pipeline + result routing task."""
        self._loop = loop
        self._pipeline = VoicePipeline(
            process_fn=self._process_audio_wrapper,
            sample_rate=sample_rate,
            channels=channels,
            loop=loop,
        )
        self._pipeline.start()
        set_active_voice_pipeline(self._pipeline)
        self._sender_task = loop.create_task(self._result_loop())
        logger.info("Platform audio pipeline started")

    async def stop_pipeline(self) -> None:
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None
        if self._pipeline:
            self._pipeline.stop()
            set_active_voice_pipeline(None)
            self._pipeline = None
        logger.info("Platform audio pipeline stopped")

    # ── Audio routing ──────────────────────────────────────

    def _on_platform_audio(
        self,
        adapter: PlatformAdapter,
        pcm_data: bytes,
        sample_rate: int,
        channels: int,
    ) -> None:
        """Called by any adapter when it receives audio from its platform."""
        if self._pipeline is None:
            return
        self._pipeline.submit_chunk(pcm_data)

    def _process_audio_wrapper(self, pcm, sr, ch, chunk_id):
        from services.voice_processor import process_audio
        return process_audio(pcm, sr, ch, chunk_id)

    async def _result_loop(self) -> None:
        """Read processed results from the pipeline and route to adapters."""
        if not self._pipeline:
            return
        try:
            while True:
                result, put_time = await self._pipeline.get_result()
                for adapter in self._adapters.values():
                    if adapter.streaming:
                        try:
                            adapter.send_audio(result.pcm_data)
                        except Exception as exc:
                            logger.debug(
                                "send_audio error (%s): %s",
                                adapter.platform, exc,
                            )
        except asyncio.CancelledError:
            return

    # ── Status ─────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "platforms": self.list_platforms(),
            "pipeline": get_active_voice_pipeline_metrics(),
            "registered_count": len(self._adapters),
        }


platform_manager = PlatformManager()