"""
Telegram integration via Bot API + MTProto.

Two modes:
  1. Voice messages — bot receives voice notes, processes, sends back
  2. Voice calls (MTProto) — real-time voice call via Telethon/Pyrogram

This adapter focuses on voice calls for real-time changing.
Voice message mode can be added as a sub-strategy later.

Library: https://docs.python-telegram-bot.org/ (Bot API)
         https://docs.telethon.dev/ (MTProto voice calls)
"""
from __future__ import annotations

from services.platform_base import PlatformAdapter


class TelegramAdapter(PlatformAdapter):
    platform = "telegram"
    display_name = "Telegram"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._bot_token: str = ""
        self._api_id: str = ""
        self._api_hash: str = ""
        self._mode: str = "voice_call"  # voice_call | voice_message

    def connect(self, **kwargs) -> dict:
        """Authenticate with Telegram Bot API + MTProto.

        kwargs: bot_token, api_id, api_hash
        TODO: init Bot + Telethon client, start polling.
        """
        self._bot_token = kwargs.get(
            "bot_token", self._config.get("bot_token", "")
        )
        self._api_id = kwargs.get("api_id", "")
        self._api_hash = kwargs.get("api_hash", "")
        self._connected = True
        return self.get_status()

    def disconnect(self) -> dict:
        self._connected = False
        self._streaming = False
        return self.get_status()

    def start_stream(self) -> dict:
        """Start listening for incoming voice calls.

        TODO: register incoming call handler that, on answer,
        calls self._on_incoming_audio(pcm, sr, ch) from the call's
        audio stream.
        """
        self._streaming = True
        return self.get_status()

    def stop_stream(self) -> dict:
        self._streaming = False
        return self.get_status()

    def send_audio(self, pcm_data: bytes) -> None:
        """Send processed PCM back through the voice call.

        TODO: write to the MTProto call's outgoing audio stream.
        """
        pass

    def get_status(self) -> dict:
        return {
            "platform": self.platform,
            "connected": self._connected,
            "streaming": self._streaming,
            "mode": self._mode,
            "bot_token_configured": bool(self._bot_token),
            "api_id": self._api_id,
        }