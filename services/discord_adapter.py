"""
Discord integration via discord.py voice client.

Architecture:
  • discord.py VoiceClient connects to a Discord voice channel
  • Incoming: Opus-decoded PCM via AudioSink / voice_receive hook
  • Outgoing: PCMAudio source fed back as bot's audio
  • Bot token + guild ID + channel ID required

Library: https://github.com/Rapptz/discord.py (voice support)
"""
from __future__ import annotations

from services.platform_base import PlatformAdapter


class DiscordAdapter(PlatformAdapter):
    platform = "discord"
    display_name = "Discord"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._bot_token: str = ""
        self._guild_id: str = ""
        self._channel_id: str = ""

    def connect(self, **kwargs) -> dict:
        """Start the Discord bot and connect to a voice channel.

        kwargs: token, guild_id, channel_id
        TODO: create discord.Client, connect to voice channel.
        """
        self._bot_token = kwargs.get("token", self._config.get("token", ""))
        self._guild_id = kwargs.get("guild_id", "")
        self._channel_id = kwargs.get("channel_id", "")
        self._connected = True
        return self.get_status()

    def disconnect(self) -> dict:
        """Leave voice channel and shut down the bot."""
        self._connected = False
        self._streaming = False
        return self.get_status()

    def start_stream(self) -> dict:
        """Begin receiving voice audio and sending processed audio.

        TODO: register voice_receive callback that calls
        self._on_incoming_audio(pcm, 48000, 2).
        Create a PCMAudio source from the processed buffer.
        """
        self._streaming = True
        return self.get_status()

    def stop_stream(self) -> dict:
        self._streaming = False
        return self.get_status()

    def send_audio(self, pcm_data: bytes) -> None:
        """Send processed PCM as the bot's voice output.

        TODO: write to the bot's audio source buffer (48 kHz stereo).
        """
        pass

    def get_status(self) -> dict:
        return {
            "platform": self.platform,
            "connected": self._connected,
            "streaming": self._streaming,
            "guild_id": self._guild_id,
            "channel_id": self._channel_id,
            "token_configured": bool(self._bot_token),
        }