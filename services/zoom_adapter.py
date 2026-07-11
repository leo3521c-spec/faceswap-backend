"""
Zoom integration via Zoom Meeting SDK.

Architecture:
  • Zoom Meeting SDK (C++ core) with Python bindings
  • AudioRawData callback — receives per-participant 16-bit PCM
  • ZoomAudioRawDataSender — injects processed audio as virtual mic
  • Requires SDK key + secret from Zoom Marketplace

SDK: https://marketplace.zoom.us/docs/sdk/meeting-sdks/
"""
from __future__ import annotations

from services.platform_base import PlatformAdapter


class ZoomAdapter(PlatformAdapter):
    platform = "zoom"
    display_name = "Zoom"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._meeting_id: str = ""
        self._sdk_key: str = ""
        self._participant_count: int = 0

    def connect(self, **kwargs) -> dict:
        """Initialize Zoom SDK and join a meeting.

        kwargs: sdk_key, sdk_secret, meeting_id, meeting_password
        TODO: init SDK, auth, join meeting.
        """
        self._sdk_key = kwargs.get("sdk_key", self._config.get("sdk_key", ""))
        self._meeting_id = kwargs.get("meeting_id", "")
        self._connected = True
        return self.get_status()

    def disconnect(self) -> dict:
        """Leave meeting and release SDK resources."""
        self._connected = False
        self._streaming = False
        return self.get_status()

    def start_stream(self) -> dict:
        """Subscribe to AudioRawData and start the virtual mic sender.

        TODO: register audio callback, create ZoomAudioRawDataSender.
        The callback should call self._on_incoming_audio(pcm, sr, ch).
        """
        self._streaming = True
        return self.get_status()

    def stop_stream(self) -> dict:
        self._streaming = False
        return self.get_status()

    def send_audio(self, pcm_data: bytes) -> None:
        """Send processed PCM as virtual microphone input.

        TODO: feed to ZoomAudioRawDataSender.send().
        """
        pass

    def get_status(self) -> dict:
        return {
            "platform": self.platform,
            "connected": self._connected,
            "streaming": self._streaming,
            "meeting_id": self._meeting_id,
            "sdk_configured": bool(self._sdk_key),
            "participant_count": self._participant_count,
        }