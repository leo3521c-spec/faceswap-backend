"""
Browser WebRTC integration via aiortc.

Architecture:
  • Browser captures mic audio via WebRTC (getUserMedia)
  • Peer connection established with this server (aiortc)
  • Incoming: RemoteAudioTrack receives Opus → decoded PCM
  • Outgoing: ProcessedAudioTrack sends processed PCM → encoded Opus
  • Signaling via the existing /ws/voice WebSocket or HTTP polling

This adapter manages a single peer connection. For multi-user,
each user gets their own adapter instance.

Library: https://github.com/aiortc/aiortc
"""
from __future__ import annotations

from services.platform_base import PlatformAdapter


class WebRTCAdapter(PlatformAdapter):
    platform = "webrtc"
    display_name = "Browser WebRTC"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._peer_id: str = ""
        self._ice_servers: list = []
        self._sample_rate: int = 48000

    def connect(self, **kwargs) -> dict:
        """Create a WebRTC peer connection.

        kwargs: peer_id, ice_servers, sdp_offer
        TODO: create RTCPeerConnection with ICE servers,
        set remote description from SDP offer, create answer.
        """
        self._peer_id = kwargs.get("peer_id", "")
        self._ice_servers = kwargs.get(
            "ice_servers", [{"urls": "stun:stun.l.google.com:19302"}]
        )
        self._connected = True
        return self.get_status()

    def disconnect(self) -> dict:
        """Close the peer connection."""
        self._connected = False
        self._streaming = False
        return self.get_status()

    def start_stream(self) -> dict:
        """Begin receiving audio track and sending processed track.

        TODO: subscribe to RemoteAudioTrack.recv(),
        call self._on_incoming_audio(pcm, sr, ch) for each frame.
        Replace outgoing track with ProcessedAudioTrack.
        """
        self._streaming = True
        return self.get_status()

    def stop_stream(self) -> dict:
        self._streaming = False
        return self.get_status()

    def send_audio(self, pcm_data: bytes) -> None:
        """Feed processed PCM into the outgoing audio track.

        TODO: push to ProcessedAudioTrack (an aiortc MediaStreamTrack).
        """
        pass

    def get_status(self) -> dict:
        return {
            "platform": self.platform,
            "connected": self._connected,
            "streaming": self._streaming,
            "peer_id": self._peer_id,
            "ice_servers": self._ice_servers,
            "sample_rate": self._sample_rate,
        }