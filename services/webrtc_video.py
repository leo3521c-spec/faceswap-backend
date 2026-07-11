"""
WebRTC video transport for real-time face swap.

Replaces the JPEG-over-WebSocket transport with a peer-to-peer
video stream when enabled.

Architecture
────────────
  • Browser sends webcam video track via WebRTC (RTP/SRTP)
  • Server receives track (aiortc RemoteStreamTrack)
  • Each frame: VideoFrame → BGR numpy → JPEG → face swap pipeline → JPEG → BGR → VideoFrame
  • Server sends processed frames back via outgoing VideoStreamTrack
  • Signaling: HTTP POST /webrtc/offer (SDP offer → SDP answer)
  • ICE/TURN server support via RTCConfiguration
  • Automatic cleanup on connection state change

Library: aiortc (https://github.com/aiortc/aiortc)
"""
from __future__ import annotations

import asyncio
import cv2
import numpy as np
from fractions import Fraction
from typing import Optional, Set

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
)
from av import VideoFrame

from utils.logger import setup_logger
from services.metrics import metrics
from services.model_manager import model_manager
from services.face_processor import process_frame, extract_source_face
from services.face_tracker import face_tracker
from services.gpu_manager import gpu_manager

logger = setup_logger("webrtc_video")

TIME_BASE = Fraction(1, 90000)  # 90 kHz RTP clock


class FaceSwapVideoTrack(MediaStreamTrack):
    """
    Outgoing video track that delivers face-swapped frames.

    Reads frames from the incoming (browser webcam) track, runs each
    through the face swap pipeline, and delivers the processed result.
    """

    kind = "video"

    def __init__(self, incoming_track: MediaStreamTrack, source_face):
        super().__init__()
        self.incoming = incoming_track
        self.source_face = source_face
        self._frame_count = 0
        self._stopped = False

    async def recv(self) -> VideoFrame:
        if self._stopped:
            raise RuntimeError("Track stopped")

        frame = await self.incoming.recv()

        try:
            # VideoFrame → BGR numpy array
            img = frame.to_ndarray(format="bgr24")

            # Encode to JPEG for the face swap pipeline
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return frame

            # Run the full face swap pipeline (in thread to avoid blocking)
            result = await asyncio.to_thread(
                process_frame, buf.tobytes(), self.source_face
            )

            # Decode swapped frame back to numpy
            swapped = cv2.imdecode(
                np.frombuffer(result.jpeg_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if swapped is None:
                return frame

            # Record metrics
            metrics.record_frame(
                result.inference_time_ms,
                face_count=result.face_count,
                confidence=result.detection_confidence,
            )
            gpu_manager.record_inference(result.inference_time_ms)

            self._frame_count += 1

            # BGR → RGB → VideoFrame
            rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
            new_frame = VideoFrame.from_ndarray(rgb, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base or TIME_BASE
            return new_frame

        except Exception as exc:
            logger.warning("Frame processing error: %s", exc)
            return frame

    def stop(self):
        self._stopped = True
        super().stop()


class WebRTCVideoManager:
    """
    Manages WebRTC peer connections for video face swap.

    Each client gets its own RTCPeerConnection. Connections are
    automatically cleaned up when the browser disconnects.
    """

    def __init__(self):
        self._pcs: Set[RTCPeerConnection] = set()
        self._ice_servers: list = []

    def set_ice_servers(self, ice_servers: list) -> None:
        self._ice_servers = ice_servers or []

    async def handle_offer(
        self,
        sdp: str,
        source_face_b64: str,
        ice_servers: Optional[list] = None,
    ) -> dict:
        """
        Process an SDP offer from the browser and return an answer.

        Args:
            sdp: Browser's SDP offer string.
            source_face_b64: Source face image as base64-encoded JPEG.
            ice_servers: Optional ICE/TURN server config from client.

        Returns:
            {"sdp": answer_sdp, "type": "answer"}
        """
        import base64

        servers = ice_servers if ice_servers else self._ice_servers
        rtc_config = RTCConfiguration(
            iceServers=[self._build_ice_server(s) for s in servers]
        )

        pc = RTCPeerConnection(rtc_config)
        self._pcs.add(pc)
        logger.info("Created peer connection (%d total)", len(self._pcs))

        # Decode source face
        source_face = None
        try:
            source_bytes = base64.b64decode(source_face_b64)
            source_face = await asyncio.to_thread(
                extract_source_face, source_bytes
            )
        except Exception as exc:
            logger.error("Failed to extract source face: %s", exc)

        if source_face is None:
            await pc.close()
            self._pcs.discard(pc)
            return {"error": "No face detected in source image"}, 400

        # Handle incoming tracks from browser
        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                logger.info("Received video track from browser")
                face_tracker.reset()
                outgoing = FaceSwapVideoTrack(track, source_face)
                pc.addTrack(outgoing)
                logger.info("Added face swap output track")

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            logger.info("Peer connection state: %s", state)
            if state in ("failed", "closed"):
                await pc.close()
                self._pcs.discard(pc)
                logger.info(
                    "Peer connection closed (%d remaining)", len(self._pcs)
                )

        # Set remote description (browser's offer)
        offer = RTCSessionDescription(sdp=sdp, type="offer")
        await pc.setRemoteDescription(offer)

        # Create and set local description (answer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    def _build_ice_server(self, config: dict) -> RTCIceServer:
        """Build an RTCIceServer from a config dict."""
        kwargs = {"urls": config["urls"]}
        if "username" in config:
            kwargs["username"] = config["username"]
        if "credential" in config:
            kwargs["credential"] = config["credential"]
        return RTCIceServer(**kwargs)

    async def cleanup(self) -> None:
        """Close all peer connections (called on shutdown)."""
        for pc in list(self._pcs):
            try:
                await pc.close()
            except Exception:
                pass
        self._pcs.clear()
        logger.info("All peer connections closed")

    def get_status(self) -> dict:
        return {
            "active_connections": len(self._pcs),
            "ice_servers": self._ice_servers,
        }


# Singleton
webrtc_video_manager = WebRTCVideoManager()