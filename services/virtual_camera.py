"""
OBS Virtual Camera output manager.

Uses pyvirtualcam to write processed frames to a virtual camera device
that OBS Studio (or any virtual-camera consumer) can capture as a source.

Features
────────
• Output to OBS Virtual Camera on Windows / Linux / macOS
• Supported resolutions: 720p (1280×720), 1080p (1920×1080)
• Supported frame rates: 24 FPS, 30 FPS
• Enable / disable at runtime — no restart required
• Metrics: output resolution, dropped frames, output FPS

Dependencies
────────────
  pip install pyvirtualcam

  • Windows: OBS Virtual Camera must be installed (comes with OBS Studio)
  • Linux:   v4l2loopback kernel module (sudo modprobe v4l2loopback exclusive_caps=1)
  • macOS:   OBS Virtual Camera (comes with OBS Studio)
"""
from __future__ import annotations

import threading
import time
import cv2
import numpy as np
from collections import deque
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("virtual_camera")

# ── Resolution presets ────────────────────────────────────────

RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}

VALID_FPS = [24, 30]


class VirtualCameraManager:
    """
    Manages the lifecycle and frame output of a virtual camera.

    Thread-safe: frames are submitted from the pipeline's sender task
    (async) and written by an internal writer thread that enforces the
    target FPS via sleep pacing.
    """

    def __init__(self) -> None:
        self._cam = None
        self._lock = threading.Lock()
        self._enabled = False
        self._active = False
        self._resolution: str = "720p"
        self._fps: int = 30
        self._device: str = ""

        # Writer thread
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        # Metrics
        self._frames_sent: int = 0
        self._frames_dropped: int = 0
        self._fps_times: deque = deque(maxlen=120)
        self._target_interval: float = 1.0 / self._fps

    # ── Properties ─────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def active(self) -> bool:
        return self._active

    @property
    def resolution(self) -> str:
        return self._resolution

    @property
    def width(self) -> int:
        return RESOLUTIONS.get(self._resolution, (0, 0))[0]

    @property
    def height(self) -> int:
        return RESOLUTIONS.get(self._resolution, (0, 0))[1]

    @property
    def fps(self) -> int:
        return self._fps

    # ── Lifecycle ──────────────────────────────────────────

    def enable(
        self,
        resolution: Optional[str] = None,
        fps: Optional[int] = None,
    ) -> dict:
        """
        Enable the virtual camera with optional resolution/fps override.
        Opens the pyvirtualcam device and starts the writer thread.
        """
        with self._lock:
            if self._active:
                logger.info("Virtual camera already active — reconfiguring")
                self._stop_internal()

            if resolution is not None:
                if resolution not in RESOLUTIONS:
                    return {"error": f"Invalid resolution: {resolution}"}
                self._resolution = resolution
            if fps is not None:
                if fps not in VALID_FPS:
                    return {"error": f"Invalid FPS: {fps}"}
                self._fps = fps

            self._target_interval = 1.0 / self._fps
            self._enabled = True

            try:
                import pyvirtualcam

                w, h = RESOLUTIONS[self._resolution]
                self._cam = pyvirtualcam.Camera(
                    width=w, height=h, fps=self._fps
                )
                self._device = self._cam.device
                self._active = True

                # Reset metrics
                self._frames_sent = 0
                self._frames_dropped = 0
                self._fps_times.clear()
                self._latest_frame = None

                # Start writer thread
                self._stop_event.clear()
                self._writer_thread = threading.Thread(
                    target=self._writer_loop,
                    name="virtual-camera-writer",
                    daemon=True,
                )
                self._writer_thread.start()

                logger.info(
                    "Virtual camera enabled: %s (%dx%d @ %d fps) on %s",
                    self._resolution,
                    w,
                    h,
                    self._fps,
                    self._device,
                )
                return self.get_status()
            except ImportError:
                logger.error(
                    "pyvirtualcam is not installed — "
                    "install with: pip install pyvirtualcam"
                )
                self._enabled = False
                self._active = False
                return {
                    "error": "pyvirtualcam is not installed. "
                    "Install with: pip install pyvirtualcam"
                }
            except Exception as exc:
                logger.error("Failed to open virtual camera: %s", exc)
                self._enabled = False
                self._active = False
                self._cam = None
                return {"error": f"Failed to open virtual camera: {exc}"}

    def disable(self) -> dict:
        """Disable the virtual camera and release the device."""
        with self._lock:
            self._stop_internal()
            logger.info("Virtual camera disabled")
            return self.get_status()

    def _stop_internal(self) -> None:
        """Internal shutdown — caller must hold the lock."""
        self._enabled = False
        self._active = False
        self._stop_event.set()
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=2.0)
        self._writer_thread = None
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        self._device = ""
        self._latest_frame = None

    # ── Frame submission ───────────────────────────────────

    def submit_frame(self, frame: np.ndarray) -> bool:
        """
        Submit a processed frame for virtual camera output.

        Called from the pipeline sender. Non-blocking: if a frame is
        already pending, it is replaced (latest-wins). The frame is
        resized to the target resolution if needed.
        Returns True if the frame was accepted, False if the camera
        is not active.
        """
        if not self._active or self._cam is None:
            return False

        target_w, target_h = RESOLUTIONS[self._resolution]

        # Resize if the frame doesn't match the target resolution
        h, w = frame.shape[:2]
        if (w, h) != (target_w, target_h):
            frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        # pyvirtualcam expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self._frame_lock:
            if self._latest_frame is not None:
                # Previous frame wasn't consumed yet — count as dropped
                self._frames_dropped += 1
            self._latest_frame = frame_rgb

        return True

    # ── Writer thread ──────────────────────────────────────

    def _writer_loop(self) -> None:
        """
        Writer thread: reads the latest pending frame and sends it to
        the virtual camera at the target FPS. Paces output with sleep
        to avoid exceeding the device's frame rate.
        """
        logger.info("Virtual camera writer thread started")
        while not self._stop_event.is_set():
            t_start = time.perf_counter()

            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is not None and self._cam is not None:
                try:
                    self._cam.send(frame)
                    self._frames_sent += 1
                    self._fps_times.append(time.time())
                except Exception as exc:
                    logger.warning("Virtual camera write error: %s", exc)
                    self._frames_dropped += 1

            # Pace to target FPS
            elapsed = time.perf_counter() - t_start
            sleep_time = self._target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Virtual camera writer thread stopped")

    # ── Metrics ────────────────────────────────────────────

    @property
    def output_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        if elapsed <= 0:
            return 0.0
        return len(self._fps_times) / elapsed

    def get_status(self) -> dict:
        """Return full virtual camera status and metrics."""
        return {
            "enabled": self._enabled,
            "active": self._active,
            "resolution": self._resolution,
            "width": self.width,
            "height": self.height,
            "fps": self._fps,
            "device": self._device,
            "frames_sent": self._frames_sent,
            "frames_dropped": self._frames_dropped,
            "output_fps": round(self.output_fps, 2),
            "available_resolutions": list(RESOLUTIONS.keys()),
            "available_fps": VALID_FPS,
        }


# ── Singleton ─────────────────────────────────────────────────

virtual_camera = VirtualCameraManager()