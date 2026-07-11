# ═══════════════════════════════════════════════════════════════
#  Test Module 1: Webcam Capture
#  Verifies frame generation, encoding, and capture pipeline basics
# ═══════════════════════════════════════════════════════════════
import cv2
import numpy as np
import pytest

from tests.conftest import (
    generate_synthetic_face_frame,
    generate_empty_frame,
    encode_to_jpeg,
    generate_synthetic_face_frame as _gen,
)


class TestWebcamCapture:
    """Test frame capture, encoding, and format validation."""

    def test_frame_dimensions(self):
        """Frames have correct width/height."""
        frame = generate_synthetic_face_frame(640, 480)
        assert frame.shape == (480, 640, 3), f"Expected (480,640,3), got {frame.shape}"

    def test_frame_dtype(self):
        """Frames are uint8 BGR."""
        frame = generate_synthetic_face_frame()
        assert frame.dtype == np.uint8

    def test_frame_not_empty(self):
        """Generated frames contain non-zero pixel data."""
        frame = generate_synthetic_face_frame()
        assert frame.std() > 0, "Frame has zero variance — likely blank"

    def test_jpeg_encoding_roundtrip(self):
        """JPEG encode → decode preserves dimensions."""
        original = generate_synthetic_face_frame(640, 480)
        jpeg_bytes = encode_to_jpeg(original)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape == original.shape

    def test_jpeg_bytes_not_empty(self):
        """JPEG encoding produces non-empty bytes."""
        frame = generate_synthetic_face_frame()
        jpeg = encode_to_jpeg(frame)
        assert len(jpeg) > 0
        assert isinstance(jpeg, bytes)

    def test_multiple_resolutions(self):
        """Frames can be generated at various resolutions."""
        for w, h in [(320, 240), (640, 480), (1280, 720), (1920, 1080)]:
            frame = generate_synthetic_face_frame(w, h)
            assert frame.shape == (h, w, 3)

    def test_capture_consistency(self):
        """Two captures at same params produce same dimensions."""
        f1 = generate_synthetic_face_frame(640, 480)
        f2 = generate_synthetic_face_frame(640, 480)
        assert f1.shape == f2.shape

    def test_brightness_control(self):
        """Brightness parameter affects pixel values."""
        dark = generate_synthetic_face_frame(brightness=10)
        bright = generate_synthetic_face_frame(brightness=200)
        assert dark.mean() < bright.mean()

    def test_empty_frame_generation(self):
        """Empty frames (no face) can be generated."""
        frame = generate_empty_frame(320, 240)
        assert frame.shape == (240, 320, 3)
        # Uniform brightness
        assert frame.std() == 0 or frame.std() < 1