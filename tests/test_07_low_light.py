# ═══════════════════════════════════════════════════════════════
#  Test Module 7: Low Light Conditions
#  Tests pipeline behavior with dark/underexposed frames
# ═══════════════════════════════════════════════════════════════
import numpy as np
import cv2
import pytest

from tests.conftest import (
    generate_low_light_frame,
    generate_synthetic_face_frame,
    encode_to_jpeg,
    MockModelManager,
)


class TestLowLight:
    """Test face swap under low-light conditions."""

    def test_low_light_frame_is_dark(self):
        """Low-light frame has low mean brightness."""
        frame = generate_low_light_frame()
        assert frame.mean() < 30, f"Expected dark frame, got mean={frame.mean():.1f}"

    def test_low_light_jpeg_encoding(self):
        """Low-light frame can be JPEG-encoded."""
        frame = generate_low_light_frame()
        jpeg = encode_to_jpeg(frame)
        assert len(jpeg) > 0

    def test_low_light_decode(self):
        """Low-light JPEG can be decoded."""
        frame = generate_low_light_frame()
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape == frame.shape

    def test_low_light_detector_runs(self):
        """Detector runs without error on low-light frame."""
        manager = MockModelManager()
        frame = generate_low_light_frame()
        faces = manager.detector.get(frame)
        # May or may not detect — but shouldn't crash
        assert isinstance(faces, list)

    def test_brightness_gradient(self):
        """Frames at different brightness levels are distinguishable."""
        levels = [10, 50, 100, 150, 200, 250]
        means = []
        for b in levels:
            frame = generate_synthetic_face_frame(brightness=b)
            means.append(frame.mean())
        # Means should be monotonically increasing
        for i in range(1, len(means)):
            assert means[i] > means[i - 1]

    def test_low_light_swapper_runs(self):
        """Swapper runs without error on low-light frame."""
        manager = MockModelManager()
        frame = generate_low_light_frame()
        faces = manager.detector.get(frame)
        if faces:
            result = manager.swapper.get(frame, faces[0], faces[0], paste_back=True)
            assert result.shape == frame.shape

    def test_histogram_stretching(self):
        """Low-light frame can be histogram-stretched (preprocessing)."""
        frame = generate_low_light_frame()
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        assert enhanced.mean() > frame.mean()

    def test_low_light_no_crash(self):
        """Full pipeline (encode → decode → detect → swap) doesn't crash."""
        frame = generate_low_light_frame()
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        manager = MockModelManager()
        faces = manager.detector.get(decoded)
        if faces:
            result = manager.swapper.get(decoded, faces[0], faces[0])
            assert result is not None