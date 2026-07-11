# ═══════════════════════════════════════════════════════════════
#  Test Module 9: Glasses
#  Tests pipeline with faces wearing glasses
# ═══════════════════════════════════════════════════════════════
import numpy as np
import cv2
import pytest

from tests.conftest import (
    generate_glasses_frame,
    generate_synthetic_face_frame,
    encode_to_jpeg,
    MockModelManager,
)


class TestGlasses:
    """Test face swap with glasses-wearing faces."""

    def test_glasses_frame_generation(self):
        """Glasses frame can be generated."""
        frame = generate_glasses_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_glasses_vs_no_glasses_differ(self):
        """Glasses frame differs from no-glasses frame."""
        no_glasses = generate_synthetic_face_frame(glasses=False)
        with_glasses = generate_glasses_frame()
        assert not np.array_equal(no_glasses, with_glasses)

    def test_glasses_jpeg_encoding(self):
        """Glasses frame can be JPEG-encoded."""
        frame = generate_glasses_frame()
        jpeg = encode_to_jpeg(frame)
        assert len(jpeg) > 0

    def test_glasses_detector_runs(self):
        """Detector runs without error on glasses face."""
        manager = MockModelManager()
        frame = generate_glasses_frame()
        faces = manager.detector.get(frame)
        assert isinstance(faces, list)

    def test_glasses_swapper_runs(self):
        """Swapper runs without error on glasses face."""
        manager = MockModelManager()
        frame = generate_glasses_frame()
        faces = manager.detector.get(frame)
        if faces:
            result = manager.swapper.get(frame, faces[0], faces[0])
            assert result.shape == frame.shape

    def test_glasses_frame_has_variation(self):
        """Glasses frame has pixel variation (not uniform)."""
        frame = generate_glasses_frame()
        assert frame.std() > 0

    def test_glasses_full_pipeline(self):
        """Full pipeline works with glasses frame."""
        manager = MockModelManager()
        frame = generate_glasses_frame()
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        faces = manager.detector.get(decoded)
        if faces:
            result = manager.swapper.get(decoded, faces[0], faces[0])
            assert result is not None