# ═══════════════════════════════════════════════════════════════
#  Test Module 10: Beard
#  Tests pipeline with faces having facial hair
# ═══════════════════════════════════════════════════════════════
import numpy as np
import cv2
import pytest

from tests.conftest import (
    generate_beard_frame,
    generate_synthetic_face_frame,
    encode_to_jpeg,
    MockModelManager,
)


class TestBeard:
    """Test face swap with bearded faces."""

    def test_beard_frame_generation(self):
        """Beard frame can be generated."""
        frame = generate_beard_frame()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_beard_vs_no_beard_differ(self):
        """Beard frame differs from clean-shaven frame."""
        clean = generate_synthetic_face_frame(beard=False)
        bearded = generate_beard_frame()
        assert not np.array_equal(clean, bearded)

    def test_beard_jpeg_encoding(self):
        """Beard frame can be JPEG-encoded."""
        frame = generate_beard_frame()
        jpeg = encode_to_jpeg(frame)
        assert len(jpeg) > 0

    def test_beard_detector_runs(self):
        """Detector runs without error on bearded face."""
        manager = MockModelManager()
        frame = generate_beard_frame()
        faces = manager.detector.get(frame)
        assert isinstance(faces, list)

    def test_beard_swapper_runs(self):
        """Swapper runs without error on bearded face."""
        manager = MockModelManager()
        frame = generate_beard_frame()
        faces = manager.detector.get(frame)
        if faces:
            result = manager.swapper.get(frame, faces[0], faces[0])
            assert result.shape == frame.shape

    def test_beard_frame_has_variation(self):
        """Beard frame has pixel variation."""
        frame = generate_beard_frame()
        assert frame.std() > 0

    def test_beard_full_pipeline(self):
        """Full pipeline works with beard frame."""
        manager = MockModelManager()
        frame = generate_beard_frame()
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        faces = manager.detector.get(decoded)
        if faces:
            result = manager.swapper.get(decoded, faces[0], faces[0])
            assert result is not None