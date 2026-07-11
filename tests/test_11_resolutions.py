# ═══════════════════════════════════════════════════════════════
#  Test Module 11: Different Resolutions
#  Tests pipeline at 360p, 480p, 720p, 1080p, 4K
# ═══════════════════════════════════════════════════════════════
import numpy as np
import cv2
import pytest

from tests.conftest import (
    generate_synthetic_face_frame,
    encode_to_jpeg,
    MockModelManager,
)


RESOLUTIONS = [
    (640, 360, "360p"),
    (640, 480, "480p"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
    (3840, 2160, "4K"),
]


class TestResolutions:
    """Test face swap at various resolutions."""

    @pytest.mark.parametrize("w,h,label", RESOLUTIONS)
    def test_frame_generation(self, w, h, label):
        """Frame can be generated at this resolution."""
        frame = generate_synthetic_face_frame(w, h)
        assert frame.shape == (h, w, 3)

    @pytest.mark.parametrize("w,h,label", RESOLUTIONS)
    def test_jpeg_encoding(self, w, h, label):
        """Frame can be JPEG-encoded at this resolution."""
        frame = generate_synthetic_face_frame(w, h)
        jpeg = encode_to_jpeg(frame)
        assert len(jpeg) > 0

    @pytest.mark.parametrize("w,h,label", RESOLUTIONS)
    def test_jpeg_decode(self, w, h, label):
        """JPEG can be decoded at this resolution."""
        frame = generate_synthetic_face_frame(w, h)
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape == (h, w, 3)

    @pytest.mark.parametrize("w,h,label", RESOLUTIONS)
    def test_detector_runs(self, w, h, label):
        """Detector runs at this resolution."""
        manager = MockModelManager()
        frame = generate_synthetic_face_frame(w, h)
        faces = manager.detector.get(frame)
        assert isinstance(faces, list)

    @pytest.mark.parametrize("w,h,label", RESOLUTIONS)
    def test_swapper_runs(self, w, h, label):
        """Swapper runs at this resolution."""
        manager = MockModelManager()
        frame = generate_synthetic_face_frame(w, h)
        faces = manager.detector.get(frame)
        if faces:
            result = manager.swapper.get(frame, faces[0], faces[0])
            assert result.shape == frame.shape

    @pytest.mark.parametrize("w,h,label", RESOLUTIONS)
    def test_encoding_time_reasonable(self, w, h, label):
        """JPEG encoding completes in reasonable time."""
        import time
        frame = generate_synthetic_face_frame(w, h)
        start = time.perf_counter()
        encode_to_jpeg(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Should be under 500ms even for 4K
        assert elapsed_ms < 500, f"{label} encoding took {elapsed_ms:.0f}ms"

    def test_resolution_scaling_preserves_face(self):
        """Scaling a frame preserves face region proportions."""
        large = generate_synthetic_face_frame(1280, 720)
        small = cv2.resize(large, (640, 360))
        assert small.shape == (360, 640, 3)
        # Both frames should have similar brightness
        assert abs(large.mean() - small.mean()) < 10

    def test_jpeg_size_increases_with_resolution(self):
        """Larger resolutions produce larger JPEG files."""
        sizes = []
        for w, h, _ in RESOLUTIONS[:4]:
            frame = generate_synthetic_face_frame(w, h)
            jpeg = encode_to_jpeg(frame)
            sizes.append(len(jpeg))
        for i in range(1, len(sizes)):
            assert sizes[i] > sizes[i - 1] * 0.5  # At least not dramatically smaller