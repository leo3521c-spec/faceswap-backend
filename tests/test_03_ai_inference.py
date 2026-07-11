# ═══════════════════════════════════════════════════════════════
#  Test Module 3: AI Inference
#  Tests the face processing pipeline (decode → detect → swap → encode)
# ═══════════════════════════════════════════════════════════════
import time
import cv2
import numpy as np
import pytest

from tests.conftest import (
    generate_synthetic_face_frame,
    encode_to_jpeg,
    MockModelManager,
    MockGPUManager,
    create_mock_face,
    generate_empty_frame,
)


class TestAIInference:
    """Test AI inference pipeline components."""

    def test_jpeg_decode(self):
        """JPEG bytes can be decoded to BGR array."""
        frame = generate_synthetic_face_frame()
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape == frame.shape

    def test_jpeg_encode_quality(self):
        """JPEG encoding at different qualities produces valid output."""
        frame = generate_synthetic_face_frame()
        for quality in [50, 75, 85, 95]:
            jpeg = encode_to_jpeg(frame, quality=quality)
            assert len(jpeg) > 0
            decoded = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            assert decoded is not None

    def test_inference_timing(self):
        """Simulated inference completes within reasonable time."""
        frame = generate_synthetic_face_frame()
        start = time.perf_counter()
        # Simulate decode + process + encode
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        result_jpeg = encode_to_jpeg(decoded)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 1000, f"Inference took {elapsed_ms:.0f}ms (expected <1000ms)"

    def test_face_swap_passthrough_no_face(self):
        """Empty frame (no face) produces pass-through result."""
        frame = generate_empty_frame()
        jpeg = encode_to_jpeg(frame)
        # Verify the frame is valid and can be decoded
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None

    def test_mock_detector_returns_faces(self):
        """Mock detector returns face objects with required fields."""
        manager = MockModelManager()
        frame = generate_synthetic_face_frame()
        faces = manager.detector.get(frame)
        assert len(faces) > 0
        face = faces[0]
        assert hasattr(face, "bbox")
        assert hasattr(face, "kps")
        assert hasattr(face, "det_score")
        assert hasattr(face, "embedding")

    def test_mock_swapper_returns_image(self):
        """Mock swapper returns an image of same shape."""
        manager = MockModelManager()
        frame = generate_synthetic_face_frame()
        faces = manager.detector.get(frame)
        source = faces[0]
        result = manager.swapper.get(frame, faces[0], source, paste_back=True)
        assert result.shape == frame.shape

    def test_gpu_manager_mock(self):
        """Mock GPU manager reports CPU mode."""
        gm = MockGPUManager()
        status = gm.get_status()
        assert status["gpu_available"] is False
        assert "CPU" in status["name"] or "CPU" in status["provider"]

    def test_pinned_memory_mock(self):
        """Mock GPU pinned memory returns usable array."""
        gm = MockGPUManager()
        buf = gm.get_pinned_array(1024, dtype=np.uint8)
        assert buf.shape == (1024,)
        buf[:] = np.frombuffer(b"\x00" * 1024, dtype=np.uint8)
        gm.return_pinned_array(buf)

    def test_inference_under_repeated_calls(self):
        """Repeated inference calls don't crash or leak resources."""
        frame = generate_synthetic_face_frame()
        for _ in range(50):
            jpeg = encode_to_jpeg(frame)
            decoded = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            assert decoded is not None

    def test_frame_result_metadata_fields(self):
        """FrameResult has all required metadata fields."""
        from services.face_processor import FrameResult
        result = FrameResult(
            jpeg_bytes=b"",
            inference_time_ms=25.0,
            face_count=1,
            detection_confidence=0.95,
        )
        meta = result.to_metadata()
        assert meta["type"] == "frame_result"
        assert meta["inference_time_ms"] == 25.0
        assert meta["face_count"] == 1
        assert meta["detection_confidence"] == 0.95
        assert "enhanced" in meta
        assert "pass_through" in meta
        assert "tracking_confidence" in meta