# ═══════════════════════════════════════════════════════════════
#  Test Module 6: Multiple Face Support
#  Tests detection and handling of 2, 3, and many faces
# ═══════════════════════════════════════════════════════════════
import numpy as np
import pytest

from tests.conftest import (
    generate_multi_face_frame,
    generate_synthetic_face_frame,
    MockModelManager,
    create_mock_face,
    encode_to_jpeg,
)


class TestMultipleFaceSupport:
    """Test multi-face detection and swap."""

    def test_single_face_frame(self):
        """Frame with 1 face is valid."""
        frame = generate_synthetic_face_frame(face_count=1)
        assert frame is not None
        assert frame.shape[2] == 3

    def test_two_face_frame(self):
        """Frame with 2 faces is valid."""
        frame = generate_multi_face_frame(640, 480, count=2)
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_three_face_frame(self):
        """Frame with 3 faces is valid."""
        frame = generate_multi_face_frame(640, 480, count=3)
        assert frame is not None

    def test_many_face_frame(self):
        """Frame with 6 faces is valid."""
        frame = generate_multi_face_frame(1280, 480, count=6)
        assert frame is not None

    def test_detector_returns_multiple_faces(self):
        """Mock detector can return multiple faces."""
        manager = MockModelManager()
        manager.detector._face_count = 3
        frame = generate_synthetic_face_frame()
        faces = manager.detector.get(frame)
        assert len(faces) == 3

    def test_multiple_face_bboxes_non_overlapping(self):
        """Multiple face bboxes don't significantly overlap."""
        manager = MockModelManager()
        manager.detector._face_count = 3
        frame = generate_synthetic_face_frame(640, 480)
        faces = manager.detector.get(frame)

        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                # Check that bboxes are spatially separated
                cx_i = (faces[i].bbox[0] + faces[i].bbox[2]) / 2
                cx_j = (faces[j].bbox[0] + faces[j].bbox[2]) / 2
                assert abs(cx_i - cx_j) > 50, "Faces too close together"

    def test_multiple_face_embeddings_distinct(self):
        """Multiple faces have distinct embeddings."""
        faces = [create_mock_face(cx=100+i*200) for i in range(3)]
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                sim = float(np.dot(
                    faces[i].normed_embedding,
                    faces[j].normed_embedding,
                ))
                assert sim < 0.99, f"Faces {i} and {j} have nearly identical embeddings"

    def test_swapper_handles_multiple_faces(self):
        """Swapper processes each face without error."""
        manager = MockModelManager()
        manager.detector._face_count = 3
        frame = generate_synthetic_face_frame()
        faces = manager.detector.get(frame)
        source = faces[0]

        result = frame.copy()
        for face in faces:
            result = manager.swapper.get(result, face, source, paste_back=True)
        assert result.shape == frame.shape

    def test_multi_face_jpeg_encoding(self):
        """Multi-face frame can be JPEG-encoded."""
        frame = generate_multi_face_frame(count=3)
        jpeg = encode_to_jpeg(frame)
        assert len(jpeg) > 0

    def test_face_count_in_metadata(self):
        """FrameResult reports correct face count for multi-face."""
        from services.face_processor import FrameResult
        result = FrameResult(
            jpeg_bytes=b"",
            inference_time_ms=30.0,
            face_count=3,
            detection_confidence=0.9,
        )
        assert result.to_metadata()["face_count"] == 3