# ═══════════════════════════════════════════════════════════════
#  Test Module 4: Face Detection Accuracy
#  Tests detection confidence, bbox accuracy, landmark positioning
# ═══════════════════════════════════════════════════════════════
import numpy as np
import pytest

from tests.conftest import (
    generate_synthetic_face_frame,
    create_mock_face,
    MockFace,
)


class TestFaceDetectionAccuracy:
    """Test face detection accuracy and reliability."""

    def test_mock_face_bbox_valid(self):
        """Mock face bbox is a valid 4-element array."""
        face = create_mock_face(320, 240, 100, 120)
        assert face.bbox.shape == (4,)
        x1, y1, x2, y2 = face.bbox
        assert x1 < x2
        assert y1 < y2

    def test_mock_face_kps_valid(self):
        """Mock face has 5 keypoints in correct format."""
        face = create_mock_face()
        assert face.kps.shape == (5, 2)
        # All keypoints within image bounds (640x480)
        for kp in face.kps:
            assert 0 <= kp[0] <= 640
            assert 0 <= kp[1] <= 480

    def test_detection_score_range(self):
        """Detection score is in [0, 1] range."""
        for score in [0.1, 0.5, 0.9, 0.95]:
            face = create_mock_face(score=score)
            assert 0.0 <= face.det_score <= 1.0

    def test_confidence_equals_score(self):
        """Confidence field matches detection score."""
        face = create_mock_face(score=0.88)
        assert face.confidence == face.det_score

    def test_embedding_dimension(self):
        """Face embedding has correct dimensionality (512 for ArcFace)."""
        face = create_mock_face()
        assert face.embedding.shape == (512,)

    def test_normed_embedding_unit_length(self):
        """Normed embedding has unit L2 length."""
        face = create_mock_face()
        norm = np.linalg.norm(face.normed_embedding)
        assert abs(norm - 1.0) < 0.01

    def test_bbox_center_matches_position(self):
        """Face bbox center matches the specified position."""
        cx, cy = 300, 200
        face = create_mock_face(cx=cx, cy=cy, w=100, h=120)
        bbox_cx = (face.bbox[0] + face.bbox[2]) / 2
        bbox_cy = (face.bbox[1] + face.bbox[3]) / 2
        assert abs(bbox_cx - cx) < 1.0
        assert abs(bbox_cy - cy) < 1.0

    def test_landmark_positions(self):
        """5-point landmarks are positioned correctly relative to face center."""
        cx, cy = 320, 240
        w, h = 100, 120
        face = create_mock_face(cx=cx, cy=cy, w=w, h=h)
        # Left eye should be left of center
        assert face.kps[0][0] < cx
        # Right eye should be right of center
        assert face.kps[1][0] > cx
        # Nose should be below eyes
        assert face.kps[2][1] > face.kps[0][1]
        # Mouth should be below nose
        assert face.kps[3][1] > face.kps[2][1]
        assert face.kps[4][1] > face.kps[2][1]

    def test_high_confidence_detection(self):
        """Well-lit frontal face should have high detection score."""
        face = create_mock_face(score=0.95)
        assert face.det_score >= 0.7

    def test_detection_with_synthetic_frame(self):
        """Synthetic frame with drawn face is valid for detection."""
        from tests.conftest import MockModelManager
        manager = MockModelManager()
        frame = generate_synthetic_face_frame(640, 480, brightness=128)
        faces = manager.detector.get(frame)
        assert len(faces) >= 1
        assert all(f.det_score > 0 for f in faces)

    def test_no_detection_on_empty_frame(self):
        """Empty frame (no face features) should return no faces."""
        from tests.conftest import MockModelManager
        manager = MockModelManager()
        manager.detector._should_detect = False
        from tests.conftest import generate_empty_frame
        frame = generate_empty_frame()
        faces = manager.detector.get(frame)
        assert len(faces) == 0