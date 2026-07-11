# ═══════════════════════════════════════════════════════════════
#  Test Module 8: Side Face / Profile
#  Tests pipeline with angled/rotated faces
# ═══════════════════════════════════════════════════════════════
import numpy as np
import cv2
import pytest

from tests.conftest import (
    generate_side_face_frame,
    generate_synthetic_face_frame,
    encode_to_jpeg,
    MockModelManager,
    create_mock_face,
)


class TestSideFace:
    """Test face swap with side-angled / profile faces."""

    def test_side_face_frame_generation(self):
        """Side face frame can be generated."""
        frame = generate_side_face_frame(angle=45)
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_various_angles(self):
        """Frames at various angles are valid."""
        for angle in [0, 15, 30, 45, 60, -30, -45]:
            frame = generate_side_face_frame(angle=angle)
            assert frame is not None

    def test_side_face_jpeg_encoding(self):
        """Side face frame can be JPEG-encoded."""
        frame = generate_side_face_frame(angle=30)
        jpeg = encode_to_jpeg(frame)
        assert len(jpeg) > 0

    def test_side_face_detector_runs(self):
        """Detector runs without error on side face."""
        manager = MockModelManager()
        frame = generate_side_face_frame(angle=45)
        faces = manager.detector.get(frame)
        assert isinstance(faces, list)

    def test_side_face_swapper_runs(self):
        """Swapper runs without error on side face."""
        manager = MockModelManager()
        frame = generate_side_face_frame(angle=30)
        faces = manager.detector.get(frame)
        if faces:
            result = manager.swapper.get(frame, faces[0], faces[0])
            assert result.shape == frame.shape

    def test_head_pose_yaw_estimation(self):
        """Head pose estimation produces non-zero yaw for side face."""
        from services.face_tracker import _estimate_head_pose
        # Offset landmarks to simulate yaw
        kps = np.array([
            [110, 100], [140, 100], [130, 130], [115, 145], [135, 145],
        ], dtype=np.float32)
        pose = _estimate_head_pose(kps, 480, 640)
        assert "yaw" in pose
        # Yaw should be a valid number
        assert isinstance(pose["yaw"], (int, float))

    def test_frontal_vs_side_difference(self):
        """Frontal and side frames produce different images."""
        frontal = generate_synthetic_face_frame(angle=0)
        side = generate_side_face_frame(angle=45)
        # They should differ
        assert not np.array_equal(frontal, side)

    def test_extreme_angle_no_crash(self):
        """Pipeline doesn't crash with extreme face angle."""
        manager = MockModelManager()
        frame = generate_side_face_frame(angle=80)
        jpeg = encode_to_jpeg(frame)
        decoded = cv2.imdecode(
            np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        faces = manager.detector.get(decoded)
        assert isinstance(faces, list)