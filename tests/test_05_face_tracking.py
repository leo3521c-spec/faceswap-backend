# ═══════════════════════════════════════════════════════════════
#  Test Module 5: Face Tracking
#  Tests IoU, embedding similarity, track ID persistence, recovery
# ═══════════════════════════════════════════════════════════════
import numpy as np
import pytest

from services.face_tracker import (
    TrackedFace,
    FaceTracker,
    _iou,
    _embedding_sim,
    _kps_to_bbox,
    _estimate_head_pose,
    _rotation_to_euler,
)
from tests.conftest import create_mock_face, generate_synthetic_face_frame


class TestGeometryHelpers:
    """Test geometry utility functions."""

    def test_iou_identical_boxes(self):
        """IoU of identical boxes = 1.0."""
        box = np.array([10, 10, 50, 50], dtype=np.float32)
        assert _iou(box, box) == 1.0

    def test_iou_no_overlap(self):
        """IoU of non-overlapping boxes = 0.0."""
        a = np.array([0, 0, 10, 10], dtype=np.float32)
        b = np.array([100, 100, 110, 110], dtype=np.float32)
        assert _iou(a, b) == 0.0

    def test_iou_partial_overlap(self):
        """IoU of partially overlapping boxes is between 0 and 1."""
        a = np.array([0, 0, 20, 20], dtype=np.float32)
        b = np.array([10, 10, 30, 30], dtype=np.float32)
        iou = _iou(a, b)
        assert 0.0 < iou < 1.0

    def test_embedding_sim_identical(self):
        """Cosine similarity of identical embeddings = 1.0."""
        emb = np.random.randn(512).astype(np.float32)
        sim = _embedding_sim(emb, emb)
        assert abs(sim - 1.0) < 0.001

    def test_embedding_sim_orthogonal(self):
        """Cosine similarity of orthogonal embeddings ≈ 0.0."""
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        sim = _embedding_sim(a, b)
        assert abs(sim) < 0.01

    def test_embedding_sim_none(self):
        """None embeddings return 0.0."""
        assert _embedding_sim(None, np.array([1, 2, 3])) == 0.0
        assert _embedding_sim(np.array([1, 2, 3]), None) == 0.0

    def test_kps_to_bbox(self):
        """kps_to_bbox produces a valid 4-element bbox."""
        kps = np.array([
            [100, 100], [150, 100], [125, 120], [110, 140], [140, 140],
        ], dtype=np.float32)
        bbox = _kps_to_bbox(kps)
        assert bbox.shape == (4,)
        assert bbox[0] < bbox[2]
        assert bbox[1] < bbox[3]

    def test_head_pose_estimation(self):
        """Head pose estimation returns yaw/pitch/roll."""
        kps = np.array([
            [100, 100], [150, 100], [125, 130], [110, 145], [140, 145],
        ], dtype=np.float32)
        pose = _estimate_head_pose(kps, 480, 640)
        assert "yaw" in pose
        assert "pitch" in pose
        assert "roll" in pose

    def test_rotation_to_euler_identity(self):
        """Identity rotation gives (0, 0, 0) euler angles."""
        rmat = np.eye(3, dtype=np.float64)
        angles = _rotation_to_euler(rmat)
        assert abs(angles[0]) < 0.1  # pitch
        assert abs(angles[1]) < 0.1  # yaw
        assert abs(angles[2]) < 0.1  # roll


class TestTrackedFace:
    """Test the TrackedFace class."""

    def test_initialization(self):
        """TrackedFace initializes with correct fields from detection."""
        face = create_mock_face(score=0.9)
        tracked = TrackedFace(track_id=1, face=face)
        assert tracked.track_id == 1
        assert tracked.confidence == 0.9
        assert tracked.miss_count == 0
        assert tracked.frames_tracked == 1

    def test_update_from_detection(self):
        """update_from_detection refreshes all fields."""
        face = create_mock_face(score=0.8)
        tracked = TrackedFace(track_id=1, face=face)
        new_face = create_mock_face(cx=350, cy=250, score=0.95)
        tracked.update_from_detection(new_face)
        assert tracked.confidence == 0.95
        assert tracked.frames_tracked == 2
        assert tracked.miss_count == 0

    def test_update_from_tracking(self):
        """update_from_tracking decays confidence."""
        face = create_mock_face(score=1.0)
        tracked = TrackedFace(track_id=1, face=face)
        original_conf = tracked.confidence
        new_kps = face.kps + np.array([5, 5], dtype=np.float32)
        tracked.update_from_tracking(new_kps)
        assert tracked.confidence < original_conf
        assert tracked.miss_count == 0
        assert tracked.frames_tracked == 2

    def test_confidence_decay_floor(self):
        """Confidence doesn't decay below 0.1."""
        face = create_mock_face(score=0.2)
        tracked = TrackedFace(track_id=1, face=face)
        for _ in range(50):
            tracked.update_from_tracking(face.kps)
        assert tracked.confidence >= 0.1

    def test_to_dict(self):
        """to_dict returns all required fields."""
        face = create_mock_face()
        tracked = TrackedFace(track_id=5, face=face)
        d = tracked.to_dict()
        assert d["track_id"] == 5
        assert "confidence" in d
        assert "miss_count" in d
        assert "frames_tracked" in d
        assert "head_pose" in d
        assert "bbox" in d


class TestFaceTracker:
    """Test the FaceTracker orchestrator."""

    def test_initialization(self):
        """Tracker initializes in idle state."""
        tracker = FaceTracker()
        assert tracker._state == "idle"
        assert len(tracker._tracks) == 0

    def test_reset(self):
        """Reset clears all tracking state."""
        tracker = FaceTracker()
        tracker._tracks = ["fake"]
        tracker._frame_count = 100
        tracker._next_id = 5
        tracker.reset()
        assert len(tracker._tracks) == 0
        assert tracker._frame_count == 0
        assert tracker._next_id == 0
        assert tracker._state == "idle"

    def test_get_metrics(self):
        """get_metrics returns all required fields."""
        tracker = FaceTracker()
        m = tracker.get_metrics()
        assert "confidence" in m
        assert "state" in m
        assert "active_tracks" in m
        assert "fps" in m
        assert "total_tracks_created" in m
        assert "total_detections" in m