"""
Landmark-based face tracker with persistent IDs, optical-flow tracking,
automatic recovery, head-pose estimation, and multi-face support.

Architecture
────────────
  Frame N   (detection) → InsightFace detect → match to tracks (IoU + embedding)
                          → update matched / create new / expire stale
  Frame N+k (tracking)   → KLT optical flow on 5-point landmarks
                          → update kps/bbox → estimate head pose via solvePnP
  Failure                → force full detection on next frame (auto-recovery)

Exposes: tracking_confidence, tracking_fps, tracking_state
"""
from __future__ import annotations

import time
import cv2
import numpy as np
from collections import deque
from typing import Optional

from config import get_settings
from utils.logger import setup_logger
from services.model_manager import model_manager

logger = setup_logger("face_tracker")

# ── 3D face model for solvePnP head-pose estimation ───────────
# Generic canonical face — proportional values are sufficient;
# solvePnP finds the rotation that best maps these to 2D landmarks.
_3D_FACE_MODEL = np.array(
    [
        [-31.0, -72.0, -26.0],  # left eye
        [31.0, -72.0, -26.0],   # right eye
        [0.0, 10.0, 45.0],      # nose tip
        [-24.0, 48.0, 10.0],    # left mouth corner
        [24.0, 48.0, 10.0],     # right mouth corner
    ],
    dtype=np.float64,
)


# ── Geometry helpers ──────────────────────────────────────────


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _embedding_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Cosine similarity between two embeddings."""
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _kps_to_bbox(kps: np.ndarray) -> np.ndarray:
    """Compute a bounding box from 5-point landmarks."""
    x_min, x_max = float(kps[:, 0].min()), float(kps[:, 0].max())
    y_min, y_max = float(kps[:, 1].min()), float(kps[:, 1].max())
    w = x_max - x_min
    h = y_max - y_min
    # Expand — landmarks only cover eyes/nose/mouth, not forehead/chin
    return np.array(
        [x_min - w * 0.4, y_min - h * 0.6, x_max + w * 0.4, y_max + h * 0.3],
        dtype=np.float32,
    )


def _rotation_to_euler(rmat: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to (pitch, yaw, roll) in degrees."""
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw = np.arctan2(-rmat[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


def _estimate_head_pose(
    kps: np.ndarray, frame_h: int, frame_w: int
) -> dict:
    """Estimate head pose (yaw, pitch, roll) via solvePnP."""
    focal = float(frame_w)
    cam_matrix = np.array(
        [
            [focal, 0.0, frame_w / 2.0],
            [0.0, focal, frame_h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, _ = cv2.solvePnP(
            _3D_FACE_MODEL, kps.astype(np.float64), cam_matrix, dist
        )
        if not ok:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        rmat, _ = cv2.Rodrigues(rvec)
        angles = _rotation_to_euler(rmat)
        return {
            "yaw": round(float(angles[1]), 2),
            "pitch": round(float(angles[0]), 2),
            "roll": round(float(angles[2]), 2),
        }
    except Exception:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


# ── TrackedFace ───────────────────────────────────────────────


class TrackedFace:
    """A face with a persistent track ID, tracked across frames.

    On detection frames, all fields are refreshed from InsightFace.
    On tracking frames, kps/bbox are updated via optical flow; the
    embedding is reused from the last detection (no re-extraction).
    """

    def __init__(self, track_id: int, face) -> None:
        self.track_id = track_id
        self.bbox = np.asarray(face.bbox, dtype=np.float32).copy()
        self.kps = np.asarray(face.kps, dtype=np.float32).copy()
        emb = getattr(face, "embedding", None)
        nemb = getattr(face, "normed_embedding", None)
        self.embedding = np.asarray(emb).copy() if emb is not None else None
        self.normed_embedding = (
            np.asarray(nemb).copy() if nemb is not None else None
        )
        self.det_score = float(face.det_score)
        self.confidence = float(face.det_score)
        self.miss_count = 0
        self.frames_tracked = 1
        self.head_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    def update_from_detection(self, face) -> None:
        """Refresh all fields from a fresh InsightFace detection."""
        self.bbox = np.asarray(face.bbox, dtype=np.float32).copy()
        self.kps = np.asarray(face.kps, dtype=np.float32).copy()
        emb = getattr(face, "embedding", None)
        nemb = getattr(face, "normed_embedding", None)
        if emb is not None:
            self.embedding = np.asarray(emb).copy()
        if nemb is not None:
            self.normed_embedding = np.asarray(nemb).copy()
        self.det_score = float(face.det_score)
        self.confidence = float(face.det_score)
        self.miss_count = 0
        self.frames_tracked += 1

    def update_from_tracking(self, new_kps: np.ndarray) -> None:
        """Update landmarks from optical-flow tracking."""
        self.kps = new_kps.astype(np.float32)
        self.bbox = _kps_to_bbox(new_kps)
        # Confidence decays with each tracking frame
        self.confidence = max(self.confidence * 0.93, 0.1)
        self.miss_count = 0
        self.frames_tracked += 1

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "confidence": round(self.confidence, 4),
            "miss_count": self.miss_count,
            "frames_tracked": self.frames_tracked,
            "head_pose": self.head_pose,
            "bbox": [round(float(v), 1) for v in self.bbox],
        }


# ── FaceTracker ───────────────────────────────────────────────


class FaceTracker:
    """
    Multi-face landmark tracker.

    Alternates between full InsightFace detection (every *detection_interval*
    frames) and KLT optical-flow landmark tracking (intermediate frames).
    Recovers automatically by forcing detection when tracking fails.
    """

    def __init__(self) -> None:
        self._tracks: list[TrackedFace] = []
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0
        self._last_detection_frame = -1
        self._force_detection = True
        self._next_id = 0
        self._state = "idle"

        # Metrics
        self._fps_times: deque = deque(maxlen=120)
        self._confidences: deque = deque(maxlen=120)
        self._total_tracks_created = 0
        self._total_detections = 0
        self._total_tracking_frames = 0

        # Runtime overrides (set by auto-optimization)
        self._detection_interval_override: int | None = None

    # ── Public API ─────────────────────────────────────────

    def set_detection_interval(self, interval: int | None) -> None:
        """Override detection interval at runtime (None = use config default)."""
        self._detection_interval_override = interval

    def get_detection_interval(self) -> int:
        if self._detection_interval_override is not None:
            return self._detection_interval_override
        return get_settings().tracking_detection_interval

    def reset(self) -> None:
        """Clear all tracking state (call when a new session starts)."""
        self._tracks = []
        self._prev_gray = None
        self._frame_count = 0
        self._last_detection_frame = -1
        self._force_detection = True
        self._next_id = 0
        self._state = "idle"
        self._fps_times.clear()
        self._confidences.clear()
        self._total_tracks_created = 0
        self._total_detections = 0
        self._total_tracking_frames = 0
        logger.info("Tracker reset")

    def update(
        self, frame: np.ndarray
    ) -> tuple[list[TrackedFace], dict]:
        """
        Process a frame: detect or track faces.

        Returns (tracked_faces, tracking_info) where tracking_info has:
            confidence, state, active_tracks, detection_frame
        """
        settings = get_settings()
        self._frame_count += 1
        now = time.time()
        self._fps_times.append(now)

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Decide: full detection or optical-flow tracking?
        detect = self._should_detect()

        if detect:
            faces = self._run_detection(frame, settings)
            self._match_and_update(faces, settings)
            self._last_detection_frame = self._frame_count
            self._force_detection = False
            self._total_detections += 1
            self._state = "detecting" if self._tracks else "idle"
        else:
            self._run_tracking(gray, h, w)
            self._total_tracking_frames += 1
            self._state = "tracking" if self._tracks else "idle"

        # Estimate head pose for all active tracks
        if settings.tracking_enable_head_pose:
            for t in self._tracks:
                t.head_pose = _estimate_head_pose(t.kps, h, w)

        # Expire stale tracks
        self._expire_stale(settings)

        # Store previous frame for next optical-flow pass
        self._prev_gray = gray

        # Compute metrics
        avg_conf = self._avg_confidence()
        self._confidences.append(avg_conf)

        info = {
            "confidence": round(avg_conf, 4),
            "state": self._state,
            "active_tracks": len(self._tracks),
            "detection_frame": detect,
        }
        return list(self._tracks), info

    def get_metrics(self) -> dict:
        """Return detailed tracking metrics for the /tracking endpoint."""
        return {
            "state": self._state,
            "confidence": round(self._avg_confidence(), 4),
            "fps": round(self._current_fps(), 2),
            "active_tracks": len(self._tracks),
            "total_tracks_created": self._total_tracks_created,
            "total_detections": self._total_detections,
            "total_tracking_frames": self._total_tracking_frames,
            "detection_interval": self.get_detection_interval(),
            "tracks": [t.to_dict() for t in self._tracks],
        }

    # ── Detection vs tracking decision ─────────────────────

    def _should_detect(self) -> bool:
        """True if a full detection should run this frame."""
        if not self._tracks:
            return True
        if self._force_detection:
            return True
        interval = self.get_detection_interval()
        if self._frame_count - self._last_detection_frame >= interval:
            return True
        return False

    # ── Detection frame ────────────────────────────────────

    def _run_detection(self, frame: np.ndarray, settings):
        """Run InsightFace detection and filter by threshold."""
        detector = model_manager.detector
        if detector is None:
            logger.warning("Detector not loaded — skipping detection")
            return []
        faces = detector.get(frame)
        return [f for f in faces if f.det_score >= settings.swap_det_threshold]

    def _match_and_update(self, detections: list, settings) -> None:
        """
        Match new detections to existing tracks using IoU + embedding
        similarity (greedy assignment). Unmatched detections create new
        tracks; unmatched tracks increment miss_count.
        """
        if not detections:
            for t in self._tracks:
                t.miss_count += 1
            return

        if not self._tracks:
            for face in detections:
                self._create_track(face)
            return

        # Build cost matrix: lower cost = better match
        n_tracks = len(self._tracks)
        n_det = len(detections)
        costs = np.full((n_tracks, n_det), 1.0, dtype=np.float32)

        for i, track in enumerate(self._tracks):
            for j, det in enumerate(detections):
                iou = _iou(track.bbox, det.bbox)
                sim = _embedding_sim(track.embedding, det.embedding)
                # Combined similarity (higher = better); cost = 1 - similarity
                combined = 0.4 * iou + 0.6 * sim
                costs[i, j] = 1.0 - combined

        # Greedy assignment: pick the lowest-cost pair repeatedly
        matched_tracks = set()
        matched_dets = set()
        flat = []
        for i in range(n_tracks):
            for j in range(n_det):
                flat.append((costs[i, j], i, j))
        flat.sort()

        iou_thr = settings.tracking_iou_threshold
        sim_thr = settings.tracking_embedding_threshold

        for cost, i, j in flat:
            if i in matched_tracks or j in matched_dets:
                continue
            # Accept if IoU OR embedding similarity is high enough
            det = detections[j]
            iou = _iou(self._tracks[i].bbox, det.bbox)
            sim = _embedding_sim(self._tracks[i].embedding, det.embedding)
            if iou >= iou_thr or sim >= sim_thr:
                self._tracks[i].update_from_detection(det)
                matched_tracks.add(i)
                matched_dets.add(j)

        # Unmatched detections → new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._create_track(det)

        # Unmatched tracks → increment miss
        for i, track in enumerate(self._tracks):
            if i not in matched_tracks:
                track.miss_count += 1
                track.confidence = max(track.confidence * 0.8, 0.1)

    def _create_track(self, face) -> TrackedFace:
        track = TrackedFace(self._next_id, face)
        self._next_id += 1
        self._tracks.append(track)
        self._total_tracks_created += 1
        logger.info(
            "Created track %d (score %.3f)", track.track_id, track.det_score
        )
        return track

    # ── Tracking frame (optical flow) ──────────────────────

    def _run_tracking(self, gray: np.ndarray, h: int, w: int) -> None:
        """
        Track each face's 5 landmarks via KLT optical flow with
        forward-backward error check. Force re-detection on failure.
        """
        if self._prev_gray is None:
            self._force_detection = True
            return

        lk_params = dict(
            winSize=(31, 31),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        any_failed = False

        for track in self._tracks:
            pts = track.kps.reshape(-1, 1, 2).astype(np.float32)

            # Forward flow: prev → curr
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, pts, None, **lk_params
            )
            if next_pts is None or status is None:
                track.miss_count += 1
                track.confidence = max(track.confidence * 0.7, 0.1)
                any_failed = True
                continue

            # Backward flow: curr → prev (for error check)
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._prev_gray, next_pts, None, **lk_params
            )

            # Forward-backward error
            if back_pts is not None and back_status is not None:
                fb_error = np.linalg.norm(pts - back_pts, axis=2).reshape(-1)
                good = (
                    (status.reshape(-1) == 1)
                    & (back_status.reshape(-1) == 1)
                    & (fb_error < 2.0)
                )
            else:
                good = status.reshape(-1) == 1

            # Need at least 4 of 5 landmarks tracked
            if good.sum() < 4:
                track.miss_count += 1
                track.confidence = max(track.confidence * 0.7, 0.1)
                any_failed = True
                continue

            # Use tracked landmarks (fill any lost points with originals)
            new_kps = next_pts.reshape(-1, 2).astype(np.float32)
            for k in range(5):
                if not good[k]:
                    new_kps[k] = track.kps[k]

            # Clamp to frame bounds
            new_kps[:, 0] = np.clip(new_kps[:, 0], 0, w - 1)
            new_kps[:, 1] = np.clip(new_kps[:, 1], 0, h - 1)

            track.update_from_tracking(new_kps)

        # If any track failed, force detection on the next frame (recovery)
        if any_failed:
            self._force_detection = True
            self._state = "recovering"

        # If any track's confidence is too low, force re-detection
        threshold = get_settings().tracking_confidence_threshold
        if any(t.confidence < threshold for t in self._tracks):
            self._force_detection = True

    # ── Track lifecycle ────────────────────────────────────

    def _expire_stale(self, settings) -> None:
        """Remove tracks that have been missed for too long."""
        max_missed = settings.tracking_max_missed
        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if t.miss_count < max_missed]
        removed = before - len(self._tracks)
        if removed > 0:
            logger.info("Expired %d stale track(s)", removed)
            if not self._tracks:
                self._state = "lost"
                self._force_detection = True

    # ── Metrics helpers ────────────────────────────────────

    def _avg_confidence(self) -> float:
        if not self._tracks:
            return 0.0
        return sum(t.confidence for t in self._tracks) / len(self._tracks)

    def _current_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        if elapsed <= 0:
            return 0.0
        return len(self._fps_times) / elapsed


# ── Singleton ─────────────────────────────────────────────────

face_tracker = FaceTracker()