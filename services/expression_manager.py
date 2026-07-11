"""
Expression Preservation — maintain original facial expressions on the swapped face.

Pipeline
────────
  1. Extract dense landmarks (106-point) from original and swapped frames
  2. Compute displacement: swapped_lmks → orig_lmks
  3. Build dense flow map via coarse-grid accumulation + Gaussian smoothing
  4. Apply warp via cv2.remap (smooth piecewise deformation)
  5. Result: swapped face deformed to match original expressions

Preserves:
  • Eye movement  • Blink   • Eyebrows     • Mouth movement
  • Lip sync      • Smile   • Head rotation • Facial muscles

Primary: LivePortrait (expression transfer model, when available)
Fallback: Dense landmark warping (106-point, or 5-point affine)
"""
from __future__ import annotations

import time
import cv2
import numpy as np
from collections import deque
from typing import Optional, Any

from utils.logger import setup_logger
from services.model_manager import model_manager

logger = setup_logger("expression_manager")


class ExpressionPreservationManager:
    """
    Preserves original facial expressions on the swapped face.

    Stateless across frames except for configuration and rolling metrics.
    """

    def __init__(self) -> None:
        self._enabled: bool = True
        self._warp_strength: float = 1.0
        self._grid_size: int = 32
        self._liveportrait: Any = None

        # Metrics
        self._warp_times: deque = deque(maxlen=120)
        self._fps_times: deque = deque(maxlen=120)
        self._total_warped: int = 0
        self._total_skipped: int = 0
        self._lp_used: int = 0
        self._dense_used: int = 0
        self._affine_used: int = 0

    # ── Configuration ───────────────────────────────────────

    def configure(
        self,
        enabled: Optional[bool] = None,
        warp_strength: Optional[float] = None,
        grid_size: Optional[int] = None,
    ) -> None:
        if enabled is not None:
            self._enabled = enabled
        if warp_strength is not None:
            self._warp_strength = max(0.0, min(2.0, float(warp_strength)))
        if grid_size is not None:
            self._grid_size = max(8, min(64, int(grid_size)))

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_warp_strength(self, strength: float) -> bool:
        if strength < 0.0 or strength > 2.0:
            return False
        self._warp_strength = strength
        return True

    def set_grid_size(self, size: int) -> bool:
        if size < 8 or size > 64:
            return False
        self._grid_size = size
        return True

    def set_liveportrait(self, instance: Any) -> None:
        """Attach a LivePortrait pipeline instance for expression transfer."""
        self._liveportrait = instance
        logger.info("LivePortrait expression transfer attached")

    @property
    def liveportrait_available(self) -> bool:
        return self._liveportrait is not None

    # ── Public API ──────────────────────────────────────────

    def preserve_expression(
        self, original: np.ndarray, swapped: np.ndarray, face
    ) -> tuple[np.ndarray, dict]:
        """
        Warp the swapped face to match the original frame's expressions.

        Args:
            original: Original BGR frame (pre-swap — source of expressions).
            swapped:  Swapped BGR frame (post InSwapper — has source identity).
            face:     InsightFace face object with .bbox, .kps.

        Returns:
            (warped_frame, info_dict)
        """
        if not self._enabled:
            self._total_skipped += 1
            return swapped, self._info(skipped=True)

        t_start = time.perf_counter()

        # ── Try LivePortrait first ──────────────────────────
        if self._liveportrait is not None:
            result, lp_info = self._run_liveportrait(original, swapped, face)
            if result is not None:
                elapsed = (time.perf_counter() - t_start) * 1000
                self._record(elapsed)
                self._lp_used += 1
                lp_info["time_ms"] = round(elapsed, 2)
                lp_info["enabled"] = True
                lp_info["skipped"] = False
                lp_info["features"] = self._compute_features(face)
                return result, lp_info

        # ── Landmark-based warping ──────────────────────────
        orig_lmks = self._get_landmarks(original, face, re_detect=False)
        swapped_lmks = self._get_landmarks(swapped, face, re_detect=True)

        if orig_lmks is None or swapped_lmks is None:
            self._total_skipped += 1
            return swapped, self._info(skipped=True)

        orig_lmks = np.asarray(orig_lmks, dtype=np.float32)
        swapped_lmks = np.asarray(swapped_lmks, dtype=np.float32)
        n_points = len(orig_lmks)

        features = self._compute_features(face)

        if n_points >= 68:
            result = self._dense_warp(swapped, swapped_lmks, orig_lmks, face)
            method = "dense_warp"
            self._dense_used += 1
        elif n_points >= 3:
            result = self._affine_warp(swapped, swapped_lmks, orig_lmks)
            method = "affine_warp"
            self._affine_used += 1
        else:
            self._total_skipped += 1
            return swapped, self._info(skipped=True)

        elapsed = (time.perf_counter() - t_start) * 1000
        self._record(elapsed)

        return result, self._info(
            skipped=False,
            method=method,
            time_ms=elapsed,
            n_points=n_points,
            features=features,
        )

    # ── Landmark extraction ─────────────────────────────────

    def _get_landmarks(
        self, frame: np.ndarray, face, re_detect: bool = False
    ) -> Optional[np.ndarray]:
        """
        Get dense facial landmarks from a frame.

        If re_detect is False, tries the face object's landmark_2d_106 first.
        Otherwise, re-detects on the face ROI to get fresh landmarks.
        Falls back to 5-point kps if dense landmarks are unavailable.
        """
        if not re_detect:
            lmks = getattr(face, "landmark_2d_106", None)
            if lmks is not None:
                return np.asarray(lmks, dtype=np.float32)

        bbox = face.bbox.astype(int)
        h, w = frame.shape[:2]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        pad = int(max(bw, bh) * 0.2)
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(w, bbox[2] + pad)
        y2 = min(h, bbox[3] + pad)

        if x2 <= x1 or y2 <= y1:
            return np.asarray(face.kps, dtype=np.float32)

        roi = frame[y1:y2, x1:x2]
        try:
            detected = model_manager.detector.get(roi)
        except Exception as exc:
            logger.debug("Landmark re-detection failed: %s", exc)
            return np.asarray(face.kps, dtype=np.float32)

        if not detected:
            return np.asarray(face.kps, dtype=np.float32)

        best = max(detected, key=lambda f: f.det_score)
        lmks = getattr(best, "landmark_2d_106", None)
        if lmks is not None:
            lmks = np.asarray(lmks, dtype=np.float32).copy()
            lmks[:, 0] += x1
            lmks[:, 1] += y1
            return lmks

        # Fall back to 5-point kps
        return np.asarray(face.kps, dtype=np.float32)

    # ── Dense landmark warping ──────────────────────────────

    def _dense_warp(
        self,
        swapped: np.ndarray,
        src_lmks: np.ndarray,
        dst_lmks: np.ndarray,
        face,
    ) -> np.ndarray:
        """
        Warp the swapped frame so src_lmks align to dst_lmks using a
        dense displacement field applied via cv2.remap.

        The warp is limited to the face ROI (bbox + padding) to avoid
        deforming the background.
        """
        bbox = face.bbox.astype(int)
        h, w = swapped.shape[:2]
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        pad = int(max(bw, bh) * 0.3)
        px1 = max(0, bbox[0] - pad)
        py1 = max(0, bbox[1] - pad)
        px2 = min(w, bbox[2] + pad)
        py2 = min(h, bbox[3] + pad)

        if px2 <= px1 or py2 <= py1:
            return swapped

        roi = swapped[py1:py2, px1:px2].copy()
        rh, rw = roi.shape[:2]

        # Shift landmarks to ROI-local coordinates
        src_roi = src_lmks.copy()
        src_roi[:, 0] -= px1
        src_roi[:, 1] -= py1
        dst_roi = dst_lmks.copy()
        dst_roi[:, 0] -= px1
        dst_roi[:, 1] -= py1

        # Clip to ROI bounds
        src_roi[:, 0] = np.clip(src_roi[:, 0], 0, rw - 1)
        src_roi[:, 1] = np.clip(src_roi[:, 1], 0, rh - 1)
        dst_roi[:, 0] = np.clip(dst_roi[:, 0], 0, rw - 1)
        dst_roi[:, 1] = np.clip(dst_roi[:, 1], 0, rh - 1)

        # Build dense displacement field
        map_x, map_y = self._build_flow_map(src_roi, dst_roi, (rh, rw))

        # Apply warp
        warped_roi = cv2.remap(
            roi, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        result = swapped.copy()
        result[py1:py2, px1:px2] = warped_roi
        return result

    def _build_flow_map(
        self,
        src_lmks: np.ndarray,
        dst_lmks: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build a dense displacement field from landmark correspondences.

        1. Accumulate per-landmark displacements onto a coarse grid
        2. Gaussian-blur the grid to spread displacements smoothly
        3. Mask by influence weight (zero displacement far from landmarks)
        4. Resize to full ROI resolution
        5. Return (map_x, map_y) for cv2.remap
        """
        rh, rw = shape
        gw = gh = self._grid_size

        # Per-landmark displacement (where each landmark should move)
        deltas = (dst_lmks - src_lmks) * self._warp_strength  # (N, 2)

        # Accumulate on coarse grid
        dx_grid = np.zeros((gh, gw), dtype=np.float32)
        dy_grid = np.zeros((gh, gw), dtype=np.float32)
        w_grid = np.zeros((gh, gw), dtype=np.float32)

        for i in range(len(src_lmks)):
            sx, sy = src_lmks[i]
            dx, dy = deltas[i]

            gx = int(sx / max(rw, 1) * (gw - 1))
            gy = int(sy / max(rh, 1) * (gh - 1))
            gx = max(0, min(gw - 1, gx))
            gy = max(0, min(gh - 1, gy))

            dx_grid[gy, gx] += dx
            dy_grid[gy, gx] += dy
            w_grid[gy, gx] += 1.0

        # Normalize by weight
        mask = w_grid > 0
        dx_grid = np.where(mask, dx_grid / np.maximum(w_grid, 1e-8), 0)
        dy_grid = np.where(mask, dy_grid / np.maximum(w_grid, 1e-8), 0)

        # Gaussian blur to spread displacements into neighboring grid cells
        ksize = max(3, gw // 2)
        if ksize % 2 == 0:
            ksize += 1
        sigma = ksize * 0.3

        dx_grid = cv2.GaussianBlur(dx_grid, (ksize, ksize), sigma)
        dy_grid = cv2.GaussianBlur(dy_grid, (ksize, ksize), sigma)
        w_blur = cv2.GaussianBlur(
            w_grid.astype(np.float32), (ksize, ksize), sigma
        )

        # Zero out displacement in regions far from any landmark
        w_max = float(w_blur.max())
        if w_max > 0:
            w_norm = w_blur / w_max
            dx_grid *= w_norm
            dy_grid *= w_norm

        # Resize to full ROI resolution
        dx_full = cv2.resize(dx_grid, (rw, rh))
        dy_full = cv2.resize(dy_grid, (rw, rh))

        # Build remap coordinates: target_pixel = source_pixel + displacement
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
        map_x = xx + dx_full
        map_y = yy + dy_full

        return map_x, map_y

    # ── Affine warp (5-point fallback) ──────────────────────

    def _affine_warp(
        self,
        swapped: np.ndarray,
        src_lmks: np.ndarray,
        dst_lmks: np.ndarray,
    ) -> np.ndarray:
        """
        Global affine warp for when only 5-point landmarks are available.
        Handles head rotation and translation but not fine expressions.
        """
        transform = cv2.estimateAffinePartial2D(src_lmks, dst_lmks)[0]
        if transform is None:
            return swapped
        h, w = swapped.shape[:2]
        return cv2.warpAffine(
            swapped, transform, (w, h), borderMode=cv2.BORDER_REFLECT
        )

    # ── LivePortrait ────────────────────────────────────────

    def _run_liveportrait(
        self, original: np.ndarray, swapped: np.ndarray, face
    ) -> tuple[Optional[np.ndarray], dict]:
        """
        Run LivePortrait expression transfer.

        LivePortrait animates a source image with a driving frame's motion.
        Here, 'swapped' is the source (identity) and 'original' is the driver
        (expressions to copy).
        """
        try:
            if hasattr(self._liveportrait, "execute"):
                result = self._liveportrait.execute(swapped, original)
                return result, {"method": "liveportrait"}
            elif hasattr(self._liveportrait, "animate"):
                result = self._liveportrait.animate(swapped, original)
                return result, {"method": "liveportrait"}
            else:
                logger.warning("LivePortrait instance has no known API")
                return None, {}
        except Exception as exc:
            logger.warning("LivePortrait failed, falling back: %s", exc)
            return None, {}

    # ── Expression features ─────────────────────────────────

    def _compute_features(self, face) -> dict:
        """
        Compute expression feature metrics from the 5-point kps.

        These are proxies for expression state, useful for monitoring
        and diagnostics. The dense warp itself handles the actual
        expression transfer via all landmark points.
        """
        kps = np.asarray(face.kps, dtype=np.float32)
        l_eye, r_eye, nose, l_mouth, r_mouth = kps

        eye_dist = float(np.linalg.norm(r_eye - l_eye))
        mouth_w = float(np.linalg.norm(r_mouth - l_mouth))
        eye_center = (l_eye + r_eye) / 2.0
        mouth_center = (l_mouth + r_mouth) / 2.0
        face_h = float(np.linalg.norm(eye_center - mouth_center))
        smile_ratio = mouth_w / max(eye_dist, 1e-6)

        return {
            "eye_distance": round(eye_dist, 2),
            "mouth_width": round(mouth_w, 2),
            "face_height": round(face_h, 2),
            "smile_ratio": round(smile_ratio, 4),
        }

    # ── Info / metrics ──────────────────────────────────────

    def _record(self, elapsed_ms: float) -> None:
        self._warp_times.append(elapsed_ms)
        self._fps_times.append(time.time())
        self._total_warped += 1

    def _info(
        self,
        skipped: bool = False,
        method: str = "none",
        time_ms: float = 0.0,
        n_points: int = 0,
        features: Optional[dict] = None,
    ) -> dict:
        return {
            "enabled": self._enabled and not skipped,
            "method": method,
            "time_ms": round(time_ms, 2),
            "landmark_count": n_points,
            "warp_strength": self._warp_strength,
            "features": features or {},
            "skipped": skipped,
        }

    def get_status(self) -> dict:
        avg_time = (
            sum(self._warp_times) / len(self._warp_times)
            if self._warp_times
            else 0.0
        )
        fps = 0.0
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed > 0:
                fps = len(self._fps_times) / elapsed

        return {
            "enabled": self._enabled,
            "liveportrait_available": self.liveportrait_available,
            "warp_strength": self._warp_strength,
            "grid_size": self._grid_size,
            "avg_time_ms": round(avg_time, 2),
            "fps": round(fps, 2),
            "total_warped": self._total_warped,
            "total_skipped": self._total_skipped,
            "liveportrait_used": self._lp_used,
            "dense_warp_used": self._dense_used,
            "affine_warp_used": self._affine_used,
            "preservation_targets": [
                "eye_movement",
                "blink",
                "eyebrows",
                "mouth_movement",
                "lip_sync",
                "smile",
                "head_rotation",
                "facial_muscles",
            ],
        }

    def reset_metrics(self) -> None:
        self._warp_times.clear()
        self._fps_times.clear()
        self._total_warped = 0
        self._total_skipped = 0
        self._lp_used = 0
        self._dense_used = 0
        self._affine_used = 0


# ── Singleton ────────────────────────────────────────────────

expression_manager = ExpressionPreservationManager()