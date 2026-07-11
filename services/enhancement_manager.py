"""
Adaptive face enhancement manager.

Features
────────
• Supports GFPGAN and CodeFormer (loaded by ModelManager)
• Adaptive: assesses frame quality and skips enhancement when already high
• Three modes: fast, balanced, ultra
• Returns enhancement time per frame
• Runtime mode switching via set_mode() — no restart required

Mode behaviour
──────────────
  fast     — GFPGAN only, only when quality is low; lowest latency
  balanced — GFPGAN when quality is medium-low; CodeFormer when quality is low
  ultra    — CodeFormer always (highest quality, highest latency)

Quality assessment
──────────────────
Uses Laplacian variance (blur metric) on the face region:
  • high   → skip enhancement entirely
  • medium → GFPGAN (fast, light restoration)
  • low    → CodeFormer (balanced/ultra) or GFPGAN (fast)
"""
from __future__ import annotations

import time
import cv2
import numpy as np
from collections import deque
from typing import Optional

from utils.logger import setup_logger
from services.model_manager import model_manager

logger = setup_logger("enhancement_manager")

# ── Quality thresholds (Laplacian variance) ───────────────────
# Higher variance = sharper image. These thresholds are calibrated
# for face crops of ~128–512px. Overridden from config at startup.
QUALITY_HIGH = 150.0     # above this → skip enhancement
QUALITY_MEDIUM = 80.0    # above this but below HIGH → light enhance


def _sync_thresholds() -> None:
    """Sync quality thresholds from config (called once at startup)."""
    global QUALITY_HIGH, QUALITY_MEDIUM
    try:
        from config import get_settings
        s = get_settings()
        QUALITY_HIGH = s.enhancement_quality_high
        QUALITY_MEDIUM = s.enhancement_quality_medium
    except Exception:
        pass

# ── Modes ─────────────────────────────────────────────────────

VALID_MODES = ("off", "fast", "balanced", "ultra")


class EnhancementManager:
    """
    Adaptive face enhancement with runtime mode switching.

    The manager is stateless across frames except for:
      • current mode (switchable at runtime)
      • rolling metrics (enhancement time, skip rate)
    """

    def __init__(self) -> None:
        _sync_thresholds()
        self._mode: str = "balanced"
        self._lock_time: float = 0.0  # not used for threading; reserved
        # Metrics
        self._enhance_times: deque = deque(maxlen=120)
        self._fps_times: deque = deque(maxlen=120)
        self._total_enhanced: int = 0
        self._total_skipped: int = 0
        self._total_frames: int = 0
        self._mode_switch_count: int = 0

    # ── Mode management (runtime switching) ────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> bool:
        """Switch enhancement mode at runtime. Returns True if accepted."""
        mode = mode.lower().strip()
        if mode not in VALID_MODES:
            return False
        if mode == self._mode:
            return True
        old = self._mode
        self._mode = mode
        self._mode_switch_count += 1
        logger.info("Enhancement mode switched: %s → %s", old, mode)
        return True

    # ── Available enhancers ────────────────────────────────

    def _get_enhancer(self, name: str):
        """Retrieve a loaded enhancer instance by key."""
        enhancers = getattr(model_manager, "enhancers", {})
        if not enhancers:
            return None
        return enhancers.get(name)

    @property
    def available_enhancers(self) -> list[str]:
        enhancers = getattr(model_manager, "enhancers", {})
        return list(enhancers.keys()) if enhancers else []

    # ── Quality assessment ─────────────────────────────────

    def _assess_quality(self, face_crop: np.ndarray) -> str:
        """
        Assess face quality using Laplacian variance (blur metric).
        Returns 'high', 'medium', or 'low'.
        """
        if face_crop is None or face_crop.size == 0:
            return "low"
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var >= QUALITY_HIGH:
            return "high"
        if laplacian_var >= QUALITY_MEDIUM:
            return "medium"
        return "low"

    # ── Enhancement dispatch ───────────────────────────────

    def _run_gfpgan(self, frame: np.ndarray) -> np.ndarray | None:
        """Run GFPGAN enhancement. Returns enhanced frame or None on failure."""
        enhancer = self._get_enhancer("gfpgan_v14")
        if enhancer is None:
            return None
        try:
            _, _, result = enhancer.enhance(frame, paste_back=True)
            return result
        except Exception as exc:
            logger.warning("GFPGAN enhancement failed: %s", exc)
            return None

    def _run_codeformer(self, frame: np.ndarray) -> np.ndarray | None:
        """Run CodeFormer enhancement. Returns enhanced frame or None on failure."""
        enhancer = self._get_enhancer("codeformer")
        if enhancer is None:
            return None
        try:
            if hasattr(enhancer, "enhance"):
                # GFPGANer-compatible API (codeformer wrapped)
                _, _, result = enhancer.enhance(frame, paste_back=True)
                return result
            elif hasattr(enhancer, "inference"):
                # Raw CodeFormer API
                result = enhancer.inference(frame)
                return result
            else:
                logger.warning("CodeFormer enhancer has no known API")
                return None
        except Exception as exc:
            logger.warning("CodeFormer enhancement failed: %s", exc)
            return None

    # ── Public API ─────────────────────────────────────────

    def enhance(
        self, frame: np.ndarray, face_crop: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, dict]:
        """
        Adaptively enhance a frame.

        Args:
            frame: The full BGR frame (already face-swapped).
            face_crop: Optional face region crop for quality assessment.
                       If None, the full frame is used.

        Returns:
            (enhanced_frame, info) where info contains:
                enhanced (bool), enhancer (str|None), enhancement_time_ms (float),
                quality (str), mode (str), skipped (bool)
        """
        self._total_frames += 1
        self._fps_times.append(time.time())

        info = {
            "enhanced": False,
            "enhancer": None,
            "enhancement_time_ms": 0.0,
            "quality": "unknown",
            "mode": self._mode,
            "skipped": False,
        }

        if self._mode == "off":
            info["skipped"] = True
            self._total_skipped += 1
            return frame, info

        # ── Quality assessment ──────────────────────────────
        assess_region = face_crop if face_crop is not None else frame
        quality = self._assess_quality(assess_region)
        info["quality"] = quality

        # ── Skip if quality is already high ─────────────────
        if quality == "high":
            info["skipped"] = True
            self._total_skipped += 1
            return frame, info

        # ── Mode + quality → enhancer selection ─────────────
        enhancer_name = self._select_enhancer(quality)
        info["enhancer"] = enhancer_name

        if enhancer_name is None:
            info["skipped"] = True
            self._total_skipped += 1
            return frame, info

        # ── Run enhancement ─────────────────────────────────
        t_start = time.perf_counter()

        if enhancer_name == "gfpgan":
            result = self._run_gfpgan(frame)
        elif enhancer_name == "codeformer":
            result = self._run_codeformer(frame)
        else:
            result = None

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        info["enhancement_time_ms"] = round(elapsed_ms, 2)
        self._enhance_times.append(elapsed_ms)

        if result is not None:
            info["enhanced"] = True
            self._total_enhanced += 1
            return result, info
        else:
            # Enhancer failed — return original, mark as not enhanced
            info["enhancer"] = None
            info["skipped"] = True
            self._total_skipped += 1
            return frame, info

    def _select_enhancer(self, quality: str) -> Optional[str]:
        """
        Decide which enhancer to use based on mode and quality.

        Mode      | Quality | Enhancer
        �──────────┼─────────┼──────────
        fast      | medium  | gfpgan
        fast      | low     | gfpgan
        balanced  | medium  | gfpgan
        balanced  | low     | codeformer (fallback gfpgan)
        ultra     | medium  | codeformer (fallback gfpgan)
        ultra     | low     | codeformer (fallback gfpgan)
        """
        available = set(self.available_enhancers)

        if self._mode == "fast":
            if "gfpgan_v14" in available:
                return "gfpgan"
            return None

        if self._mode == "balanced":
            if quality == "medium":
                if "gfpgan_v14" in available:
                    return "gfpgan"
                return None
            # quality == "low"
            if "codeformer" in available:
                return "codeformer"
            if "gfpgan_v14" in available:
                return "gfpgan"
            return None

        if self._mode == "ultra":
            if "codeformer" in available:
                return "codeformer"
            if "gfpgan_v14" in available:
                return "gfpgan"
            return None

        return None

    # ── Metrics ────────────────────────────────────────────

    def get_metrics(self) -> dict:
        avg_time = (
            sum(self._enhance_times) / len(self._enhance_times)
            if self._enhance_times
            else 0.0
        )
        fps = 0.0
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed > 0:
                fps = len(self._fps_times) / elapsed

        skip_rate = 0.0
        if self._total_frames > 0:
            skip_rate = self._total_skipped / self._total_frames

        return {
            "mode": self._mode,
            "available_enhancers": self.available_enhancers,
            "avg_enhancement_time_ms": round(avg_time, 2),
            "fps": round(fps, 2),
            "total_frames": self._total_frames,
            "total_enhanced": self._total_enhanced,
            "total_skipped": self._total_skipped,
            "skip_rate": round(skip_rate, 4),
            "mode_switch_count": self._mode_switch_count,
            "quality_thresholds": {
                "high": QUALITY_HIGH,
                "medium": QUALITY_MEDIUM,
            },
        }

    def reset_metrics(self) -> None:
        self._enhance_times.clear()
        self._fps_times.clear()
        self._total_enhanced = 0
        self._total_skipped = 0
        self._total_frames = 0


# ── Singleton ─────────────────────────────────────────────────

enhancement_manager = EnhancementManager()