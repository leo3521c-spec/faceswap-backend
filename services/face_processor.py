"""
Minimal real-time face swap pipeline:

  JPEG → Decode → Detect → InSwapper128 (single best face) → Encode → Binary

No expression preservation, no semantic masking, no enhancement.
Only the highest-confidence face is swapped.
Runs in a worker thread — never blocks the WebSocket event loop.
"""
import time
import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque

from config import get_settings
from utils.logger import setup_logger
from services.metrics import metrics
from services.model_manager import model_manager
from services.gpu_manager import gpu_manager
from services.face_tracker import face_tracker

logger = setup_logger("face_processor")


@dataclass
class FrameResult:
    """Minimal result from the face-swap pipeline."""
    jpeg_bytes: bytes
    inference_time_ms: float
    face_detected: bool
    confidence: float

    def to_metadata(self) -> dict:
        return {
            "type": "frame_result",
            "latency_ms": round(self.inference_time_ms, 2),
            "fps": round(metrics.current_fps, 1),
            "face_detected": self.face_detected,
            "confidence": round(self.confidence, 4),
        }


# ── JPEG helpers ─────────────────────────────────────────────


def decode_jpeg(data: bytes) -> np.ndarray | None:
    """Decode JPEG bytes into a BGR numpy array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ── Source face ──────────────────────────────────────────────


def extract_source_face(source_bytes: bytes):
    """Decode the source image and return the highest-confidence face."""
    img = decode_jpeg(source_bytes)
    if img is None:
        logger.error("Could not decode source face image")
        return None

    faces = model_manager.detector.get(img)
    if not faces:
        logger.error("No face detected in source image")
        return None

    best = max(faces, key=lambda f: f.det_score)
    logger.info(
        "Source face extracted — score %.3f, embedding dim %d",
        float(best.det_score),
        best.normed_embedding.shape[0],
    )
    return best


# ── Full pipeline ────────────────────────────────────────────


def process_frame(frame_bytes: bytes, source_face) -> FrameResult:
    """
    Minimal pipeline:
      1. Decode JPEG
      2. Face detection/tracking
      3. Pick single highest-confidence face
      4. InSwapper128 swap
      5. Encode JPEG
    """
    settings = get_settings()
    t_start = time.perf_counter()

    # ── 1 · Decode ──────────────────────────────────────────
    frame = decode_jpeg(frame_bytes)

    if frame is None:
        return FrameResult(
            jpeg_bytes=frame_bytes,
            inference_time_ms=0,
            face_detected=False,
            confidence=0,
        )

    # ── 2 · Face Tracking (detection every N frames, tracking in between) ──
    faces, tracking = face_tracker.update(frame)

    if not faces:
        result_bytes = encode_jpeg(frame, settings.jpeg_quality)
        return FrameResult(
            jpeg_bytes=result_bytes,
            inference_time_ms=(time.perf_counter() - t_start) * 1000,
            face_detected=False,
            confidence=0,
        )

    # ── 3 · Single face only — pick highest confidence ──────
    best_face = max(faces, key=lambda f: f.det_score)
    confidence = float(best_face.det_score)

    # ── 4 · InSwapper128 ────────────────────────────────────
    result = model_manager.swapper.get(
        frame, best_face, source_face, paste_back=True
    )

    # ── 5 · Encode JPEG ─────────────────────────────────────
    result_bytes = encode_jpeg(result, settings.jpeg_quality)
    inference_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "Frame: %dms, face_conf=%.3f, track_state=%s, jpeg=%d bytes",
        inference_ms, confidence, tracking["state"], len(result_bytes),
    )

    gpu_manager.record_inference(inference_ms)

    return FrameResult(
        jpeg_bytes=result_bytes,
        inference_time_ms=inference_ms,
        face_detected=True,
        confidence=confidence,
    )