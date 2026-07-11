"""
Real GPU inference pipeline:
  JPEG → Decode → Detect → Embed → Align → Swap → Enhance → Merge → Encode → Binary

Every step is timed. Results include inference_time_ms, face_count,
and detection_confidence. Runs entirely in a worker thread — never
blocks the WebSocket event loop.
"""
import time
import cv2
import numpy as np
from dataclasses import dataclass, field

from config import get_settings
from utils.logger import setup_logger
from services.metrics import metrics
from services.model_manager import model_manager
from services.gpu_manager import gpu_manager
from services.face_tracker import face_tracker
from services.enhancement_manager import enhancement_manager
from services.face_masking import face_masking_manager
from services.expression_manager import expression_manager

logger = setup_logger("face_processor")


# ── Result structure ─────────────────────────────────────────


@dataclass
class FrameResult:
    """Structured result from the face-swap pipeline."""
    jpeg_bytes: bytes
    inference_time_ms: float
    face_count: int
    detection_confidence: float
    face_scores: list[float] = field(default_factory=list)
    enhanced: bool = False
    pass_through: bool = False
    tracking_confidence: float = 0.0
    tracking_state: str = "idle"
    track_ids: list[int] = field(default_factory=list)
    head_poses: list[dict] = field(default_factory=list)
    enhancement_time_ms: float = 0.0
    enhancement_mode: str = "balanced"
    enhancement_quality: str = "unknown"
    enhancer_used: str | None = None
    enhancement_skipped: bool = False
    masking_applied: bool = False
    masking_method: str = "none"
    masking_time_ms: float = 0.0
    masking_color_corrected: bool = False
    masking_lighting_corrected: bool = False
    masking_occlusion_handled: bool = False
    masking_feather_radius: int = 0
    expression_preserved: bool = False
    expression_method: str = "none"
    expression_time_ms: float = 0.0
    expression_landmark_count: int = 0
    expression_warp_strength: float = 1.0

    def to_metadata(self) -> dict:
        return {
            "type": "frame_result",
            "inference_time_ms": round(self.inference_time_ms, 2),
            "face_count": self.face_count,
            "detection_confidence": round(self.detection_confidence, 4),
            "face_scores": [round(s, 4) for s in self.face_scores],
            "enhanced": self.enhanced,
            "pass_through": self.pass_through,
            "tracking_confidence": round(self.tracking_confidence, 4),
            "tracking_state": self.tracking_state,
            "track_ids": self.track_ids,
            "head_poses": self.head_poses,
            "enhancement_time_ms": round(self.enhancement_time_ms, 2),
            "enhancement_mode": self.enhancement_mode,
            "enhancement_quality": self.enhancement_quality,
            "enhancer_used": self.enhancer_used,
            "enhancement_skipped": self.enhancement_skipped,
            "masking_applied": self.masking_applied,
            "masking_method": self.masking_method,
            "masking_time_ms": round(self.masking_time_ms, 2),
            "masking_color_corrected": self.masking_color_corrected,
            "masking_lighting_corrected": self.masking_lighting_corrected,
            "masking_occlusion_handled": self.masking_occlusion_handled,
            "masking_feather_radius": self.masking_feather_radius,
            "expression_preserved": self.expression_preserved,
            "expression_method": self.expression_method,
            "expression_time_ms": round(self.expression_time_ms, 2),
            "expression_landmark_count": self.expression_landmark_count,
            "expression_warp_strength": self.expression_warp_strength,
        }


# ── JPEG helpers ─────────────────────────────────────────────


def decode_jpeg(data: bytes) -> np.ndarray | None:
    """Decode JPEG bytes into a BGR numpy array.

    Uses pinned memory for the source buffer when available,
    enabling faster CPU→GPU transfers during ONNX inference.
    """
    # Copy bytes into a pinned numpy buffer for fast H2D transfer
    buf = gpu_manager.get_pinned_array(len(data), dtype=np.uint8)
    buf[:] = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    gpu_manager.return_pinned_array(buf)
    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ── Face alignment ───────────────────────────────────────────

# ArcFace standard 5-point reference (112×112 aligned face template)
_ARCFACE_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def compute_alignment_matrix(kps: np.ndarray, size: int = 128):
    """
    Compute the affine transform that aligns a face (given its 5-point
    landmarks) to the ArcFace 112×112 reference grid, scaled to *size*.
    Returns the forward transform matrix or None.
    """
    src = np.asarray(kps, dtype=np.float32)
    ref = _ARCFACE_REF * (size / 112.0)
    return cv2.estimateAffinePartial2D(src, ref)[0]


def align_face(frame: np.ndarray, kps: np.ndarray, size: int = 128):
    """
    Warp a face region to an aligned *size*×*size* crop.
    Returns (aligned_crop, transform_matrix) or (None, None).
    """
    transform = compute_alignment_matrix(kps, size)
    if transform is None:
        return None, None
    aligned = cv2.warpAffine(frame, transform, (size, size), borderValue=0)
    return aligned, transform


# ── Source face ──────────────────────────────────────────────


def extract_source_face(source_bytes: bytes):
    """Decode the source image and return the highest-confidence face
    (with embedding already extracted by InsightFace)."""
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
    Run the complete GPU inference pipeline on a single JPEG frame.

    Steps:
      1. Decode JPEG
      2. Face tracking (detection every N frames, optical-flow tracking
         in between — assigns persistent IDs, supports head rotation)
      3. Face alignment (affine warp via 5-point landmarks)
      4. InSwapper 128 swap (paste_back=True)
      5. Expression preservation — dense landmark warping (106-point)
         or affine fallback → matches original expressions (eyes, brows,
         mouth, smile, blink, head rotation, facial muscles)
      6. Semantic face masking — BiSeNet parsing / landmark fallback →
         build swap mask (skin, eyes, brows, nose, lips only) →
         preserve hair, beard, neck, ears, glasses →
         color correction (LAB) + lighting correction (Y-channel) →
         edge feathering → alpha blend
      7. Adaptive enhancement (GFPGAN / CodeFormer, optional)
      8. Encode JPEG

    Supports multiple faces and no-face pass-through.
    Returns a FrameResult with JPEG bytes + full diagnostics including
    tracking confidence, tracking state, track IDs, and head poses.
    """
    settings = get_settings()
    t_start = time.perf_counter()

    # ── 1 · Decode ──────────────────────────────────────────
    frame = decode_jpeg(frame_bytes)
    if frame is None:
        return FrameResult(
            jpeg_bytes=frame_bytes,
            inference_time_ms=0,
            face_count=0,
            detection_confidence=0,
            pass_through=True,
        )

    # ── 2 · Face Tracking (detection or optical-flow tracking) ──
    faces, tracking = face_tracker.update(frame)
    logger.info("Face detected: %d face(s), tracking=%s", len(faces), tracking["state"])

    if not faces:
        # No face — pass through original to keep stream smooth
        result_bytes = encode_jpeg(frame, settings.jpeg_quality)
        return FrameResult(
            jpeg_bytes=result_bytes,
            inference_time_ms=(time.perf_counter() - t_start) * 1000,
            face_count=0,
            detection_confidence=0,
            pass_through=True,
            tracking_confidence=tracking["confidence"],
            tracking_state=tracking["state"],
        )

    face_scores = [f.confidence for f in faces]
    best_confidence = max(face_scores)

    # ── 3 · Face Alignment ──────────────────────────────────
    # Pre-compute alignment matrices for logging / diagnostics.
    # InSwapper uses the same kps landmarks internally for its
    # affine warp, so we pass the full frame + face object.
    for face in faces:
        aligned, _ = align_face(frame, face.kps, size=128)
        if aligned is None:
            logger.warning("Alignment failed for face score %.3f", face.det_score)

    # ── 4 · InSwapper 128 + 5 · Expression Preservation + 6 · Masking ──
    # Swap → preserve original expressions → semantic masking for
    # hair/beard/neck/ear/glasses preservation + seamless blending.
    result = frame
    masking_infos = []
    expression_infos = []
    for face in faces:
        swapped = model_manager.swapper.get(
            result, face, source_face, paste_back=True
        )
        logger.info("Face swapped: track=%s, score=%.3f", getattr(face, 'track_id', '?'), float(face.det_score))
        expression_corrected, expr_info = expression_manager.preserve_expression(
            frame, swapped, face
        )
        result, mask_info = face_masking_manager.blend_face(
            result, expression_corrected, face
        )
        masking_infos.append(mask_info)
        expression_infos.append(expr_info)

    # ── 5 · Adaptive Enhancement (GFPGAN / CodeFormer) ──────
    # Assess quality and enhance only when needed. Mode is
    # switchable at runtime via enhancement_manager.set_mode().
    face_crop = None
    if faces:
        # Use the first face's bbox region for quality assessment
        bbox = faces[0].bbox.astype(int)
        h, w = result.shape[:2]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        if x2 > x1 and y2 > y1:
            face_crop = result[y1:y2, x1:x2]

    result, enh_info = enhancement_manager.enhance(result, face_crop)
    enhanced = enh_info["enhanced"]

    # ── 7 · Encode JPEG ─────────────────────────────────────
    result_bytes = encode_jpeg(result, settings.jpeg_quality)
    logger.info("Frame encoded: %d bytes, total inference=%.1fms", len(result_bytes), (time.perf_counter() - t_start) * 1000)
    inference_ms = (time.perf_counter() - t_start) * 1000

    # Track inference speed on the GPU manager
    gpu_manager.record_inference(inference_ms)

    return FrameResult(
        jpeg_bytes=result_bytes,
        inference_time_ms=inference_ms,
        face_count=len(faces),
        detection_confidence=best_confidence,
        face_scores=face_scores,
        enhanced=enhanced,
        tracking_confidence=tracking["confidence"],
        tracking_state=tracking["state"],
        track_ids=[f.track_id for f in faces],
        head_poses=[f.head_pose for f in faces],
        enhancement_time_ms=enh_info["enhancement_time_ms"],
        enhancement_mode=enh_info["mode"],
        enhancement_quality=enh_info["quality"],
        enhancer_used=enh_info["enhancer"],
        enhancement_skipped=enh_info["skipped"],
        masking_applied=masking_infos[0].get("enabled", False) if masking_infos else False,
        masking_method=masking_infos[0].get("method", "none") if masking_infos else "none",
        masking_time_ms=sum(m.get("masking_time_ms", 0) for m in masking_infos),
        masking_color_corrected=masking_infos[0].get("color_corrected", False) if masking_infos else False,
        masking_lighting_corrected=masking_infos[0].get("lighting_corrected", False) if masking_infos else False,
        masking_occlusion_handled=masking_infos[0].get("occlusion_handled", False) if masking_infos else False,
        masking_feather_radius=masking_infos[0].get("feather_radius", 0) if masking_infos else 0,
        expression_preserved=expression_infos[0].get("enabled", False) if expression_infos else False,
        expression_method=expression_infos[0].get("method", "none") if expression_infos else "none",
        expression_time_ms=sum(e.get("time_ms", 0) for e in expression_infos),
        expression_landmark_count=expression_infos[0].get("landmark_count", 0) if expression_infos else 0,
        expression_warp_strength=expression_infos[0].get("warp_strength", 1.0) if expression_infos else 1.0,
    )