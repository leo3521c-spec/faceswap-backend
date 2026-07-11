"""
Semantic Face Masking — seamless face blending with region preservation.

Pipeline
────────
  1. Face parsing (BiSeNet ONNX) or landmark-based fallback → semantic segmentation
  2. Build swap mask: skin, brows, eyes, nose, lips only
     — excludes hair, beard, neck, ears, glasses, clothing, hat
  3. Beard preservation (heuristic: darkness / texture check in chin region)
  4. Occlusion handling (parser excludes non-face pixels; edge-discontinuity check)
  5. Color correction (LAB color transfer: swapped → original)
  6. Lighting correction (Y-channel luminance matching)
  7. Edge feathering (Gaussian blur on mask boundaries)
  8. Alpha blend: result = original × (1−mask) + corrected_swapped × mask

BiSeNet 19-class CelebAMask-HQ mapping
  0=background  1=skin  2=l_brow  3=r_brow  4=l_eye  5=r_eye
  6=eye_glass   7=l_ear 8=r_ear   9=ear_ring 10=nose  11=mouth
  12=u_lip      13=l_lip 14=neck  15=necklace 16=cloth 17=hair 18=hat

Swap mask includes:     {1, 2, 3, 4, 5, 10, 11, 12, 13}  — inner face
Preserve (exclude):     {0, 6, 7, 8, 9, 14, 15, 16, 17, 18}
"""
from __future__ import annotations

import time
import cv2
import numpy as np
from collections import deque
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("face_masking")

# ── BiSeNet class IDs ────────────────────────────────────────

CLASS_BACKGROUND = 0
CLASS_SKIN = 1
CLASS_L_BROW = 2
CLASS_R_BROW = 3
CLASS_L_EYE = 4
CLASS_R_EYE = 5
CLASS_EYE_GLASS = 6
CLASS_L_EAR = 7
CLASS_R_EAR = 8
CLASS_EAR_RING = 9
CLASS_NOSE = 10
CLASS_MOUTH = 11
CLASS_U_LIP = 12
CLASS_L_LIP = 13
CLASS_NECK = 14
CLASS_NECKLACE = 15
CLASS_CLOTH = 16
CLASS_HAIR = 17
CLASS_HAT = 18

# Classes included in the swap mask (inner face only)
SWAP_CLASSES = frozenset({
    CLASS_SKIN, CLASS_L_BROW, CLASS_R_BROW,
    CLASS_L_EYE, CLASS_R_EYE,
    CLASS_NOSE, CLASS_MOUTH, CLASS_U_LIP, CLASS_L_LIP,
})

# ImageNet normalization (standard for BiSeNet face parsing)
_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Manager ──────────────────────────────────────────────────


class FaceMaskingManager:
    """
    Semantic face masking with runtime configuration.

    Stateles across frames except for:
      • configuration (switchable at runtime)
      • rolling metrics
    """

    def __init__(self) -> None:
        self._enabled: bool = True
        self._feather_radius: int = 15
        self._color_correction: bool = True
        self._lighting_correction: bool = True
        self._occlusion_handling: bool = True
        self._parser = None  # ONNX InferenceSession
        self._parser_size: int = 512

        # Metrics
        self._mask_times: deque = deque(maxlen=120)
        self._fps_times: deque = deque(maxlen=120)
        self._total_blended: int = 0
        self._total_skipped: int = 0
        self._parser_used: int = 0
        self._landmark_used: int = 0

    # ── Configuration ───────────────────────────────────────

    def configure(
        self,
        enabled: Optional[bool] = None,
        feather_radius: Optional[int] = None,
        color_correction: Optional[bool] = None,
        lighting_correction: Optional[bool] = None,
        occlusion_handling: Optional[bool] = None,
        parser_size: Optional[int] = None,
    ) -> None:
        if enabled is not None:
            self._enabled = enabled
        if feather_radius is not None:
            self._feather_radius = max(1, min(60, int(feather_radius)))
        if color_correction is not None:
            self._color_correction = color_correction
        if lighting_correction is not None:
            self._lighting_correction = lighting_correction
        if occlusion_handling is not None:
            self._occlusion_handling = occlusion_handling
        if parser_size is not None:
            self._parser_size = int(parser_size)

    def set_parser(self, session) -> None:
        """Attach an ONNX InferenceSession for BiSeNet face parsing."""
        self._parser = session
        logger.info("Face parser session attached")

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def set_feather_radius(self, radius: int) -> None:
        self._feather_radius = max(1, min(60, int(radius)))

    def set_color_correction(self, enabled: bool) -> None:
        self._color_correction = enabled

    def set_lighting_correction(self, enabled: bool) -> None:
        self._lighting_correction = enabled

    def set_occlusion_handling(self, enabled: bool) -> None:
        self._occlusion_handling = enabled

    @property
    def parser_available(self) -> bool:
        return self._parser is not None

    # ── Public API ──────────────────────────────────────────

    def blend_face(
        self,
        original: np.ndarray,
        swapped: np.ndarray,
        face,
    ) -> tuple[np.ndarray, dict]:
        """
        Blend a swapped face into the original frame using semantic masking.

        Args:
            original: Original BGR frame (pre-swap).
            swapped:  Swapped BGR frame (same dimensions, post InSwapper paste_back).
            face:     InsightFace face object with .bbox and .kps.

        Returns:
            (blended_frame, info_dict)
        """
        if not self._enabled:
            self._total_skipped += 1
            return swapped, self._info(skipped=True)

        t_start = time.perf_counter()
        h, w = original.shape[:2]
        bbox = face.bbox.astype(int)

        # ── Define ROI with padding for context ────────────
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        pad = int(max(bw, bh) * 0.3)
        px1 = max(0, bbox[0] - pad)
        py1 = max(0, bbox[1] - pad)
        px2 = min(w, bbox[2] + pad)
        py2 = min(h, bbox[3] + pad)

        if px2 <= px1 or py2 <= py1:
            self._total_skipped += 1
            return swapped, self._info(skipped=True)

        roi_orig = original[py1:py2, px1:px2]
        roi_swap = swapped[py1:py2, px1:px2]

        # ── 1 · Build semantic mask ────────────────────────
        if self._parser is not None:
            mask = self._build_parser_mask(roi_orig)
            method = "bisenet"
            self._parser_used += 1
        else:
            mask = self._build_landmark_mask(roi_orig, face, px1, py1)
            method = "landmark"
            self._landmark_used += 1

        # ── 2 · Beard preservation ─────────────────────────
        mask = self._preserve_beard(roi_orig, mask, face, px1, py1)

        # ── 3 · Morphological cleanup ──────────────────────
        mask = self._clean_mask(mask)

        # ── 4 · Occlusion handling ─────────────────────────
        if self._occlusion_handling:
            mask = self._handle_occlusion(roi_orig, roi_swap, mask)

        # ── 5 · Color correction ───────────────────────────
        color_corrected = False
        if self._color_correction:
            roi_swap = self._color_correct(roi_orig, roi_swap, mask)
            color_corrected = True

        # ── 6 · Lighting correction ────────────────────────
        lighting_corrected = False
        if self._lighting_correction:
            roi_swap = self._lighting_correct(roi_orig, roi_swap, mask)
            lighting_corrected = True

        # ── 7 · Edge feathering ────────────────────────────
        mask = self._feather(mask)

        # ── 8 · Alpha blend ────────────────────────────────
        mask_f = (mask.astype(np.float32) / 255.0)[..., np.newaxis]
        blended_roi = (
            roi_orig.astype(np.float32) * (1.0 - mask_f)
            + roi_swap.astype(np.float32) * mask_f
        )

        result = original.copy()
        result[py1:py2, px1:px2] = blended_roi.astype(np.uint8)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._mask_times.append(elapsed_ms)
        self._fps_times.append(time.time())
        self._total_blended += 1

        info = self._info(
            skipped=False,
            method=method,
            time_ms=elapsed_ms,
            color_corrected=color_corrected,
            lighting_corrected=lighting_corrected,
            occlusion_handled=self._occlusion_handling,
            roi=(px1, py1, px2, py2),
        )
        return result, info

    # ── Mask builders ───────────────────────────────────────

    def _build_parser_mask(self, roi: np.ndarray) -> np.ndarray:
        """Build a swap mask using BiSeNet face parsing (ONNX)."""
        size = self._parser_size
        h, w = roi.shape[:2]

        # Preprocess: resize → RGB → normalize → NCHW
        resized = cv2.resize(roi, (size, size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        img = (img - _NORM_MEAN) / _NORM_STD
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # 1×3×H×W

        # Inference
        input_name = self._parser.get_inputs()[0].name
        output = self._parser.run(None, {input_name: img})[0]

        # Postprocess: argmax → label map (H×W)
        if output.ndim == 4:
            label_map = np.argmax(output[0], axis=0)
        elif output.ndim == 3:
            label_map = np.argmax(output[0], axis=0)
        else:
            label_map = output.squeeze()

        # Build swap mask from inner-face classes
        mask = np.zeros_like(label_map, dtype=np.uint8)
        for cls in SWAP_CLASSES:
            mask[label_map == cls] = 255

        # Resize back to ROI size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def _build_landmark_mask(
        self, roi: np.ndarray, face, off_x: int, off_y: int
    ) -> np.ndarray:
        """
        Landmark-based fallback when no face parsing model is available.
        Constructs an inner-face ellipse from the 5-point kps.
        """
        h, w = roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        kps = face.kps.copy().astype(np.float32)
        kps[:, 0] -= off_x
        kps[:, 1] -= off_y

        # kps: [left_eye, right_eye, nose, left_mouth, right_mouth]
        l_eye, r_eye, nose, l_mouth, r_mouth = kps

        eye_dist = float(np.linalg.norm(r_eye - l_eye))
        mouth_dist = float(np.linalg.norm(r_mouth - l_mouth))
        face_width = max(eye_dist, mouth_dist) * 1.8

        # Vertical extent: from above brows to below lower lip
        eye_cy = (l_eye[1] + r_eye[1]) / 2
        top = eye_cy - eye_dist * 0.9
        mouth_cy = (l_mouth[1] + r_mouth[1]) / 2
        bottom = mouth_cy + eye_dist * 0.7
        face_height = bottom - top
        center_y = (top + bottom) / 2
        center_x = (l_eye[0] + r_eye[0]) / 2

        # Ellipse covering inner face (narrower to exclude ears)
        ellipse_w = int(face_width * 0.85)
        ellipse_h = int(face_height * 1.0)
        if ellipse_w < 5 or ellipse_h < 5:
            return mask

        cv2.ellipse(
            mask,
            (int(center_x), int(center_y)),
            (ellipse_w // 2, ellipse_h // 2),
            0, 0, 360, 255, -1,
        )

        # Exclude forehead (preserve hair)
        brow_y = int(eye_cy - eye_dist * 0.3)
        mask[: max(0, brow_y - 5), :] = 0

        # Exclude below chin (preserve neck / beard)
        chin_y = int(bottom + eye_dist * 0.35)
        if chin_y < h:
            mask[chin_y:, :] = 0

        return mask

    # ── Preservation heuristics ────────────────────────────

    def _preserve_beard(
        self, roi: np.ndarray, mask: np.ndarray, face, off_x: int, off_y: int
    ) -> np.ndarray:
        """
        Detect beard in the chin region and exclude it from the swap mask.
        Uses brightness + texture variance to identify facial hair.
        """
        h, w = roi.shape[:2]
        kps = face.kps.copy().astype(np.float32)
        kps[:, 0] -= off_x
        kps[:, 1] -= off_y

        l_mouth, r_mouth = kps[3], kps[4]
        mouth_cx = (l_mouth[0] + r_mouth[0]) / 2
        mouth_cy = (l_mouth[1] + r_mouth[1]) / 2
        mouth_w = float(np.linalg.norm(r_mouth - l_mouth))

        # Chin region: below the mouth, spanning the jaw
        cy1 = int(mouth_cy + mouth_w * 0.35)
        cy2 = int(mouth_cy + mouth_w * 1.6)
        cx1 = int(mouth_cx - mouth_w * 0.95)
        cx2 = int(mouth_cx + mouth_w * 0.95)

        cy1 = max(0, min(h, cy1))
        cy2 = max(0, min(h, cy2))
        cx1 = max(0, min(w, cx1))
        cx2 = max(0, min(w, cx2))

        if cy2 <= cy1 or cx2 <= cx1:
            return mask

        chin_region = roi[cy1:cy2, cx1:cx2]
        if chin_region.size == 0:
            return mask

        gray = cv2.cvtColor(chin_region, cv2.COLOR_BGR2GRAY)

        # Beard detection: darker than face mean + high local variance
        face_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        face_mean = float(np.mean(face_gray))
        chin_mean = float(np.mean(gray))

        # Laplacian variance = texture roughness (beards are rough)
        chin_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        is_beard = chin_mean < face_mean - 20 or chin_var > 200
        if is_beard:
            mask[cy1:cy2, cx1:cx2] = 0

        return mask

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Morphological close + open to fill holes and remove noise."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def _handle_occlusion(
        self, original: np.ndarray, swapped: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Detect occluding objects (hands, phones) within the face region
        and reduce the mask weight there.

        Strategy: the face parser already excludes non-face pixels. For the
        landmark fallback, we additionally check for sharp depth-like
        discontinuities in the original frame that don't correspond to face
        features, and erode the mask at those boundaries.
        """
        if self._parser is not None:
            # Parser-based segmentation already handles occlusion
            return mask

        # Landmark fallback: detect strong edges that might indicate occluders
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)

        # Dilate edges to create exclusion zones
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edge_zones = cv2.dilate(edges, kernel, iterations=1)

        # Reduce mask weight at edge zones (but not too aggressively)
        edge_mask = (edge_zones > 0).astype(np.uint8) * 255
        # Only reduce where both mask and edges are present
        overlap = cv2.bitwise_and(mask, edge_mask)
        # Erode the overlap regions from the mask
        reduction = cv2.dilate(overlap, kernel, iterations=1)
        mask = cv2.subtract(mask, reduction // 2)

        return mask

    # ── Color / lighting correction ─────────────────────────

    def _color_correct(
        self, original: np.ndarray, swapped: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        LAB color transfer: match the swapped face's color statistics
        to the original frame within the mask region.
        """
        mask_bool = mask > 127
        if mask_bool.sum() < 20:
            return swapped

        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        swap_lab = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
        result = swap_lab.copy()

        for c in range(3):
            o_vals = orig_lab[..., c][mask_bool]
            s_vals = swap_lab[..., c][mask_bool]
            o_mean, o_std = float(o_vals.mean()), float(o_vals.std())
            s_mean, s_std = float(s_vals.mean()), float(s_vals.std())

            if s_std > 0.5:
                result[..., c] = (
                    (swap_lab[..., c] - s_mean) / s_std * o_std + o_mean
                )
            else:
                result[..., c] = swap_lab[..., c] - s_mean + o_mean

        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def _lighting_correct(
        self, original: np.ndarray, swapped: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Match the luminance (Y channel) of the swapped face to the
        original frame, correcting for lighting differences.
        """
        mask_bool = mask > 127
        if mask_bool.sum() < 20:
            return swapped

        orig_ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
        swap_ycrcb = cv2.cvtColor(swapped, cv2.COLOR_BGR2YCrCb)

        orig_y = orig_ycrcb[..., 0].astype(np.float32)
        swap_y = swap_ycrcb[..., 0].astype(np.float32)

        o_vals = orig_y[mask_bool]
        s_vals = swap_y[mask_bool]
        o_mean, o_std = float(o_vals.mean()), float(o_vals.std())
        s_mean, s_std = float(s_vals.mean()), float(s_vals.std())

        if s_std > 0.5:
            corrected_y = (swap_y - s_mean) / s_std * o_std + o_mean
        else:
            corrected_y = swap_y - s_mean + o_mean

        corrected_y = np.clip(corrected_y, 0, 255).astype(np.uint8)
        swap_ycrcb[..., 0] = corrected_y
        return cv2.cvtColor(swap_ycrcb, cv2.COLOR_YCrCb2BGR)

    # ── Edge feathering ─────────────────────────────────────

    def _feather(self, mask: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to feather mask edges for seamless blending."""
        radius = self._feather_radius
        if radius < 2:
            return mask
        # Gaussian blur with kernel size = 2*radius+1 (must be odd)
        ksize = radius * 2 + 1
        return cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # ── Info / metrics ──────────────────────────────────────

    def _info(
        self,
        skipped: bool = False,
        method: str = "none",
        time_ms: float = 0.0,
        color_corrected: bool = False,
        lighting_corrected: bool = False,
        occlusion_handled: bool = False,
        roi=None,
    ) -> dict:
        return {
            "enabled": self._enabled and not skipped,
            "method": method,
            "masking_time_ms": round(time_ms, 2),
            "color_corrected": color_corrected,
            "lighting_corrected": lighting_corrected,
            "occlusion_handled": occlusion_handled,
            "feather_radius": self._feather_radius if not skipped else 0,
            "skipped": skipped,
            "roi": list(roi) if roi else None,
        }

    def get_status(self) -> dict:
        avg_time = (
            sum(self._mask_times) / len(self._mask_times)
            if self._mask_times
            else 0.0
        )
        fps = 0.0
        if len(self._fps_times) >= 2:
            elapsed = self._fps_times[-1] - self._fps_times[0]
            if elapsed > 0:
                fps = len(self._fps_times) / elapsed

        return {
            "enabled": self._enabled,
            "parser_available": self.parser_available,
            "parser_size": self._parser_size,
            "feather_radius": self._feather_radius,
            "color_correction": self._color_correction,
            "lighting_correction": self._lighting_correction,
            "occlusion_handling": self._occlusion_handling,
            "avg_masking_time_ms": round(avg_time, 2),
            "fps": round(fps, 2),
            "total_blended": self._total_blended,
            "total_skipped": self._total_skipped,
            "parser_used_count": self._parser_used,
            "landmark_used_count": self._landmark_used,
            "preservation_targets": [
                "hair", "beard", "neck", "ears", "glasses",
            ],
        }

    def reset_metrics(self) -> None:
        self._mask_times.clear()
        self._fps_times.clear()
        self._total_blended = 0
        self._total_skipped = 0
        self._parser_used = 0
        self._landmark_used = 0


# ── Singleton ────────────────────────────────────────────────

face_masking_manager = FaceMaskingManager()