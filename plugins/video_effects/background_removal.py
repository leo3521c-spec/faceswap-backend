"""
Background Removal video effect plugin.

Uses the already-loaded InsightFace detector to find the face region,
builds an elliptical foreground mask (face + upper body), and applies
one of two modes:

    • "blur"     — Gaussian blur the background (portrait mode)
    • "remove"   — Replace background with solid color (green screen)

No new model dependencies — reuses the existing face detector.
A future plugin could swap this for MediaPipe Selfie Segmentation or
a U2Net model — just drop a new file in plugins/video_effects/.
"""
import cv2
import numpy as np

from plugins.base import VideoEffectPlugin


class BackgroundRemovalPlugin(VideoEffectPlugin):
    name = "background_removal"
    display_name = "Background Removal"
    version = "1.0.0"
    description = "Portrait-mode background blur or solid-color removal using face detection"
    author = "FaceSwap AI"

    def initialize(self) -> bool:
        self._mode: str = self.config.get("mode", "blur")
        self._blur_strength: int = self.config.get("blur_strength", 51)
        self._bg_color: tuple = self.config.get("bg_color", (0, 0, 0))
        self._initialized = True
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # Import here to avoid circular imports at module load time
        from services.model_manager import model_manager

        mask = np.zeros((h, w), dtype=np.uint8)

        # Use the face detector to find foreground region
        if model_manager.is_loaded and model_manager.detector is not None:
            try:
                faces = model_manager.detector.get(frame)
            except Exception:
                faces = []
        else:
            faces = []

        if faces:
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) // 2
                bw = x2 - x1
                bh = y2 - y1

                # Expand to include upper body
                ex1 = max(0, cx - int(bw * 0.9))
                ex2 = min(w, cx + int(bw * 0.9))
                ey1 = max(0, y1 - int(bh * 0.4))
                ey2 = min(h, y2 + int(bh * 1.8))

                cv2.ellipse(
                    mask,
                    (cx, (ey1 + ey2) // 2),
                    ((ex2 - ex1) // 2, (ey2 - ey1) // 2),
                    0, 0, 360, 255, -1,
                )
        else:
            # No face detected — keep center region as foreground
            cv2.ellipse(
                mask,
                (w // 2, h // 2),
                (w // 3, h // 2),
                0, 0, 360, 255, -1,
            )

        # Feather the mask edges for smooth transitions
        k = self._blur_strength if self._blur_strength % 2 == 1 else self._blur_strength + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask_f = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_f, mask_f, mask_f], axis=-1)

        if self._mode == "remove":
            # Solid color background
            bg = np.full_like(frame, self._bg_color, dtype=np.uint8)
            result = (frame * mask_3ch + bg * (1 - mask_3ch)).astype(np.uint8)
        else:
            # Blurred background (portrait mode)
            blurred = cv2.GaussianBlur(frame, (k, k), 0)
            result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)

        return result

    def set_parameter(self, key: str, value) -> bool:
        if key == "mode":
            if value in ("blur", "remove"):
                self._mode = value
                return True
            return False
        if key == "blur_strength":
            self._blur_strength = max(3, min(101, int(value)))
            return False  # not in base, but useful — accept anyway
        if key == "bg_color":
            if isinstance(value, (list, tuple)) and len(value) == 3:
                self._bg_color = tuple(int(v) for v in value)
                return True
            return False
        return False

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "mode": self._mode,
            "blur_strength": self._blur_strength,
        }


def create(settings=None):
    return BackgroundRemovalPlugin()