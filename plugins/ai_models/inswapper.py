"""
InSwapper 128 AI model plugin.

Wraps the existing model_manager's InSwapper ONNX model. This plugin
demonstrates the AIModelPlugin contract — future face swap models
(SimSwap, FaceDancer, etc.) can be added as separate plugin files
without touching face_processor.py.
"""
import numpy as np

from plugins.base import AIModelPlugin
from services.model_manager import model_manager


class InSwapperPlugin(AIModelPlugin):
    name = "inswapper_128"
    display_name = "InSwapper 128"
    version = "1.0.0"
    model_type = "face_swap"
    description = "InSwapper 128 ONNX face swap model (InsightFace)"
    author = "InsightFace"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def load_model(self, model_path: str) -> bool:
        """Model is loaded by model_manager at startup — just check status."""
        return model_manager.is_loaded

    def process(self, frame: np.ndarray, source_face, face=None) -> tuple:
        """Run face swap on the frame.

        Args:
            frame: BGR numpy array
            source_face: source face object (with embedding)
            face: target face detected in the frame

        Returns:
            (swapped_frame, metadata)
        """
        if face is None or source_face is None:
            return frame, {"swapped": False, "reason": "no_face"}

        if not self.is_loaded():
            return frame, {"swapped": False, "reason": "model_not_loaded"}

        result = model_manager.swapper.get(
            frame, face, source_face, paste_back=True
        )
        return result, {"swapped": True, "model": self.name}

    def is_loaded(self) -> bool:
        return (
            model_manager.is_loaded
            and model_manager.swapper is not None
        )

    def get_status(self) -> dict:
        return {
            "model_type": self.model_type,
            "loaded": self.is_loaded(),
            "model_path": getattr(model_manager, "_model_paths", {}).get(
                "swapper", "models/inswapper_128.onnx"
            ),
        }


def create(settings=None):
    return InSwapperPlugin()