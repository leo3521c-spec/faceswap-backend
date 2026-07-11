"""
Base classes for the plugin system.

Every plugin extends Plugin and picks a category-specific base.
The core engine interacts with plugins only through these interfaces —
never through concrete implementations.

Plugin lifecycle:
    create(settings) → Plugin instance
    registry.register(plugin) → plugin.initialize()
    ... app runs ...
    registry.shutdown_all() → plugin.shutdown()
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np

from services.platform_base import PlatformAdapter


class Plugin(ABC):
    """Root interface for all plugins.

    Subclasses set the class-level metadata (name, display_name, etc.)
    and implement initialize() + get_status().
    """

    # ── Metadata (set by subclasses) ──────────────────────
    name: str = ""
    display_name: str = ""
    category: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = config or {}
        self._initialized: bool = False

    # ── Lifecycle ─────────────────────────────────────────

    @abstractmethod
    def initialize(self) -> bool:
        """Called once after construction. Return True if ready."""
        ...

    @abstractmethod
    def get_status(self) -> dict:
        """Return plugin status for /plugins endpoint."""
        ...

    def shutdown(self) -> None:
        """Cleanup on app shutdown. Override if resources need releasing."""
        pass

    @property
    def initialized(self) -> bool:
        return self._initialized

    def to_dict(self) -> dict:
        """Serializable metadata + status for API responses."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "initialized": self._initialized,
            "status": self.get_status(),
        }


# ── Platform Plugin ──────────────────────────────────────────

class PlatformPlugin(Plugin):
    """Plugin for communication platforms (OBS, Zoom, Discord, etc.).

    Each platform plugin wraps or extends a PlatformAdapter, providing
    the connect/disconnect/stream contract plus plugin metadata.

    The registry creates the adapter and registers it with
    PlatformManager during startup — see get_adapter().
    """

    category = "platform"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._adapter: Optional[PlatformAdapter] = None

    @abstractmethod
    def create_adapter(self) -> PlatformAdapter:
        """Return a PlatformAdapter instance for this platform.

        Called by the registry to register with PlatformManager.
        """
        ...

    def get_adapter(self) -> Optional[PlatformAdapter]:
        """Return the adapter (creates it if not yet created)."""
        if self._adapter is None:
            self._adapter = self.create_adapter()
        return self._adapter

    def get_status(self) -> dict:
        if self._adapter:
            return self._adapter.get_status()
        return {"initialized": self._initialized, "connected": False}

    def shutdown(self) -> None:
        if self._adapter and self._adapter.connected:
            self._adapter.disconnect()


# ── AI Model Plugin ──────────────────────────────────────────

class AIModelPlugin(Plugin):
    """Plugin for AI inference models (face swap, enhancers, future models).

    The face processing pipeline can query available AI model plugins
    and delegate inference to them — allowing new models to be added
    without modifying face_processor.py.
    """

    category = "ai_model"
    model_type: str = ""  # "face_swap", "enhancer", "super_res", etc.

    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load the model from the given path. Return True on success."""
        ...

    @abstractmethod
    def process(
        self, frame: np.ndarray, source_face: Any, face: Any = None
    ) -> tuple[np.ndarray, dict]:
        """Run inference on a frame.

        Args:
            frame: BGR numpy array (the input frame)
            source_face: source face embedding/object
            face: target face detected in the frame (optional)

        Returns:
            (result_frame, metadata_dict)
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """True if the model is loaded and ready for inference."""
        ...

    def get_status(self) -> dict:
        return {
            "model_type": self.model_type,
            "loaded": self.is_loaded(),
        }


# ── Voice Effect Plugin ──────────────────────────────────────

class VoiceEffectPlugin(Plugin):
    """Plugin for voice changing / audio effects.

    The voice processor can chain VoiceEffectPlugins, allowing new
    voice effects to be added without modifying voice_processor.py.
    """

    category = "voice_effect"

    @abstractmethod
    def process_audio(
        self, pcm_data: bytes, sample_rate: int, channels: int
    ) -> bytes:
        """Process a PCM audio chunk. Return processed PCM bytes."""
        ...

    @abstractmethod
    def set_parameter(self, key: str, value: Any) -> bool:
        """Set a runtime parameter (pitch, intensity, etc.).

        Return True if the parameter was accepted.
        """
        ...

    def get_status(self) -> dict:
        return {"initialized": self._initialized}


# ── Video Effect Plugin ──────────────────────────────────────

class VideoEffectPlugin(Plugin):
    """Plugin for video effects (background removal, blur, filters).

    The face processing pipeline can apply VideoEffectPlugins after
    the face swap step, allowing new visual effects without modifying
    face_processor.py.
    """

    category = "video_effect"

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply the effect to a BGR frame. Return the processed frame."""
        ...

    def get_status(self) -> dict:
        return {"initialized": self._initialized}