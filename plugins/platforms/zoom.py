"""Zoom platform plugin — Meeting SDK audio bridge."""
from plugins.base import PlatformPlugin
from services.zoom_adapter import ZoomAdapter


class ZoomPlugin(PlatformPlugin):
    name = "zoom"
    display_name = "Zoom"
    version = "1.0.0"
    description = "Zoom Meeting SDK integration for real-time voice changing in meetings"
    author = "FaceSwap AI"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def create_adapter(self) -> ZoomAdapter:
        return ZoomAdapter(self.config)


def create(settings=None):
    config = {}
    if settings and hasattr(settings, "platform_zoom_sdk_key"):
        config["sdk_key"] = settings.platform_zoom_sdk_key
    return ZoomPlugin(config=config)