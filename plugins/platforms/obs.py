"""OBS Studio platform plugin — virtual camera + audio cable."""
from plugins.base import PlatformPlugin
from services.obs_adapter import OBSAudioAdapter


class OBSPlugin(PlatformPlugin):
    name = "obs"
    display_name = "OBS Studio"
    version = "1.0.0"
    description = "Virtual camera output + audio cable bridge for OBS Studio"
    author = "FaceSwap AI"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def create_adapter(self) -> OBSAudioAdapter:
        return OBSAudioAdapter(self.config)


def create(settings=None):
    return OBSPlugin()