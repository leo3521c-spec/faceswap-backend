"""Google Meet platform plugin — virtual audio device bridge."""
from plugins.base import PlatformPlugin
from services.google_meet_adapter import GoogleMeetAdapter


class GoogleMeetPlugin(PlatformPlugin):
    name = "google_meet"
    display_name = "Google Meet"
    version = "1.0.0"
    description = "Virtual audio device bridge for Google Meet voice changing"
    author = "FaceSwap AI"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def create_adapter(self) -> GoogleMeetAdapter:
        return GoogleMeetAdapter(self.config)


def create(settings=None):
    return GoogleMeetPlugin()