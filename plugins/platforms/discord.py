"""Discord platform plugin — bot voice channel bridge."""
from plugins.base import PlatformPlugin
from services.discord_adapter import DiscordAdapter


class DiscordPlugin(PlatformPlugin):
    name = "discord"
    display_name = "Discord"
    version = "1.0.0"
    description = "Discord bot voice channel integration for real-time voice changing"
    author = "FaceSwap AI"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def create_adapter(self) -> DiscordAdapter:
        return DiscordAdapter(self.config)


def create(settings=None):
    config = {}
    if settings and hasattr(settings, "platform_discord_token"):
        config["token"] = settings.platform_discord_token
    return DiscordPlugin(config=config)