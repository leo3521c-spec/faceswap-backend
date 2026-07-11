"""Telegram platform plugin — bot voice call bridge."""
from plugins.base import PlatformPlugin
from services.telegram_adapter import TelegramAdapter


class TelegramPlugin(PlatformPlugin):
    name = "telegram"
    display_name = "Telegram"
    version = "1.0.0"
    description = "Telegram bot voice call integration for real-time voice changing"
    author = "FaceSwap AI"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def create_adapter(self) -> TelegramAdapter:
        return TelegramAdapter(self.config)


def create(settings=None):
    config = {}
    if settings and hasattr(settings, "platform_telegram_bot_token"):
        config["bot_token"] = settings.platform_telegram_bot_token
    return TelegramPlugin(config=config)