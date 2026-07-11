"""
Plugin architecture for FaceSwap AI.

Drop-in plugins — no core engine modification required.

Categories:
    • platform      — OBS, Zoom, Google Meet, Discord, Telegram
    • ai_model      — face swap models, future AI models
    • voice_effect  — voice changers, audio effects
    • video_effect  — background removal, blur, filters

Adding a new plugin:
    1. Create a .py file in the appropriate plugins/<category>/ directory
    2. Define a class extending the base for that category
    3. Export a create(settings=None) factory function
    4. Done — auto-discovery handles registration on startup

The core engine (face_processor, voice_processor, main pipeline) never
imports plugin code directly — it queries the registry.
"""
from plugins.base import (
    Plugin,
    PlatformPlugin,
    AIModelPlugin,
    VoiceEffectPlugin,
    VideoEffectPlugin,
)
from plugins.registry import plugin_registry

__all__ = [
    "Plugin",
    "PlatformPlugin",
    "AIModelPlugin",
    "VoiceEffectPlugin",
    "VideoEffectPlugin",
    "plugin_registry",
]