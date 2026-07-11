# Plugin Architecture

Drop-in plugins for FaceSwap AI — **no core engine modification required**.

## How It Works

```
plugins/
├── base.py              ← Abstract base classes (the plugin contract)
├── registry.py          ← Auto-discovery + central registry
├── platforms/           ← OBS, Zoom, Google Meet, Discord, Telegram
├── ai_models/           ← Face swap models, future AI models
├── voice_effects/       ← Voice changers, audio effects
└── video_effects/       ← Background removal, blur, filters
```

On startup, `plugin_registry.discover()` scans each subdirectory, imports
every `.py` file, calls its `create(settings)` factory, and registers
the plugin. Platform plugins' adapters are then registered with
`PlatformManager` automatically.

## Adding a New Plugin

1. **Pick the right subdirectory** (e.g. `plugins/video_effects/`).
2. **Create a `.py` file** (e.g. `my_effect.py`).
3. **Define a class** extending the appropriate base:

```python
from plugins.base import VideoEffectPlugin
import numpy as np

class MyEffect(VideoEffectPlugin):
    name = "my_effect"
    display_name = "My Effect"
    description = "Does something cool to video frames"

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Your processing logic here
        return frame

def create(settings=None):
    return MyEffect()
```

4. **Done.** The plugin auto-registers on next startup.

## Plugin Categories

### Platform (`plugins/platforms/`)
Implements `PlatformPlugin` — wraps a `PlatformAdapter` for audio/video
routing to external services. Each plugin's `create_adapter()` returns
an adapter that gets registered with `PlatformManager`.

### AI Model (`plugins/ai_models/`)
Implements `AIModelPlugin` — face swap models, enhancers, future models.
Provides `load_model()`, `process(frame, source_face, face)`, `is_loaded()`.

### Voice Effect (`plugins/voice_effects/`)
Implements `VoiceEffectPlugin` — voice changers, pitch shifters.
Provides `process_audio(pcm, sample_rate, channels)` and `set_parameter()`.

### Video Effect (`plugins/video_effects/`)
Implements `VideoEffectPlugin` — background removal, blur, filters.
Provides `process_frame(frame) -> frame`.

## What Never Changes

Adding a plugin requires **zero modifications** to:
- `face_processor.py` — the AI inference pipeline
- `voice_processor.py` — the voice processing chain
- `audio_pipeline.py` / `frame_queue.py` — the threading pipeline
- `main.py` — only calls `discover()` + `register_platforms()`, never per-plugin