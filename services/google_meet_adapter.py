"""
Google Meet integration.

Architecture:
  • No official audio SDK — Google Meet runs in the browser
  • Approach: virtual audio device (VB-Cable / BlackHole / pulse null sink)
    • Processed audio → virtual output device → Meet selects it as mic
    • Meet's speaker output → virtual input device → captured as source
  • Alternatively: Chrome Extension API for tab audio capture
  • This adapter manages the virtual audio device bridge

Virtual device tools:
  • Linux: PulseAudio null sink/module-loopback
  • macOS: BlackHole
  • Windows: VB-Audio Virtual Cable
"""
from __future__ import annotations

from services.platform_base import PlatformAdapter


class GoogleMeetAdapter(PlatformAdapter):
    platform = "google_meet"
    display_name = "Google Meet"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._virtual_output_device: str = ""
        self._virtual_input_device: str = ""

    def connect(self, **kwargs) -> dict:
        """Set up virtual audio devices for Meet bridging.

        kwargs: output_device, input_device
        TODO: create/verify virtual audio devices.
        """
        self._virtual_output_device = kwargs.get(
            "output_device", "virtual_mic"
        )
        self._virtual_input_device = kwargs.get(
            "input_device", "virtual_speaker"
        )
        self._connected = True
        return self.get_status()

    def disconnect(self) -> dict:
        """Release virtual audio devices."""
        self._connected = False
        self._streaming = False
        return self.get_status()

    def start_stream(self) -> dict:
        """Start capturing from virtual input and writing to virtual output.

        TODO: open sounddevice streams on the virtual devices.
        Capture callback should call self._on_incoming_audio(pcm, sr, ch).
        """
        self._streaming = True
        return self.get_status()

    def stop_stream(self) -> dict:
        self._streaming = False
        return self.get_status()

    def send_audio(self, pcm_data: bytes) -> None:
        """Write processed PCM to the virtual output device (Meet's mic).

        TODO: write to sounddevice OutputStream on virtual_output_device.
        """
        pass

    def get_status(self) -> dict:
        return {
            "platform": self.platform,
            "connected": self._connected,
            "streaming": self._streaming,
            "virtual_output_device": self._virtual_output_device,
            "virtual_input_device": self._virtual_input_device,
        }