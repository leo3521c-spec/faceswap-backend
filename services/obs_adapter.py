"""
OBS Studio integration via virtual audio cable.

Architecture:
  • Processed audio is written to a virtual audio output device
  • OBS captures from that virtual device as a mic/aux source
  • Source audio (what OBS hears) is captured from a virtual input
    device or system loopback
  • Complements the existing virtual_camera.py (which handles video)

Virtual device tools:
  • Linux: PulseAudio null sink
  • macOS: BlackHole
  • Windows: VB-Audio Virtual Cable

No OBS API needed — OBS simply captures from the virtual device.
"""
from __future__ import annotations

from services.platform_base import PlatformAdapter


class OBSAudioAdapter(PlatformAdapter):
    platform = "obs"
    display_name = "OBS Studio"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._output_device: str = ""
        self._input_device: str = ""
        self._sample_rate: int = 48000

    def connect(self, **kwargs) -> dict:
        """Configure virtual audio devices for OBS bridging.

        kwargs: output_device, input_device, sample_rate
        TODO: verify devices exist via sounddevice.query_devices().
        """
        self._output_device = kwargs.get(
            "output_device", "virtual_mic"
        )
        self._input_device = kwargs.get(
            "input_device", "virtual_speaker"
        )
        self._sample_rate = kwargs.get("sample_rate", 48000)
        self._connected = True
        return self.get_status()

    def disconnect(self) -> dict:
        self._connected = False
        self._streaming = False
        return self.get_status()

    def start_stream(self) -> dict:
        """Start capture from input device and playback to output device.

        TODO: open sounddevice InputStream on input_device (capture)
        and OutputStream on output_device (playback).
        Capture callback → self._on_incoming_audio(pcm, sr, ch).
        """
        self._streaming = True
        return self.get_status()

    def stop_stream(self) -> dict:
        self._streaming = False
        return self.get_status()

    def send_audio(self, pcm_data: bytes) -> None:
        """Write processed PCM to the virtual output device.

        OBS picks this up as a mic/aux source.
        TODO: write to sounddevice OutputStream.
        """
        pass

    def get_status(self) -> dict:
        return {
            "platform": self.platform,
            "connected": self._connected,
            "streaming": self._streaming,
            "output_device": self._output_device,
            "input_device": self._input_device,
            "sample_rate": self._sample_rate,
        }