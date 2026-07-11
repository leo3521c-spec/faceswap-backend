"""
Central configuration via environment variables.
All settings use the FACESWAP_ prefix.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── Server ──────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_path: str = "/ws/swap"

    # ── Models ──────────────────────────────────────────────
    inswapper_model_path: str = "models/inswapper_128.onnx"
    detector_model: str = "buffalo_l"
    gfpgan_model_path: str = "models/GFPGANv1.4.pth"

    # ── Processing ──────────────────────────────────────────
    enable_enhancer: bool = False
    face_detection_size: int = 640
    swap_det_threshold: float = 0.5
    jpeg_quality: int = 85
    batch_size: int = 1

    # ── Enhancement ────────────────────────────────────────
    enhancement_mode: str = "balanced"  # off | fast | balanced | ultra
    enhancement_quality_high: float = 150.0  # Laplacian variance above → skip
    enhancement_quality_medium: float = 80.0  # above → light enhance

    # ── Semantic Face Masking ──────────────────────────────
    masking_enabled: bool = True
    masking_feather_radius: int = 15  # Gaussian blur radius for edge feathering
    masking_color_correction: bool = True  # LAB color transfer
    masking_lighting_correction: bool = True  # Y-channel luminance matching
    masking_occlusion_handling: bool = True  # detect & exclude occluders
    masking_face_parser_model_path: str = "models/face_parsing_512.onnx"
    masking_face_parser_size: int = 512  # BiSeNet input dimension

    # ── Expression Preservation ────────────────────────────
    expression_enabled: bool = False  # Re-runs face detection per frame (1200ms) — disabled for speed
    expression_warp_strength: float = 1.0  # 0-2, how strongly to match expressions
    expression_grid_size: int = 32  # dense flow grid resolution

    # ── WebRTC Video Transport ──────────────────────────────
    webrtc_enabled: bool = False  # use WebRTC instead of WebSocket for video
    webrtc_signaling_path: str = "/webrtc/offer"
    webrtc_ice_servers: list = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
    ]
    webrtc_turn_servers: list = []  # [{"urls":"turn:...","username":"...","credential":"..."}]

    # ── Virtual Camera (OBS) ──────────────────────────────
    vc_enabled: bool = False  # start disabled — enable at runtime
    vc_resolution: str = "720p"  # 720p | 1080p
    vc_fps: int = 30  # 24 | 30

    # ── Face Tracking ──────────────────────────────────────
    tracking_detection_interval: int = 5  # full detection every N frames
    tracking_max_missed: int = 10  # remove track after N consecutive misses
    tracking_iou_threshold: float = 0.3  # min IoU for track-detection match
    tracking_embedding_threshold: float = 0.5  # min cosine sim for match
    tracking_confidence_threshold: float = 0.3  # force re-detect below this
    tracking_enable_head_pose: bool = True

    # ── GPU / ONNX ──────────────────────────────────────────
    gpu_device_id: int = -1  # -1 = auto-select best GPU
    gpu_mem_limit_mb: int = 4096  # ONNX arena memory limit
    enable_tensorrt: bool = False  # TensorRT rebuilds engines on every call — use CUDA EP instead
    enable_fp16: bool = True  # Half-precision inference
    enable_cuda_graph: bool = True  # CUDA Graph capture for fixed shapes
    enable_pinned_memory: bool = True  # Page-locked memory for fast transfers
    cudnn_exhaustive: bool = False  # EXHAUSTIVE conv algo search
    trt_workspace_mb: int = 2048  # TensorRT max workspace size

    # ── Voice / Audio ──────────────────────────────────────
    voice_websocket_path: str = "/ws/voice"
    voice_enabled: bool = False
    voice_sample_rate: int = 24000  # Hz
    voice_chunk_duration_ms: int = 20  # 20 ms chunks
    voice_channels: int = 1  # mono
    voice_input_device: int = -1  # -1 = default

    # Noise suppression
    voice_noise_suppression: bool = True
    voice_noise_aggressiveness: int = 2  # 0-4

    # Voice conversion
    voice_conversion_enabled: bool = False
    voice_model_path: str = ""
    voice_pitch_shift: int = 0  # semitones (-12 to +12)

    # Echo cancellation
    voice_echo_cancellation: bool = True
    voice_echo_tail_length_ms: int = 128

    # Mute
    voice_muted: bool = False

    # ── Platform Integrations ──────────────────────────────
    platform_default_sample_rate: int = 48000
    platform_default_channels: int = 1

    # Zoom
    platform_zoom_enabled: bool = False
    platform_zoom_sdk_key: str = ""

    # Google Meet (virtual audio devices)
    platform_meet_enabled: bool = False

    # Discord
    platform_discord_enabled: bool = False
    platform_discord_token: str = ""

    # Telegram
    platform_telegram_enabled: bool = False
    platform_telegram_bot_token: str = ""

    # OBS (virtual audio cable)
    platform_obs_enabled: bool = False

    # WebRTC
    platform_webrtc_enabled: bool = False
    platform_webrtc_ice_servers: list = [
        {"urls": "stun:stun.l.google.com:19302"}
    ]

    # ── CORS ────────────────────────────────────────────────
    cors_origins: list[str] = ["*"]

    # ── Logging ─────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_prefix": "FACESWAP_",
        "env_file_encoding": "utf-8",
    }

    @property
    def onnx_providers(self) -> list:
        """Provider list is now managed by GPUManager — this property
        is kept for backward compatibility with code that reads it."""
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


@lru_cache
def get_settings() -> Settings:
    return Settings()