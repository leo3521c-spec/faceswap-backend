"""
Minimal configuration for real-time face swap.
Pipeline: JPEG Decode → InsightFace Detection → InSwapper128 → JPEG Encode → Binary WS Response
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
    face_detection_size: int = 320  # smaller = faster detection
    swap_det_threshold: float = 0.5
    jpeg_quality: int = 70  # lower = smaller frames, faster transfer
    batch_size: int = 1

    # ── Enhancement (disabled by default — not in realtime pipeline) ──
    enable_enhancer: bool = False
    enhancement_mode: str = "off"

    # ── Face Tracking ──────────────────────────────────────
    tracking_detection_interval: int = 15  # full detection every N frames
    tracking_max_missed: int = 10
    tracking_iou_threshold: float = 0.3
    tracking_embedding_threshold: float = 0.5
    tracking_confidence_threshold: float = 0.3
    tracking_enable_head_pose: bool = False  # disabled — unnecessary compute

    # ── GPU / ONNX ──────────────────────────────────────────
    gpu_device_id: int = -1  # -1 = auto-select best GPU
    gpu_mem_limit_mb: int = 4096
    enable_tensorrt: bool = False  # TensorRT not installed on RunPod — use CUDA EP
    enable_fp16: bool = True
    enable_cuda_graph: bool = True
    enable_pinned_memory: bool = True
    cudnn_exhaustive: bool = False
    trt_workspace_mb: int = 4096
    trt_engine_cache_path: str = "trt_cache"  # persist TRT engines between runs

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
        """Provider chain: TensorRT → CUDA → CPU (no silent CPU fallback)."""
        return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


@lru_cache
def get_settings() -> Settings:
    return Settings()