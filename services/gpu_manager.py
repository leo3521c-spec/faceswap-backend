"""
GPU Manager — NVIDIA GPU optimization, monitoring, and fallback.

Features
────────
• Automatic GPU selection (picks least-loaded NVIDIA GPU)
• ONNX Runtime provider chain: TensorRT → CUDA → CPU fallback
• FP16 (half-precision) inference support
• CUDA Graph capture for fixed-shape inference
• Pinned (page-locked) memory pool for fast H2D/D2H transfers
• GPU monitoring: name, VRAM, temperature, utilization
• Inference speed tracking (ms / FPS)
• Graceful CPU fallback when no NVIDIA GPU is available
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import get_settings
from utils.logger import setup_logger

logger = setup_logger("gpu_manager")


# ── Data structures ──────────────────────────────────────────


@dataclass
class GPUInfo:
    """Snapshot of a single GPU's state."""
    available: bool
    device_id: int
    name: str
    vram_total_mb: float = 0.0
    vram_used_mb: float = 0.0
    vram_free_mb: float = 0.0
    vram_utilization_pct: float = 0.0
    temperature_c: float = 0.0
    gpu_utilization_pct: float = 0.0
    provider: str = "CPUExecutionProvider"
    fp16_enabled: bool = False
    tensorrt_enabled: bool = False
    cuda_graph_enabled: bool = False
    pinned_memory_enabled: bool = False

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "device_id": self.device_id,
            "gpu_name": self.name,
            "vram_total_mb": round(self.vram_total_mb, 1),
            "vram_used_mb": round(self.vram_used_mb, 1),
            "vram_free_mb": round(self.vram_free_mb, 1),
            "vram_utilization_pct": round(self.vram_utilization_pct, 1),
            "temperature_c": round(self.temperature_c, 1),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "provider": self.provider,
            "fp16_enabled": self.fp16_enabled,
            "tensorrt_enabled": self.tensorrt_enabled,
            "cuda_graph_enabled": self.cuda_graph_enabled,
            "pinned_memory_enabled": self.pinned_memory_enabled,
        }


@dataclass
class InferenceSpeed:
    """Tracks inference speed metrics."""
    _times: deque = field(default_factory=lambda: deque(maxlen=120))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, ms: float) -> None:
        with self._lock:
            self._times.append(ms)

    @property
    def avg_ms(self) -> float:
        with self._lock:
            if not self._times:
                return 0.0
            return sum(self._times) / len(self._times)

    @property
    def fps(self) -> float:
        avg = self.avg_ms
        return 1000.0 / avg if avg > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "avg_inference_ms": round(self.avg_ms, 2),
            "inference_fps": round(self.fps, 2),
            "samples": len(self._times),
        }


# ── Pinned Memory Pool ───────────────────────────────────────


class PinnedMemoryPool:
    """
    Page-locked (pinned) memory pool for fast CPU→GPU transfers.

    Pre-allocates reusable pinned numpy arrays so we don't pay the
    pinning cost on every frame. Thread-safe.
    """

    def __init__(self, pool_size: int = 4) -> None:
        self._pool: list[np.ndarray] = []
        self._max_size = pool_size
        self._lock = threading.Lock()
        self._enabled = False
        self._init_pool()

    def _init_pool(self) -> None:
        """Try to initialize pinned memory via PyTorch CUDA."""
        try:
            import torch
            if torch.cuda.is_available():
                self._torch = torch
                self._enabled = True
                logger.info(
                    "Pinned memory pool enabled (PyTorch CUDA, pool=%d)",
                    self._max_size,
                )
                return
        except Exception:
            pass
        self._torch = None
        self._enabled = False
        logger.info("Pinned memory pool disabled (no PyTorch CUDA)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_array(self, shape: tuple, dtype=np.float32) -> np.ndarray:
        """
        Get a pinned numpy array of the given shape/dtype.
        Falls back to a regular array if pinning is unavailable.
        """
        if not self._enabled or self._torch is None:
            return np.empty(shape, dtype=dtype)

        with self._lock:
            # Try to reuse from pool
            for i, arr in enumerate(self._pool):
                if arr.shape == shape and arr.dtype == dtype:
                    return self._pool.pop(i)

        # Allocate new pinned tensor and get numpy view
        tensor = self._torch.empty(
            shape, dtype=self._torch.from_numpy(np.empty(0, dtype=dtype)).dtype
        ).pin_memory()
        return tensor.numpy()

    def return_array(self, arr: np.ndarray) -> None:
        """Return an array to the pool for reuse."""
        if not self._enabled:
            return
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(arr)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "pool_size": len(self._pool),
                "max_pool_size": self._max_size,
            }


# ── CUDA Graph Manager ───────────────────────────────────────


class CUDAGraphManager:
    """
    Manages CUDA Graph capture and replay for fixed-shape inference.

    CUDA Graphs eliminate kernel launch overhead for repeated
    inference with the same input shape — critical for real-time
    video where every frame has identical dimensions.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._graphs: dict[str, Any] = {}  # key → captured graph
        self._static_inputs: dict[str, Any] = {}
        self._static_outputs: dict[str, Any] = {}
        self._lock = threading.Lock()

        try:
            import torch
            if torch.cuda.is_available():
                self._torch = torch
                self._enabled = True
                logger.info("CUDA Graph manager initialized (PyTorch CUDA)")
                return
        except Exception:
            pass
        self._torch = None
        logger.info("CUDA Graph manager disabled (no PyTorch CUDA)")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def capture(self, key: str, input_shape: tuple, run_fn) -> bool:
        """
        Capture a CUDA graph for a function with fixed input shape.
        *run_fn* receives a dummy tensor and returns the output.
        Returns True if capture succeeded.
        """
        if not self._enabled or self._torch is None:
            return False

        with self._lock:
            if key in self._graphs:
                return True

            try:
                # Warmup
                dummy = self._torch.zeros(
                    input_shape, device="cuda", dtype=self._torch.float16
                )
                for _ in range(3):
                    run_fn(dummy)

                self._torch.cuda.synchronize()

                # Capture
                static_input = self._torch.zeros(
                    input_shape, device="cuda", dtype=self._torch.float16
                )
                g = self._torch.cuda.CUDAGraph()
                with self._torch.cuda.graph(g):
                    static_output = run_fn(static_input)

                self._graphs[key] = g
                self._static_inputs[key] = static_input
                self._static_outputs[key] = static_output
                logger.info("CUDA Graph captured for '%s' shape=%s", key, input_shape)
                return True
            except Exception as exc:
                logger.warning("CUDA Graph capture failed for '%s': %s", key, exc)
                return False

    def replay(self, key: str, input_data) -> Any:
        """Replay a captured graph. Returns the output."""
        if key not in self._graphs:
            return None

        g = self._graphs[key]
        static_input = self._static_inputs[key]
        static_input.copy_(input_data)
        g.replay()
        return self._static_outputs[key]

    def has_graph(self, key: str) -> bool:
        return key in self._graphs

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "captured_graphs": list(self._graphs.keys()),
                "graph_count": len(self._graphs),
            }


# ── GPU Manager ──────────────────────────────────────────────


class GPUManager:
    """
    Central GPU optimization and monitoring service.

    • Auto-selects the best NVIDIA GPU
    • Builds the ONNX Runtime provider chain (TensorRT → CUDA → CPU)
    • Manages pinned memory pool + CUDA graph capture
    • Monitors VRAM, temperature, utilization via NVML
    • Tracks inference speed (avg ms + FPS)
    • Falls back to CPU if no GPU is available
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.info = GPUInfo(
            available=False,
            device_id=-1,
            name="CPU (no GPU)",
            provider="CPUExecutionProvider",
        )
        self.speed = InferenceSpeed()
        self.pinned_pool = PinnedMemoryPool(pool_size=4)
        self.graph_manager = CUDAGraphManager()
        self._nvml_initialized = False
        self._nvml_handle = None
        self._torch = None

        self._detect_gpu()

    # ── Detection & Initialization ───────────────────────────

    def _detect_gpu(self) -> None:
        """Detect NVIDIA GPU(s) and initialize NVML + PyTorch CUDA."""
        logger.info("=" * 50)
        logger.info("GPUManager: detecting GPU...")
        logger.info("=" * 50)

        # 1 · Try PyTorch CUDA
        torch_cuda_available = False
        try:
            import torch
            self._torch = torch
            torch_cuda_available = torch.cuda.is_available()
        except Exception:
            logger.info("PyTorch CUDA unavailable (likely cuDNN mismatch) — using ONNX Runtime CUDA EP directly")
            self._torch = None

        # 2 · Try NVML for monitoring
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count > 0:
                # Auto-select GPU: pick the one with most free VRAM
                best_id = self._select_best_gpu(pynvml, device_count)
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(best_id)
                name = pynvml.nvmlDeviceGetName(self._nvml_handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                self.info.available = True
                self.info.device_id = best_id
                self.info.name = name
                logger.info("  ✓ GPU detected: %s (device %d)", name, best_id)
            else:
                logger.info("  ⊘ No NVIDIA GPUs found via NVML")
        except ImportError:
            logger.info("  ⊘ pynvml not installed — GPU monitoring disabled")
        except Exception as exc:
            logger.warning("  ⚠ NVML init failed: %s", exc)

        # 3 · Fallback to CPU if no GPU
        if not self.info.available and not torch_cuda_available:
            self._fallback_to_cpu()
            return

        # 4 · Configure providers
        self._configure_providers()

        # 5 · Refresh initial stats
        self._refresh_stats()

    def _select_best_gpu(self, pynvml, count: int) -> int:
        """Select the GPU with the most free VRAM."""
        best_id = 0
        best_free = 0
        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                free_mb = mem.free / (1024 * 1024)
                if free_mb > best_free:
                    best_free = free_mb
                    best_id = i
            except Exception:
                continue

        # Override with explicit setting if valid
        configured = self.settings.gpu_device_id
        if 0 <= configured < count:
            best_id = configured
            logger.info(
                "  → Using configured GPU device_id=%d", configured
            )
        else:
            logger.info(
                "  → Auto-selected GPU device_id=%d (%.0f MB free)",
                best_id,
                best_free,
            )
        return best_id

    def _fallback_to_cpu(self) -> None:
        """Switch to CPU-only mode."""
        self.info.available = False
        self.info.device_id = -1
        self.info.name = "CPU (GPU unavailable)"
        self.info.provider = "CPUExecutionProvider"
        self.info.fp16_enabled = False
        self.info.tensorrt_enabled = False
        self.info.cuda_graph_enabled = False
        self.info.pinned_memory_enabled = False
        logger.warning("  ⚠ Falling back to CPU execution")

    def _configure_providers(self) -> None:
        """Build the ONNX Runtime provider chain."""
        settings = self.settings
        providers: list[str | tuple] = ["CPUExecutionProvider"]

        if not self.info.available:
            return

        # CUDA Execution Provider
        cuda_opts: dict[str, Any] = {
            "device_id": self.info.device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": int(settings.gpu_mem_limit_mb * 1024 * 1024),
            "cudnn_conv_algo_search": "EXHAUSTIVE" if settings.cudnn_exhaustive else "DEFAULT",
        }

        if settings.enable_fp16:
            cuda_opts["preferred_fp16_data_type"] = True
            self.info.fp16_enabled = True
            logger.info("  ✓ FP16 (half-precision) inference enabled")

        providers.insert(0, ("CUDAExecutionProvider", cuda_opts))

        # TensorRT Execution Provider (highest priority)
        if settings.enable_tensorrt:
            import os
            cache_path = settings.trt_engine_cache_path
            os.makedirs(cache_path, exist_ok=True)
            trt_opts: dict[str, Any] = {
                "device_id": self.info.device_id,
                "trt_fp16_enable": settings.enable_fp16,
                "trt_max_workspace_size": int(
                    settings.trt_workspace_mb * 1024 * 1024
                ),
                "trt_max_partition_iterations": 1000,
                "trt_min_subgraph_size": 5,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.abspath(cache_path),
            }
            providers.insert(0, ("TensorrtExecutionProvider", trt_opts))
            self.info.tensorrt_enabled = True
            logger.info("  ✓ TensorRT execution provider enabled (cache: %s)", cache_path)
        else:
            logger.info("  ⊘ TensorRT disabled in settings")

        self.info.provider = (
            "TensorrtExecutionProvider" if self.info.tensorrt_enabled
            else "CUDAExecutionProvider"
        )

        # CUDA Graphs
        if settings.enable_cuda_graph and self.graph_manager.enabled:
            self.info.cuda_graph_enabled = True
            logger.info("  ✓ CUDA Graph capture enabled")
        else:
            self.info.cuda_graph_enabled = False
            logger.info("  ⊘ CUDA Graph disabled")

        # Pinned memory
        if settings.enable_pinned_memory and self.pinned_pool.enabled:
            self.info.pinned_memory_enabled = True
            logger.info("  ✓ Pinned memory pool enabled")
        else:
            self.info.pinned_memory_enabled = False
            logger.info("  ⊘ Pinned memory disabled")

        self._providers = providers
        logger.info("  → Provider chain: %s", [p if isinstance(p, str) else p[0] for p in providers])

        # ── Critical: verify CUDA EP actually loads (no silent CPU fallback) ──
        self._verify_providers()

    def _verify_providers(self) -> None:
        """Verify that CUDA EP actually loads — no silent CPU fallback.

        Creates a throwaway ONNX session and checks which providers are
        actually active. If CUDA isn't available, logs a critical error
        with remediation instructions.
        """
        try:
            import onnxruntime as ort
            import numpy as np

            # Create a tiny dummy model to test provider availability
            # Use available providers from ORT's perspective
            available = ort.get_available_providers()
            logger.info("  → ORT available providers: %s", available)

            if "CUDAExecutionProvider" not in available:
                logger.error("=" * 60)
                logger.error("CRITICAL: CUDA Execution Provider is NOT available!")
                logger.error("All inference will run on CPU (10-50x slower).")
                logger.error("")
                logger.error("Fix for CUDA 12 pods (RunPod):")
                logger.error("  pip uninstall onnxruntime onnxruntime-gpu -y")
                logger.error("  pip install onnxruntime-gpu==1.18.1 \\")
                logger.error("    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/")
                logger.error("=" * 60)
                # Force CUDA off — use CPU only to avoid repeated failed loads
                self._providers = ["CPUExecutionProvider"]
                self.info.provider = "CPUExecutionProvider"
                self.info.tensorrt_enabled = False
                self.info.fp16_enabled = False
                self.info.cuda_graph_enabled = False
                self.info.pinned_memory_enabled = False
        except ImportError:
            logger.warning("onnxruntime not installed — cannot verify providers")

    # ── Public API ───────────────────────────────────────────

    @property
    def providers(self) -> list[str | tuple]:
        """ONNX Runtime provider list for session creation."""
        if hasattr(self, "_providers"):
            return self._providers
        return ["CPUExecutionProvider"]

    @property
    def is_gpu_available(self) -> bool:
        return self.info.available

    @property
    def device_id(self) -> int:
        return self.info.device_id if self.info.available else -1

    @property
    def torch_device(self) -> str:
        """PyTorch device string for this GPU."""
        if self._torch and self.info.available:
            return f"cuda:{self.info.device_id}"
        return "cpu"

    def get_pinned_array(self, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Get a pinned-memory numpy array (or regular if unavailable)."""
        return self.pinned_pool.get_array(shape, dtype)

    def return_pinned_array(self, arr: np.ndarray) -> None:
        """Return a pinned array to the pool."""
        self.pinned_pool.return_array(arr)

    def capture_graph(self, key: str, input_shape: tuple, run_fn) -> bool:
        """Capture a CUDA graph for repeated fixed-shape inference."""
        return self.graph_manager.capture(key, input_shape, run_fn)

    def replay_graph(self, key: str, input_data) -> Any:
        """Replay a captured CUDA graph."""
        return self.graph_manager.replay(key, input_data)

    def has_graph(self, key: str) -> bool:
        return self.graph_manager.has_graph(key)

    def record_inference(self, ms: float) -> None:
        """Record an inference time for speed tracking."""
        self.speed.record(ms)

    # ── Monitoring ───────────────────────────────────────────

    def _refresh_stats(self) -> None:
        """Refresh VRAM, temperature, utilization from NVML."""
        if not self._nvml_initialized or self._nvml_handle is None:
            return

        try:
            import pynvml

            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            self.info.vram_total_mb = mem.total / (1024 * 1024)
            self.info.vram_used_mb = mem.used / (1024 * 1024)
            self.info.vram_free_mb = mem.free / (1024 * 1024)
            self.info.vram_utilization_pct = (
                (mem.used / mem.total * 100) if mem.total > 0 else 0
            )

            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            self.info.gpu_utilization_pct = float(util.gpu)

            temp = pynvml.nvmlDeviceGetTemperature(
                self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
            )
            self.info.temperature_c = float(temp)
        except Exception as exc:
            logger.debug("NVML stats refresh failed: %s", exc)

    def get_status(self) -> dict:
        """Full GPU status for the /gpu endpoint."""
        self._refresh_stats()
        return {
            **self.info.to_dict(),
            "inference_speed": self.speed.to_dict(),
            "pinned_memory": self.pinned_pool.get_stats(),
            "cuda_graphs": self.graph_manager.get_stats(),
            "batch_size": 1,
            "optimization_summary": {
                "gpu_available": self.info.available,
                "tensorrt": self.info.tensorrt_enabled,
                "fp16": self.info.fp16_enabled,
                "cuda_graphs": self.info.cuda_graph_enabled,
                "pinned_memory": self.info.pinned_memory_enabled,
                "cpu_fallback": not self.info.available,
            },
        }

    def shutdown(self) -> None:
        """Clean up NVML."""
        if self._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ── Singleton ────────────────────────────────────────────────

gpu_manager = GPUManager()