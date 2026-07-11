# ═══════════════════════════════════════════════════════════════
#  FaceSwap AI Backend — Test Suite
#  Run: pytest tests/ -v
#  In Docker: ./deploy.sh test
# ═══════════════════════════════════════════════════════════════
import pytest
import sys
import os

# Ensure backend dir is on the path when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Test configuration loading from environment variables."""

    def test_settings_defaults(self):
        from config import Settings
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.websocket_path == "/ws/swap"

    def test_settings_env_prefix(self):
        os.environ["FACESWAP_PORT"] = "9999"
        from config import Settings
        s = Settings()
        assert s.port == 9999
        del os.environ["FACESWAP_PORT"]

    def test_gpu_defaults(self):
        from config import Settings
        s = Settings()
        assert s.enable_tensorrt is True
        assert s.enable_fp16 is True
        assert s.gpu_mem_limit_mb == 4096

    def test_voice_config(self):
        from config import Settings
        s = Settings()
        assert s.voice_sample_rate == 24000
        assert s.voice_channels == 1
        assert s.voice_websocket_path == "/ws/voice"

    def test_platform_config(self):
        from config import Settings
        s = Settings()
        assert s.platform_default_sample_rate == 48000
        assert s.platform_zoom_enabled is False
        assert s.platform_webrtc_enabled is False


class TestMetrics:
    """Test the metrics collector."""

    def test_metrics_singleton(self):
        from services.metrics import metrics
        assert metrics is not None

    def test_metrics_initial_state(self):
        from services.metrics import metrics
        data = metrics.to_dict()
        assert "fps" in data
        assert "avg_latency_ms" in data
        assert "total_frames" in data


class TestLogger:
    """Test logger setup."""

    def test_logger_creation(self):
        from utils.logger import setup_logger
        logger = setup_logger("test")
        assert logger.name == "test"
        assert len(logger.handlers) > 0

    def test_logger_idempotent(self):
        from utils.logger import setup_logger
        logger1 = setup_logger("test_idem")
        handler_count = len(logger1.handlers)
        logger2 = setup_logger("test_idem")
        assert len(logger2.handlers) == handler_count


class TestHealthEndpoint:
    """Test the /health endpoint via FastAPI TestClient."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        # Import after env setup to avoid model loading
        try:
            from main import app
            return TestClient(app)
        except Exception:
            pytest.skip("Could not create test client — models not available")

    def test_health_response(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "gpu_available" in data