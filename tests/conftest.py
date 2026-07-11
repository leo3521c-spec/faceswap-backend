# ═══════════════════════════════════════════════════════════════
#  FaceSwap AI — Shared Test Fixtures
#  Synthetic frames, mock faces, dummy models, and shared config
# ═══════════════════════════════════════════════════════════════
import sys
import os
import time
import types
import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import cv2
import pytest

# Ensure backend dir is on sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

# Use CPU-only mode for tests to avoid GPU dependency
os.environ.setdefault("FACESWAP_GPU_DEVICE_ID", "-1")
os.environ.setdefault("FACESWAP_ENABLE_TENSORRT", "false")
os.environ.setdefault("FACESWAP_ENABLE_FP16", "false")
os.environ.setdefault("FACESWAP_ENABLE_CUDA_GRAPH", "false")
os.environ.setdefault("FACESWAP_LOG_LEVEL", "WARNING")


# ═══════════════════════════════════════════════════════════════
#  Synthetic Frame Generators
# ═══════════════════════════════════════════════════════════════

def _draw_face_features(img, cx, cy, w, h, glasses=False, beard=False):
    """Draw minimal facial features (eyes, nose, mouth) on an image."""
    # Eyes
    eye_y = int(cy - h * 0.1)
    eye_offset = int(w * 0.15)
    eye_r = max(2, int(w * 0.04))
    cv2.circle(img, (cx - eye_offset, eye_y), eye_r, (30, 30, 30), -1)
    cv2.circle(img, (cx + eye_offset, eye_y), eye_r, (30, 30, 30), -1)

    if glasses:
        cv2.rectangle(
            img,
            (cx - eye_offset - eye_r - 4, eye_y - eye_r - 4),
            (cx + eye_offset + eye_r + 4, eye_y + eye_r + 4),
            (200, 200, 200), 2,
        )

    # Nose
    nose_pts = np.array([
        [cx, int(cy + h * 0.05)],
        [cx - int(w * 0.05), int(cy + h * 0.15)],
        [cx + int(w * 0.05), int(cy + h * 0.15)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [nose_pts], (40, 40, 40))

    # Mouth
    mouth_y = int(cy + h * 0.25)
    cv2.ellipse(
        img, (cx, mouth_y),
        (int(w * 0.15), int(h * 0.04)),
        0, 0, 180, (20, 20, 20), 2,
    )

    # Beard
    if beard:
        beard_pts = np.array([
            [cx - int(w * 0.2), int(cy + h * 0.1)],
            [cx + int(w * 0.2), int(cy + h * 0.1)],
            [cx + int(w * 0.15), int(cy + h * 0.35)],
            [cx, int(cy + h * 0.42)],
            [cx - int(w * 0.15), int(cy + h * 0.35)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [beard_pts], (60, 50, 40))


def generate_synthetic_face_frame(
    width=640,
    height=480,
    face_count=1,
    brightness=128,
    face_angle=0,
    glasses=False,
    beard=False,
):
    """Generate a synthetic BGR frame with drawn face(s).

    Args:
        face_count: number of faces to draw
        brightness: background brightness (0=black, 255=white)
        face_angle: degrees to offset face position (simulates side face)
        glasses: draw glasses
        beard: draw beard

    Returns:
        np.ndarray BGR image
    """
    img = np.full((height, width, 3), brightness, dtype=np.uint8)

    if face_count == 1:
        positions = [(width // 2, height // 2)]
    elif face_count == 2:
        positions = [(width // 3, height // 2), (2 * width // 3, height // 2)]
    elif face_count == 3:
        positions = [
            (width // 4, height // 2),
            (width // 2, height // 2),
            (3 * width // 4, height // 2),
        ]
    else:
        positions = []
        for i in range(face_count):
            row = i // 3
            col = i % 3
            positions.append((
                int((col + 1) * width / 4),
                int((row + 1) * height / (face_count // 3 + 1)),
            ))

    fw = int(width * 0.15)
    fh = int(height * 0.25)

    for cx, cy in positions:
        # Apply angle offset to simulate side face
        offset_x = int(np.sin(np.radians(face_angle)) * fw * 0.3)
        cx_adj = cx + offset_x

        # Face oval (skin tone)
        skin_color = (180 + np.random.randint(-20, 20),
                      150 + np.random.randint(-20, 20),
                      130 + np.random.randint(-20, 20))
        cv2.ellipse(img, (cx_adj, cy), (fw, fh), 0, 0, 360, skin_color, -1)

        # Facial features
        _draw_face_features(img, cx_adj, cy, fw * 2, fh * 2,
                            glasses=glasses, beard=beard)

    # Add slight noise for realism
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_low_light_frame(width=640, height=480):
    """Generate a very dark frame with a barely visible face."""
    img = generate_synthetic_face_frame(width, height, brightness=15)
    return img


def generate_side_face_frame(width=640, height=480, angle=45):
    """Generate a frame with a side-angled face."""
    return generate_synthetic_face_frame(
        width, height, face_angle=angle,
    )


def generate_glasses_frame(width=640, height=480):
    """Generate a frame with a face wearing glasses."""
    return generate_synthetic_face_frame(width, height, glasses=True)


def generate_beard_frame(width=640, height=480):
    """Generate a frame with a face having a beard."""
    return generate_synthetic_face_frame(width, height, beard=True)


def generate_multi_face_frame(width=640, height=480, count=3):
    """Generate a frame with multiple faces."""
    return generate_synthetic_face_frame(width, height, face_count=count)


def generate_empty_frame(width=640, height=480):
    """Generate a frame with no faces (blank background)."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def encode_to_jpeg(frame, quality=85):
    """Encode a BGR frame to JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "JPEG encoding failed"
    return buf.tobytes()


# ═══════════════════════════════════════════════════════════════
#  Mock Face Object (mimics InsightFace detection)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MockFace:
    """Mimics insightface.app.common.Face object."""
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: np.ndarray = None
    normed_embedding: np.ndarray = None
    gender: int = 0
    age: int = 30
    track_id: int = -1
    head_pose: dict = field(default_factory=lambda: {"yaw": 0, "pitch": 0, "roll": 0})
    confidence: float = 0.0

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.random.randn(512).astype(np.float32)
        if self.normed_embedding is None:
            norm = np.linalg.norm(self.embedding)
            self.normed_embedding = (
                self.embedding / norm if norm > 0 else self.embedding
            )
        self.confidence = self.det_score


def create_mock_face(cx=320, cy=240, w=100, h=120, score=0.95):
    """Create a MockFace at the given position with 5-point landmarks."""
    bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)
    kps = np.array([
        [cx - w*0.15, cy - h*0.1],   # left eye
        [cx + w*0.15, cy - h*0.1],   # right eye
        [cx, cy + h*0.05],           # nose
        [cx - w*0.12, cy + h*0.25],  # left mouth
        [cx + w*0.12, cy + h*0.25],  # right mouth
    ], dtype=np.float32)
    return MockFace(bbox=bbox, kps=kps, det_score=score)


# ═══════════════════════════════════════════════════════════════
#  Mock Model Manager (avoids loading real ONNX models)
# ═══════════════════════════════════════════════════════════════

class MockDetector:
    """Mock InsightFace detector."""
    def __init__(self):
        self._should_detect = True
        self._face_count = 1
        self._score = 0.95

    def get(self, img):
        if not self._should_detect:
            return []
        h, w = img.shape[:2]
        faces = []
        count = self._face_count
        for i in range(count):
            cx = w * (i + 1) / (count + 1)
            cy = h / 2
            fw = min(w, h) * 0.25
            fh = fw * 1.2
            faces.append(create_mock_face(
                cx=int(cx), cy=int(cy),
                w=int(fw), h=int(fh),
                score=self._score,
            ))
        return faces


class MockSwapper:
    """Mock face swapper — returns the frame unchanged."""
    def get(self, img, face, source_face, paste_back=True):
        return img


class MockModelManager:
    """Mock model_manager that avoids loading real models."""
    def __init__(self):
        self.is_loaded = True
        self.detector = MockDetector()
        self.swapper = MockSwapper()
        self.enhancers = {}

    def load_models(self):
        pass

    def get_model_status(self):
        return {
            "is_loaded": True,
            "required_models_ready": True,
            "models": {},
        }


class MockGPUManager:
    """Mock GPU manager for CPU-only test environments."""
    def __init__(self):
        self.is_gpu_available = False
        self.info = types.SimpleNamespace(
            name="CPU (test mode)",
            vram_total_mb=0,
            vram_used_mb=0,
            temperature_c=0,
            utilization_pct=0,
            provider="CPUExecutionProvider",
            fp16_enabled=False,
            tensorrt_enabled=False,
            avg_inference_ms=0.0,
        )

    def get_status(self):
        return {
            "gpu_available": False,
            "name": self.info.name,
            "provider": self.info.provider,
            "fp16_enabled": False,
            "tensorrt_enabled": False,
            "vram_total_mb": 0,
            "vram_used_mb": 0,
            "temperature_c": 0,
            "utilization_pct": 0,
            "avg_inference_ms": 0.0,
        }

    def record_inference(self, ms):
        pass

    def get_pinned_array(self, size, dtype=np.uint8):
        return np.empty(size, dtype=dtype)

    def return_pinned_array(self, buf):
        pass

    def shutdown(self):
        pass


# ═══════════════════════════════════════════════════════════════
#  Pytest Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_face_frame():
    """A standard synthetic face frame (640x480)."""
    return generate_synthetic_face_frame()


@pytest.fixture
def synthetic_face_jpeg():
    """JPEG-encoded synthetic face frame."""
    frame = generate_synthetic_face_frame()
    return encode_to_jpeg(frame)


@pytest.fixture
def low_light_jpeg():
    """JPEG-encoded low-light frame."""
    return encode_to_jpeg(generate_low_light_frame())


@pytest.fixture
def side_face_jpeg():
    """JPEG-encoded side-angle face frame."""
    return encode_to_jpeg(generate_side_face_frame())


@pytest.fixture
def glasses_jpeg():
    """JPEG-encoded frame with glasses."""
    return encode_to_jpeg(generate_glasses_frame())


@pytest.fixture
def beard_jpeg():
    """JPEG-encoded frame with beard."""
    return encode_to_jpeg(generate_beard_frame())


@pytest.fixture
def multi_face_jpeg():
    """JPEG-encoded frame with 3 faces."""
    return encode_to_jpeg(generate_multi_face_frame(count=3))


@pytest.fixture
def empty_jpeg():
    """JPEG-encoded frame with no faces."""
    return encode_to_jpeg(generate_empty_frame())


@pytest.fixture
def mock_model_manager():
    """Mock model manager — no real model loading."""
    return MockModelManager()


@pytest.fixture
def mock_gpu_manager():
    """Mock GPU manager — CPU mode."""
    return MockGPUManager()


@pytest.fixture
def mock_face():
    """A single mock face object."""
    return create_mock_face()