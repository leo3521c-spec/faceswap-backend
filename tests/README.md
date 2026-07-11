# FaceSwap AI — Test Suite Documentation

Comprehensive automated test suite covering 16 modules with 149 test cases.

---

## Quick Start

```bash
cd backend

# Run full suite with reports
./run_tests.sh

# Run with pytest directly
python -m pytest tests/ -v

# Run specific module
python -m pytest tests/test_05_face_tracking.py -v

# Run with report generation
./run_tests.sh --report
```

---

## Test Modules

| # | Module | Tests | What's Covered |
|---|--------|-------|----------------|
| 01 | Webcam Capture | 9 | Frame dimensions, dtype, JPEG roundtrip, multiple resolutions, brightness control |
| 02 | WebSocket Stability | 19 | LatestFrameQueue, PipelineMetrics, FramePipeline lifecycle, thread safety, stale frame drops |
| 03 | AI Inference | 10 | JPEG decode/encode, inference timing, mock detector/swapper, pinned memory, FrameResult metadata |
| 04 | Face Detection Accuracy | 11 | BBox validation, 5-point landmarks, detection score range, embedding dims, normed embedding, landmark positions |
| 05 | Face Tracking | 17 | IoU, cosine similarity, kps→bbox, head pose, TrackedFace lifecycle, confidence decay, tracker reset/metrics |
| 06 | Multiple Face Support | 10 | 1/2/3/6 faces, non-overlapping bboxes, distinct embeddings, multi-face swap, metadata |
| 07 | Low Light | 8 | Dark frame generation, brightness gradient, CLAHE enhancement, no-crash under low light |
| 08 | Side Face | 8 | Various angles (0-80°), head pose yaw, frontal vs side difference, no-crash at extreme angles |
| 09 | Glasses | 7 | Glasses frame generation, detector/swapper compatibility, full pipeline with glasses |
| 10 | Beard | 7 | Beard frame generation, detector/swapper compatibility, full pipeline with beard |
| 11 | Resolutions | 8×5 | 360p/480p/720p/1080p/4K — generation, encoding, decode, detector, swapper, timing |
| 12 | GPU Overload | 7 | Burst frames, drop tracking, 8-thread concurrency, latency bounding, throughput |
| 13 | Network Interruption | 8 | Queue timeout, recovery, clear/disconnect, reconnect, partial frames, resume after gap |
| 14 | Memory Leak | 8 | Queue, JPEG, MockFace, MetricsCollector, frame generation, threads, numpy, GC |
| 15 | Frame Drops | 10 | Drop counting, no-drop when consuming, drop ratio, concurrent drops, wait time with drops |
| 16 | Long Session Stability | 10 | 10K cycles, sustained metrics, latency stability, FPS, repeated start/stop, concurrent session, memory bounded |

**Total: 149 test cases across 17 test classes**

---

## Architecture

```
tests/
├── conftest.py                    # Shared fixtures: synthetic frames, mock faces, mock managers
├── generate_report.py             # HTML + JSON + console report generator
├── test_01_webcam_capture.py      # Frame capture & encoding
├── test_02_websocket_stability.py # Queue + pipeline + thread safety
├── test_03_ai_inference.py        # Decode → detect → swap → encode
├── test_04_face_detection.py      # Detection accuracy & landmarks
├── test_05_face_tracking.py       # IoU, embedding sim, track persistence
├── test_06_multiple_faces.py      # Multi-face detection & swap
├── test_07_low_light.py           # Dark frame handling
├── test_08_side_face.py           # Angled/profile faces
├── test_09_glasses.py             # Glasses occlusion
├── test_10_beard.py               # Facial hair occlusion
├── test_11_resolutions.py         # 360p → 4K
├── test_12_gpu_overload.py        # Burst load & throughput
├── test_13_network_interruption.py# Disconnect & recovery
├── test_14_memory_leak.py         # Memory growth detection
├── test_15_frame_drops.py         # Drop counting & metrics
└── test_16_long_session.py        # Sustained operation stability
```

---

## Test Fixtures (conftest.py)

### Synthetic Frame Generators

| Function | Description |
|----------|-------------|
| `generate_synthetic_face_frame()` | BGR frame with drawn face(s) — configurable count, brightness, angle, glasses, beard |
| `generate_low_light_frame()` | Very dark frame (brightness=15) with barely visible face |
| `generate_side_face_frame()` | Frame with face offset to simulate side angle |
| `generate_glasses_frame()` | Frame with glasses drawn on face |
| `generate_beard_frame()` | Frame with beard drawn on face |
| `generate_multi_face_frame()` | Frame with 2-6 faces |
| `generate_empty_frame()` | Blank frame (no face) |
| `encode_to_jpeg()` | Encode BGR frame to JPEG bytes |

### Mock Objects

| Class | Mocks |
|-------|-------|
| `MockFace` | `insightface.app.common.Face` — bbox, kps, det_score, embedding, normed_embedding |
| `MockDetector` | `insightface.app.FaceAnalysis` — `.get()` returns mock faces |
| `MockSwapper` | `inswapper` ONNX model — `.get()` returns frame unchanged |
| `MockModelManager` | `services.model_manager.ModelManager` — avoids loading real models |
| `MockGPUManager` | `services.gpu_manager.GPUManager` — CPU-only test mode |

### Pytest Fixtures

```python
@pytest.fixture
def synthetic_face_frame():     # 640×480 BGR frame with face

@pytest.fixture
def synthetic_face_jpeg():      # JPEG-encoded face frame

@pytest.fixture
def low_light_jpeg():           # Dark JPEG frame

@pytest.fixture
def side_face_jpeg():           # Side-angle JPEG frame

@pytest.fixture
def glasses_jpeg():             # Glasses JPEG frame

@pytest.fixture
def beard_jpeg():               # Beard JPEG frame

@pytest.fixture
def multi_face_jpeg():          # 3-face JPEG frame

@pytest.fixture
def empty_jpeg():               # No-face JPEG frame

@pytest.fixture
def mock_model_manager():       # Mock model manager (no real models)

@pytest.fixture
def mock_gpu_manager():         # Mock GPU manager (CPU mode)

@pytest.fixture
def mock_face():                # Single MockFace object
```

---

## Test Reports

Reports are generated in `logs/test-reports/`:

### HTML Report (`test-report.html`)

- Color-coded pass/fail per module
- Per-test-case breakdown (expandable)
- Summary cards: modules, tests, passed, failed, errors, skipped, duration
- Overall status badge
- Dark theme, responsive

### JSON Report (`test-report.json`)

```json
{
  "generated_at": "2025-01-15T10:30:00Z",
  "summary": {
    "total_modules": 16,
    "modules_passed": 16,
    "modules_failed": 0,
    "total_tests": 149,
    "total_passed": 149,
    "total_failed": 0,
    "overall_status": "PASS"
  },
  "modules": [...]
}
```

### Console Report

```
════════════════════════════════════════════════════════════════
  🎭 FaceSwap AI — TEST REPORT SUMMARY
════════════════════════════════════════════════════════════════

  #    Module                         Status   Pass   Fail    Err   Skip    Time
  ---------------------------------------------------------------------------
  01   Webcam Capture                 PASS        9      0      0      0    0.5s
  02   WebSocket Stability            PASS       19      0      0      0    1.2s
  ...
  16   Long Session Stability         PASS       10      0      0      0    3.1s
  ---------------------------------------------------------------------------
       TOTAL                          PASS      149      0      0      0   15.3s
```

### JUnit XML (`junit-*.xml`)

Per-module JUnit XML files for CI/CD integration.

---

## Running in Docker

```bash
# Using the deployment script
./deploy.sh test

# Manual
docker compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit
```

The test overlay runs in CPU-only mode (no GPU required).

---

## CI/CD Integration

The CI/CD pipeline (see `ci-cd-template.yml`) runs:

1. **Lint** — `py_compile` on all source files
2. **Test** — `pytest tests/ -v --tb=short`
3. **Build** — Docker image build + smoke test
4. **Push** — Image to GHCR (main branch only)
5. **Deploy** — SSH deploy (main branch only)

---

## Adding New Tests

1. Create `test_NN_name.py` in `tests/`
2. Import fixtures from `conftest.py`:
   ```python
   from tests.conftest import generate_synthetic_face_frame, MockModelManager
   ```
3. Write test classes inheriting from nothing (pytest convention)
4. Test functions must start with `test_`
5. Add module to `MODULES` list in `generate_report.py`

---

## Test Categories

### Unit Tests
- Config loading, metrics, logger, geometry helpers
- Queue operations, pipeline metrics
- MockFace, MockDetector, MockSwapper

### Integration Tests
- FramePipeline start/stop/submit/get_result
- Full encode → decode → detect → swap → encode cycle
- Multi-face swap chain

### Stress Tests
- 10,000 sustained queue operations
- 8-thread concurrent producers
- 1000-frame burst with drop tracking
- Repeated pipeline start/stop (20 cycles)

### Memory Tests
- Memory growth < 50MB across 5000 operations
- Rolling window bounded at 120 entries
- GC properly collects frame objects
- Thread creation/destroy doesn't leak

### Edge Case Tests
- Empty frames (no face)
- Extreme face angles (80°)
- Very low light (brightness=15)
- Partial/empty frame data
- Network timeout + recovery