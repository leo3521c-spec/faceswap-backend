# ═══════════════════════════════════════════════════════════════
#  Test Module 14: Memory Leak Detection
#  Tests for memory growth across repeated operations
# ═══════════════════════════════════════════════════════════════
import sys
import gc
import time
import threading
import numpy as np
import pytest

from services.frame_queue import LatestFrameQueue, PipelineMetrics
from tests.conftest import (
    generate_synthetic_face_frame,
    encode_to_jpeg,
    create_mock_face,
    MockModelManager,
)


class TestMemoryLeak:
    """Detect memory leaks in repeated operations."""

    def _get_memory_mb(self):
        """Get current process memory in MB."""
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def test_queue_no_leak(self):
        """Queue doesn't leak memory across many put/get cycles."""
        q = LatestFrameQueue()
        gc.collect()
        mem_before = self._get_memory_mb()

        for _ in range(5000):
            q.put(b"x" * 1024)  # 1KB frames
            q.get(timeout=0.5)

        gc.collect()
        mem_after = self._get_memory_mb()
        # Memory growth should be < 50MB
        growth = mem_after - mem_before
        assert growth < 50, f"Memory grew {growth:.1f}MB (expected <50MB)"

    def test_jpeg_encode_decode_no_leak(self):
        """JPEG encode/decode doesn't leak memory."""
        frame = generate_synthetic_face_frame(640, 480)
        gc.collect()
        mem_before = self._get_memory_mb()

        for _ in range(1000):
            jpeg = encode_to_jpeg(frame)
            import cv2
            decoded = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            del jpeg, decoded

        gc.collect()
        mem_after = self._get_memory_mb()
        growth = mem_after - mem_before
        assert growth < 50, f"Memory grew {growth:.1f}MB"

    def test_mock_face_creation_no_leak(self):
        """MockFace creation doesn't leak memory."""
        gc.collect()
        mem_before = self._get_memory_mb()

        faces = []
        for _ in range(5000):
            face = create_mock_face()
            faces.append(face)
            if len(faces) > 100:
                faces.clear()  # Don't keep references
                gc.collect()

        faces.clear()
        gc.collect()
        mem_after = self._get_memory_mb()
        growth = mem_after - mem_before
        assert growth < 50, f"Memory grew {growth:.1f}MB"

    def test_metrics_collector_no_leak(self):
        """MetricsCollector doesn't leak memory with rolling window."""
        m = PipelineMetrics()
        gc.collect()
        mem_before = self._get_memory_mb()

        for i in range(10000):
            m.record_capture(i)
            m.record_processed(i, 25.0, 5.0)
            m.record_sent(i, 50.0)

        gc.collect()
        mem_after = self._get_memory_mb()
        growth = mem_after - mem_before
        # Rolling window should prevent unbounded growth
        assert growth < 30, f"Memory grew {growth:.1f}MB"

    def test_frame_generation_no_leak(self):
        """Frame generation doesn't leak memory."""
        gc.collect()
        mem_before = self._get_memory_mb()

        for _ in range(500):
            frame = generate_synthetic_face_frame(640, 480)
            del frame

        gc.collect()
        mem_after = self._get_memory_mb()
        growth = mem_after - mem_before
        assert growth < 50, f"Memory grew {growth:.1f}MB"

    def test_thread_creation_no_leak(self):
        """Thread creation/destroy doesn't leak memory."""
        gc.collect()
        mem_before = self._get_memory_mb()

        for _ in range(100):
            q = LatestFrameQueue()

            def producer():
                for i in range(50):
                    q.put(f"f{i}".encode())
                # consumer
                for i in range(50):
                    q.get(timeout=0.5)

            t = threading.Thread(target=producer)
            t.start()
            t.join()
            del q

        gc.collect()
        mem_after = self._get_memory_mb()
        growth = mem_after - mem_before
        assert growth < 50, f"Memory grew {growth:.1f}MB"

    def test_numpy_array_no_leak(self):
        """Numpy array allocation/dealloc doesn't leak."""
        gc.collect()
        mem_before = self._get_memory_mb()

        for _ in range(1000):
            arr = np.random.randn(480, 640, 3).astype(np.uint8)
            del arr

        gc.collect()
        mem_after = self._get_memory_mb()
        growth = mem_after - mem_before
        assert growth < 30, f"Memory grew {growth:.1f}MB"

    def test_gc_collects_frame_objects(self):
        """Garbage collector properly collects frame objects."""
        gc.collect()
        obj_count_before = len(gc.get_objects())

        for _ in range(100):
            frame = generate_synthetic_face_frame(320, 240)
            jpeg = encode_to_jpeg(frame)
            del frame, jpeg

        gc.collect()
        obj_count_after = len(gc.get_objects())
        # Object count shouldn't grow significantly
        growth = obj_count_after - obj_count_before
        assert growth < 500, f"Object count grew by {growth}"