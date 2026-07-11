# ═══════════════════════════════════════════════════════════════
#  Test Module 16: Long Session Stability
#  Tests sustained operation, metric stability, and no degradation
# ═══════════════════════════════════════════════════════════════
import time
import threading
import pytest

from services.frame_queue import LatestFrameQueue, PipelineMetrics
from services.metrics import MetricsCollector
from tests.conftest import (
    generate_synthetic_face_frame,
    encode_to_jpeg,
    MockModelManager,
)


class TestLongSessionStability:
    """Test stability over extended simulated sessions."""

    def test_sustained_queue_operations(self):
        """Queue handles 10,000 sustained put/get cycles."""
        q = LatestFrameQueue()
        for i in range(10000):
            q.put(f"frame_{i}".encode())
            result = q.get(timeout=0.5)
            assert result is not None
            assert result.data == f"frame_{i}".encode()
        assert q.dropped_count == 0

    def test_sustained_metrics_collection(self):
        """Metrics collector handles 10,000 records without error."""
        m = PipelineMetrics()
        for i in range(10000):
            m.record_capture(i)
            m.record_processed(i, 25.0, 5.0)
            m.record_sent(i, 50.0)
        d = m.to_dict()
        assert d["frames_captured"] == 10000
        assert d["frames_processed"] == 10000
        assert d["frames_sent"] == 10000

    def test_latency_stable_over_time(self):
        """Average latency remains stable over many operations."""
        m = PipelineMetrics()
        latencies = []
        for i in range(1000):
            lat = 25.0 + (i % 10)  # Varies between 25-34ms
            m.record_sent(i, lat)
            latencies.append(lat)

        avg = m.avg_latency_ms
        expected_avg = sum(latencies) / len(latencies)
        assert abs(avg - expected_avg) < 1.0

    def test_fps_calculation_stable(self):
        """FPS calculation is stable over time."""
        m = PipelineMetrics()
        for i in range(100):
            m.record_sent(i, 30.0)
            time.sleep(0.01)  # 100 FPS target
        fps = m.current_fps
        # Should be around 50-150 FPS (with overhead)
        assert 10 < fps < 200

    def test_no_state_corruption_long_session(self):
        """No state corruption after prolonged queue usage."""
        q = LatestFrameQueue()
        for session in range(10):
            # Each "session" = 1000 frames
            for i in range(1000):
                q.put(f"s{session}_f{i}".encode())
                result = q.get(timeout=0.5)
                assert result is not None
            # Verify session boundary
            assert q.dropped_count == 0
            assert q.queue_size == 0

    def test_repeated_pipeline_start_stop(self):
        """Pipeline can be started/stopped repeatedly without issues."""
        import asyncio
        import types

        loop = asyncio.new_event_loop()

        def fake_process(data, source):
            return types.SimpleNamespace(
                jpeg_bytes=data,
                inference_time_ms=10.0,
                face_count=0,
                detection_confidence=0.0,
                to_metadata=lambda: {},
            )

        from services.frame_queue import FramePipeline

        for _ in range(20):
            pipeline = FramePipeline(
                process_fn=fake_process,
                source_face=None,
                loop=loop,
            )
            pipeline.start()
            time.sleep(0.05)
            pipeline.stop()

        loop.close()

    def test_concurrent_long_session(self):
        """Concurrent producers + consumer over extended period."""
        q = LatestFrameQueue()
        stop = threading.Event()
        errors = []

        def producer():
            i = 0
            while not stop.is_set():
                q.put(f"f{i}".encode())
                i += 1
                time.sleep(0.001)

        def consumer():
            while not stop.is_set():
                try:
                    result = q.get(timeout=0.1)
                    if result and result.data != b"":
                        pass  # OK
                except Exception as e:
                    errors.append(e)

        prod = threading.Thread(target=producer)
        cons = threading.Thread(target=consumer)
        prod.start()
        cons.start()

        time.sleep(2.0)  # Run for 2 seconds
        stop.set()
        prod.join(timeout=1.0)
        cons.join(timeout=1.0)

        assert len(errors) == 0
        assert q._frame_counter > 0  # Processed some frames

    def test_memory_stable_over_session(self):
        """Memory usage doesn't grow unboundedly over a long session."""
        import gc
        import resource

        def get_mem():
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        gc.collect()
        mem_start = get_mem()

        q = LatestFrameQueue()
        for i in range(5000):
            q.put(f"frame_{i}".encode() * 100)  # ~700 bytes each
            q.get(timeout=0.5)

        gc.collect()
        mem_end = get_mem()
        growth = mem_end - mem_start
        assert growth < 50, f"Memory grew {growth:.1f}MB over 5000 cycles"

    def test_metrics_rolling_window_bounded(self):
        """Metrics rolling window stays bounded (maxlen=120)."""
        m = PipelineMetrics()
        for i in range(10000):
            m.record_processed(i, float(i), 1.0)
        # Rolling window should cap at 120
        assert len(m._processing_times) <= 120
        assert len(m._wait_times) <= 120

    def test_global_metrics_rolling_window_bounded(self):
        """Global metrics rolling window stays bounded."""
        collector = MetricsCollector()
        for i in range(10000):
            collector.record_frame(latency_ms=float(i), face_count=1, confidence=0.9)
        assert len(collector._latencies) <= 120
        assert len(collector._fps_times) <= 120