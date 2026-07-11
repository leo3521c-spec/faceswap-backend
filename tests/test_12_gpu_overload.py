# ═══════════════════════════════════════════════════════════════
#  Test Module 12: GPU Overload
#  Tests pipeline behavior under high processing load
# ═══════════════════════════════════════════════════════════════
import time
import threading
import numpy as np
import pytest

from services.frame_queue import LatestFrameQueue, PipelineMetrics
from tests.conftest import generate_synthetic_face_frame, encode_to_jpeg


class TestGPUOverload:
    """Test pipeline behavior under simulated GPU overload."""

    def test_queue_handles_burst_frames(self):
        """Queue drops stale frames during burst input."""
        q = LatestFrameQueue()
        # Rapidly put 50 frames — only latest should survive
        for i in range(50):
            q.put(f"frame_{i}".encode())
        frame = q.get(timeout=0.5)
        assert frame is not None
        # Latest frame is the one we get
        assert frame.data == b"frame_49"
        # 49 frames were dropped
        assert q.dropped_count == 49

    def test_metrics_track_drops(self):
        """Pipeline metrics correctly track dropped frames under load."""
        m = PipelineMetrics()
        q = LatestFrameQueue()
        for i in range(100):
            q.put(f"frame_{i}".encode())
            if i % 10 == 0:
                m.record_drop()
        assert m.frames_dropped == 10

    def test_concurrent_load_no_crash(self):
        """Concurrent frame submission doesn't crash."""
        q = LatestFrameQueue()
        errors = []

        def producer(thread_id):
            try:
                for i in range(200):
                    q.put(f"t{thread_id}_f{i}".encode())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Total puts = 8 * 200 = 1600
        assert q._frame_counter == 1600

    def test_processing_delay_drops_frames(self):
        """When processing is slow, the queue drops intermediate frames."""
        q = LatestFrameQueue()
        # Put frame 1
        q.put(b"frame_1")
        # Simulate slow processing — put frames 2-5 while "processing"
        for i in range(2, 6):
            q.put(f"frame_{i}".encode())
        # Get the result — should be the latest (frame 5)
        result = q.get(timeout=0.5)
        assert result.data == b"frame_5"
        # 4 frames were dropped (frames 1-4 replaced by 5)
        assert q.dropped_count == 4

    def test_latency_stays_bounded_under_load(self):
        """Under load, queue wait time stays bounded (latest-wins)."""
        q = LatestFrameQueue()
        # Put 100 frames rapidly
        for i in range(100):
            q.put(f"f{i}".encode())
        # Get the latest
        frame = q.get(timeout=0.5)
        # Wait time should be near zero (frame was just put)
        assert q.avg_wait_time_ms < 100

    def test_throughput_measurement(self):
        """Pipeline can handle high frame throughput."""
        q = LatestFrameQueue()
        start = time.perf_counter()
        count = 500
        for i in range(count):
            q.put(f"f{i}".encode())
        elapsed = time.perf_counter() - start
        throughput = count / elapsed
        # Should handle at least 1000 puts/sec
        assert throughput > 1000, f"Throughput {throughput:.0f} puts/sec too low"

    def test_metrics_under_load(self):
        """Metrics are accurate under high load."""
        m = PipelineMetrics()
        for i in range(100):
            m.record_capture(i)
            if i % 3 == 0:
                m.record_processed(i, 25.0, 5.0)
            if i % 5 == 0:
                m.record_sent(i, 50.0)
            if i % 7 == 0:
                m.record_drop()
        d = m.to_dict()
        assert d["frames_captured"] == 100
        assert d["frames_processed"] == 34  # 0,3,6,...99
        assert d["frames_sent"] == 20  # 0,5,10,...95
        assert d["frames_dropped"] == 15  # 0,7,14,...98