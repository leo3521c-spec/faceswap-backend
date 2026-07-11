# ═══════════════════════════════════════════════════════════════
#  Test Module 15: Frame Drops
#  Tests frame drop detection, counting, and metrics accuracy
# ═══════════════════════════════════════════════════════════════
import time
import threading
import pytest

from services.frame_queue import LatestFrameQueue, PipelineMetrics
from services.metrics import MetricsCollector, metrics


class TestFrameDrops:
    """Test frame drop detection and handling."""

    def test_drop_count_increments(self):
        """Each dropped frame increments the drop counter."""
        q = LatestFrameQueue()
        q.put(b"f1")
        assert q.dropped_count == 0
        q.put(b"f2")  # f1 is dropped
        assert q.dropped_count == 1
        q.put(b"f3")  # f2 is dropped
        assert q.dropped_count == 2

    def test_no_drop_when_consuming(self):
        """No drops when consumer keeps up with producer."""
        q = LatestFrameQueue()
        for i in range(10):
            q.put(f"f{i}".encode())
            q.get(timeout=0.5)
        assert q.dropped_count == 0

    def test_drop_ratio_calculation(self):
        """Drop ratio is correctly calculated."""
        q = LatestFrameQueue()
        total = 100
        for i in range(total):
            q.put(f"f{i}".encode())
        # Only consume the last frame
        q.get(timeout=0.5)
        drop_ratio = q.dropped_count / total
        assert drop_ratio > 0.9  # 99 out of 100 dropped

    def test_pipeline_metrics_drop_tracking(self):
        """PipelineMetrics tracks drops independently."""
        m = PipelineMetrics()
        for _ in range(50):
            m.record_drop()
        assert m.frames_dropped == 50

    def test_global_metrics_drop_tracking(self):
        """Global MetricsCollector tracks dropped frames."""
        collector = MetricsCollector()
        initial_dropped = collector.frames_dropped
        collector.record_drop()
        collector.record_drop()
        collector.record_drop()
        assert collector.frames_dropped == initial_dropped + 3

    def test_drop_under_sustained_load(self):
        """Drops are tracked correctly under sustained high load."""
        q = LatestFrameQueue()
        # Produce 1000 frames without consuming
        for i in range(1000):
            q.put(f"f{i}".encode())
        # Only the last frame is available
        result = q.get(timeout=0.5)
        assert result.frame_id == 1000
        assert q.dropped_count == 999

    def test_drop_recovery(self):
        """After drops, pipeline recovers when load decreases."""
        q = LatestFrameQueue()
        # High load
        for i in range(50):
            q.put(f"f{i}".encode())
        q.get(timeout=0.5)
        drops_high = q.dropped_count

        # Low load — no more drops
        for i in range(10):
            q.put(f"low_{i}".encode())
            q.get(timeout=0.5)

        assert q.dropped_count == drops_high  # No new drops

    def test_concurrent_drops_counted(self):
        """Drops from concurrent producers are counted correctly."""
        q = LatestFrameQueue()

        def producer(thread_id):
            for i in range(100):
                q.put(f"t{thread_id}_f{i}".encode())

        threads = [threading.Thread(target=producer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 400 puts, only 1 frame remains, 399 dropped
        assert q.dropped_count == 399

    def test_drop_metrics_in_to_dict(self):
        """Drop count appears in metrics dict."""
        m = PipelineMetrics()
        m.record_drop()
        m.record_drop()
        d = m.to_dict()
        assert "frames_dropped" in d
        assert d["frames_dropped"] == 2

    def test_avg_wait_time_with_drops(self):
        """Average wait time is low when frames are being dropped (latest-wins)."""
        q = LatestFrameQueue()
        # Rapidly put frames
        for i in range(100):
            q.put(f"f{i}".encode())
        # Get the latest
        q.get(timeout=0.5)
        # Wait time should be very low (frame was just put)
        assert q.avg_wait_time_ms < 10