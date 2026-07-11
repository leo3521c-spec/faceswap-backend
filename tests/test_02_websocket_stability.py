# ═══════════════════════════════════════════════════════════════
#  Test Module 2: WebSocket Stability
#  Tests the FramePipeline, LatestFrameQueue, and async output queue
# ═══════════════════════════════════════════════════════════════
import asyncio
import time
import threading
import pytest

from services.frame_queue import (
    LatestFrameQueue,
    PipelineMetrics,
    FramePipeline,
    set_active_pipeline,
    get_active_pipeline_metrics,
)


class TestLatestFrameQueue:
    """Test the latest-frame-wins queue."""

    def test_put_and_get(self):
        """Basic put → get roundtrip."""
        q = LatestFrameQueue()
        frame_id = q.put(b"frame_data")
        assert frame_id == 1
        frame = q.get(timeout=1.0)
        assert frame is not None
        assert frame.data == b"frame_data"
        assert frame.frame_id == 1

    def test_latest_wins(self):
        """Putting a second frame replaces the first (dropped)."""
        q = LatestFrameQueue()
        q.put(b"frame1")
        q.put(b"frame2")
        frame = q.get(timeout=1.0)
        assert frame.data == b"frame2"
        assert q.dropped_count == 1

    def test_get_timeout(self):
        """Get returns None on timeout when queue is empty."""
        q = LatestFrameQueue()
        result = q.get(timeout=0.1)
        assert result is None

    def test_queue_size(self):
        """Queue size is 0 when empty, 1 when has frame."""
        q = LatestFrameQueue()
        assert q.queue_size == 0
        q.put(b"data")
        assert q.queue_size == 1
        q.get(timeout=0.5)
        assert q.queue_size == 0

    def test_clear(self):
        """Clear discards pending frame."""
        q = LatestFrameQueue()
        q.put(b"data")
        q.clear()
        assert q.queue_size == 0
        result = q.get(timeout=0.1)
        assert result is None

    def test_frame_id_increment(self):
        """Frame IDs are monotonically increasing."""
        q = LatestFrameQueue()
        id1 = q.put(b"f1")
        q.get(timeout=0.5)
        id2 = q.put(b"f2")
        assert id2 > id1

    def test_dropped_count_accuracy(self):
        """Dropping 5 frames reports correct count."""
        q = LatestFrameQueue()
        q.put(b"f0")
        for i in range(5):
            q.put(f"f{i+1}".encode())
        assert q.dropped_count == 5

    def test_thread_safety(self):
        """Concurrent puts from multiple threads don't crash."""
        q = LatestFrameQueue()
        results = []

        def producer(start):
            for i in range(100):
                fid = q.put(f"thread-{start}-{i}".encode())
                results.append(fid)

        threads = [
            threading.Thread(target=producer, args=(i,)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All puts succeeded — no crash
        assert len(results) == 400
        # Frame IDs are unique
        assert len(set(results)) == 400

    def test_get_metrics(self):
        """Queue metrics dict has all required fields."""
        q = LatestFrameQueue()
        q.put(b"data")
        q.get(timeout=0.5)
        m = q.get_metrics()
        assert "queue_size" in m
        assert "max_size" in m
        assert "dropped_frames" in m
        assert "last_frame_id" in m
        assert "avg_wait_time_ms" in m


class TestPipelineMetrics:
    """Test the PipelineMetrics collector."""

    def test_initial_state(self):
        m = PipelineMetrics()
        d = m.to_dict()
        assert d["frames_captured"] == 0
        assert d["frames_processed"] == 0
        assert d["frames_sent"] == 0
        assert d["frames_dropped"] == 0
        assert d["avg_latency_ms"] == 0.0

    def test_record_capture(self):
        m = PipelineMetrics()
        m.record_capture(1)
        m.record_capture(2)
        assert m.frames_captured == 2

    def test_record_processed(self):
        m = PipelineMetrics()
        m.record_processed(1, 25.0, 5.0)
        m.record_processed(2, 30.0, 3.0)
        assert m.frames_processed == 2
        assert m.avg_processing_time_ms == 27.5

    def test_record_sent(self):
        m = PipelineMetrics()
        m.record_sent(1, 50.0)
        m.record_sent(2, 60.0)
        assert m.frames_sent == 2
        assert m.avg_latency_ms == 55.0

    def test_latency_ok_threshold(self):
        """latency_ok is True when avg < 100ms."""
        m = PipelineMetrics()
        m.record_sent(1, 50.0)
        d = m.to_dict()
        assert d["latency_ok"] is True

    def test_latency_not_ok(self):
        """latency_ok is False when avg >= 100ms."""
        m = PipelineMetrics()
        m.record_sent(1, 150.0)
        d = m.to_dict()
        assert d["latency_ok"] is False


class TestFramePipeline:
    """Test the full 3-thread pipeline lifecycle."""

    def test_pipeline_start_stop(self):
        """Pipeline starts and stops cleanly."""
        loop = asyncio.new_event_loop()

        def fake_process(data, source):
            return types.SimpleNamespace(
                jpeg_bytes=data,
                inference_time_ms=10.0,
                face_count=1,
                detection_confidence=0.9,
                to_metadata=lambda: {"type": "frame_result"},
            )

        import types
        pipeline = FramePipeline(
            process_fn=fake_process,
            source_face=None,
            loop=loop,
        )
        pipeline.start()
        assert pipeline._proc_thread.is_alive()
        time.sleep(0.2)
        pipeline.stop()
        assert not pipeline._proc_thread.is_alive()
        loop.close()

    def test_pipeline_submit_and_get_result(self):
        """Frame submitted to pipeline produces a result."""
        loop = asyncio.new_event_loop()

        async def run_test():
            import types

            def fake_process(data, source):
                return types.SimpleNamespace(
                    jpeg_bytes=data,
                    inference_time_ms=10.0,
                    face_count=1,
                    detection_confidence=0.9,
                    to_metadata=lambda: {"type": "frame_result"},
                )

            pipeline = FramePipeline(
                process_fn=fake_process,
                source_face=None,
                loop=loop,
            )
            pipeline.start()

            # Submit a frame
            pipeline.submit_frame(b"test_frame")

            # Get result
            result, frame_id, put_time = await asyncio.wait_for(
                pipeline.get_result(), timeout=2.0
            )
            assert result.jpeg_bytes == b"test_frame"
            assert frame_id == 1

            pipeline.stop()

        loop.run_until_complete(run_test())
        loop.close()

    def test_pipeline_drops_stale_frames(self):
        """When processing is slow, stale frames are dropped."""
        loop = asyncio.new_event_loop()

        async def run_test():
            import types
            import time as _time

            def slow_process(data, source):
                _time.sleep(0.3)  # Slow processing
                return types.SimpleNamespace(
                    jpeg_bytes=data,
                    inference_time_ms=300.0,
                    face_count=1,
                    detection_confidence=0.9,
                    to_metadata=lambda: {},
                )

            pipeline = FramePipeline(
                process_fn=slow_process,
                source_face=None,
                loop=loop,
            )
            pipeline.start()

            # Submit multiple frames quickly while processing is slow
            for i in range(5):
                pipeline.submit_frame(f"frame_{i}".encode())

            # Get at least one result
            result, frame_id, _ = await asyncio.wait_for(
                pipeline.get_result(), timeout=3.0
            )
            # Some frames should have been dropped
            assert pipeline.metrics.frames_dropped >= 1

            pipeline.stop()

        loop.run_until_complete(run_test())
        loop.close()

    def test_active_pipeline_reference(self):
        """Global active pipeline reference works."""
        assert get_active_pipeline_metrics()["active"] is False

        loop = asyncio.new_event_loop()
        import types

        def fake_process(data, source):
            return types.SimpleNamespace(
                jpeg_bytes=data,
                inference_time_ms=10.0,
                face_count=0,
                detection_confidence=0.0,
                to_metadata=lambda: {},
            )

        pipeline = FramePipeline(
            process_fn=fake_process,
            source_face=None,
            loop=loop,
        )
        set_active_pipeline(pipeline)
        metrics = get_active_pipeline_metrics()
        assert metrics["active"] is True

        set_active_pipeline(None)
        assert get_active_pipeline_metrics()["active"] is False
        loop.close()