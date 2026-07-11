# ═══════════════════════════════════════════════════════════════
#  Test Module 13: Network Interruption
#  Tests pipeline recovery from disconnections and timeouts
# ═══════════════════════════════════════════════════════════════
import asyncio
import time
import pytest

from services.frame_queue import LatestFrameQueue, FramePipeline


class TestNetworkInterruption:
    """Test WebSocket/pipeline recovery from network issues."""

    def test_queue_timeout_on_no_input(self):
        """Queue.get() returns None on timeout (simulates no network input)."""
        q = LatestFrameQueue()
        start = time.perf_counter()
        result = q.get(timeout=0.2)
        elapsed = time.perf_counter() - start
        assert result is None
        assert elapsed >= 0.15  # Waited at least ~0.2s

    def test_queue_recovers_after_timeout(self):
        """Queue recovers and returns frame after a timeout period."""
        q = LatestFrameQueue()
        # First get times out
        result1 = q.get(timeout=0.1)
        assert result1 is None
        # Now put a frame
        q.put(b"recovered_frame")
        result2 = q.get(timeout=0.5)
        assert result2 is not None
        assert result2.data == b"recovered_frame"

    def test_clear_simulates_disconnect(self):
        """Clear simulates flushing stale frames on disconnect."""
        q = LatestFrameQueue()
        q.put(b"stale_frame_1")
        q.put(b"stale_frame_2")
        q.clear()
        # After clear, queue is empty
        assert q.queue_size == 0
        result = q.get(timeout=0.1)
        assert result is None

    def test_reconnect_scenario(self):
        """Simulate disconnect → reconnect → new frames flow."""
        q = LatestFrameQueue()
        # Session 1
        q.put(b"session1_frame")
        f1 = q.get(timeout=0.5)
        assert f1.data == b"session1_frame"
        # Disconnect (clear)
        q.clear()
        # Reconnect — new frames
        q.put(b"session2_frame")
        f2 = q.get(timeout=0.5)
        assert f2.data == b"session2_frame"

    def test_pipeline_stop_on_disconnect(self):
        """Pipeline stops cleanly on simulated disconnect."""
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
        pipeline.start()
        assert pipeline._proc_thread.is_alive()

        # Simulate disconnect
        pipeline.stop()
        assert not pipeline._proc_thread.is_alive()

        # Can restart
        pipeline.start()
        assert pipeline._proc_thread.is_alive()
        pipeline.stop()
        loop.close()

    def test_partial_frame_handling(self):
        """Queue handles partial/incomplete frame data gracefully."""
        q = LatestFrameQueue()
        # Empty bytes (simulates partial frame)
        q.put(b"")
        result = q.get(timeout=0.5)
        assert result is not None
        assert result.data == b""

    def test_resume_after_gap(self):
        """Pipeline resumes correctly after a gap in input."""
        q = LatestFrameQueue()
        # Initial frames
        for i in range(5):
            q.put(f"f{i}".encode())
        f = q.get(timeout=0.5)

        # Gap — no frames for 0.3s
        time.sleep(0.3)
        result = q.get(timeout=0.1)
        assert result is None

        # Resume
        q.put(b"resumed_frame")
        f = q.get(timeout=0.5)
        assert f.data == b"resumed_frame"

    def test_frame_id_continuity_after_disconnect(self):
        """Frame IDs continue incrementing after a gap."""
        q = LatestFrameQueue()
        id1 = q.put(b"f1")
        q.get(timeout=0.5)
        # Gap
        time.sleep(0.1)
        id2 = q.put(b"f2")
        assert id2 > id1