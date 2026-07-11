"""
FastAPI entry point — minimal real-time face swap backend.

Pipeline: WebSocket → JPEG Decode → InsightFace Detection → InSwapper128 → JPEG Encode → Binary Response

Endpoints
─────────
GET  /health      — liveness probe + model status
GET  /metrics     — FPS, latency, frame counts
GET  /gpu         — GPU name, VRAM, temperature, utilization
GET  /logs        — last 100 lines of backend log
WS   /ws/swap     — real-time face swap stream
"""
import asyncio
import base64
import json
import time
import cv2
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response

from config import get_settings
from utils.logger import setup_logger
from services.metrics import metrics
from services.model_manager import model_manager
from services.gpu_manager import gpu_manager
from services.frame_queue import FramePipeline, set_active_pipeline, get_active_pipeline_metrics
from services.face_processor import process_frame, extract_source_face
from services.face_tracker import face_tracker

settings = get_settings()
logger = setup_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FaceSwap backend...")
    await asyncio.to_thread(model_manager.load_models)
    logger.info("Backend ready — accepting connections.")
    logger.info(
        "GPU: %s | Provider: %s | FP16: %s | TensorRT: %s",
        gpu_manager.info.name,
        gpu_manager.info.provider,
        gpu_manager.info.fp16_enabled,
        gpu_manager.info.tensorrt_enabled,
    )
    yield
    logger.info("Shutting down...")
    gpu_manager.shutdown()


app = FastAPI(
    title="FaceSwap AI Backend",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ──────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if model_manager.is_loaded else "loading",
        "models_loaded": model_manager.is_loaded,
        "gpu_available": gpu_manager.is_gpu_available,
        "gpu_name": gpu_manager.info.name,
        "provider": gpu_manager.info.provider,
    }


@app.get("/metrics")
async def get_metrics():
    return {
        **metrics.to_dict(),
        "pipeline": get_active_pipeline_metrics(),
        "tracking": face_tracker.get_metrics(),
        "gpu": gpu_manager.get_status(),
    }


@app.get("/gpu")
async def get_gpu_status():
    return gpu_manager.get_status()


@app.get("/logs")
async def get_logs():
    """Return the last 100 lines of the backend log."""
    import subprocess
    try:
        result = subprocess.run(
            ["tail", "-100", "/tmp/faceswap.log"],
            capture_output=True, text=True, timeout=5,
        )
        return {"lines": result.stdout.splitlines()}
    except Exception as exc:
        return {"error": str(exc)}


# ── WebSocket ───────────────────────────────────────────────

@app.websocket(settings.websocket_path)
async def swap_websocket(websocket: WebSocket):
    await websocket.accept()
    client = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "unknown"
    )
    logger.info("Client connected: %s", client)

    try:
        source_face = await _receive_source_face(websocket)
        if source_face is None:
            await websocket.send_json(
                {"type": "error", "message": "No face detected in source image"}
            )
            await websocket.close()
            return

        await websocket.send_json({"type": "ready"})
        logger.info("Source face locked for %s", client)

        await _run_swap_loop(websocket, source_face, client)

    except WebSocketDisconnect:
        logger.info("Client disconnected: %s", client)
    except Exception as exc:
        logger.error("WebSocket error [%s]: %s", client, exc)
    finally:
        logger.info("Connection closed: %s", client)


async def _receive_source_face(websocket: WebSocket):
    """Wait for the first message and extract the source face."""
    msg = await websocket.receive()

    if msg.get("type") == "websocket.disconnect":
        return None

    if msg.get("bytes"):
        logger.info("Received source face: %d bytes", len(msg["bytes"]))
        return await asyncio.to_thread(extract_source_face, msg["bytes"])

    text = msg.get("text", "")
    try:
        data = json.loads(text)
        if data.get("type") == "init" and data.get("source_face"):
            raw = base64.b64decode(data["source_face"])
            return await asyncio.to_thread(extract_source_face, raw)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Failed to parse source face: %s", exc)

    return None


async def _run_swap_loop(
    websocket: WebSocket, source_face, client: str
) -> None:
    """
    3-thread pipeline:

    ┌─ capture ─→ FramePipeline.input_queue ─→ processing ─→ output_queue ─→ sender ─┐
    │  (async)       (max_size=1,              (OS thread,     (asyncio)       (async)  │
    │                 latest-wins)              GPU inference)                           │
    └─ reads from WebSocket; drops stale frames so latency stays minimal ───────────────┘
    """
    face_tracker.reset()
    loop = asyncio.get_running_loop()
    pipeline = FramePipeline(
        process_fn=process_frame,
        source_face=source_face,
        loop=loop,
    )
    pipeline.start()
    set_active_pipeline(pipeline)

    async def capture():
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                frame = msg.get("bytes")
                if frame:
                    pipeline.submit_frame(frame)
        except WebSocketDisconnect:
            return

    async def sender():
        try:
            while True:
                result, frame_id, put_time = await pipeline.get_result()

                latency_ms = (time.perf_counter() - put_time) * 1000

                metrics.record_frame(
                    result.inference_time_ms,
                    face_count=1 if result.face_detected else 0,
                    confidence=result.confidence,
                )
                pipeline.metrics.record_sent(frame_id, latency_ms)

                # One JSON metadata message, then one binary JPEG
                await websocket.send_json(result.to_metadata())
                await websocket.send_bytes(result.jpeg_bytes)

        except WebSocketDisconnect:
            return

    tasks = [
        asyncio.create_task(capture()),
        asyncio.create_task(sender()),
    ]
    done, pending = await asyncio.wait(
        tasks, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    pipeline.stop()
    set_active_pipeline(None)


# ── Entry ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=1,
    )