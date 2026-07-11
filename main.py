"""
FastAPI entry point.

Endpoints
─────────
GET  /health      — liveness probe + model status
GET  /metrics     — FPS, latency, frame counts, pipeline metrics
GET  /models      — model metadata + health status
GET  /gpu         — GPU name, VRAM, temperature, utilization, inference speed
GET  /tracking    — face tracking confidence, FPS, state, per-track details
GET  /enhancement — enhancement mode, metrics, available enhancers
PUT  /enhancement — switch mode at runtime ({"mode":"fast|balanced|ultra|off"})
GET  /masking     — semantic face masking status (parser, feather, color/lighting)
PUT  /masking     — configure masking at runtime (enabled, feather_radius, etc.)
GET  /expression  — expression preservation status (method, landmarks, features)
PUT  /expression  — configure expression preservation (enabled, warp_strength, etc.)
GET  /virtual-camera — virtual camera status (resolution, FPS, dropped frames)
PUT  /virtual-camera — enable/disable or reconfigure at runtime
WS   /ws/swap     — real-time face swap stream (3-thread pipeline)

GET  /voice               — voice pipeline status (all sub-systems)
GET  /voice/microphone    — microphone capture status + available devices
PUT  /voice/microphone    — start/stop capture, change device
GET  /voice/noise         — noise suppression status
PUT  /voice/noise         — enable/disable, set aggressiveness (0-4)
GET  /voice/conversion    — voice conversion status, loaded model, pitch
PUT  /voice/conversion    — enable/disable, set pitch, load model
GET  /voice/echo          — echo cancellation status
PUT  /voice/echo          — enable/disable, set tail length
GET  /voice/mute          — mute status
PUT  /voice/mute          — mute/unmute/toggle
WS   /ws/voice            — real-time voice stream (3-thread pipeline)

GET  /platforms               — list all registered platform adapters
GET  /platforms/{platform}    — single platform status
PUT  /platforms/{platform}/connect    — connect to a platform (body = config)
PUT  /platforms/{platform}/disconnect — disconnect from a platform
PUT  /platforms/{platform}/stream     — start/stop audio stream ({"action":"start|stop"})

WebSocket protocol
───────────────────
1. Client connects.
2. Client sends source face as a binary JPEG (first message).
   ─ or a JSON text frame: {"type":"init","source_face":"<base64>"}
3. Server responds: {"type":"ready"}
4. Client streams binary JPEG frames.
5. Server responds with two messages per frame:
   a. JSON text: {"type":"frame_result","inference_time_ms":...,
      "face_count":...,"detection_confidence":...,"enhanced":...}
   b. Binary JPEG (swapped frame)

Pipeline: capture (async) → LatestFrameQueue (max=1, latest-wins)
         → processing (OS thread, GPU) → output queue → sender (async)

The event loop is never blocked by GPU inference.
Target end-to-end latency: < 100 ms.
"""
import asyncio
import base64
import json
import time
import cv2
import numpy as np
from contextlib import asynccontextmanager
from types import SimpleNamespace

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
from services.enhancement_manager import enhancement_manager
from services.virtual_camera import virtual_camera
from services.audio_pipeline import (
    VoicePipeline,
    set_active_voice_pipeline,
    get_active_voice_pipeline_metrics,
)
from services.voice_processor import process_audio
from services.microphone_capture import microphone_capture
from services.noise_suppressor import noise_suppressor
from services.voice_converter import voice_converter
from services.echo_canceller import echo_canceller
from services.mute_manager import mute_manager
from services.platform_manager import platform_manager
from services.face_masking import face_masking_manager
from services.expression_manager import expression_manager
from services.webrtc_video import webrtc_video_manager
from plugins import plugin_registry

settings = get_settings()
logger = setup_logger("main")


# ── Lifespan ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FaceSwap backend...")
    # Load + warmup models in a thread so the loop stays alive
    await asyncio.to_thread(model_manager.load_models)
    enhancement_manager.set_mode(settings.enhancement_mode)
    # Semantic face masking — configure from settings
    face_masking_manager.configure(
        enabled=settings.masking_enabled,
        feather_radius=settings.masking_feather_radius,
        color_correction=settings.masking_color_correction,
        lighting_correction=settings.masking_lighting_correction,
        occlusion_handling=settings.masking_occlusion_handling,
        parser_size=settings.masking_face_parser_size,
    )
    if model_manager.face_parser is not None:
        face_masking_manager.set_parser(model_manager.face_parser)
        logger.info("Face masking: BiSeNet parser active")
    else:
        logger.info("Face masking: landmark-based fallback (no parser model)")
    # Expression preservation — configure from settings
    expression_manager.configure(
        enabled=settings.expression_enabled,
        warp_strength=settings.expression_warp_strength,
        grid_size=settings.expression_grid_size,
    )
    if model_manager.liveportrait is not None:
        expression_manager.set_liveportrait(model_manager.liveportrait)
        logger.info("Expression preservation: LivePortrait active")
    else:
        logger.info("Expression preservation: landmark-based warping")
    logger.info(
        "Enhancement mode: %s | Enhancers: %s",
        enhancement_manager.mode,
        enhancement_manager.available_enhancers,
    )
    if settings.vc_enabled:
        virtual_camera.enable(
            resolution=settings.vc_resolution, fps=settings.vc_fps
        )
    # Voice pipeline — initialize config (services start on demand)
    voice_converter.set_enabled(settings.voice_conversion_enabled)
    noise_suppressor.set_enabled(settings.voice_noise_suppression)
    echo_canceller.set_enabled(settings.voice_echo_cancellation)
    mute_manager.set_muted(settings.voice_muted)

    # WebRTC video transport — configure ICE/TURN servers
    all_ice = list(settings.webrtc_ice_servers) + list(settings.webrtc_turn_servers)
    webrtc_video_manager.set_ice_servers(all_ice)
    logger.info(
        "WebRTC video transport: %s | ICE servers: %d",
        "enabled" if settings.webrtc_enabled else "disabled",
        len(all_ice),
    )

    # ── Plugin Architecture ────────────────────────────────
    # Auto-discover all plugins (platforms, AI models, voice effects,
    # video effects) from the plugins/ directory — no manual registration.
    plugin_registry.discover(settings=settings)
    plugin_registry.register_platforms(platform_manager)
    logger.info(
        "Plugins: %d total | %d platform adapters",
        plugin_registry.count,
        len(platform_manager.list_platforms()),
    )

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
    plugin_registry.shutdown_all()
    virtual_camera.disable()
    microphone_capture.stop()
    await platform_manager.stop_pipeline()
    for adapter in platform_manager.list_platforms():
        p = platform_manager.get_adapter(adapter["platform"])
        if p and p.connected:
            p.disconnect()
    await webrtc_video_manager.cleanup()
    gpu_manager.shutdown()


# ── App ─────────────────────────────────────────────────────

app = FastAPI(
    title="FaceSwap AI Backend",
    version="1.0.0",
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
    status = model_manager.get_model_status()
    return {
        "status": "ok" if model_manager.is_loaded else "loading",
        "models_loaded": model_manager.is_loaded,
        "enhancer_enabled": bool(getattr(model_manager, "enhancers", {})),
        "gpu_available": gpu_manager.is_gpu_available,
        "gpu_name": gpu_manager.info.name,
        "provider": gpu_manager.info.provider,
        "required_models_ready": status["required_models_ready"],
    }


@app.get("/metrics")
async def get_metrics():
    return {
        **metrics.to_dict(),
        "pipeline": get_active_pipeline_metrics(),
        "tracking": face_tracker.get_metrics(),
        "enhancement": enhancement_manager.get_metrics(),
        "masking": face_masking_manager.get_status(),
        "expression": expression_manager.get_status(),
        "virtual_camera": virtual_camera.get_status(),
        "voice": {
            "microphone": microphone_capture.get_status(),
            "echo_cancellation": echo_canceller.get_status(),
            "noise_suppression": noise_suppressor.get_status(),
            "voice_conversion": voice_converter.get_status(),
            "mute": mute_manager.get_status(),
            "pipeline": get_active_voice_pipeline_metrics(),
        },
        "platforms": platform_manager.get_status(),
    }


@app.get("/models")
async def get_models():
    """Return detailed metadata and health status for every managed model."""
    return model_manager.get_model_status()


@app.get("/gpu")
async def get_gpu_status():
    """Return GPU name, VRAM, temperature, utilization, and inference speed."""
    return gpu_manager.get_status()


@app.get("/tracking")
async def get_tracking_status():
    """Return face tracking confidence, FPS, state, and per-track details."""
    return face_tracker.get_metrics()


@app.put("/tracking/config")
async def configure_tracking(body: dict):
    """Configure face tracking at runtime.

    Body: {"detection_interval": 5}   # frames between full detections
    """
    if "detection_interval" in body:
        val = body["detection_interval"]
        if not isinstance(val, int) or val < 1 or val > 60:
            return Response.json(
                {"error": "detection_interval must be 1-60"}, status_code=400
            )
        face_tracker.set_detection_interval(val)
    return {
        "detection_interval": face_tracker.get_detection_interval(),
    }


@app.get("/enhancement")
async def get_enhancement_status():
    """Return enhancement mode, metrics, and available enhancers."""
    return enhancement_manager.get_metrics()


@app.put("/enhancement")
async def set_enhancement_mode(body: dict):
    """Switch enhancement mode at runtime — no restart required.

    Body: {"mode": "off" | "fast" | "balanced" | "ultra"}
    """
    mode = body.get("mode", "")
    if not enhancement_manager.set_mode(mode):
        return Response.json(
            {"error": f"Invalid mode '{mode}'. Valid: off, fast, balanced, ultra"},
            status_code=400,
        )
    return {
        "mode": enhancement_manager.mode,
        "message": f"Enhancement mode set to '{enhancement_manager.mode}'",
    }


@app.get("/virtual-camera")
async def get_virtual_camera_status():
    """Return virtual camera status: resolution, FPS, dropped frames."""
    return virtual_camera.get_status()


@app.put("/virtual-camera")
async def configure_virtual_camera(body: dict):
    """Enable, disable, or reconfigure the virtual camera at runtime.

    Body options:
      {"enabled": true, "resolution": "720p", "fps": 30}
      {"enabled": false}
    """
    enabled = body.get("enabled")
    resolution = body.get("resolution")
    fps = body.get("fps")

    if enabled is False:
        return virtual_camera.disable()

    if enabled is True:
        return virtual_camera.enable(resolution=resolution, fps=fps)

    return Response.json(
        {"error": "Provide 'enabled' (true/false) in the request body"},
        status_code=400,
    )


# ── Semantic Face Masking ────────────────────────────────────

@app.get("/masking")
async def get_masking_status():
    """Return semantic face masking status and configuration."""
    return face_masking_manager.get_status()


@app.put("/masking")
async def configure_masking(body: dict):
    """Configure semantic face masking at runtime.

    Body options:
      {"enabled": true}
      {"feather_radius": 15}
      {"color_correction": true}
      {"lighting_correction": true}
      {"occlusion_handling": true}
    """
    if "enabled" in body:
        face_masking_manager.set_enabled(body["enabled"])
    if "feather_radius" in body:
        face_masking_manager.set_feather_radius(body["feather_radius"])
    if "color_correction" in body:
        face_masking_manager.set_color_correction(body["color_correction"])
    if "lighting_correction" in body:
        face_masking_manager.set_lighting_correction(body["lighting_correction"])
    if "occlusion_handling" in body:
        face_masking_manager.set_occlusion_handling(body["occlusion_handling"])
    return face_masking_manager.get_status()


# ── Expression Preservation ──────────────────────────────────

@app.get("/expression")
async def get_expression_status():
    """Return expression preservation status and configuration."""
    return expression_manager.get_status()


@app.put("/expression")
async def configure_expression(body: dict):
    """Configure expression preservation at runtime.

    Body options:
      {"enabled": true}
      {"warp_strength": 1.0}     # 0.0–2.0
      {"grid_size": 32}          # 8–64
    """
    if "enabled" in body:
        expression_manager.set_enabled(body["enabled"])
    if "warp_strength" in body:
        if not expression_manager.set_warp_strength(body["warp_strength"]):
            return Response.json(
                {"error": "Warp strength must be 0.0–2.0"}, status_code=400
            )
    if "grid_size" in body:
        if not expression_manager.set_grid_size(body["grid_size"]):
            return Response.json(
                {"error": "Grid size must be 8–64"}, status_code=400
            )
    return expression_manager.get_status()


# ── WebRTC Video Transport ───────────────────────────────────

@app.get("/webrtc/status")
async def get_webrtc_status():
    """Return WebRTC video transport status and ICE server config."""
    all_ice = list(settings.webrtc_ice_servers) + list(settings.webrtc_turn_servers)
    return {
        "enabled": settings.webrtc_enabled,
        **webrtc_video_manager.get_status(),
        "ice_servers": all_ice,
    }


@app.post("/webrtc/offer")
async def webrtc_offer(body: dict):
    """WebRTC signaling endpoint — exchange SDP offer/answer.

    Body: {"sdp": "<offer_sdp>", "type": "offer",
           "source_face": "<base64_jpeg>",
           "ice_servers": [...]}
    Returns: {"sdp": "<answer_sdp>", "type": "answer"}
    """
    sdp = body.get("sdp", "")
    source_face_b64 = body.get("source_face", "")
    ice_servers = body.get("ice_servers")

    if not sdp or not source_face_b64:
        return Response.json(
            {"error": "Missing 'sdp' or 'source_face' in request body"},
            status_code=400,
        )

    result = await webrtc_video_manager.handle_offer(
        sdp=sdp,
        source_face_b64=source_face_b64,
        ice_servers=ice_servers,
    )

    if isinstance(result, tuple):
        data, status = result
        return Response.json(data, status_code=status)

    return result


@app.get("/webrtc/ice-servers")
async def get_ice_servers():
    """Return configured ICE/TURN servers for the browser client."""
    return {
        "ice_servers": list(settings.webrtc_ice_servers)
        + list(settings.webrtc_turn_servers),
    }


# ── Voice / Audio ────────────────────────────────────────────

@app.get("/voice")
async def get_voice_status():
    """Return aggregated status for the entire voice pipeline."""
    return {
        "microphone": microphone_capture.get_status(),
        "echo_cancellation": echo_canceller.get_status(),
        "noise_suppression": noise_suppressor.get_status(),
        "voice_conversion": voice_converter.get_status(),
        "mute": mute_manager.get_status(),
        "pipeline": get_active_voice_pipeline_metrics(),
    }


@app.get("/voice/microphone")
async def get_microphone_status():
    """Return microphone capture status and available devices."""
    return microphone_capture.get_status()


@app.put("/voice/microphone")
async def configure_microphone(body: dict):
    """Start/stop microphone capture or change device.

    Body: {"action": "start"|"stop"|"list", "device": <index>}
    """
    action = body.get("action", "")
    if action == "start":
        if "device" in body:
            microphone_capture._device = body["device"]
        return microphone_capture.start()
    if action == "stop":
        return microphone_capture.stop()
    if action == "list":
        return {"devices": microphone_capture.list_devices()}
    return Response.json(
        {"error": "Provide 'action' (start|stop|list) in the request body"},
        status_code=400,
    )


@app.get("/voice/noise")
async def get_noise_status():
    """Return noise suppression status."""
    return noise_suppressor.get_status()


@app.put("/voice/noise")
async def configure_noise(body: dict):
    """Enable/disable noise suppression or set aggressiveness (0-4).

    Body: {"enabled": true, "aggressiveness": 2}
    """
    if "enabled" in body:
        noise_suppressor.set_enabled(body["enabled"])
    if "aggressiveness" in body:
        if not noise_suppressor.set_aggressiveness(body["aggressiveness"]):
            return Response.json(
                {"error": "Aggressiveness must be 0-4"}, status_code=400
            )
    return noise_suppressor.get_status()


@app.get("/voice/conversion")
async def get_conversion_status():
    """Return voice conversion status, loaded model, and pitch."""
    return voice_converter.get_status()


@app.put("/voice/conversion")
async def configure_conversion(body: dict):
    """Enable/disable voice conversion, set pitch, or load a model.

    Body options:
      {"enabled": true}
      {"pitch_shift": -3}              # -12 to +12 semitones
      {"model_path": "models/voice.onnx"}
    """
    if "enabled" in body:
        voice_converter.set_enabled(body["enabled"])
    if "pitch_shift" in body:
        if not voice_converter.set_pitch(body["pitch_shift"]):
            return Response.json(
                {"error": "Pitch shift must be -12 to +12 semitones"},
                status_code=400,
            )
    if "model_path" in body:
        voice_converter.load_model(body["model_path"])
    return voice_converter.get_status()


@app.get("/voice/echo")
async def get_echo_status():
    """Return echo cancellation status."""
    return echo_canceller.get_status()


@app.put("/voice/echo")
async def configure_echo(body: dict):
    """Enable/disable echo cancellation or set tail length.

    Body: {"enabled": true, "tail_length_ms": 128}
    """
    if "enabled" in body:
        echo_canceller.set_enabled(body["enabled"])
    if "tail_length_ms" in body:
        echo_canceller.set_tail_length(body["tail_length_ms"])
    return echo_canceller.get_status()


@app.get("/voice/mute")
async def get_mute_status():
    """Return mute status."""
    return mute_manager.get_status()


@app.put("/voice/mute")
async def configure_mute(body: dict):
    """Mute, unmute, or toggle.

    Body: {"muted": true}    — set explicit state
    Body: {"toggle": true}   — toggle current state
    """
    if body.get("toggle"):
        mute_manager.toggle()
    elif "muted" in body:
        mute_manager.set_muted(body["muted"])
    return mute_manager.get_status()


# ── Plugins ──────────────────────────────────────────────────

@app.get("/plugins")
async def list_plugins():
    """List all registered plugins across all categories."""
    return {
        "plugins": plugin_registry.list_all(),
        "count": plugin_registry.count,
    }


@app.get("/plugins/{category}")
async def list_plugins_by_category(category: str):
    """List plugins in a specific category.

    Categories: platforms, ai_models, voice_effects, video_effects
    """
    plugins = plugin_registry.list_category(category)
    if not plugins and category not in (
        "platform", "ai_model", "voice_effect", "video_effect",
    ):
        return Response.json(
            {"error": f"Unknown category '{category}'"}, status_code=404
        )
    return {"category": category, "plugins": plugins, "count": len(plugins)}


@app.get("/plugins/{category}/{name}")
async def get_plugin_detail(category: str, name: str):
    """Return detailed status for a single plugin."""
    plugin = plugin_registry.get(name)
    if plugin is None:
        return Response.json(
            {"error": f"Plugin '{name}' not found"}, status_code=404
        )
    return plugin.to_dict()


# ── Platforms ────────────────────────────────────────────────

@app.get("/platforms")
async def list_platforms():
    """List all registered platform adapters and their status."""
    return {"platforms": platform_manager.list_platforms()}


@app.get("/platforms/{platform}")
async def get_platform_status(platform: str):
    """Return status for a single platform adapter."""
    adapter = platform_manager.get_adapter(platform)
    if not adapter:
        return Response.json(
            {"error": f"Unknown platform '{platform}'"}, status_code=404
        )
    return adapter.get_status()


@app.put("/platforms/{platform}/connect")
async def connect_platform(platform: str, body: dict):
    """Connect to a platform with the given configuration.

    Body contents are platform-specific (tokens, device names, etc.).
    """
    adapter = platform_manager.get_adapter(platform)
    if not adapter:
        return Response.json(
            {"error": f"Unknown platform '{platform}'"}, status_code=404
        )
    return adapter.connect(**body)


@app.put("/platforms/{platform}/disconnect")
async def disconnect_platform(platform: str):
    """Disconnect from a platform."""
    adapter = platform_manager.get_adapter(platform)
    if not adapter:
        return Response.json(
            {"error": f"Unknown platform '{platform}'"}, status_code=404
        )
    return adapter.disconnect()


@app.put("/platforms/{platform}/stream")
async def stream_platform(platform: str, body: dict):
    """Start or stop the audio stream for a platform.

    Body: {"action": "start" | "stop"}

    On first start, the platform audio pipeline is created.
    """
    adapter = platform_manager.get_adapter(platform)
    if not adapter:
        return Response.json(
            {"error": f"Unknown platform '{platform}'"}, status_code=404
        )
    if not adapter.connected:
        return Response.json(
            {"error": f"Platform '{platform}' is not connected"},
            status_code=400,
        )

    action = body.get("action", "")
    if action == "start":
        if not platform_manager._pipeline:
            loop = asyncio.get_running_loop()
            platform_manager.start_pipeline(
                sample_rate=settings.platform_default_sample_rate,
                channels=settings.platform_default_channels,
                loop=loop,
            )
        return adapter.start_stream()
    if action == "stop":
        result = adapter.stop_stream()
        if not any(a.streaming for a in platform_manager._adapters.values()):
            await platform_manager.stop_pipeline()
        return result

    return Response.json(
        {"error": "Provide 'action' (start|stop) in the request body"},
        status_code=400,
    )


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


# ── WebSocket helpers ───────────────────────────────────────

async def _receive_source_face(websocket: WebSocket):
    """Wait for the first message and extract the source face."""
    msg = await websocket.receive()

    if msg.get("type") == "websocket.disconnect":
        return None

    # Binary JPEG
    if msg.get("bytes"):
        return await asyncio.to_thread(extract_source_face, msg["bytes"])

    # JSON text with base64
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
    3-thread pipeline architecture:

    ┌─ capture ─→ FramePipeline.input_queue ─→ processing ─→ output_queue ─→ sender ─┐
    │  (async)       (max_size=1,              (OS thread,     (asyncio)       (async)  │
    │                 latest-wins)              GPU inference)                           │
    └─ reads from WebSocket; drops stale frames so latency stays under 100 ms ─────────┘

    • Capture task  — async, reads binary JPEGs from the WebSocket
    • Processing    — real OS thread, runs GPU inference off the event loop
    • Sender task   — async, sends JSON metadata + binary JPEG to client

    The WebSocket thread (event loop) is NEVER blocked by GPU work.
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
        """Capture task — reads frames from WebSocket, submits to pipeline."""
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
        """Sender task — reads processed results, sends to WebSocket."""
        try:
            while True:
                result, frame_id, put_time = await pipeline.get_result()

                # End-to-end latency: from frame capture to result sent
                latency_ms = (time.perf_counter() - put_time) * 1000

                metrics.record_frame(
                    result.inference_time_ms,
                    face_count=result.face_count,
                    confidence=result.detection_confidence,
                )
                pipeline.metrics.record_sent(frame_id, latency_ms)

                # Send diagnostics as JSON text frame, then binary JPEG
                await websocket.send_json(result.to_metadata())
                await websocket.send_bytes(result.jpeg_bytes)

                # Output to OBS virtual camera if enabled
                if virtual_camera.active and not result.pass_through:
                    try:
                        frame = cv2.imdecode(
                            np.frombuffer(
                                result.jpeg_bytes, dtype=np.uint8
                            ),
                            cv2.IMREAD_COLOR,
                        )
                        if frame is not None:
                            virtual_camera.submit_frame(frame)
                    except Exception as exc:
                        logger.debug("VC submit error: %s", exc)
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


# ── Voice WebSocket ──────────────────────────────────────────

@app.websocket(settings.voice_websocket_path)
async def voice_websocket(websocket: WebSocket):
    """Real-time voice changing WebSocket.

    Protocol:
    1. Client connects.
    2. Client sends JSON: {"type":"init","sample_rate":24000,"channels":1}
    3. Server responds: {"type":"ready"}
    4. Client streams binary PCM chunks (16-bit signed LE).
    5. Server responds per chunk:
       a. JSON text: {"type":"audio_result","processing_time_ms":...,
          "muted":...,"noise_suppressed":...,"voice_converted":...}
       b. Binary PCM (processed audio)
    """
    await websocket.accept()
    client = (
        f"{websocket.client.host}:{websocket.client.port}"
        if websocket.client
        else "unknown"
    )
    logger.info("Voice client connected: %s", client)

    try:
        init = await websocket.receive()
        if init.get("type") == "websocket.disconnect":
            return

        text = init.get("text", "")
        data = json.loads(text)
        if data.get("type") != "init":
            await websocket.send_json(
                {"type": "error", "message": "Expected init message"}
            )
            await websocket.close()
            return

        sample_rate = data.get("sample_rate", settings.voice_sample_rate)
        channels = data.get("channels", settings.voice_channels)

        await websocket.send_json({"type": "ready"})
        logger.info(
            "Voice stream ready for %s (rate=%d, ch=%d)",
            client, sample_rate, channels,
        )

        await _run_voice_loop(websocket, sample_rate, channels, client)

    except WebSocketDisconnect:
        logger.info("Voice client disconnected: %s", client)
    except Exception as exc:
        logger.error("Voice WebSocket error [%s]: %s", client, exc)
    finally:
        logger.info("Voice connection closed: %s", client)


async def _run_voice_loop(
    websocket: WebSocket, sample_rate: int, channels: int, client: str
) -> None:
    """3-thread voice pipeline — same architecture as the face swap loop.

    Capture (async) -> LatestChunkQueue -> Processing (thread) -> OutputQueue -> Sender (async)
    """
    loop = asyncio.get_running_loop()
    pipeline = VoicePipeline(
        process_fn=process_audio,
        sample_rate=sample_rate,
        channels=channels,
        loop=loop,
    )
    pipeline.start()
    set_active_voice_pipeline(pipeline)

    async def capture():
        """Capture task — reads PCM chunks from WebSocket."""
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                chunk = msg.get("bytes")
                if chunk:
                    pipeline.submit_chunk(chunk)
        except WebSocketDisconnect:
            return

    async def sender():
        """Sender task — sends processed audio + metadata to WebSocket."""
        try:
            while True:
                result, put_time = await pipeline.get_result()

                latency_ms = (time.perf_counter() - put_time) * 1000
                pipeline.metrics.record_sent(latency_ms)

                await websocket.send_json(result.to_metadata())
                await websocket.send_bytes(result.pcm_data)
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
    set_active_voice_pipeline(None)


# ── Entry ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=1,  # single worker — GPU is the bottleneck, not the GIL
    )