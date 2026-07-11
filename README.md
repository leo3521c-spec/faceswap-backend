# FaceSwap AI — FastAPI Backend

Real-time AI face swap backend optimized for **< 100 ms** latency and
**24–30 FPS** on NVIDIA RTX GPUs.

## Architecture

```
Camera → WebSocket → Frame Queue (latest-only) → AI Worker
  → Face Detection (InsightFace) → Face Swap (InSwapper 128)
  → Enhancement (GFPGAN, optional) → JPEG Encode → Return Frame
```

## Quick Start

### 1 · Download models

```bash
mkdir models
# InSwapper 128
wget -O models/inswapper_128.onnx \
  https://huggingface.co/ezk77/inswapper_128/resolve/main/inswapper_128.onnx
# InsightFace will auto-download buffalo_l on first run
# (optional) GFPGAN
wget -O models/GFPGANv1.4.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

### 2 · Local dev

```bash
cp .env.example .env
pip install -r requirements.txt
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3 · Docker (GPU)

```bash
cp .env.example .env
./deploy.sh          # One-command deploy (builds + starts + SSL + health check)
# or manual:
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

> **Full deployment guide**: `DEPLOY.md` — covers NGINX, HTTPS, CI/CD,
> GPU monitoring, environments (prod/dev/test), and troubleshooting.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model status |
| GET | `/metrics` | FPS, latency, frame counts |
| WS | `/ws/swap` | Real-time swap stream |

### WebSocket protocol

1. Connect to `ws://host:8000/ws/swap`
2. Send source face as binary JPEG (first message)
3. Receive `{"type":"ready"}`
4. Stream binary JPEG frames → receive swapped JPEG frames

```python
# Minimal client example
import websockets, asyncio, cv2

async def client():
    async with websockets.connect("ws://localhost:8000/ws/swap") as ws:
        # Send source face
        source = cv2.imread("source.jpg")
        _, jpg = cv2.imencode(".jpg", source)
        await ws.send(jpg.tobytes())
        await ws.recv()  # {"type":"ready"}

        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok: break
            _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            await ws.send(jpg.tobytes())
            result = await ws.recv()
            swapped = cv2.imdecode(
                np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            cv2.imshow("swapped", swapped)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

asyncio.run(client())
```

## Configuration

All settings are environment variables prefixed with `FACESWAP_`.
See `.env.example` for the full list.

## Project Structure

```
backend/
├── main.py              # FastAPI app, WebSocket, health, metrics
├── config.py            # Env-var settings
├── requirements.txt
├── Dockerfile           # Multi-stage GPU build
├── docker-compose.yml   # Base compose (shared)
├── docker-compose.prod.yml  # Production overlay
├── docker-compose.dev.yml   # Development overlay (hot reload)
├── docker-compose.test.yml  # Testing overlay (pytest)
├── deploy.sh            # One-command deployment script
├── .env.example         # All environment variables
├── .env.production      # Production env template
├── DEPLOY.md            # Full deployment guide
├── ci-cd-template.yml   # CI/CD pipeline (copy to .github/workflows/)
├── nginx/
│   └── nginx.conf       # Reverse proxy: SSL, WS, rate limiting
├── scripts/
│   ├── entrypoint.sh    # Container entrypoint (GPU check + start)
│   ├── healthcheck.sh   # Docker health check
│   └── gpu-monitor.sh   # GPU monitoring daemon
├── tests/
│   └── test_backend.py  # pytest test suite
├── utils/
│   └── logger.py        # Structured logging
└── services/
    ├── model_manager.py # Model loading + GPU warmup
    ├── face_processor.py# Decode → detect → swap → enhance → encode
    ├── frame_queue.py   # Latest-frame-only async queue
    └── metrics.py       # FPS / latency collector