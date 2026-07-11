# FaceSwap AI Backend

Real-time GPU face swap backend — FastAPI + WebSocket + ONNX Runtime.

## Quick Start
```bash
chmod +x setup-runpod.sh
./setup-runpod.sh
```

## Architecture
- FastAPI WebSocket streaming pipeline
- ONNX Runtime GPU inference (InsightFace + InSwapper)
- GFPGAN / CodeFormer enhancement
- Sub-100ms end-to-end latency
- Docker containerized
