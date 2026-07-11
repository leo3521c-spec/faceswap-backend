#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  FaceSwap AI — RunPod One-Command Setup (Native, No Docker)
#  RunPod PyTorch template already has Python 3.11 + CUDA + PyTorch
# ═══════════════════════════════════════════════════════════════

echo "============================================"
echo "  FaceSwap AI — RunPod Setup (Native)"
echo "============================================"

# ── Step 1: Check GPU ─────────────────────────────────────────
echo "Step 1: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "  ✓ NVIDIA GPU detected: $GPU_NAME"
else
    echo "  ✗ No NVIDIA GPU found! Make sure you selected a GPU pod."
    exit 1
fi

# ── Step 2: Check Python ──────────────────────────────────────
echo "Step 2: Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "  ✗ Python 3 not found!"
    exit 1
fi
PY_VER=$(python3 --version 2>&1)
echo "  ✓ $PY_VER"

echo "  Installing Python dependencies..."
pip install --quiet --upgrade pip
# Install all deps EXCEPT torch/torchvision (already in RunPod template)
grep -v -E '^(torch|torchvision)==' requirements.txt > /tmp/req_nogpu.txt
pip install --quiet -r /tmp/req_nogpu.txt
echo "  ✓ Dependencies installed"

# ── Step 3: Download AI Models ────────────────────────────────
echo "Step 3: Downloading AI models..."
mkdir -p models

if [ ! -f "models/inswapper_128.onnx" ]; then
    echo "  Downloading InSwapper 128 model (~530MB)..."
    curl -L --progress-bar -o models/inswapper_128.onnx \
        https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
    if [ -s models/inswapper_128.onnx ]; then
        echo "  ✓ InSwapper model downloaded"
    else
        echo "  ✗ InSwapper download FAILED!"
        exit 1
    fi
else
    echo "  ✓ InSwapper model already exists"
fi

if [ ! -f "models/GFPGANv1.4.pth" ]; then
    echo "  Downloading GFPGAN enhancer model (~333MB)..."
    curl -L --progress-bar -o models/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
    if [ -s models/GFPGANv1.4.pth ]; then
        echo "  ✓ GFPGAN model downloaded"
    else
        echo "  ✗ GFPGAN download FAILED!"
        exit 1
    fi
else
    echo "  ✓ GFPGAN model already exists"
fi

# ── Step 4: Kill any old instance ─────────────────────────────
echo "Step 4: Starting backend..."
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

# ── Step 5: Start backend ─────────────────────────────────────
export FACESWAP_HOST=0.0.0.0
export FACESWAP_PORT=8000
export FACESWAP_LOG_LEVEL=INFO
export FACESWAP_CORS_ORIGINS='["*"]'
export FACESWAP_ENABLE_TENSORRT=true
export FACESWAP_ENABLE_FP16=true
export FACESWAP_ENABLE_CUDA_GRAPH=true

nohup python3 main.py > /tmp/faceswap.log 2>&1 &
BACKEND_PID=$!
echo "  ✓ Backend started (PID: $BACKEND_PID)"

# ── Step 5b: Start Cloudflare Tunnel for HTTPS ────────────────
# The app runs on HTTPS, so it needs wss:// (secure WebSocket).
# Cloudflare Tunnel gives a free HTTPS URL that proxies to local HTTP.
echo "Step 5b: Starting Cloudflare Tunnel (for HTTPS/WSS)..."
pkill -f "cloudflared tunnel" 2>/dev/null || true

if ! command -v cloudflared &> /dev/null; then
    echo "  Installing cloudflared..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# Start tunnel in background, capture URL from log
nohup cloudflared tunnel --url http://localhost:8000 > /tmp/cloudflared.log 2>&1 &
TUNNEL_PID=$!
echo "  ✓ Tunnel started (PID: $TUNNEL_PID), waiting for URL..."

# Wait for tunnel URL to appear in log (up to 30s)
TUNNEL_URL=""
for i in $(seq 1 15); do
    TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log 2>/dev/null | head -1)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
    sleep 2
done

if [ -n "$TUNNEL_URL" ]; then
    echo "  ✓ Tunnel URL: $TUNNEL_URL"
else
    echo "  ⚠ Tunnel URL not found yet. Check: cat /tmp/cloudflared.log"
    TUNNEL_URL="https://CHECK_CLOUDFLARE_LOG"
fi

echo ""
        echo "Step 5: Waiting for backend to start (models loading, ~60-120s)..."
echo "  Checking health..."
for i in $(seq 1 24); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo "============================================"
        echo "  ✓ Backend is LIVE!"
        echo "============================================"
        echo ""
        PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "YOUR_POD_IP")
        echo "  Backend URL:  http://$PUBLIC_IP:8000"
        echo "  Health:       http://$PUBLIC_IP:8000/health"
        echo "  WebSocket:    ws://$PUBLIC_IP:8000/ws/swap"
        echo ""
        echo "  ⚠ IMPORTANT: Use the HTTPS tunnel URL below (not the HTTP IP)."
        echo "  The app runs on HTTPS, so ws:// will be blocked by the browser."
        echo ""
        echo "  Paste this HTTPS URL in your app's Settings → GPU Server URL:"
        echo "  ┌──────────────────────────────────────────────────────────────┐"
        echo "  │  $TUNNEL_URL"
        echo "  └──────────────────────────────────────────────────────────────┘"
        echo ""
        echo "  Backend (HTTP, local):  http://$PUBLIC_IP:8000"
        echo "  Health:                 http://$PUBLIC_IP:8000/health"
        echo "  WebSocket (WSS):        $TUNNEL_URL/ws/swap"
        echo ""
        echo "  View backend logs:   tail -f /tmp/faceswap.log"
        echo "  View tunnel logs:    tail -f /tmp/cloudflared.log"
        echo "  Stop backend:        kill $BACKEND_PID"
        echo "  Stop tunnel:         kill $TUNNEL_PID"
        echo "============================================"
        exit 0
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "  ⚠ Backend not ready yet. Last 30 lines of log:"
echo "  ──────────────────────────────────────────────"
tail -30 /tmp/faceswap.log 2>/dev/null
echo "  ──────────────────────────────────────────────"
echo "  Full log:  cat /tmp/faceswap.log"
echo "============================================"