#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  FaceSwap AI — RunPod One-Command Setup
#  Run this script on your RunPod GPU instance
# ═══════════════════════════════════════════════════════════════
# No set -e — we handle errors manually

echo "============================================"
echo "  FaceSwap AI — RunPod Setup"
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

# ── Step 2: Check Docker ──────────────────────────────────────
echo "Step 2: Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "  Installing Docker..."
    curl -fsSL https://get.docker.com | sh
fi
echo "  ✓ Docker installed"

if ! docker compose version &> /dev/null; then
    echo "  Installing Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin
fi
echo "  ✓ Docker Compose ready"

# Start Docker daemon if not running (RunPod doesn't auto-start it)
if ! docker info &> /dev/null; then
    echo "  Starting Docker daemon..."
    
    # Try method 1: service command
    service docker start > /tmp/dockerd.log 2>&1 || true
    sleep 3
    
    # If still not running, try method 2: dockerd with DinD-compatible flags
    if ! docker info &> /dev/null; then
        echo "  Trying direct dockerd with container-compatible flags..."
        dockerd \
            --host=unix:///var/run/docker.sock \
            --storage-driver=vfs \
            --iptables=false \
            --ip6tables=false \
            --bridge=none \
            --ip-masq=false \
            > /tmp/dockerd.log 2>&1 &
        sleep 5
    fi
    
    # Wait up to 40s for daemon to be ready
    for i in $(seq 1 20); do
        if docker info &> /dev/null; then
            echo "  ✓ Docker daemon started"
            break
        fi
        sleep 2
    done
    
    if ! docker info &> /dev/null; then
        echo "  ✗ Docker daemon failed to start."
        echo "  ── Last 30 lines of dockerd log ──"
        tail -30 /tmp/dockerd.log 2>/dev/null || echo "  (no log found)"
        echo "  ──────────────────────────────────"
        exit 1
    fi
else
    echo "  ✓ Docker daemon already running"
fi

# ── Step 3: Download AI Models ────────────────────────────────
echo "Step 3: Downloading AI models..."
mkdir -p models

if [ ! -f "models/inswapper_128.onnx" ]; then
    echo "  Downloading InSwapper 128 model (~530MB)..."
    curl -L --progress-bar -o models/inswapper_128.onnx \
        https://huggingface.co/ezk77/inswapper_128/resolve/main/inswapper_128.onnx
    if [ -s models/inswapper_128.onnx ]; then
        echo "  ✓ InSwapper model downloaded"
    else
        echo "  ✗ InSwapper download FAILED! Check your internet connection."
        echo "  Try manually: curl -L -o models/inswapper_128.onnx https://huggingface.co/ezk77/inswapper_128/resolve/main/inswapper_128.onnx"
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

# ── Step 4: Build & Start ─────────────────────────────────────
echo "Step 4: Building and starting backend..."
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0
docker compose -f docker-compose.runpod.yml build --no-cache
docker compose -f docker-compose.runpod.yml up -d

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
        # Get public IP
        PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "YOUR_POD_IP")
        echo "  Backend URL:  http://$PUBLIC_IP:8000"
        echo "  Health:       http://$PUBLIC_IP:8000/health"
        echo "  WebSocket:    ws://$PUBLIC_IP:8000/ws/swap"
        echo ""
        echo "  Paste this URL in your app's Settings → GPU Server URL:"
        echo "  ┌─────────────────────────────────────────────┐"
        echo "  │  http://$PUBLIC_IP:8000            │"
        echo "  └─────────────────────────────────────────────┘"
        echo ""
        echo "  View logs:  docker compose -f docker-compose.runpod.yml logs -f"
        echo "============================================"
        exit 0
    fi
    echo -n "."
    sleep 5
done

echo ""
echo "  ⚠ Backend not ready yet. Check logs:"
echo "  docker compose -f docker-compose.runpod.yml logs -f"
echo "============================================"