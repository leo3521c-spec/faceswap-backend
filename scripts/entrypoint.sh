#!/bin/bash
# ── Container entrypoint ─────────────────────────────────────
# 1. Verify GPU is accessible
# 2. Verify required models exist (or warn)
# 3. Start uvicorn
set -e

echo "============================================"
echo "  FaceSwap AI Backend — Starting"
echo "============================================"
echo "  Time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  GPU Check:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader || echo "  ⚠ nvidia-smi found but failed"
else
    echo "  ⚠ nvidia-smi not found — running in CPU-only mode"
fi
echo "============================================"

# ── Model verification ───────────────────────────────────────
MODELS_DIR="${FACESWAP_MODELS_DIR:-/app/models}"
INSWAPPER="${FACESWAP_INSWAPPER_MODEL_PATH:-models/inswapper_128.onnx}"

if [ ! -f "$INSWAPPER" ]; then
    echo "⚠ Warning: InSwapper model not found at $INSWAPPER"
    echo "  The backend will start but face swap will fail until models are downloaded."
    echo "  See DEPLOY.md → Model Download section."
fi

# ── Start Uvicorn ─────────────────────────────────────────────
echo "→ Starting uvicorn on ${FACESWAP_HOST:-0.0.0.0}:${FACESWAP_PORT:-8000}"
exec uvicorn main:app \
    --host "${FACESWAP_HOST:-0.0.0.0}" \
    --port "${FACESWAP_PORT:-8000}" \
    --workers 1 \
    --log-level "${FACESWAP_LOG_LEVEL:-info}" \
    --access-log \
    --no-server-header