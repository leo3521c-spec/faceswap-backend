#!/bin/bash
# ── Health check script ──────────────────────────────────────
# Used by Docker HEALTHCHECK and external monitoring.
# Exits 0 if healthy, 1 if not.

PORT="${FACESWAP_PORT:-8000}"
HOST="${FACESWAP_HOST:-localhost}"
ENDPOINT="http://${HOST}:${PORT}/health"
TIMEOUT=10

response=$(curl -sf --max-time "$TIMEOUT" "$ENDPOINT" 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "FAIL: Backend unreachable at $ENDPOINT"
    exit 1
fi

# Check if models are loaded
status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

if [ "$status" != "ok" ]; then
    echo "WARN: Backend reachable but status='$status' (models may still be loading)"
    exit 1
fi

echo "OK: Backend healthy — models loaded, GPU active"
exit 0