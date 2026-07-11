#!/bin/bash
# ── GPU Monitoring Daemon ────────────────────────────────────
# Polls nvidia-smi + the backend /gpu endpoint and logs metrics.
# Intended to run as a sidecar or cron job alongside the backend.

PORT="${FACESWAP_PORT:-8000}"
INTERVAL="${GPU_MONITOR_INTERVAL:-5}"
LOG_FILE="/app/logs/gpu-monitor.log"

mkdir -p /app/logs

echo "GPU monitor started — polling every ${INTERVAL}s → $LOG_FILE"

while true; do
    TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

    # nvidia-smi metrics
    if command -v nvidia-smi &> /dev/null; then
        GPU_STATS=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits 2>/dev/null || echo "N/A,N/A,N/A,N/A,N/A")
    else
        GPU_STATS="N/A,N/A,N/A,N/A,N/A"
    fi

    # Backend /gpu endpoint
    BACKEND_GPU=$(curl -sf --max-time 3 "http://localhost:${PORT}/gpu" 2>/dev/null || echo "{}")
    INFERENCE_MS=$(echo "$BACKEND_GPU" | python3 -c "import sys,json; d=json.load(sys.stdin); print(round(d.get('avg_inference_ms',0),2))" 2>/dev/null || echo "0")

    IFS=',' read -r TEMP UTIL MEM_USED MEM_TOTAL POWER <<< "$GPU_STATS"

    echo "${TIMESTAMP} | temp=${TEMP}C util=${UTIL}% mem=${MEM_USED}/${MEM_TOTAL}MiB power=${POWER}W inference=${INFERENCE_MS}ms" >> "$LOG_FILE"

    # Alert on high temperature
    if [ "$TEMP" != "N/A" ] && [ "$TEMP" -gt 85 ] 2>/dev/null; then
        echo "${TIMESTAMP} | ⚠ ALERT: GPU temperature ${TEMP}°C exceeds 85°C threshold" >> "$LOG_FILE"
    fi

    # Alert on high memory usage (>90%)
    if [ "$MEM_USED" != "N/A" ] && [ "$MEM_TOTAL" != "N/A" ] && [ "$MEM_TOTAL" -gt 0 ] 2>/dev/null; then
        PCT=$((MEM_USED * 100 / MEM_TOTAL))
        if [ "$PCT" -gt 90 ]; then
            echo "${TIMESTAMP} | ⚠ ALERT: GPU memory at ${PCT}% (${MEM_USED}/${MEM_TOTAL}MiB)" >> "$LOG_FILE"
        fi
    fi

    sleep "$INTERVAL"
done