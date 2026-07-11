# ── Multi-stage build: builder → runtime ──────────────────────
# Stage 1: Install dependencies in a throwaway layer
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# ── Stage 2: Lean runtime image ───────────────────────────────
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    nvidia-smi-no-root \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:${FACESWAP_PORT:-8000}/health || exit 1

EXPOSE 8000

# Entrypoint handles GPU check, model verification, then starts uvicorn
ENTRYPOINT ["/app/scripts/entrypoint.sh"]