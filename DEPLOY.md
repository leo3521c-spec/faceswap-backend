# FaceSwap AI — Deployment Guide

Complete deployment package with Docker, NGINX, HTTPS, CI/CD, GPU
monitoring, and one-command deployment.

---

## Quick Start (One Command)

```bash
cd backend
./deploy.sh          # Production deploy
```

That's it. The script will:
1. Check prerequisites (Docker, GPU)
2. Create `.env` from `.env.example`
3. Generate SSL certificates
4. Build and start all containers
5. Wait for health check

---

## Commands

```bash
./deploy.sh prod      # Production deploy (default)
./deploy.sh dev       # Development with hot reload
./deploy.sh test      # Run test suite
./deploy.sh stop      # Stop all services
./deploy.sh logs      # Tail logs (add service name to filter)
./deploy.sh status    # Service + GPU + health status
./deploy.sh health    # Check /health endpoint
./deploy.sh ssl       # Generate SSL certificates
./deploy.sh gpu       # Start GPU monitor sidecar
./deploy.sh clean     # Remove all containers + volumes
```

---

## Prerequisites

### 1. Docker + Docker Compose v2

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in

# Verify
docker --version
docker compose version
```

### 2. NVIDIA Drivers + Container Toolkit (GPU)

```bash
# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

### 3. Download AI Models

```bash
cd backend
mkdir -p models

# InSwapper 128 (required)
wget -O models/inswapper_128.onnx \
    https://huggingface.co/ezk77/inswapper_128/resolve/main/inswapper_128.onnx

# GFPGAN (optional — face enhancement)
wget -O models/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

> InsightFace's `buffalo_l` detector auto-downloads on first run.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │              Docker Network               │
                    │                                          │
   Port 443 ──────► │  ┌──────────┐    ┌─────────────────────┐ │
   Port 80  ──────► │  │  NGINX   │───►│  FaceSwap Backend    │ │
   (HTTPS)          │  │  SSL/TLS │    │  (FastAPI + GPU)     │ │
                    │  │  Proxy   │    │  Port 8000           │ │
                    │  └──────────┘    └─────────────────────┘ │
                    │                         │                │
                    │                    ┌────┴────┐           │
                    │                    │  GPU    │           │
                    │                    │ Monitor │           │
                    │                    │ Sidecar │           │
                    │                    └─────────┘           │
                    └──────────────────────────────────────────┘
```

### Containers

| Container         | Role                                           |
|-------------------|------------------------------------------------|
| `faceswap-backend`| FastAPI app — WebSocket face swap + voice      |
| `faceswap-nginx`  | Reverse proxy — SSL, rate limiting, WS routing |
| `gpu-monitor`     | Sidecar — polls nvidia-smi every 5s            |

---

## Environments

### Production (`docker-compose.prod.yml`)

```bash
cp .env.production .env    # Edit with your domain + tokens
./deploy.sh prod
```

- NGINX with HTTPS (port 80 → 301 → 443)
- TensorRT + FP16 + CUDA Graphs enabled
- CORS restricted to your domain
- GPU monitor sidecar running
- `restart: always` on all containers
- Log rotation (50MB × 5 files)
- Memory limit: 8GB

### Development (`docker-compose.dev.yml`)

```bash
./deploy.sh dev
```

- Hot reload (`uvicorn --reload`)
- Source code mounted as volume
- NGINX disabled (direct access to :8000)
- DEBUG logging
- TensorRT/FP16 disabled (faster startup)
- `restart: no`

### Testing (`docker-compose.test.yml`)

```bash
./deploy.sh test
```

- CPU-only (no GPU reservation)
- Runs `pytest` then exits
- JUnit XML results in `logs/test-results.xml`
- NGINX + GPU monitor disabled

---

## NGINX Configuration

`nginx/nginx.conf` provides:

- **HTTPS termination** — TLS 1.2/1.3 with self-signed certs
- **WebSocket proxying** — `/ws/swap` and `/ws/voice` with upgrade headers
- **Rate limiting** — 30 req/s for REST API, 10 req/s for WebSocket
- **Security headers** — HSTS, X-Frame-Options, X-Content-Type-Options
- **gzip compression** — JSON, JS, CSS, XML
- **HTTP → HTTPS redirect** — all port 80 traffic redirected to 443
- **Health passthrough** — `/health` accessible without SSL

### Let's Encrypt (Production SSL)

Replace self-signed certs with Let's Encrypt:

```bash
# Install certbot
sudo apt-get install -y certbot

# Generate certs
sudo certbot certonly --standalone -d your-domain.com

# Copy to nginx/ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem

# Set up auto-renewal (crontab)
echo "0 3 * * * certbot renew --quiet && docker restart faceswap-nginx" | sudo tee -a /var/spool/cron/crontabs/root
```

---

## GPU Monitoring

### Sidecar Container (`gpu-monitor`)

Runs alongside the backend in production:

```bash
./deploy.sh gpu         # Start the monitor
./deploy.sh logs gpu-monitor   # View metrics
```

Logs to `/app/logs/gpu-monitor.log` with format:

```
2025-01-15T10:30:00Z | temp=62C util=85% mem=3200/8192MiB power=145W inference=28ms
```

**Alerts** (written to same log):
- GPU temperature > 85°C
- GPU memory > 90% utilization

### Backend `/gpu` Endpoint

```bash
curl https://localhost/gpu | jq .
```

```json
{
    "name": "NVIDIA GeForce RTX 4070",
    "vram_total_mb": 8192,
    "vram_used_mb": 3200,
    "temperature_c": 62,
    "utilization_pct": 85,
    "avg_inference_ms": 28.5,
    "provider": "CUDAExecutionProvider",
    "fp16_enabled": true,
    "tensorrt_enabled": true
}
```

---

## Health Check

### Docker Health Check

The Dockerfile includes a `HEALTHCHECK` that calls `/health`:

```bash
docker inspect --format='{{.State.Health.Status}}' faceswap-backend
# healthy / unhealthy / starting
```

### Manual Check

```bash
./deploy.sh health
# or
curl -sf http://localhost:8000/health | jq .
```

```json
{
    "status": "ok",
    "models_loaded": true,
    "enhancer_enabled": false,
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 4070",
    "provider": "CUDAExecutionProvider",
    "required_models_ready": true
}
```

---

## Automatic Restart

All containers use `restart: always` (production) or `restart: unless-stopped` (base):

| Policy              | Behavior                                    |
|---------------------|---------------------------------------------|
| `always`            | Restart on crash, always (even after reboot)|
| `unless-stopped`    | Restart on crash, unless manually stopped   |

The backend has a 60-second `start_period` in the health check to allow
model loading before Docker considers it unhealthy.

---

## Logs

### Log Rotation

All containers use Docker's `json-file` driver with rotation:

```yaml
logging:
  driver: json-file
  options:
    max-size: "50m"    # Rotate at 50MB
    max-file: "5"      # Keep 5 rotated files
```

### Viewing Logs

```bash
# All services
./deploy.sh logs

# Specific service
./deploy.sh logs faceswap-backend
./deploy.sh logs nginx
./deploy.sh logs gpu-monitor

# Direct Docker
docker compose logs -f faceswap-backend
```

### NGINX Logs

```bash
# Access log
docker exec faceswap-nginx cat /var/log/nginx/access.log

# Error log
docker exec faceswap-nginx cat /var/log/nginx/error.log
```

### GPU Monitor Logs

```bash
cat logs/gpu-monitor.log
```

---

## Environment Variables

All settings use the `FACESWAP_` prefix. See `.env.example` for the full list.

### Key Variables

| Variable                          | Default          | Description                        |
|-----------------------------------|------------------|------------------------------------|
| `FACESWAP_PORT`                   | 8000             | Backend port                       |
| `FACESWAP_DOMAIN`                 | localhost        | Domain for SSL + CORS              |
| `FACESWAP_GPU_DEVICE_ID`          | 0                | GPU index (-1 for CPU)             |
| `FACESWAP_ENABLE_TENSORRT`        | true             | TensorRT execution provider        |
| `FACESWAP_ENABLE_FP16`            | true             | Half-precision inference           |
| `FACESWAP_ENABLE_CUDA_GRAPH`      | true             | CUDA Graph capture                 |
| `FACESWAP_LOG_LEVEL`              | INFO             | Logging level                      |
| `FACESWAP_CORS_ORIGINS`           | ["*"]            | Allowed CORS origins               |
| `FACESWAP_VOICE_ENABLED`          | false            | Enable voice pipeline              |
| `FACESWAP_PLATFORM_DISCORD_TOKEN` | (empty)          | Discord bot token                  |

### Production `.env`

```bash
cp .env.production .env
# Edit FACESWAP_DOMAIN, CORS origins, platform tokens
```

---

## CI/CD Pipeline

`ci-cd-template.yml` is the CI/CD pipeline template. GitHub workflows
cannot be written from the app builder — copy it to your repo:

```bash
cp backend/ci-cd-template.yml .github/workflows/backend-cicd.yml
git add .github/workflows/backend-cicd.yml
git commit -m "Add backend CI/CD pipeline"
git push
```

The pipeline runs on every PR and push to `main`.

### Stages

```
PR ──────────────────────────────────────────────────────
  │
  ├─ test    → lint (py_compile) + pytest (CPU mode)
  │
  └─ build   → Docker build + smoke test (container starts)

Push to main ────────────────────────────────────────────
  │
  ├─ test    → same as PR
  ├─ build   → same as PR
  ├─ push    → tag + push image to GHCR
  └─ deploy  → SSH to server → git pull → docker compose up
```

### Required GitHub Secrets (for deploy stage)

| Secret              | Description                        |
|---------------------|------------------------------------|
| `DEPLOY_HOST`       | SSH host of production server      |
| `DEPLOY_USER`       | SSH username                       |
| `DEPLOY_SSH_KEY`    | SSH private key                    |

### Skip CI/CD

Add `[skip ci]` to a commit message to skip the pipeline.

---

## File Structure

```
backend/
├── deploy.sh                  # One-command deployment
├── Dockerfile                 # Multi-stage GPU build
├── .dockerignore
├── docker-compose.yml         # Base config (shared)
├── docker-compose.prod.yml    # Production overlay
├── docker-compose.dev.yml     # Development overlay
├── docker-compose.test.yml    # Testing overlay
├── .env.example               # All env vars (template)
├── .env.production            # Production env template
├── .gitignore
├── pytest.ini
├── nginx/
│   ├── nginx.conf             # Reverse proxy + SSL + WS
│   └── ssl/                   # SSL certificates (generated)
├── scripts/
│   ├── entrypoint.sh          # Container entrypoint
│   ├── healthcheck.sh         # Docker health check
│   └── gpu-monitor.sh         # GPU monitoring daemon
├── tests/
│   └── test_backend.py        # pytest test suite
├── logs/                      # Runtime logs (gitignored)
├── models/                    # AI models (gitignored)
└── ...                        # Application code
```

---

## API Endpoints

| Method | Path                          | Description                   |
|--------|-------------------------------|-------------------------------|
| GET    | `/health`                     | Liveness + model status       |
| GET    | `/metrics`                    | FPS, latency, pipeline stats  |
| GET    | `/gpu`                        | GPU status + inference speed  |
| GET    | `/models`                     | Model metadata + health       |
| GET    | `/tracking`                   | Face tracking status          |
| GET    | `/enhancement`                | Enhancement mode + metrics    |
| PUT    | `/enhancement`                | Switch enhancement mode       |
| GET    | `/virtual-camera`             | Virtual camera status         |
| PUT    | `/virtual-camera`             | Enable/disable/configure VC   |
| GET    | `/voice`                      | Voice pipeline status         |
| PUT    | `/voice/microphone`           | Start/stop mic capture        |
| PUT    | `/voice/noise`                | Configure noise suppression   |
| PUT    | `/voice/conversion`           | Configure voice conversion    |
| PUT    | `/voice/echo`                 | Configure echo cancellation   |
| PUT    | `/voice/mute`                 | Mute/unmute/toggle            |
| GET    | `/platforms`                  | List platform adapters        |
| PUT    | `/platforms/{p}/connect`      | Connect to a platform         |
| PUT    | `/platforms/{p}/disconnect`   | Disconnect                    |
| PUT    | `/platforms/{p}/stream`       | Start/stop audio stream       |
| WS     | `/ws/swap`                    | Real-time face swap stream    |
| WS     | `/ws/voice`                   | Real-time voice change stream |

---

## Troubleshooting

### GPU not detected in container

```bash
# Check host
nvidia-smi

# Check Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If the above fails, reinstall nvidia-container-toolkit
```

### Backend unhealthy after deploy

```bash
# Check logs
./deploy.sh logs faceswap-backend

# Common causes:
# 1. Models not downloaded → see Model Download section
# 2. GPU not available → check nvidia-smi
# 3. Port conflict → change FACESWAP_PORT
```

### WebSocket connection fails through NGINX

```bash
# Check NGINX error log
docker exec faceswap-nginx cat /var/log/nginx/error.log

# Verify WebSocket upgrade headers are set (see nginx.conf)
# Test with wscat:
npm install -g wscat
wscat -c wss://localhost/ws/swap --no-check
```

### Out of GPU memory

```bash
# Check VRAM usage
nvidia-smi

# Reduce memory limit in .env:
FACESWAP_GPU_MEM_LIMIT_MB=2048

# Disable TensorRT (uses more VRAM for workspace):
FACESWAP_ENABLE_TENSORRT=false

# Disable CUDA Graphs:
FACESWAP_ENABLE_CUDA_GRAPH=false
`