#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  FaceSwap AI — One-Command Deployment Script
# ═══════════════════════════════════════════════════════════════
#  Usage:
#    ./deploy.sh              → production deploy (default)
#    ./deploy.sh dev          → development mode
#    ./deploy.sh test         → run tests
#    ./deploy.sh stop         → stop all services
#    ./deploy.sh logs         → tail logs
#    ./deploy.sh status       → show service status
#    ./deploy.sh ssl          → generate self-signed SSL certs
#    ./deploy.sh gpu          → start GPU monitor sidecar
#    ./deploy.sh health       → check health endpoint
#    ./deploy.sh clean        → remove containers + volumes
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }
err()  { echo -e "${RED}✗${NC} $1"; }

COMMAND="${1:-prod}"
COMPOSE="docker compose"

# ── Pre-flight checks ────────────────────────────────────────
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        err "Docker is not installed. Install: https://docs.docker.com/get-docker/"
        exit 1
    fi
    ok "Docker found"

    if ! docker info &> /dev/null; then
        err "Docker daemon is not running. Start it with: sudo systemctl start docker"
        exit 1
    fi
    ok "Docker daemon running"

    if ! $COMPOSE version &> /dev/null; then
        err "Docker Compose v2 not found. Install: https://docs.docker.com/compose/install/"
        exit 1
    fi
    ok "Docker Compose v2 found"

    # GPU check (only for prod/dev)
    if [ "$COMMAND" != "test" ] && [ "$COMMAND" != "clean" ]; then
        if command -v nvidia-smi &> /dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            ok "NVIDIA GPU detected: $GPU_NAME"
        else
            warn "nvidia-smi not found — backend will run in CPU-only mode (very slow)"
            warn "For GPU support, install NVIDIA drivers + nvidia-container-toolkit"
        fi
    fi
}

# ── Ensure .env exists ────────────────────────────────────────
ensure_env() {
    if [ ! -f .env ]; then
        log "Creating .env from .env.example..."
        cp .env.example .env
        ok ".env created — edit it to set your configuration"
    fi
}

# ── Ensure models directory exists ────────────────────────────
ensure_models() {
    if [ ! -d "models" ]; then
        log "Creating models directory..."
        mkdir -p models
        warn "Models directory created but is empty!"
        warn "Download models before starting:"
        warn "  wget -O models/inswapper_128.onnx https://huggingface.co/ezk77/inswapper_128/resolve/main/inswapper_128.onnx"
        warn "  wget -O models/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    fi
}

# ── SSL certificate generation ────────────────────────────────
generate_ssl() {
    log "Generating self-signed SSL certificates..."
    mkdir -p nginx/ssl

    if [ -f nginx/ssl/cert.pem ] && [ -f nginx/ssl/key.pem ]; then
        warn "SSL certificates already exist. Overwrite? (y/N)"
        read -r confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            ok "Keeping existing certificates"
            return
        fi
    fi

    DOMAIN="${FACESWAP_DOMAIN:-localhost}"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/key.pem \
        -out nginx/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=FaceSwap/CN=$DOMAIN" \
        -addext "subjectAltName=DNS:$DOMAIN,DNS:www.$DOMAIN,IP:127.0.0.1" 2>/dev/null

    ok "SSL certificates generated (nginx/ssl/cert.pem, nginx/ssl/key.pem)"
    warn "For production, replace with Let's Encrypt certificates"
}

# ── Commands ──────────────────────────────────────────────────

cmd_prod() {
    log "🚀 Deploying FaceSwap AI — PRODUCTION"
    check_prerequisites
    ensure_env
    ensure_models
    generate_ssl

    log "Building and starting production containers..."
    $COMPOSE -f docker-compose.yml -f docker-compose.prod.yml up -d --build

    log "Waiting for backend to become healthy..."
    sleep 10
    if curl -sf http://localhost:8000/health &> /dev/null; then
        ok "Backend is healthy!"
    else
        warn "Backend not yet healthy — models may still be loading (up to 60s)"
        warn "Check status with: ./deploy.sh status"
    fi

    ok "Production deployment complete!"
    echo ""
    echo -e "  ${CYAN}Backend:${NC}  http://localhost:8000"
    echo -e "  ${CYAN}HTTPS:${NC}     https://localhost"
    echo -e "  ${CYAN}Health:${NC}    ./deploy.sh health"
    echo -e "  ${CYAN}Logs:${NC}      ./deploy.sh logs"
    echo ""
}

cmd_dev() {
    log "🔧 Starting FaceSwap AI — DEVELOPMENT"
    check_prerequisites
    ensure_env

    $COMPOSE -f docker-compose.yml -f docker-compose.dev.yml up -d --build

    sleep 5
    ok "Development environment started!"
    echo ""
    echo -e "  ${CYAN}Backend:${NC}  http://localhost:8000"
    echo -e "  ${CYAN}Docs:${NC}     http://localhost:8000/docs"
    echo -e "  ${CYAN}Hot reload:${NC} enabled — edit code and watch changes"
    echo ""
}

cmd_test() {
    log "🧪 Running FaceSwap AI — TESTS"
    check_prerequisites

    $COMPOSE -f docker-compose.yml -f docker-compose.test.yml up -d --build
    sleep 5

    log "Running test suite..."
    $COMPOSE -f docker-compose.yml -f docker-compose.test.yml exec faceswap-backend \
        python -m pytest tests/ -v --tb=short || true

    log "Cleaning up test containers..."
    $COMPOSE -f docker-compose.yml -f docker-compose.test.yml down -v
    ok "Tests complete"
}

cmd_stop() {
    log "Stopping all FaceSwap services..."
    $COMPOSE -f docker-compose.yml -f docker-compose.prod.yml down 2>/dev/null || true
    $COMPOSE -f docker-compose.yml -f docker-compose.dev.yml down 2>/dev/null || true
    $COMPOSE down 2>/dev/null || true
    ok "All services stopped"
}

cmd_logs() {
    SERVICE="${2:-}"
    if [ -n "$SERVICE" ]; then
        $COMPOSE logs -f "$SERVICE"
    else
        $COMPOSE logs -f
    fi
}

cmd_status() {
    log "Service status:"
    $COMPOSE ps
    echo ""
    log "Health check:"
    if curl -sf http://localhost:8000/health 2>/dev/null; then
        echo ""
        ok "Backend is healthy"
    else
        err "Backend is not responding"
    fi
    echo ""
    log "GPU status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total \
            --format=csv
    else
        warn "nvidia-smi not available"
    fi
}

cmd_health() {
    log "Checking backend health..."
    if curl -sf http://localhost:8000/health 2>/dev/null | python3 -m json.tool; then
        ok "Backend healthy"
    else
        err "Backend not responding"
        exit 1
    fi
}

cmd_gpu_monitor() {
    log "Starting GPU monitor sidecar..."
    $COMPOSE -f docker-compose.yml -f docker-compose.prod.yml up -d gpu-monitor
    ok "GPU monitor started — logs at: ./deploy.sh logs gpu-monitor"
}

cmd_clean() {
    warn "This will remove ALL containers, volumes, and networks. Continue? (y/N)"
    read -r confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        $COMPOSE -f docker-compose.yml -f docker-compose.prod.yml down -v --rmi local 2>/dev/null || true
        $COMPOSE -f docker-compose.yml -f docker-compose.dev.yml down -v --rmi local 2>/dev/null || true
        $COMPOSE down -v --rmi local 2>/dev/null || true
        ok "All containers, volumes, and images removed"
    else
        ok "Clean cancelled"
    fi
}

# ── Main ──────────────────────────────────────────────────────
case "$COMMAND" in
    prod|up)        cmd_prod ;;
    dev)            cmd_dev ;;
    test)           cmd_test ;;
    stop|down)      cmd_stop ;;
    logs)           cmd_logs "$2" ;;
    status|ps)      cmd_status ;;
    health)         cmd_health ;;
    ssl)            generate_ssl ;;
    gpu)            cmd_gpu_monitor ;;
    clean)          cmd_clean ;;
    *)
        echo "Usage: $0 {prod|dev|test|stop|logs|status|health|ssl|gpu|clean}"
        echo ""
        echo "Commands:"
        echo "  prod     Deploy production stack (default)"
        echo "  dev      Start development environment with hot reload"
        echo "  test     Run test suite"
        echo "  stop     Stop all services"
        echo "  logs     Tail logs (optional: service name)"
        echo "  status   Show service + GPU + health status"
        echo "  health   Check /health endpoint"
        echo "  ssl      Generate self-signed SSL certificates"
        echo "  gpu      Start GPU monitoring sidecar"
        echo "  clean    Remove all containers, volumes, images"
        exit 1
        ;;
esac