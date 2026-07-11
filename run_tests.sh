#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  FaceSwap AI — Test Runner
#  Runs the full 16-module test suite and generates reports
# ═══════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  🎭 FaceSwap AI — Comprehensive Test Suite"
echo "  16 modules covering all system components"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ── Colors ────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Check Python ──────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

echo -e "${CYAN}Python:${NC} $(python3 --version)"

# ─<arg_value> Install test dependencies ───────────────────────────────────
echo -e "${CYAN}Installing test dependencies...${NC}"
pip install -q pytest pytest-asyncio httpx 2>/dev/null || true

# ── Run tests ──────────────────────────────────────────────────
if [ "$1" == "--report" ] || [ "$1" == "-r" ]; then
    # Run with custom report generator
    echo -e "${CYAN}Running full suite with report generation...${NC}"
    python3 tests/generate_report.py
else
    # Run with pytest directly
    echo -e "${CYAN}Running pytest...${NC}"
    echo ""
    python3 -m pytest tests/ \
        -v \
        --tb=short \
        --junitxml=logs/test-reports/junit-full.xml \
        -p no:cacheprovider \
        || true

    # Also generate the HTML/JSON report
    echo ""
    echo -e "${CYAN}Generating detailed reports...${NC}"
    python3 tests/generate_report.py || true
fi

echo ""
echo -e "${CYAN}Reports saved to:${NC} logs/test-reports/"
echo -e "  ${CYAN}HTML:${NC} logs/test-reports/test-report.html"
echo -e "  ${CYAN}JSON:${NC} logs/test-reports/test-report.json"