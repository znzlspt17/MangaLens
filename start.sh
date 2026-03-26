#!/usr/bin/env bash
# MangaLens — 원클릭 실행 스크립트
# 사용법: ./start.sh [--port PORT] [--host HOST] [--skip-download] [--dev]
set -euo pipefail

# ── 색상 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[MangaLens]${NC} $*"; }
success() { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*" >&2; }

# ── 옵션 파싱 ─────────────────────────────────────────────────────────────
PORT=20399        # FastAPI 서버 (프론트엔드 + API 통합)
HOST=0.0.0.0
SKIP_DOWNLOAD=0
DEV_MODE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)        PORT="$2";         shift 2 ;;
    --host)        HOST="$2";         shift 2 ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --dev)         DEV_MODE=1;        shift ;;
    -h|--help)
      echo "사용법: $0 [--port PORT] [--host HOST] [--skip-download] [--dev]"
      echo "  --port PORT           서버 포트 (기본: 20399)"
      echo "  --host HOST           바인드 주소 (기본: 0.0.0.0)"
      echo "  --skip-download       모델/폰트 다운로드 건너뜀"
      echo "  --dev                 uvicorn --reload 개발 모드"
      exit 0 ;;
    *) error "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

# ── 작업 디렉토리 ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "작업 경로: $SCRIPT_DIR"

# ── uv 확인 ───────────────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  error "uv 패키지 매니저를 찾을 수 없습니다."
  echo "  설치 방법: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
success "uv $(uv --version)"

# ── .env 파일 ─────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    warn ".env 파일이 없습니다. .env.example 에서 복사합니다."
    cp .env.example .env
  else
    error ".env 파일이 없습니다."
    exit 1
  fi
fi

# ── ROCm 환경 자동 설정 ───────────────────────────────────────────────────
if command -v rocm-smi &>/dev/null; then
  if [[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    # Detect actual GPU architecture from rocminfo
    if command -v rocminfo &>/dev/null; then
      _gfx=$(rocminfo 2>/dev/null | grep -oP 'gfx\K[0-9]+' | head -1)
      if [[ -n "$_gfx" ]]; then
        # Convert e.g. 1201 -> 12.0.1
        _major=${_gfx:0:2}
        _minor=${_gfx:2:1}
        _patch=${_gfx:3:1}
        export HSA_OVERRIDE_GFX_VERSION="${_major}.${_minor}.${_patch}"
        export PYTORCH_ROCM_ARCH="gfx${_gfx}"
        info "ROCm gfx${_gfx} 감지 → HSA_OVERRIDE_GFX_VERSION=${_major}.${_minor}.${_patch}"
      fi
    fi
    if [[ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
      export HSA_OVERRIDE_GFX_VERSION=12.0.1
      export PYTORCH_ROCM_ARCH=gfx1201
      info "ROCm 기본값 → HSA_OVERRIDE_GFX_VERSION=12.0.1"
    fi
  fi
  if [[ -z "${PYTORCH_ROCM_ARCH:-}" ]]; then
    export PYTORCH_ROCM_ARCH=gfx1201
  fi
fi

# ── 의존성 설치 ───────────────────────────────────────────────────────────
info "의존성 확인 중..."
uv sync --quiet
success "의존성 설치 완료"

# ── 모델 & 폰트 다운로드 ──────────────────────────────────────────────────
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  MODELS_PRESENT=1
  for f in models/comictextdetector.pt models/big-lama.pt \
            models/RealESRGAN_x2plus.pth models/RealESRGAN_x4plus.pth; do
    [[ -f "$f" ]] || { MODELS_PRESENT=0; break; }
  done
  FONT_PRESENT=0
  ls fonts/*.ttf &>/dev/null 2>&1 && FONT_PRESENT=1

  if [[ $MODELS_PRESENT -eq 0 ]] || [[ $FONT_PRESENT -eq 0 ]]; then
    info "모델/폰트 다운로드 중... (최초 실행 시 수 분 소요)"
    uv run python -m server.download
    success "모델/폰트 다운로드 완료"
  else
    success "모델 & 폰트 이미 존재 — 다운로드 건너뜀"
  fi
else
  warn "모델/폰트 다운로드 건너뜀 (--skip-download)"
fi

# ── 서버 실행 ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  MangaLens 서버 시작"
echo -e "  URL: http://localhost:${PORT}"
echo -e "  종료: Ctrl+C"
echo -e "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ── FastAPI 서버 (프론트엔드 + API 통합, 포트 20399) ──────────────────────
if [[ $DEV_MODE -eq 1 ]]; then
  HOST="$HOST" PORT="$PORT" uv run uvicorn server.main:app --host "$HOST" --port "$PORT" --reload
else
  HOST="$HOST" PORT="$PORT" uv run python -m server.main
fi
