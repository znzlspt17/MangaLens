#!/usr/bin/env bash
# MangaLens — 서버 종료 스크립트
# 이 디렉터리에서 실행된 server.main / uvicorn 프로세스만 정확히 종료합니다.

# ── 색상 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[MangaLens]${NC} $*"; }
success() { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*" >&2; }

# ── 작업 디렉토리 ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── MangaLens 프로세스 탐색 ───────────────────────────────────────────────
# 조건: cmdline에 'server.main' 또는 'uvicorn server.main:app' 포함
#        AND /proc/<pid>/cwd 가 이 디렉터리와 일치
find_manglens_pids() {
    local -a result=()
    local -a candidates=()

    # server.main 패턴으로 후보 수집 (uv + python 두 프로세스 모두 포함됨)
    while IFS= read -r pid; do
        candidates+=("$pid")
    done < <(pgrep -f "server[.]main" 2>/dev/null)

    for pid in "${candidates[@]}"; do
        # /proc/<pid>/cwd 가 SCRIPT_DIR 와 일치하는지 확인
        local cwd
        cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null) || continue
        if [[ "$cwd" == "$SCRIPT_DIR" ]]; then
            result+=("$pid")
        fi
    done

    printf '%s\n' "${result[@]}"
}

mapfile -t PIDS < <(find_manglens_pids)

if [[ ${#PIDS[@]} -eq 0 ]]; then
    warn "실행 중인 MangaLens 프로세스가 없습니다."
    exit 0
fi

# ── 프로세스 정보 출력 ────────────────────────────────────────────────────
info "종료 대상 프로세스 (${#PIDS[@]}개):"
for pid in "${PIDS[@]}"; do
    cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-80)
    printf "  PID %-7s  %s\n" "$pid" "$cmdline"
done
echo ""

# ── SIGTERM (정상 종료 요청) ──────────────────────────────────────────────
for pid in "${PIDS[@]}"; do
    if kill -TERM "$pid" 2>/dev/null; then
        info "SIGTERM → PID $pid"
    fi
done

# ── 최대 5초 대기 ─────────────────────────────────────────────────────────
TIMEOUT=5
ELAPSED=0
while [[ $ELAPSED -lt $TIMEOUT ]]; do
    ALIVE=0
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && { ALIVE=1; break; }
    done
    [[ $ALIVE -eq 0 ]] && break
    sleep 1
    (( ELAPSED++ )) || true
done

# ── 여전히 살아있으면 SIGKILL ─────────────────────────────────────────────
KILLED=0
for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        warn "SIGKILL (강제) → PID $pid"
        kill -KILL "$pid" 2>/dev/null && (( KILLED++ )) || true
    fi
done

if [[ $KILLED -gt 0 ]]; then
    warn "${KILLED}개 프로세스를 강제 종료했습니다."
fi

success "MangaLens 서버가 종료되었습니다."
