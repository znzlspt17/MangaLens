#!/usr/bin/env bash
# =============================================================================
#  MangaLens — 실사용 테스트 스크립트
#  resources/test/ 의 실제 만화 이미지를 서버에 업로드하고
#  번역 결과를 output/test_run/ 에 저장합니다.
#
#  사용법:
#    ./test_run.sh              # 1.png 단일 이미지 테스트 (빠름)
#    ./test_run.sh --bulk       # 1~16.png 전체 일괄 업로드 테스트
#    ./test_run.sh --image 3    # 특정 번호 이미지 단일 테스트
#    ./test_run.sh --all        # 1~16.png 를 한 장씩 순서대로 테스트
#
#  옵션:
#    --host HOST   서버 주소 (기본: http://localhost:20399)
#    --out  DIR    결과 저장 디렉토리 (기본: output/test_run)
#    --poll SECS   상태 폴링 간격 초 (기본: 3)
#    --wait SECS   최대 대기 시간 초 (기본: 600)
# =============================================================================
set -euo pipefail

# ─── 기본값 ──────────────────────────────────────────────────────────────────
HOST="http://localhost:20399"
OUT_DIR="output/test_run"
POLL=3
WAIT=600
MODE="single"   # single | bulk | all
IMAGE_NUM="1"

# ─── 인자 파싱 ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bulk)   MODE="bulk";  shift ;;
    --all)    MODE="all";   shift ;;
    --image)  MODE="single"; IMAGE_NUM="$2"; shift 2 ;;
    --host)   HOST="$2";   shift 2 ;;
    --out)    OUT_DIR="$2"; shift 2 ;;
    --poll)   POLL="$2";   shift 2 ;;
    --wait)   WAIT="$2";   shift 2 ;;
    *) echo "알 수 없는 옵션: $1"; exit 1 ;;
  esac
done

TEST_DIR="resources/test"
mkdir -p "$OUT_DIR"

# ─── 색상 ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
fail() { echo -e "${RED}[FAIL]${RESET} $*"; exit 1; }

# ─── 의존성 확인 ──────────────────────────────────────────────────────────────
command -v curl &>/dev/null || fail "curl 이 없습니다. 설치 후 재시도하세요."

# jq 가 없으면 python3 로 대체 (간단한 .key // default | floor 패턴 지원)
if command -v jq &>/dev/null; then
  _jq() { jq "$@"; }
else
  warn "jq 없음 — python3 로 JSON 파싱합니다."
  _jq() {
    local filter=""
    for arg in "$@"; do
      case "$arg" in -r) ;; *) filter="$arg" ;; esac
    done
    python3 -c "
import sys, json, re
try:
    data = json.load(sys.stdin)
except Exception:
    print('null'); sys.exit(0)
f = sys.argv[1]
has_floor = bool(re.search(r'\|\s*floor', f))
f = re.sub(r'\s*\|\s*floor', '', f).strip()
m = re.match(r'^\.([\w.]+)(?:\s*//\s*(.+))?$', f)
if not m:
    print('null'); sys.exit(0)
val = data
for k in m.group(1).split('.'):
    val = val.get(k) if isinstance(val, dict) else None
    if val is None: break
if val is None and m.group(2):
    d = m.group(2).strip().strip('\"')
    try: val = int(d)
    except ValueError:
        try: val = float(d)
        except ValueError: val = d
if has_floor and val is not None:
    try: val = int(float(val))
    except Exception: pass
print('' if val is None else val)
" "$filter"
  }
fi

# ─── 서버 연결 확인 ───────────────────────────────────────────────────────────
log "서버 연결 확인: $HOST/api/health"
if ! curl -sf "$HOST/api/health" > /dev/null 2>&1; then
  fail "서버에 연결할 수 없습니다.\n  서버를 먼저 시작하세요: ./start.sh"
fi
ok "서버 응답 확인"

# ─── 상태 폴링 함수 ───────────────────────────────────────────────────────────
poll_until_done() {
  local task_id="$1"
  local elapsed=0
  while true; do
    local resp
    resp=$(curl -sf "$HOST/api/status/$task_id" 2>/dev/null) || {
      warn "상태 조회 실패 (task=$task_id), 재시도..."
      sleep "$POLL"; elapsed=$((elapsed + POLL)); continue
    }
    local status progress completed total failed
    status=$(echo "$resp"   | _jq -r '.status')
    progress=$(echo "$resp" | _jq -r '.progress // 0 | floor')
    completed=$(echo "$resp"| _jq -r '.completed_images // 0')
    total=$(echo "$resp"    | _jq -r '.total_images // 0')
    failed=$(echo "$resp"   | _jq -r '.failed_images // 0')

    printf "\r  상태: %-12s  진행률: %3d%%  완료: %d/%d  실패: %d   " \
      "$status" "$progress" "$completed" "$total" "$failed"

    case "$status" in
      completed) echo; ok "번역 완료!"; return 0 ;;
      partial)   echo; warn "일부 실패 (failed=$failed / total=$total)"; return 0 ;;
      failed)    echo; fail "파이프라인 실패 (task=$task_id)" ;;
    esac

    sleep "$POLL"
    elapsed=$((elapsed + POLL))
    if [[ $elapsed -ge $WAIT ]]; then
      echo; fail "타임아웃: ${WAIT}초 초과 (마지막 상태: $status)"
    fi
  done
}

# ─── 결과 다운로드 함수 ───────────────────────────────────────────────────────
download_result() {
  local task_id="$1"
  local dest="$2"
  log "결과 다운로드: $dest"
  local http_code
  http_code=$(curl -sf -w "%{http_code}" \
    "$HOST/api/result/$task_id" \
    -o "$dest" 2>/dev/null) || true

  if [[ "$http_code" == "200" ]]; then
    local size
    size=$(du -sh "$dest" 2>/dev/null | cut -f1)
    ok "저장 완료: $dest ($size)"
  else
    warn "다운로드 실패 (HTTP $http_code, task=$task_id)"
  fi
}

# =============================================================================
#  모드별 실행
# =============================================================================

echo -e "\n${BOLD}=== MangaLens 실사용 테스트 (mode=$MODE) ===${RESET}\n"

# ─── 단일 이미지 테스트 ───────────────────────────────────────────────────────
if [[ "$MODE" == "single" ]]; then
  IMG="$TEST_DIR/${IMAGE_NUM}.png"
  [[ -f "$IMG" ]] || fail "이미지 없음: $IMG"

  log "단일 업로드: $IMG"
  resp=$(curl -sf -X POST "$HOST/api/upload" \
    -F "file=@$IMG;type=image/png") || fail "업로드 요청 실패"

  task_id=$(echo "$resp" | _jq -r '.task_id')
  log "태스크 생성: $task_id"

  poll_until_done "$task_id"
  download_result "$task_id" "$OUT_DIR/${IMAGE_NUM}_translated.png"
fi

# ─── 전체 일괄 ZIP 업로드 ────────────────────────────────────────────────────
if [[ "$MODE" == "bulk" ]]; then
  ZIP_TMP=$(mktemp /tmp/mangalens_test_XXXXXX.zip)
  trap 'rm -f "$ZIP_TMP"' EXIT

  log "ZIP 패키징: resources/test/1~16.png → $ZIP_TMP"
  cd "$TEST_DIR" && zip -q "$ZIP_TMP" ./*.png && cd - > /dev/null
  ok "ZIP 생성 완료: $(du -sh "$ZIP_TMP" | cut -f1)"

  log "일괄 업로드 (ZIP)"
  resp=$(curl -sf -X POST "$HOST/api/upload/bulk" \
    -F "files=@$ZIP_TMP;type=application/zip") || fail "업로드 요청 실패"

  task_id=$(echo "$resp" | _jq -r '.task_id')
  log "태스크 생성: $task_id  (16장 일괄 처리)"

  poll_until_done "$task_id"
  DEST="$OUT_DIR/bulk_result_${task_id}.zip"
  download_result "$task_id" "$DEST"

  # ZIP이면 자동 압축 해제
  if file "$DEST" 2>/dev/null | grep -q "Zip archive"; then
    UNZIP_DIR="$OUT_DIR/bulk_${task_id}"
    mkdir -p "$UNZIP_DIR"
    unzip -q "$DEST" -d "$UNZIP_DIR"
    ok "압축 해제: $UNZIP_DIR ($(ls "$UNZIP_DIR" | wc -l)장)"
  fi
fi

# ─── 1~16 순차 단일 테스트 ───────────────────────────────────────────────────
if [[ "$MODE" == "all" ]]; then
  PASS=0; FAIL=0
  for i in $(seq 1 16); do
    IMG="$TEST_DIR/${i}.png"
    [[ -f "$IMG" ]] || { warn "$IMG 없음, 건너뜀"; continue; }

    echo -e "\n${BOLD}--- 이미지 $i/16 ---${RESET}"
    log "업로드: $IMG"

    resp=$(curl -sf -X POST "$HOST/api/upload" \
      -F "file=@$IMG;type=image/png") || {
      warn "업로드 실패: $IMG"
      FAIL=$((FAIL + 1)); continue
    }

    task_id=$(echo "$resp" | _jq -r '.task_id')
    log "태스크: $task_id"

    if poll_until_done "$task_id"; then
      download_result "$task_id" "$OUT_DIR/${i}_translated.png"
      PASS=$((PASS + 1))
    else
      FAIL=$((FAIL + 1))
    fi
  done

  echo -e "\n${BOLD}=== 결과 요약 ===${RESET}"
  echo -e "  통과: ${GREEN}${PASS}${RESET} / 전체: 16"
  [[ $FAIL -gt 0 ]] && echo -e "  실패: ${RED}${FAIL}${RESET}"
fi

echo -e "\n${CYAN}결과 저장 위치:${RESET} $OUT_DIR/"
ls -lh "$OUT_DIR"/ 2>/dev/null || true
echo
