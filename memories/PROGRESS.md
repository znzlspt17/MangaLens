# MangaLens — 진행 상황 추적

> 최종 갱신: 2026-04-03 (Phase 19: 연결 말풍선 분리 + bbox 확장 + 비말풍선 허위 양성 방지)

---

## 현재 시스템 상태

| 구성 요소 | 상태 | 비고 |
|-----------|------|------|
| 서버 (FastAPI) | ✅ 정상 동작 | port 20399, uvicorn |
| GPU | ✅ CUDA | NVIDIA RTX 4090, VRAM 24564 MB, CUDA 12.8 |
| BubbleDetector | ✅ ML 모드 | comic-text-detector YOLOv5 on CUDA (기본) |
| MagiDetector | ✅ 정상 (P0 해결) | `USE_MAGI_DETECTOR=true`로 활성화 — bbox 마스크 자동 생성 |
| Preprocessor (Real-ESRGAN) | ✅ 정상 | device=cuda, anime_6B 기본 (x4plus fallback) |
| OCREngine (manga-ocr) | ✅ 정상 | kha-white/manga-ocr-base on CUDA |
| TextEraser (LaMa) | ✅ 정상 | models/big-lama.pt |
| TextRenderer (Pillow) | ✅ 정상 | fonts/NotoSansKR-Regular.ttf |
| 번역 (Hunyuan-MT-7B) | ✅ 로컬 모델 | tencent/Hunyuan-MT-7B, 7B파라미터, fp16, 외부 API 키 불필요 |
| 프론트엔드 | ✅ 정상 | index.html, 다크모드, 반응형 |

---

## 완료된 작업 이력

### 2026-03-25 (오늘)

#### 버그 수정
| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `vendor/basicsr-1.4.2/setup.py` | Python 3.14 PEP 667 — `get_version()` exec namespace 명시적 dict 사용 |
| 2 | `vendor/basicsr-1.4.2/basicsr/data/degradations.py` | `torchvision.transforms.functional_tensor` 제거됨 → try/except ImportError fallback |
| 3 | `.venv/.../basicsr/data/degradations.py` | 동일 패치 |
| 4 | `start.sh` | `set -eu pipefail` → `set -euo pipefail` (bash 문법 오류 수정) |
| 5 | `server/main.py` | warm-up에서 `await` 누락 → RuntimeWarning 3건 수정 |
| 6 | `server/pipeline/orchestrator.py` | `font_dir` 하드코딩 `"./fonts"` → `_server_settings.font_dir` |
| 7 | `server/pipeline/text_renderer.py` | 비-variable 폰트에 `set_variation_by_axes()` 반복 호출 → `_is_variable_font` 초기화 시 캐시 |
| 8 | `.env` | `DEEPL_API_KEY=your_deepl_api_key_here` (placeholder) → 비워둠 |
| 9 | `server/routers/settings.py` | `secure=True` 쿠키 → HTTP 환경에서 저장 안 됨 → `request.url.scheme == "https"` 조건부 |
| 10 | `pyproject.toml` | `ultralytics>=8.0` 누락 → BubbleDetector가 OpenCV fallback 사용 → ML 모드 복원 |

#### 기능 추가
| # | 파일 | 추가 내용 |
|---|------|-----------|
| 1 | `server/schemas/models.py` | `UserSettingsResponse`에 `session_id: str \| None` 필드 추가 |
| 2 | `server/routers/settings.py` | GET/POST `/api/settings` 응답에 `session_id` 포함 |
| 3 | `frontend/js/api.js` | localStorage 기반 세션 관리 (`_SESSION_KEY`, `setSessionId()`), 모든 요청에 `X-Session-Id` 헤더 자동 주입 |
| 4 | `frontend/js/settings.js` | 저장/조회 후 `API.setSessionId(data.session_id)` 호출, `closeModal()` 시 입력값 유지, `checkAndShowBanner()` 초기화 |
| 5 | `frontend/js/upload.js` | `setServerReady()` 추가, 버튼 파일 수 표시, 서버 미준비 시 비활성화+툴팁 |
| 6 | `frontend/js/result.js` | 다운로드 버튼 로딩 피드백, 다시 시도 버튼 연결 |
| 7 | `frontend/js/app.js` | `_updateSteps()` 단계 인디케이터, `Upload.setServerReady()` 헬스체크 연동 |
| 8 | `frontend/index.html` | 3단계 진행 바, API 키 경고 배너, 히어로 배너, 업로드 카드 그룹핑, 다시 시도 버튼 |
| 9 | `frontend/css/style.css` | step-bar, api-key-banner, hero, upload-card, 배경 orb 애니메이션, 헤더 glassmorphism |
| 10 | `frontend/css/themes.css` | 다크/라이트 CSS 변수 전면 정비 (`--gradient-text`, `--header-bg`, `--shadow-md` 등) |

#### 2026-03-25 (오늘) — 2차 작업

##### 버그 수정
| # | 파일 | 수정 내용 |
|---|------|-----------|
| 11 | `server/pipeline/text_renderer.py` | `render()` 반환 타입 `np.ndarray` → `tuple[np.ndarray, int]` — `font_size_used` 항상 0으로 기록되던 문제 수정 |
| 12 | `server/pipeline/orchestrator.py` | `renderer.render()` 튜플 언패킹 + `ctx.font_size_used` 업데이트 연결 |
| 13 | `server/pipeline/text_renderer.py` | `_draw_vertical()` — `max_cols = usable_w // col_width` 기준 `chars_per_col` 적응형 계산, 마지막 컬럼 글자 누락(잘림) 버그 수정 |

##### 기능 추가
| # | 파일 | 추가 내용 |
|---|------|-----------|
| 11 | `server/pipeline/preprocessor.py` | `remove_furigana()` — connected component 높이 중앙값 60% 미만 글리프를 후리가나로 판단해 흰색 마스킹, 업스케일 직후 OCR 전 자동 적용 |

---

#### PM 프로젝트 전체 검토
| # | 항목 | 내용 |
|---|------|------|
| 1 | 코드베이스 규모 | **7,787줄** / 47개 파일 (서버 2,588 + 파이프라인 1,544 + 프론트 2,286 + 테스트 2,104) |
| 2 | PLAN.md 원칙 준수 | P1~P7 완벽 준수, P8(GPU 메모리 관리) 부분 미흡 |
| 3 | 종합 점수 | **7.9 / 10** |

---

## 2026-03-26 — Phase 15: 렌더링 버그 수정

### 현상
말풍선 텍스트 줄바꿈 후 가장 긴 줄의 **왼쪽 글자(스트로크 포함)가 오버레이 경계 밖으로 잘리는** 문제 발생.

### 원인 분석
`_wrap_text()`에 전달되는 `usable_w`가 스트로크 너비를 고려하지 않아, `line_w ≈ usable_w`인 긴 줄이 `x = _PADDING(6px)`에서 시작할 때 스트로크가 `_PADDING - stroke_width < 0` 좌표로 그려져 클리핑됨.

### 수정 내용

| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `server/pipeline/text_renderer.py` | `render()` — `wrap_w = max(usable_w - 2 × stroke_width, 1)` 도입하여 줄바꿈 기준 너비를 스트로크 여유분만큼 축소 |
| 2 | `server/pipeline/text_renderer.py` | `_draw_horizontal()` — `x = max(stroke_width, _PADDING + …)` 하한 보장으로 스트로크 좌측 경계 음수 좌표 방지 |

---

## 2026-03-27 — Phase 16: 탁음/반탁음 + 텍스트 크기 버그 수정

### 버그 1: 탁음/반탁음이 후리가나로 오인식되어 지워지는 문제

**원인:** `remove_furigana()`의 근접도 패딩이 `median_h × 0.15`(≈4px)로 너무 작아 실제 11-29px 떨어진 탁음 점(゛゜)이나 ellipsis(．)까지 삭제됨. 또한 후리가나는 별도 서브컬럼에 위치하므로 column-overlap 기준이 없었음.

**수정:**
| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `server/pipeline/preprocessor.py` | `pad = max(median_h * 0.15, 3.0)` → `max(median_h * 0.30, 3.0)` (패딩 2× 증가) |
| 2 | `server/pipeline/preprocessor.py` | column-overlap 체크 추가: 소형 component의 x범위가 대형 component 중 하나와 겹치면 보존 (`if sx1 < lx2 and sx2 > lx1: near_large = True`) |

**검증:** P22 ID3 기준 잘못 삭제된 픽셀 612 → 0

---

### 버그 2: 번역 텍스트가 너무 크거나 말풍선 밖으로 넘치는 문제

**원인:** 수직 말풍선(세로쓰기 원본)에서 `fit_h = max(bw, _MIN_FONT_SIZE * 3)` — 좁은 bw=24px 말풍선에서 `fit_h=36, usable_h=24` → 폰트 바이너리 탐색이 항상 최소값(12px) 반환. 또한 `actual_render_h = needed_h`에 상한이 없어 bh 초과 가능.

**수정:**
| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `server/pipeline/text_renderer.py` | `fit_h = max(bw, _MIN_FONT_SIZE * 3)` → `fit_h = bh` (전체 말풍선 높이 사용) |
| 2 | `server/pipeline/text_renderer.py` | `actual_render_h = needed_h` → `actual_render_h = min(needed_h, bh)` (bh 초과 방지) |

**검증:** 폰트 크기 12px → 25px 이상으로 정상화, 세로 오버플로우 제거

---

### Phase 16 최종 결과

| 항목 | 수치 |
|------|------|
| 전체 테스트 페이지 | 33/33 성공 |
| 총 번역 버블 | 282개 |
| 미번역 버블 | 0개 (0%) |
| 평균 OCR 신뢰도 | 0.935 |
| pytest 통과 수 | 126/126 ✅ |

---

## 알려진 이슈 (PM 검토 결과)

### P0 — 모두 해결됨 ✅

| 이슈 | 파일 | 상태 |
|------|------|------|
| 모델 캐시 Race Condition | `server/pipeline/orchestrator.py` | ✅ 해결 — `async def _get_cached` + `asyncio.Lock` 이중 확인 잠금 |

### P1 — 모두 해결됨 ✅

| 이슈 | 파일 | 상태 |
|------|------|------|
| GPU 텐서 미해제 | `server/pipeline/text_eraser.py` | ✅ 해결 — `del img_t, mask_t, result` + `torch.cuda.empty_cache()` |
| task_store 동시성 | `server/state.py` | ✅ 해결 — pub/sub Queue 인프라 추가 (`subscribe/unsubscribe/notify_task_changed`) |
| Semaphore 초기화 경합 | `server/routers/upload.py` | ✅ 해결 — `asyncio.Lock` 보호 `async def _get_semaphore()` + notify 3개소 |

### P2 — 모두 해결됨 ✅

| 이슈 | 파일 | 상태 |
|------|------|------|
| WebSocket 폴링 비확장 | `server/routers/ws.py` | ✅ 해결 — 1초 폴링 → 이벤트 기반 Queue (5초 heartbeat) |
| font.getlength() 반복 | `server/pipeline/text_renderer.py` | ✅ 해결 — 모듈 레벨 `_font_cache` dict |
| 라이트모드 색상 대비 | `frontend/css/themes.css` | ✅ 해결 — WCAG AA: `#6b7280`→`#4b5563` (secondary), `#9ca3af`→`#6b7280` (muted) |
| WS 메시지 파싱 에러 누락 | `frontend/js/progress.js` | ✅ 해결 — `console.warn` 로깅 추가 |

### 낮음 — 기존 이슈

| 이슈 | 심각도 | 내용 |
|------|--------|------|
| HF_TOKEN 미설정 경고 | 낮음 | HuggingFace rate limit 경고. 기능에는 영향 없음 |
| basicsr `.venv` 패치 | 낮음 | `uv sync` 재실행 시 재패치 필요 (자동화 미구현) |
| Hunyuan 첫 요청 지연 | 낮음 | 마지막 비활성 시 ~5초 VRAM 로드. lifespan warm-up으로 개선 가능 (D-014 후속고려) |
| 효과음 검출 불가 | 낮음 | `effects_skipped = 0` — 효과음(effect 타입) 말풍선이 감지되지 않음 |

---

## Phase 상태 요약

| Phase | 완료 여부 |
|-------|-----------|
| Phase 1: 기반 구축 | ✅ |
| Phase 2: 핵심 파이프라인 | ✅ |
| Phase 3: 렌더링 & 합성 | ✅ |
| Phase 4: 서버 완성 | ✅ |
| Phase 5: 테스트 & 최적화 | ✅ |
| Phase 6: 프론트엔드 | ✅ |
| Phase 7: UI 개선 & 버그 수정 | ✅ |
| PM 전체 검토 | ✅ 종합 7.9/10 |
| Phase 8: 모델 업그레이드 | ✅ |
| Phase 9: 탁점 보호 & UX 개선 | ✅ |
| Phase 10: Hunyuan-MT-7B 번역 엔진 마이그레이션 | ✅ |
| Phase 11: P0–P2 버그 수정 스프린트 | ✅ |
| Phase 12: 전체 코드 리뷰 (5개 담당 검토) | ✅ 검토 완료 |
| Phase 13: 전체 이슈 수정 (26건 P0-P2) | ✅ 181 tests passed |
| Phase 14: 로깅 시스템 + 런타임 버그 수정 | ✅ 181 tests passed |
| Phase 15: 렌더링 말풍선 좌측 글자 잘림 수정 | ✅ |
| Phase 16: 탁음/반탁음 오삭제 + 텍스트 크기 과대/과소 수정 | ✅ 126 tests passed, 33/33 pages |
| Phase 17: 노이즈 근본 원인 7건 수정 (N1~N7) | ✅ 187 tests passed |
| Phase 18: 타원 말풍선 안전 여백 강화 (_PADDING_PCT 0.08→0.15) | ✅ 64 tests passed |
| Phase 19: 연결 말풍선 분리 + bbox 확장 + 어두운 내부 허위 양성 방지 | ✅ |

---

## 2026-03-27 — Phase 17: 노이즈 근본 원인 분석 + 수정

### 분석 데이터
- `output/phase19_final/` — 33페이지, 총 273개 말풍선 전수 분석
- 미번역 8건(3%), 폰트 12px 강제 67건(25%), 세로 버블 전체가 24px 캡에 묶힘

### 수정 목록 (7건)

| # | 코드 | 파일 | 변경 전 | 변경 후 |
|---|------|------|---------|---------|
| N1 | `_PROXIMITY_GAP` | `bubble_detector.py` | 10 | 25 (세로쓰기 컬럼 합치기 개선) |
| N2 | scale-down 제거 | `text_renderer.py` | `font_size × 0.85` (60px 초과 시) | 제거 — `_find_best_font_size`에 위임 |
| N3 | `_MAX_VERT_FONT_SIZE` | `text_renderer.py` | 24 | 36 (세로→가로 변환 캡 완화) |
| N4 | 가나 감지 | `translator.py` | fallback = 원문 반환 | `_JA_KANA_RE` — 가나 포함 시 `""` 반환 |
| N5 | 간결성 지시 | `translator.py` | 6개 규칙 | 7번째 규칙 "원문과 유사한 분량 유지" 추가 |
| N6 | inpainting dilate | `text_eraser.py` | `(9,9)` kernel + iterations=3 | `(7,7)` kernel + iterations=2 |
| N7 | 자동 글자색 | `text_renderer.py` | 항상 검정 | bbox 평균 밝기 < 128 → 흰 글씨 / 검정 윤곽 |

### Phase 17 최종 결과

| 항목 | 수치 |
|------|------|
| pytest 통과 수 | **187/187** ✅ |
| 수정 파일 수 | 4 (`bubble_detector.py`, `text_renderer.py`, `text_eraser.py`, `translator.py`) |

---

## 2026-03-31 — Phase 18: 타원 말풍선 안전 여백 강화

### 현상
이미지 14에서 말풍선 텍스트(특히 "게요.") 하단이 타원 경계 밖으로 잘리는 클리핑 발생. 대형 타원형 말풍선(bh=715~968)에서 텍스트가 위/아래 타원 호 바깥에 그려짐.

### 원인 분석
`_PADDING_PCT = 0.08` — 타원 내에 내접하는 직사각형의 이론 여백 `(1 - 1/√2)/2 ≈ 0.146`보다 작아 실제 타원형 말풍선의 곡률을 커버하지 못함. 직사각형 bbox 기준으로는 여백처럼 보여도 타원 호 안쪽으로는 공간이 더  좁기 때문.

### 수정 내용

| # | 파일 | 수정 내용 |
|---|------|-----------|
| 1 | `server/pipeline/text_renderer.py` | `_PADDING_PCT = 0.08` → `0.15` (타원 내접 직사각형 이론값 ≥ 0.146에 안전 여유 포함) |

### Phase 18 최종 결과

| 항목 | 수치 |
|------|------|
| pytest 통과 수 | **64/64** ✅ |
| 수정 파일 수 | 1 (`text_renderer.py`) |
| 영향 말풍선 | 모든 가로쓰기 말풍선 — usable 영역 bw×84% → bw×70% |
| 클리핑 해소 | 이미지 14 balloon[0]/[4]/[5] 포함 대형 타원 전체 정상 확인 |

---

## 2026-04-03 — Phase 19: 연결 말풍선 분리 + bbox 확장 + 허위 양성 방지

### 배경
실제 번역 결과에서 (1) 세로로 붙어 있는 두 말풍선이 하나로 뭉쳐 검출, (2) YOLO bbox가 실제 말풍선보다 좁아 텍스트 렌더링 공간 부족·글자 잘림, (3) 어두운 장면 텍스트(장르 특유의 효과음/배경 대사) bbox가 확장으로 인해 주변 영역까지 침범하는 허위 양성 발생.

### Magi v2 패키지 수정

| # | 파일 | 수정 내용 |
|---|------|----------|
| 1 | `pyproject.toml` | `einops>=0.8`, `pulp>=2.9` 의존성 추가 — `USE_MAGI_DETECTOR=true` 시 `ModuleNotFoundError` 해소 |

### bubble_detector.py 신규 함수 3개

#### 1. `_split_tall_boxes()` — 연결 말풍선 자동 분리

**현상**: `bubble_005.png` — 두 말풍선이 가운데서 좁게 연결돼 하나의 크고 세로로 긴 bbox로 검출  
**원인**: YOLO가 연결 영역을 포함하여 단일 박스로 예측  
**해결**: 텍스트 세그 마스크 행 밀도 스캔 → 핀치 포인트(peak의 30% 미만 골짜기) 탐지 → 상하로 분리  

| 조건 | 기준 |
|------|------|
| 분리 시도 기준 비율 | `h/w ≥ 3.0` |
| 최소 높이 | `h ≥ 80px` |
| 골짜기 임계값 | `밀도 < peak × 0.30` |
| 각 파트 최소 크기 | `≥ 30px` |

#### 2. `_expand_bbox_to_balloon()` — 실제 말풍선 경계까지 bbox 확장

**현상**: 좁은 YOLO bbox 안에 텍스트가 잘려 12px 폰트 강제, 글자 누락  
**해결**: 원본 이미지 밝기 기준으로 4방향 탐색 → 흰 말풍선 경계 직전까지 확장  

| 상수 | 값 | 설명 |
|------|-----|------|
| `_EXPAND_BRIGHTNESS` | 200 | 확장 중단 밝기 임계값 |
| `_EXPAND_MAX_PX` | 80 | 방향별 최대 확장 픽셀 |
| `_EXPAND_INSET` | 4 | 안전 여백 (경계에서 안쪽) |

**seg_full 검증**: 확장 면적이 1.5배 이상이면 세그 밀도 비교 → 1/3 미만이면 확장 취소  
**허위 양성 방지** (D-037): 내부 평균 밝기 < 160이면 확장 건너뜀  

**제거**: `text_renderer.py`의 `_probe_balloon_width()` (80줄) — 역할이 탐지기 계층으로 완전 이전됨

#### 3. `_dedup_expanded_boxes()` — 확장 후 중복 박스 제거

**현상**: 분리 + 확장 후 인접 bbox가 서로 50% 이상 침범하는 경우 발생  
**해결**: `intersection / min(area1, area2) ≥ 0.50` 이면 작은 박스를 큰 박스로 흡수  

### 파이프라인 처리 순서 (최신)

```
NMS → _merge_overlapping_boxes → _merge_proximity_boxes
  → 스케일 변환(모델→원본 좌표)
  → _split_tall_boxes
  → _expand_bbox_to_balloon  ← 허위 양성: 내부 밝기 < 160이면 건너뜀
  → _dedup_expanded_boxes
  → BubbleInfo 생성
```

### Phase 19 수정 파일 목록

| # | 파일 | 수정 내용 |
|---|------|----------|
| 1 | `pyproject.toml` | `einops>=0.8`, `pulp>=2.9` 추가 |
| 2 | `server/pipeline/bubble_detector.py` | `_split_tall_boxes()`, `_expand_bbox_to_balloon()`, `_dedup_expanded_boxes()` 추가, 상수 6개, 내부 밝기 가드, detect() 파이프라인 순서 통합 |
| 3 | `server/pipeline/text_renderer.py` | `_probe_balloon_width()` (80줄) 제거, `render()`의 프로빙 로직 제거 |

---

### 배경
번역 실행 시 "번역 실패" 발생. 로그가 부족하여 원인 파악 불가 → 프론트엔드+백엔드 전면 로깅 구축 후 런타임 버그 2건 발견·수정.

### 로깅 시스템 구축 (14개 파일, +260/-39줄)

#### 백엔드 로깅
| # | 파일 | 추가 내용 |
|---|------|-----------|
| 1 | `server/utils/logger.py` | `RotatingFileHandler` 추가 — `logs/mangalens.log`, 10MB × 5 backups, UTF-8 |
| 2 | `server/main.py` | HTTP 미들웨어 — `/api/*` 요청/응답 로깅 (method, path, status, latency) |
| 3 | `server/routers/upload.py` | 파이프라인 라이프사이클 상세 로깅 — 큐, 세마포어 대기, 이미지별 처리시간, 실패 예외 |
| 4 | `server/routers/ws.py` | WebSocket connect/disconnect/error, 터미널 상태 전송, `send_json` 실패 catch |
| 5 | `server/routers/result.py` | 상태 조회, 다운로드 거부(409), 누락 파일 진단 (디렉토리 리스팅) |
| 6 | `server/pipeline/orchestrator.py` | GPU fallback 경고, 번역 성공/실패, 마스크 소스 통계, CUDA cleanup 실패 로깅 |

#### 프론트엔드 로깅
| # | 파일 | 추가 내용 |
|---|------|-----------|
| 7 | `frontend/js/api.js` | `Logger` 모듈 (IIFE) — console + 200-entry 링 버퍼, `Logger.dump()` 진단용 |
| 8 | `frontend/js/upload.js` | 파일 검증, 업로드 시작/성공/실패, 서버 준비 상태 |
| 9 | `frontend/js/progress.js` | WebSocket 연결/메시지/종료/오류/재연결 |
| 10 | `frontend/js/result.js` | 결과 표시, 다운로드 시작/실패 |
| 11 | `frontend/js/app.js` | 헬스체크, 부팅 시퀀스 |

#### 기타
| # | 파일 | 추가 내용 |
|---|------|-----------|
| 12 | `.gitignore` | `logs/` 디렉토리 제외 |

### 런타임 버그 수정
| # | 파일 | 이슈 | 원인 | 수정 |
|---|------|------|------|------|
| 1 | `server/pipeline/preprocessor.py` | `NameError: name 'asyncio' is not defined` | Phase 13에서 `asyncio.to_thread()` 래핑 추가 시 `import asyncio` 누락 | `import asyncio` 추가 |
| 2 | `server/pipeline/translator.py` | `AttributeError` on `encoded.shape[1]` | transformers 5.3.0의 `apply_chat_template(return_tensors="pt")`가 `BatchEncoding`(dict-like) 반환 — `.shape` 직접 접근 불가 | `hasattr(result, "input_ids")` 검사 후 `result["input_ids"]` 추출 |

- **테스트 결과**: **181 passed**, 0 failed (기존 테스트 전체 통과)
- **핵심 성과**: 로깅 시스템으로 무증상 런타임 버그 2건 즉시 발견·해결, 향후 디버깅 인프라 확보

---

## 2026-03-26 — Phase 12: 전체 코드 리뷰 (5개 담당 검토)

### 번역 환각 수정 (이전 시점에 완료)
| # | 파일 | 변경 내용 |
|---|------|-----------|
| 1 | `server/pipeline/translator.py` | chat template 적용 (`apply_chat_template`), 동적 `max_new_tokens`, `repetition_penalty=1.15`, `no_repeat_ngram_size=5`, `_postprocess()` 후처리 추가 |
| 2 | `server/pipeline/magi_detector.py` | BGR→RGB 변환, `asyncio.to_thread`, GPU fp16 로딩 추가 |

### 5개 담당 검토 결과 종합

**테스트: 171 passed, 0 failed**

#### P0 Critical (운영 불가) — 2건 ✅ 전체 해결
| # | 담당 | 파일 | 이슈 | 상태 |
|---|------|------|------|------|
| 1 | Pipeline | `orchestrator.py`+`magi_detector.py` | Magi v2 — `BubbleInfo.mask=None` → 인페인팅 마스크 0 → 원본 일본어 위에 한국어 겹침 | ✅ bbox 기반 마스크 자동 생성 |
| 2 | GPU | `start.bat` | WSL 경로 `/home/user/test0320` 하드코딩 → Windows 실행 불가 | ✅ `wslpath -u` 동적 변환 |

#### P1 High (기능 결함) — 10건 ✅ 전체 해결
| # | 담당 | 파일 | 이슈 | 상태 |
|---|------|------|------|------|
| 3 | Pipeline | `bubble_detector.py` | `detect()` GPU 추론에 `asyncio.to_thread` 누락 → 이벤트 루프 블로킹 | ✅ |
| 4 | Pipeline | `preprocessor.py` | `crop_and_upscale()` Real-ESRGAN 추론에 `asyncio.to_thread` 누락 | ✅ |
| 5 | Pipeline | `orchestrator.py` | `_get_cached()` 모델 로딩이 이벤트 루프에서 동기 실행 | ✅ |
| 6 | Pipeline | `translator.py` | 개별 텍스트 에러 핸들링 없음 → 1건 실패 시 전체 배치 손실 | ✅ per-text try/except + fallback |
| 7 | Server | `routers/upload.py` | ZIP 내부 이미지 미검증 (magic bytes, Pillow verify 스킵) | ✅ Pillow verify 추가 |
| 8 | Server | `routers/result.py` | `partial` 상태 결과 다운로드 불가 | ✅ `not in ("completed", "partial")` |
| 9 | Server | `routers/upload.py` | ZIP bomb 방어 없음 (해제 크기 제한 없음) | ✅ per-entry + total size limit |
| 10 | Frontend | `api.js` | D-014 잔존: 세션 관리 코드 (`X-Session-Id`) 미제거 | ✅ 삭제 |
| 11 | Frontend | `progress.js` | WS 재연결 3회 실패 후 사용자 무응답 상태 | ✅ Toast.error + 상태 표시 |
| 12 | GPU | `.env.example` | 제거된 API 키 잔존 → 사용자 혼란 | ✅ DEEPL/GOOGLE 키 삭제 |

#### P2 Medium — 14건 ✅ 전체 해결
| # | 담당 | 파일 | 이슈 | 상태 |
|---|------|------|------|------|
| 13 | Pipeline | `compositor.py` | RGB↔BGR 채널 불일치 잠복 | ✅ overlay RGB→BGR 변환 추가 |
| 14 | Pipeline | `bubble_detector.py` | 효과음 미분류 → 불필요 번역 | ✅ cid==2 → "effect" 분류 |
| 15 | Pipeline | `translator.py` | `source_lang`/`target_lang` 프롬프트 미반영 | ✅ 동적 언어명 삽입 |
| 16 | Pipeline | `translator.py`, `bubble_detector.py` | GPU 텐서 명시적 정리 누락 | ✅ del + to_thread 분리 |
| 17 | Pipeline | `translator.py` | PLAN 명시 재시도 로직 미구현 | ✅ per-text fallback |
| 18 | Server | `routers/result.py` | `task_id` 경로 순회 검증 부재 | ✅ regex 검증 추가 |
| 19 | Server | `state.py` | `_task_watchers` 빈 리스트 무한 누적 | ✅ 빈 리스트 del 추가 |
| 20 | Server | `state.py` | `task_store` bound 제한 없음 | ✅ OrderedDict + 500건 제한 |
| 21 | GPU | `start.sh` | ROCm 아키텍처 무조건 gfx1201 가정 | ✅ rocminfo 동적 감지 |
| 22 | GPU | `download.py` | 체크섬 검증 없음 | ⏭️ 후순위 (기능 영향 없음) |
| 23 | Frontend | `index.html` | settings 버튼 title에 "API 키" 잔존 | ✅ 삭제 |
| 24 | Frontend | `style.css` | Dead CSS ~115줄 (API 키 가이드 관련) | ✅ 삭제 |
| 25 | QA | `test_pipeline.py` | translator 내부 함수 단위 테스트 부재 | ✅ 7개 테스트 추가 |
| 26 | QA | `test_pipeline.py` | Magi BGR→RGB 변환 검증 부재 | ✅ compositor RGB→BGR 테스트 추가 |

### 종합 평가
- **점수**: **8.5 / 10** (Phase 13 수정 완료 후 상향)
- **테스트**: **181 passed**, 0 failed (171 → +10 신규)
- **해결**: P0 2건, P1 10건, P2 13건 = 총 25/26건 해결 (P2#22 download 체크섬만 후순위)
- **핵심 개선**: Magi 모드 정상화, 이벤트 루프 블로킹 3건 제거, ZIP 보안 강화, 세션 잔존 코드 제거, compositor 색상 수정

---

## 2026-03-26 — Phase 13: 전체 이슈 수정 (26건)

### 수정 파일 총 17개
| # | 파일 | 변경 내용 | 심각도 |
|---|------|-----------|--------|
| 1 | `server/pipeline/orchestrator.py` | Magi mask=None → bbox 기반 마스크 자동 생성; `_get_cached` `asyncio.to_thread` | P0, P1 |
| 2 | `start.bat` | WSL 하드코딩 경로 → `wslpath -u` 동적 변환 | P0 |
| 3 | `server/pipeline/bubble_detector.py` | `detect()` GPU 추론 `asyncio.to_thread` 래핑; `asyncio` import 추가; effect 분류 | P1, P2 |
| 4 | `server/pipeline/preprocessor.py` | `crop_and_upscale()` Real-ESRGAN `asyncio.to_thread` 래핑 | P1 |
| 5 | `server/pipeline/translator.py` | per-text try/except fallback; `_SYSTEM_MSG_TEMPLATE` 동적 언어명; GPU 텐서 del; 언어 파라미터 전달 | P1, P2 |
| 6 | `server/routers/upload.py` | ZIP entry Pillow verify; ZIP bomb 방어 (per-entry + total size); `add_task()` 사용 | P1 |
| 7 | `server/routers/result.py` | `partial` 상태 다운로드 허용; `task_id` regex 검증; `re` import | P1, P2 |
| 8 | `server/state.py` | `OrderedDict` + 500건 제한 `add_task()`; `unsubscribe()` 빈 리스트 정리 | P2 |
| 9 | `server/pipeline/compositor.py` | overlay RGBA→BGR 변환 후 블렌딩 | P2 |
| 10 | `frontend/js/api.js` | `_SESSION_KEY`, `_getSessionId()`, `setSessionId()`, `X-Session-Id` 헤더 제거 | P1 |
| 11 | `frontend/js/progress.js` | WS 재연결 실패 시 `Toast.error` + 상태 텍스트 표시 | P1 |
| 12 | `frontend/index.html` | settings 버튼 title "API 키" 제거 | P2 |
| 13 | `frontend/css/style.css` | `.api-key-guide*` dead CSS ~115줄 삭제 | P2 |
| 14 | `.env.example` | `DEEPL_API_KEY`, `GOOGLE_API_KEY` 삭제 | P1 |
| 15 | `start.sh` | ROCm `rocminfo` 동적 gfx 감지; API 키 경고 메시지 제거 | P2 |
| 16 | `tests/test_integration.py` | task_id 포맷 검증에 맞춰 테스트 ID 업데이트; `test_invalid_task_id_returns_400` 추가 | 테스트 |
| 17 | `tests/test_pipeline.py` | `_postprocess`, `_dynamic_max_new_tokens`, 언어명 반영, compositor RGB→BGR 등 9개 테스트 추가 | 테스트 |

- **테스트 결과**: **181 passed**, 0 failed (171 → +10 신규)
- **해결**: P0×2, P1×10, P2×13 = 25/26건 (P2#22 download 체크섬만 후순위 — 기능 영향 없음)

---

## 2026-03-26 — Phase 11: P0–P2 버그 수정 스프린트

### PM 검토 기반 전체 이슈 해결 ✅

| # | 파일 | 변경 내용 | 심각도 |
|---|------|-----------|--------|
| 1 | `server/pipeline/orchestrator.py` | `_get_cached` → `async def` + `asyncio.Lock` 이중 확인 잠금, 6개 call site `await` 추가 | P0 |
| 2 | `server/pipeline/text_eraser.py` | LaMa 추론 후 `del img_t, mask_t, result` + `torch.cuda.empty_cache()` 추가 | P1 |
| 3 | `server/state.py` | pub/sub Queue 인프라 — `subscribe()`, `unsubscribe()`, `notify_task_changed()` 추가 | P1 |
| 4 | `server/routers/upload.py` | `asyncio.Lock` 보호 `async def _get_semaphore()`, 3개소 `notify_task_changed()` 호출 추가 | P1 |
| 5 | `server/routers/ws.py` | 1초 폴링 → 이벤트 기반 `asyncio.Queue` (5초 heartbeat timeout) | P2 |
| 6 | `server/pipeline/text_renderer.py` | 모듈 레벨 `_font_cache: dict[tuple, ...]` — `_load_font()` 캐시 조회 우선 | P2 |
| 7 | `frontend/css/themes.css` | 라이트모드 WCAG AA: `--text-secondary` `#6b7280`→`#4b5563` (6.1:1), `--text-muted` `#9ca3af`→`#6b7280` | P2 |
| 8 | `frontend/js/progress.js` | WebSocket JSON 파싱 에러 `console.warn` 로깅 추가 | P2 |
| 9 | `tests/test_pipeline.py` | `_get_cached` async 전환에 따른 3개 테스트 `@pytest.mark.asyncio` + `await` 업데이트 | 테스트 |

- **테스트 결과**: **171 passed**, 0 failed (168 → +3 신규 asyncio 테스트 통과)
- **핵심 개선**: 동시 요청 시 모델 이중 로딩 방지 (P0), VRAM 누수 제거 (P1), WS 이벤트 드리븐으로 응답성 향상 (P2)

---

## 2026-03-26 — Phase 10: Hunyuan-MT-7B 번역 엔진 마이그레이션

### DeepL/Google → Hunyuan-MT-7B 로컬 인퍼런스 ✅

| # | 파일 | 변경 내용 |
|---|------|-----------|
| 1 | `server/pipeline/translator.py` | 완전 재작성 — Hunyuan-MT-7B 로컬 CausalLM 인퍼런스 |
| 2 | `server/config.py` | `deepl_api_key`, `google_api_key` 필드 제거 |
| 3 | `server/schemas/models.py` | `UserSettings`, `UserSettingsResponse`, `MAX_USER_API_KEY_LENGTH` 제거 |
| 4 | `server/routers/settings.py` | **파일 삭제** — POST/GET `/api/settings` 엔드포인트 제거 |
| 5 | `server/routers/upload.py` | `session_store` 임포트, `_extract_user_settings()` 함수, Request 파라미터 제거 |
| 6 | `server/pipeline/orchestrator.py` | Translator() 호출에서 `deepl_key`/`google_key` 인수 제거 |
| 7 | `tests/test_pipeline.py` | TestTranslator를 7개 Hunyuan 특화 테스트로 교체 |
| 8 | `tests/test_integration.py` | TestSettingsEndpoints 클래스 + `_clean_session_store` 픽스처 제거 |
| 9 | `tests/test_frontend.py` | `test_api_settings_still_works` 제거 |

- **테스트 결과**: test_pipeline 7/7 + test_integration 18/18 + test_frontend 20/20 = 45 passed
- **HuggingFace 모델**: `tencent/Hunyuan-MT-7B` — WMT25 1위, 첫 요청 시 HF Hub에서 ~15GB 자동 다운로드
- **추론 패턴**: 모듈 레벨 싱글턴 + `threading.Lock` 이중 확인 + `asyncio.to_thread()` 비동기

---

## 2026-03-26 — Phase 8: 모델 업그레이드

### Tier 1 — RealESRGAN anime_6B (x4 업스케일러 교체) ✅

| # | 파일 | 변경 내용 |
|---|------|-----------|
| 1 | `server/download.py` | `MODELS` 리스트에 anime_6B 항목 추가 (`v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth`, 17MB) |
| 2 | `server/config.py` | `upscaler_variant: str = "anime_6b"` 설정 추가 (`anime_6b` \| `x4plus`) |
| 3 | `server/pipeline/preprocessor.py` | `_pick_x4_variant()` 함수 추가 — anime_6B(`num_block=6`) 우선, x4plus(`num_block=23`) fallback |
| 4 | `models/RealESRGAN_x4plus_anime_6B.pth` | 17.1MB 모델 가중치 다운로드 완료 |
| 5 | `tests/test_pipeline.py` | `test_pick_x4_variant_anime_6b`, `_fallback_to_x4plus`, `_explicit_x4plus` 3개 테스트 추가 |

- **테스트 결과**: 42→45 passed (기존 42 + 신규 3), 0 failed

### Tier 2 — Magi v2 (A/B 감지기 탐지기 통합) ✅

| # | 파일 | 변경 내용 |
|---|------|-----------|
| 1 | `server/config.py` | `use_magi_detector: bool = False`, `magi_vram_threshold_mb: int = 4096` 추가 |
| 2 | `server/pipeline/magi_detector.py` | **신규** — `MagiDetector` 클래스 (HuggingFace `ragavsachdeva/magiv2` 래핑) |
|   |  | - `_xyxy_to_xywh()`: bbox 형식 변환 `(x1,y1,x2,y2)` → `(x,y,w,h)` |
|   |  | - `is_essential_text` → `speech`/`effect` 분류 |
|   |  | - Magi 내장 reading order 보존 |
|   |  | - VRAM 임계값 검사(`get_vram_mb` < 4096 시 skip) |
| 3 | `server/pipeline/orchestrator.py` | Stage 1 조건부 분기: `use_magi_detector=True` → `MagiDetector` 사용, 별도 캐시 키 `"magi_detector"` |
|   |  | Magi 사용 시 `sort_bubbles_rtl()` 바이패스 |
| 4 | `pyproject.toml` | `accelerate>=1.0` 의존성 추가 |
| 5 | `tests/test_pipeline.py` | `TestMagiDetector` 5개 + `TestMagiCacheKeySeparation` 1개 테스트 추가 |

- **테스트 결과**: 48 passed (pipeline), 131 passed (전체 핵심), 0 failed
- **활성화 방법**: `.env`에 `USE_MAGI_DETECTOR=true` 설정 (기본 `false` = YOLOv5 유지)
- **사이드이펙트 대비**: bbox 변환, 캐시 키 분리, reading order 분기 모두 구현

### 버그 수정 — 탁점/반탁점 보호 ✅

| # | 파일 | 변경 내용 |
|---|------|----------|
| 1 | `server/pipeline/preprocessor.py` | `remove_furigana()` — 작은 컴포넌트가 큰 글리프 bbox에 근접(median_h×15%, 최소 3px)하면 탁점/반탁점으로 판단해 보존, 멀리 떨어진 것만 후리가나로 제거 |
| 2 | `tests/test_pipeline.py` | `test_remove_furigana_preserves_dakuten`, `test_remove_furigana_removes_distant_small_components` 2개 테스트 추가 |

- **원인**: 탁점(゛)/반탁점(゜)이 이진화 후 별도 connected component로 분리되어 높이+면적 필터에 걸려 삭제됨 (예: が→か, ぱ→は)
- **해결**: 작은 컴포넌트 ↔ 큰 글리프 bbox 근접 검사로 탁점 보호

### 기능 개선 — 출력 디렉토리 네이밍 ✅

| # | 파일 | 변경 내용 |
|---|------|----------|
| 1 | `server/routers/upload.py` | `task_id` 생성 방식: `uuid.uuid4().hex` → `YYYYMMDD_HHMMSS_fff_xxxxxx` (날짜+시간+밀리초+6자리 UUID 접미사) |

- **테스트 결과**: 182 passed, 0 failed
