# MangaLens — 진행 상황 추적

> 최종 갱신: 2026-03-26 (Phase 11: P0–P2 버그 수정 스프린트)

---

## 현재 시스템 상태

| 구성 요소 | 상태 | 비고 |
|-----------|------|------|
| 서버 (FastAPI) | ✅ 정상 동작 | port 20399, uvicorn |
| GPU | ✅ CUDA | NVIDIA RTX 4090, VRAM 24564 MB, CUDA 12.8 |
| BubbleDetector | ✅ ML 모드 | comic-text-detector YOLOv5 on CUDA (기본) |
| MagiDetector | ⚠️ 비활성 | `USE_MAGI_DETECTOR=true`로 활성화 (ragavsachdeva/magiv2) |
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
| HF_TOKEN 미설정 경고 | 낙음 | HuggingFace rate limit 경고. 기능에는 영향 없음 |
| basicsr `.venv` 패치 | 낙음 | `uv sync` 재실행 시 재패치 필요 (자동화 미구현) |
| Hunyuan 첫 요청 지연 | 낙음 | 마지막 비활성 시 ~5내가 VRAM 로드. lifespan warm-up으로 개선 가능 (D-014 후속고려) |

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
