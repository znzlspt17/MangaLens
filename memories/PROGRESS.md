# MangaLens — 진행 상황 추적

> 최종 갱신: 2026-03-25

---

## 현재 시스템 상태

| 구성 요소 | 상태 | 비고 |
|-----------|------|------|
| 서버 (FastAPI) | ✅ 정상 동작 | port 20399, uvicorn |
| GPU | ✅ CUDA | NVIDIA RTX 4090, VRAM 24564 MB, CUDA 12.8 |
| BubbleDetector | ✅ ML 모드 | comic-text-detector YOLOv5 on CUDA |
| Preprocessor (Real-ESRGAN) | ✅ 정상 | device=cuda |
| OCREngine (manga-ocr) | ✅ 정상 | kha-white/manga-ocr-base on CUDA |
| TextEraser (LaMa) | ✅ 정상 | models/big-lama.pt |
| TextRenderer (Pillow) | ✅ 정상 | fonts/NotoSansKR-Regular.ttf |
| 번역 (DeepL/Google) | ✅ 세션 키 | HTTP 세션 쿠키 우회 → X-Session-Id 헤더 방식 |
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

### P0 — 즉시 수정 필요

| 이슈 | 파일 | 내용 |
|------|------|------|
| 모델 캐시 Race Condition | `server/pipeline/orchestrator.py` | `_model_cache` dict에 `asyncio.Lock` 없이 동시 접근 → 모델 2중 로딩·GPU OOM 위험 |
| httpx 클라이언트 미해제 | `server/pipeline/translator.py` | `AsyncClient.close()` 정의만 있고 호출 없음 → 연결 누수 |
| TOCTOU 버그 | `server/routers/result.py` | `delete_after_download=True` 시 `StreamingResponse` 전송 완료 전 파일 삭제 가능 |

### P1 — 중요 개선

| 이슈 | 파일 | 내용 |
|------|------|------|
| GPU 텐서 미해제 | `server/pipeline/text_eraser.py` | `del img_t, mask_t` + `torch.cuda.empty_cache()` 누락 → VRAM 누적 |
| task_store 동시성 | `server/state.py` | 비원자적 다중 필드 업데이트, `asyncio.Lock` 혹은 dataclass 교체 필요 |
| 세션 TTL 없음 | `server/routers/settings.py` | `session_store` 무기한 메모리 점유 → TTL 정리 로직 추가 필요 |
| Semaphore 초기화 경합 | `server/routers/upload.py` | 동시 첫 요청 시 Semaphore 2중 생성 가능 |

### P2 — 개선 권장

| 이슈 | 파일 | 내용 |
|------|------|------|
| WebSocket 폴링 비확장 | `server/routers/ws.py` | 1초 폴링 → 대규모 연결 시 이벤트 기반 전환 권장 |
| font.getlength() 반복 | `server/pipeline/text_renderer.py` | 이진탐색 내 반복 호출 → 성능 병목 가능 |
| 라이트모드 색상 대비 | `frontend/css/style.css` | WCAG AA 기준 일부 미달 |
| WS 메시지 파싱 에러 누락 | `frontend/js/progress.js` | JSON 파싱 에러 로깅 없음 |

### 낮음 — 기존 이슈

| 이슈 | 심각도 | 내용 |
|------|--------|------|
| 세션 서버 재시작 시 소멸 | 낮음 | 의도된 설계 (P7 원칙). 재시작 후 API 키 재입력 필요 |
| HF_TOKEN 미설정 경고 | 낮음 | HuggingFace rate limit 경고. 기능에는 영향 없음 |
| basicsr `.venv` 패치 | 낮음 | `uv sync` 재실행 시 재패치 필요 (자동화 미구현) |

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
