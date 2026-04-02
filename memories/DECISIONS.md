# MangaLens — 기술 결정 기록 (DECISIONS.md)

> 주요 기술 선택의 근거를 기록한 문서. 최종 갱신: 2026-03-27 (D-031)

---

## D-001: 패키지 매니저 — uv 선택

- **결정**: pip/poetry 대신 `uv` 사용
- **근거**: Rust 기반으로 설치 속도 10~100x 빠름, `pyproject.toml` PEP 621 준수, lock 파일 기반 재현 가능 빌드, GPU 인덱스 분리(`[[tool.uv.index]]`) 지원
- **대안**: poetry, conda — 속도/GPU 인덱스 분리 면에서 열세

---

## D-002: 말풍선 검출 — comic-text-detector YOLOv5

- **결정**: ultralytics YOLOv5 백본 + comic-text-detector 가중치
- **근거**: 만화 특화, 세로쓰기 방향 판별 내장, 말풍선/효과음/나레이션 박스 구분
- **주의**: `ultralytics` PyPI 패키지 필요 (`>=8.0`). 미설치 시 OpenCV contour fallback 사용 — 검출 품질 대폭 저하
- **이슈 이력**: 2026-03-25 `ultralytics` 의존성 누락으로 fallback 작동 → `pyproject.toml`에 추가 후 수정

---

## D-003: 텍스트 렌더링 — Pillow (OpenCV 배제)

- **결정**: `cv2.putText` 대신 Pillow + FreeType
- **근거**: CJK 유니코드 완벽 지원, 안티앨리어싱, 자동 줄바꿈, 세로→가로 레이아웃 변환, 폰트 크기 자동 조절
- **트레이드오프**: Pillow ↔ numpy 변환 오버헤드 존재하나 품질 우위로 선택

---

## D-004: LaMa 인페인팅 — simple-lama-inpainting 래퍼 배제

- **결정**: `simple-lama-inpainting` PyPI 래퍼 미사용, torch로 직접 추론 구현
- **근거**: 래퍼가 `Pillow<10` 요구 → 최신 Pillow와 버전 충돌 불가피
- **구현**: `models/big-lama.pt` 직접 로드, LaMa 아키텍처 직접 구현

---

## D-005: API 키 세션 — 서버 메모리 저장 (비영구) ~~[D-014로 대체됨]~~

- **결정**: API 키를 DB가 아닌 서버 메모리 `session_store` dict에 세션 수명으로 저장
- **근거**: P7 원칙 "서버가 API 키를 영구 저장하지 않음", 보안상 키를 디스크에 쓰지 않음
- **식별자**: `session_id` — Cookie 또는 `X-Session-Id` 헤더로 전달
- **⚠️ 폐기**: D-014 (Hunyuan-MT-7B 로컬 인퍼런스) 채택으로 외부 API 키 불필요. `session_store`, `settings.py` 라우터, 관련 스키마 전면 제거됨

---

## D-006: HTTP 세션 쿠키 우회 — X-Session-Id 헤더 + localStorage ~~[D-014로 대체됨]~~

- **결정**: `secure=True` 쿠키 대신 `localStorage` + `X-Session-Id` 헤더 방식 병행
- **근거**: HTTP (localhost) 환경에서 `secure=True` 쿠키는 브라우저가 저장하지 않음 → API 키 세션 유실
- **구현**:
  - 서버: `GET/POST /api/settings` 응답 JSON에 `session_id` 포함
  - 클라이언트: `api.js`에서 `localStorage['mangalens_session_id']` 저장/조회, 모든 요청 헤더에 `X-Session-Id` 자동 추가
  - HTTPS 환경에서는 쿠키도 병행 작동 (`secure` 조건부 설정)
- **⚠️ 폐기**: D-014 채택으로 세션 인프라 전면 제거됨

---

## D-007: basicsr — vendor 패치 배포

- **결정**: `basicsr==1.4.2`를 `vendor/basicsr-1.4.2/`에 소스 복사 후 패치
- **근거**: Python 3.14 PEP 667로 `exec()` 결과가 `locals()`에 반영되지 않음 → `setup.py`의 `get_version()` KeyError 발생
- **패치 내용**:
  1. `setup.py`: `ns = {}; exec(..., ns); return ns['__version__']`
  2. `basicsr/data/degradations.py`: `torchvision.transforms.functional_tensor` 제거됨 → `try/except ImportError` fallback

---

## D-008: PyTorch 인덱스 — CUDA 12.8

- **결정**: `https://download.pytorch.org/whl/cu128` 인덱스 사용
- **근거**: 운영 환경 NVIDIA RTX 4090 + CUDA 12.8 (driver 590.48.01) 최적 빌드
- **설정**: `pyproject.toml` `[tool.uv.sources]`에서 torch, torchvision에 명시적 인덱스 지정

---

## D-009: 후리가나 제거 — OCR 이전 이미지 레벨 처리

- **결정**: OCR 전(업스케일 직후) connected component 높이 필터링으로 후리가나 마스킹
- **근거**: OCR 텍스트 레벨에서는 후리가나와 오쿠리가나를 구분 불가 (예: `走はしる` → `は・し`가 후리가나인지 가나 단어인지 판별 불가). 이미지 레벨에서는 후리가나가 시각적으로 본문 글자 크기의 절반 이하임을 이용 가능
- **구현**: `preprocessor.py::remove_furigana()` — 글리프 높이 중앙값의 60% 미만 connected component를 흰색으로 마스킹
- **보존**: 오쿠리가나(`走る`의 `る`)는 본문과 동일한 크기이므로 영향 없음
- **위치**: `Preprocessor.crop_and_upscale()` 반환 직전에 적용 (Real-ESRGAN 및 cv2 fallback 양쪽)

---

## D-010: x4 업스케일러 — anime_6B 기본 선택

- **결정**: `RealESRGAN_x4plus_anime_6B.pth` (RRDBNet 6블록, 17MB)를 x4 업스케일러 기본값으로 설정, 기존 `x4plus` (23블록, 64MB)는 fallback으로 유지
- **근거**: anime_6B는 애니메이션/만화 라인아트+텍스트에 특화 학습되어 OCR 전 업스케일 시 획 선명도가 우수하고, 6블록 경량 구조로 추론 속도 약 2배 빠름. 만화 번역 서비스 특성상 실사 모델보다 적합
- **설정**: `UPSCALER_VARIANT` 환경변수 (`anime_6b` | `x4plus`), `_pick_x4_variant()` 함수가 모델 경로+블록 수 반환
- **fallback**: anime_6B 파일 미존재 시 자동으로 x4plus 사용

---

## D-011: Magi v2 통합 — A/B 토글 방식 도입

- **결정**: Magi v2 (`ragavsachdeva/magiv2`)를 기존 YOLOv5 검출기와 A/B 토글 방식으로 통합. 기본값 비활성 (`USE_MAGI_DETECTOR=false`)
- **근거**: Magi v2는 말풍선·패널·캐릭터 검출 + 화자 연결 + 내장 reading order를 단일 모델로 제공하나, Transformer 기반으로 VRAM ~4GB+ 필요하고 추론이 YOLOv5보다 느림. 점진적 검증을 위해 A/B 토글 방식 채택
- **구현**:
  - `server/pipeline/magi_detector.py` 신규 생성 — `MagiDetector` 클래스
  - bbox 형식 변환 `(x1,y1,x2,y2)` → `(x,y,w,h)` 레이어 (`_xyxy_to_xywh()`)
  - `is_essential_text` → `speech`/`effect` 매핑
  - 별도 캐시 키 `"magi_detector"` — YOLOv5 `"bubble_detector"`와 분리
  - Magi 사용 시 `sort_bubbles_rtl()` 바이패스 (Magi 내장 순서 보존)
  - VRAM 임계값 검사 (`magi_vram_threshold_mb: 4096`) — 미만 시 로딩 건너뜀
- **의존성**: `accelerate>=1.0`, `transformers>=5.0` (기존 manga-ocr 의존성)

---

## D-012: 후리가나 제거 — 탁점/반탁점 보호 로직 추가 (Phase 9 초기, Phase 16 강화)

- **결정**: `remove_furigana()` 에서 작은 connected component가 큰 글리프 bbox에 근접하거나 **같은 텍스트 컬럼**에 있으면 삭제하지 않음 (탁점/반탁점 보존)
- **근거**: 탁점(゛)과 반탁점(゜)은 이진화 후 문자 본체와 별도 connected component로 분리되나, 문자 본체 바로 옆에 위치. 탁점 없이 OCR할 경우 `が→か`, `ぱ→は` 등 의미가 완전히 달라져 번역 품질 심각하게 저하. 또한 세로쓰기 말풍선에서 ellipsis(．)와 구두점도 별도 컴포넌트로 분리됨
- **Phase 9 구현**: padding = `max(median_h * 0.15, 3px)`. 근접하면 보존, 멀면 후리가나로 제거
- **Phase 16 강화 → D-023**: 실제 탁음 점이 11-29px 거리였는데 패딩 4px로 커버 안 됨 → 별도 결정 D-023으로 보완
- **관련 결정**: D-009 (후리가나 제거 최초 구현), D-023 (패딩 강화 + column check)

---

---

## D-014: 번역 엔진 — Hunyuan-MT-7B 로컬 인퍼런스 (DeepL/Google 대체)

- **결정**: DeepL API(기본) + Google Translate(fallback) 외부 API 방식을 `tencent/Hunyuan-MT-7B` 로컬 모델 인퍼런스로 완전 대체
- **근거**: WMT25 경진대회 30/31 언어쌍 1위, JA→KO 지원, 외부 API 키 종속성 제거, 인터넷 없는 환경에서도 번역 가능
- **모델**: `tencent/Hunyuan-MT-7B` (CausalLM, 7B 파라미터, fp16 ~14GB VRAM)
- **추론**: 싱글턴 패턴 + `threading.Lock` 이중 확인, `asyncio.to_thread()`로 비동기 처리
- **프롬프트**: `"Translate the following segment into Korean, without additional explanation.\n{text}"`
- **fallback**: 추론 실패 시 원문 그대로 반환 (번역 건너뜀)
- **제거된 코드**: `session_store`, `POST/GET /api/settings`, `UserSettings` 스키마, `deepl_api_key`/`google_api_key` 설정 필드, `X-DeepL-Key`/`X-Google-Key`/`X-Session-Id` CORS 헤더
- **D-005, D-006 폐기**: 외부 API 키 관리 인프라 전면 제거

---

## D-013: 출력 디렉토리 네이밍 — 날짜+시간 형식

- **결정**: `task_id`를 `uuid.uuid4().hex` 대신 `YYYYMMDD_HHMMSS_fff_xxxxxx` 형식으로 생성
- **근거**: UUID hex(예: `30927c97f0a3...`)는 사람이 식별 불가. 날짜+시간+밀리초 형식으로 변경하면 출력물 정렬과 시간 확인이 용이
- **충돌 방지**: 6자리 UUID 접미사(`_xxxxxx`) 추가로 동시 업로드 시에도 고유성 보장
- **예시**: `20260326_112302_385_176b5e`

---

## D-015: 모델 캐시 동시성 — asyncio.Lock 이중 확인 잠금

- **결정**: `_get_cached()` → `async def` 전환 + 모듈 레벨 `_model_cache_lock: asyncio.Lock` 이중 확인 잠금 (DCL)
- **근거**: 동시 요청 시 캐시 miss가 여러 코루틴에서 동시 감지되어 동일 모델이 2번 이상 로딩될 수 있음 → VRAM 이중 점유 또는 OOM. asyncio는 단일 스레드이므로 `async with lock` 만으로 충분
- **패턴**: fast path(잠금 없는 조회) → slow path(잠금 + 재확인 → 생성)
- **영향**: `_get_cached` 호출 6개소 모두 `await` 추가, 관련 테스트 3개 `@pytest.mark.asyncio` + `async def` 전환

---

## D-016: WebSocket 업데이트 — 폴링 → 이벤트 기반 pub/sub

- **결정**: `ws.py` 1초 폴링 `asyncio.sleep(1)` → `state.py` pub/sub Queue + `asyncio.wait_for(q.get(), timeout=5.0)` heartbeat
- **근거**: 1초 폴링은 연결 수 × 초당 1회 상태 조회로 확장성 한계. 이벤트 기반 전환 시 상태 변경이 있을 때만 전송 → CPU/메모리 절약, 응답 지연 ~0으로 감소
- **구현**:
  - `state.py`: `_task_watchers: dict[str, list[asyncio.Queue[bool]]]` + `subscribe/unsubscribe/notify_task_changed`
  - `ws.py`: `subscribe()` 호출 후 `asyncio.wait_for(q.get(), 5.0)` — 5초 timeout 시 heartbeat 전송
  - `upload.py`: 처리 시작, 이미지 완료, 최종 상태 3개소에서 `notify_task_changed()` 호출
- **안전성**: `finally` 블록에서 `unsubscribe()` 보장으로 연결 종료 시 Queue 누수 없음

---

## D-017: task_id 경로 순회 방어 — regex 검증

- **결정**: `result.py`의 `GET /api/status/{task_id}` 및 `GET /api/result/{task_id}`에 `^[0-9]{8}_[0-9]{6}_[0-9]{3}_[0-9a-f]{6}$` 정규식 검증 추가
- **근거**: `task_id`가 `../../../etc/passwd` 같은 경로를 포함할 경우 `Path(output_dir) / task_id`로 임의 파일 접근 가능. 생성 시 사용하는 정확한 형식만 허용하여 경로 순회 공격 차단
- **영향**: 유효하지 않은 형식의 task_id → HTTP 400 Bad Request 반환

---

## D-018: task_store 메모리 제한 — OrderedDict + 자동 퇴출

- **결정**: `task_store`를 `dict` → `OrderedDict`로 교체 + 최대 500건 제한, 초과 시 가장 오래된 항목 자동 퇴출
- **근거**: 장기 운영 시 완료/실패 태스크가 무한 누적되어 메모리 소진. OrderedDict의 FIFO 특성을 활용하여 최소한의 코드로 LRU-like 제한 구현
- **구현**: `add_task()` 함수로 삽입 통일, `while len > _MAX_TASKS: popitem(last=False)`

---

## D-019: 번역 프롬프트 — 동적 언어명 템플릿

- **결정**: system message와 fallback prompt에서 하드코딩된 "Japanese"/"Korean" → `{src}`/`{tgt}` 플레이스홀더 + `_LANG_NAMES` 매핑 사용
- **근거**: `Translator(source_lang="JA", target_lang="KO")` 파라미터가 실제 프롬프트에 반영되지 않아, 향후 다국어 지원 시 프롬프트 수정 필요. 동적 언어명 삽입으로 확장성 확보

---

## D-020: 로깅 아키텍처 — RotatingFileHandler + 프론트엔드 링 버퍼

- **결정**: 백엔드에 `RotatingFileHandler`(10MB × 5 backups), 프론트엔드에 200-entry 인메모리 링 버퍼 방식의 이중 로깅 시스템 구축
- **근거**: 번역 실패 시 원인 추적이 불가능했음 (로그 없음). 파일 로그는 서버 재시작 후에도 보존 가능, 프론트엔드 링 버퍼는 `Logger.dump()`로 브라우저 콘솔에서 즉시 진단 가능
- **백엔드 구현**: `server/utils/logger.py`에 파일 핸들러 추가, HTTP 미들웨어(`main.py`), 라우터/파이프라인 전 계층에 구조화된 로깅
- **프론트엔드 구현**: `api.js`에 `Logger` IIFE 모듈 — `info/warn/error` + timestamp, 200개 초과 시 FIFO 삭제
- **로그 경로**: `logs/mangalens.log` (`.gitignore`에 추가)

---

## D-022: 텍스트 렌더링 — 스트로크 인식 줄바꿈 너비 + x 하한 보장

- **결정**: `_wrap_text()` 호출 시 `wrap_w = usable_w - 2 × stroke_width` 사용, `_draw_horizontal()`에서 `x = max(stroke_width, _PADDING + …)` 하한 적용
- **근거**: `stroke_width`가 커질수록(폰트 크기 비례) 스트로크가 오버레이 왼쪽 경계를 침범하여 글자가 잘림. 줄바꿈 기준 너비를 미리 축소하면 가장 긴 줄도 스트로크 포함 시 오버레이 안에 들어감. x 하한은 이중 안전장치
- **영향 범위**: 소형 버블(stroke_width=1)은 2px, 대형 버블(font_size≥40, stroke_width=2)은 4px 너비 감소 — 텍스트 양에 따라 줄 수가 1줄 늘어날 수 있으나 잘림 없이 정상 표시
- **관련 결정**: D-003 (Pillow 렌더링), Phase 15 수정

---

## D-021: transformers BatchEncoding 호환 — apply_chat_template 반환 타입 분기

- **결정**: `translator.py`의 `_build_input_ids`에서 `apply_chat_template(return_tensors="pt")` 반환값에 대해 `hasattr(result, "input_ids")` 분기 처리
- **근거**: transformers 5.3.0에서 `apply_chat_template`이 `return_tensors="pt"` 옵션 사용 시 bare tensor가 아닌 `BatchEncoding`(dict-like 객체)을 반환하도록 변경됨. 기존 코드는 `result.shape[1]`을 직접 호출하여 `AttributeError` 발생
- **구현**: `BatchEncoding`이면 `result["input_ids"]` 추출, bare tensor면 그대로 사용. 두 버전 모두 호환
- **교훈**: transformers 메이저 업데이트 시 반환 타입 변경에 주의. 테스트에서 mock 사용으로 해당 경로가 커버되지 않았음

---

## D-023: 후리가나 제거 — 탁음/반탁음 보호 패딩 강화 + column-overlap 체크 (Phase 16)

- **결정**: `remove_furigana()` 보호 패딩을 `median_h × 0.15` → `× 0.30`으로 2배 증가, 추가로 소형 component의 x범위가 대형 component의 x범위와 겹치면 같은 텍스트 컬럼으로 보고 무조건 보존
- **근거**: 실제 측정 결과 탁점(゛)이 주변 글자에서 11–29px 떨어져 있었으나 `pad ≈ 4px` (median_h=27.5)라 detection 불가. 또한 후리가나는 주 텍스트와 x범위가 겹치지 않는 별도 서브컬럼 위치 → column-overlap 기준으로 구분 가능
- **구현**:
  - `pad = max(median_h * 0.30, 3.0)` — 기존 0.15 → 0.30
  - 근접도 루프에 column-overlap 체크 추가: `if sx1 < lx2 and sx2 > lx1: near_large = True`
- **검증**: P22 ID3 기준 잘못 삭제된 픽셀 612 → 0
- **관련 결정**: D-009, D-012

---

## D-024: 텍스트 렌더러 — 수직 말풍선 폰트 크기 수정 (fit_h = bh)

- **결정**: 수직 원본(세로쓰기) 말풍선 렌더링 시 `fit_h = max(bw, _MIN_FONT_SIZE * 3)` → `fit_h = bh`로 변경, `actual_render_h = needed_h` → `min(needed_h, bh)` 상한 추가
- **근거**: 세로쓰기 말풍선은 bw(너비)가 좁고(예: 24px) bh(높이)가 긴 형태. 기존 코드는 bw 기반으로 `fit_h=36, usable_h=24`를 계산해 font binary search에서 항상 최솟값(12px)을 반환. 실제로 렌더링 공간은 bh만큼 넓게 존재하므로 bh를 상한으로 써야 함. `actual_render_h` 상한이 없으면 가늘고 긴 오버레이가 말풍선 밖으로 삐져나갈 수 있음
- **수정 위치**: `server/pipeline/text_renderer.py::TextRenderer.render()`, 수직 소스 분기(`if is_vert_src:`) 내부
- **검증**: P22 기준 폰트 12px(최솟값 고정) → 25px+(말풍선 크기 비례), 세로 오버플로우 제거
- **관련 결정**: D-003 (Pillow 렌더링), D-022 (스트로크 인식 줄바꿈)

---

## D-025: 텍스트 렌더러 — 초좁은 수직 말풍선 캔버스 폭 상한 (_vert_cap)

- **결정**: 수직 원본 말풍선 렌더링 시 `render_w = bh` / `fit_h = bh` → `_vert_cap = max(bw*4, _MIN_FONT_SIZE*6)`을 상한으로 `render_w = min(bh, _vert_cap)`, `fit_h = min(bh, _vert_cap)`로 변경
- **근거**: D-024(Phase 16)에서 `fit_h = bh`로 넓혔더니 초좁은 세로 말풍선(bw=8~20, bh=80~140, 비율 bh/bw ≥ 4)에서 두 가지 버그 발생:
  1. **폰트 과대**: `fit_h=bh` → binary search `hi=bh-padding=~90` → font 21~29px (정상 말풍선에 비해 3배+)
  2. **캔버스 오버플로우**: `render_w=bh=102` 캔버스가 말풍선(bw=11) 중심에서 ±46px 밖으로 확장 → LaMa가 지운 영역 외부 manga art 위에 텍스트 렌더링
- **공식**: `_vert_cap = max(bw * 4, _MIN_FONT_SIZE * 6) = max(bw*4, 72)`. `bw*4≥bh`인 "충분히 넓은" 말풍선은 cap이 bh 이상 → Phase 16 동작 그대로 보존. `bw*4<bh`인 초좁은 말풍선만 제한
- **수정 위치**: `server/pipeline/text_renderer.py::TextRenderer.render()`, 수직 분기(`if is_vert_src:`) 내부 3줄
- **검증**: 33/33 페이지 OK, 282 버블. 영향 페이지(P1/P3/P16) 시각 확인 — 텍스트가 말풍선 내부에 적절한 크기로 배치됨. pytest 64/64 통과
- **관련 결정**: D-024 (Phase 16 fit_h=bh), D-003 (Pillow 렌더링)

---

## D-026: 말풍선 근접 병합 갭 확대 (_PROXIMITY_GAP 10→25) — Phase 17 N1

- **결정**: `_PROXIMITY_GAP = 10` → `25` (세로쓰기 텍스트 컬럼 병합 갭)
- **근거**: phase19_final 분석에서 세로쓰기 말풍선의 각 문자 컬럼이 너무 좁은 개별 bbox(w=15~27px)로 잡혀 30건(11%)의 초협소 말풍선 생성. 갭 10px은 컬럼 간 간격(약 12~20px)보다 작아 병합 실패. 25px로 확대하면 세로쓰기 컬럼들이 정상 병합됨.
- **리스크**: 근접하지만 실제로 분리된 별개 말풍선을 잘못 합칠 가능성 → phase19_final 33페이지에서 검증 후 허용 범위로 판단
- **파일**: `server/pipeline/bubble_detector.py`

---

## D-027: scale-down 제거 (font_size×0.85 폐기) — Phase 17 N2

- **결정**: `if render_w > 60 and bh > 60: font_size = max(int(font_size * 0.85), _MIN_FONT_SIZE)` 블록 전면 제거
- **근거**: `_find_best_font_size()`가 이미 bbox에 맞는 최대 폰트를 binary search로 찾음. 추가 0.85 scale-down은 horizontal 버블 82%를 fs=12~13으로 강제 수축 — 가독성 저하의 주요 원인. scale-down 없이도 줄바꿈 + 세로 overflow 축소 루프가 안전 보장을 제공함.
- **B-3 주석 제거**: 기존 "B-3: Adaptive scale-down" 주석도 함께 제거
- **파일**: `server/pipeline/text_renderer.py`

---

## D-028: _MAX_VERT_FONT_SIZE 캡 완화 (24→36) — Phase 17 N3

- **결정**: `_MAX_VERT_FONT_SIZE = 24` → `36`
- **근거**: 세로→가로 변환 말풍선 전체(229건)가 최대 24px에 묶여 가독성 저하. 24px는 원래 "artwork 아래에 텍스트가 덮이는 것 방지" 목적이었으나 결과적으로 과도하게 작음. 36px로 완화하면 정상 크기의 말풍선에서 18~28px 폰트가 가능해지면서 artwork 침범은 `_vert_cap` 기반 overlay 크기 제한이 별도로 방어.
- **파일**: `server/pipeline/text_renderer.py`

---

## D-029: 번역 실패 감지 — 일본어 가나 정규식 (_JA_KANA_RE) — Phase 17 N4

- **결정**: `_postprocess()` 마지막에 히라가나/가타카나 감지(`[\u3040-\u30FF]`) → 발견 시 `""` 반환
- **근거**: 배치 번역 실패 + 개별 fallback도 원문 반환하는 8건에서 일본어 가나가 그대로 번역 결과에 포함됨. 한국어에는 가나가 절대 포함되지 않으므로, 가나 존재 = 번역 실패의 신뢰도 높은 신호. `""` 반환 → `render()` 조기 종료 → 말풍선 blank → 원문 일본어 노출보다 나은 사용자 경험.
- **파일**: `server/pipeline/translator.py`

---

## D-030: 번역 간결성 규칙 추가 (규칙 #6/#7) — Phase 17 N5

- **결정**: `_SYSTEM_MSG_TEMPLATE` 및 `_BATCH_SYSTEM_MSG_TEMPLATE` 양쪽에 "원문과 유사한 분량으로 번역 유지" 규칙 추가
- **근거**: JA→KO 번역 팽창 평균 1.5x, 최대 10x — 말풍선 내 텍스트 넘침의 주요 원인. 기존 6배 hallucination guard는 너무 관대. 명시적 간결성 지시로 모델 출력 길이를 1.0~1.5x로 유도.
- **파일**: `server/pipeline/translator.py`

---

## D-031: inpainting 마스크 dilate 축소 (9×9 iter=3 → 7×7 iter=2) — Phase 17 N6

- **결정**: `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))` + `iterations=3` → `(7,7)` + `iterations=2`
- **근거**: 실효 팽창 반경: 기존 (9×9 iter=3) ≈ 13.5px, 변경 후 (7×7 iter=2) ≈ 7px. 좁은 말풍선(bw=15~30px)에서 마스크가 13px 확장되면 말풍선 경계를 벗어나 주변 artwork까지 inpainting → 아티팩트 발생. 7px로 축소하면 텍스트 anti-aliased edge+stroke 커버에 충분하면서 경계 침범 최소화.
- **파일**: `server/pipeline/text_eraser.py`

---

## D-032: 텍스트 렌더러 — 타원 말풍선 안전 여백 강화 (_PADDING_PCT 0.08→0.15) — Phase 18

- **결정**: `_PADDING_PCT = 0.08` → `0.15` (타원형 말풍선 내접 직사각형 안전 여백)
- **근거**: 타원(장축 a, 단축 b) 내에 내접하는 직사각형의 최대 크기는 `a/√2 × b/√2`이고, 타원 반경 대비 여유 거리는 `(1 - 1/√2)/2 ≈ 0.146`. 기존 0.08은 이 이론값보다 작아 실제 타원형 말풍선(bw=117~342, bh=715~968)에서 텍스트 상단/하단이 타원 경계 밖으로 삐져나오는 클리핑 발생. 0.15는 이론 하한(0.146) 이상으로 실제 말풍선 형태 편차를 커버함.
- **수식**: 내접 직사각형 여백 = $(1 - 1/\sqrt{2})/2 \approx 0.146$ per side → 반올림하여 0.15 채택
- **영향**: 모든 가로쓰기 말풍선의 `usable_w = bw * (1 - 2×0.15)`, `usable_h = bh * (1 - 2×0.15)` → 70% 사용. 이전(84%) 대비 가로/세로 각 7% 추가 여백.
- **검증**: 이미지 14 balloon[0](bw=342, bh=968), balloon[4](bw=117, bh=715), balloon[5](bw=86, bh=870) — 모두 타원 경계 내에 텍스트 완전 수납 확인. "게요." 말소리 완전 표시 확인.
- **파일**: `server/pipeline/text_renderer.py`
- **관련 결정**: D-022 (스트로크 인식 줄바꿈), D-027 (scale-down 제거)

---

## D-033: Magi v2 필수 패키지 추가 (einops + pulp) — Phase 19

- **결정**: `pyproject.toml`에 `"einops>=0.8"`, `"pulp>=2.9"` 추가 후 `uv sync`로 설치 (einops 0.8.2, pulp 3.3.0)
- **근거**: `USE_MAGI_DETECTOR=true` 설정 시 `magiv2` 모델 로딩에서 `ModuleNotFoundError: No module named 'einops'` 및 `'pulp'` 발생. 두 패키지 모두 `ragavsachdeva/magiv2`의 직접 의존성이나 `pyproject.toml`에 누락되어 있었음
- **파일**: `pyproject.toml`
- **관련 결정**: D-011 (Magi v2 통합)

---

## D-034: 연결 말풍선 자동 분리 — _split_tall_boxes() 추가 — Phase 19

- **결정**: `bubble_detector.py`에 `_split_tall_boxes()` 함수 추가. NMS → 병합 → 근접 병합 → 스케일 변환 이후, bbox 확장 전에 실행
- **근거**: 세로로 높은 하나의 bbox 내에 두 개의 연결된 말풍선이 묶여 검출되는 케이스 발생 (예: 가운데가 좁아진 모래시계 형태). 텍스트 세그멘테이션 마스크(`seg_full`)의 행 방향 밀도를 스캔하여 "핀치 포인트"(골짜기)를 탐색하고, 조건 만족 시 두 박스로 분리
- **분리 조건**: `h/w ≥ _SPLIT_ASPECT(3.0)`, `h ≥ _SPLIT_MIN_H(80px)`, 골짜기 밀도 `< peak × _SPLIT_VALLEY_RATIO(0.30)`, 분리된 각 파트 `≥ 30px`
- **상수**: `_SPLIT_ASPECT=3.0`, `_SPLIT_MIN_H=80`, `_SPLIT_VALLEY_RATIO=0.30`
- **파일**: `server/pipeline/bubble_detector.py`

---

## D-035: 말풍선 bbox 자동 확장 — _expand_bbox_to_balloon() 추가 — Phase 19

- **결정**: `bubble_detector.py`에 `_expand_bbox_to_balloon()` 함수 추가. `_split_tall_boxes()` 이후, `_dedup_expanded_boxes()` 전에 실행
- **근거**: YOLO 검출 bbox가 실제 말풍선보다 좁게 잡혀 텍스트 렌더링 영역이 부족하고 글자가 잘리는 문제. 특히 세로쓰기 말풍선(좁고 긴 형태)에서 빈번. 원본 이미지 픽셀 밝기를 기준으로 4방향 확장을 시도하여 실제 말풍선 경계까지 bbox를 넓힘
- **확장 로직**: 각 방향으로 최대 `_EXPAND_MAX_PX(80px)` 탐색. 픽셀 평균 밝기 `< _EXPAND_BRIGHTNESS(200)`이면 확장 중단. 확장 후 `_EXPAND_INSET(4px)` 안전 여백 적용
- **허위 양성 방지**: bbox 내부 평균 밝기 `< 160` → 어두운 배경(바지, 어두운 패널 등) 위의 장면 텍스트로 판단, 확장 건너뜀. 말풍선 내부는 흰색(평균 200+), 비말풍선 배경은 훨씬 낮음
- **검증 강화**: `seg_full` 전달 시 확장 면적이 1.5배 이상으로 커지면 seg 밀도 비교 — 원본 밀도 대비 1/3 미만이면 확장 취소
- **제거**: `text_renderer.py`의 `_probe_balloon_width()` 80줄 제거 (역할이 탐지기 계층으로 이전됨)
- **상수**: `_EXPAND_BRIGHTNESS=200`, `_EXPAND_MAX_PX=80`, `_EXPAND_INSET=4`
- **파일**: `server/pipeline/bubble_detector.py`, `server/pipeline/text_renderer.py`

---

## D-036: 확장 후 중복 박스 제거 — _dedup_expanded_boxes() 추가 — Phase 19

- **결정**: `bubble_detector.py`에 `_dedup_expanded_boxes()` 함수 추가. `_expand_bbox_to_balloon()` 직후 실행
- **근거**: 분리(split)  + 확장(expand) 후 인접한 두 bbox가 서로 침범하는 경우 발생 (예: bubble 1 (375,36,57,187)과 bubble 2 (378,130,52,93)가 확장 후 겹침). 겹침이 작은 박스 면적의 50% 이상이면 큰 박스로 흡수(병합)
- **병합 기준**: `intersection / min(area1, area2) ≥ containment_ratio(0.50)` → 작은 박스를 큰 박스에 흡수
- **파일**: `server/pipeline/bubble_detector.py`

---

## D-037: 어두운 내부 검사 — 비말풍선 영역 확장 방지 — Phase 19

- **결정**: `_expand_bbox_to_balloon()` 내에서 각 박스 처리 전 bbox 내부 평균 밝기를 검사하여 160 미만이면 확장 없이 원본 bbox 반환
- **근거**: YOLO 검출기가 어두운 배경 위 장면 텍스트(scene text, 예: 검은 바지/옷 위의 `そして、`)도 검출함. 이 경우 `_expand_bbox_to_balloon()`이 주변 밝은 배경 픽셀로 가장자리를 탐색하다 말풍선 외부 영역까지 bbox를 크게 확장하는 허위 양성 발생. 말풍선 내부는 흰색(평균 밝기 ~200–250)이고 장면 텍스트 배경은 훨씬 어두우므로(< 160), 이 임계값으로 안전하게 구분 가능
- **임계값**: 160 (말풍선 흰색 ~200+ vs 장면 배경 ~50–130)
- **로깅**: `logger.debug("Expand skipped ... dark interior (mean=...)")` 진단 로그 포함
- **파일**: `server/pipeline/bubble_detector.py`
- **관련 결정**: D-035 (확장 함수 전체 설계)

