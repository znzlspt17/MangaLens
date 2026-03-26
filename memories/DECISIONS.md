# MangaLens — 기술 결정 기록 (DECISIONS.md)

> 주요 기술 선택의 근거를 기록한 문서. 최종 갱신: 2026-03-26 (D-019)

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

## D-012: 후리가나 제거 — 탁점/반탁점 보호 로직 추가

- **결정**: `remove_furigana()` 에서 작은 connected component가 큰 글리프 bbox에 근접하면 삭제하지 않음 (탁점/반탁점 보존)
- **근거**: 탁점(゛)과 반탁점(゜)은 이진화 후 문자 본체와 별도 connected component로 분리되나, 문자 본체 바로 옆에 위치. 탁점 없이 OCR할 경우 `が→か`, `ぱ→は` 등 의미가 완전히 달라져 번역 품질 심각하게 저하
- **구현**: 작은 컴포넌트 bbox와 모든 큰 글리프 bbox 간 근접도 검사 — padding = `max(median_h * 0.15, 3px)`. 근접하면 보존, 멀면 후리가나로 제거
- **관련 결정**: D-009 (후리가나 제거 최초 구현) 보완

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
