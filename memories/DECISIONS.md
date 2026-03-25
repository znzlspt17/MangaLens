# MangaLens — 기술 결정 기록 (DECISIONS.md)

> 주요 기술 선택의 근거를 기록한 문서. 최종 갱신: 2026-03-25

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

## D-005: API 키 세션 — 서버 메모리 저장 (비영구)

- **결정**: API 키를 DB가 아닌 서버 메모리 `session_store` dict에 세션 수명으로 저장
- **근거**: P7 원칙 "서버가 API 키를 영구 저장하지 않음", 보안상 키를 디스크에 쓰지 않음
- **식별자**: `session_id` — Cookie 또는 `X-Session-Id` 헤더로 전달

---

## D-006: HTTP 세션 쿠키 우회 — X-Session-Id 헤더 + localStorage

- **결정**: `secure=True` 쿠키 대신 `localStorage` + `X-Session-Id` 헤더 방식 병행
- **근거**: HTTP (localhost) 환경에서 `secure=True` 쿠키는 브라우저가 저장하지 않음 → API 키 세션 유실
- **구현**:
  - 서버: `GET/POST /api/settings` 응답 JSON에 `session_id` 포함
  - 클라이언트: `api.js`에서 `localStorage['mangalens_session_id']` 저장/조회, 모든 요청 헤더에 `X-Session-Id` 자동 추가
  - HTTPS 환경에서는 쿠키도 병행 작동 (`secure` 조건부 설정)

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
