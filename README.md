# MangaLens

일본어 만화 이미지를 자동으로 번역하는 서비스입니다. 말풍선을 감지하고, OCR → 번역 → 텍스트 재렌더링하여 번역된 이미지를 출력합니다.

## 주요 기능

- **말풍선 검출** — comic-text-detector (YOLOv5s 기반)로 말풍선/나레이션 박스 자동 감지
- **이미지 업스케일** — Real-ESRGAN으로 OCR 정확도 향상
- **OCR** — manga-ocr로 일본어 세로쓰기 텍스트 인식
- **번역** — DeepL API (기본) / Google Translate (대체), JA → KO
- **텍스트 제거** — LaMa inpainting으로 원본 텍스트 자연스럽게 제거
- **텍스트 렌더링** — Pillow + CJK 폰트, 자동 폰트 크기/줄바꿈
- **이미지 합성** — 알파 블렌딩으로 자연스러운 최종 출력
- **GPU 자동 감지** — CUDA / ROCm / CPU 자동 전환

## 요구 사항

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (패키지 매니저)
- GPU 권장 (CUDA 또는 ROCm), CPU도 가능

## 빠른 시작

### 1. 설치

```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치 (가상환경 자동 생성)
uv sync --group dev
```

### 2. 모델 & 폰트 다운로드

```bash
uv run python -m server.download
```

### 3. 환경 설정

```bash
cp .env.example .env
# .env 파일을 열어 DeepL API 키 등을 설정
```

### 4. 서버 실행

```bash
uv run python -m server.main
```

서버가 `http://localhost:20399` 에서 시작됩니다.

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/health` | 서버 상태 확인 |
| POST | `/api/upload` | 단일 이미지 업로드 & 번역 |
| POST | `/api/upload/bulk` | 다수 이미지 Bulk 업로드 (multipart/ZIP) |
| GET | `/api/status/{task_id}` | 작업 진행 상태 조회 |
| GET | `/api/result/{task_id}` | 번역 결과 다운로드 |
| POST | `/api/settings` | 사용자 API 키 설정 |
| GET | `/api/settings` | 현재 설정 조회 |
| GET | `/api/system/gpu` | GPU 환경 정보 |
| WS | `/ws/progress/{task_id}` | 실시간 진행률 알림 |

## 사용 예시

### 단일 이미지 번역

```bash
curl -X POST http://localhost:20399/api/upload \
  -F "file=@manga_page.jpg"

# 응답: {"task_id": "abc123...", "status": "queued"}

# 상태 확인
curl http://localhost:20399/api/status/abc123...

# 결과 다운로드
curl -o translated.png http://localhost:20399/api/result/abc123...
```

### 사용자 API 키 설정

```bash
curl -X POST http://localhost:20399/api/settings \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: my-session" \
  -d '{"deepl_api_key": "your-key-here"}'
```

## GPU 설정

### NVIDIA (CUDA)

`pyproject.toml`의 인덱스 URL이 기본으로 CUDA를 가리킵니다:

```bash
uv sync
```

### AMD (ROCm gfx1201)

```bash
# 환경변수 설정
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export PYTORCH_ROCM_ARCH=gfx1201

# pyproject.toml에서 인덱스 URL을 rocm6.3으로 변경 후:
uv sync
```

### CPU 전용

```bash
# pyproject.toml에서 인덱스 URL을 cpu로 변경 후:
uv sync
```

## 테스트

```bash
uv run pytest tests/ -v
```

## 프로젝트 구조

```
server/
├── main.py              # FastAPI 앱 진입점
├── config.py            # 환경변수 관리
├── gpu.py               # GPU 자동 감지
├── download.py          # 모델/폰트 다운로드
├── state.py             # 공유 태스크 상태
├── routers/
│   ├── upload.py        # 업로드 API
│   ├── result.py        # 결과 다운로드 API
│   ├── settings.py      # 사용자 설정 API
│   └── ws.py            # WebSocket 진행률
├── pipeline/
│   ├── orchestrator.py  # 7단계 파이프라인 오케스트레이션
│   ├── bubble_detector.py
│   ├── preprocessor.py
│   ├── ocr_engine.py
│   ├── translator.py
│   ├── text_eraser.py
│   ├── text_renderer.py
│   └── compositor.py
├── schemas/
│   └── models.py        # Pydantic 모델
└── utils/
    ├── image.py         # 이미지 보안 검증
    ├── reading_order.py # 우→좌 읽기 순서
    └── logger.py        # 로깅 설정
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DEEPL_API_KEY` | (없음) | DeepL 번역 API 키 |
| `GOOGLE_API_KEY` | (없음) | Google 번역 API 키 |
| `GPU_BACKEND` | `auto` | GPU 백엔드 (auto/cuda/rocm/cpu) |
| `MAX_UPLOAD_SIZE` | `52428800` | 최대 업로드 크기 (50MB) |
| `MAX_CONCURRENT_TASKS` | `1` | 동시 파이프라인 수 |
| `RESULT_TTL_SECONDS` | `3600` | 결과 보존 시간 (1시간) |
| `SKIP_WARMUP` | `false` | 모델 워밍업 건너뛰기 |
| `ALLOWED_ORIGINS` | (비어 있음) | CORS 허용 출처. 라이브 배포 시 명시적 도메인만 설정 |

## 라이브 배포 전 점검

- 현재 기본 설정은 **same-origin 배포**를 기준으로 합니다. 외부 프론트엔드에서 API를 호출해야 할 때만 `ALLOWED_ORIGINS` 에 정확한 도메인을 설정하세요.
- 브라우저 API 키는 서버 세션(HttpOnly cookie 기반)에만 저장되며, 프론트엔드 `sessionStorage` 에는 저장하지 않습니다.
- `task_store`, `session_store` 는 메모리 기반이라 서버 재시작 시 초기화됩니다. 따라서 **단일 인스턴스 배포** 또는 외부 상태 저장소 도입 전까지는 수평 확장에 적합하지 않습니다.
- 공개 인터넷에 노출할 경우, 리버스 프록시/로드밸런서에서 HTTPS 강제, 요청 크기 제한, 요청 속도 제한을 함께 설정하는 것을 권장합니다.
- 프록시 뒤에서 HTTPS 를 종료한다면, 프록시가 `X-Forwarded-Proto` 같은 전달 헤더를 **직접 설정하고 클라이언트가 보낸 값을 제거/덮어쓰도록** 구성하세요.

## 라이선스

이 프로젝트는 개인 사용 목적입니다.
