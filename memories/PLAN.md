# 일본어 만화 번역 서비스 — PLAN.md

> **프로젝트명**: MangaLens  
> **목표**: 일본어 만화/도서 이미지를 입력받아 말풍선을 감지하고, OCR → 번역 → 텍스트 재렌더링하여 번역된 이미지를 출력하는 서비스  
> **최종 수정**: 2026-03-25  

---

## 1. 핵심 원칙

| # | 원칙 | 설명 |
|---|------|------|
| P1 | 우→좌 읽기 순서 | 일본 도서는 오른쪽에서 왼쪽으로 페이지를 넘기고, 문장도 우→좌 순서 |
| P2 | 세로쓰기(縦書き) | 대사는 세로쓰기이며, 번역 시에도 이 레이아웃을 고려 |
| P3 | 말풍선 우선 탐지 | 전체 화면 OCR 금지 — 반드시 말풍선(balloon)을 먼저 정확히 검출 |
| P4 | 전처리 후 OCR | 말풍선 크롭 → 업스케일링(SR) → OCR 순서 준수 |
| P5 | 원본 대조 기록 | OCR 원문과 번역문을 구조화된 형식으로 기록 |
| P6 | 고품질 텍스트 렌더링 | OpenCV `putText`가 아닌 전용 렌더링 엔진 사용 |
| P7 | 사용자 API 키 관리 | 서버에 기본값은 있되, 서버가 책임지지 말고 사용자가 자신의 API 키를 입력할 수 있어야 함. 서버는 API 키를 영구 저장하지 않음 (세션 수명) |
| P8 | GPU 자동 감지 | CUDA 130 혹은 ROCm 6.3 gfx1201 환경을 자동 감지하여 적절한 백엔드 선택 |

---

## 2. 파이프라인 아키텍처

```
[입력 이미지]
     │
     ▼
┌─────────────────────┐
│ ① 말풍선 검출        │  ← comic-text-detector (YOLO 기반)
│   (Bubble Detection) │     말풍선 bbox + mask 추출
└─────────┬───────────┘
          │ bbox 목록 (우→좌 정렬)
          ▼
┌─────────────────────┐
│ ② 크롭 & 업스케일    │  ← Real-ESRGAN (x2/x4 업스케일링)
│   (Crop & Upscale)   │     작은 말풍선도 OCR 정확도 확보
└─────────┬───────────┘
          │ 업스케일된 크롭 이미지
          ▼
┌─────────────────────┐
│ ③ OCR 수행           │  ← manga-ocr (일본어 세로쓰기 특화)
│   (Text Recognition) │     세로쓰기 자동 인식
└─────────┬───────────┘
          │ 일본어 텍스트
          ▼
┌─────────────────────┐
│ ④ 번역               │  ← DeepL API (기본) / Google Translate (대체)
│   (Translation)      │     사용자 제공 API 키 우선
└─────────┬───────────┘
          │ 번역된 텍스트          ┌──── ④와 ⑤는 서로 독립적이므로
          ▼                      │    asyncio.gather로 병렬 실행 가능
┌─────────────────────┐          │
│ ⑤ 원본 텍스트 제거    │  ← LaMa │ Inpainting Model
│   (Text Erasure)     │     말풍선 내부 텍스트만 정밀 제거
└─────────┬───────────┘
          │ 깨끗한 말풍선
          ▼
┌─────────────────────┐
│ ⑥ 번역 텍스트 렌더링  │  ← Pillow + CJK 폰트 렌더링 엔진
│   (Text Rendering)   │     자동 폰트 크기 조절, 줄바꿈 처리
└─────────┬───────────┘
          │ 텍스트가 그려진 말풍선
          ▼
┌─────────────────────┐
│ ⑦ 합성               │  ← 원본 이미지에 번역된 말풍선 오버레이
│   (Composition)      │     알파 블렌딩으로 자연스러운 합성
└─────────┬───────────┘
          │
          ▼
   [번역된 이미지 출력]
```

---

## 3. 기술 스택 상세

### 3.1 말풍선 검출 — comic-text-detector

| 항목 | 내용 |
|------|------|
| **모델** | [comic-text-detector](https://github.com/dmMaze/comic-text-detector) |
| **아키텍처** | YOLO 기반 텍스트 영역 + 말풍선 검출 |
| **출력** | 말풍선 bounding box, 텍스트 영역 mask, 텍스트 방향(세로/가로) |
| **선택 이유** | 만화 특화, 세로쓰기 방향 판별 내장, 말풍선과 효과음 구분 가능 |
| **읽기 순서** | 검출 후 bbox를 우→좌, 위→아래로 정렬 (일본 만화 읽기 순서) |
| **나레이션 박스** | 사각형 나레이션/모노로그 박스도 말풍선과 동일하게 검출·처리 |

### 3.2 이미지 업스케일링 — Real-ESRGAN

| 항목 | 내용 |
|------|------|
| **모델** | Real-ESRGAN (x2/x4) |
| **용도** | 작은 크롭된 말풍선 이미지를 업스케일하여 OCR 정확도 향상 |
| **배율** | 기본 x2, 작은 버블은 x4 (자동 판별) |
| **GPU 지원** | CUDA / ROCm 자동 선택 |

### 3.3 OCR 엔진 — manga-ocr

| 항목 | 내용 |
|------|------|
| **모델** | [manga-ocr](https://github.com/kha-white/manga-ocr) |
| **특징** | 일본 만화 세로쓰기 텍스트에 특화된 OCR |
| **백엔드** | TrOCR 기반 (Vision Encoder-Decoder) |
| **입력** | 업스케일된 말풍선 크롭 이미지 |
| **선택 이유** | 일반 OCR (Tesseract 등) 대비 만화 세로쓰기 인식률 월등 |

### 3.4 번역 — DeepL API

| 항목 | 내용 |
|------|------|
| **기본 엔진** | DeepL API (JA → KO) |
| **대체 엔진** | Google Cloud Translation API v2 (Basic) |
| **API 키 관리** | 서버 기본값 `.env` + 사용자 입력 키 우선 적용 |
| **배치 처리** | 한 페이지의 모든 말풍선 텍스트를 배치로 번역 (API 호출 최적화) |
| **문맥 번역** | 같은 페이지의 말풍선들을 읽기 순서대로 연결하여 대화 문맥을 유지한 채 번역 (앞뒤 말풍선을 context로 함께 전달) |

### 3.5 텍스트 제거 — LaMa Inpainting

| 항목 | 내용 |
|------|------|
| **모델** | [LaMa](https://github.com/advimman/lama) (Large Mask Inpainting) |
| **역할** | 말풍선 내부의 일본어 텍스트를 자연스럽게 제거 |
| **마스크 생성** | comic-text-detector의 텍스트 mask 활용 |
| **OpenCV 인페인팅 대비** | 경계 자연스러움, 배경 패턴 복원력 월등히 우수 |

### 3.6 텍스트 렌더링 — Pillow + CJK 폰트 엔진

| 항목 | 내용 |
|------|------|
| **렌더링 엔진** | **Pillow (PIL)** + FreeType 기반 |
| **OpenCV 대비 장점** | ① CJK 유니코드 완벽 지원 ② 안티앨리어싱 ③ 자동 줄바꿈 ④ 세로/가로 텍스트 레이아웃 ⑤ 폰트 크기 자동 조절 ⑥ 텍스트 외곽선/그림자 |
| **폰트** | Noto Sans KR (기본), 사용자 커스텀 폰트 지원 |
| **레이아웃 전략** | 말풍선 bbox 내부에서 최대 폰트 크기 자동 계산, 여백(padding) 확보 |
| **세로쓰기 → 가로쓰기** | 일본어 세로쓰기는 한국어 가로쓰기로 변환하되, 말풍선 형태에 따라 자동 판단 |

### 3.7 합성 — Alpha Blending Compositor

| 항목 | 내용 |
|------|------|
| **방식** | 인페인팅된 깨끗한 말풍선 + 렌더링된 텍스트 → 원본 이미지에 오버레이 |
| **알파 블렌딩** | 경계 부분 부드러운 합성 (feathering) |
| **출력 해상도** | 원본과 동일 (업스케일은 OCR 전처리 전용, 최종 출력에 미반영) |
| **출력 포맷** | PNG (무손실) / JPEG (선택) |

---

## 4. OCR 대조 기록 형식

원본 OCR 텍스트와 번역문을 비교할 수 있는 구조화된 JSON 형식:

```json
{
  "version": "1.0",
  "source_file": "manga_page_001.jpg",
  "processed_at": "2026-03-24T10:30:00+09:00",
  "reading_direction": "rtl",
  "image_size": { "width": 1200, "height": 1800 },
  "bubbles": [
    {
      "id": 1,
      "reading_order": 1,
      "bbox": { "x": 800, "y": 100, "w": 200, "h": 300 },
      "text_direction": "vertical",
      "original_text": "お前はもう死んでいる",
      "ocr_confidence": 0.97,
      "translated_text": "너는 이미 죽어 있다",
      "translation_engine": "deepl",
      "font_size_used": 24,
      "bubble_type": "speech",
      "crop_file": "crops/manga_page_001_bubble_001.png"
    },
    {
      "id": 2,
      "reading_order": 2,
      "bbox": { "x": 400, "y": 150, "w": 180, "h": 250 },
      "text_direction": "vertical",
      "original_text": "何だと！",
      "ocr_confidence": 0.94,
      "translated_text": "뭐라고!",
      "translation_engine": "deepl",
      "font_size_used": 28,
      "bubble_type": "speech",
      "crop_file": "crops/manga_page_001_bubble_002.png"
    }
  ],
  "summary": {
    "total_bubbles": 2,
    "avg_confidence": 0.955,
    "processing_time_ms": 3200
  }
}
```

**기록 파일 위치**: `output/<원본파일명>/translation_log.json`

---

## 5. 서버 아키텍처

### 5.1 프레임워크: FastAPI

```
서버 구조:

FastAPI (비동기 ASGI)
├── GET  /api/health            ← 서버 상태 확인 (헬스체크)
├── POST /api/upload           ← 단일 이미지 업로드 & 번역
├── POST /api/upload/bulk      ← 다수 이미지 Bulk 업로드
├── GET  /api/status/{task_id} ← 작업 진행 상태 조회
├── GET  /api/result/{task_id} ← 번역 결과 다운로드
├── POST /api/settings         ← 사용자 API 키 설정
├── GET  /api/settings         ← 현재 설정 조회
├── GET  /api/system/gpu       ← GPU 환경 정보 조회
└── WebSocket /ws/progress/{task_id}  ← 실시간 진행률 알림
```

### 5.2 Bulk 업로드 처리

```
설계 방향:
1. 다수 이미지를 ZIP 또는 multipart/form-data로 수신
2. 백그라운드 태스크 큐 (asyncio.Queue 기반)
3. 이미지당 독립적 파이프라인 실행
4. WebSocket으로 이미지별 진행률 전송
5. 완료 시 ZIP으로 일괄 다운로드
```

### 5.3 사용자 API 키 관리

```
"서버가 책임지지 말고 API를 사용자한테서 입력 받을 수 있어야 돼"

우선순위:
1. 요청 헤더의 사용자 API 키 (X-DeepL-Key, X-Google-Key) ← 최우선
2. 세션 설정으로 저장된 키 (POST /api/settings)
3. 서버 .env 파일의 기본 키 ← fallback

* 서버는 API 키를 영구 저장하지 않음 (세션 수명)
* 세션 = 서버 메모리 내 dict (key: session_id). 서버 재시작 시 소멸.
* session_id는 클라이언트가 Cookie 또는 헤더로 전달. 없으면 서버가 UUID 발급.
```

### 5.4 GPU 환경 자동 감지

```
WSL2 환경 감지 로직:

서버 시작 시
├── nvidia-smi 실행 가능? → CUDA 경로
│   ├── torch.cuda.is_available() 확인
│   ├── CUDA 버전/GPU 모델 로깅
│   └── device = "cuda"
├── rocm-smi 실행 가능? → ROCm 경로
│   ├── torch.hip.is_available() 확인  (ROCm)
│   ├── ROCm 버전 / gfx 아키텍처 확인 (gfx1201 등)
│   ├── HSA_OVERRIDE_GFX_VERSION 설정
│   └── device = "cuda" (ROCm PyTorch도 cuda API 사용)
└── 둘 다 없음 → CPU fallback
    └── device = "cpu" (경고 로그)
```

**환경 변수 자동 설정** (ROCm gfx1201):
```bash
HSA_OVERRIDE_GFX_VERSION=12.0.1
PYTORCH_ROCM_ARCH=gfx1201
```

---

## 6. 프로젝트 디렉토리 구조

```
test0320/
├── README.md                       # 프로젝트 소개 & 실행 가이드
├── .env                            # 기본 API 키 (gitignore 대상)
├── .env.example                    # API 키 템플릿
├── pyproject.toml                  # 패키지 및 의존성 관리 (uv)
├── uv.lock                         # uv 의존성 잠금 파일
│
├── .github/
│   └── agents/
│       ├── pipeline.agent.md       # ML 파이프라인 Agent
│       ├── server.agent.md         # 서버/API Agent
│       ├── gpu.agent.md            # GPU/환경 Agent
│       ├── qa.agent.md             # 테스트 Agent
│       └── frontend.agent.md       # 프론트엔드 Agent
│
├── server/
│   ├── __init__.py
│   ├── main.py                     # FastAPI 앱 진입점
│   ├── config.py                   # 환경변수, 설정 관리
│   ├── gpu.py                      # GPU 감지 (CUDA/ROCm/CPU)
│   ├── state.py                    # 태스크 상태 저장소 (task_store dict)
│   ├── download.py                 # 모델/폰트 다운로드 (python -m server.download)
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── upload.py               # 업로드 엔드포인트 (단일/벌크)
│   │   ├── result.py               # 결과 조회/다운로드
│   │   ├── settings.py             # 사용자 설정 (API 키 등)
│   │   └── ws.py                   # WebSocket 진행률 알림
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # 파이프라인 전체 오케스트레이션
│   │   ├── bubble_detector.py      # 말풍선 검출 (comic-text-detector)
│   │   ├── preprocessor.py         # 크롭 & 업스케일 (Real-ESRGAN)
│   │   ├── ocr_engine.py           # OCR (manga-ocr)
│   │   ├── translator.py           # 번역 (DeepL / Google)
│   │   ├── text_eraser.py          # 텍스트 제거 (LaMa Inpainting)
│   │   ├── text_renderer.py        # 텍스트 렌더링 (Pillow CJK)
│   │   └── compositor.py           # 최종 합성
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── models.py               # Pydantic 모델 (요청/응답)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image.py                # 이미지 유틸리티
│       ├── reading_order.py        # 우→좌 읽기 순서 정렬
│       └── logger.py               # 로깅 설정
│
├── fonts/                          # 번역 텍스트 렌더링용 폰트
│   └── .gitkeep
│
├── models/                         # ML 모델 가중치 (gitignore)
│   └── .gitkeep
│
├── output/                         # 번역 결과 출력 (gitignore)
│   └── .gitkeep
│
├── frontend/                       # 웹 프론트엔드 (정적 파일)
│   ├── index.html                  # 메인 SPA 페이지
│   ├── css/
│   │   ├── style.css               # 메인 스타일시트
│   │   └── themes.css              # 다크모드/라이트모드 테마 변수
│   ├── js/
│   │   ├── app.js                  # 앱 초기화 & 라우팅
│   │   ├── upload.js               # 업로드 로직 (드래그앤드롭, FormData)
│   │   ├── progress.js             # WebSocket 진행률 관리
│   │   ├── result.js               # 결과 뷰어 & 다운로드
│   │   ├── settings.js             # API 키 설정 관리
│   │   └── api.js                  # API 호출 유틸리티 (fetch 래퍼)
│   └── assets/
│       └── icons/                  # SVG 아이콘
│
├── memories/                       # PM 추적 문서
│   ├── PLAN.md                     # 이 문서 (마스터 기획서)
│   ├── idea.txt                    # 원본 요구사항 + 보강 내용
│   ├── ARCHITECTURE.md             # 아키텍처 결정 기록
│   ├── PROGRESS.md                 # 진행 상황 추적
│   └── DECISIONS.md                # 기술 선택 근거 기록
│
└── tests/
    ├── __init__.py
    ├── conftest.py                 # 공유 픽스처 (mock_gpu, app, client)
    ├── test_frontend.py            # 프론트엔드 정적 파일 서빙 테스트
    ├── test_gpu.py                 # GPU 감지 16개 테스트
    ├── test_integration.py         # API 통합 테스트 (Settings/Status/Result/WS/Upload/TTL)
    ├── test_pipeline.py            # 파이프라인 모듈 단위 테스트 + 모델 캐시
    ├── test_reading_order.py       # 읽기 순서 정렬 테스트
    ├── test_security.py            # 보안 검증 27개 테스트
    └── test_upload.py              # 업로드 엔드포인트 테스트
```

---

## 7. Agent 구성

PM(프로젝트 관리)은 memories를 통해 직접 수행. 코드 작성은 4개 Agent가 담당:

| Agent | 역할 | 담당 영역 | 파일 위치 |
|-------|------|-----------|-----------|
| **Pipeline** | ML 파이프라인 전문가 | 말풍선 검출, 업스케일, OCR, 인페인팅, 렌더링, 합성 | `server/pipeline/` |
| **Server** | 서버/API 전문가 | FastAPI, 라우터, WebSocket, Bulk, 보안, 동시 처리 | `server/routers/`, `server/main.py` |
| **GPU** | GPU/환경 전문가 | CUDA/ROCm 감지, 모델 다운로드, 환경 셋업 | `server/gpu.py`, `pyproject.toml` |
| **QA** | 테스트 전문가 | 단위/통합/API/보안 테스트, 원칙 위반 검증 | `tests/` |
| **Frontend** | 프론트엔드 전문가 | 웹 UI, 업로드, 진행률, 결과 뷰어, 설정, 다크모드 | `frontend/` |

### Agent 간 경계 규칙
- Pipeline은 서버 라우터/GPU 감지 코드를 작성하지 않는다
- Server는 ML 모델 로딩/추론 코드를 작성하지 않는다
- GPU는 ML 추론/API 라우터 코드를 작성하지 않는다
- QA는 프로덕션 코드를 수정하지 않는다 — 버그 발견 시 보고만
- Frontend는 백엔드 Python 코드를 작성하지 않는다 — API 호출만
- Server가 정적 파일 서빙 설정(`StaticFiles`)을 담당, Frontend가 실제 UI 파일 작성

---

## 8. 의존성

### 종속성 관리: uv

| 항목 | 내용 |
|------|------|
| **패키지 매니저** | [uv](https://docs.astral.sh/uv/) (Astral) |
| **설정 파일** | `pyproject.toml` (PEP 621) |
| **잠금 파일** | `uv.lock` (커밋 대상) |
| **가상환경** | `uv sync` 시 `.venv` 자동 생성 |
| **실행** | `uv run python -m server.main` |
| **의존성 추가** | `uv add <패키지>` |
| **개발 의존성** | `uv add --group dev <패키지>` |
| **GPU 의존성** | `[tool.uv.index]` + `[tool.uv.sources]`로 PyTorch 인덱스 분리 |

### 핵심 패키지

| 패키지 | 용도 | 버전 기준 |
|---------|------|-----------|
| `fastapi` | 웹 서버 | >=0.115 |
| `uvicorn[standard]` | ASGI 서버 | >=0.32 |
| `python-multipart` | 파일 업로드 | >=0.0.18 |
| `torch` | ML 프레임워크 (CUDA/ROCm) | >=2.6 |
| `torchvision` | 이미지 처리 | >=0.21 |
| `manga-ocr` | 일본어 만화 OCR | >=0.1.14 |
| `Pillow` | 이미지 처리 & 텍스트 렌더링 | >=11.0 |
| `opencv-python-headless` | 이미지 전처리 | >=4.10 |
| `numpy` | 수치 연산 | >=1.26 |
| `httpx` | 비동기 HTTP 클라이언트 (번역 API) | >=0.28 |
| `pydantic` | 데이터 검증 | >=2.10 |
| `pydantic-settings` | 환경변수 설정 | >=2.7 |
| `websockets` | WebSocket 지원 | >=14.0 |
| `aiofiles` | 비동기 파일 I/O | >=24.1 |
| `ultralytics` | YOLOv5 backbone (BubbleDetector) | >=8.0 |

### ML 모델 관련

| 패키지 | 용도 | 설치 방식 |
|---------|------|-----------|
| `comic-text-detector` | 말풍선/텍스트 영역 검출 | pip 패키지 아님 — models/에 가중치 직접 배치, torch로 직접 추론 |
| `realesrgan` | 이미지 업스케일링 | PyPI (>=0.3.0) |
| `ultralytics` | YOLOv5 백본 (comic-text-detector 의존) | PyPI (>=8.0) |
| `LaMa` | 텍스트 인페인팅 | pip 래퍼(simple-lama-inpainting)가 Pillow<10 요구하여 충돌 — torch로 직접 추론 구현 |

### 개발 의존성 (uv group: dev)

| 패키지 | 용도 |
|---------|------|
| `pytest` | 테스트 프레임워크 |
| `pytest-asyncio` | 비동기 테스트 |
| `httpx` | FastAPI TestClient |

### PyTorch GPU 인덱스 설정 (pyproject.toml)

```toml
[tool.uv]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchvision = { index = "pytorch-cu128" }
basicsr = { path = "vendor/basicsr-1.4.2" }  # Python 3.14 PEP 667 패치 적용
```

> ROCm 환경에서는 인덱스 URL을 `https://download.pytorch.org/whl/rocm6.3`으로 변경.
> CPU 환경에서는 `https://download.pytorch.org/whl/cpu`로 변경.

---

## 9. 마일스톤

### Phase 1: 기반 구축 ✅ 완료
- [x] PLAN.md 작성
- [x] 프로젝트 디렉토리 구조 생성
- [x] pyproject.toml 작성 (uv)
- [x] GPU 감지 모듈 구현
- [x] FastAPI 서버 스캐폴딩

### Phase 2: 핵심 파이프라인 ✅ 완료
- [x] 말풍선 검출 모듈 (comic-text-detector)
- [x] 크롭 & 업스케일 모듈 (Real-ESRGAN)
- [x] OCR 모듈 (manga-ocr)
- [x] 번역 모듈 (DeepL API)
- [x] OCR 대조 기록 시스템

### Phase 3: 렌더링 & 합성 ✅ 완료
- [x] LaMa 텍스트 제거 모듈
- [x] Pillow 텍스트 렌더링 엔진
- [x] 말풍선 합성 모듈
- [x] 읽기 순서 정렬 (우→좌)

### Phase 4: 서버 완성 ✅ 완료
- [x] 단일 이미지 업로드 API
- [x] Bulk 업로드 + 백그라운드 처리
- [x] 사용자 API 키 관리
- [x] WebSocket 진행률 알림
- [x] 결과 다운로드 (단일/ZIP)

### Phase 5: 테스트 & 최적화 ✅ 완료
- [x] 단위 테스트 작성 (159개 PASSED)
- [x] 통합 테스트
- [x] GPU 메모리 최적화 (글로벌 모델 캐시)
- [x] 배치 처리 성능 튜닝 (torch.no_grad, empty_cache)

### Phase 6: 프론트엔드 ✅ 완료
- [x] FastAPI 정적 파일 서빙 설정 (StaticFiles)
- [x] 메인 페이지 (index.html) — 레이아웃 & 네비게이션
- [x] 이미지 업로드 UI — 드래그앤드롭, 파일 선택, 미리보기
- [x] Bulk 업로드 UI — 다수 파일 선택, 진행률 표시
- [x] WebSocket 진행률 패널 — 실시간 단계별 상태
- [x] 결과 뷰어 — 원본↔번역본 비교, 확대/축소, 다운로드
- [x] 설정 패널 — API 키 입력, 마스킹, 세션 관리
- [x] 시스템 정보 표시 — GPU 상태, 서버 헬스
- [x] 다크모드/라이트모드 — CSS 변수 테마 전환
- [x] 반응형 디자인 — 모바일/태블릿/데스크톱 대응
- [x] 에러 UX — 사용자 친화적 에러 메시지, 재시도

### Phase 7: UI 개선 & 버그 수정 ✅ 완료 (2026-03-25)
- [x] 3단계 진행 인디케이터 바 (업로드 → 번역 중 → 완료)
- [x] API 키 경고 배너 (키 미설정 감지, 설정 화면 바로가기)
- [x] 다시 시도 버튼 (실패 시 표시)
- [x] 히어로 배너 (서비스 설명 문구 + AI 번역 뱃지)
- [x] 업로드 카드 컴포넌트 (업로드 영역 그룹핑)
- [x] 배경 장식 orb 애니메이션 (CSS pseudo-element)
- [x] 헤더 글래스모피즘 (backdrop-filter + 그라디언트 로고)
- [x] 번역 시작 버튼 파일 수 표시 + 서버 준비 전 비활성화
- [x] 다운로드 버튼 로딩 피드백 & 성공 토스트
- [x] 설정 모달 닫기 시 입력값 유지
- [x] HTTP 세션 쿠키 문제 해결 (X-Session-Id 헤더 + localStorage)
- [x] session_id JSON 응답 포함 (UserSettingsResponse)
- [x] ultralytics 누락 의존성 추가 → ML 말풍선 검출 복원
- [x] basicsr Python 3.14 PEP 667 패치 (vendor/basicsr-1.4.2)

---

## 10. 효과음(오노마토페) 처리 정책

만화에는 말풍선 밖에 그림 위에 직접 그려진 효과음(「ドカーン」「ザワザワ」등)이 존재한다.

```
처리 정책:
- Phase 1~4에서는 효과음을 번역 대상에서 제외 (말풍선만 처리)
- comic-text-detector가 효과음 영역도 검출하므로, 타입을 구분하여 스킵
- 향후 옵션으로 효과음 번역 on/off 토글 제공 가능

이유:
- 효과음은 그림 위에 직접 그려져 있어 인페인팅 난이도가 높음
- 배경 복원이 불완전하면 원본보다 품질이 떨어짐
- 말풍선 번역이 우선 목표이므로, 효과음은 별도 Phase로 분리
```

---

## 11. 에러 복구 및 부분 실패 전략

```
말풍선 단위:
- OCR 실패 (confidence < 0.3) → 해당 말풍선 원본 유지, 로그에 기록
- 번역 API 에러 → 최대 3회 재시도 (exponential backoff)
- 재시도 실패 → 원문 그대로 렌더링 + 로그에 "translation_failed" 표기

Bulk 처리:
- 이미지별 독립 실행 → 1장 실패해도 나머지 계속 진행
- 실패한 이미지는 결과에 포함하되 상태를 "failed"로 표기
- 최종 ZIP에 translation_log.json 포함 (처리 결과 기록)

번역 API 키 에러:
- 401/403 에러 → 즉시 중단, 사용자에게 키 재입력 요청
- 429 (Rate Limit) → backoff 후 재시도
```

---

## 12. 모델 및 리소스 다운로드 전략

```
필요한 모델 (총 약 2~3 GB):
- comic-text-detector 가중치 (~200 MB)
- manga-ocr 가중치 (~400 MB, HuggingFace 자동 캐시)
- Real-ESRGAN 가중치 (~60 MB)
- LaMa 가중치 (~200 MB)

필요한 폰트:
- Noto Sans KR (~16 MB, Google Fonts)

다운로드 시점:
- 서버 최초 시작 시 자동 확인 + 다운로드
- 이미 다운로드된 모델은 스킵 (MODEL_CACHE_DIR 기준)
- 다운로드 진행률을 로그로 출력
- 오프라인 환경 대비: 수동 배치 스크립트 (`python -m server.download`) 제공

캐시 위치:
- 모델: ./models/ (설정 가능)
- 폰트: ./fonts/ (설정 가능)
- HuggingFace 모델: ~/.cache/huggingface/ (기본)
```

---

## 13. 동시 처리 제한 (Concurrency)

```
GPU 메모리 보호:
- 파이프라인 실행은 asyncio.Semaphore로 동시 실행 수 제한
- 기본값: 1 (GPU 1개 기준, 순차 처리)
- 설정 가능: MAX_CONCURRENT_TASKS 환경변수
- VRAM 8GB 미만 → 강제 1, 8GB 이상 → 최대 2

Bulk 업로드:
- BackgroundTasks + asyncio.Semaphore로 순차 실행
- 이미지별 순차 파이프라인 처리 (for 루프)
- Semaphore로 동시 태스크 수 제한
```

---

## 14. 업로드 파일 보안 검증

```
검증 단계:
1. 확장자 검사: .jpg, .jpeg, .png, .webp, .bmp, .tiff만 허용
2. Magic bytes 검사: 파일 헤더로 실제 이미지 포맷 확인
   - JPEG: FF D8 FF
   - PNG: 89 50 4E 47
   - WebP: 52 49 46 46 ... 57 45 42 50
3. 파일 크기 제한: 기본 50MB (MAX_UPLOAD_SIZE 설정 가능)
4. 파일명 정규화: 경로 탐색 문자 (../ 등) 제거, UUID로 내부 저장
5. Pillow.Image.verify()로 이미지 무결성 검증
```

---

## 15. CORS 정책

```
허용 Origin:
- 개발: http://localhost:*, http://127.0.0.1:*
- 프로덕션: 환경변수 ALLOWED_ORIGINS로 설정 (콤마 구분)
- 기본값: ["*"] (로컬 개발 용도)

허용 메서드: GET, POST, OPTIONS
허용 헤더: Content-Type, X-DeepL-Key, X-Google-Key, X-Session-Id
자격 증명(credentials): true (세션 쿠키 전달 용)
```

---

## 16. 결과 파일 보존 정책 (TTL)

```
디스크 성장 방지:
- output/ 디렉토리 내 결과 파일에 TTL 적용
- 기본 TTL: 1시간 (RESULT_TTL_SECONDS 환경변수로 설정 가능)
- 정리 주기: 5분마다 백그라운드 태스크로 만료 결과 삭제
- 삭제 대상: 번역 결과 이미지 + 크롭 + translation_log.json + 하위 디렉토리
- 클라이언트가 아직 다운로드하지 않은 결과는 다운로드 후 즉시 삭제 옵션(DELETE_AFTER_DOWNLOAD) 제공
```

---

## 17. 모델 워밍업 (Warm-up)

```
첫 요청 cold-start 제거:
- 서버 시작 시 (lifespan) 모든 ML 모델을 GPU/CPU에 로드
- 더미 추론 1회 실행 (128x128 빈 이미지)
- 워밍업 완료 후에만 요청 수락 (/api/health에서 ready=true 반환)
- SKIP_WARMUP=true 환경변수로 스킵 가능 (개발 시)
```

---

## 18. 기술 선택 근거 요약

| 결정 | 선택 | 대안 | 근거 |
|------|------|------|------|
| 말풍선 검출 | comic-text-detector | paddle-detection, CRAFT | 만화 특화, 세로/가로 방향 판별, 말풍선 mask 제공 |
| OCR | manga-ocr | EasyOCR, Tesseract | 만화 세로쓰기 인식률 최고, TrOCR 기반 |
| 번역 | DeepL API | Google, Papago | JA→KO 번역 품질 최상급, 사용자 API 키 지원 |
| 텍스트 제거 | LaMa (torch 직접 추론) | OpenCV inpaint, DeepFill | 배경 패턴 복원력, 경계 자연스러움 우수. simple-lama-inpainting은 Pillow<10 충돌로 제외 |
| 텍스트 렌더링 | Pillow + FreeType | OpenCV, Cairo | CJK 완벽 지원, 안티앨리어싱, 자동 레이아웃 |
| 업스케일링 | Real-ESRGAN | waifu2x, SwinIR | 만화/애니 특화 모델 보유, GPU 가속 |
| 서버 | FastAPI | Flask, Django | 비동기 지원, WebSocket 내장, Pydantic 통합 |

---

## 19. 환경 설정 가이드

### uv 설치
```bash
# 공식 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 pip으로 설치
pip install uv
```

### 프로젝트 초기화 & 의존성 설치
```bash
# 가상환경 생성 + 의존성 설치 (uv.lock 기반)
uv sync

# 개발 의존성 포함
uv sync --group dev
```

### CUDA 130 (NVIDIA)
```bash
# pyproject.toml의 [tool.uv.index]에 cu130 인덱스 설정 후:
uv sync
```

### ROCm (AMD gfx1201)
```bash
# 환경변수 설정
export HSA_OVERRIDE_GFX_VERSION=12.0.1
export PYTORCH_ROCM_ARCH=gfx1201

# pyproject.toml의 [tool.uv.index] URL을 rocm6.3으로 변경 후:
uv sync
```

### CPU (Fallback)
```bash
# pyproject.toml의 [tool.uv.index] URL을 cpu로 변경 후:
uv sync
```

### 서버 실행
```bash
uv run python -m server.main
```

### 모델 수동 다운로드 (오프라인 환경)
```bash
uv run python -m mangalens.download
```

---

## 20. 프론트엔드 아키텍처

### 20.1 기술 선택

| 항목 | 선택 | 근거 |
|------|------|------|
| **프레임워크** | 없음 (Vanilla HTML+CSS+JS) | 빌드 단계 제거, FastAPI StaticFiles로 바로 서빙, 의존성 최소화 |
| **스타일링** | CSS3 + CSS Custom Properties | 다크모드 전환, 반응형 디자인 |
| **HTTP 클라이언트** | fetch API (ES2020+) | 브라우저 내장, 외부 의존성 불필요 |
| **실시간 통신** | WebSocket API | 진행률 스트리밍, 브라우저 내장 |
| **서빙 방식** | FastAPI `StaticFiles` 마운트 | `/static/` → `frontend/` 디렉토리 |

### 20.2 SPA 구조

```
[메인 페이지 (index.html)]
├── 헤더: 로고 + 다크모드 토글 + 설정 버튼
├── 업로드 영역
│   ├── 드래그앤드롭 존
│   ├── 파일 선택 버튼
│   ├── 단일/Bulk 모드 전환
│   └── 업로드 전 미리보기 (썸네일 그리드)
├── 진행률 패널
│   ├── 전체 진행률 바
│   ├── 이미지별 개별 상태 카드
│   └── 현재 단계 표시 (검출→업스케일→OCR→번역→제거→렌더링→합성)
├── 결과 뷰어
│   ├── 원본 ↔ 번역본 비교 (슬라이더)
│   ├── 확대/축소
│   └── 다운로드 버튼 (개별/ZIP)
├── 설정 모달
│   ├── DeepL API 키 입력
│   ├── Google API 키 입력
│   └── 서버 상태/GPU 정보 표시
└── 푸터: 시스템 정보
```

### 20.3 API 연동 흐름

```
1. 업로드 시작
   POST /api/upload (단일) 또는 POST /api/upload/bulk (Bulk)
   → task_id 수신

2. 진행률 추적
   WebSocket /ws/progress/{task_id} 연결
   → 실시간 JSON 메시지 수신
   → 프로그레스바 + 단계 표시 업데이트

3. 완료 감지
   WebSocket에서 status="completed" 수신
   → 결과 뷰어 활성화

4. 결과 다운로드
   GET /api/result/{task_id}
   → 번역된 이미지 또는 ZIP 수신

5. 설정 관리
   POST /api/settings (키 저장)
   GET /api/settings (키 조회)
   → 세션 기반, 서버 재시작 시 소멸
```

### 20.4 다크모드 전략

```css
/* CSS Custom Properties로 테마 관리 */
:root {
  --bg-primary: #ffffff;
  --text-primary: #1a1a1a;
  --accent: #4f46e5;
}

[data-theme="dark"] {
  --bg-primary: #0f0f0f;
  --text-primary: #e5e5e5;
  --accent: #818cf8;
}

/* 시스템 설정 감지 */
@media (prefers-color-scheme: dark) { ... }
```

### 20.5 반응형 브레이크포인트

| 디바이스 | 너비 | 레이아웃 |
|---------|------|---------|
| 모바일 | < 640px | 단일 컬럼, 풀위드 업로드 존 |
| 태블릿 | 640–1024px | 2컬럼 결과 뷰 |
| 데스크톱 | > 1024px | 사이드바 + 메인 콘텐츠 |
