# MangaLens

> 일본어 만화 이미지를 AI로 자동 번역하는 웹 서비스

만화 페이지 이미지를 업로드하면 말풍선을 자동으로 감지하고, 일본어 텍스트를 OCR → 번역 → 텍스트 재렌더링하여 번역된 이미지를 출력합니다.

## 개발 기간

2026년 3월 24일 ~ 현재 (진행 중)

---

## 사용 라이브러리

### 핵심 프레임워크

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **FastAPI** | ≥0.115 | 비동기 웹 서버 (REST API + WebSocket) |
| **uvicorn** | ≥0.32 | ASGI 서버 |
| **PyTorch** | ≥2.6 | ML 모델 추론 프레임워크 |
| **torchvision** | ≥0.21 | 이미지 전처리 유틸리티 |

### ML / 이미지 처리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **manga-ocr** | ≥0.1.14 | 일본어 만화 OCR (TrOCR 기반) |
| **ultralytics** | ≥8.0 | YOLOv5 백본 (말풍선 검출) |
| **realesrgan** | ≥0.3.0 | 이미지 업스케일링 (OCR 전처리) |
| **basicsr** | 1.4.2 | Real-ESRGAN 의존성 (vendor 패치) |
| **Pillow** | ≥11.0 | CJK 텍스트 렌더링 + 이미지 처리 |
| **opencv-python-headless** | ≥4.10 | 이미지 I/O, 마스크 처리, fallback 인페인팅 |
| **numpy** | ≥1.26 | 행렬 연산 |

### 네트워크 / 유틸리티

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **httpx** | ≥0.28 | 비동기 HTTP 클라이언트 (모델/폰트 다운로드) |
| **pydantic** | ≥2.10 | 요청/응답 데이터 검증 |
| **pydantic-settings** | ≥2.7 | 환경변수 설정 관리 |
| **websockets** | ≥14.0 | 실시간 진행률 전송 |
| **aiofiles** | ≥24.1 | 비동기 파일 I/O |
| **python-multipart** | ≥0.0.18 | 파일 업로드 처리 |

### 프론트엔드

- **Vanilla HTML5 + CSS3 + JavaScript (ES2020+)** — 프레임워크/번들러 없음
- CSS 변수 기반 다크/라이트 테마, CSS Grid/Flexbox 반응형

---

## 사용 모델

| 단계 | 모델 | 아키텍처 | 크기 | 역할 |
|------|------|---------|------|------|
| ① 말풍선 검출 | **comic-text-detector** | YOLOv5s | ~12MB | 말풍선/나레이션 bbox + 텍스트 마스크 + 방향 판별 |
| ② 이미지 업스케일 | **Real-ESRGAN** | RRDBNet (ESRGAN) | ~17~64MB | 저해상도 크롭 업스케일 (OCR 정확도 향상) — `anime_6B` 기본 |
| ③ OCR | **manga-ocr** | TrOCR (ViT + GPT Decoder) | ~450MB | 일본어 세로쓰기 텍스트 인식 |
| ⑤ 텍스트 제거 | **LaMa** | FFC + ResNet | ~200MB | 말풍선 내 텍스트 인페인팅 (배경 복원) |
| ④ 번역 | **Hunyuan-MT-7B** | CausalLM (7B파라미터) | ~14GB VRAM | JA → KO 로컬 인퍼런스 (WMT25 1위) |

> 4개 로컬 모델 합계 약 **726MB** VRAM + Hunyuan-MT-7B ~14GB VRAM. 첫 요청 시 1회 로드 후 글로벌 캐시에 유지됩니다.

---

## 업데이트 사항 — 모델 선택 근거

### ① 말풍선 검출: comic-text-detector (YOLOv5s)

**이슈**: 만화 페이지에서 텍스트를 찾아야 하는데, 일반 텍스트 검출기(EAST, CRAFT)는 효과음·배경 텍스트까지 모두 잡아 오인식이 심했음

**해결**: 만화 특화 모델인 comic-text-detector를 선택.
- 말풍선/나레이션 박스를 우선 검출하고 효과음(`effect`)은 필터링
- **세로쓰기 방향 판별**이 내장되어 있어 후속 OCR에 방향 정보 전달 가능
- ultralytics YOLOv5 백본으로 NMS 후처리까지 표준화

### ② 업스케일: Real-ESRGAN (RRDBNet)

**이슈**: 저해상도 스캔본에서 크롭한 말풍선이 너무 작아(50×80px 등) OCR 인식률이 급격히 떨어짐

**해결**: Real-ESRGAN으로 크롭 이미지를 x2/x4 업스케일 후 OCR에 전달.
- 작은 크롭은 x4, 큰 크롭은 x2 자동 판별
- **x4 기본 변형**: `anime_6B` (만화/애니 특화 6블록 경량 모델) — `UPSCALER_VARIANT` 환경변수로 `x4plus` 전환 가능
- 업스케일 직후 **후리가나 제거** (connected component 높이 필터링 + 탁음/반탁음 열 보존) — OCR 정확도 향상
- **업스케일은 OCR 전처리 전용** — 최종 출력은 원본 해상도 유지

### ③ OCR: manga-ocr (TrOCR)

**이슈**: Tesseract/EasyOCR 등 범용 OCR은 만화 세로쓰기(縦書き) 텍스트 인식률이 매우 낮음 (특히 손글씨 스타일 폰트)

**해결**: manga-ocr (TrOCR 기반, ViT Encoder + GPT Decoder)를 사용.
- 일본 만화 세로쓰기에 특화 훈련된 Vision-Language 모델
- 신뢰도 < 0.3이면 번역을 건너뛰고 원문 유지 (오역 방지)

### ④ 번역: Hunyuan-MT-7B (로컬 LLM)

**이슈**: 로컬 번역 모델(Helsinki-NLP 등)은 만화 대사체/구어체 번역 품질이 낙어, 대화 문맥을 유지하기 어려움. 외부 API(DeepL/Google)는 키 관리 부담과 운영 비용 발생.

**해결**: WMT25 다국어 번역 경진대회 30/31 언어쌍 1위를 차지한 **tencent/Hunyuan-MT-7B** 로컬 인퍼런스 채택.
- 외부 API 키 없이 온프레미스 동작
- `asyncio.to_thread()`로 비동기 실행, GPU fp16 추론
- 실패 시 원문 그대로 반환 (fallback)

### ⑤ 텍스트 제거: LaMa (Large Mask Inpainting)

**이슈**: OpenCV `cv2.inpaint()`(Telea/NS)는 단색 배경에서만 작동하고, 만화의 스크린톤·그라데이션 배경에서는 뭉개짐이 심했음. `simple-lama-inpainting` PyPI 래퍼는 `Pillow<10` 요구하여 최신 Pillow와 충돌.

**해결**: LaMa TorchScript 모델(`big-lama.pt`)을 직접 로드하여 추론 구현.
- comic-text-detector의 텍스트 마스크를 그대로 활용 → 정밀한 텍스트만 제거
- 배경 패턴(스크린톤, 선, 그라데이션)까지 자연스럽게 복원
- 모델 없을 시 OpenCV Telea로 graceful degradation

### ⑥ 텍스트 렌더링: Pillow + FreeType

**이슈**: OpenCV `cv2.putText()`는 CJK 유니코드를 렌더링하지 못하고, 안티앨리어싱/줄바꿈 불가

**해결**: Pillow + FreeType 기반 CJK 렌더링 엔진 직접 구현.
- Noto Sans KR 폰트로 한국어 완벽 지원
- 말풍선 bbox 내부에서 이진탐색으로 최적 폰트 크기 자동 계산
- 텍스트 외곽선(stroke) + 자동 줄바꿈
- 세로쓰기 → 가로쓰기 레이아웃 자동 변환

### 오케스트레이션 최적화: 4+5 병렬 실행

**이슈**: Stage 4(번역)는 GPU 코어를 오래 점유하고 Stage 5(텍스트 제거)는 개별적으로 실행 가능

**해결**: Stage 4와 Stage 5는 서로 독립적이므로 `asyncio.gather`로 **동시 실행**.

```
          ┌── Stage 4: 번역 (Hunyuan-MT-7B, GPU)
Stage 3 ──┤                               ──→ Stage 6
          └── Stage 5: 텍스트 제거 (LaMa, GPU)
```

---

## Quick Start

### 1. 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv 설치
uv sync --group dev                                 # 의존성 설치
```

### 2. 모델 & 폰트 다운로드

```bash
uv run python -m server.download
```

### 3. 서버 실행

```bash
./start.sh        # Linux/macOS (모델 다운로드 + 서버 시작)
start.bat          # Windows
# 또는: uv run python -m server.main
```

### 4. 사용

1. 브라우저에서 `http://localhost:20399` 접속
2. 만화 이미지 드래그 앤 드롭
3. 실시간 진행률 확인 → 원본/번역 비교 슬라이더 → 다운로드

> **주의**: Hunyuan-MT-7B 모델은 첫 요청 시 HuggingFace Hub에서 ~15GB 자동 다운로드됩니다.

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/health` | 서버 상태 확인 |
| POST | `/api/upload` | 단일 이미지 업로드 & 번역 |
| POST | `/api/upload/bulk` | 다수 이미지 Bulk 업로드 |
| GET | `/api/status/{task_id}` | 작업 진행 상태 조회 |
| GET | `/api/result/{task_id}` | 번역 결과 다운로드 |
| GET | `/api/system/gpu` | GPU 환경 정보 |
| WS | `/ws/progress/{task_id}` | 실시간 진행률 알림 |

## 프로젝트 구조

```
server/
├── main.py              # FastAPI 앱 진입점
├── config.py            # 환경변수 관리
├── gpu.py               # GPU 자동 감지 (CUDA/ROCm/CPU)
├── download.py          # 모델/폰트 자동 다운로드
├── state.py             # 공유 태스크 상태
├── routers/
│   ├── upload.py        # 업로드 API
│   ├── result.py        # 결과 다운로드 API
│   └── ws.py            # WebSocket 진행률
├── pipeline/
│   ├── orchestrator.py  # 7단계 파이프라인 오케스트레이션
│   ├── bubble_detector.py  # ① 말풍선 검출 (YOLOv5s)
│   ├── preprocessor.py     # ② 크롭 & 업스케일 (Real-ESRGAN)
│   ├── ocr_engine.py       # ③ OCR (manga-ocr)
│   ├── translator.py       # ④ 번역 (Hunyuan-MT-7B 로컬 LLM)
│   ├── text_eraser.py      # ⑤ 텍스트 제거 (LaMa)
│   ├── text_renderer.py    # ⑥ 텍스트 렌더링 (Pillow)
│   └── compositor.py       # ⑦ 알파 블렌딩 합성
├── schemas/
│   └── models.py        # Pydantic 모델
└── utils/
    ├── image.py         # 이미지 보안 검증 (4단계)
    ├── reading_order.py # 우→좌 읽기 순서 정렬
    └── logger.py        # 로깅 설정

frontend/                # Vanilla HTML+CSS+JS SPA
├── index.html
├── css/style.css, themes.css
├── js/app.js, api.js, upload.js, progress.js, result.js, settings.js
└── assets/icons/

tests/                   # pytest 테스트 (187 케이스)
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GPU_BACKEND` | `auto` | GPU 백엔드 (auto/cuda/rocm/cpu) |
| `MAX_UPLOAD_SIZE` | `52428800` | 최대 업로드 크기 (50MB) |
| `MAX_CONCURRENT_TASKS` | `1` | 동시 파이프라인 수 |
| `RESULT_TTL_SECONDS` | `3600` | 결과 보존 시간 (1시간) |
| `SKIP_WARMUP` | `false` | 모델 워밍업 건너뛰기 |
| `ALLOWED_ORIGINS` | (비어 있음) | CORS 허용 출처 |
| `USE_MAGI_DETECTOR` | `false` | Magi v2 감지기 활성화 |
| `UPSCALER_VARIANT` | `anime_6b` | 업스케일러 변형 선택 (`anime_6b` / `x4plus`) |

## 라이선스

이 프로젝트는 개인 사용 목적입니다.
