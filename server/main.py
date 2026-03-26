"""MangaLens — FastAPI application entry point."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.config import settings
from server.gpu import detect_gpu, get_gpu_info
from server.schemas.models import GPUInfoResponse, HealthResponse
from server.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
_ready = False


# ---------------------------------------------------------------------------
# Background: Result TTL cleanup
# ---------------------------------------------------------------------------

async def _ttl_cleanup_loop() -> None:
    """Periodically remove expired result directories (every 5 min)."""
    while True:
        await asyncio.sleep(300)  # 5 minutes
        from server.state import task_store

        output_path = Path(settings.output_dir)
        if not output_path.is_dir():
            continue

        now = time.time()
        removed = 0
        for entry in output_path.iterdir():
            if not entry.is_dir():
                continue
            try:
                mtime = entry.stat().st_mtime
                if now - mtime > settings.result_ttl_seconds:
                    task_id = entry.name
                    shutil.rmtree(entry)
                    task_store.pop(task_id, None)
                    removed += 1
            except Exception:
                logger.warning("TTL cleanup: failed to remove %s", entry, exc_info=True)
        if removed:
            logger.info("TTL cleanup: removed %d expired result(s)", removed)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready

    # 1. GPU detection
    gpu_info = detect_gpu(force_backend=settings.gpu_backend)
    logger.info(
        "GPU: %s (%s) — VRAM %d MB",
        gpu_info.gpu_name,
        gpu_info.backend,
        gpu_info.vram_mb,
    )

    # 2. Model warm-up
    if not settings.skip_warmup:
        import numpy as np

        t0 = time.time()
        device = gpu_info.device
        logger.info("Starting model warm-up on device=%s ...", device)

        # 더미 이미지 생성 (128x128 RGB) — PLAN.md §17
        _dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)

        warmup_steps = [
            ("BubbleDetector", lambda: __import__("server.pipeline.bubble_detector", fromlist=["BubbleDetector"]).BubbleDetector(device)),
            ("Preprocessor", lambda: __import__("server.pipeline.preprocessor", fromlist=["Preprocessor"]).Preprocessor(device)),
            ("OCREngine", lambda: __import__("server.pipeline.ocr_engine", fromlist=["OCREngine"]).OCREngine(device)),
            ("TextEraser", lambda: __import__("server.pipeline.text_eraser", fromlist=["TextEraser"]).TextEraser(device)),
            ("TextRenderer", lambda: __import__("server.pipeline.text_renderer", fromlist=["TextRenderer"]).TextRenderer(settings.font_dir)),
        ]
        for name, loader in warmup_steps:
            try:
                instance = loader()
                logger.info("  %s loaded", name)
                # 더미 추론 (PLAN.md §17)
                try:
                    if name == "BubbleDetector":
                        await instance.detect(_dummy_img)
                    elif name == "Preprocessor":
                        await instance.crop_and_upscale(_dummy_img, {"x": 0, "y": 0, "w": 64, "h": 64})
                    elif name == "TextEraser":
                        _dummy_mask = np.zeros((128, 128), dtype=np.uint8)
                        await instance.erase(_dummy_img, _dummy_mask)
                    # OCREngine/TextRenderer — 로드만으로 충분
                    logger.info("  %s warm-up inference done", name)
                except Exception:
                    logger.debug("  %s warm-up inference skipped (non-fatal)", name)
            except Exception:
                logger.warning("  %s failed to load (non-fatal)", name, exc_info=True)

        elapsed = time.time() - t0
        logger.info("Model warm-up finished in %.1fs", elapsed)
    else:
        logger.info("Model warm-up skipped (SKIP_WARMUP=true)")

    _ready = True

    # 3. Start TTL cleanup background task
    cleanup_task = asyncio.create_task(_ttl_cleanup_loop())

    # Ensure output directory exists
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("MangaLens server started — ready to accept requests")

    yield

    # Shutdown
    _ready = False
    cleanup_task.cancel()
    logger.info("MangaLens server shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MangaLens",
    version="0.1.0",
    description="Japanese manga image translation service",
    lifespan=lifespan,
)

# CORS middleware
_allowed_origins = settings.get_allowed_origins()
if _allowed_origins:
    if "*" in _allowed_origins:
        logger.warning(
            "ALLOWED_ORIGINS contains '*'; credentialed CORS is disabled. "
            "Set explicit origins before public deployment."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=settings.allow_cors_credentials(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
else:
    logger.info("CORS middleware disabled because ALLOWED_ORIGINS is not configured")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from server.routers import result, upload, ws  # noqa: E402

app.include_router(upload.router)
app.include_router(result.router)
app.include_router(ws.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Server health check. Returns ready=true once warm-up is complete."""
    gpu = get_gpu_info()
    return HealthResponse(
        status="ok",
        ready=_ready,
        gpu_info=GPUInfoResponse(
            backend=gpu.backend,
            device=gpu.device,
            gpu_name=gpu.gpu_name,
            vram_mb=gpu.vram_mb,
            driver_version=gpu.driver_version,
        ),
    )


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

@app.get("/api/system/gpu", response_model=GPUInfoResponse, tags=["system"])
async def system_gpu() -> GPUInfoResponse:
    """Return detected GPU environment information."""
    gpu = get_gpu_info()
    return GPUInfoResponse(
        backend=gpu.backend,
        device=gpu.device,
        gpu_name=gpu.gpu_name,
        vram_mb=gpu.vram_mb,
        driver_version=gpu.driver_version,
    )


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    logger.info("Frontend served from %s", _frontend_dir)
else:
    logger.warning("Frontend directory not found: %s — static files will not be served", _frontend_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    _host = os.environ.get("HOST", "0.0.0.0")
    _port = int(os.environ.get("PORT", "20399"))
    _reload = os.environ.get("DEV", "").lower() in ("1", "true")
    uvicorn.run(
        "server.main:app",
        host=_host,
        port=_port,
        reload=_reload,
    )
