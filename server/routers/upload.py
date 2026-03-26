"""Upload endpoints for MangaLens — single and bulk image upload."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import uuid
import zipfile
import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks

from server.config import settings
from server.pipeline.orchestrator import run_pipeline, UserTranslationSettings
from server.schemas.models import ErrorResponse, UploadResponse
from server.utils.image import (
    ALLOWED_EXTENSIONS,
    ImageValidationError,
    validate_image_file,
    save_upload,
)
from server.state import task_store, notify_task_changed
from server.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

# Semaphore to limit concurrent pipeline executions
_pipeline_semaphore: asyncio.Semaphore | None = None
_semaphore_lock: asyncio.Lock = asyncio.Lock()  # P1: prevents double-init race on first concurrent requests


async def _get_semaphore() -> asyncio.Semaphore:
    global _pipeline_semaphore
    # Fast path
    if _pipeline_semaphore is not None:
        return _pipeline_semaphore
    # Slow path — only one coroutine initialises the semaphore
    async with _semaphore_lock:
        if _pipeline_semaphore is None:
            limit = settings.max_concurrent_tasks
            # VRAM 기반 자동 조정 (PLAN.md §13)
            if limit == 1:  # 기본값인 경우 VRAM 기반 자동 조정
                from server.gpu import get_gpu_info
                gpu = get_gpu_info()
                if gpu.vram_mb >= 8192:
                    limit = 2
                    logger.info("VRAM %d MB >= 8GB — auto-set max_concurrent_tasks=2", gpu.vram_mb)
            _pipeline_semaphore = asyncio.Semaphore(limit)
    return _pipeline_semaphore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(
    task_id: str,
    image_paths: list[Path],
    user_settings: UserTranslationSettings | None = None,
) -> None:
    """Run the translation pipeline for the given images.

    Each image is processed independently — a single failure does not
    abort remaining images (PLAN.md §11).
    """
    sem = await _get_semaphore()
    async with sem:
        task_store[task_id]["status"] = "processing"
        task_store[task_id]["total_images"] = len(image_paths)
        task_store[task_id]["result_paths"] = []
        await notify_task_changed(task_id)

        ts = user_settings or UserTranslationSettings()
        output_base = Path(settings.output_dir) / task_id
        completed = 0
        failed = 0

        for idx, img_path in enumerate(image_paths):
            try:
                result = await run_pipeline(
                    image_path=img_path,
                    settings=ts,
                    output_dir=output_base,
                )
                task_store[task_id]["result_paths"].append(result.translated_image_path)
                completed += 1
                logger.info(
                    "Task %s — image %d/%d completed: %s",
                    task_id, idx + 1, len(image_paths), img_path.name,
                )
            except Exception:
                failed += 1
                logger.exception(
                    "Task %s — image %d/%d failed: %s",
                    task_id, idx + 1, len(image_paths), img_path.name,
                )

            task_store[task_id]["completed_images"] = completed
            task_store[task_id]["failed_images"] = failed
            task_store[task_id]["progress"] = (
                (completed + failed) / len(image_paths) * 100.0
            )
            await notify_task_changed(task_id)

        if failed == len(image_paths):
            task_store[task_id]["status"] = "failed"
        elif failed > 0:
            task_store[task_id]["status"] = "partial"
        else:
            task_store[task_id]["status"] = "completed"
        await notify_task_changed(task_id)


# ---------------------------------------------------------------------------
# POST /api/upload  — single image
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_single(
    file: Annotated[UploadFile, File(description="Manga image file")],
    background_tasks: BackgroundTasks,
) -> UploadResponse:
    """Upload a single manga image for translation."""
    # Validate image
    try:
        contents = await validate_image_file(file)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Generate task ID — datetime for human readability + short UUID suffix for uniqueness
    now = datetime.now(timezone.utc).astimezone()
    task_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3] + f"_{uuid.uuid4().hex[:6]}"

    # Reset file position and save
    file.file.seek(0)
    saved_path = await save_upload(file, task_id)

    # Initialise task state
    task_store[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "total_images": 1,
        "completed_images": 0,
        "failed_images": 0,
    }

    # Schedule pipeline in background
    background_tasks.add_task(_run_pipeline, task_id, [saved_path], UserTranslationSettings())

    return UploadResponse(task_id=task_id, status="queued")


# ---------------------------------------------------------------------------
# POST /api/upload/bulk — multiple images (multipart or ZIP)
# ---------------------------------------------------------------------------

_MAX_BULK_IMAGES = 100


@router.post(
    "/upload/bulk",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def upload_bulk(
    files: Annotated[
        list[UploadFile],
        File(description="Multiple manga image files or a single ZIP"),
    ],
    background_tasks: BackgroundTasks,
) -> UploadResponse:
    """Upload multiple manga images (multipart) or a single ZIP archive."""
    now = datetime.now(timezone.utc).astimezone()
    task_id = now.strftime("%Y%m%d_%H%M%S_%f")[:-3] + f"_{uuid.uuid4().hex[:6]}"
    image_paths: list[Path] = []

    # Detect ZIP upload (single file with .zip extension)
    if len(files) == 1 and files[0].filename and files[0].filename.lower().endswith(".zip"):
        zip_contents = await files[0].read()
        _check_bulk_size(len(zip_contents))
        try:
            zf = zipfile.ZipFile(io.BytesIO(zip_contents))
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file.")

        entries = [
            n for n in zf.namelist()
            if not n.endswith("/") and Path(n).suffix.lower() in ALLOWED_EXTENSIONS
        ]

        if len(entries) > _MAX_BULK_IMAGES:
            raise HTTPException(
                status_code=429,
                detail=f"Too many images ({len(entries)}). Maximum: {_MAX_BULK_IMAGES}.",
            )
        if not entries:
            raise HTTPException(status_code=400, detail="No valid images found in ZIP.")

        task_dir = Path(settings.output_dir) / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        for entry_name in entries:
            data = zf.read(entry_name)
            ext = Path(entry_name).suffix.lower()
            safe_name = f"{uuid.uuid4().hex}{ext}"
            dest = task_dir / safe_name
            dest.write_bytes(data)
            image_paths.append(dest)
        zf.close()
    else:
        # Multipart form-data with multiple files
        if len(files) > _MAX_BULK_IMAGES:
            raise HTTPException(
                status_code=429,
                detail=f"Too many images ({len(files)}). Maximum: {_MAX_BULK_IMAGES}.",
            )

        for f in files:
            try:
                contents = await validate_image_file(f)
            except ImageValidationError as exc:
                raise HTTPException(status_code=400, detail=f"{f.filename}: {exc}")
            f.file.seek(0)
            saved = await save_upload(f, task_id)
            image_paths.append(saved)

    if not image_paths:
        raise HTTPException(status_code=400, detail="No valid images provided.")

    # Initialise task state
    task_store[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "total_images": len(image_paths),
        "completed_images": 0,
        "failed_images": 0,
    }

    user_settings = UserTranslationSettings()
    background_tasks.add_task(_run_pipeline, task_id, image_paths, user_settings)

    return UploadResponse(task_id=task_id, status="queued")


def _check_bulk_size(size: int) -> None:
    # Allow bulk uploads up to max_upload_size * number of images cap
    limit = settings.max_upload_size * _MAX_BULK_IMAGES
    if size > limit:
        raise HTTPException(
            status_code=413,
            detail=f"Upload too large ({size} bytes).",
        )
