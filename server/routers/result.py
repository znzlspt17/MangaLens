"""Result and task status endpoints for MangaLens."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from server.config import settings
from server.schemas.models import ErrorResponse, TaskStatus, TranslationResult
from server.state import task_store
from server.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["result"])

_TASK_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9]{3}_[0-9a-f]{6}$")


def _validate_task_id(task_id: str) -> None:
    """Reject task_id values that don't match the expected format."""
    if not _TASK_ID_RE.match(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID format.")


# ---------------------------------------------------------------------------
# GET /api/status/{task_id}
# ---------------------------------------------------------------------------

@router.get(
    "/status/{task_id}",
    response_model=TaskStatus,
    responses={404: {"model": ErrorResponse}},
)
async def get_task_status(task_id: str) -> TaskStatus:
    """Query the status of a translation task."""
    _validate_task_id(task_id)
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found.")

    info = task_store[task_id]
    return TaskStatus(
        task_id=task_id,
        status=info["status"],
        progress=info.get("progress", 0.0),
        total_images=info.get("total_images", 0),
        completed_images=info.get("completed_images", 0),
        failed_images=info.get("failed_images", 0),
    )


# ---------------------------------------------------------------------------
# GET /api/result/{task_id}
# ---------------------------------------------------------------------------

@router.get(
    "/result/{task_id}",
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def get_result(task_id: str) -> FileResponse:
    """Download the translated result for a completed task.

    Returns a single image or a ZIP for bulk tasks.
    """
    _validate_task_id(task_id)
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found.")

    info = task_store[task_id]
    if info["status"] not in ("completed", "partial"):
        raise HTTPException(
            status_code=409,
            detail=f"Task is not completed (current: {info['status']}).",
        )

    task_dir = Path(settings.output_dir) / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Result files not found.")

    # Collect result image files (pipeline outputs *_translated.png)
    result_files = sorted(task_dir.glob("*_translated.*"))

    if not result_files:
        # Fallback: any image in the directory excluding originals
        result_files = sorted(
            p for p in task_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and ("_translated" in p.stem or "_result" in p.stem)
        )

    if not result_files:
        raise HTTPException(status_code=404, detail="No result images found.")

    # Determine media type from file extension
    _MEDIA_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }

    # Single file → direct download
    if len(result_files) == 1:
        media = _MEDIA_TYPES.get(result_files[0].suffix.lower(), "application/octet-stream")
        if settings.delete_after_download:
            # Read file into memory first to avoid race with rmtree
            import io
            from fastapi.responses import StreamingResponse

            data = result_files[0].read_bytes()
            response = StreamingResponse(
                io.BytesIO(data),
                media_type=media,
                headers={
                    "Content-Disposition": f'inline; filename="{result_files[0].name}"',
                },
            )
        else:
            response = FileResponse(
                path=str(result_files[0]),
                filename=result_files[0].name,
                media_type=media,
            )
    else:
        # Multiple files → ZIP
        import io
        import zipfile

        from fastapi.responses import StreamingResponse

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in result_files:
                zf.write(f, arcname=f.name)
            # Include translation log if present
            log_path = task_dir / "translation_log.json"
            if log_path.exists():
                zf.write(log_path, arcname="translation_log.json")

        buf.seek(0)
        response = StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{task_id}.zip"',
            },
        )

    # Optionally delete after download
    if settings.delete_after_download:
        shutil.rmtree(task_dir, ignore_errors=True)
        task_store.pop(task_id, None)
        logger.info("Deleted result for task %s after download", task_id)

    return response
