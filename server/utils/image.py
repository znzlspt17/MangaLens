"""Image validation and storage utilities for MangaLens."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from server.config import settings
from server.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Allowed extensions
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# ---------------------------------------------------------------------------
# Magic bytes mapping
# ---------------------------------------------------------------------------

MAGIC_BYTES: list[tuple[bytes, set[str]]] = [
    (b"\xff\xd8\xff", {".jpg", ".jpeg"}),
    (b"\x89PNG\r\n\x1a\n", {".png"}),
    # WebP: starts with RIFF....WEBP
    (b"RIFF", {".webp"}),
    # BMP
    (b"BM", {".bmp"}),
    # TIFF (little-endian / big-endian)
    (b"II\x2a\x00", {".tiff"}),
    (b"MM\x00\x2a", {".tiff"}),
]


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""


def _check_extension(filename: str) -> str:
    """Return the lowercase extension if allowed, else raise."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def _check_magic_bytes(header: bytes, ext: str) -> None:
    """Verify the file header matches the claimed extension."""
    for magic, exts in MAGIC_BYTES:
        if header.startswith(magic):
            if ext in exts:
                return
            # Special case: WebP has RIFF header but needs WEBP at offset 8
            if magic == b"RIFF" and ext == ".webp":
                if len(header) >= 12 and header[8:12] == b"WEBP":
                    return
    raise ImageValidationError(
        "File content does not match its extension (magic bytes mismatch)."
    )


def _check_size(size: int) -> None:
    """Enforce max upload size."""
    if size > settings.max_upload_size:
        raise ImageValidationError(
            f"File too large ({size} bytes). "
            f"Maximum allowed: {settings.max_upload_size} bytes."
        )


def _sanitize_filename(filename: str) -> str:
    """Remove path traversal characters and dangerous patterns."""
    # Strip directory components
    name = os.path.basename(filename)
    # Remove any remaining path traversal
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    # Remove non-printable / unusual characters
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "unnamed"


async def validate_image_file(file: UploadFile) -> bytes:
    """Validate an uploaded image file and return its contents.

    Checks: extension, magic bytes, file size, Pillow verify.
    Raises ImageValidationError on any failure.
    """
    filename = file.filename or "unnamed"

    # 1. Extension check
    ext = _check_extension(filename)

    # 2. Read contents
    contents = await file.read()

    # 3. Size check
    _check_size(len(contents))

    # 4. Magic bytes check
    _check_magic_bytes(contents[:16], ext)

    # 5. Pillow integrity verification
    import io

    from PIL import Image

    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception as exc:
        raise ImageValidationError(
            f"Image integrity check failed: {exc}"
        ) from exc

    return contents


async def save_upload(file: UploadFile, task_id: str) -> Path:
    """Save an uploaded file with a UUID filename under the task directory.

    Returns the path to the saved file.
    """
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{ext}"

    task_dir = Path(settings.output_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    dest = task_dir / safe_name
    contents = await file.read()
    dest.write_bytes(contents)

    logger.info("Saved upload: %s → %s", _sanitize_filename(filename), dest)
    return dest
