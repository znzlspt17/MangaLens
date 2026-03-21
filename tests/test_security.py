"""Upload security tests for MangaLens (server/utils/image.py).

Tests cover (§14 of PLAN.md):
- Allowed extensions pass validation
- Disallowed extensions are rejected
- Magic bytes mismatch rejection
- File size limit enforcement
- Path traversal character sanitization
- Pillow.verify() failure rejection
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from server.utils.image import (
    ALLOWED_EXTENSIONS,
    ImageValidationError,
    _check_extension,
    _check_magic_bytes,
    _check_size,
    _sanitize_filename,
    validate_image_file,
)


# ---------------------------------------------------------------------------
# Extension checks
# ---------------------------------------------------------------------------

class TestExtensionCheck:
    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"])
    def test_allowed_extensions_pass(self, ext):
        result = _check_extension(f"test_image{ext}")
        assert result == ext

    @pytest.mark.parametrize("ext", [".exe", ".py", ".sh", ".js", ".svg", ".gif", ".pdf"])
    def test_disallowed_extensions_raise(self, ext):
        with pytest.raises(ImageValidationError, match="Unsupported file extension"):
            _check_extension(f"malicious{ext}")

    def test_case_insensitive(self):
        result = _check_extension("photo.JPG")
        assert result == ".jpg"

    def test_empty_extension_raises(self):
        with pytest.raises(ImageValidationError):
            _check_extension("noextension")


# ---------------------------------------------------------------------------
# Magic bytes checks
# ---------------------------------------------------------------------------

class TestMagicBytes:
    def test_jpeg_header_matches_jpg(self):
        header = b"\xff\xd8\xff\xe0" + b"\x00" * 12
        _check_magic_bytes(header, ".jpg")  # should not raise

    def test_jpeg_header_matches_jpeg(self):
        header = b"\xff\xd8\xff\xe0" + b"\x00" * 12
        _check_magic_bytes(header, ".jpeg")  # should not raise

    def test_png_header_matches_png(self):
        header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        _check_magic_bytes(header, ".png")  # should not raise

    def test_bmp_header_matches_bmp(self):
        header = b"BM" + b"\x00" * 14
        _check_magic_bytes(header, ".bmp")  # should not raise

    def test_tiff_le_header_matches_tiff(self):
        header = b"II\x2a\x00" + b"\x00" * 12
        _check_magic_bytes(header, ".tiff")  # should not raise

    def test_tiff_be_header_matches_tiff(self):
        header = b"MM\x00\x2a" + b"\x00" * 12
        _check_magic_bytes(header, ".tiff")  # should not raise

    def test_jpg_extension_with_png_header_raises(self):
        """Magic bytes mismatch: .jpg extension but PNG header."""
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        with pytest.raises(ImageValidationError, match="magic bytes mismatch"):
            _check_magic_bytes(png_header, ".jpg")

    def test_png_extension_with_jpeg_header_raises(self):
        """Magic bytes mismatch: .png extension but JPEG header."""
        jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 12
        with pytest.raises(ImageValidationError, match="magic bytes mismatch"):
            _check_magic_bytes(jpeg_header, ".png")

    def test_unknown_header_raises(self):
        """Completely unrecognised file header."""
        header = b"\x00\x00\x00\x00" + b"\x00" * 12
        with pytest.raises(ImageValidationError, match="magic bytes mismatch"):
            _check_magic_bytes(header, ".jpg")


# ---------------------------------------------------------------------------
# File size check
# ---------------------------------------------------------------------------

class TestSizeCheck:
    def test_within_limit_passes(self):
        _check_size(1024)  # 1 KB — should not raise

    def test_at_limit_passes(self):
        from server.config import settings
        _check_size(settings.max_upload_size)  # exactly at limit

    def test_over_limit_raises(self):
        from server.config import settings
        with pytest.raises(ImageValidationError, match="File too large"):
            _check_size(settings.max_upload_size + 1)


# ---------------------------------------------------------------------------
# Filename sanitization (path traversal prevention)
# ---------------------------------------------------------------------------

class TestFilenameSanitization:
    def test_normal_filename(self):
        result = _sanitize_filename("manga_page.jpg")
        assert result == "manga_page.jpg"

    def test_path_traversal_dotdot_slash(self):
        result = _sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_path_traversal_backslash(self):
        result = _sanitize_filename("..\\..\\windows\\system32\\cmd.exe")
        assert "\\" not in result
        assert ".." not in result

    def test_absolute_path_stripped(self):
        result = _sanitize_filename("/etc/shadow")
        assert result == "shadow"

    def test_windows_path_stripped(self):
        result = _sanitize_filename("C:\\Users\\hack\\evil.jpg")
        assert "\\" not in result
        # basename should be extracted
        assert "evil.jpg" in result or "evil_jpg" in result

    def test_empty_becomes_unnamed(self):
        result = _sanitize_filename("")
        assert result == "unnamed"

    def test_dots_only_becomes_unnamed(self):
        result = _sanitize_filename(".....")
        assert result  # should not be empty


# ---------------------------------------------------------------------------
# Full validate_image_file (async)
# ---------------------------------------------------------------------------

class TestValidateImageFile:
    @pytest.mark.asyncio
    async def test_valid_png_passes(self, sample_image_bytes):
        """Valid PNG with correct extension passes all checks."""
        mock_file = AsyncMock()
        mock_file.filename = "test.png"
        mock_file.read = AsyncMock(return_value=sample_image_bytes)

        result = await validate_image_file(mock_file)
        assert result == sample_image_bytes

    @pytest.mark.asyncio
    async def test_valid_jpeg_passes(self, sample_jpeg_bytes):
        mock_file = AsyncMock()
        mock_file.filename = "test.jpg"
        mock_file.read = AsyncMock(return_value=sample_jpeg_bytes)

        result = await validate_image_file(mock_file)
        assert result == sample_jpeg_bytes

    @pytest.mark.asyncio
    async def test_exe_extension_rejected(self):
        mock_file = AsyncMock()
        mock_file.filename = "malware.exe"
        mock_file.read = AsyncMock(return_value=b"MZ" + b"\x00" * 100)

        with pytest.raises(ImageValidationError, match="Unsupported file extension"):
            await validate_image_file(mock_file)

    @pytest.mark.asyncio
    async def test_magic_mismatch_rejected(self, sample_image_bytes):
        """PNG bytes but .jpg extension → magic bytes mismatch."""
        mock_file = AsyncMock()
        mock_file.filename = "trick.jpg"
        mock_file.read = AsyncMock(return_value=sample_image_bytes)  # PNG bytes

        with pytest.raises(ImageValidationError, match="magic bytes mismatch"):
            await validate_image_file(mock_file)

    @pytest.mark.asyncio
    async def test_oversized_file_rejected(self, sample_image_bytes):
        """File exceeding max_upload_size is rejected."""
        from server.config import settings

        big_data = b"\xff\xd8\xff\xe0" + b"\x00" * (settings.max_upload_size + 100)
        mock_file = AsyncMock()
        mock_file.filename = "big.jpg"
        mock_file.read = AsyncMock(return_value=big_data)

        with pytest.raises(ImageValidationError, match="File too large"):
            await validate_image_file(mock_file)

    @pytest.mark.asyncio
    async def test_pillow_verify_failure_rejected(self):
        """Corrupted image data fails Pillow verify."""
        # Valid JPEG header but garbage body
        corrupted = b"\xff\xd8\xff\xe0" + b"\x00\x10JFIF" + b"\xff" * 200
        mock_file = AsyncMock()
        mock_file.filename = "corrupt.jpg"
        mock_file.read = AsyncMock(return_value=corrupted)

        with pytest.raises(ImageValidationError, match="integrity check failed"):
            await validate_image_file(mock_file)

    @pytest.mark.asyncio
    async def test_no_filename_defaults_unnamed(self, sample_image_bytes):
        """File with no filename uses 'unnamed' and fails extension check."""
        mock_file = AsyncMock()
        mock_file.filename = None
        mock_file.read = AsyncMock(return_value=sample_image_bytes)

        # 'unnamed' has no extension → fails extension check
        with pytest.raises(ImageValidationError):
            await validate_image_file(mock_file)
