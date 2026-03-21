"""Shared test fixtures for MangaLens test suite."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Create a minimal 100x100 white PNG image as bytes."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_jpeg_bytes() -> bytes:
    """Create a minimal 100x100 white JPEG image as bytes."""
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def mock_gpu():
    """Mock GPU detection to return CPU fallback."""
    from server.gpu import GPUInfo

    cpu_info = GPUInfo(
        backend="cpu",
        device="cpu",
        gpu_name="CPU",
        vram_mb=0,
        driver_version="",
    )
    with patch("server.gpu._cached_info", cpu_info), \
         patch("server.gpu.detect_gpu", return_value=cpu_info), \
         patch("server.main.detect_gpu", return_value=cpu_info), \
         patch("server.main.get_gpu_info", return_value=cpu_info):
        yield cpu_info


@pytest.fixture()
def app(mock_gpu):
    """Provide a FastAPI app instance with GPU mocked."""
    from server.main import app as _app
    return _app


@pytest.fixture()
async def client(app):
    """Provide an httpx.AsyncClient bound to the FastAPI app."""
    import httpx
    from httpx._transports.asgi import ASGITransport

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
