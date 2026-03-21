"""API endpoint tests for MangaLens (upload, health, gpu).

Tests cover:
- POST /api/upload — single valid image → 200 + task_id
- POST /api/upload — invalid file → 400
- POST /api/upload/bulk — multiple images → 200
- GET  /api/health → 200 + ready state
- GET  /api/system/gpu → 200 + GPU info

Uses httpx.AsyncClient + FastAPI ASGITransport.
"""

from __future__ import annotations

import io

import pytest

from server.state import task_store


# ---------------------------------------------------------------------------
# Helper to clear task_store between tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_task_store():
    task_store.clear()
    yield
    task_store.clear()


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_has_status_ok(self, client):
        resp = await client.get("/api/health")
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_has_gpu_info(self, client):
        resp = await client.get("/api/health")
        data = resp.json()
        assert "gpu_info" in data
        gpu = data["gpu_info"]
        assert "backend" in gpu
        assert "device" in gpu
        assert "gpu_name" in gpu
        assert "vram_mb" in gpu


# ---------------------------------------------------------------------------
# GET /api/system/gpu
# ---------------------------------------------------------------------------

class TestGPUEndpoint:
    @pytest.mark.asyncio
    async def test_gpu_returns_200(self, client):
        resp = await client.get("/api/system/gpu")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_gpu_response_fields(self, client):
        resp = await client.get("/api/system/gpu")
        data = resp.json()
        assert data["backend"] in ("cuda", "rocm", "cpu")
        assert data["device"] in ("cuda", "cpu")
        assert isinstance(data["vram_mb"], int)
        assert isinstance(data["gpu_name"], str)
        assert isinstance(data["driver_version"], str)


# ---------------------------------------------------------------------------
# POST /api/upload — single image
# ---------------------------------------------------------------------------

class TestSingleUpload:
    @pytest.mark.asyncio
    async def test_valid_png_upload(self, client, sample_image_bytes):
        resp = await client.post(
            "/api/upload",
            files={"file": ("test.png", io.BytesIO(sample_image_bytes), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_valid_jpeg_upload(self, client, sample_jpeg_bytes):
        resp = await client.post(
            "/api/upload",
            files={"file": ("photo.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data

    @pytest.mark.asyncio
    async def test_invalid_extension_returns_400(self, client):
        resp = await client.post(
            "/api/upload",
            files={"file": ("evil.exe", io.BytesIO(b"MZ" + b"\x00" * 100), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported file extension" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_magic_mismatch_returns_400(self, client, sample_image_bytes):
        """PNG data uploaded with .jpg extension → magic bytes mismatch."""
        resp = await client.post(
            "/api/upload",
            files={"file": ("trick.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "magic bytes" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_task_appears_in_store(self, client, sample_image_bytes):
        resp = await client.post(
            "/api/upload",
            files={"file": ("page.png", io.BytesIO(sample_image_bytes), "image/png")},
        )
        data = resp.json()
        assert data["task_id"] in task_store


# ---------------------------------------------------------------------------
# POST /api/upload/bulk — multiple images
# ---------------------------------------------------------------------------

class TestBulkUpload:
    @pytest.mark.asyncio
    async def test_bulk_two_images(self, client, sample_image_bytes, sample_jpeg_bytes):
        resp = await client.post(
            "/api/upload/bulk",
            files=[
                ("files", ("page1.png", io.BytesIO(sample_image_bytes), "image/png")),
                ("files", ("page2.jpg", io.BytesIO(sample_jpeg_bytes), "image/jpeg")),
            ],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "queued"

    @pytest.mark.asyncio
    async def test_bulk_one_invalid_returns_400(self, client, sample_image_bytes):
        resp = await client.post(
            "/api/upload/bulk",
            files=[
                ("files", ("page1.png", io.BytesIO(sample_image_bytes), "image/png")),
                ("files", ("evil.exe", io.BytesIO(b"MZ" + b"\x00" * 100), "application/octet-stream")),
            ],
        )
        assert resp.status_code == 400
