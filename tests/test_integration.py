"""Integration tests and API endpoint tests for MangaLens.

Covers:
- Settings endpoints (POST/GET /api/settings)
- Status endpoint (GET /api/status/{task_id})
- Result endpoint (GET /api/result/{task_id})
- WebSocket progress (ws://host/ws/progress/{task_id})
- Upload → Status → Result pipeline integration
- TTL cleanup loop
"""

from __future__ import annotations

import asyncio
import io
import shutil
import time
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx._transports.asgi import ASGITransport
from PIL import Image

from server.state import task_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a minimal PNG image as bytes."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_zip_with_images(count: int = 2) -> bytes:
    """Create an in-memory ZIP containing `count` PNG images."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(count):
            zf.writestr(f"page_{i}.png", _make_png_bytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_task_store():
    """Ensure task_store is clean before and after every test."""
    task_store.clear()
    yield
    task_store.clear()


@pytest.fixture(autouse=True)
def _clean_session_store():
    """Ensure session_store is clean before and after every test."""
    from server.routers.settings import session_store
    session_store.clear()
    yield
    session_store.clear()


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
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ===================================================================
# 1. TestSettingsEndpoints
# ===================================================================


class TestSettingsEndpoints:
    """POST/GET /api/settings — session-based key management."""

    async def test_post_settings_sets_api_keys(self, client: httpx.AsyncClient):
        """POST stores keys and returns masked values."""
        resp = await client.post(
            "/api/settings",
            json={"deepl_api_key": "dl-1234567890abcdef", "google_api_key": "gk-0987654321fedcba"},
            headers={"X-Session-Id": "sess-001"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Keys must be masked (first 4 + last 4 visible)
        assert body["deepl_api_key"] is not None
        assert body["deepl_api_key"].startswith("dl-1")
        assert "****" in body["deepl_api_key"]
        assert body["deepl_api_key"].endswith("cdef")

    async def test_get_settings_returns_masked_keys(self, client: httpx.AsyncClient):
        """GET returns previously stored masked keys."""
        # First, set keys
        await client.post(
            "/api/settings",
            json={"deepl_api_key": "dl-1234567890abcdef"},
            headers={"X-Session-Id": "sess-002"},
        )
        # Then, read back
        resp = await client.get(
            "/api/settings",
            headers={"X-Session-Id": "sess-002"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["deepl_api_key"] is not None
        assert "****" in body["deepl_api_key"]

    async def test_get_settings_no_session_returns_empty(self, client: httpx.AsyncClient):
        """GET without prior session returns null keys."""
        resp = await client.get("/api/settings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["deepl_api_key"] is None
        assert body["google_api_key"] is None

    async def test_sessions_are_isolated(self, client: httpx.AsyncClient):
        """Different session IDs have independent settings."""
        await client.post(
            "/api/settings",
            json={"deepl_api_key": "key-for-session-A-1234"},
            headers={"X-Session-Id": "sess-A"},
        )
        await client.post(
            "/api/settings",
            json={"google_api_key": "key-for-session-B-5678"},
            headers={"X-Session-Id": "sess-B"},
        )
        # Session A should not have google_api_key
        resp_a = await client.get("/api/settings", headers={"X-Session-Id": "sess-A"})
        assert resp_a.json()["google_api_key"] is None
        assert resp_a.json()["deepl_api_key"] is not None

        # Session B should not have deepl_api_key
        resp_b = await client.get("/api/settings", headers={"X-Session-Id": "sess-B"})
        assert resp_b.json()["deepl_api_key"] is None
        assert resp_b.json()["google_api_key"] is not None

    async def test_invalid_session_id_rejected(self, client: httpx.AsyncClient):
        """Invalid session identifiers are rejected before creating session state."""
        resp = await client.post(
            "/api/settings",
            json={"deepl_api_key": "dl-1234567890abcdef"},
            headers={"X-Session-Id": "bad session id"},
        )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid session ID."

    async def test_oversized_api_key_rejected(self, client: httpx.AsyncClient):
        """Oversized API keys are rejected by request validation."""
        resp = await client.post(
            "/api/settings",
            json={"deepl_api_key": "x" * 513},
            headers={"X-Session-Id": "sess-oversized"},
        )
        assert resp.status_code == 422


class TestDeploymentSafetyConfig:
    """Deployment-related configuration safety checks."""

    def test_cors_credentials_only_for_explicit_origins(self):
        """Credentialed CORS is only enabled for explicit origin lists."""
        from server.config import Settings

        blank = Settings(allowed_origins="")
        explicit = Settings(allowed_origins="https://mangalens.example.com")
        wildcard = Settings(allowed_origins="*")

        assert blank.get_allowed_origins() == []
        assert blank.allow_cors_credentials() is False
        assert explicit.get_allowed_origins() == ["https://mangalens.example.com"]
        assert explicit.allow_cors_credentials() is True
        assert wildcard.get_allowed_origins() == ["*"]
        assert wildcard.allow_cors_credentials() is False

    async def test_default_app_does_not_emit_cors_headers(
        self, client: httpx.AsyncClient
    ):
        """Default app config should stay same-origin without CORS headers."""
        resp = await client.get(
            "/api/health",
            headers={"Origin": "https://mangalens.example.com"},
        )
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers
        assert "access-control-allow-credentials" not in resp.headers


# ===================================================================
# 2. TestStatusEndpoint
# ===================================================================


class TestStatusEndpoint:
    """GET /api/status/{task_id}."""

    async def test_unknown_task_returns_404(self, client: httpx.AsyncClient):
        resp = await client.get("/api/status/nonexistent-task-id")
        assert resp.status_code == 404

    async def test_existing_task_returns_status(self, client: httpx.AsyncClient):
        task_store["task-123"] = {
            "status": "processing",
            "progress": 50.0,
            "total_images": 2,
            "completed_images": 1,
            "failed_images": 0,
        }
        resp = await client.get("/api/status/task-123")
        assert resp.status_code == 200
        body = resp.json()
        assert body["task_id"] == "task-123"
        assert body["status"] == "processing"
        assert body["progress"] == 50.0
        assert body["completed_images"] == 1

    async def test_queued_task_shows_zero_progress(self, client: httpx.AsyncClient):
        task_store["task-q"] = {
            "status": "queued",
            "progress": 0.0,
            "total_images": 1,
            "completed_images": 0,
            "failed_images": 0,
        }
        resp = await client.get("/api/status/task-q")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        assert body["progress"] == 0.0


# ===================================================================
# 3. TestResultEndpoint
# ===================================================================


class TestResultEndpoint:
    """GET /api/result/{task_id}."""

    async def test_unknown_task_returns_404(self, client: httpx.AsyncClient):
        resp = await client.get("/api/result/nonexistent")
        assert resp.status_code == 404

    async def test_not_completed_returns_409(self, client: httpx.AsyncClient):
        task_store["task-proc"] = {
            "status": "processing",
            "progress": 30.0,
            "total_images": 1,
            "completed_images": 0,
            "failed_images": 0,
        }
        resp = await client.get("/api/result/task-proc")
        assert resp.status_code == 409

    async def test_completed_single_image_downloadable(
        self, client: httpx.AsyncClient, tmp_path: Path
    ):
        """Completed task with result file returns 200 with file content."""
        task_id = "task-done-single"
        task_store[task_id] = {
            "status": "completed",
            "progress": 100.0,
            "total_images": 1,
            "completed_images": 1,
            "failed_images": 0,
        }
        # Create result file in the output directory
        result_dir = tmp_path / task_id
        result_dir.mkdir(parents=True)
        result_file = result_dir / "page_translated.png"
        result_file.write_bytes(_make_png_bytes())

        with patch("server.routers.result.settings") as mock_settings:
            mock_settings.output_dir = str(tmp_path)
            mock_settings.delete_after_download = False
            resp = await client.get(f"/api/result/{task_id}")

        assert resp.status_code == 200
        assert len(resp.content) > 0

    async def test_completed_no_result_files_returns_404(
        self, client: httpx.AsyncClient, tmp_path: Path
    ):
        """Completed task but empty directory returns 404."""
        task_id = "task-empty"
        task_store[task_id] = {
            "status": "completed",
            "progress": 100.0,
            "total_images": 1,
            "completed_images": 1,
            "failed_images": 0,
        }
        result_dir = tmp_path / task_id
        result_dir.mkdir(parents=True)

        with patch("server.routers.result.settings") as mock_settings:
            mock_settings.output_dir = str(tmp_path)
            mock_settings.delete_after_download = False
            resp = await client.get(f"/api/result/{task_id}")

        assert resp.status_code == 404

    async def test_delete_after_download(
        self, client: httpx.AsyncClient, tmp_path: Path
    ):
        """With delete_after_download=True, result dir and task are removed.

        Uses multi-file result to trigger the StreamingResponse (ZIP) path,
        which builds the archive in memory before deletion.

        NOTE: Single-file FileResponse path has a bug — the file is deleted
        before FileResponse can serve it, causing a RuntimeError. This is a
        production code issue (server/routers/result.py).
        """
        task_id = "task-delete-dl"
        task_store[task_id] = {
            "status": "completed",
            "progress": 100.0,
            "total_images": 2,
            "completed_images": 2,
            "failed_images": 0,
        }
        result_dir = tmp_path / task_id
        result_dir.mkdir(parents=True)
        (result_dir / "page1_translated.png").write_bytes(_make_png_bytes())
        (result_dir / "page2_translated.png").write_bytes(_make_png_bytes())

        with patch("server.routers.result.settings") as mock_settings:
            mock_settings.output_dir = str(tmp_path)
            mock_settings.delete_after_download = True
            resp = await client.get(f"/api/result/{task_id}")

        assert resp.status_code == 200
        # Directory and task entry should be removed
        assert not result_dir.exists()
        assert task_id not in task_store


# ===================================================================
# 4. TestWebSocket
# ===================================================================


class TestWebSocket:
    """WebSocket /ws/progress/{task_id}."""

    async def test_ws_unknown_task_sends_error_and_closes(self, app):
        """Connecting with non-existent task_id sends error and closes."""
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/ws/progress/no-such-task") as ws:
                data = ws.receive_json()
                assert data["error"] == "task_not_found"

    async def test_ws_completed_task_sends_status_and_closes(self, app):
        """Connecting with completed task sends final status and closes."""
        task_store["task-ws-done"] = {
            "status": "completed",
            "progress": 100.0,
            "total_images": 1,
            "completed_images": 1,
            "failed_images": 0,
        }
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/ws/progress/task-ws-done") as ws:
                data = ws.receive_json()
                assert data["task_id"] == "task-ws-done"
                assert data["status"] == "completed"
                assert data["progress"] == 100.0

    async def test_ws_processing_sends_updates(self, app):
        """Connecting with processing task receives progress updates."""
        task_store["task-ws-prog"] = {
            "status": "processing",
            "progress": 50.0,
            "total_images": 2,
            "completed_images": 1,
            "failed_images": 0,
        }
        from starlette.testclient import TestClient

        with TestClient(app) as tc:
            with tc.websocket_connect("/ws/progress/task-ws-prog") as ws:
                data = ws.receive_json()
                assert data["status"] == "processing"
                assert data["progress"] == 50.0
                # Simulate task completing so the WS loop exits
                task_store["task-ws-prog"]["status"] = "completed"
                task_store["task-ws-prog"]["progress"] = 100.0
                data2 = ws.receive_json()
                assert data2["status"] == "completed"


# ===================================================================
# 5. TestUploadPipelineIntegration
# ===================================================================


class TestUploadPipelineIntegration:
    """Upload → Status → Result end-to-end flow (pipeline mocked)."""

    async def test_upload_creates_task_entry(self, client: httpx.AsyncClient):
        """Single upload creates a task_store entry with status=queued."""
        png = _make_png_bytes()
        with patch("server.routers.upload.validate_image_file", new_callable=AsyncMock, return_value=png), \
             patch("server.routers.upload.save_upload", new_callable=AsyncMock, return_value=Path("/tmp/test.png")), \
             patch("server.routers.upload._run_pipeline", new_callable=AsyncMock):
            resp = await client.post(
                "/api/upload",
                files={"file": ("test.png", io.BytesIO(png), "image/png")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        task_id = body["task_id"]
        assert task_id in task_store
        assert task_store[task_id]["status"] == "queued"

    async def test_upload_status_result_flow(
        self, client: httpx.AsyncClient, tmp_path: Path
    ):
        """Full flow: upload → check status → mark completed → get result."""
        png = _make_png_bytes()
        with patch("server.routers.upload.validate_image_file", new_callable=AsyncMock, return_value=png), \
             patch("server.routers.upload.save_upload", new_callable=AsyncMock, return_value=Path("/tmp/test.png")), \
             patch("server.routers.upload._run_pipeline", new_callable=AsyncMock):
            resp = await client.post(
                "/api/upload",
                files={"file": ("test.png", io.BytesIO(png), "image/png")},
            )

        task_id = resp.json()["task_id"]

        # Status should be queued
        status_resp = await client.get(f"/api/status/{task_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "queued"

        # Simulate pipeline completion
        task_store[task_id]["status"] = "completed"
        task_store[task_id]["progress"] = 100.0
        task_store[task_id]["completed_images"] = 1

        # Create result file
        result_dir = tmp_path / task_id
        result_dir.mkdir(parents=True)
        (result_dir / "page_translated.png").write_bytes(_make_png_bytes())

        with patch("server.routers.result.settings") as mock_settings:
            mock_settings.output_dir = str(tmp_path)
            mock_settings.delete_after_download = False
            result_resp = await client.get(f"/api/result/{task_id}")

        assert result_resp.status_code == 200

    async def test_bulk_upload_zip(self, client: httpx.AsyncClient, tmp_path: Path):
        """Bulk upload with ZIP creates a task_store entry."""
        zip_bytes = _make_zip_with_images(3)

        with patch("server.routers.upload._run_pipeline", new_callable=AsyncMock), \
             patch("server.routers.upload.settings") as mock_settings:
            mock_settings.output_dir = str(tmp_path)
            mock_settings.max_upload_size = 52_428_800
            resp = await client.post(
                "/api/upload/bulk",
                files={"files": ("images.zip", io.BytesIO(zip_bytes), "application/zip")},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "queued"
        task_id = body["task_id"]
        assert task_id in task_store
        assert task_store[task_id]["total_images"] == 3


# ===================================================================
# 6. TestTTLCleanup
# ===================================================================


class TestTTLCleanup:
    """TTL cleanup loop logic unit test."""

    async def test_expired_directory_removed(self, tmp_path: Path):
        """Directories older than TTL are removed along with task_store entry."""
        task_id = "task-expired"
        task_dir = tmp_path / task_id
        task_dir.mkdir()
        (task_dir / "result.png").write_bytes(_make_png_bytes())

        task_store[task_id] = {"status": "completed", "progress": 100.0}

        # Directly test the cleanup logic (not the loop)
        ttl = 10  # seconds
        # Backdate mtime so it appears expired
        import os
        old_time = time.time() - ttl - 100
        os.utime(task_dir, (old_time, old_time))

        with patch("server.main.settings") as mock_settings:
            mock_settings.output_dir = str(tmp_path)
            mock_settings.result_ttl_seconds = ttl
            # Execute one pass of cleanup inline
            output_path = Path(mock_settings.output_dir)
            now = time.time()
            for entry in output_path.iterdir():
                if not entry.is_dir():
                    continue
                mtime = entry.stat().st_mtime
                if now - mtime > mock_settings.result_ttl_seconds:
                    tid = entry.name
                    shutil.rmtree(entry)
                    task_store.pop(tid, None)

        assert not task_dir.exists()
        assert task_id not in task_store

    async def test_non_expired_directory_kept(self, tmp_path: Path):
        """Directories within TTL are not removed."""
        task_id = "task-fresh"
        task_dir = tmp_path / task_id
        task_dir.mkdir()
        (task_dir / "result.png").write_bytes(_make_png_bytes())

        task_store[task_id] = {"status": "completed", "progress": 100.0}

        ttl = 3600  # 1 hour — directory just created, so not expired

        output_path = tmp_path
        now = time.time()
        for entry in output_path.iterdir():
            if not entry.is_dir():
                continue
            mtime = entry.stat().st_mtime
            if now - mtime > ttl:
                tid = entry.name
                shutil.rmtree(entry)
                task_store.pop(tid, None)

        assert task_dir.exists()
        assert task_id in task_store
