"""Tests for frontend static file serving and API coexistence."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
class TestFrontendServing:
    """Static file serving tests."""

    async def test_index_html_served(self, client):
        """GET / → 200, Content-Type text/html, contains 'MangaLens'."""
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "MangaLens" in resp.text

    async def test_css_served(self, client):
        """GET /css/style.css → 200, Content-Type text/css."""
        resp = await client.get("/css/style.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    async def test_themes_css_served(self, client):
        """GET /css/themes.css → 200."""
        resp = await client.get("/css/themes.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]

    @pytest.mark.parametrize(
        "js_file",
        ["app.js", "api.js", "upload.js", "progress.js", "result.js", "settings.js"],
    )
    async def test_js_files_served(self, client, js_file):
        """GET /js/<file> → 200 for every JS module."""
        resp = await client.get(f"/js/{js_file}")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]

    async def test_nonexistent_file(self, client):
        """GET /nonexistent.xyz → SPA fallback (200 html) or 404."""
        resp = await client.get("/nonexistent.xyz")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
class TestFrontendContent:
    """HTML content validation tests."""

    @pytest.fixture()
    async def html(self, client):
        resp = await client.get("/")
        return resp.text

    async def test_has_upload_section(self, html):
        """index.html contains upload-section."""
        assert 'id="upload-section"' in html

    async def test_has_progress_section(self, html):
        """index.html contains progress-section."""
        assert 'id="progress-section"' in html

    async def test_has_result_section(self, html):
        """index.html contains result-section."""
        assert 'id="result-section"' in html

    async def test_has_settings_modal(self, html):
        """index.html contains settings-modal."""
        assert 'id="settings-modal"' in html

    async def test_has_dark_mode_toggle(self, html):
        """index.html contains theme-toggle element."""
        assert 'id="theme-toggle"' in html

    async def test_has_drag_drop_zone(self, html):
        """index.html contains dropzone element."""
        assert 'id="dropzone"' in html

    async def test_js_modules_loaded(self, html):
        """index.html has script tags for all JS modules."""
        for js in ("app.js", "api.js", "upload.js", "progress.js", "result.js", "settings.js"):
            assert f'src="/js/{js}"' in html


@pytest.mark.asyncio
class TestFrontendApiCoexistence:
    """API endpoints remain functional alongside frontend serving."""

    async def test_api_health_still_works(self, client):
        """GET /api/health → 200, JSON response."""
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert "status" in data

    async def test_api_settings_still_works(self, client):
        """GET /api/settings → 200, JSON response."""
        resp = await client.get("/api/settings")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")

    async def test_api_system_gpu_still_works(self, client):
        """GET /api/system/gpu → 200, JSON response."""
        resp = await client.get("/api/system/gpu")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        data = resp.json()
        assert "backend" in data

    async def test_frontend_and_api_independent(self, client):
        """/ → HTML and /api/health → JSON both work independently."""
        html_resp = await client.get("/")
        api_resp = await client.get("/api/health")

        assert html_resp.status_code == 200
        assert "text/html" in html_resp.headers["content-type"]

        assert api_resp.status_code == 200
        assert api_resp.headers["content-type"].startswith("application/json")
