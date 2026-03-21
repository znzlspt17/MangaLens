"""Pydantic request / response models for MangaLens API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    ready: bool = False
    gpu_info: GPUInfoResponse | None = None


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

class GPUInfoResponse(BaseModel):
    backend: str
    device: str
    gpu_name: str
    vram_mb: int
    driver_version: str


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    task_id: str
    status: str = "queued"


# ---------------------------------------------------------------------------
# Task status
# ---------------------------------------------------------------------------

class TaskStatus(BaseModel):
    task_id: str
    status: str  # queued | processing | completed | failed
    progress: float = 0.0
    total_images: int = 0
    completed_images: int = 0
    failed_images: int = 0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class TranslationResult(BaseModel):
    task_id: str
    images: list[str] = Field(default_factory=list)
    translation_log_url: str | None = None


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

class UserSettings(BaseModel):
    """Input model for POST /api/settings."""
    deepl_api_key: str | None = None
    google_api_key: str | None = None


class UserSettingsResponse(BaseModel):
    """Response model with masked keys."""
    deepl_api_key: str | None = None
    google_api_key: str | None = None


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str


# Rebuild HealthResponse so the forward reference to GPUInfoResponse resolves.
HealthResponse.model_rebuild()
