"""Environment variable and settings management for MangaLens.

Uses pydantic-settings to load configuration from .env file
and environment variables.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # GPU
    gpu_backend: str = "auto"  # auto | cuda | rocm | cpu

    # Upscaler
    upscaler_variant: str = "anime_6b"  # anime_6b | x4plus

    # Detector
    use_magi_detector: bool = False  # True → Magi v2, False → YOLOv5
    magi_vram_threshold_mb: int = 4096  # Magi v2 requires ≥4GB VRAM

    # Server
    max_upload_size: int = 52_428_800  # 50 MB
    max_concurrent_tasks: int = 1
    result_ttl_seconds: int = 3600  # 1 hour
    delete_after_download: bool = False
    allowed_origins: str = ""  # comma-separated
    skip_warmup: bool = False

    # Paths
    model_cache_dir: str = "./models"
    font_dir: str = "./fonts"
    output_dir: str = "./output"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    def get_allowed_origins(self) -> list[str]:
        """Parse ALLOWED_ORIGINS into a list of origin strings."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    def allow_cors_credentials(self) -> bool:
        """Allow credentialed CORS only for explicit origins."""
        origins = self.get_allowed_origins()
        return bool(origins) and "*" not in origins


settings = Settings()
