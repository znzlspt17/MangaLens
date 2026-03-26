"""Structured logging configuration for MangaLens.

Logs to both console and a rotating file (logs/mangalens.log).
"""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path


def _is_production() -> bool:
    return os.getenv("MANGALENS_ENV", "development").lower() == "production"


_JSON_FORMAT = (
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"name":"%(name)s","message":"%(message)s"}'
)
_DEV_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Ensure logs directory exists
_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = str(_LOG_DIR / "mangalens.log")

LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {"format": _JSON_FORMAT, "datefmt": "%Y-%m-%dT%H:%M:%S"},
        "dev": {"format": _DEV_FORMAT, "datefmt": "%H:%M:%S"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json" if _is_production() else "dev",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": _LOG_FILE,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "formatter": "dev",
            "encoding": "utf-8",
        },
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "handlers": ["console", "file"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
