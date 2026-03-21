"""Structured logging configuration for MangaLens."""

from __future__ import annotations

import logging
import logging.config
import os
import sys


def _is_production() -> bool:
    return os.getenv("MANGALENS_ENV", "development").lower() == "production"


_JSON_FORMAT = (
    '{"time":"%(asctime)s","level":"%(levelname)s",'
    '"name":"%(name)s","message":"%(message)s"}'
)
_DEV_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

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
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
