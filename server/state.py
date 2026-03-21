"""Shared in-memory task state for MangaLens.

Keeps task_store in one place so routers don't cross-import each other.
"""

from __future__ import annotations

# Structure: { task_id: { status, progress, total_images, completed_images, failed_images } }
task_store: dict[str, dict] = {}
