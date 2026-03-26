"""Shared in-memory task state for MangaLens.

Keeps task_store in one place so routers don't cross-import each other.
"""

from __future__ import annotations

import asyncio

# Structure: { task_id: { status, progress, total_images, completed_images, failed_images } }
task_store: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# P1/P2: WebSocket pub/sub — per-task notification queues
# ---------------------------------------------------------------------------
# Each connected WebSocket client subscribes by calling subscribe(task_id),
# which returns a Queue.  When task state changes, notify_task_changed()
# puts a token into every subscribed queue so the WS handler wakes up
# immediately instead of polling every second.
# ---------------------------------------------------------------------------
_task_watchers: dict[str, list[asyncio.Queue[bool]]] = {}


def subscribe(task_id: str) -> asyncio.Queue[bool]:
    """Register a WebSocket watcher for *task_id*. Returns a notification queue."""
    q: asyncio.Queue[bool] = asyncio.Queue(maxsize=1)
    _task_watchers.setdefault(task_id, []).append(q)
    return q


def unsubscribe(task_id: str, q: asyncio.Queue[bool]) -> None:
    """Unregister a WebSocket watcher."""
    watchers = _task_watchers.get(task_id)
    if watchers:
        try:
            watchers.remove(q)
        except ValueError:
            pass


async def notify_task_changed(task_id: str) -> None:
    """Notify all WebSocket watchers that *task_id*'s state has changed."""
    for q in _task_watchers.get(task_id, []):
        try:
            q.put_nowait(True)  # non-blocking; skip if subscriber has a pending notification
        except asyncio.QueueFull:
            pass
