"""WebSocket endpoint for real-time task progress notifications."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.state import task_store, subscribe, unsubscribe
from server.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_TERMINAL_STATUSES = frozenset({"completed", "partial", "failed"})
# Maximum seconds to wait for a state-change notification before sending a
# heartbeat.  Replaces the old 1-second blind poll — the WebSocket now wakes
# up only when state actually changes, reducing CPU overhead under many
# concurrent connections.
_HEARTBEAT_TIMEOUT = 5.0


@router.websocket("/ws/progress/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str) -> None:
    """Stream task progress updates over WebSocket.

    Instead of polling every second, the handler waits on a per-task
    notification queue that is signalled by the pipeline runner whenever
    state changes.  A heartbeat is sent every ``_HEARTBEAT_TIMEOUT`` seconds
    even when no change occurs, so the client can detect stale connections.
    """
    await websocket.accept()

    # Validate that the task exists
    if task_id not in task_store:
        await websocket.send_json({"error": "task_not_found", "task_id": task_id})
        await websocket.close(code=1008)
        return

    notify_queue = subscribe(task_id)
    try:
        while True:
            task = task_store.get(task_id)
            if task is None:
                await websocket.send_json({"error": "task_not_found", "task_id": task_id})
                await websocket.close(code=1008)
                return

            message = {
                "task_id": task_id,
                "status": task.get("status", "unknown"),
                "progress": task.get("progress", 0.0),
                "total_images": task.get("total_images", 0),
                "completed_images": task.get("completed_images", 0),
                "failed_images": task.get("failed_images", 0),
            }
            await websocket.send_json(message)

            if message["status"] in _TERMINAL_STATUSES:
                await websocket.close(code=1000)
                return

            # Wait for a state-change notification (or heartbeat timeout)
            try:
                await asyncio.wait_for(notify_queue.get(), timeout=_HEARTBEAT_TIMEOUT)
            except asyncio.TimeoutError:
                pass  # heartbeat — loop and resend current state

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected for task %s", task_id)
    finally:
        unsubscribe(task_id, notify_queue)
