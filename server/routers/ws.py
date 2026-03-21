"""WebSocket endpoint for real-time task progress notifications."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.state import task_store
from server.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

_TERMINAL_STATUSES = frozenset({"completed", "partial", "failed"})
_POLL_INTERVAL = 1  # seconds


@router.websocket("/ws/progress/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str) -> None:
    """Stream task progress updates over WebSocket at 1-second intervals.

    Sends JSON messages with task status until the task reaches a terminal
    state (completed / partial / failed), then closes the connection.
    """
    await websocket.accept()

    # Validate that the task exists
    if task_id not in task_store:
        await websocket.send_json({"error": "task_not_found", "task_id": task_id})
        await websocket.close(code=1008)
        return

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

            await asyncio.sleep(_POLL_INTERVAL)
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected for task %s", task_id)
