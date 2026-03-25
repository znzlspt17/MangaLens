"""User settings endpoints for MangaLens (session-based API key management)."""

from __future__ import annotations

import re
import uuid

from fastapi import APIRouter, Cookie, Header, HTTPException, Request, Response

from server.schemas.models import UserSettings, UserSettingsResponse
from server.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# ---------------------------------------------------------------------------
# In-memory session store (lost on server restart — by design)
# ---------------------------------------------------------------------------
# { session_id: { "deepl_api_key": "...", "google_api_key": "..." } }
session_store: dict[str, dict[str, str]] = {}


def _validate_session_id(sid: str) -> str:
    """Validate a client-provided session ID before using it as a store key."""
    if not _SESSION_ID_PATTERN.fullmatch(sid):
        raise HTTPException(status_code=400, detail="Invalid session ID.")
    return sid


def _set_session_cookie(response: Response, sid: str, request: Request) -> None:
    """Set the session cookie with secure defaults for the current scheme."""
    response.set_cookie(
        key="session_id",
        value=sid,
        httponly=True,
        samesite="lax",
        secure=True,
    )


def _resolve_session_id(
    x_session_id: str | None = None,
    session_id_cookie: str | None = None,
) -> tuple[str, bool]:
    """Return (session_id, is_new).

    Priority: X-Session-Id header > cookie > generate new UUID.
    """
    sid = x_session_id or session_id_cookie
    if sid:
        sid = _validate_session_id(sid)
    if sid and sid in session_store:
        return sid, False
    if sid:
        # Known ID format but not in store yet — create entry
        session_store[sid] = {}
        return sid, False
    new_sid = uuid.uuid4().hex
    session_store[new_sid] = {}
    return new_sid, True


def _mask_key(key: str | None) -> str | None:
    """Mask an API key for safe display (show first 4 + last 4 chars)."""
    if not key:
        return None
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


# ---------------------------------------------------------------------------
# POST /api/settings
# ---------------------------------------------------------------------------

@router.post("/settings", response_model=UserSettingsResponse)
async def update_settings(
    request: Request,
    body: UserSettings,
    response: Response,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    session_id: str | None = Cookie(None, alias="session_id"),
) -> UserSettingsResponse:
    """Store user API keys in the session (in-memory, not persisted)."""
    sid, is_new = _resolve_session_id(x_session_id, session_id)

    if body.deepl_api_key is not None:
        session_store[sid]["deepl_api_key"] = body.deepl_api_key
    if body.google_api_key is not None:
        session_store[sid]["google_api_key"] = body.google_api_key

    # Set cookie so the client remembers the session
    _set_session_cookie(response, sid, request)

    data = session_store[sid]
    logger.info("Session %s settings updated", sid[:8])
    return UserSettingsResponse(
        deepl_api_key=_mask_key(data.get("deepl_api_key")),
        google_api_key=_mask_key(data.get("google_api_key")),
    )


# ---------------------------------------------------------------------------
# GET /api/settings
# ---------------------------------------------------------------------------

@router.get("/settings", response_model=UserSettingsResponse)
async def get_settings(
    request: Request,
    response: Response,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
    session_id: str | None = Cookie(None, alias="session_id"),
) -> UserSettingsResponse:
    """Return current session settings (keys are masked)."""
    sid, is_new = _resolve_session_id(x_session_id, session_id)

    if is_new:
        _set_session_cookie(response, sid, request)

    data = session_store.get(sid, {})
    return UserSettingsResponse(
        deepl_api_key=_mask_key(data.get("deepl_api_key")),
        google_api_key=_mask_key(data.get("google_api_key")),
    )
