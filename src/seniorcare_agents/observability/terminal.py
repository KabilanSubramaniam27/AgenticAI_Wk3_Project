from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

_request_id: ContextVar[str] = ContextVar("seniorcare_request_id", default="-")
_SENSITIVE = {
    "api_key",
    "authorization",
    "date_of_birth",
    "dateofbirth",
    "dob",
    "embedding",
    "embeddings",
    "password",
    "token",
}

logger = logging.getLogger("seniorcare.flow")
if not logger.handlers:
    formatter = logging.Formatter("%(message)s")
    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(formatter)
    logger.addHandler(terminal_handler)

    project_root = Path.cwd()
    application_log = Path(os.getenv("APPLICATION_LOG_PATH", "application.log")).expanduser()
    if not application_log.is_absolute():
        application_log = project_root / application_log
    application_log.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(application_log, mode="a", encoding="utf-8", delay=True)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def begin_request(request_id: str | None = None) -> Token[str]:
    return _request_id.set(request_id or f"REQ-{uuid4().hex[:12]}")


def end_request(token: Token[str]) -> None:
    _request_id.reset(token)


def current_request_id() -> str:
    return _request_id.get()


def _safe(value: Any, key: str = "") -> Any:
    if key.casefold() in _SENSITIVE:
        return "[REDACTED]"
    if isinstance(value, str):
        return value if len(value) <= 300 else f"{value[:297]}..."
    if isinstance(value, dict):
        items = list(value.items())[:16]
        result = {str(item_key): _safe(item_value, str(item_key)) for item_key, item_value in items}
        if len(value) > len(items):
            result["_additionalFields"] = len(value) - len(items)
        return result
    if isinstance(value, (list, tuple)):
        sample = [_safe(item) for item in value[:3]]
        return {"count": len(value), "sample": sample}
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)[:300]}
    return value


def flow_event(
    component: str,
    operation: str,
    direction: str,
    payload: Any = None,
    *,
    request_id: str | None = None,
) -> None:
    """Print one safe, compact JSON event at a component boundary."""
    row = {
        "timestamp": datetime.now(UTC).strftime("%m:%d:%Y %H:%M:%S.%f")[:-3],
        "requestId": request_id or current_request_id(),
        "component": component,
        "operation": operation,
        "direction": direction,
    }
    if payload is not None:
        row["payload"] = _safe(payload)
    logger.info(json.dumps(row, default=str, separators=(",", ":")))
