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
    "address",
    "api_key",
    "authorization",
    "date_of_birth",
    "dateofbirth",
    "dob",
    "embedding",
    "embeddings",
    "password",
    "pickup_address",
    "destination_address",
    "token",
}
_VERBOSE_FIELDS = {
    "conversationhistory",
    "conversation_history",
    "currentmembercontext",
    "membercontext",
    "member_context",
    "messages",
    "prompt",
    "retrievedchunks",
    "retrieved_chunks",
    "retrievalresults",
    "retrieval_results",
    "recentconversationturns",
    "relevantconversationturns",
    "rollingconversationsummary",
    "structuredrecords",
    "toolresults",
    "tool_results",
}
_MAX_SUMMARY_FIELDS = max(4, int(os.getenv("APPLICATION_LOG_MAX_FIELDS", "12")))
_MAX_SUMMARY_STRING = max(40, int(os.getenv("APPLICATION_LOG_MAX_STRING", "160")))
_IDENTIFIER_LIST_FIELDS = {"agents", "selectedagents", "selected_agents", "selectedtools"}

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


def _shape(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {"type": "object", "fieldCount": len(value), "keys": list(value)[:8]}
    if isinstance(value, (list, tuple, set)):
        return {"type": "array", "count": len(value)}
    if isinstance(value, str):
        return {"type": "text", "characters": len(value)}
    return {"type": type(value).__name__}


def _safe(value: Any, key: str = "") -> Any:
    if key.casefold() in _SENSITIVE:
        return "[REDACTED]"
    if key.casefold() in _VERBOSE_FIELDS:
        return _shape(value)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        return (
            value
            if len(value) <= _MAX_SUMMARY_STRING
            else f"{value[: _MAX_SUMMARY_STRING - 3]}..."
        )
    if isinstance(value, dict):
        items = list(value.items())[:_MAX_SUMMARY_FIELDS]
        result: dict[str, Any] = {}
        for item_key, item_value in items:
            item_key = str(item_key)
            if isinstance(item_value, (dict, list, tuple, set)):
                result[item_key] = (
                    _safe(item_value, item_key)
                    if item_key.casefold() in _VERBOSE_FIELDS | _IDENTIFIER_LIST_FIELDS
                    else _shape(item_value)
                )
            else:
                result[item_key] = _safe(item_value, item_key)
        if len(value) > len(items):
            result["_additionalFields"] = len(value) - len(items)
        return result
    if isinstance(value, (list, tuple, set)):
        if key.casefold() in _IDENTIFIER_LIST_FIELDS and all(
            isinstance(item, str) for item in value
        ):
            return list(value)[:8]
        return _shape(value)
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
    selected_agent: str | None = None,
    selected_tool: str | None = None,
    duration_ms: float | None = None,
    status: str | None = None,
) -> None:
    """Print one safe, compact JSON event at a component boundary."""
    row: dict[str, Any] = {
        "timestamp": datetime.now(UTC).strftime("%m:%d:%Y %H:%M:%S.%f")[:-3],
        "requestId": request_id or current_request_id(),
        "component": component,
        "operation": operation,
        "direction": direction,
        "status": status
        or ("error" if direction == "error" else "success" if direction == "output" else "started"),
    }
    if selected_agent:
        row["selectedAgent"] = selected_agent
    elif component in {"agent", "subgraph"}:
        row["selectedAgent"] = operation.split("_", 1)[0]
    elif "Agent_" in operation:
        row["selectedAgent"] = operation.split("_", 1)[0]
    if isinstance(payload, dict):
        payload_agent = payload.get("agent") or payload.get("agentName")
        if payload_agent and "selectedAgent" not in row:
            row["selectedAgent"] = str(payload_agent)
        payload_agents = payload.get("selectedAgents") or payload.get("agents")
        if isinstance(payload_agents, (list, tuple)):
            row["selectedAgents"] = [str(item) for item in payload_agents[:8]]
    if selected_tool or component == "mcp_tool":
        row["selectedTool"] = selected_tool or operation
    if duration_ms is not None:
        row["durationMs"] = round(duration_ms, 2)
    if isinstance(payload, BaseException):
        row["error"] = _safe(payload)
    elif direction == "error" and isinstance(payload, dict) and "error" in payload:
        row["error"] = _safe(payload["error"], "error")
        remaining = {key: value for key, value in payload.items() if key != "error"}
        if remaining:
            row["summary"] = _safe(remaining)
    elif payload is not None:
        row["summary"] = _safe(payload)
    logger.info(json.dumps(row, default=str, separators=(",", ":")))
