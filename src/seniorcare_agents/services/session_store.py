from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4


@dataclass
class ChatMessage:
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AgentSession:
    session_id: str
    user_id: str
    recipient_id: str | None = None
    active_case_id: str | None = None
    last_query: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    pending_action_ids: list[str] = field(default_factory=list)
    last_response: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class InMemoryAgentSessionStore:
    """Thread-safe session state with optional durable JSON persistence."""

    def __init__(
        self,
        ttl_minutes: int = 120,
        max_sessions: int = 1_000,
        storage_path: Path | None = None,
    ):
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_sessions = max_sessions
        self.storage_path = storage_path
        self._sessions: dict[str, AgentSession] = {}
        self._action_sessions: dict[str, str] = {}
        self._lock = RLock()
        self._load()

    def get_or_create(self, user_id: str, session_id: str | None = None) -> AgentSession:
        with self._lock:
            self._evict()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                if session.user_id != user_id:
                    raise PermissionError("Session does not belong to this member")
                self._touch(session)
                self._persist()
                return deepcopy(session)
            identifier = session_id or f"SESSION-{uuid4().hex[:16]}"
            session = AgentSession(session_id=identifier, user_id=user_id)
            self._sessions[identifier] = session
            self._enforce_limit()
            self._persist()
            return deepcopy(session)

    def record_exchange(
        self, session_id: str, query: str, response: dict[str, Any]
    ) -> AgentSession:
        with self._lock:
            session = self._require(session_id)
            session.last_query = query
            session.messages.append(ChatMessage(role="user", content=query))
            final_response = str(response.get("final_response") or "")
            if final_response:
                session.messages.append(ChatMessage(role="assistant", content=final_response))
            session.last_response = deepcopy(response)
            action_ids = [
                str(action["action_id"])
                for action in response.get("proposed_actions", [])
                if action.get("action_id")
            ]
            session.pending_action_ids = action_ids
            for action_id in action_ids:
                self._action_sessions[action_id] = session_id
            self._touch(session)
            self._persist()
            return deepcopy(session)

    def resolve_action(
        self, action_id: str, status: Literal["executed", "rejected", "failed"]
    ) -> AgentSession | None:
        with self._lock:
            session_id = self._action_sessions.pop(action_id, None)
            if not session_id or session_id not in self._sessions:
                return None
            session = self._sessions[session_id]
            session.pending_action_ids = [
                value for value in session.pending_action_ids if value != action_id
            ]
            session.messages.append(
                ChatMessage(role="system", content=f"Local action {status}: {action_id}")
            )
            self._touch(session)
            self._persist()
            return deepcopy(session)

    def discard_pending_actions(self, session_id: str) -> list[str]:
        """Detach all unapproved proposals when the conversation moves to a new turn."""
        with self._lock:
            session = self._require(session_id)
            action_ids = list(session.pending_action_ids)
            session.pending_action_ids = []
            for action_id in action_ids:
                self._action_sessions.pop(action_id, None)
            if action_ids:
                session.messages.append(
                    ChatMessage(
                        role="system",
                        content="Previous unapproved action proposal discarded.",
                    )
                )
            self._touch(session)
            self._persist()
            return action_ids

    def set_active_case(self, session_id: str, case_id: str) -> AgentSession:
        with self._lock:
            session = self._require(session_id)
            session.active_case_id = case_id
            self._touch(session)
            self._persist()
            return deepcopy(session)

    def set_recipient(self, session_id: str, recipient_id: str) -> AgentSession:
        with self._lock:
            session = self._require(session_id)
            session.recipient_id = recipient_id
            self._touch(session)
            self._persist()
            return deepcopy(session)

    def find_by_action(self, action_id: str, user_id: str) -> AgentSession | None:
        with self._lock:
            session_id = self._action_sessions.get(action_id)
            session = self._sessions.get(session_id or "")
            if not session or session.user_id != user_id:
                return None
            return deepcopy(session)

    def snapshot(self, session_id: str, user_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._require(session_id)
            if session.user_id != user_id:
                raise PermissionError("Session does not belong to this member")
            return asdict(deepcopy(session))

    def _require(self, session_id: str) -> AgentSession:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError("Agent session not found or expired")
        return session

    @staticmethod
    def _touch(session: AgentSession) -> None:
        session.updated_at = datetime.now(UTC).isoformat()

    def _evict(self) -> None:
        cutoff = datetime.now(UTC) - self.ttl
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if datetime.fromisoformat(session.updated_at) < cutoff
        ]
        for session_id in expired:
            self._drop(session_id)

    def _enforce_limit(self) -> None:
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].updated_at)
            self._drop(oldest)

    def _drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._action_sessions = {
            action_id: owner
            for action_id, owner in self._action_sessions.items()
            if owner != session_id
        }

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._sessions = {
                key: AgentSession(
                    **{
                        **value,
                        "messages": [ChatMessage(**message) for message in value.get("messages", [])],
                    }
                )
                for key, value in payload.get("sessions", {}).items()
            }
            self._action_sessions = {
                str(key): str(value) for key, value in payload.get("actionSessions", {}).items()
            }
            self._evict()
        except (OSError, ValueError, TypeError):
            self._sessions = {}
            self._action_sessions = {}

    def _persist(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "sessions": {key: asdict(value) for key, value in self._sessions.items()},
                    "actionSessions": self._action_sessions,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.storage_path)


class PersistentAgentSessionStore(InMemoryAgentSessionStore):
    """Durable API session/checkpoint metadata restored after process restart."""

    def __init__(self, storage_path: Path, ttl_minutes: int = 1440, max_sessions: int = 1_000):
        super().__init__(ttl_minutes, max_sessions, storage_path)
