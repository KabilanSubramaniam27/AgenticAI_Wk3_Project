import json
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any


class InMemoryActionStore:
    """Process-local pending-action state owned by the agent API."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self._rows: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self._load()

    def create(self, row: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            action_id = str(row["action_id"])
            if action_id in self._rows:
                raise ValueError(f"Duplicate action ID: {action_id}")
            self._rows[action_id] = deepcopy(row)
            self._persist()
            return deepcopy(row)

    def get(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get(action_id)
            return deepcopy(row) if row else None

    def update(self, action_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if action_id not in self._rows:
                raise KeyError(f"Unknown action ID: {action_id}")
            self._rows[action_id].update(deepcopy(changes))
            self._persist()
            return deepcopy(self._rows[action_id])

    def delete(self, action_id: str) -> dict[str, Any] | None:
        """Remove an unneeded pending action and return its former value."""
        with self._lock:
            row = self._rows.pop(action_id, None)
            if row is not None:
                self._persist()
            return deepcopy(row) if row is not None else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(list(self._rows.values()))

    def for_user(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy([row for row in self._rows.values() if row.get("user_id") == user_id])

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            rows = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self._rows = {str(row["action_id"]): row for row in rows if row.get("action_id")}
        except (OSError, ValueError, TypeError):
            self._rows = {}

    def _persist(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.storage_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(list(self._rows.values()), indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.storage_path)


class PersistentActionStore(InMemoryActionStore):
    def __init__(self, storage_path: Path):
        super().__init__(storage_path)
