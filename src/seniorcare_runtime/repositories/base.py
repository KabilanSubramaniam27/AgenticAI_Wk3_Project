import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any


class JsonRepository:
    """Small deterministic JSON repository with atomic file replacement."""

    _lock = threading.RLock()

    def __init__(self, path: Path, id_field: str):
        self.path = path
        self.id_field = id_field

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {self.path}")
        return value

    def get(self, record_id: str) -> dict[str, Any] | None:
        return next((row for row in self.all() if row.get(self.id_field) == record_id), None)

    def find(self, **filters: Any) -> list[dict[str, Any]]:
        return [
            row
            for row in self.all()
            if all(row.get(key) == value for key, value in filters.items())
        ]

    def save_all(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                json.dump(rows, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                temporary = Path(handle.name)
            os.replace(temporary, self.path)

    def create(self, row: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rows = self.all()
            record_id = row.get(self.id_field)
            if not record_id:
                raise ValueError(f"Missing {self.id_field}")
            if any(item.get(self.id_field) == record_id for item in rows):
                raise ValueError(f"Duplicate {self.id_field}: {record_id}")
            rows.append(row)
            self.save_all(rows)
        return row

    def update(self, record_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rows = self.all()
            for index, row in enumerate(rows):
                if row.get(self.id_field) == record_id:
                    updated = {**row, **changes, self.id_field: record_id}
                    rows[index] = updated
                    self.save_all(rows)
                    return updated
        raise KeyError(f"Unknown {self.id_field}: {record_id}")

    def next_id(self, prefix: str) -> str:
        numbers = []
        for row in self.all():
            value = str(row.get(self.id_field, ""))
            if value.startswith(prefix) and value[len(prefix) :].isdigit():
                numbers.append(int(value[len(prefix) :]))
        return f"{prefix}{max(numbers, default=1000) + 1}"

    def create_with_generated_id(self, prefix: str, row: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            rows = self.all()
            numbers = []
            for item in rows:
                value = str(item.get(self.id_field, ""))
                if value.startswith(prefix) and value[len(prefix) :].isdigit():
                    numbers.append(int(value[len(prefix) :]))
            created = {**row, self.id_field: f"{prefix}{max(numbers, default=1000) + 1}"}
            rows.append(created)
            self.save_all(rows)
            return created
