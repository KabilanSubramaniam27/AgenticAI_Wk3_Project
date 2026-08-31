import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ObservabilityService:
    SENSITIVE = {"dateofbirth", "date_of_birth", "dob", "api_key", "authorization", "embedding"}

    def __init__(self, path: Path):
        self.path = path

    def emit(self, event: str, **fields: Any) -> None:
        safe = {key: value for key, value in fields.items() if key.casefold() not in self.SENSITIVE}
        row = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **safe}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
