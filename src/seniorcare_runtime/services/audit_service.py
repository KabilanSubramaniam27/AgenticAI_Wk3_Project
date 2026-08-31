import json
import threading
from datetime import UTC, datetime
from typing import Any

from seniorcare_runtime.config import RuntimeSettings


class AuditService:
    _lock = threading.Lock()

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings

    def record(self, action: str, entity_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        safe_payload = {
            key: value
            for key, value in payload.items()
            if key.casefold()
            not in {"dateofbirth", "date_of_birth", "dob", "api_key", "authorization"}
        }
        event = {
            "requestId": safe_payload.get("requestId"),
            "userId": safe_payload.get("userId") or safe_payload.get("seniorId"),
            "seniorId": safe_payload.get("seniorId"),
            "caseId": safe_payload.get("caseId"),
            "agent": safe_payload.get("agent", "seniorcare_runtime"),
            "tool": safe_payload.get("tool", action),
            "timestamp": datetime.now(UTC).isoformat(),
            "action": action,
            "status": safe_payload.get("status", "success"),
            "entityId": entity_id,
            "simulation": True,
            "externalActionPerformed": False,
            "payload": safe_payload,
        }
        self.settings.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.settings.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event

    def events_for(self, senior_id: str, case_id: str | None = None, limit: int = 20) -> list[dict]:
        if not self.settings.audit_path.exists():
            return []
        rows = [
            json.loads(line)
            for line in self.settings.audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [
            row
            for row in rows
            if row.get("seniorId") == senior_id or row.get("userId") == senior_id
        ]
        if case_id:
            matching = [
                row
                for row in matching
                if row.get("caseId") == case_id or row.get("entityId") == case_id
            ]
        return matching[-limit:][::-1]
