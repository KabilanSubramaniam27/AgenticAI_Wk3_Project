import json
from datetime import UTC, datetime

from seniorcare_ingestion.config import Settings


class Manifest:
    def __init__(self, settings: Settings):
        self.path = settings.manifest_path
        self.data = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"runs": {}, "chunks": {}, "sources": {}}
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")

    def begin(self) -> str:
        run_id = f"RUN-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
        self.data["runs"][run_id] = {
            "startedAt": datetime.now(UTC).isoformat(),
            "status": "running",
        }
        self.save()
        return run_id

    def finish(self, run_id: str, status: str = "completed") -> None:
        self.data["runs"][run_id].update(completedAt=datetime.now(UTC).isoformat(), status=status)
        self.save()
