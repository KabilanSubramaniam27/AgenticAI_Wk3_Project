import json

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class MedicationRepository:
    def __init__(self, settings: RuntimeSettings):
        self.medications = JsonRepository(
            settings.synthetic_dir / "medications.json", "medicationId"
        )
        self.refills = JsonRepository(settings.synthetic_dir / "pharmacy_refills.json", "refillId")
        self.reference_path = settings.project_root / "data/normalized/medications.jsonl"

    def for_senior(self, senior_id: str) -> list[dict]:
        return self.medications.find(seniorId=senior_id)

    def refills_for(self, senior_id: str) -> list[dict]:
        return self.refills.find(seniorId=senior_id)

    def reference(self, name: str, limit: int = 20) -> list[dict]:
        term = name.casefold()
        results: list[dict] = []
        if not self.reference_path.exists():
            return results
        for line in self.reference_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            names = f"{row.get('brand_name', '')} {row.get('generic_name', '')} {' '.join(row.get('substance_names', []))}"
            if term in names.casefold():
                results.append(row)
            if len(results) >= limit:
                break
        return results
