import json

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class ProviderRepository:
    def __init__(self, settings: RuntimeSettings):
        self.local = JsonRepository(settings.synthetic_dir / "providers.json", "providerId")
        self.public_path = settings.project_root / "data/normalized/providers.jsonl"

    def get(self, provider_id: str) -> dict | None:
        return self.local.get(provider_id)

    def search(
        self,
        specialty: str | None = None,
        county: str | None = None,
        city: str | None = None,
        limit: int = 20,
        include_public: bool = True,
    ) -> list[dict]:
        terms = {"specialty": specialty, "county": county, "city": city}
        rows = [
            row
            for row in self.local.all()
            if all(
                not value or value.casefold() in str(row.get(key, "")).casefold()
                for key, value in terms.items()
            )
        ]
        if include_public and len(rows) < limit and self.public_path.exists():
            for line in self.public_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                public_terms = {
                    "specialty": row.get("specialty"),
                    "county": None,
                    "city": row.get("city"),
                }
                if (
                    specialty
                    and specialty.casefold() not in str(public_terms["specialty"] or "").casefold()
                ):
                    continue
                if city and city.casefold() not in str(public_terms["city"] or "").casefold():
                    continue
                if county:
                    continue  # CMS rows do not currently carry a normalized county.
                rows.append(row)
                if len(rows) >= limit:
                    break
        return rows[:limit]
