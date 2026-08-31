from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class SocialRepository:
    def __init__(self, settings: RuntimeSettings):
        self.activities = JsonRepository(
            settings.synthetic_dir / "social_activities.json", "activityId"
        )
        self.registrations = JsonRepository(
            settings.synthetic_dir / "activity_registrations.json", "registrationId"
        )

    def search(self, county: str | None = None, activity_type: str | None = None) -> list[dict]:
        return [
            row
            for row in self.activities.all()
            if (not county or row.get("county") == county)
            and (
                not activity_type
                or activity_type.casefold() in row.get("activityType", "").casefold()
            )
        ]
