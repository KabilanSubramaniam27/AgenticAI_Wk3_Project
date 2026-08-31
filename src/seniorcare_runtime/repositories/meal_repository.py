from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class MealRepository:
    def __init__(self, settings: RuntimeSettings):
        self.services = JsonRepository(
            settings.synthetic_dir / "meal_services.json", "mealServiceId"
        )
        self.enrollments = JsonRepository(
            settings.synthetic_dir / "meal_enrollments.json", "mealEnrollmentId"
        )

    def services_for(
        self, service_area: str | None = None, service_type: str | None = None
    ) -> list[dict]:
        return [
            row
            for row in self.services.all()
            if (not service_area or row.get("serviceArea") == service_area)
            and (not service_type or row.get("serviceType") == service_type)
        ]

    def enrollments_for(self, senior_id: str) -> list[dict]:
        return self.enrollments.find(seniorId=senior_id)
