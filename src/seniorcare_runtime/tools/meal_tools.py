from datetime import date

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import MealRepository, SeniorRepository
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


class MealTools:
    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.repo = MealRepository(settings)
        self.seniors = SeniorRepository(settings)
        self.audit = AuditService(settings)

    def find_meal_services(
        self, service_area: str | None = None, service_type: str | None = None
    ) -> dict:
        return simulated(self.repo.services_for(service_area, service_type))

    def enroll_dummy_meal_service(
        self, senior_id: str, meal_service_id: str, recipient_id: str | None = None
    ) -> dict:
        member = self.seniors.get(senior_id)
        if not member:
            raise KeyError(f"Unknown seniorId: {senior_id}")
        if not self.seniors.get_care_recipient(member, recipient_id):
            raise PermissionError("Care recipient does not belong to this account")
        if not self.repo.services.get(meal_service_id):
            raise KeyError(f"Unknown mealServiceId: {meal_service_id}")
        enrollment = {
            "mealEnrollmentId": self.repo.enrollments.next_id("MEN"),
            "seniorId": senior_id,
            "requestedByUserId": senior_id,
            "recipientId": recipient_id,
            "careRecipient": self.seniors.public_care_recipient(member, recipient_id),
            "mealServiceId": meal_service_id,
            "requestDate": str(date.today()),
            "status": "requested",
            "simulation": True,
        }
        self.repo.enrollments.create(enrollment)
        self.audit.record("dummy_meal_enrollment", str(enrollment["mealEnrollmentId"]), enrollment)
        return simulated(enrollment)
