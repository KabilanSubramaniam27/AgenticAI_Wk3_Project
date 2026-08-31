from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import CaseRepository, SeniorRepository
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


class ReminderTools:
    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.repo = CaseRepository(settings)
        self.seniors = SeniorRepository(settings)
        self.audit = AuditService(settings)

    def list_reminders(self, senior_id: str) -> dict:
        return simulated(self.repo.reminders_for(senior_id))

    def schedule_dummy_reminder(
        self,
        senior_id: str,
        reminder_type: str,
        related_entity_id: str,
        reminder_date: str,
        reminder_time: str,
        message: str,
        caregiver_id: str | None = None,
        delivery_method: str = "app",
    ) -> dict:
        if not self.seniors.get(senior_id):
            raise KeyError(f"Unknown seniorId: {senior_id}")
        reminder = {
            "reminderId": self.repo.reminders.next_id("REM"),
            "seniorId": senior_id,
            "caregiverId": caregiver_id,
            "reminderType": reminder_type,
            "relatedEntityId": related_entity_id,
            "reminderDate": reminder_date,
            "reminderTime": reminder_time,
            "deliveryMethod": delivery_method,
            "status": "scheduled",
            "message": message,
            "simulation": True,
        }
        self.repo.reminders.create(reminder)
        self.audit.record("dummy_reminder_scheduled", str(reminder["reminderId"]), reminder)
        return simulated(reminder)
