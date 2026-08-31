from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import SeniorRepository, SocialRepository
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


class EventTools:
    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.repo = SocialRepository(settings)
        self.seniors = SeniorRepository(settings)
        self.audit = AuditService(settings)

    def find_events(self, county: str | None = None, activity_type: str | None = None) -> dict:
        return simulated(self.repo.search(county, activity_type))

    def register_dummy_event(
        self, senior_id: str, activity_id: str, recipient_id: str | None = None
    ) -> dict:
        member = self.seniors.get(senior_id)
        if not member:
            raise KeyError(f"Unknown seniorId: {senior_id}")
        if not self.seniors.get_care_recipient(member, recipient_id):
            raise PermissionError("Care recipient does not belong to this account")
        if not self.repo.activities.get(activity_id):
            raise KeyError(f"Unknown activityId: {activity_id}")
        registration = {
            "registrationId": self.repo.registrations.next_id("REG"),
            "seniorId": senior_id,
            "requestedByUserId": senior_id,
            "recipientId": recipient_id,
            "careRecipient": self.seniors.public_care_recipient(member, recipient_id),
            "activityId": activity_id,
            "status": "registered",
            "simulation": True,
        }
        self.repo.registrations.create(registration)
        self.audit.record(
            "dummy_event_registered", str(registration["registrationId"]), registration
        )
        return simulated(registration)
