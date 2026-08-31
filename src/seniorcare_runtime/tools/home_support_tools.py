from datetime import date

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import HomeSupportRepository, SeniorRepository
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


class HomeSupportTools:
    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.repo = HomeSupportRepository(settings)
        self.members = SeniorRepository(settings)
        self.audit = AuditService(settings)

    def request_dummy_home_support(
        self,
        senior_id: str,
        request_type: str,
        priority: str = "medium",
        notes: str = "",
        recipient_id: str | None = None,
    ) -> dict:
        member = self.members.get(senior_id)
        if not member:
            raise KeyError(f"Unknown seniorId: {senior_id}")
        if not self.members.get_care_recipient(member, recipient_id):
            raise PermissionError("Care recipient does not belong to this account")
        request = self.repo.requests.create_with_generated_id(
            "HOME",
            {
                "seniorId": senior_id,
                "requestedByUserId": senior_id,
                "recipientId": recipient_id,
                "careRecipient": self.members.public_care_recipient(member, recipient_id),
                "requestType": request_type,
                "requestDate": str(date.today()),
                "status": "requested",
                "assignedResourceId": None,
                "priority": priority,
                "notes": notes,
                "simulation": True,
            },
        )
        self.audit.record(
            "dummy_home_support_requested", str(request["homeSupportRequestId"]), request
        )
        return simulated(request)
