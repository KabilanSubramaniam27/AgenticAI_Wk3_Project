from datetime import date, timedelta

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import MedicationRepository, SeniorRepository
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


class MedicationTools:
    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.repo = MedicationRepository(settings)
        self.seniors = SeniorRepository(settings)
        self.audit = AuditService(settings)

    def list_medications(self, senior_id: str) -> dict:
        return simulated(self.repo.for_senior(senior_id))

    def lookup_reference(self, medication_name: str) -> dict:
        return simulated(self.repo.reference(medication_name))

    def request_dummy_refill(
        self,
        senior_id: str,
        medication_id: str,
        method: str = "pickup",
        recipient_id: str | None = None,
    ) -> dict:
        member = self.seniors.get(senior_id)
        if not member:
            raise KeyError(f"Unknown seniorId: {senior_id}")
        if not self.seniors.get_care_recipient(member, recipient_id):
            raise PermissionError("Care recipient does not belong to this account")
        medication = self.repo.medications.get(medication_id)
        if not medication or medication.get("seniorId") != senior_id:
            raise ValueError("Medication does not belong to senior")
        refill = {
            "refillId": self.repo.refills.next_id("RFL"),
            "medicationId": medication_id,
            "seniorId": senior_id,
            "requestedByUserId": senior_id,
            "recipientId": recipient_id,
            "careRecipient": self.seniors.public_care_recipient(member, recipient_id),
            "pharmacyId": medication.get("pharmacyId"),
            "requestDate": str(date.today()),
            "status": "requested",
            "pickupOrDelivery": method,
            "estimatedReadyDate": str(date.today() + timedelta(days=2)),
            "simulation": True,
        }
        self.repo.refills.create(refill)
        self.audit.record("dummy_refill_requested", str(refill["refillId"]), refill)
        return simulated(refill)
