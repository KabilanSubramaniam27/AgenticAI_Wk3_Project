from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import (
    AppointmentRepository,
    ProviderRepository,
    SeniorRepository,
)
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


class AppointmentTools:
    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.repo = AppointmentRepository(settings)
        self.seniors = SeniorRepository(settings)
        self.providers = ProviderRepository(settings)
        self.audit = AuditService(settings)

    def list_appointments(self, senior_id: str) -> dict:
        return simulated(self.repo.for_senior(senior_id))

    def find_available_slots(self, provider_id: str | None = None) -> dict:
        return simulated(self.repo.available_slots(provider_id))

    def book_dummy_appointment(
        self,
        senior_id: str,
        provider_id: str,
        availability_id: str,
        reason: str,
        transportation_required: bool = False,
        recipient_id: str | None = None,
    ) -> dict:
        member = self.seniors.get(senior_id)
        if not member:
            raise KeyError(f"Unknown seniorId: {senior_id}")
        if not self.seniors.get_care_recipient(member, recipient_id):
            raise PermissionError("Care recipient does not belong to this account")
        if not self.providers.get(provider_id):
            raise KeyError(f"Unknown providerId: {provider_id}")
        slot = self.repo.availability.get(availability_id)
        if not slot or slot.get("providerId") != provider_id or slot.get("status") != "available":
            raise ValueError("Selected local availability slot is not available")
        appointment = {
            "appointmentId": self.repo.appointments.next_id("APT"),
            "seniorId": senior_id,
            "requestedByUserId": senior_id,
            "recipientId": recipient_id,
            "careRecipient": self.seniors.public_care_recipient(member, recipient_id),
            "providerId": provider_id,
            "referralId": None,
            "appointmentDate": slot["availableDate"],
            "appointmentTime": slot["availableTime"],
            "status": "scheduled",
            "reason": reason,
            "transportationRequired": transportation_required,
            "transportationRideId": None,
            "simulation": True,
        }
        self.repo.appointments.create(appointment)
        self.repo.availability.update(availability_id, {"status": "booked"})
        self.audit.record("dummy_appointment_booked", appointment["appointmentId"], appointment)
        return simulated(appointment)

    def cancel_dummy_appointment(self, appointment_id: str) -> dict:
        appointment = self.repo.appointments.update(appointment_id, {"status": "cancelled"})
        self.audit.record("dummy_appointment_cancelled", appointment_id, appointment)
        return simulated(appointment)

    def reschedule_dummy_appointment(self, appointment_id: str, availability_id: str) -> dict:
        appointment = self.repo.get(appointment_id)
        slot = self.repo.availability.get(availability_id)
        if not appointment:
            raise KeyError(f"Unknown appointmentId: {appointment_id}")
        if (
            not slot
            or slot.get("providerId") != appointment.get("providerId")
            or slot.get("status") != "available"
        ):
            raise ValueError("Selected local availability slot is not available")
        updated = self.repo.appointments.update(
            appointment_id,
            {
                "appointmentDate": slot["availableDate"],
                "appointmentTime": slot["availableTime"],
                "status": "scheduled",
            },
        )
        self.repo.availability.update(availability_id, {"status": "booked"})
        self.audit.record("dummy_appointment_rescheduled", appointment_id, updated)
        return simulated(updated)
