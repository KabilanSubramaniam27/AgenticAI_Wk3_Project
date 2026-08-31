from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class AppointmentRepository:
    def __init__(self, settings: RuntimeSettings):
        self.appointments = JsonRepository(
            settings.synthetic_dir / "appointments.json", "appointmentId"
        )
        self.availability = JsonRepository(
            settings.synthetic_dir / "provider_availability.json", "availabilityId"
        )
        self.referrals = JsonRepository(settings.synthetic_dir / "referrals.json", "referralId")

    def for_senior(self, senior_id: str) -> list[dict]:
        return self.appointments.find(seniorId=senior_id)

    def get(self, appointment_id: str) -> dict | None:
        return self.appointments.get(appointment_id)

    def available_slots(self, provider_id: str | None = None) -> list[dict]:
        rows = self.availability.find(status="available")
        return [row for row in rows if not provider_id or row.get("providerId") == provider_id]

    def referrals_for(self, senior_id: str) -> list[dict]:
        return self.referrals.find(seniorId=senior_id)
