from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import (
    AppointmentRepository,
    CaseRepository,
    HomeSupportRepository,
    MedicationRepository,
    SeniorRepository,
    TransportationRepository,
)


class SeniorContextService:
    def __init__(self, settings: RuntimeSettings):
        self.seniors = SeniorRepository(settings)
        self.appointments = AppointmentRepository(settings)
        self.transportation = TransportationRepository(settings)
        self.medications = MedicationRepository(settings)
        self.home = HomeSupportRepository(settings)
        self.cases = CaseRepository(settings)

    def get_context(self, senior_id: str) -> dict:
        senior = self.seniors.get(senior_id)
        if not senior:
            raise KeyError(f"Unknown seniorId: {senior_id}")
        return {
            "senior": senior,
            "caregivers": self.seniors.caregivers_for(senior_id),
            "appointments": self.appointments.for_senior(senior_id),
            "referrals": self.appointments.referrals_for(senior_id),
            "rides": self.transportation.rides_for(senior_id),
            "medications": self.medications.for_senior(senior_id),
            "refills": self.medications.refills_for(senior_id),
            "homeSupportRequests": self.home.for_senior(senior_id),
            "caseTasks": self.cases.tasks_for(senior_id),
            "reminders": self.cases.reminders_for(senior_id),
            "dischargeTasks": self.cases.discharge_tasks_for(senior_id),
            "benefitApplications": self.cases.benefits_for(senior_id),
            "cases": self.cases.cases_for(senior_id),
        }
