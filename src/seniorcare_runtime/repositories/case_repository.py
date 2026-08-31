from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class CaseRepository:
    def __init__(self, settings: RuntimeSettings):
        root = settings.synthetic_dir
        self.cases = JsonRepository(root / "cases.json", "caseId")
        self.tasks = JsonRepository(root / "case_tasks.json", "caseTaskId")
        self.reminders = JsonRepository(root / "reminders.json", "reminderId")
        self.discharges = JsonRepository(root / "discharge_tasks.json", "dischargeTaskId")
        self.benefits = JsonRepository(root / "benefit_applications.json", "benefitApplicationId")

    def tasks_for(self, senior_id: str) -> list[dict]:
        return self.tasks.find(seniorId=senior_id)

    def reminders_for(self, senior_id: str) -> list[dict]:
        return self.reminders.find(seniorId=senior_id)

    def discharge_tasks_for(self, senior_id: str) -> list[dict]:
        return self.discharges.find(seniorId=senior_id)

    def benefits_for(self, senior_id: str) -> list[dict]:
        return self.benefits.find(seniorId=senior_id)

    def cases_for(self, senior_id: str) -> list[dict]:
        return sorted(
            self.cases.find(seniorId=senior_id),
            key=lambda row: row.get("updatedAt", ""),
            reverse=True,
        )

    def get_case(self, case_id: str) -> dict | None:
        return self.cases.get(case_id)

    def create_case(self, case: dict) -> dict:
        return self.cases.create_with_generated_id("CASE", case)

    def update_case(self, case_id: str, changes: dict) -> dict:
        return self.cases.update(case_id, changes)
