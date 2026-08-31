from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class HomeSupportRepository:
    def __init__(self, settings: RuntimeSettings):
        self.requests = JsonRepository(
            settings.synthetic_dir / "home_support_requests.json", "homeSupportRequestId"
        )

    def for_senior(self, senior_id: str) -> list[dict]:
        return self.requests.find(seniorId=senior_id)
