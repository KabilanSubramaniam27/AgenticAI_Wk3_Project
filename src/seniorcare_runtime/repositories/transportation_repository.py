from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class TransportationRepository:
    def __init__(self, settings: RuntimeSettings):
        self.services = JsonRepository(
            settings.synthetic_dir / "transportation_services.json", "transportationServiceId"
        )
        self.rides = JsonRepository(settings.synthetic_dir / "rides.json", "rideId")
        self.vehicles = JsonRepository(
            settings.synthetic_dir / "transportation_vehicles.json", "vehicleId"
        )

    def rides_for(self, senior_id: str) -> list[dict]:
        return self.rides.find(seniorId=senior_id)

    def services_for(
        self, service_area: str | None = None, wheelchair: bool | None = None
    ) -> list[dict]:
        return [
            row
            for row in self.services.all()
            if (not service_area or row.get("serviceArea") == service_area)
            and (wheelchair is None or row.get("wheelchairAccessible") is wheelchair)
        ]
