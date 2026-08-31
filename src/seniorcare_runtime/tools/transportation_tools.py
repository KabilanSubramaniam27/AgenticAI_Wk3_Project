import re
from datetime import datetime, timedelta

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import (
    AppointmentRepository,
    SeniorRepository,
    TransportationRepository,
)
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


class TransportationTools:
    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.repo = TransportationRepository(settings)
        self.seniors = SeniorRepository(settings)
        self.appointments = AppointmentRepository(settings)
        self.audit = AuditService(settings)

    def find_services(
        self, service_area: str | None = None, wheelchair: bool | None = None
    ) -> dict:
        return simulated(self.repo.services_for(service_area, wheelchair))

    def list_rides(self, senior_id: str) -> dict:
        return simulated(self.repo.rides_for(senior_id))

    def estimate_dummy_travel(self, origin_zip: str, destination_zip: str) -> dict:
        difference = abs(int(origin_zip[:5]) - int(destination_zip[:5]))
        miles = round(3 + difference % 22, 1)
        return simulated(
            {
                "originZip": origin_zip,
                "destinationZip": destination_zip,
                "distanceMiles": miles,
                "durationMinutes": round(12 + miles * 1.8),
                "estimatedAt": datetime.now().isoformat(),
            }
        )

    def find_available_transportation(
        self,
        destination_address: str,
        appointment_time: str,
        appointment_date: str,
        pickup_address: str,
        wheelchair_required: bool,
    ) -> dict:
        """Plan a deterministic local demo trip and select an available vehicle."""
        origin_zip = self._zip_code(pickup_address)
        destination_zip = self._zip_code(destination_address)
        difference = (
            abs(int(origin_zip) - int(destination_zip)) if origin_zip and destination_zip else 7
        )
        distance_miles = round(3 + difference % 22, 1)
        travel_minutes = round(12 + distance_miles * 1.8)
        appointment_at = datetime.strptime(
            f"{appointment_date} {appointment_time}", "%Y-%m-%d %H:%M"
        )
        pickup_at = appointment_at - timedelta(minutes=travel_minutes + 15)
        services = [
            row
            for row in self.repo.services.all()
            if row.get("status") == "active"
            and row.get("supportsMedicalTrips") is True
            and (not wheelchair_required or row.get("wheelchairAccessible") is True)
        ]
        occupied_vehicle_ids = {
            row.get("vehicleId")
            for row in self.repo.rides.all()
            if row.get("pickupDate") == appointment_date
            and row.get("pickupTime") == pickup_at.strftime("%H:%M")
            and row.get("status") in {"confirmed", "pending"}
        }
        service_ids = {row.get("transportationServiceId") for row in services}
        vehicle = next(
            (
                row
                for row in self.repo.vehicles.all()
                if row.get("status") == "available"
                and row.get("transportationServiceId") in service_ids
                and (not wheelchair_required or row.get("wheelchairAccessible") is True)
                and row.get("vehicleId") not in occupied_vehicle_ids
            ),
            None,
        )
        service = next(
            (
                row
                for row in services
                if vehicle
                and row.get("transportationServiceId") == vehicle.get("transportationServiceId")
            ),
            None,
        )
        return simulated(
            {
                "available": service is not None,
                "transportationServiceId": service.get("transportationServiceId")
                if service
                else None,
                "serviceName": service.get("serviceName") if service else None,
                "vehicleId": vehicle.get("vehicleId") if vehicle else None,
                "vehicleType": vehicle.get("vehicleType") if vehicle else None,
                "wheelchairRequired": wheelchair_required,
                "wheelchairAccessible": service.get("wheelchairAccessible") if service else None,
                "pickupAddress": pickup_address,
                "destinationAddress": destination_address,
                "appointmentDate": appointment_date,
                "appointmentTime": appointment_time,
                "pickupDate": pickup_at.strftime("%Y-%m-%d"),
                "pickupTime": pickup_at.strftime("%H:%M"),
                "estimatedDistanceMiles": distance_miles,
                "estimatedTravelMinutes": travel_minutes,
                "arrivalBufferMinutes": 15,
                "simulation": True,
            }
        )

    @staticmethod
    def _zip_code(address: str) -> str | None:
        match = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
        return match.group(1) if match else None

    def book_dummy_ride(
        self,
        senior_id: str,
        appointment_id: str,
        service_id: str,
        pickup_date: str,
        pickup_time: str,
        pickup_address: str,
        destination_address: str,
        appointment_date: str,
        appointment_time: str,
        wheelchair_required: bool,
        vehicle_id: str,
        estimated_travel_minutes: int,
        accommodation: str = "none",
        recipient_id: str | None = None,
        return_ride_required: bool = False,
    ) -> dict:
        member = self.seniors.get(senior_id)
        if not member:
            raise KeyError(f"Unknown seniorId: {senior_id}")
        if not self.seniors.get_care_recipient(member, recipient_id):
            raise PermissionError("Care recipient does not belong to this account")
        appointment = self.appointments.get(appointment_id)
        if not appointment or appointment.get("seniorId") != senior_id:
            raise ValueError("Appointment does not belong to senior")
        if appointment.get("recipientId") and appointment.get("recipientId") != recipient_id:
            raise PermissionError("Appointment belongs to a different care recipient")
        if not self.repo.services.get(service_id):
            raise KeyError(f"Unknown transportationServiceId: {service_id}")
        ride = {
            "rideId": self.repo.rides.next_id("RIDE"),
            "seniorId": senior_id,
            "requestedByUserId": senior_id,
            "recipientId": recipient_id,
            "careRecipient": self.seniors.public_care_recipient(member, recipient_id),
            "appointmentId": appointment_id,
            "transportationServiceId": service_id,
            "pickupDate": pickup_date,
            "pickupTime": pickup_time,
            "pickupAddress": pickup_address,
            "destinationAddress": destination_address,
            "appointmentDate": appointment_date,
            "appointmentTime": appointment_time,
            "wheelchairRequired": wheelchair_required,
            "vehicleId": vehicle_id,
            "estimatedTravelMinutes": estimated_travel_minutes,
            "returnRideRequired": return_ride_required,
            "mobilityAccommodation": accommodation,
            "status": "confirmed",
            "simulation": True,
        }
        self.repo.rides.create(ride)
        self.appointments.appointments.update(
            appointment_id, {"transportationRideId": ride["rideId"], "transportationRequired": True}
        )
        self.audit.record("dummy_ride_booked", str(ride["rideId"]), ride)
        return simulated(ride)

    def modify_dummy_ride(self, ride_id: str, pickup_date: str, pickup_time: str) -> dict:
        ride = self.repo.rides.update(
            ride_id, {"pickupDate": pickup_date, "pickupTime": pickup_time, "status": "confirmed"}
        )
        self.audit.record("dummy_ride_modified", ride_id, ride)
        return simulated(ride)

    def cancel_dummy_ride(self, ride_id: str) -> dict:
        ride = self.repo.rides.update(ride_id, {"status": "cancelled"})
        self.audit.record("dummy_ride_cancelled", ride_id, ride)
        return simulated(ride)
