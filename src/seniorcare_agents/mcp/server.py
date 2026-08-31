from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from seniorcare_agents.retrieval import HybridRetriever
from seniorcare_runtime.agents import MemberCaseAgent
from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import (
    AppointmentRepository,
    CaseRepository,
    HomeSupportRepository,
    MealRepository,
    MedicationRepository,
    ProviderRepository,
    SeniorRepository,
    SocialRepository,
    TransportationRepository,
)
from seniorcare_runtime.repositories.base import JsonRepository
from seniorcare_runtime.services import AuditService, RiskDetectionService, SeniorContextService
from seniorcare_runtime.tools import (
    AppointmentTools,
    EventTools,
    HomeSupportTools,
    MealTools,
    MedicationTools,
    ReminderTools,
    TransportationTools,
)

READ_ONLY = ToolAnnotations(readOnlyHint=True)
CREATE_ONLY = ToolAnnotations(destructiveHint=False, idempotentHint=False)
IDEMPOTENT_WRITE = ToolAnnotations(destructiveHint=False, idempotentHint=True)
DESTRUCTIVE_WRITE = ToolAnnotations(destructiveHint=True, idempotentHint=False)
PUBLIC_KNOWLEDGE_POLICIES: dict[str, frozenset[str]] = {
    "TransportationAgent": frozenset({"transportation"}),
    "MealsFoodAgent": frozenset({"food_meals", "benefits_financial"}),
    "SocialWellbeingAgent": frozenset({"social_wellbeing"}),
    "HomeSupportSafetyAgent": frozenset(
        {"home_support", "caregiver_support", "benefits_financial"}
    ),
    "evaluation": frozenset(
        {
            "healthcare_access",
            "transportation",
            "medication_reference",
            "discharge_support",
            "food_meals",
            "benefits_financial",
            "home_support",
            "caregiver_support",
            "social_wellbeing",
        }
    ),
}


def create_seniorcare_mcp_server(
    settings: RuntimeSettings, retriever: HybridRetriever | None = None
) -> FastMCP:
    """Build the independent MCP server over existing domain services."""
    server = FastMCP(
        "seniorcare-connect",
        instructions=(
            "All writes affect local synthetic JSON only. Never claim an external provider, "
            "pharmacy, ride company, or emergency service was contacted."
        ),
    )
    members = SeniorRepository(settings)
    appointments = AppointmentRepository(settings)
    providers = ProviderRepository(settings)
    transportation = TransportationRepository(settings)
    medications = MedicationRepository(settings)
    meals = MealRepository(settings)
    social = SocialRepository(settings)
    home = HomeSupportRepository(settings)
    cases = CaseRepository(settings)
    caregivers = JsonRepository(settings.synthetic_dir / "caregivers.json", "caregiverId")
    risks = RiskDetectionService(settings)
    audit = AuditService(settings)
    context = SeniorContextService(settings)
    member_cases = MemberCaseAgent(settings)

    @server.tool(annotations=READ_ONLY)
    def server_status() -> dict[str, Any]:
        """Return safe MCP service health and corpus information."""
        return {
            "status": "OK",
            "server": "seniorcare-connect",
            "simulation": settings.simulation_mode,
            "externalMutationsAllowed": settings.allow_external_mutations,
            "ragChunks": len(retriever.bm25.rows) if retriever else 0,
        }

    @server.tool(annotations=READ_ONLY)
    def get_member(user_id: str) -> dict[str, Any] | None:
        """Get a synthetic member by SeniorCare user ID."""
        member = members.get(user_id)
        return members.public_member(member) if member else None

    @server.tool(annotations=CREATE_ONLY)
    def register_member(
        first_name: str,
        last_name: str,
        date_of_birth: str,
        city: str | None = None,
        county: str | None = None,
        state: str = "VA",
        zip_code: str | None = None,
        care_for: str = "self",
        relationship_to_care_recipient: str = "self",
        care_recipient_first_name: str | None = None,
        care_recipient_last_name: str | None = None,
        care_recipient_date_of_birth: str | None = None,
    ) -> dict[str, Any]:
        """Register a local synthetic member after explicit form submission."""
        return member_cases.register_member(
            {
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": date_of_birth,
                "city": city,
                "county": county,
                "state": state,
                "zip_code": zip_code,
                "care_for": care_for,
                "relationship_to_care_recipient": relationship_to_care_recipient,
                "care_recipient_first_name": care_recipient_first_name,
                "care_recipient_last_name": care_recipient_last_name,
                "care_recipient_date_of_birth": care_recipient_date_of_birth,
            }
        )

    @server.tool(annotations=CREATE_ONLY)
    def add_care_recipient(
        user_id: str,
        first_name: str,
        last_name: str,
        date_of_birth: str,
        relationship_to_account_holder: str,
    ) -> dict[str, Any]:
        """Add another local synthetic care recipient to an adult account."""
        return member_cases.add_care_recipient(
            user_id,
            {
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": date_of_birth,
                "relationship_to_account_holder": relationship_to_account_holder,
            },
        )

    @server.tool(annotations=READ_ONLY)
    def get_member_context(user_id: str) -> dict[str, Any]:
        """Get member, caregiver, case, appointment, ride, and task context."""
        return context.get_context(user_id)

    @server.tool(annotations=READ_ONLY)
    def validate_action_context(
        user_id: str, case_id: str | None = None, recipient_id: str | None = None
    ) -> dict[str, Any]:
        """Validate member existence and optional case ownership for an approved action."""
        member = members.get(user_id)
        case = cases.get_case(case_id) if case_id else None
        resolved_recipient = members.get_care_recipient(member, recipient_id) if member else None
        resolved_recipient_id = (
            resolved_recipient.get("recipientId") if resolved_recipient else None
        )
        return {
            "memberExists": member is not None,
            "caseMatches": case_id is None
            or bool(
                case
                and case.get("seniorId") == user_id
                and (
                    not case.get("recipientId") or case.get("recipientId") == resolved_recipient_id
                )
            ),
            "accountRole": member.get("accountRole", "self_care") if member else None,
            "recipientMatches": bool(resolved_recipient),
            "careRecipient": (
                members.public_care_recipient(member, recipient_id) if member else None
            ),
        }

    @server.tool(annotations=READ_ONLY)
    def list_appointments(user_id: str) -> list[dict[str, Any]]:
        """List a member's simulated appointments."""
        return appointments.for_senior(user_id)

    @server.tool(annotations=READ_ONLY)
    def list_referrals(user_id: str) -> list[dict[str, Any]]:
        """List a member's simulated referrals."""
        return appointments.referrals_for(user_id)

    @server.tool(annotations=READ_ONLY)
    def search_providers(
        specialty: str | None = None,
        county: str | None = None,
        limit: int = 10,
        include_public: bool = True,
    ) -> list[dict[str, Any]]:
        """Search local synthetic and public provider records."""
        return providers.search(
            specialty=specialty,
            county=county,
            limit=limit,
            include_public=include_public,
        )

    @server.tool(annotations=READ_ONLY)
    def get_provider(provider_id: str) -> dict[str, Any] | None:
        """Get one local simulated provider for approval-detail enrichment."""
        return providers.get(provider_id)

    @server.tool(annotations=READ_ONLY)
    def list_available_slots(provider_id: str | None = None) -> list[dict[str, Any]]:
        """List local simulated provider availability."""
        return appointments.available_slots(provider_id)

    @server.tool(annotations=READ_ONLY)
    def get_available_slot(availability_id: str) -> dict[str, Any] | None:
        """Get one local simulated provider-availability record."""
        return appointments.availability.get(availability_id)

    @server.tool(annotations=READ_ONLY)
    def list_rides(user_id: str) -> list[dict[str, Any]]:
        """List a member's simulated rides."""
        return transportation.rides_for(user_id)

    @server.tool(annotations=READ_ONLY)
    def search_transportation_services(
        county: str | None = None, wheelchair_accessible: bool | None = None
    ) -> list[dict[str, Any]]:
        """Search local simulated transportation services."""
        return transportation.services_for(county, wheelchair_accessible)

    @server.tool(annotations=READ_ONLY)
    def find_available_transportation(
        destination_address: str,
        appointment_time: str,
        appointment_date: str,
        pickup_address: str,
        wheelchair_required: bool,
    ) -> dict[str, Any]:
        """Estimate a local demo trip and return an available simulated vehicle."""
        return transportation_tools.find_available_transportation(
            destination_address,
            appointment_time,
            appointment_date,
            pickup_address,
            wheelchair_required,
        )

    @server.tool(annotations=READ_ONLY)
    def list_medications(user_id: str) -> list[dict[str, Any]]:
        """List a member's synthetic medication coordination records."""
        return medications.for_senior(user_id)

    @server.tool(annotations=READ_ONLY)
    def list_refills(user_id: str) -> list[dict[str, Any]]:
        """List a member's simulated refill requests."""
        return medications.refills_for(user_id)

    @server.tool(annotations=READ_ONLY)
    def search_medication_references(name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search structured public openFDA medication references by brand, generic, or substance."""
        return medications.reference(name, min(max(limit, 1), 50))

    @server.tool(annotations=READ_ONLY)
    def search_meal_services(county: str | None = None) -> list[dict[str, Any]]:
        """Search local simulated meal services."""
        return meals.services_for(county)

    @server.tool(annotations=READ_ONLY)
    def search_social_activities(county: str | None = None) -> list[dict[str, Any]]:
        """Search local simulated social activities."""
        return social.search(county)

    @server.tool(annotations=READ_ONLY)
    def list_home_support_requests(user_id: str) -> list[dict[str, Any]]:
        """List a member's simulated home-support requests."""
        return home.for_senior(user_id)

    @server.tool(annotations=READ_ONLY)
    def list_cases(user_id: str) -> list[dict[str, Any]]:
        """List cases belonging to a member."""
        return cases.cases_for(user_id)

    @server.tool(annotations=IDEMPOTENT_WRITE)
    def close_due_cases(user_id: str) -> dict[str, Any]:
        """Close local active cases whose appointment or explicit target date has passed."""
        return member_cases.close_due_cases(user_id)

    @server.tool(annotations=READ_ONLY)
    def get_case_related_records(user_id: str, entity_ids: list[str]) -> list[dict[str, Any]]:
        """Resolve case-linked records into member-owned confirmation details."""
        resolved: list[dict[str, Any]] = []
        for entity_id in entity_ids:
            record: dict[str, Any] | None = None
            record_type = "related record"
            details: dict[str, Any] = {}
            if entity_id.startswith("APT"):
                record = appointments.get(entity_id)
                provider_id = record.get("providerId") if record else None
                provider = providers.get(str(provider_id)) if provider_id else None
                record_type = "appointment"
                details = {
                    "Doctor": provider.get("providerName") if provider else None,
                    "Specialty": provider.get("specialty") if provider else None,
                    "Facility": provider.get("facilityName") if provider else None,
                    "Location": ", ".join(
                        str(value)
                        for value in (
                            provider.get("city") if provider else None,
                            provider.get("state") if provider else None,
                            provider.get("zipCode") if provider else None,
                        )
                        if value
                    ),
                    "Date": record.get("appointmentDate") if record else None,
                    "Time": record.get("appointmentTime") if record else None,
                    "Reason": record.get("reason") if record else None,
                }
            elif entity_id.startswith("RIDE"):
                record = transportation.rides.get(entity_id)
                service_id = record.get("transportationServiceId") if record else None
                service = transportation.services.get(str(service_id)) if service_id else None
                record_type = "transportation"
                details = {
                    "Transportation service": service.get("serviceName") if service else None,
                    "Service area": service.get("serviceArea") if service else None,
                    "Pickup date": record.get("pickupDate") if record else None,
                    "Pickup time": record.get("pickupTime") if record else None,
                    "Pickup address": record.get("pickupAddress") if record else None,
                    "Destination": record.get("destinationAddress") if record else None,
                    "Appointment date": record.get("appointmentDate") if record else None,
                    "Appointment time": record.get("appointmentTime") if record else None,
                    "Vehicle ID": record.get("vehicleId") if record else None,
                    "Estimated travel minutes": record.get("estimatedTravelMinutes")
                    if record
                    else None,
                    "Wheelchair required": record.get("wheelchairRequired") if record else None,
                    "Related appointment": record.get("appointmentId") if record else None,
                    "Return ride required": record.get("returnRideRequired") if record else None,
                    "Mobility accommodation": record.get("mobilityAccommodation")
                    if record
                    else None,
                    "Contact": service.get("bookingPhone") if service else None,
                }
            elif entity_id.startswith("RFL"):
                record = medications.refills.get(entity_id)
                medication_id = record.get("medicationId") if record else None
                medication = (
                    medications.medications.get(str(medication_id)) if medication_id else None
                )
                record_type = "medication refill"
                details = {
                    "Medication": medication.get("medicationName") if medication else None,
                    "Strength": medication.get("strength") if medication else None,
                    "Pharmacy ID": record.get("pharmacyId") if record else None,
                    "Request date": record.get("requestDate") if record else None,
                    "Pickup or delivery": record.get("pickupOrDelivery") if record else None,
                    "Estimated ready date": record.get("estimatedReadyDate") if record else None,
                }
            elif entity_id.startswith("MEN"):
                record = meals.enrollments.get(entity_id)
                meal_service_id = record.get("mealServiceId") if record else None
                service = meals.services.get(str(meal_service_id)) if meal_service_id else None
                record_type = "meal service"
                details = {
                    "Meal service": service.get("serviceName") if service else None,
                    "Service type": service.get("serviceType") if service else None,
                    "Service area": service.get("serviceArea") if service else None,
                    "Delivery days": service.get("deliveryDays") if service else None,
                    "Request date": record.get("requestDate") if record else None,
                }
            elif entity_id.startswith("REG"):
                record = social.registrations.get(entity_id)
                activity_id = record.get("activityId") if record else None
                activity = social.activities.get(str(activity_id)) if activity_id else None
                record_type = "social activity"
                details = {
                    "Activity": activity.get("activityName") if activity else None,
                    "Activity type": activity.get("activityType") if activity else None,
                    "Location": activity.get("locationName") if activity else None,
                    "City": activity.get("city") if activity else None,
                    "Date": activity.get("activityDate") if activity else None,
                    "Start time": activity.get("startTime") if activity else None,
                }
            elif entity_id.startswith("HOME"):
                record = home.requests.get(entity_id)
                record_type = "home support"
                details = {
                    "Request type": record.get("requestType") if record else None,
                    "Priority": record.get("priority") if record else None,
                    "Request date": record.get("requestDate") if record else None,
                    "Assigned resource": record.get("assignedResourceId") if record else None,
                    "Notes": record.get("notes") if record else None,
                }
            elif entity_id.startswith("DIS"):
                record = cases.discharges.get(entity_id)
                record_type = "hospital discharge follow-up"
                details = {
                    "Hospital": record.get("hospitalName") if record else None,
                    "Discharge date": record.get("dischargeDate") if record else None,
                    "Task": record.get("taskType") if record else None,
                    "Due date": record.get("dueDate") if record else None,
                    "Related appointment": record.get("relatedAppointmentId") if record else None,
                    "Notes": record.get("notes") if record else None,
                }
            elif entity_id.startswith("BEN"):
                record = cases.benefits.get(entity_id)
                record_type = "benefits coordination"
                details = {
                    "Benefit": record.get("benefitType") if record else None,
                    "Application date": record.get("applicationDate") if record else None,
                    "Missing documents": record.get("missingDocuments") if record else None,
                    "Next action": record.get("nextAction") if record else None,
                    "Last updated": record.get("lastUpdated") if record else None,
                }
            elif entity_id.startswith("CG"):
                record = caregivers.get(entity_id)
                record_type = "caregiver coordination"
                details = {
                    "Caregiver": (
                        f"{record.get('firstName', '')} {record.get('lastName', '')}".strip()
                        if record
                        else None
                    ),
                    "Relationship": record.get("relationship") if record else None,
                    "Preferred contact": record.get("preferredContactMethod") if record else None,
                    "Phone": record.get("phone") if record else None,
                    "Email": record.get("email") if record else None,
                    "Scheduling authorized": (
                        record.get("authorizedForScheduling") if record else None
                    ),
                    "Transportation authorized": (
                        record.get("authorizedForTransportation") if record else None
                    ),
                }
            elif entity_id.startswith("REF"):
                record = appointments.referrals.get(entity_id)
                record_type = "healthcare referral"
                details = {
                    "Provider": record.get("providerId") if record else None,
                    "Specialty": record.get("specialty") if record else None,
                    "Created date": record.get("createdDate") if record else None,
                    "Notes": record.get("notes") if record else None,
                }
            if not record or record.get("seniorId") != user_id:
                continue
            resolved.append(
                {
                    "recordType": record_type,
                    "trackingId": entity_id,
                    "status": record.get("status", "recorded"),
                    "details": {
                        key: value for key, value in details.items() if value not in (None, "", [])
                    },
                }
            )
        return resolved

    @server.tool(annotations=READ_ONLY)
    def get_case(user_id: str, case_id: str) -> dict[str, Any]:
        """Get one member-owned coordination case."""
        return member_cases.get_case(user_id, case_id)

    @server.tool(annotations=IDEMPOTENT_WRITE)
    def update_case_status(
        user_id: str, case_id: str, status: str, status_note: str
    ) -> dict[str, Any]:
        """Update a local case after an authorized request."""
        allowed = {
            "open",
            "in_progress",
            "blocked",
            "waiting_for_user",
            "resolved",
            "closed",
            "cancelled",
        }
        if status not in allowed:
            raise ValueError("Invalid case status")
        return member_cases.update_case_status(user_id, case_id, status, status_note)  # type: ignore[arg-type]

    @server.tool(annotations=IDEMPOTENT_WRITE)
    def link_case_entity(user_id: str, case_id: str, entity_id: str) -> dict[str, Any]:
        """Link an approved simulated entity to its member-owned case."""
        return member_cases.add_related_entity(user_id, case_id, entity_id)

    @server.tool(annotations=READ_ONLY)
    def evaluate_risks(user_id: str) -> list[dict[str, Any]]:
        """Run deterministic coordination-risk rules for a member."""
        return risks.evaluate(user_id)

    @server.tool(annotations=READ_ONLY)
    def list_audit_events(
        user_id: str, case_id: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """List privacy-filtered audit events for a member and optional case."""
        return audit.events_for(user_id, case_id, limit)

    @server.tool(annotations=READ_ONLY)
    async def search_public_knowledge(
        query: str,
        categories: list[str],
        state: str = "Virginia",
        county: str | None = None,
        agent_name: str = "MCPAgent",
    ) -> list[dict[str, Any]]:
        """Run hybrid BM25/vector/reranked retrieval over public SeniorCare knowledge."""
        if retriever is None:
            return []
        allowed = PUBLIC_KNOWLEDGE_POLICIES.get(agent_name)
        if allowed is None:
            raise PermissionError(f"Unknown RAG caller policy: {agent_name}")
        requested = set(categories)
        if not requested or not requested <= allowed:
            raise PermissionError(
                f"RAG categories {sorted(requested)} are not allowed for {agent_name}"
            )
        chunks = await retriever.retrieve(
            query,
            categories,
            {"state": state, "county": county},
            agent=agent_name,
        )
        return [chunk.model_dump(mode="json") for chunk in chunks]

    @server.tool(annotations=READ_ONLY)
    async def search_healthcare_knowledge(
        query: str,
        state: str = "Virginia",
        county: str | None = None,
        agent_name: str = "HealthcareAccessAgent",
    ) -> list[dict[str, Any]]:
        """Search only source-attributed healthcare-access guidance using hybrid RAG."""
        if retriever is None:
            return []
        chunks = await retriever.retrieve(
            query,
            ["healthcare_access"],
            {"state": state, "county": county},
            agent=agent_name,
        )
        return [chunk.model_dump(mode="json") for chunk in chunks]

    @server.tool(annotations=READ_ONLY)
    async def search_medication_knowledge(
        query: str,
        state: str = "Virginia",
        county: str | None = None,
        agent_name: str = "MedicationPharmacyAgent",
    ) -> list[dict[str, Any]]:
        """Search only source-attributed medication guidance using hybrid RAG."""
        if retriever is None:
            return []
        chunks = await retriever.retrieve(
            query,
            ["medication_reference"],
            {"state": state, "county": county},
            agent=agent_name,
        )
        return [chunk.model_dump(mode="json") for chunk in chunks]

    @server.tool(annotations=READ_ONLY)
    def list_knowledge_chunk_ids(categories: list[str]) -> list[str]:
        """List canonical chunk IDs for retrieval evaluation categories."""
        if retriever is None:
            return []
        selected = set(categories)
        return [
            str(row["chunk_id"]) for row in retriever.bm25.rows if row.get("category") in selected
        ]

    @server.tool(annotations=CREATE_ONLY)
    def create_case(user_id: str, request: dict[str, Any]) -> dict[str, Any]:
        """Create an approved local simulation-only coordination case."""
        return member_cases.create_case(user_id, request)

    appointment_tools = AppointmentTools(settings)
    transportation_tools = TransportationTools(settings)
    medication_tools = MedicationTools(settings)
    meal_tools = MealTools(settings)
    reminder_tools = ReminderTools(settings)
    event_tools = EventTools(settings)
    home_tools = HomeSupportTools(settings)

    @server.tool(annotations=CREATE_ONLY)
    def book_dummy_appointment(
        senior_id: str,
        provider_id: str,
        availability_id: str,
        reason: str,
        transportation_required: bool = False,
        recipient_id: str | None = None,
    ) -> dict[str, Any]:
        """Book an approved local simulated appointment; no provider is contacted."""
        return appointment_tools.book_dummy_appointment(
            senior_id,
            provider_id,
            availability_id,
            reason,
            transportation_required,
            recipient_id,
        )

    @server.tool(annotations=DESTRUCTIVE_WRITE)
    def cancel_dummy_appointment(appointment_id: str) -> dict[str, Any]:
        """Cancel an approved local simulated appointment."""
        return appointment_tools.cancel_dummy_appointment(appointment_id)

    @server.tool(annotations=CREATE_ONLY)
    def reschedule_dummy_appointment(appointment_id: str, availability_id: str) -> dict[str, Any]:
        """Reschedule an approved local simulated appointment."""
        return appointment_tools.reschedule_dummy_appointment(appointment_id, availability_id)

    @server.tool(annotations=CREATE_ONLY)
    def book_dummy_ride(
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
        accommodation: str,
        recipient_id: str | None = None,
        return_ride_required: bool = False,
    ) -> dict[str, Any]:
        """Book an approved local simulated ride; no transit provider is contacted."""
        return transportation_tools.book_dummy_ride(
            senior_id,
            appointment_id,
            service_id,
            pickup_date,
            pickup_time,
            pickup_address,
            destination_address,
            appointment_date,
            appointment_time,
            wheelchair_required,
            vehicle_id,
            estimated_travel_minutes,
            accommodation,
            recipient_id,
            return_ride_required,
        )

    @server.tool(annotations=CREATE_ONLY)
    def modify_dummy_ride(ride_id: str, pickup_date: str, pickup_time: str) -> dict[str, Any]:
        """Modify an approved local simulated ride."""
        return transportation_tools.modify_dummy_ride(ride_id, pickup_date, pickup_time)

    @server.tool(annotations=DESTRUCTIVE_WRITE)
    def cancel_dummy_ride(ride_id: str) -> dict[str, Any]:
        """Cancel an approved local simulated ride."""
        return transportation_tools.cancel_dummy_ride(ride_id)

    @server.tool(annotations=CREATE_ONLY)
    def request_dummy_refill(
        senior_id: str,
        medication_id: str,
        method: str = "pickup",
        recipient_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an approved simulated refill request; no pharmacy is contacted."""
        return medication_tools.request_dummy_refill(senior_id, medication_id, method, recipient_id)

    @server.tool(annotations=CREATE_ONLY)
    def enroll_dummy_meal_service(
        senior_id: str, meal_service_id: str, recipient_id: str | None = None
    ) -> dict[str, Any]:
        """Create an approved simulated meal enrollment."""
        return meal_tools.enroll_dummy_meal_service(senior_id, meal_service_id, recipient_id)

    @server.tool(annotations=CREATE_ONLY)
    def register_dummy_event(
        senior_id: str, activity_id: str, recipient_id: str | None = None
    ) -> dict[str, Any]:
        """Create an approved simulated activity registration."""
        return event_tools.register_dummy_event(senior_id, activity_id, recipient_id)

    @server.tool(annotations=CREATE_ONLY)
    def request_dummy_home_support(
        senior_id: str,
        request_type: str,
        priority: str,
        notes: str,
        recipient_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an approved simulated home-support request."""
        return home_tools.request_dummy_home_support(
            senior_id, request_type, priority, notes, recipient_id
        )

    @server.tool(annotations=CREATE_ONLY)
    def schedule_dummy_reminder(
        senior_id: str,
        reminder_type: str,
        related_entity_id: str,
        reminder_date: str,
        reminder_time: str,
        message: str,
        caregiver_id: str | None = None,
        delivery_method: str = "app",
    ) -> dict[str, Any]:
        """Create an approved local simulated reminder."""
        return reminder_tools.schedule_dummy_reminder(
            senior_id,
            reminder_type,
            related_entity_id,
            reminder_date,
            reminder_time,
            message,
            caregiver_id,
            delivery_method,
        )

    return server
