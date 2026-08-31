import json
import shutil
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from seniorcare_runtime.agents import MemberCaseAgent
from seniorcare_runtime.agents.member_case_agent import MemberRegistration
from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import ProviderRepository, SeniorRepository
from seniorcare_runtime.services import RiskDetectionService, SeniorContextService
from seniorcare_runtime.tools import (
    AppointmentTools,
    TransportationTools,
)


def runtime_settings(tmp_path: Path) -> RuntimeSettings:
    source = Path(__file__).parents[2] / "data/synthetic-data"
    destination = tmp_path / "data/synthetic-data"
    shutil.copytree(source, destination)
    normalized = tmp_path / "data/normalized"
    normalized.mkdir(parents=True)
    (normalized / "providers.jsonl").write_text("", encoding="utf-8")
    (normalized / "medications.jsonl").write_text("", encoding="utf-8")
    return RuntimeSettings(
        project_root=tmp_path, simulation_mode=True, allow_external_mutations=False
    )


def test_repositories_resolve_senior_and_provider(tmp_path: Path):
    settings = runtime_settings(tmp_path)
    senior = SeniorRepository(settings).get("SEN1001")
    providers = ProviderRepository(settings).search(
        specialty="Primary Care", county="Richmond City"
    )
    assert senior and senior["firstName"] == "Robert"
    assert providers and providers[0]["providerId"] == "PRV1001"
    assert providers[0]["addressLine1"]


def test_synthetic_runtime_data_covers_operational_agent_stores():
    synthetic_dir = Path(__file__).parents[2] / "data/synthetic-data"
    required_files = {
        "seniors.json",
        "caregivers.json",
        "providers.json",
        "provider_availability.json",
        "appointments.json",
        "referrals.json",
        "medications.json",
        "pharmacy_refills.json",
        "transportation_services.json",
        "transportation_vehicles.json",
        "rides.json",
        "discharge_tasks.json",
        "meal_services.json",
        "meal_enrollments.json",
        "benefit_applications.json",
        "home_support_requests.json",
        "social_activities.json",
        "activity_registrations.json",
        "case_tasks.json",
        "reminders.json",
        "cases.json",
    }
    datasets = {
        name: json.loads((synthetic_dir / name).read_text(encoding="utf-8"))
        for name in required_files
    }
    assert all(isinstance(records, list) and records for records in datasets.values())

    provider_ids = {row["providerId"] for row in datasets["providers.json"]}
    service_ids = {
        row["transportationServiceId"] for row in datasets["transportation_services.json"]
    }
    meal_service_ids = {row["mealServiceId"] for row in datasets["meal_services.json"]}
    activity_ids = {row["activityId"] for row in datasets["social_activities.json"]}

    assert all(row.get("addressLine1") for row in datasets["providers.json"])
    assert all(row["providerId"] in provider_ids for row in datasets["provider_availability.json"])
    assert all(
        row["transportationServiceId"] in service_ids
        for row in datasets["transportation_vehicles.json"]
    )
    assert all(
        row["mealServiceId"] in meal_service_ids for row in datasets["meal_enrollments.json"]
    )
    assert all(row["activityId"] in activity_ids for row in datasets["activity_registrations.json"])


def test_dummy_appointment_changes_only_temp_synthetic_data(tmp_path: Path):
    settings = runtime_settings(tmp_path)
    tool = AppointmentTools(settings)
    result = tool.book_dummy_appointment("SEN1001", "PRV1001", "AVL1001", "study test")
    assert result["simulation"] is True
    assert result["externalActionPerformed"] is False
    assert result["data"]["status"] == "scheduled"
    assert tool.repo.availability.get("AVL1001")["status"] == "booked"
    assert settings.audit_path.exists()


def test_runtime_refuses_external_mutation_mode(tmp_path: Path):
    settings = RuntimeSettings(
        project_root=tmp_path, simulation_mode=False, allow_external_mutations=True
    )
    with pytest.raises(RuntimeError, match="SIMULATION_MODE"):
        AppointmentTools(settings)


def test_context_identity_and_deterministic_risk(tmp_path: Path):
    settings = runtime_settings(tmp_path)
    context = SeniorContextService(settings).get_context("SEN1003")
    identity = SeniorRepository(settings).search(name="James Smith")
    risks = RiskDetectionService(settings).evaluate("SEN1003", as_of=date(2026, 9, 10))
    assert context["appointments"]
    assert identity[0]["seniorId"] == "SEN1003"
    assert {risk["code"] for risk in risks} >= {
        "OVERDUE_CASE_TASK",
        "REFILL_AUTHORIZATION_NEEDED",
        "APPOINTMENT_RIDE_NOT_CONFIRMED",
    }


def test_phase_one_transportation_tool_writes_local_record(tmp_path: Path):
    settings = runtime_settings(tmp_path)
    ride = TransportationTools(settings).book_dummy_ride(
        "SEN1001",
        "APT1001",
        "TRN1001",
        "2026-09-02",
        "08:15",
        "123 Main Street, Richmond, VA 23220",
        "Central Virginia Primary Care Center, Richmond, VA 23220",
        "2026-09-02",
        "09:00",
        False,
        "VEH-TRN1001-01",
        30,
        return_ride_required=True,
    )
    assert ride["simulation"] is True
    assert ride["externalActionPerformed"] is False
    assert ride["data"]["returnRideRequired"] is True
    assert ride["data"]["pickupAddress"] == "123 Main Street, Richmond, VA 23220"
    assert ride["data"]["vehicleId"] == "VEH-TRN1001-01"


def test_transportation_planner_uses_trip_schema_and_vehicle_availability(tmp_path: Path):
    settings = runtime_settings(tmp_path)
    result = TransportationTools(settings).find_available_transportation(
        destination_address="Orthopedics Center, Midlothian, VA 23113",
        appointment_time="11:30",
        appointment_date="2026-09-12",
        pickup_address="123 Main Street, Richmond, VA 23220",
        wheelchair_required=True,
    )

    plan = result["data"]
    assert plan["available"] is True
    assert plan["wheelchairAccessible"] is True
    assert plan["vehicleId"].startswith("VEH")
    assert plan["vehicleType"] in {"wheelchair_van", "accessible_minibus"}
    assert plan["pickupTime"] < plan["appointmentTime"]
    assert plan["estimatedTravelMinutes"] > 0


def test_member_case_agent_onboards_returns_and_tracks_cases(tmp_path: Path):
    settings = runtime_settings(tmp_path)
    agent = MemberCaseAgent(settings)
    welcome = agent.start()
    registered = agent.register_member(
        {
            "first_name": "Grace",
            "last_name": "Hopper",
            "date_of_birth": "1936-12-09",
            "city": "Richmond",
            "county": "Richmond City",
        }
    )
    user_id = registered["data"]["userId"]
    created = agent.create_case(
        user_id,
        {
            "case_type": "appointment_coordination",
            "title": "Arrange cardiology visit",
            "description": "Schedule a local study appointment",
            "priority": "high",
            "target_date": "2026-09-15",
        },
    )
    case_id = created["data"]["caseId"]
    agent.add_related_entity(user_id, case_id, "APT1999")
    agent.update_case_status(user_id, case_id, "in_progress", "Dummy appointment requested")
    returning = agent.start(user_id)
    assert welcome["data"]["newMemberRequiredFields"] == [
        "firstName",
        "lastName",
        "dateOfBirth",
        "careFor",
    ]
    assert user_id.startswith("SEN") and case_id.startswith("CASE")
    assert "dateOfBirth" not in returning["data"]["member"]
    assert returning["data"]["cases"][0]["status"] == "in_progress"
    assert "APT1999" in returning["data"]["cases"][0]["relatedEntityIds"]


def test_member_registration_is_idempotent_for_same_identity(tmp_path: Path):
    agent = MemberCaseAgent(runtime_settings(tmp_path))
    request = {"first_name": "New", "last_name": "Member", "date_of_birth": "1940-01-02"}
    first = agent.register_member(request)
    second = agent.register_member(request)
    assert first["data"]["created"] is True
    assert second["data"]["created"] is False
    assert first["data"]["userId"] == second["data"]["member"]["seniorId"]


def test_account_holder_must_be_at_least_21():
    today = date.today()
    under_21 = date(today.year - 20, today.month, min(today.day, 28))
    with pytest.raises(ValidationError, match="at least 21"):
        MemberRegistration(
            first_name="Young",
            last_name="Account",
            date_of_birth=under_21,
        )


def test_family_representative_tracks_parent_as_care_recipient(tmp_path: Path):
    agent = MemberCaseAgent(runtime_settings(tmp_path))
    registered = agent.register_member(
        {
            "first_name": "Adult",
            "last_name": "Child",
            "date_of_birth": "1980-05-12",
            "care_for": "family_member",
            "relationship_to_care_recipient": "mother",
            "care_recipient_first_name": "Mary",
            "care_recipient_last_name": "Child",
            "care_recipient_date_of_birth": "1945-03-10",
        }
    )
    user_id = registered["data"]["userId"]
    member = registered["data"]["member"]
    created = agent.create_case(
        user_id,
        {
            "case_type": "appointment_coordination",
            "title": "Knee-pain appointment",
            "description": "Book a simulated appointment for my mother",
        },
    )
    assert member["accountRole"] == "family_representative"
    assert member["careRecipient"]["firstName"] == "Mary"
    assert member["careRecipient"]["relationshipToAccountHolder"] == "mother"
    assert "dateOfBirth" not in member["careRecipient"]
    assert created["data"]["requestedByUserId"] == user_id
    assert created["data"]["careRecipient"]["firstName"] == "Mary"


def test_synthetic_member_dataset_satisfies_adult_and_recipient_contract():
    rows = SeniorRepository(RuntimeSettings(project_root=Path(__file__).parents[2])).seniors.all()
    assert rows
    assert all(int(row["age"]) >= 21 for row in rows)
    representative = next(row for row in rows if row["seniorId"] == "SEN1021")
    assert representative["accountRole"] == "family_representative"
    assert representative["careRecipient"]["relationshipToAccountHolder"] == "father"


def test_account_supports_multiple_owned_care_recipients_and_recipient_cases(tmp_path: Path):
    agent = MemberCaseAgent(runtime_settings(tmp_path))
    registered = agent.register_member(
        {
            "first_name": "Jordan",
            "last_name": "Lee",
            "date_of_birth": "1980-01-01",
        }
    )
    user_id = registered["data"]["userId"]
    self_id = registered["data"]["member"]["careRecipients"][0]["recipientId"]
    father = agent.add_care_recipient(
        user_id,
        {
            "first_name": "Robert",
            "last_name": "Lee",
            "date_of_birth": "1942-04-02",
            "relationship_to_account_holder": "father",
        },
    )["data"]
    mother = agent.add_care_recipient(
        user_id,
        {
            "first_name": "Maria",
            "last_name": "Lee",
            "date_of_birth": "1945-06-03",
            "relationship_to_account_holder": "mother",
        },
    )["data"]
    member = agent.start(user_id)["data"]["member"]
    assert {row["recipientId"] for row in member["careRecipients"]} == {
        self_id,
        father["recipientId"],
        mother["recipientId"],
    }
    assert all("dateOfBirth" not in row for row in member["careRecipients"])
    father_case = agent.create_case(
        user_id,
        {
            "recipient_id": father["recipientId"],
            "case_type": "appointment_coordination",
            "title": "Father appointment",
            "description": "Local study request",
        },
    )["data"]
    assert father_case["recipientId"] == father["recipientId"]
    assert father_case["careRecipient"]["firstName"] == "Robert"
    with pytest.raises(PermissionError):
        agent.create_case(
            user_id,
            {
                "recipient_id": "REC-OTHER-ACCOUNT",
                "case_type": "appointment_coordination",
                "title": "Invalid recipient",
                "description": "Must be rejected",
            },
        )
