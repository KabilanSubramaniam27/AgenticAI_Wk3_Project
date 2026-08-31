from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories import AppointmentRepository, CaseRepository, SeniorRepository
from seniorcare_runtime.services.audit_service import AuditService
from seniorcare_runtime.tools.common import simulated


def _adult_cutoff(today: date) -> date:
    try:
        return today.replace(year=today.year - 21)
    except ValueError:
        return today.replace(year=today.year - 21, day=28)


class MemberRegistration(BaseModel):
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    date_of_birth: date
    city: str | None = None
    county: str | None = None
    state: str = "VA"
    zip_code: str | None = None
    care_for: Literal["self", "family_member"] = "self"
    relationship_to_care_recipient: Literal[
        "self", "father", "mother", "parent", "spouse", "family_member", "care_recipient"
    ] = "self"
    care_recipient_first_name: str | None = Field(default=None, max_length=80)
    care_recipient_last_name: str | None = Field(default=None, max_length=80)
    care_recipient_date_of_birth: date | None = None

    @field_validator("date_of_birth")
    @classmethod
    def account_holder_must_be_adult(cls, value: date) -> date:
        if value > _adult_cutoff(date.today()):
            raise ValueError("account holder must be at least 21 years old")
        return value

    @field_validator("care_recipient_date_of_birth")
    @classmethod
    def care_recipient_birth_date_must_be_past(cls, value: date | None) -> date | None:
        if value is not None and value >= date.today():
            raise ValueError("care recipient date of birth must be in the past")
        return value

    @model_validator(mode="after")
    def family_recipient_is_complete(self) -> "MemberRegistration":
        if self.care_for == "self":
            self.relationship_to_care_recipient = "self"
            return self
        if self.relationship_to_care_recipient == "self":
            raise ValueError("select the account holder's relationship to the care recipient")
        if not all(
            (
                self.care_recipient_first_name,
                self.care_recipient_last_name,
                self.care_recipient_date_of_birth,
            )
        ):
            raise ValueError("care recipient first name, last name, and date of birth are required")
        return self


class NewCaseRequest(BaseModel):
    recipient_id: str | None = Field(default=None, min_length=1, max_length=80)
    case_type: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=4000)
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    related_entity_ids: list[str] = Field(default_factory=list)
    assigned_caregiver_id: str | None = None
    target_date: date | None = None


class CareRecipientRegistration(BaseModel):
    first_name: str = Field(min_length=1, max_length=80)
    last_name: str = Field(min_length=1, max_length=80)
    date_of_birth: date
    relationship_to_account_holder: Literal[
        "father", "mother", "parent", "spouse", "family_member", "care_recipient"
    ]

    @field_validator("date_of_birth")
    @classmethod
    def birth_date_must_be_past(cls, value: date) -> date:
        if value >= date.today():
            raise ValueError("care recipient date of birth must be in the past")
        return value


class MemberCaseAgent:
    """Conversation-facing local agent for synthetic member and case tracking."""

    def __init__(self, settings: RuntimeSettings):
        settings.require_simulation()
        self.members = SeniorRepository(settings)
        self.cases = CaseRepository(settings)
        self.appointments = AppointmentRepository(settings)
        self.audit = AuditService(settings)

    def start(self, user_id: str | None = None) -> dict:
        if not user_id:
            return simulated(
                {
                    "recognized": False,
                    "message": "Welcome. Are you a new or returning adult account holder?",
                    "newMemberRequiredFields": [
                        "firstName",
                        "lastName",
                        "dateOfBirth",
                        "careFor",
                    ],
                    "familyRepresentativeRequiredFields": [
                        "relationshipToCareRecipient",
                        "careRecipientFirstName",
                        "careRecipientLastName",
                        "careRecipientDateOfBirth",
                    ],
                    "returningMemberRequiredFields": ["userId"],
                }
            )
        member = self.members.get(user_id)
        if not member:
            return simulated(
                {
                    "recognized": False,
                    "userId": user_id,
                    "message": "User ID was not found. Please retry or register as a new member.",
                }
            )
        return simulated(
            {
                "recognized": True,
                "member": self._public_member(member),
                "cases": self.cases.cases_for(user_id),
            }
        )

    def register_member(self, registration: MemberRegistration | dict) -> dict:
        request = (
            registration
            if isinstance(registration, MemberRegistration)
            else MemberRegistration.model_validate(registration)
        )
        dob = request.date_of_birth.isoformat()
        existing = self.members.find_identity(request.first_name, request.last_name, dob)
        if existing:
            return simulated(
                {
                    "created": False,
                    "message": "Member already exists.",
                    "member": self._public_member(existing),
                    "cases": self.cases.cases_for(existing["seniorId"]),
                }
            )
        today = date.today()
        age = (
            today.year
            - request.date_of_birth.year
            - ((today.month, today.day) < (request.date_of_birth.month, request.date_of_birth.day))
        )
        recipient_dob = request.care_recipient_date_of_birth or request.date_of_birth
        recipient_age = (
            today.year
            - recipient_dob.year
            - ((today.month, today.day) < (recipient_dob.month, recipient_dob.day))
        )
        recipient = {
            "recipientId": None,
            "firstName": (
                request.first_name
                if request.care_for == "self"
                else request.care_recipient_first_name
            ),
            "lastName": (
                request.last_name
                if request.care_for == "self"
                else request.care_recipient_last_name
            ),
            "dateOfBirth": recipient_dob.isoformat(),
            "age": recipient_age,
            "relationshipToAccountHolder": request.relationship_to_care_recipient,
            "isAccountHolder": request.care_for == "self",
        }
        member = self.members.create(
            {
                "firstName": request.first_name.strip(),
                "lastName": request.last_name.strip(),
                "dateOfBirth": dob,
                "age": age,
                "gender": None,
                "city": request.city,
                "county": request.county,
                "state": request.state,
                "zipCode": request.zip_code,
                "mobilityNeeds": None,
                "livesAlone": None,
                "transportationAccess": None,
                "primaryCareProviderId": None,
                "primaryCaregiverId": None,
                "caseStatus": "active",
                "accountRole": (
                    "self_care" if request.care_for == "self" else "family_representative"
                ),
                "careRecipient": recipient,
                "careRecipients": [recipient],
                "simulation": True,
            }
        )
        recipient["recipientId"] = (
            member["seniorId"]
            if recipient["isAccountHolder"]
            else f"REC-{member['seniorId'][3:]}-1"
        )
        member = self.members.seniors.update(
            member["seniorId"], {"careRecipient": recipient, "careRecipients": [recipient]}
        )
        self.audit.record(
            "member_registered",
            str(member["seniorId"]),
            {
                "seniorId": member["seniorId"],
                "firstName": member["firstName"],
                "lastName": member["lastName"],
            },
        )
        return simulated(
            {
                "created": True,
                "message": "Demo account created. Save the User ID for future visits.",
                "userId": member["seniorId"],
                "member": self._public_member(member),
                "cases": [],
            }
        )

    def list_cases(self, user_id: str) -> dict:
        member = self._require_member(user_id)
        return simulated(
            {"member": self._public_member(member), "cases": self.cases.cases_for(user_id)}
        )

    def add_care_recipient(
        self, user_id: str, registration: CareRecipientRegistration | dict
    ) -> dict:
        self._require_member(user_id)
        request = (
            registration
            if isinstance(registration, CareRecipientRegistration)
            else CareRecipientRegistration.model_validate(registration)
        )
        today = date.today()
        age = (
            today.year
            - request.date_of_birth.year
            - ((today.month, today.day) < (request.date_of_birth.month, request.date_of_birth.day))
        )
        recipient = self.members.add_care_recipient(
            user_id,
            {
                "firstName": request.first_name.strip(),
                "lastName": request.last_name.strip(),
                "dateOfBirth": request.date_of_birth.isoformat(),
                "age": age,
                "relationshipToAccountHolder": request.relationship_to_account_holder,
            },
        )
        self.audit.record(
            "care_recipient_added",
            str(recipient["recipientId"]),
            {"userId": user_id, "recipientId": recipient["recipientId"]},
        )
        return simulated(
            self.members.public_care_recipient(
                self._require_member(user_id), recipient["recipientId"]
            )
        )

    def create_case(self, user_id: str, request: NewCaseRequest | dict) -> dict:
        member = self._require_member(user_id)
        value = (
            request
            if isinstance(request, NewCaseRequest)
            else NewCaseRequest.model_validate(request)
        )
        recipient = self.members.get_care_recipient(member, value.recipient_id)
        if not recipient:
            raise PermissionError(
                "Select a care recipient; the selected recipient must belong to this account"
            )
        recipient_id = str(recipient["recipientId"])
        now = datetime.now(UTC).isoformat()
        case = self.cases.create_case(
            {
                "seniorId": user_id,
                "requestedByUserId": user_id,
                "recipientId": recipient_id,
                "careRecipient": self.members.public_care_recipient(member, recipient_id),
                "caseType": value.case_type,
                "title": value.title,
                "description": value.description,
                "status": "open",
                "priority": value.priority,
                "relatedEntityIds": value.related_entity_ids,
                "assignedCaregiverId": value.assigned_caregiver_id,
                "openedAt": now,
                "updatedAt": now,
                "targetDate": value.target_date.isoformat() if value.target_date else None,
                "closedAt": None,
                "latestStatusNote": "Case submitted",
                "simulation": True,
            }
        )
        self.audit.record("case_created", str(case["caseId"]), case)
        return simulated(case)

    def update_case_status(
        self,
        user_id: str,
        case_id: str,
        status: Literal[
            "open", "in_progress", "blocked", "waiting_for_user", "resolved", "closed", "cancelled"
        ],
        status_note: str,
    ) -> dict:
        self._require_member(user_id)
        case = self.cases.get_case(case_id)
        if not case or case.get("seniorId") != user_id:
            raise KeyError("Case not found for this member")
        now = datetime.now(UTC).isoformat()
        changes = {
            "status": status,
            "latestStatusNote": status_note,
            "updatedAt": now,
            "closedAt": now if status in {"resolved", "closed", "cancelled"} else None,
        }
        updated = self.cases.update_case(case_id, changes)
        self.audit.record(
            "case_status_updated",
            case_id,
            {"seniorId": user_id, "caseId": case_id, "status": status, "statusNote": status_note},
        )
        return simulated(updated)

    def close_due_cases(self, user_id: str) -> dict:
        """Close active cases after their latest appointment or explicit target date."""
        self._require_member(user_id)
        today = date.today()
        closed_case_ids: list[str] = []
        terminal_statuses = {"resolved", "closed", "cancelled"}
        for case in self.cases.cases_for(user_id):
            if case.get("status") in terminal_statuses:
                continue
            appointment_dates = []
            for entity_id in case.get("relatedEntityIds", []):
                if not str(entity_id).startswith("APT"):
                    continue
                appointment = self.appointments.get(str(entity_id))
                if appointment and appointment.get("appointmentDate"):
                    appointment_dates.append(date.fromisoformat(appointment["appointmentDate"]))
            due_date = max(appointment_dates) if appointment_dates else None
            if due_date is None and case.get("targetDate"):
                due_date = date.fromisoformat(case["targetDate"])
            if due_date is None or due_date >= today:
                continue
            updated = self.update_case_status(
                user_id,
                case["caseId"],
                "closed",
                f"Automatically closed after due date {due_date.isoformat()}",
            )
            if updated:
                closed_case_ids.append(case["caseId"])
        return simulated({"closedCaseIds": closed_case_ids, "evaluatedAt": str(today)})

    def get_case(self, user_id: str, case_id: str) -> dict:
        self._require_member(user_id)
        case = self.cases.get_case(case_id)
        if not case or case.get("seniorId") != user_id:
            raise KeyError("Case not found for this member")
        return simulated(case)

    def add_related_entity(self, user_id: str, case_id: str, entity_id: str) -> dict:
        self._require_member(user_id)
        case = self.cases.get_case(case_id)
        if not case or case.get("seniorId") != user_id:
            raise KeyError("Case not found for this member")
        related = list(dict.fromkeys([*case.get("relatedEntityIds", []), entity_id]))
        updated = self.cases.update_case(
            case_id, {"relatedEntityIds": related, "updatedAt": datetime.now(UTC).isoformat()}
        )
        self.audit.record(
            "case_entity_linked",
            case_id,
            {"seniorId": user_id, "caseId": case_id, "relatedEntityId": entity_id},
        )
        return simulated(updated)

    def _require_member(self, user_id: str) -> dict:
        member = self.members.get(user_id)
        if not member:
            raise KeyError(f"Unknown userId: {user_id}")
        return member

    @staticmethod
    def _public_member(member: dict) -> dict:
        return SeniorRepository.public_member(member)
