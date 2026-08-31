from datetime import date, datetime, timedelta

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.services.senior_context_service import SeniorContextService


class RiskDetectionService:
    OPEN = {"pending", "in_progress", "blocked", "not_started", "documents_needed", "requested"}

    def __init__(self, settings: RuntimeSettings):
        self.context = SeniorContextService(settings)

    @staticmethod
    def _date(value: str | None) -> date | None:
        try:
            return datetime.fromisoformat(value).date() if value else None
        except ValueError:
            return None

    def evaluate(self, senior_id: str, as_of: date | None = None) -> list[dict]:
        today = as_of or date.today()
        context = self.context.get_context(senior_id)
        risks = []
        for task in context["caseTasks"]:
            if (
                task.get("status") in self.OPEN
                and (due := self._date(task.get("dueDate")))
                and due < today
            ):
                risks.append(
                    {
                        "code": "OVERDUE_CASE_TASK",
                        "severity": task.get("priority", "medium"),
                        "entityId": task["caseTaskId"],
                        "dueDate": str(due),
                    }
                )
        for refill in context["refills"]:
            if refill.get("status") == "needs_authorization":
                risks.append(
                    {
                        "code": "REFILL_AUTHORIZATION_NEEDED",
                        "severity": "high",
                        "entityId": refill["refillId"],
                    }
                )
        refill_by_medication = {row.get("medicationId"): row for row in context["refills"]}
        for medication in context["medications"]:
            due = self._date(medication.get("nextRefillDue"))
            refill = refill_by_medication.get(medication.get("medicationId"), {})
            if (
                due
                and today <= due <= today + timedelta(days=3)
                and refill.get("status") not in {"ready_for_pickup", "completed"}
            ):
                risks.append(
                    {
                        "code": "REFILL_DUE_SOON",
                        "severity": "attention",
                        "entityId": medication["medicationId"],
                        "dueDate": str(due),
                    }
                )
        for case in context["cases"]:
            if case.get("status") == "blocked":
                risks.append(
                    {"code": "CASE_BLOCKED", "severity": "high", "entityId": case["caseId"]}
                )
            target = self._date(case.get("targetDate"))
            if case.get("status") not in {"resolved", "closed"} and target and target < today:
                risks.append(
                    {
                        "code": "CASE_TARGET_OVERDUE",
                        "severity": case.get("priority", "medium"),
                        "entityId": case["caseId"],
                        "dueDate": str(target),
                    }
                )
            updated = self._date(case.get("updatedAt") or case.get("openedAt"))
            if (
                case.get("status") in {"blocked", "in_progress"}
                and updated
                and updated < today - timedelta(days=7)
            ):
                risks.append(
                    {
                        "code": "CASE_STUCK",
                        "severity": "high",
                        "entityId": case["caseId"],
                        "dueDate": str(updated),
                    }
                )
        rides = {row.get("appointmentId"): row for row in context["rides"]}
        for appointment in context["appointments"]:
            if appointment.get("transportationRequired") and rides.get(
                appointment["appointmentId"], {}
            ).get("status") not in {"confirmed", "completed"}:
                appointment_date = self._date(appointment.get("appointmentDate"))
                severity = (
                    "high"
                    if appointment_date and today <= appointment_date <= today + timedelta(days=2)
                    else "medium"
                )
                risks.append(
                    {
                        "code": "APPOINTMENT_RIDE_NOT_CONFIRMED",
                        "severity": severity,
                        "entityId": appointment["appointmentId"],
                    }
                )
        for referral in context["referrals"]:
            if referral.get("status") in {"pending", "missing", "denied"}:
                risks.append(
                    {
                        "code": "REFERRAL_INCOMPLETE",
                        "severity": "high",
                        "entityId": referral["referralId"],
                    }
                )
        return risks
