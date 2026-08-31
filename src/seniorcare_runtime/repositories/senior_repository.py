from uuid import uuid4

from seniorcare_runtime.config import RuntimeSettings
from seniorcare_runtime.repositories.base import JsonRepository


class SeniorRepository:
    def __init__(self, settings: RuntimeSettings):
        self.seniors = JsonRepository(settings.synthetic_dir / "seniors.json", "seniorId")
        self.caregivers = JsonRepository(settings.synthetic_dir / "caregivers.json", "caregiverId")

    def get(self, senior_id: str) -> dict | None:
        return self.seniors.get(senior_id)

    def search(self, name: str | None = None, county: str | None = None) -> list[dict]:
        rows = self.seniors.all()
        if county:
            rows = [row for row in rows if row.get("county") == county]
        if name:
            term = name.casefold()
            rows = [
                row
                for row in rows
                if term in f"{row.get('firstName', '')} {row.get('lastName', '')}".casefold()
            ]
        return rows

    def caregivers_for(self, senior_id: str) -> list[dict]:
        return self.caregivers.find(seniorId=senior_id)

    def get_caregiver(self, caregiver_id: str) -> dict | None:
        return self.caregivers.get(caregiver_id)

    def find_identity(self, first_name: str, last_name: str, date_of_birth: str) -> dict | None:
        return next(
            (
                row
                for row in self.seniors.all()
                if row.get("firstName", "").casefold() == first_name.casefold()
                and row.get("lastName", "").casefold() == last_name.casefold()
                and row.get("dateOfBirth") == date_of_birth
            ),
            None,
        )

    def create(self, profile: dict) -> dict:
        return self.seniors.create_with_generated_id("SEN", profile)

    @staticmethod
    def care_recipients(member: dict) -> list[dict]:
        values = member.get("careRecipients")
        if isinstance(values, list) and values:
            return [dict(value) for value in values if isinstance(value, dict)]
        legacy = member.get("careRecipient")
        if isinstance(legacy, dict):
            return [{"recipientId": legacy.get("recipientId") or member["seniorId"], **legacy}]
        return [
            {
                "recipientId": member["seniorId"],
                "firstName": member.get("firstName"),
                "lastName": member.get("lastName"),
                "dateOfBirth": member.get("dateOfBirth"),
                "age": member.get("age"),
                "relationshipToAccountHolder": "self",
                "isAccountHolder": True,
            }
        ]

    def get_care_recipient(self, member: dict, recipient_id: str | None = None) -> dict | None:
        recipients = self.care_recipients(member)
        if recipient_id:
            return next((row for row in recipients if row.get("recipientId") == recipient_id), None)
        return recipients[0] if len(recipients) == 1 else None

    def add_care_recipient(self, user_id: str, recipient: dict) -> dict:
        member = self.get(user_id)
        if not member:
            raise KeyError(f"Unknown userId: {user_id}")
        recipients = self.care_recipients(member)
        duplicate = next(
            (
                row
                for row in recipients
                if row.get("firstName", "").casefold() == recipient["firstName"].casefold()
                and row.get("lastName", "").casefold() == recipient["lastName"].casefold()
                and row.get("dateOfBirth") == recipient["dateOfBirth"]
            ),
            None,
        )
        if duplicate:
            return duplicate
        created = {"recipientId": f"REC-{uuid4().hex[:12]}", **recipient, "isAccountHolder": False}
        recipients.append(created)
        self.seniors.update(
            user_id,
            {
                "accountRole": "family_representative",
                "careRecipients": recipients,
                "careRecipient": recipients[0],
            },
        )
        return created

    @classmethod
    def public_care_recipient(cls, member: dict, recipient_id: str | None = None) -> dict:
        """Return a non-sensitive recipient snapshot, including legacy self-care records."""
        recipients = cls.care_recipients(member)
        value = (
            next((row for row in recipients if row.get("recipientId") == recipient_id), None)
            if recipient_id
            else (recipients[0] if len(recipients) == 1 else None)
        )
        return {key: item for key, item in (value or {}).items() if key != "dateOfBirth"}

    @classmethod
    def public_member(cls, member: dict) -> dict:
        public = {key: value for key, value in member.items() if key != "dateOfBirth"}
        recipients = cls.care_recipients(member)
        public["careRecipients"] = [
            {key: item for key, item in value.items() if key != "dateOfBirth"}
            for value in recipients
        ]
        public["careRecipient"] = public["careRecipients"][0]
        return public
