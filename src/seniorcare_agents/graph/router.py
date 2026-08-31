import re

INTENTS = {
    "member_registration": ("new member", "i am new", "register", "create user id"),
    "member_lookup": ("returning member", "user id"),
    "case_create": ("create a case", "open a case", "new case"),
    "appointments": ("appointment", "visit", "schedule", "cancel appointment"),
    "provider_search": ("doctor", "provider", "cardiolog", "specialist"),
    "referral": ("referral",),
    "discharge_support": ("hospital discharge", "discharge follow-up", "care transition"),
    "transportation": ("ride", "transport", "paratransit", "wheelchair"),
    "medication": ("medication", "medicine", "prescription"),
    "pharmacy_refill": ("refill", "pharmacy"),
    "meals": ("meal", "food", "snap", "nutrition"),
    "benefits_financial": (
        "benefit",
        "financial assistance",
        "medicaid",
        "medicare savings",
        "utility assistance",
    ),
    "benefits_home_support": (
        "home-support benefit",
        "home support benefit",
        "housing assistance",
        "utility assistance",
    ),
    "social_wellbeing": (
        "activity",
        "activities",
        "meet other",
        "social",
        "class",
        "lonely",
        "exercise",
    ),
    "home_support": (
        "home support",
        "home safety",
        "in-home",
        "aging in place",
        "grab bar",
        "ramp",
        "caregiver",
    ),
    "caregiver_support": ("caregiver", "respite"),
    "reminder": ("remind", "reminder"),
    "case_status": ("case", "pending", "blocked", "status"),
    "risk": ("risk", "overdue", "attention", "priority", "stuck", "blocked"),
}
AGENTS = {
    "appointments": "healthcare",
    "provider_search": "healthcare",
    "referral": "healthcare",
    "discharge_support": "healthcare",
    "transportation": "transportation",
    "medication": "medication",
    "pharmacy_refill": "medication",
    "meals": "meals",
    "benefits_financial": "meals",
    "benefits_home_support": "home_support",
    "social_wellbeing": "social",
    "home_support": "home_support",
    "caregiver_support": "home_support",
    "reminder": "case_status",
    "case_status": "case_status",
    "risk": "case_status",
}


def route_intents(query: str) -> tuple[list[str], list[str]]:
    lower = query.casefold()
    intents = [name for name, terms in INTENTS.items() if any(term in lower for term in terms)]
    if re.search(r"\bAPT[\w-]*\b", query, re.IGNORECASE) and "transportation" not in intents:
        intents.append("transportation")
    # Here "appointment" can identify a transportation destination rather
    # than request another healthcare booking. Retain both domains only when
    # the user explicitly asks to act on the appointment itself.
    appointment_action = re.search(
        r"\b(?:book|schedule|make|cancel|reschedule)\b(?:\s+\w+){0,3}\s+appointment\b",
        lower,
    )
    if (
        "transportation" in intents
        and "appointments" in intents
        and "provider_search" not in intents
        and appointment_action is None
    ):
        intents.remove("appointments")
    if len(set(intents)) > 1:
        intents.append("multi_domain")
    agents = list(dict.fromkeys(AGENTS[value] for value in intents if value in AGENTS))
    if not intents:
        return ["case_status"], ["case_status"]
    return intents, agents
