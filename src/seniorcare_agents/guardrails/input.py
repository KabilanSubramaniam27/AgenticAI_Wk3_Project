import re


class InputGuardrail:
    INJECTION = (
        "ignore previous",
        "ignore system",
        "reveal prompt",
        "developer message",
        "api key",
        "secret",
    )
    EXTERNAL = (
        "real doctor",
        "real pharmacy",
        "disable simulation",
        "external mutation",
        "bypass approval",
        "without confirmation",
    )
    MEDICAL = ("stop taking", "change my dose", "increase dosage", "diagnose me", "prescribe")
    EMERGENCY = (
        "chest pain",
        "can't breathe",
        "cannot breathe",
        "unconscious",
        "severe bleeding",
        "suicidal",
    )

    def evaluate(self, query: str) -> dict:
        lower = query.casefold()
        flags = []
        if any(term in lower for term in self.INJECTION):
            flags.append("prompt_injection")
        if any(term in lower for term in self.EXTERNAL):
            flags.append("external_action_attempt")
        if any(term in lower for term in self.MEDICAL):
            flags.append("medical_safety")
        emergency = any(term in lower for term in self.EMERGENCY)
        if emergency:
            flags.append("potential_emergency")
        user_ids = re.findall(r"\bSEN\d{4,}\b", query.upper())
        return {
            "allowed": not any(
                flag in flags for flag in ("prompt_injection", "external_action_attempt")
            ),
            "emergency": emergency,
            "flags": flags,
            "detectedUserIds": user_ids,
        }
