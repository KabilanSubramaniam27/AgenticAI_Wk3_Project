from seniorcare_agents.models import AgentResult


class OutputGuardrail:
    FORBIDDEN = (
        "real appointment confirmed",
        "doctor was contacted",
        "pharmacy was contacted",
        "meal provider was contacted",
        "transportation provider was contacted",
        "event organizer was contacted",
        "stop taking",
        "change your dose",
    )

    def validate(self, response: str, results: list[AgentResult]) -> list[str]:
        flags = (
            ["unsupported_external_claim"]
            if any(term in response.casefold() for term in self.FORBIDDEN)
            else []
        )
        executed = any(
            call.operation == "write" and call.status == "success"
            for result in results
            for call in result.tool_calls
        )
        if "has been booked" in response.casefold() and not executed:
            flags.append("unapproved_mutation_claim")
        return flags
