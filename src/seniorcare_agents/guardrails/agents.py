from __future__ import annotations

import re
from collections.abc import Mapping

from seniorcare_agents.models import (
    AgentResult,
    OrchestratorPlan,
    OrchestratorResponse,
    SpecialistPlan,
)


class AgentGuardrailError(ValueError):
    """Raised when a specialist or orchestrator violates its runtime contract."""


class SpecialistGuardrail:
    MAX_QUERY_CHARACTERS = 4000
    USER_ID = re.compile(r"^SEN\d{4,}$")
    CLINICAL_CLAIMS = (
        "you have ",
        "diagnosed with",
        "increase your dose",
        "decrease your dose",
        "stop taking",
        "replace your medication",
    )
    RECIPIENT_TERMS = {
        "father": ("father", "dad"),
        "mother": ("mother", "mom", "mum"),
        "parent": ("parent", "parents"),
        "spouse": ("spouse", "husband", "wife", "partner"),
    }
    SELF_CARE_TERMS = (
        "for me",
        "for myself",
        "my knee",
        "my shoulder",
        "my appointment",
        "my medication",
        "pick me up",
    )

    def validate_input(self, agent_key: str, query: str, user_id: str) -> None:
        if not query.strip():
            raise AgentGuardrailError(f"{agent_key}: query cannot be empty")
        if len(query) > self.MAX_QUERY_CHARACTERS:
            raise AgentGuardrailError(f"{agent_key}: query exceeds safe length")
        if not self.USER_ID.fullmatch(user_id):
            raise AgentGuardrailError(f"{agent_key}: invalid SeniorCare user ID")

    def recipient_issue(
        self, query: str, member: dict, recipient_id: str | None = None
    ) -> str | None:
        """Reject ambiguous cross-recipient actions before an LLM can propose them."""
        lower = query.casefold()
        recipients = member.get("careRecipients") or [member.get("careRecipient") or {}]
        if len(recipients) > 1 and not recipient_id:
            return "Please select which registered care recipient this request is for."
        recipient = next(
            (row for row in recipients if row.get("recipientId") == recipient_id),
            recipients[0] if len(recipients) == 1 else {},
        )
        if not recipient:
            return "The selected care recipient does not belong to this account."
        relationship = str(recipient.get("relationshipToAccountHolder") or "self")
        role = (
            "self_care"
            if recipient.get("isAccountHolder") or relationship == "self"
            else "family_representative"
        )
        mentioned = {
            name
            for name, terms in self.RECIPIENT_TERMS.items()
            if any(term in lower for term in terms)
        }
        if role == "self_care" and mentioned:
            return (
                "This account is registered for self-care, but the request names a family "
                "member. Register or use the family representative's account first."
            )
        if role == "family_representative":
            if any(term in lower for term in self.SELF_CARE_TERMS):
                return (
                    "This account is registered for a family care recipient, but the request "
                    "appears to be for the account holder. Please confirm or use a self-care account."
                )
            compatible = {relationship}
            if relationship in {"father", "mother"}:
                compatible.add("parent")
            if mentioned and not mentioned <= compatible:
                return (
                    f"This account's care recipient is registered as {relationship}; the request "
                    "names a different person. Please clarify before continuing."
                )
        return None

    def validate_output(
        self,
        agent_key: str,
        expected_name: str,
        result: AgentResult,
        user_id: str,
        allowed_actions: set[str],
        allowed_rag_categories: set[str] | None = None,
        recipient_id: str | None = None,
    ) -> None:
        if result.agent_name != expected_name:
            raise AgentGuardrailError(f"{agent_key}: incorrect agent identity")
        if not result.summary.strip():
            raise AgentGuardrailError(f"{agent_key}: empty summary")
        if any(call.operation == "write" for call in result.tool_calls):
            raise AgentGuardrailError(f"{agent_key}: LLM reported a forbidden write tool call")
        for action in result.proposed_actions:
            if action.action_type not in allowed_actions:
                raise AgentGuardrailError(
                    f"{agent_key}: forbidden proposed action {action.action_type!r}"
                )
            if action.user_id != user_id or action.senior_id != user_id:
                raise AgentGuardrailError(f"{agent_key}: proposed action ownership mismatch")
            if action.recipient_id != recipient_id:
                raise AgentGuardrailError(f"{agent_key}: proposed action recipient mismatch")
            if not action.simulation or not action.requires_approval or action.status != "proposed":
                raise AgentGuardrailError(f"{agent_key}: unsafe proposed action state")
        for chunk in result.retrieved_sources:
            if not chunk.chunk_id or not chunk.source_name or not chunk.category:
                raise AgentGuardrailError(f"{agent_key}: retrieved source lacks attribution")
            if allowed_rag_categories is not None and chunk.category not in allowed_rag_categories:
                raise AgentGuardrailError(
                    f"{agent_key}: retrieved source category {chunk.category!r} is not allowed"
                )
        if agent_key in {"healthcare", "medication"}:
            text = result.summary.casefold()
            if any(claim in text for claim in self.CLINICAL_CLAIMS):
                raise AgentGuardrailError(f"{agent_key}: prohibited clinical claim")

    def validate_plan(
        self,
        agent_key: str,
        plan: SpecialistPlan,
        allowed_tools: set[str] | frozenset[str],
    ) -> None:
        if not plan.task_summary.strip():
            raise AgentGuardrailError(f"{agent_key}: empty specialist plan")
        if len(plan.selected_tools) != len(set(plan.selected_tools)):
            raise AgentGuardrailError(f"{agent_key}: duplicate planned tool")
        forbidden = set(plan.selected_tools).difference(allowed_tools)
        if forbidden:
            raise AgentGuardrailError(f"{agent_key}: planned forbidden tools {sorted(forbidden)}")


class OrchestratorGuardrail:
    MAX_SELECTED_AGENTS = 7

    def validate_plan(self, plan: OrchestratorPlan, available: Mapping[str, object]) -> None:
        self.validate_selection(plan.selected_agents, available)
        if len(plan.selected_agents) > self.MAX_SELECTED_AGENTS:
            raise AgentGuardrailError("orchestrator: too many specialists selected")
        staged = [agent for stage in plan.execution_stages for agent in stage.agents]
        if len(staged) != len(set(staged)) or set(staged) != set(plan.selected_agents):
            raise AgentGuardrailError(
                "orchestrator: execution stages must contain each selected agent exactly once"
            )
        stage_numbers = {stage.stage for stage in plan.execution_stages}
        for stage in plan.execution_stages:
            if any(dependency not in plan.selected_agents for dependency in stage.depends_on):
                raise AgentGuardrailError("orchestrator: unknown stage dependency")
            if stage.stage < 1 or not stage_numbers:
                raise AgentGuardrailError("orchestrator: invalid execution stage")

    def validate_selection(self, selected: list[str], available: Mapping[str, object]) -> None:
        if not selected:
            raise AgentGuardrailError("orchestrator: no specialist selected")
        if len(selected) != len(set(selected)):
            raise AgentGuardrailError("orchestrator: duplicate specialist selection")
        unknown = set(selected).difference(available)
        if unknown:
            raise AgentGuardrailError(f"orchestrator: unknown specialists {sorted(unknown)}")

    def validate_results(
        self, selected: list[str], results: Mapping[str, AgentResult], user_id: str
    ) -> None:
        if set(results) != set(selected):
            raise AgentGuardrailError("orchestrator: result set does not match selected agents")
        action_ids: set[str] = set()
        for key, result in results.items():
            if result.status == "failed" and not result.warnings:
                raise AgentGuardrailError(f"orchestrator: {key} failed without an explanation")
            for action in result.proposed_actions:
                if action.action_id in action_ids:
                    raise AgentGuardrailError("orchestrator: duplicate proposed action ID")
                action_ids.add(action.action_id)
                if action.user_id != user_id:
                    raise AgentGuardrailError("orchestrator: cross-member action rejected")

    def validate_synthesis(
        self,
        response: OrchestratorResponse,
        results: Mapping[str, AgentResult],
    ) -> None:
        if not response.answer.strip():
            raise AgentGuardrailError("orchestrator: empty synthesized answer")
        normalized_answer = response.answer.casefold().replace("_", " ")
        if re.search(
            r"\b(?:provide|need|enter|supply)\b.{0,80}\b(?:provider|availability)\s+id\b",
            normalized_answer,
        ):
            raise AgentGuardrailError(
                "orchestrator: synthesis requested an internal provider/availability ID"
            )
        if not set(response.completed_agents).issubset(results):
            raise AgentGuardrailError("orchestrator: synthesis references an unknown agent")
        valid_actions = {
            action.action_id for result in results.values() for action in result.proposed_actions
        }
        if not set(response.action_ids).issubset(valid_actions):
            raise AgentGuardrailError("orchestrator: synthesis invented an action ID")
        valid_citations = {
            chunk.chunk_id for result in results.values() for chunk in result.retrieved_sources
        }
        if not set(response.citation_ids).issubset(valid_citations):
            raise AgentGuardrailError("orchestrator: synthesis invented a citation ID")
        verified_provider_names = {
            name.casefold()
            for result in results.values()
            for name in re.findall(r"\bDr\.\s+[A-Z][A-Za-z'-]+", result.summary)
        }
        if verified_provider_names and not any(
            name in response.answer.casefold() for name in verified_provider_names
        ):
            raise AgentGuardrailError(
                "orchestrator: synthesis discarded every verified provider name"
            )
