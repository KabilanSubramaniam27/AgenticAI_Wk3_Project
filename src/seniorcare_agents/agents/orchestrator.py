import asyncio
import inspect
import json
import re

from langchain_core.language_models import BaseChatModel

from seniorcare_agents.guardrails import OrchestratorGuardrail
from seniorcare_agents.models import (
    AgentResult,
    ExecutionStage,
    OrchestratorPlan,
    OrchestratorResponse,
)
from seniorcare_agents.observability import flow_event

AGENT_CAPABILITIES = {
    "healthcare": "Use for finding providers, viewing provider availability, listing or checking operational doctor appointment records (including APT status), proposing a new doctor appointment, referrals, discharge guidance, and healthcare-access RAG. Do not use merely to transport someone to an already scheduled APT ID.",
    "transportation": "Use for travel to any eligible booked senior-care service, including healthcare appointments and registered social activities, plus pickup/drop-off, paratransit, services, vehicles, travel estimates, accessibility, and transportation RAG. A supplied booking ID is immutable input: retrieve its type, owner, destination, date, and time instead of creating another booking.",
    "medication": "Use only for existing member medication/refill records, reference information about a medication explicitly named by the user, and medication-safety guidance. Never infer or list drugs from symptoms, diagnose, prescribe, recommend dosage changes, or create pharmacy/refill/delivery actions in Phase 1. Symptom-based treatment requests belong to HealthcareAccessAgent for safe provider guidance.",
    "meals": "Use to find meal-assistance services, food assistance, nutrition programs, and benefits/financial guidance. List available programs with retrieved contact details and direct the user to contact the program for enrollment; do not fabricate contact information.",
    "social": "Use to find nearby senior activities, companionship, classes, and social-wellbeing resources. List retrieved event and contact details and tell the user to contact the organizer directly; do not register an activity in Phase 1.",
    "home_support": "Use for home safety/support requests, caregiver and respite support, accessibility modifications, and related benefits RAG.",
    "case_status": "Use for CASE tracking records, reminders, audit history, coordination status, and rule-based risk evaluation. It does not own operational doctor appointment or ride records. Do not use for a new domain booking merely because it will later have a case.",
}

PLANNING_PROMPT = """You are the SeniorCare orchestrator planner. Select every specialist needed
for the request from the supplied capability catalog. Return a typed OrchestratorPlan. Independent
agents belong in the same execution stage; dependent agents belong in later stages. Transportation
depends on the specialist creating the destination booking only when that healthcare, social, or
other supported booking does not exist yet. Transportation for an existing booking ID is independent
and must retrieve that record without rerunning its source-domain workflow. Never select an
unknown agent. Do not make medical decisions and do not propose or execute writes. If the user only
says "appointment" without identifying its domain, put this clarification in missing_information:
"Which type of appointment do you mean: a doctor/provider appointment, transportation for an
existing appointment, or another service appointment?" Do not guess a domain."""

# These instructions deliberately give the model semantic control while retaining hard security
# boundaries in code. They are included in every orchestrator planning request.
PLANNING_PROMPT += """

Return exactly this logical shape (with values appropriate to the current request):
{
  "intents": ["meal_assistance"],
  "selected_agents": ["meals"],
  "execution_stages": [{"stage": 1, "agents": ["meals"], "depends_on": []}],
  "missing_information": [],
  "routing_summary": {"meals": "The current request asks for meal assistance."},
  "confidence": 0.9
}
`routing_summary` must be a flat object whose keys are selected agent keys and whose values are
plain explanation strings. Never put `selected_agents`, arrays, or nested objects inside it.

DO:
- Treat the current user query as authoritative. Prior cases and referenced records are optional
  context only when the current query explicitly refers to them or is a genuine follow-up.
- Use recentConversationTurns and rollingConversationSummary to resolve genuine answers to the
  immediately preceding clarification.
  Preserve the preceding request's domain and constraints when the current turn supplies only the
  requested county, address, yes/no choice, provider choice, or booking identifier.
- Read the complete agent capability catalog and referenced operational records before routing.
- Infer all user intents, select only agents that materially contribute, and explain each choice in routing_summary.
- Put independent agents in one stage and true dependencies in later stages.
- Treat a referenced scheduled APT record as authoritative context. For transportation to that APT,
  select TransportationAgent only unless the user explicitly asks to cancel, reschedule, or modify it.
- Ask one concise clarification in missing_information when the domain, recipient, appointment ID,
  or another required choice is genuinely ambiguous.
- Route broad provider-directory questions to HealthcareAccessAgent even when no specialty is
  supplied. The healthcare specialist can list matching providers and offer optional filtering;
  do not require a provider type before performing that search.
- Route a request for official information, FDA reference data, safety information, or guidance
  about a specifically named drug to MedicationPharmacyAgent, even when the word "medication" is
  absent. Example: "Find official information about lisinopril" is medication reference, not
  healthcare-provider search. Symptom assessment without a named drug remains healthcare.
- Route a new symptom, adverse-effect concern, dizziness, pain, breathing problem, or other
  clinical-assessment request to HealthcareAccessAgent. MedicationPharmacyAgent may additionally
  supply reference facts only when a specific medication is named. Never block symptom safety
  handling merely to ask for a medication name.

DO NOT:
- Continue an older domain workflow when the current query clearly requests a different domain.
- Route by a single keyword while ignoring the full request and referenced records.
- Start a new healthcare appointment workflow for transportation to an existing APT ID.
- Invent an agent, record, provider, destination, date, time, status, dependency, or tool result.
- Select an agent only because its output might be generally useful.
- Execute tools, propose writes, bypass approval, or make clinical decisions.
"""

SYNTHESIS_PROMPT = """You are the SeniorCare orchestrator response synthesizer. Combine only the
validated specialist results supplied to you into one concise, helpful answer. Preserve facts,
limitations, citations, action IDs, approval requirements, and simulation boundaries. Never invent
an ID, provider, time, location, availability, tool result, or successful external action. Return a
typed OrchestratorResponse. If a local specialist fails, explain that the local request
could not be completed. Do not advise the user to contact a real provider as a substitute for a
failed simulation.

DO:
- Use only validated specialist findings, missing-information questions, citations, and actions.
- Reconcile cross-domain dependencies and produce one non-duplicative response.
- Preserve the selected recipient and any referenced record IDs throughout the answer.
- Ask only for information that the validated specialist results show is still missing.

DO NOT:
- Re-route the request, invoke tools, or create a new action during synthesis.
- Re-ask for an address, appointment ID, date, time, provider, or recipient already retrieved.
- Turn a proposed action into a claim that it was executed or externally confirmed.
- Invent citations or present structured operational/API records as RAG citations.
- Expose internal tool names such as names containing dummy or describe local actions as simulated
  in ordinary user-facing labels; the UI separately communicates the local-study safety boundary.
"""


class SeniorCareOrchestratorAgent:
    name = "SeniorCareOrchestratorAgent"

    def __init__(
        self,
        agents: dict[str, object],
        model: BaseChatModel | None = None,
        configured: bool = False,
    ):
        self.agents = agents
        self.model = model
        self.configured = configured and model is not None
        self.guardrail = OrchestratorGuardrail()

    async def plan(
        self,
        query: str,
        member_context: dict,
        recipient: dict,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> OrchestratorPlan:
        history = conversation_history or []
        payload = {
            "currentUserRequest": query,
            "recentConversationTurns": history[-10:],
            "rollingConversationSummary": self._rolling_summary(history[:-10]),
            "resolvedEntities": {
                "careRecipient": recipient,
                "referencedAppointments": member_context.get("referencedAppointments", []),
            },
            "verifiedApplicationState": self._verified_state(member_context),
            "openQuestions": [],
            "agentRegistry": {
                key: AGENT_CAPABILITIES[key] for key in self.agents if key in AGENT_CAPABILITIES
            },
            "executionState": {
                "phase": "planning",
                "completedAgents": [],
                "activeCaseIds": [
                    row.get("caseId")
                    for row in member_context.get("cases", [])
                    if isinstance(row, dict) and row.get("status") not in {"closed", "cancelled"}
                ],
            },
        }
        flow_event("llm", "orchestrator_planning", "input", payload)
        if self.configured and self.model is not None:
            last_error: Exception | None = None
            correction = ""
            for attempt in range(1, 3):
                try:
                    # Function calling preserves typed validation while allowing the plan's
                    # intentionally dynamic explanation mapping.
                    planner = self.model.with_structured_output(
                        OrchestratorPlan, method="function_calling"
                    )
                    plan = await planner.ainvoke(
                        [
                            {"role": "system", "content": PLANNING_PROMPT},
                            {
                                "role": "user",
                                "content": json.dumps(payload, default=str) + correction,
                            },
                        ]
                    )
                    plan = OrchestratorPlan.model_validate(plan)
                    plan = self._enforce_appointment_clarification(query, plan)
                    plan = self._enforce_provider_search_clarification(query, plan)
                    plan = self._enforce_appointment_record_routing(query, plan)
                    plan = self._enforce_cross_domain_routing(query, plan)
                    self._validate_semantic_consistency(query, plan)
                    self.guardrail.validate_plan(plan, self.agents)
                    flow_event("llm", "orchestrator_planning", "output", plan.model_dump())
                    return plan
                except Exception as exc:
                    last_error = exc
                    flow_event(
                        "llm",
                        f"orchestrator_planning_attempt_{attempt}",
                        "error",
                        exc,
                    )
                    correction = (
                        "\n\nYour previous response failed typed validation. Return a corrected "
                        "OrchestratorPlan using the exact flat schema shown in the system prompt. "
                        f"Validation error: {str(exc)[:500]}"
                    )
            if last_error is not None:
                flow_event("llm", "orchestrator_planning", "error", last_error)

        # Failure handling is deterministic, but intent recognition is not: do not guess a domain
        # with keyword rules when the LLM is unavailable or cannot return a valid typed plan.
        plan = OrchestratorPlan(
            intents=["intent_clarification_required"],
            selected_agents=["case_status"],
            execution_stages=[ExecutionStage(stage=1, agents=["case_status"])],
            missing_information=[
                "I could not reliably determine which SeniorCare service should handle this "
                "request. Please try the request again."
            ],
            routing_summary={"case_status": "No specialist executes when intent planning fails."},
            confidence=0.0,
        )
        self.guardrail.validate_plan(plan, self.agents)
        flow_event("orchestrator", "intent_planning_failed", "output", plan.model_dump())
        return plan

    @staticmethod
    def _rolling_summary(history: list[dict[str, str]], limit: int = 4000) -> str:
        """Create a bounded, attributable summary of turns outside the recent window."""
        if not history:
            return "No earlier conversation turns."
        lines = [
            f"{str(turn.get('role', 'unknown')).title()}: {str(turn.get('content', '')).strip()}"
            for turn in history
            if str(turn.get("content", "")).strip()
        ]
        return "\n".join(lines)[-limit:] or "No earlier conversation turns."

    @staticmethod
    def _verified_state(member_context: dict) -> dict:
        """Expose authenticated operational state separately from conversational claims."""
        return {
            key: member_context.get(key, [])
            for key in (
                "appointments",
                "rides",
                "medications",
                "refills",
                "cases",
                "reminders",
                "homeSupportRequests",
                "socialActivities",
                "mealServices",
            )
            if key in member_context
        }

    @staticmethod
    def _validate_semantic_consistency(query: str, plan: OrchestratorPlan) -> None:
        """Reject an LLM route that contradicts an explicit completed-domain request."""
        lower = query.casefold()
        healthcare_terms = (
            "doctor",
            "provider",
            "physician",
            "orthopedic",
            "cardiology",
            "primary care",
            "knee pain",
            "shoulder pain",
        )
        meal_terms = ("meal", "food", "nutrition", "snap", "grocer")
        explicit_healthcare = any(term in lower for term in healthcare_terms)
        explicit_meals = any(term in lower for term in meal_terms)
        if explicit_healthcare and not explicit_meals and "meals" in plan.selected_agents:
            raise ValueError(
                "Semantic routing conflict: the completed request explicitly concerns a "
                "doctor/provider, but the plan selected MealsFoodAgent. Preserve the current "
                "healthcare request and route it to HealthcareAccessAgent."
            )
        if explicit_meals and not explicit_healthcare and "healthcare" in plan.selected_agents:
            raise ValueError(
                "Semantic routing conflict: the completed request explicitly concerns meal or "
                "food assistance, but the plan selected HealthcareAccessAgent. Route it to "
                "MealsFoodAgent."
            )

    @staticmethod
    def _enforce_appointment_clarification(query: str, plan: OrchestratorPlan) -> OrchestratorPlan:
        """Require domain clarification for a genuinely ambiguous appointment request."""
        lower = query.casefold()
        domain_terms = (
            "doctor",
            "provider",
            "physician",
            "clinic",
            "hospital",
            "pain",
            "medical",
            "specialist",
            "primary care",
            "primary-care",
            "orthopedic",
            "cardiology",
            "transport",
            "ride",
            "pickup",
            "pick up",
            "drop off",
            "wheelchair",
            "meal",
            "social",
            "activity",
            "home support",
            "caregiver",
            "pharmacy",
            "medication",
        )
        explicitly_typed = (
            bool(re.search(r"\bAPT[\w-]*\b", query, re.IGNORECASE))
            or any(term in lower for term in domain_terms)
            or bool(re.search(r"\bdr\.\s+[a-z][a-z'-]+", lower))
        )
        if explicitly_typed:
            plan.missing_information = [
                value
                for value in plan.missing_information
                if not (
                    "which type of appointment" in value.casefold()
                    or (
                        "doctor/provider appointment" in value.casefold()
                        and "transportation" in value.casefold()
                    )
                )
            ]
            return plan
        if "appointment" not in lower:
            return plan
        ambiguous_action = bool(
            re.search(
                r"\b(?:book|schedule|make|need|want|help)\b.{0,40}\bappointment\b",
                lower,
            )
        ) or lower.strip(" .?!") in {"appointment", "an appointment", "the appointment"}
        if not ambiguous_action:
            return plan
        question = (
            "Which type of appointment do you mean: a doctor/provider appointment, "
            "transportation for an existing appointment, or another service appointment?"
        )
        plan.missing_information = [question]
        return plan

    @staticmethod
    def _enforce_provider_search_clarification(
        query: str, plan: OrchestratorPlan
    ) -> OrchestratorPlan:
        """Allow an explicit broad provider-directory request without forced specialization."""
        lower = query.casefold()
        broad_search = (
            any(term in lower for term in ("provider", "doctor", "physician"))
            and any(term in lower for term in ("available", "find", "near", "list", "what"))
            and "healthcare" in plan.selected_agents
        )
        if broad_search:
            plan.missing_information = [
                value
                for value in plan.missing_information
                if not any(term in value.casefold() for term in ("provider type", "specialty"))
            ]
        return plan

    def _enforce_appointment_record_routing(
        self, query: str, plan: OrchestratorPlan
    ) -> OrchestratorPlan:
        """Keep doctor appointment records with the specialist that can read them."""
        lower = query.casefold()
        appointment_read = "appointment" in lower and any(
            term in lower for term in ("show", "list", "existing", "status", "upcoming")
        )
        if not appointment_read or "transport" in lower or "ride" in lower:
            return plan
        if "healthcare" not in self.agents:
            return plan
        plan.selected_agents = [value for value in plan.selected_agents if value != "case_status"]
        if "healthcare" not in plan.selected_agents:
            plan.selected_agents.append("healthcare")
        plan.execution_stages = [
            ExecutionStage(stage=1, agents=list(plan.selected_agents), depends_on=[])
        ]
        plan.routing_summary.pop("case_status", None)
        plan.routing_summary["healthcare"] = (
            "HealthcareAccessAgent owns operational doctor appointment records."
        )
        return plan

    @staticmethod
    def _enforce_cross_domain_routing(query: str, plan: OrchestratorPlan) -> OrchestratorPlan:
        """Keep an existing appointment reference out of appointment-creation workflow."""
        lower = query.casefold()
        appointment_reference = bool(re.search(r"\bAPT[\w-]*\b", query, re.IGNORECASE))
        transportation_request = any(
            term in lower
            for term in ("transport", "ride", "pickup", "pick up", "drop back", "round-trip")
        )
        appointment_change = bool(
            re.search(r"\b(?:cancel|reschedule|change|modify)\b.{0,30}\bappointment\b", lower)
        )
        if not (appointment_reference and transportation_request and not appointment_change):
            return plan
        return OrchestratorPlan(
            intents=["transportation"],
            selected_agents=["transportation"],
            execution_stages=[ExecutionStage(stage=1, agents=["transportation"])],
            missing_information=plan.missing_information,
            routing_summary={
                "transportation": (
                    "Existing appointment ID is immutable input to transportation coordination"
                )
            },
            confidence=plan.confidence,
        )

    async def run(
        self,
        selected: list[str],
        query: str,
        user_id: str,
        case_id: str | None,
        recipient_id: str | None = None,
        execution_stages: list[ExecutionStage] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        member_context: dict | None = None,
        recipient: dict | None = None,
        routing_summary: dict[str, str] | None = None,
    ) -> dict[str, AgentResult]:
        flow_event(
            "orchestrator",
            self.name,
            "input",
            {
                "selectedAgents": selected,
                "query": query,
                "userId": user_id,
                "caseId": case_id,
                "recipientId": recipient_id,
                "conversationHistory": conversation_history or [],
            },
        )
        self.guardrail.validate_selection(selected, self.agents)

        async def invoke(key: str, dependency_results: dict[str, AgentResult]):
            agent = self.agents[key]
            run = agent.run  # type: ignore[attr-defined]
            try:
                parameter_count = len(inspect.signature(run).parameters)
                task_context = {
                    "assignedObjective": (routing_summary or {}).get(key, query),
                    "taskSpecificSummary": (routing_summary or {}).get(key, query),
                    "verifiedFacts": {
                        "careRecipient": recipient or {},
                        "applicationState": self._verified_state(member_context or {}),
                    },
                    "dependencyResults": {
                        name: value.model_dump(mode="json")
                        for name, value in dependency_results.items()
                    },
                }
                if parameter_count >= 6:
                    result = await run(
                        query,
                        user_id,
                        case_id,
                        recipient_id,
                        conversation_history or [],
                        task_context,
                    )
                elif parameter_count >= 5:
                    result = await run(
                        query, user_id, case_id, recipient_id, conversation_history or []
                    )
                elif parameter_count >= 4:
                    result = await run(query, user_id, case_id, recipient_id)
                else:
                    result = await run(query, user_id, case_id)
                return key, result
            except Exception as exc:
                flow_event("agent", key, "error", exc)
                return key, AgentResult(
                    agent_name=getattr(agent, "name", key),
                    status="failed",
                    summary=(
                        f"{getattr(agent, 'name', key)} could not complete this part of the "
                        "request. Other specialist results remain available."
                    ),
                    warnings=[f"Specialist failure: {type(exc).__name__}"],
                    confidence=0.0,
                )

        results: dict[str, AgentResult] = {}
        stages = execution_stages or [ExecutionStage(stage=1, agents=selected)]
        for stage in sorted(stages, key=lambda item: item.stage):
            stage_agents = [key for key in stage.agents if key in selected]
            flow_event(
                "orchestrator",
                "specialist_stage",
                "input",
                {"stage": stage.stage, "agents": stage_agents, "dependsOn": stage.depends_on},
            )
            dependencies = {key: results[key] for key in stage.depends_on if key in results}
            pairs = await asyncio.gather(*(invoke(key, dependencies) for key in stage_agents))
            results.update(dict(pairs))
            flow_event(
                "orchestrator",
                "specialist_stage",
                "output",
                {"stage": stage.stage, "completedAgents": stage_agents},
            )
        self.guardrail.validate_results(selected, results, user_id)
        flow_event(
            "orchestrator",
            self.name,
            "output",
            {
                "agents": {
                    key: {"status": result.status, "summary": result.summary}
                    for key, result in results.items()
                }
            },
        )
        return results

    async def synthesize(
        self,
        query: str,
        recipient: dict,
        plan: OrchestratorPlan,
        results: dict[str, AgentResult],
        fallback_answer: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> OrchestratorResponse:
        history = conversation_history or []
        payload = {
            "currentUserRequest": query,
            "recentConversationTurns": history[-10:],
            "rollingConversationSummary": self._rolling_summary(history[:-10]),
            "resolvedEntities": {"careRecipient": recipient},
            "verifiedApplicationState": {
                "specialistResults": {
                    key: result.model_dump(mode="json") for key, result in results.items()
                }
            },
            "openQuestions": plan.missing_information,
            "agentRegistry": {
                key: AGENT_CAPABILITIES[key] for key in self.agents if key in AGENT_CAPABILITIES
            },
            "executionState": {
                "phase": "final_synthesis",
                "routingPlan": plan.model_dump(mode="json"),
                "completedAgents": list(results),
            },
            "responseSchema": OrchestratorResponse.model_json_schema(),
            "agentResults": {
                key: result.model_dump(mode="json") for key, result in results.items()
            },
        }
        flow_event("llm", "orchestrator_synthesis", "input", payload)
        if results and all(result.status in {"failed", "blocked"} for result in results.values()):
            response = OrchestratorResponse(
                answer=(
                    "I couldn't complete this local request because the selected "
                    "specialist did not return a usable result. No local action was created and "
                    "no external organization was contacted. Please try again after the service "
                    "is available."
                ),
                completed_agents=list(results),
                confidence=0.0,
            )
            flow_event(
                "orchestrator", "failed_specialists_fallback", "output", response.model_dump()
            )
            return response
        if self.configured and self.model is not None:
            try:
                synthesizer = self.model.with_structured_output(
                    OrchestratorResponse, method="function_calling"
                )
                raw_response = await synthesizer.ainvoke(
                    [
                        {"role": "system", "content": SYNTHESIS_PROMPT},
                        {"role": "user", "content": json.dumps(payload, default=str)},
                    ]
                )
                response = OrchestratorResponse.model_validate(raw_response)
                self.guardrail.validate_synthesis(response, results)
                flow_event("llm", "orchestrator_synthesis", "output", response.model_dump())
                return response
            except Exception as exc:
                flow_event("llm", "orchestrator_synthesis", "error", exc)
        response = OrchestratorResponse(
            answer=fallback_answer,
            completed_agents=list(results),
            citation_ids=[
                chunk.chunk_id for result in results.values() for chunk in result.retrieved_sources
            ],
            action_ids=[
                action.action_id
                for result in results.values()
                for action in result.proposed_actions
            ],
            confidence=min((result.confidence for result in results.values()), default=0.5),
        )
        flow_event("orchestrator", "synthesis_fallback", "output", response.model_dump())
        return response
