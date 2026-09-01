from __future__ import annotations

import asyncio
import json
import re
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from seniorcare_agents.guardrails import SpecialistGuardrail
from seniorcare_agents.mcp import MCPToolGateway
from seniorcare_agents.models import (
    AgentResult,
    ProposedAction,
    RetrievedChunk,
    SpecialistPlan,
    ToolCallRecord,
)
from seniorcare_agents.observability import flow_event

READ_TOOL_POLICIES: dict[str, frozenset[str]] = {
    "healthcare": frozenset(
        {
            "get_member",
            "get_provider",
            "get_available_slot",
            "list_appointments",
            "list_referrals",
            "search_providers",
            "list_available_slots",
            "search_healthcare_knowledge",
        }
    ),
    "transportation": frozenset(
        {
            "get_member",
            "list_rides",
            "list_appointments",
            "search_transportation_services",
            "find_available_transportation",
            "search_public_knowledge",
        }
    ),
    "medication": frozenset(
        {
            "get_member",
            "list_medications",
            "list_refills",
            "search_medication_references",
            "search_medication_knowledge",
        }
    ),
    "meals": frozenset({"get_member", "search_meal_services", "search_public_knowledge"}),
    "social": frozenset({"get_member", "search_social_activities", "search_public_knowledge"}),
    "home_support": frozenset(
        {"get_member", "list_home_support_requests", "search_public_knowledge"}
    ),
    "case_status": frozenset({"get_member", "list_cases", "evaluate_risks", "list_audit_events"}),
}

AGENT_NAMES = {
    "healthcare": "HealthcareAccessAgent",
    "transportation": "TransportationAgent",
    "medication": "MedicationPharmacyAgent",
    "meals": "MealsFoodAgent",
    "social": "SocialWellbeingAgent",
    "home_support": "HomeSupportSafetyAgent",
    "case_status": "CaseStatusRiskAgent",
}
WRITE_ACTIONS = {
    "healthcare": {"book_dummy_appointment"},
    "transportation": {"book_dummy_ride"},
    "medication": set(),
    "meals": set(),
    "social": set(),
    "home_support": {"request_dummy_home_support"},
    "case_status": set(),
}
RAG_CATEGORIES = {
    "healthcare": {"healthcare_access"},
    "transportation": {"transportation"},
    "medication": {"medication_reference"},
    "meals": {"food_meals", "benefits_financial"},
    "social": {"social_wellbeing"},
    "home_support": {"home_support", "caregiver_support", "benefits_financial"},
    "case_status": set(),
}

# These are system record keys, not facts a user should be expected to know. They
# must be resolved by selected MCP reads or left unused for an informational query.
SYSTEM_RESOLVABLE_FIELDS = frozenset(
    {
        "appointment_id",
        "availability_id",
        "event_id",
        "meal_service_id",
        "provider_id",
        "ride_id",
        "service_id",
        "transportation_service_id",
        "vehicle_id",
    }
)


class _GeneratedAgentResult(AgentResult):
    """LLM-authored result before trusted MCP provenance is attached.

    Models sometimes emit incomplete citation or tool-call projections. These two
    fields are intentionally permissive here and are discarded before the real
    ``AgentResult`` is constructed from server-returned MCP data.
    """

    retrieved_sources: list[dict[str, Any]] = Field(default_factory=list)  # type: ignore[assignment]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # type: ignore[assignment]


REQUIRED_ACTION_PARAMETERS: dict[str, frozenset[str]] = {
    "book_dummy_appointment": frozenset({"provider_id", "availability_id", "reason"}),
    "book_dummy_ride": frozenset(
        {
            "appointment_id",
            "service_id",
            "pickup_date",
            "pickup_time",
            "pickup_address",
            "destination_address",
            "appointment_date",
            "appointment_time",
            "wheelchair_required",
            "vehicle_id",
            "estimated_travel_minutes",
            "accommodation",
            "return_ride_required",
        }
    ),
    "request_dummy_home_support": frozenset({"request_type", "priority", "notes"}),
}
ACTION_ALIASES: dict[str, dict[str, str]] = {
    "transportation": {
        "book_ride": "book_dummy_ride",
        "book_transportation": "book_dummy_ride",
        "schedule_ride": "book_dummy_ride",
    }
}
DOMAIN_INSTRUCTIONS = {
    "healthcare": "For provider questions, call search_providers, then list_available_slots for a matched provider, and search_healthcare_knowledge when official guidance is relevant. A new appointment proposal must use a provider_id and availability_id returned as a consistent provider-slot pair. Do not propose a new appointment merely because an existing APT record is referenced for transportation.",
    "transportation": "Transportation may support any eligible stored destination booking, not healthcare alone. A booking ID may refer to a healthcare appointment or registered social activity; retrieve the record and validate recipient ownership, location, date, and time before planning. Never create or modify the source booking, and never require provider_id or availability_id for transportation to an already stored booking. Before proposing a ride, obtain explicit answers for: full pickup/home address, wheelchair assistance yes/no, and round trip yes/no. Never default either yes/no answer. Use the retrieved destination/date/time, calculate pickup time as booking time minus estimated travel time minus a 15-minute arrival buffer, and propose a ride only after an eligible service and vehicle are returned. If the current MCP tools cannot resolve the supplied booking type, report that limitation rather than inventing details.",
    "medication": "Phase 1 is read-only. Use list_medications and list_refills for recipient-specific stored state. Use search_medication_references only when the user explicitly names a medication, and search_medication_knowledge for official safety/reference guidance. Never infer, rank, or list drugs based on symptoms; that would be treatment advice. For symptom-based requests, explain that a clinician must assess the symptoms and suggest using HealthcareAccessAgent to find an appropriate provider. The user may also contact a licensed pharmacist, but never invent a nearby pharmacy or claim one was contacted. Do not propose refills, orders, delivery, prescription upload, acceptance, or any medication mutation.",
    "meals": "For nearby, available, program, service, or option questions, call search_meal_services and search_public_knowledge within food_meals or benefits_financial. List all applicable retrieved meal-assistance programs with program name, service area, phone, website, eligibility summary, and application instructions when those fields exist. Tell the user to contact the program directly to enroll. Do not propose local enrollment, claim definitive eligibility, or invent missing contact details.",
    "social": "Phase 1 is read-only. Ask for city, county, or ZIP when no usable location is available. Use search_social_activities for structured nearby activities and search_public_knowledge only within social_wellbeing. List all applicable retrieved activities with name, type, location, date, start time, phone, website, and registration requirements when available. Preserve freshness information and clearly mark missing contact fields. Tell the senior member or representative to contact the organizer directly to book. Never propose or claim registration.",
    "home_support": "Use list_home_support_requests for recipient-specific state and search_public_knowledge only within home_support, caregiver_support, or benefits_financial. Do not infer that the recipient qualifies. A proposed request must state the grounded request type, priority, and notes without inventing an assessment.",
    "case_status": "Use list_cases and list_audit_events for tracking/status questions and evaluate_risks only for rule-based risk evaluation. Report stored status and timestamps exactly. Do not create a domain booking, make a clinical risk diagnosis, or fabricate a case transition.",
}
DOMAIN_PLANNING_INSTRUCTIONS = {
    "healthcare": "Plan provider discovery, verified provider-slot matching, existing appointment reads, referrals, or healthcare guidance only. For a named provider, search the provider directory first and resolve the returned providerId before reading availability. Never substitute a different provider. Never plan healthcare creation for transportation to an existing booking ID.",
    "transportation": "Plan transportation only. Resolve the referenced eligible destination booking, then retrieve transportation services and availability. Require pickup address, wheelchair assistance yes/no, and round-trip yes/no before a proposal. Reuse the stored destination/date/time; never rerun or modify the source-domain booking.",
    "medication": "Plan read-only medication coordination. Read recipient medications/refills or official reference data for a specifically named medication. Never select tools to infer a drug from symptoms, prescribe, refill, upload a prescription, contact a pharmacy, or arrange delivery.",
    "meals": "Plan read-only discovery of meal-assistance programs and public food/benefit guidance. Retrieve structured services plus contact and enrollment facts. Never plan enrollment or invent eligibility/contact information.",
    "social": "Plan read-only discovery of social and wellbeing activities for the verified location. Retrieve structured activities and freshness/contact information. Never plan registration or claim the organizer was contacted.",
    "home_support": "Plan recipient home-support state reads and grounded home-support, caregiver, respite, accessibility, or benefit guidance. Only plan a local support request when explicitly requested and when request type, priority, and notes are grounded.",
    "case_status": "Plan only case, audit, reminder, status, and rule-based risk reads. Never plan operational appointment, ride, meal, social, medication, or home-support mutations.",
}
PROMPT = """You are {agent_name}, a SeniorCare coordination specialist. Use MCP read tools for facts.
All operational records and proposed changes are local study simulations. You have no write tools.
Return the AgentResult schema. {action_instructions} Every real proposal must have simulation=true and
requires_approval=true. Never create a placeholder action named none or no_action. Use only the user_id and case_id supplied in the current user
message. Use only the session conversation explicitly supplied for this invocation; never retain
member context inside the reusable agent. Always call get_member and use its
careRecipients list. The current recipientId and selectedCareRecipient are resolved and validated
by deterministic application code before you are invoked. Treat matching family language as positive
confirmation: if relationshipToAccountHolder is father, phrases such as "my father" or "dad" refer to
that selected recipient and must not trigger another ID request. Likewise apply mother/mom,
parent, and spouse terms to their matching registered relationship. Only return needs_user_input for
recipient identity when recipient_resolution_status is not "validated" or the supplied query visibly
names a different relationship than selectedCareRecipient. Never tell the user to avoid familial
terms or to type an ID already supplied in recipient_id. Put the exact recipient_id in every proposed
action and its parameters. Keep user_id/senior_id as the owning account key and recipient_id as the
person receiving care. Never claim a real provider, pharmacy, transit
company, or emergency service was contacted. Do not diagnose, prescribe, or
recommend medication changes. Do not use the word "dummy" in any user-facing summary or action
description; use plain user-facing terms such as appointment, ride, request, or local action instead.
Internal MCP action_type names and simulation metadata remain unchanged. Use structured
provider/API records in findings, but do not turn them into RAG citations. The application derives
retrieved_sources directly from actual public-knowledge tool results.

DOMAIN-SPECIFIC SYNTHESIS INSTRUCTIONS FOR THIS AGENT:
{domain_synthesis_instructions}

DO:
- Use relevantConversationTurns to interpret genuine follow-ups and preserve constraints established in
  earlier turns. Treat the current assigned query as authoritative when the topic changes.
- Base every finding and proposed action parameter on the authenticated context or retrieved data.
- Distinguish operational records, structured datasets/APIs, and RAG guidance in your reasoning.
- Preserve record IDs and citations exactly, and report precisely what information is still absent.
- Propose at most the action justified by the assigned task and complete validated evidence.

DO NOT:
- Re-ask for information already supplied in the query or returned by an MCP read.
- Change, replace, or create a related record unless the assigned request explicitly asks for it.
- Invent identifiers, availability, dates, addresses, eligibility, citations, or successful writes.
- Treat a proposal as executed or contact any real external organization.
"""

SPECIALIST_PLANNING_PROMPT = """You are the planning phase for {agent_name}. Given the assigned
request, authenticated member/recipient context, and allowed MCP read-tool names, return a typed
SpecialistPlan. Select only tools needed to retrieve grounded operational or public-knowledge facts.
Put each selected tool's concrete JSON arguments in tool_arguments, keyed by tool name. Trusted
user_id, senior_id, recipient_id, agent_name, and RAG categories are supplied or overridden by the
application. Do not select write tools. Do not invent tool names, IDs, or missing member data. RAG retrieval must
stay within these categories: {rag_categories}. The next phase will execute the selected tools and
synthesize the result.

The tool_arguments value must be an object keyed by selected tool name. Each value must itself be
an object of arguments. Example:
{{
  "selected_tools": ["search_providers", "list_available_slots"],
  "tool_arguments": {{
    "search_providers": {{"specialty": "orthopedics", "limit": 10}},
    "list_available_slots": {{}}
  }}
}}
Never flatten argument names such as specialty, county, or limit directly under tool_arguments.

DOMAIN-SPECIFIC PLANNING INSTRUCTIONS FOR THIS AGENT:
{domain_planning_instructions}

DO:
- Use relevantConversationTurns to interpret genuine follow-ups and clarification answers. Operational
  IDs and record details must still come from authenticated context or MCP reads.
- Inspect every allowed tool name and JSON schema before selecting tools.
- Use operational tools for member-specific state, structured dataset/API tools for structured
  facts, and RAG tools only for public guidance within the allowed categories.
- Retrieve an existing record before reasoning about its status or proposing a related action.
- Put only schema-supported arguments in tool_arguments and identify genuinely missing user data in
  missing_information.
- Minimize calls while collecting enough evidence for a grounded decision.
- Use missing_information only for user-provided facts that truly block the current request, phrased
  as a clear question. Never put an internal record key such as provider_id, availability_id,
  service_id, or meal_service_id in missing_information; obtain such IDs from MCP results. A provider
  search does not require the user to choose a location when matching providers can already be
  returned; list the matches first and offer location filtering as an optional refinement.

DO NOT:
- Select a write tool, an unlisted tool, or a tool belonging to another specialist.
- Invent IDs, dates, addresses, availability, eligibility, clinical facts, or tool arguments.
- Ask the user for data already present in authenticated operational records.
- Treat structured provider/service records as RAG citations.
- Cross the supplied member, recipient, action, or RAG-category boundary.
"""


class SpecialistGraphState(TypedDict, total=False):
    query: str
    conversation_history: list[dict[str, str]]
    user_id: str
    case_id: str | None
    recipient_id: str | None
    member: dict[str, Any]
    selected_recipient: dict[str, Any]
    task_context: dict[str, Any]
    plan: dict[str, Any]
    retrieval_results: dict[str, Any]
    retrieval_errors: list[dict[str, str]]
    result: dict[str, Any]


class LangGraphSpecialist:
    """Explicit LangGraph specialist subgraph using LangChain model/MCP primitives."""

    def __init__(
        self, key: str, gateway: MCPToolGateway, model: BaseChatModel, configured: bool = True
    ):
        self.key, self.name, self.gateway, self.model = key, AGENT_NAMES[key], gateway, model
        self.configured = configured
        self.guardrail = SpecialistGuardrail()
        self._agent: Any | None = None
        self._tools: dict[str, BaseTool] = {}
        self._initialization_lock = asyncio.Lock()
        self.initialization_error: str | None = None

    @property
    def initialized(self) -> bool:
        return self._agent is not None

    async def initialize(self) -> bool:
        """Discover tools and compile this stateless specialist graph exactly once."""
        if not self.configured:
            self.initialization_error = "LLM model/API key are not configured"
            return False
        if self._agent is not None:
            return True
        async with self._initialization_lock:
            if self._agent is not None:
                return True
            try:
                tools = await self.gateway.get_tools(READ_TOOL_POLICIES[self.key])
                self._tools = {tool.name: tool for tool in tools}
                graph = StateGraph(SpecialistGraphState)
                graph.add_node("specialist_plan_llm", self._planning_node)
                graph.add_node("validate_tool_plan", self._validate_plan_node)
                graph.add_node("execute_mcp_reads", self._execute_reads_node)
                graph.add_node("validate_retrieval", self._validate_retrieval_node)
                graph.add_node("specialist_synthesis_llm", self._synthesis_node)
                graph.add_node("validate_agent_result", self._validate_result_node)
                graph.add_edge(START, "specialist_plan_llm")
                graph.add_edge("specialist_plan_llm", "validate_tool_plan")
                graph.add_edge("validate_tool_plan", "execute_mcp_reads")
                graph.add_edge("execute_mcp_reads", "validate_retrieval")
                graph.add_edge("validate_retrieval", "specialist_synthesis_llm")
                graph.add_edge("specialist_synthesis_llm", "validate_agent_result")
                graph.add_edge("validate_agent_result", END)
                self._agent = graph.compile()
                self.initialization_error = None
            except Exception as exc:
                self.initialization_error = str(exc)
                return False
        return True

    def invalidate(self) -> None:
        """Discard the compiled graph so changed MCP tools can be rediscovered and rebound."""
        self._agent = None
        self._tools = {}
        self.initialization_error = None

    async def run(
        self,
        query: str,
        user_id: str,
        case_id: str | None = None,
        recipient_id: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        task_context: dict[str, Any] | None = None,
    ) -> AgentResult:
        flow_event(
            "agent",
            self.name,
            "input",
            {
                "query": query,
                "userId": user_id,
                "caseId": case_id,
                "recipientId": recipient_id,
                "conversationHistory": conversation_history or [],
            },
        )
        self.guardrail.validate_input(self.key, query, user_id)
        member = await self.gateway.call("get_member", user_id=user_id)
        if not member:
            return AgentResult(
                agent_name=self.name,
                status="needs_user_input",
                summary="A valid SeniorCare User ID is required.",
            )
        if issue := self.guardrail.recipient_issue(query, member, recipient_id):
            return AgentResult(
                agent_name=self.name,
                status="needs_user_input",
                summary=issue,
                warnings=["No action was proposed because the care recipient was not confirmed."],
            )
        recipients = member.get("careRecipients") or [member.get("careRecipient") or {}]
        selected_recipient = next(
            (value for value in recipients if value.get("recipientId") == recipient_id),
            recipients[0] if len(recipients) == 1 else {},
        )
        if self.key == "transportation" and self._is_appointment_transport_request(query):
            confirmation = await self._transportation_appointment_confirmation(
                query,
                user_id,
                member,
                selected_recipient,
                case_id,
                conversation_history or [],
            )
            if confirmation is not None:
                flow_event("agent", self.name, "output", confirmation.model_dump(mode="json"))
                return confirmation
        if not await self.initialize() or self._agent is None:
            reason = self.initialization_error or f"Unable to initialize {self.name}"
            return AgentResult(
                agent_name=self.name,
                status="blocked",
                summary=(
                    "This specialist agent is not configured yet. Set LLM_MODEL and "
                    "OPENAI_API_KEY in .env, then restart seniorcare-api."
                ),
                warnings=[reason],
                confidence=0.0,
            )
        try:
            response = await self._agent.ainvoke(
                {
                    "query": query,
                    "conversation_history": conversation_history or [],
                    "user_id": user_id,
                    "case_id": case_id,
                    "recipient_id": recipient_id,
                    "member": member,
                    "selected_recipient": selected_recipient,
                    "task_context": task_context or {},
                }
            )
        except Exception as exc:
            flow_event("subgraph", self.name, "error", exc)
            raise
        result = AgentResult.model_validate(response["result"])
        self._discard_noop_actions(result)
        if result.proposed_actions and not self._explicit_write_context(
            query, conversation_history or []
        ):
            result.proposed_actions = []
            result.warnings.append(
                "An unsolicited mutation proposal was removed; the request was informational."
            )
        for proposed in result.proposed_actions:
            alias = ACTION_ALIASES.get(self.key, {}).get(proposed.action_type.strip().casefold())
            if alias:
                result.warnings.append(
                    f"Normalized action type {proposed.action_type!r} to {alias!r}."
                )
                proposed.action_type = alias
        for proposed in result.proposed_actions:
            if proposed.agent_name not in {"", self.name}:
                return self._context_conflict("agent identity")
            if proposed.user_id not in {"", user_id}:
                return self._context_conflict("account owner")
            if proposed.senior_id not in {"", user_id}:
                return self._context_conflict("member account")
            if proposed.recipient_id not in {None, recipient_id}:
                return self._context_conflict("care recipient")
            proposed.agent_name = self.name
            proposed.user_id = user_id
            proposed.senior_id = user_id
            proposed.recipient_id = recipient_id
            parameter_owner = proposed.parameters.get("senior_id")
            if parameter_owner not in {None, user_id}:
                return self._context_conflict("action parameter account")
            proposed.parameters["senior_id"] = user_id
            if recipient_id:
                parameter_recipient = proposed.parameters.get("recipient_id")
                if parameter_recipient not in {None, recipient_id}:
                    return self._context_conflict("care recipient parameters")
                proposed.parameters["recipient_id"] = recipient_id
            if proposed.action_type == "book_dummy_appointment":
                await self._complete_appointment_parameters(proposed, query, member)
            elif proposed.action_type == "book_dummy_ride":
                await self._complete_transportation_parameters(
                    proposed,
                    query,
                    member,
                    selected_recipient,
                    case_id,
                )
            required = REQUIRED_ACTION_PARAMETERS.get(proposed.action_type, frozenset())
            missing = sorted(
                name
                for name in required
                if proposed.parameters.get(name) is None or proposed.parameters.get(name) == ""
            )
            if missing:
                action_label = {
                    "book_dummy_appointment": "appointment",
                    "book_dummy_ride": "transportation",
                    "request_dummy_home_support": "home-support request",
                }.get(proposed.action_type, "action")
                if proposed.action_type == "book_dummy_appointment" and {
                    "availability_id",
                    "provider_id",
                }.intersection(missing):
                    return AgentResult(
                        agent_name=self.name,
                        status="partial",
                        summary=(
                            "I found matching provider records, but none currently has a verified "
                            "available appointment slot. No appointment approval was created."
                        ),
                        warnings=["No matching provider and available-slot pair was found."],
                        confidence=0.8,
                    )
                return AgentResult(
                    agent_name=self.name,
                    status="needs_user_input",
                    summary=(
                        f"I could not prepare the {action_label} approval because "
                        f"{', '.join(missing)} could not be resolved from the available records."
                    ),
                    warnings=[f"Incomplete local action: missing {', '.join(missing)}."],
                    confidence=0.0,
                )
            if proposed.action_type == "book_dummy_appointment":
                await self._enrich_appointment_description(proposed, selected_recipient)
            elif proposed.action_type == "book_dummy_ride":
                await self._enrich_transportation_description(proposed, selected_recipient)
        self.guardrail.validate_output(
            self.key,
            self.name,
            result,
            user_id,
            WRITE_ACTIONS[self.key],
            RAG_CATEGORIES[self.key],
            recipient_id,
        )
        flow_event(
            "agent",
            self.name,
            "output",
            {
                "status": result.status,
                "summary": result.summary,
                "actions": [action.action_type for action in result.proposed_actions],
            },
        )
        return result

    async def _planning_node(self, state: SpecialistGraphState) -> dict[str, Any]:
        history = state.get("conversation_history", [])
        task_context = state.get("task_context", {})
        payload = {
            "assignedObjective": task_context.get("assignedObjective", state["query"]),
            "relevantConversationTurns": history[-5:],
            "taskSpecificSummary": task_context.get("taskSpecificSummary", state["query"]),
            "verifiedFacts": {
                **task_context.get("verifiedFacts", {}),
                "member": state["member"],
                "selectedCareRecipient": state["selected_recipient"],
                "userId": state["user_id"],
                "caseId": state.get("case_id"),
                "recipientId": state.get("recipient_id"),
            },
            "dependencyResults": task_context.get("dependencyResults", {}),
            "allowedTools": {name: self._tool_schema(tool) for name, tool in self._tools.items()},
            "constraints": {
                "allowedRagCategories": sorted(RAG_CATEGORIES[self.key]),
                "readOnlyPlanning": True,
                "requireVerifiedIds": True,
                "memberRecipientBoundary": True,
            },
            "responseSchema": SpecialistPlan.model_json_schema(),
        }
        flow_event("llm", f"{self.name}_planning", "input", payload)
        try:
            planner = self.model.with_structured_output(SpecialistPlan, method="function_calling")
            raw_plan = await planner.ainvoke(
                [
                    {
                        "role": "system",
                        "content": SPECIALIST_PLANNING_PROMPT.format(
                            agent_name=self.name,
                            rag_categories=sorted(RAG_CATEGORIES[self.key]),
                            domain_planning_instructions=DOMAIN_PLANNING_INSTRUCTIONS.get(
                                self.key, "Use only relevant allowed read tools."
                            ),
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ]
            )
            plan = SpecialistPlan.model_validate(raw_plan)
            ignored_internal_fields = [
                item for item in plan.missing_information if self._mentions_system_field(item)
            ]
            if ignored_internal_fields:
                plan.missing_information = [
                    item
                    for item in plan.missing_information
                    if not self._mentions_system_field(item)
                ]
                flow_event(
                    "guardrail",
                    f"{self.name}_planning_missing_information",
                    "output",
                    {"ignoredSystemFields": ignored_internal_fields},
                )
            if self.key == "healthcare":
                plan = self._normalize_healthcare_plan(plan, state["query"])
        except Exception as exc:
            flow_event("llm", f"{self.name}_planning", "error", exc)
            # The specialist still operates safely when planning output is unavailable:
            # its startup-bound read-only policy remains the maximum capability set.
            plan = SpecialistPlan(
                task_summary=state["query"],
                selected_tools=self._fallback_tools(),
                confidence=0.5,
            )
            if self.key == "healthcare":
                plan = self._normalize_healthcare_plan(plan, state["query"])
        flow_event("llm", f"{self.name}_planning", "output", plan.model_dump())
        return {"plan": plan.model_dump(mode="json")}

    @staticmethod
    def _missing_field_key(value: str) -> str:
        """Normalize a planner's missing-data label for policy comparison."""
        return value.strip().casefold().replace(" ", "_").rstrip("?:.")

    @classmethod
    def _mentions_system_field(cls, value: str) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
        return any(field in normalized for field in SYSTEM_RESOLVABLE_FIELDS)

    def _validate_plan_node(self, state: SpecialistGraphState) -> dict[str, Any]:
        plan = SpecialistPlan.model_validate(state["plan"])
        flow_event("guardrail", f"{self.name}_plan_validation", "input", plan.model_dump())
        self.guardrail.validate_plan(self.key, plan, READ_TOOL_POLICIES[self.key])
        unknown_arguments = set(plan.tool_arguments).difference(plan.selected_tools)
        if unknown_arguments:
            raise ValueError(
                f"{self.name} supplied arguments for unselected tools: {sorted(unknown_arguments)}"
            )
        flow_event("guardrail", f"{self.name}_plan_validation", "output", {"valid": True})
        return {}

    async def _execute_reads_node(self, state: SpecialistGraphState) -> dict[str, Any]:
        plan = SpecialistPlan.model_validate(state["plan"])
        results: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        flow_event(
            "subgraph",
            f"{self.name}_execute_mcp_reads",
            "input",
            {"tools": plan.selected_tools},
        )
        for tool_name in plan.selected_tools:
            arguments = self._trusted_tool_arguments(tool_name, plan, state, results)
            try:
                results[tool_name] = await self.gateway.call(
                    tool_name, _retry_safe=True, **arguments
                )
            except Exception as exc:
                errors.append({"tool": tool_name, "error": str(exc)[:300]})
                flow_event("subgraph", f"{self.name}_{tool_name}", "error", exc)
        if self.key == "healthcare":
            appointments = results.get("list_appointments")
            if isinstance(appointments, list):
                enriched_appointments: list[dict[str, Any]] = []
                for appointment in appointments:
                    if not isinstance(appointment, dict):
                        continue
                    enriched = dict(appointment)
                    provider_id = appointment.get("providerId")
                    if provider_id:
                        try:
                            provider = await self.gateway.call(
                                "get_provider", _retry_safe=True, provider_id=provider_id
                            )
                            if isinstance(provider, dict):
                                enriched["provider"] = provider
                        except Exception as exc:
                            errors.append({"tool": "get_provider", "error": str(exc)[:300]})
                    enriched_appointments.append(enriched)
                results["list_appointments"] = enriched_appointments
            providers = results.get("search_providers")
            slots = results.get("list_available_slots")
            provider_rows = providers if isinstance(providers, list) else [providers]
            requested_provider = self._provider_name_for_query(state["query"])
            if requested_provider:
                provider_rows = [
                    row
                    for row in provider_rows
                    if isinstance(row, dict)
                    and str(row.get("providerName", "")).casefold() == requested_provider.casefold()
                ]
                results["search_providers"] = provider_rows
            provider_ids = {
                row.get("providerId")
                for row in provider_rows
                if isinstance(row, dict) and row.get("providerId")
            }
            if provider_ids and isinstance(slots, list):
                results["list_available_slots"] = [
                    row
                    for row in slots
                    if isinstance(row, dict) and row.get("providerId") in provider_ids
                ]
        flow_event(
            "subgraph",
            f"{self.name}_execute_mcp_reads",
            "output",
            {"completed": list(results), "errors": errors},
        )
        return {"retrieval_results": results, "retrieval_errors": errors}

    def _validate_retrieval_node(self, state: SpecialistGraphState) -> dict[str, Any]:
        results = state.get("retrieval_results", {})
        flow_event("guardrail", f"{self.name}_retrieval_validation", "input", list(results))
        if not isinstance(results, dict):
            raise ValueError(f"{self.name} retrieval output must be a mapping")
        allowed_categories = RAG_CATEGORIES[self.key]
        for value in results.values():
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                if not isinstance(row, dict) or "category" not in row:
                    continue
                if allowed_categories and row["category"] not in allowed_categories:
                    raise ValueError(
                        f"{self.name} retrieved forbidden RAG category {row['category']!r}"
                    )
        flow_event("guardrail", f"{self.name}_retrieval_validation", "output", {"valid": True})
        return {}

    async def _synthesis_node(self, state: SpecialistGraphState) -> dict[str, Any]:
        plan = SpecialistPlan.model_validate(state["plan"])
        history = state.get("conversation_history", [])
        task_context = state.get("task_context", {})
        payload = {
            "assignedObjective": task_context.get("assignedObjective", state["query"]),
            "relevantConversationTurns": history[-5:],
            "taskSpecificSummary": task_context.get("taskSpecificSummary", state["query"]),
            "verifiedFacts": {
                **task_context.get("verifiedFacts", {}),
                "selectedCareRecipient": state["selected_recipient"],
                "recipientResolutionStatus": "validated",
                "retrievedInformation": state.get("retrieval_results", {}),
            },
            "dependencyResults": task_context.get("dependencyResults", {}),
            "allowedTools": sorted(self._tools),
            "constraints": {
                "allowedRagCategories": sorted(RAG_CATEGORIES[self.key]),
                "allowedWriteProposals": sorted(WRITE_ACTIONS[self.key]),
                "citationsMustComeFromMcp": True,
                "humanApprovalRequiredForWrites": True,
            },
            "specialistPlan": plan.model_dump(mode="json"),
            "retrievalErrors": state.get("retrieval_errors", []),
            "responseSchema": _GeneratedAgentResult.model_json_schema(),
            # Retained as the canonical request used by deterministic validation helpers.
            "query": state["query"],
        }
        flow_event("llm", f"{self.name}_synthesis", "input", payload)
        requested_provider = (
            self._provider_name_for_query(state["query"]) if self.key == "healthcare" else None
        )
        retrieved = state.get("retrieval_results", {})
        provider_rows = retrieved.get("search_providers", [])
        available_slots = retrieved.get("list_available_slots", [])
        grounded_result = self._grounded_read_result(state)
        if grounded_result is not None:
            result = grounded_result
        elif (
            requested_provider
            and self._explicit_write_request(state["query"])
            and isinstance(provider_rows, list)
            and provider_rows
            and isinstance(available_slots, list)
            and not available_slots
        ):
            result = AgentResult(
                agent_name=self.name,
                status="needs_user_input",
                summary=(
                    f"I found {requested_provider}, but there is no verified available "
                    "appointment slot for that provider in the current local records. "
                    "Please choose another orthopedic provider or try again when new slots "
                    "are available."
                ),
                warnings=["No available slot was returned for the requested provider."],
                confidence=1.0,
            )
        elif (
            self.key == "healthcare"
            and self._specialty_for_query(state["query"])
            and "list_available_slots" in retrieved
            and isinstance(provider_rows, list)
            and provider_rows
            and isinstance(available_slots, list)
            and not available_slots
        ):
            provider_names = [
                str(row["providerName"])
                for row in provider_rows
                if isinstance(row, dict) and row.get("providerName")
            ]
            names = ", ".join(provider_names) or "matching providers"
            result = AgentResult(
                agent_name=self.name,
                status="success",
                summary=(
                    f"I found {names}, but there are no verified available appointment "
                    "slots for those providers in the current local records."
                ),
                warnings=["No available slot was returned for the matched providers."],
                confidence=1.0,
            )
        elif plan.missing_information:
            result = AgentResult(
                agent_name=self.name,
                status="needs_user_input",
                summary=" ".join(plan.missing_information),
                warnings=["The specialist retrieval plan requires additional information."],
                confidence=plan.confidence,
            )
        else:
            prompt = PROMPT.format(
                agent_name=self.name,
                action_instructions=(
                    f"Only propose these action types: {sorted(WRITE_ACTIONS[self.key])}."
                    if WRITE_ACTIONS[self.key]
                    else "Do not propose actions; proposed_actions must be an empty list."
                ),
                domain_synthesis_instructions=DOMAIN_INSTRUCTIONS.get(
                    self.key, "Use the supplied retrieved information when relevant."
                ),
            )
            synthesizer = self.model.with_structured_output(
                _GeneratedAgentResult, method="function_calling"
            )
            raw_result = await synthesizer.ainvoke(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ]
            )
            generated = _GeneratedAgentResult.model_validate(
                raw_result.model_dump() if isinstance(raw_result, AgentResult) else raw_result
            )
            result = AgentResult.model_validate(
                generated.model_dump(exclude={"retrieved_sources", "tool_calls"})
            )
            # Citations are security-sensitive provenance, not generative content. The LLM may
            # summarize structured provider/API records in findings, but only an actual RAG tool
            # result can become a RetrievedChunk citation.
        if self.key == "healthcare" and self._explicit_write_request(state["query"]):
            result = self._ensure_grounded_appointment_proposal(result, state)
        # Provenance is always attached from trusted MCP output, including deterministic
        # domain renderers. It is never accepted from model-authored content.
        result.retrieved_sources = (
            []
            if self.key == "healthcare" and self._is_appointment_list_query(state["query"])
            else self._rag_sources(state.get("retrieval_results", {}), state["query"])
        )
        result.tool_calls = [
            ToolCallRecord(tool=name, operation="read", status="completed")
            for name in state.get("retrieval_results", {})
        ] + [
            ToolCallRecord(tool=row["tool"], operation="read", status="failed")
            for row in state.get("retrieval_errors", [])
        ]
        flow_event(
            "llm",
            f"{self.name}_synthesis",
            "output",
            {
                "status": result.status,
                "summary": result.summary,
                "findingCount": len(result.findings),
                "actionCount": len(result.proposed_actions),
                "sourceCount": len(result.retrieved_sources),
            },
        )
        return {"result": result.model_dump(mode="json")}

    def _ensure_grounded_appointment_proposal(
        self, result: AgentResult, state: SpecialistGraphState
    ) -> AgentResult:
        """Recover an omitted proposal using only a verified provider/available-slot pair."""
        if any(
            action.action_type == "book_dummy_appointment" for action in result.proposed_actions
        ):
            return result
        retrieved = state.get("retrieval_results", {})
        raw_providers = retrieved.get("search_providers", [])
        raw_slots = retrieved.get("list_available_slots", [])
        providers = (
            [row for row in raw_providers if isinstance(row, dict)]
            if isinstance(raw_providers, list)
            else []
        )
        slots = (
            [row for row in raw_slots if isinstance(row, dict)]
            if isinstance(raw_slots, list)
            else []
        )
        provider_by_id = {str(row["providerId"]): row for row in providers if row.get("providerId")}
        pair = next(
            (
                (provider_by_id[str(slot["providerId"])], slot)
                for slot in slots
                if slot.get("status") == "available"
                and str(slot.get("providerId") or "") in provider_by_id
                and slot.get("availabilityId")
            ),
            None,
        )
        if pair is None:
            return result
        provider, slot = pair
        provider_name = str(provider.get("providerName") or provider["providerId"])
        schedule = f"{slot.get('availableDate')} at {slot.get('availableTime')}"
        result.status = "success"
        result.summary = (
            f"I found {provider_name} and a verified available appointment on {schedule}. "
            "The appointment is ready for your approval."
        )
        result.proposed_actions = [
            ProposedAction(
                action_id=f"ACT-{uuid4().hex[:12]}",
                action_type="book_dummy_appointment",
                description=f"Appointment with {provider_name} on {schedule}",
                parameters={
                    "provider_id": provider["providerId"],
                    "availability_id": slot["availabilityId"],
                    "appointment_date": slot.get("availableDate"),
                    "appointment_time": slot.get("availableTime"),
                    "reason": state["query"],
                },
                agent_name=self.name,
                user_id=state["user_id"],
                senior_id=state["user_id"],
                recipient_id=state.get("recipient_id"),
                case_id=state.get("case_id"),
            )
        ]
        result.warnings = [
            warning
            for warning in result.warnings
            if "missing" not in warning.casefold() and "unable to find" not in warning.casefold()
        ]
        result.confidence = 1.0
        return result

    def _validate_result_node(self, state: SpecialistGraphState) -> dict[str, Any]:
        result = AgentResult.model_validate(state["result"])
        flow_event("guardrail", f"{self.name}_result_validation", "input", result.model_dump())
        if result.agent_name != self.name or not result.summary.strip():
            raise ValueError(f"{self.name} returned an invalid structured result")
        if any(call.operation != "read" for call in result.tool_calls):
            raise ValueError(f"{self.name} reported a non-read retrieval call")
        flow_event("guardrail", f"{self.name}_result_validation", "output", {"valid": True})
        return {}

    @staticmethod
    def _tool_schema(tool: BaseTool) -> dict[str, Any]:
        schema = tool.args_schema
        if schema is None:
            return {}
        if isinstance(schema, dict):
            return schema
        return schema.model_json_schema()  # type: ignore[union-attr]

    def _fallback_tools(self) -> list[str]:
        dependent = {
            "get_provider",
            "get_available_slot",
            "list_available_slots",
            "find_available_transportation",
        }
        tools = set(READ_TOOL_POLICIES[self.key]).difference(dependent)
        # Provider availability accepts an optional provider filter, so healthcare fallback can
        # safely retrieve all available slots and join them to search results after execution.
        if self.key == "healthcare" and "list_available_slots" in self._tools:
            tools.add("list_available_slots")
        return sorted(tools)

    def _normalize_healthcare_plan(self, plan: SpecialistPlan, query: str) -> SpecialistPlan:
        """Resolve provider names through search tools instead of treating them as IDs."""
        if self._is_appointment_list_query(query):
            plan.selected_tools = [
                tool for tool in ("get_member", "list_appointments") if tool in self._tools
            ]
            plan.tool_arguments = {
                tool: plan.tool_arguments.get(tool, {}) for tool in plan.selected_tools
            }
            plan.missing_information = []
            return plan
        requested_provider = self._provider_name_for_query(query)
        specialty = self._specialty_for_query(query)
        provider_request = bool(
            re.search(r"\b(?:find|search|show|list|book|schedule|need|want)\b", query, re.I)
            and re.search(
                r"\b(?:doctor|provider|physician|specialist|appointment|orthopedic)\b",
                query,
                re.I,
            )
        )
        if not requested_provider and not specialty and not provider_request:
            return plan

        # LLMs commonly place a display name such as ``Dr. Carter`` in an ID field.
        # IDs are trusted only after an MCP search returns them, so retrieve providers and
        # slots broadly and join them in the read-validation node.
        selected = [
            name
            for name in plan.selected_tools
            if name not in {"get_provider", "get_available_slot"}
        ]
        for required in ("search_providers", "list_available_slots"):
            if required in self._tools and required not in selected:
                selected.append(required)
        plan.selected_tools = selected
        plan.tool_arguments.pop("get_provider", None)
        plan.tool_arguments.pop("get_available_slot", None)
        plan.tool_arguments["search_providers"] = {"specialty": specialty, "limit": 10}
        plan.tool_arguments["list_available_slots"] = {}
        plan.missing_information = [
            item for item in plan.missing_information if not self._mentions_system_field(item)
        ]
        return plan

    @staticmethod
    def _provider_name_for_query(query: str) -> str | None:
        # ``doctor`` is also a common noun ("find a doctor for my father").
        # Exclude grammatical words so they cannot become fictitious provider
        # names such as ``Dr. For`` and filter out all valid search results.
        match = re.search(r"\b(?:Dr\.|Doctor)\s+([A-Z][A-Za-z'-]+)", query, flags=re.IGNORECASE)
        if not match:
            return None
        surname = match.group(1)
        if surname.casefold() in {
            "appointment",
            "for",
            "in",
            "near",
            "provider",
            "that",
            "to",
            "with",
        }:
            return None
        return f"Dr. {surname.title()}"

    def _trusted_tool_arguments(
        self,
        tool_name: str,
        plan: SpecialistPlan,
        state: SpecialistGraphState,
        previous_results: dict[str, Any],
    ) -> dict[str, Any]:
        tool = self._tools[tool_name]
        schema = self._tool_schema(tool)
        properties = schema.get("properties", {})
        arguments = dict(plan.tool_arguments.get(tool_name, {}))
        query = state["query"]
        member = state["member"]
        recipient = state["selected_recipient"]
        trusted = {
            "user_id": state["user_id"],
            "senior_id": state["user_id"],
            "recipient_id": state.get("recipient_id"),
            "agent_name": self.name,
        }
        for name, value in trusted.items():
            if name in properties:
                arguments[name] = value
        if "query" in properties and not arguments.get("query"):
            arguments["query"] = query
        if "name" in properties and not arguments.get("name"):
            arguments["name"] = self._medication_name_for_query(query)
        if "categories" in properties:
            arguments["categories"] = sorted(RAG_CATEGORIES[self.key])
        if "county" in properties and not arguments.get("county"):
            arguments["county"] = recipient.get("county") or member.get("county")
        if "specialty" in properties and not arguments.get("specialty"):
            arguments["specialty"] = self._specialty_for_query(query)
        if "limit" in properties and not arguments.get("limit"):
            arguments["limit"] = 10
        if "include_public" in properties:
            arguments["include_public"] = False
        if (
            "provider_id" in properties
            and tool_name != "list_available_slots"
            and not arguments.get("provider_id")
        ):
            arguments["provider_id"] = self._first_value(previous_results, "providerId")
        if "availability_id" in properties and not arguments.get("availability_id"):
            arguments["availability_id"] = self._first_value(previous_results, "availabilityId")
        return {
            name: value
            for name, value in arguments.items()
            if name in properties and value is not None
        }

    def _grounded_read_result(self, state: SpecialistGraphState) -> AgentResult | None:
        """Render sensitive structured reads without allowing synthesis to discard facts."""
        query = state["query"]
        results = state.get("retrieval_results", {})
        if self.key == "healthcare" and self._is_appointment_list_query(query):
            return self._appointment_list_result(results)
        if (
            self.key == "healthcare"
            and self._is_provider_discovery_query(query)
            and not self._explicit_write_request(query)
        ):
            return self._provider_discovery_result(results)
        if self.key == "medication" and self._medication_name_for_query(query):
            return self._medication_reference_result(query, results)
        if self.key == "meals":
            source_result = self._named_source_detail_result(query, results)
            if source_result is not None:
                return source_result
            if self._is_meal_discovery_query(query):
                return self._meal_services_result(results)
        return None

    @staticmethod
    def _is_provider_discovery_query(query: str) -> bool:
        text = query.casefold()
        return bool(
            re.search(r"\b(?:find|search|show|list|other|available)\b", text)
            and re.search(r"\b(?:doctor|provider|physician|specialist|orthopedic)\b", text)
        )

    def _provider_discovery_result(self, results: dict[str, Any]) -> AgentResult:
        """Render verified provider/slot pairs so synthesis cannot discard directory facts."""
        raw_providers = results.get("search_providers", [])
        raw_slots = results.get("list_available_slots", [])
        providers = (
            [row for row in raw_providers if isinstance(row, dict)]
            if isinstance(raw_providers, list)
            else []
        )
        slots = (
            [row for row in raw_slots if isinstance(row, dict)]
            if isinstance(raw_slots, list)
            else []
        )
        slots_by_provider: dict[str, list[dict[str, Any]]] = {}
        for slot in slots:
            provider_id = str(slot.get("providerId") or "")
            if provider_id and slot.get("status") == "available":
                slots_by_provider.setdefault(provider_id, []).append(slot)
        if not providers:
            return AgentResult(
                agent_name=self.name,
                status="partial",
                summary="I did not find a matching provider in the verified local provider records.",
                confidence=1.0,
            )
        lines = ["I found these matching providers in the verified local records:"]
        related_ids: list[str] = []
        for provider in providers[:10]:
            provider_id = str(provider.get("providerId") or "")
            name = str(provider.get("providerName") or provider_id or "Provider name unavailable")
            specialty = provider.get("specialty")
            facility = provider.get("facilityName")
            location = ", ".join(
                str(value)
                for value in (provider.get("city"), provider.get("state"), provider.get("zipCode"))
                if value
            )
            details = [str(value) for value in (specialty, facility, location) if value]
            line = f"- {name} ({provider_id})"
            if details:
                line += " — " + "; ".join(details)
            provider_slots = sorted(
                slots_by_provider.get(provider_id, []),
                key=lambda row: (
                    str(row.get("availableDate", "")),
                    str(row.get("availableTime", "")),
                ),
            )
            if provider_slots:
                rendered_slots = ", ".join(
                    f"{row.get('availableDate')} at {row.get('availableTime')} ({row.get('availabilityId')})"
                    for row in provider_slots[:3]
                )
                line += f". Available: {rendered_slots}."
            else:
                line += ". No verified slot is currently available."
            lines.append(line)
            if provider_id:
                related_ids.append(provider_id)
        return AgentResult(
            agent_name=self.name,
            status="success",
            summary="\n".join(lines),
            related_entity_ids=related_ids,
            confidence=1.0,
        )

    @staticmethod
    def _is_appointment_list_query(query: str) -> bool:
        text = query.casefold()
        return bool(
            re.search(r"\b(?:show|list|view|see|check|what are)\b", text)
            and re.search(
                r"\b(?:existing|current|scheduled|upcoming|my|father|mother|dad|mom)", text
            )
            and re.search(r"\b(?:doctor|provider|medical)?\s*appointments?\b", text)
        )

    def _appointment_list_result(self, results: dict[str, Any]) -> AgentResult:
        rows = results.get("list_appointments", [])
        appointments = (
            [
                row
                for row in rows
                if isinstance(row, dict)
                and str(row.get("status", "")).casefold() not in {"cancelled", "canceled"}
            ]
            if isinstance(rows, list)
            else []
        )
        if not appointments:
            return AgentResult(
                agent_name=self.name,
                status="success",
                summary="I found no stored doctor appointments for this care recipient.",
                confidence=1.0,
            )
        lines: list[str] = ["Here are the stored doctor appointments for this care recipient:"]
        related_ids: list[str] = []
        for row in appointments:
            appointment_id = str(row.get("appointmentId", "Not provided"))
            raw_provider = row.get("provider")
            provider: dict[str, Any] = raw_provider if isinstance(raw_provider, dict) else {}
            provider_name = (
                provider.get("providerName") or row.get("providerName") or row.get("providerId")
            )
            specialty = provider.get("specialty")
            facility = provider.get("facilityName") or provider.get("organizationName")
            location = ", ".join(
                str(value)
                for value in (provider.get("city"), provider.get("state"), provider.get("zipCode"))
                if value
            )
            provider_text = str(provider_name or "Provider not recorded")
            if specialty:
                provider_text += f" ({specialty})"
            place = ", ".join(value for value in (facility, location) if value)
            schedule = (
                " ".join(
                    str(value)
                    for value in (row.get("appointmentDate"), row.get("appointmentTime"))
                    if value
                )
                or "Date/time not recorded"
            )
            reason = row.get("reason") or row.get("appointmentReason") or "Reason not recorded"
            status = str(row.get("status", "unknown")).replace("_", " ").title()
            line = f"- {appointment_id} — {status} — {schedule} — {provider_text}"
            if place:
                line += f" at {place}"
            line += f" — {reason}"
            lines.append(line)
            if appointment_id != "Not provided":
                related_ids.append(appointment_id)
        return AgentResult(
            agent_name=self.name,
            status="success",
            summary="\n".join(lines),
            related_entity_ids=related_ids,
            confidence=1.0,
        )

    @staticmethod
    def _medication_name_for_query(query: str) -> str | None:
        patterns = (
            r"\b(?:information|guidance|reference|safety|details)\s+(?:about|for|on)\s+([a-z][a-z0-9-]{2,})\b",
            r"\b(?:about|regarding)\s+([a-z][a-z0-9-]{2,})\b",
            r"\b([a-z][a-z0-9-]{2,})\s+(?:safety|information|guidance|reference)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, query.casefold())
            if match:
                candidate = match.group(1)
                if candidate not in {"official", "medication", "medicine", "drug", "prescription"}:
                    return candidate
        return None

    def _medication_reference_result(self, query: str, results: dict[str, Any]) -> AgentResult:
        medication = self._medication_name_for_query(query) or "the requested medication"
        value = results.get("search_medication_references", [])
        rows = value if isinstance(value, list) else [value]
        references = [
            row
            for row in rows
            if isinstance(row, dict)
            and any(row.get(field) for field in ("generic_name", "brand_name", "substance_names"))
        ]
        if not references:
            return AgentResult(
                agent_name=self.name,
                status="partial",
                summary=(
                    f"I could not retrieve a medication-specific structured FDA record for "
                    f"{medication.title()}. The retrieved public guidance is general medication "
                    "safety information, so I will not attribute drug-specific warnings to it. "
                    "Please use the current FDA label or ask a licensed pharmacist or clinician."
                ),
                warnings=["No medication-specific structured reference was returned."],
                confidence=1.0,
            )
        lines = [
            f"I found these structured openFDA product records matching {medication.title()}. "
            "These records identify products; they are not prescribing instructions or a complete safety label:"
        ]
        for row in references[:5]:
            name = row.get("brand_name") or row.get("generic_name") or medication.title()
            generic = row.get("generic_name")
            manufacturer = row.get("manufacturer_name")
            form = row.get("dosage_form")
            routes = row.get("route") or []
            ndc = row.get("product_ndc")
            details = [
                f"generic: {generic}" if generic else None,
                f"manufacturer: {manufacturer}" if manufacturer else None,
            ]
            details.extend(
                [
                    f"form: {form}" if form else None,
                    f"route: {', '.join(routes)}" if routes else None,
                    f"NDC: {ndc}" if ndc else None,
                ]
            )
            lines.append(f"- {name} — " + "; ".join(item for item in details if item))
        lines.append(
            "For contraindications, boxed warnings, interactions, or individual advice, consult the "
            "current FDA-approved label and a licensed clinician or pharmacist."
        )
        return AgentResult(
            agent_name=self.name,
            status="success",
            summary="\n".join(lines),
            related_entity_ids=[
                str(row["medication_id"]) for row in references[:5] if row.get("medication_id")
            ],
            confidence=1.0,
        )

    @staticmethod
    def _is_meal_discovery_query(query: str) -> bool:
        text = query.casefold()
        return "meal" in text or "food assistance" in text or "nutrition program" in text

    def _meal_services_result(self, results: dict[str, Any]) -> AgentResult:
        value = results.get("search_meal_services", [])
        rows = [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []
        if not rows:
            return AgentResult(
                agent_name=self.name,
                status="partial",
                summary="I did not find a matching structured meal-assistance service in the local records.",
                confidence=1.0,
            )
        lines = [f"I found {len(rows)} meal-assistance service record(s):"]
        related_ids: list[str] = []
        for row in rows[:10]:
            service_id = str(row.get("mealServiceId", ""))
            name = row.get("serviceName") or "Unnamed meal service"
            service_type = str(row.get("serviceType", "service")).replace("_", " ")
            location = ", ".join(
                str(value)
                for value in (row.get("serviceArea"), row.get("city"), row.get("zipCode"))
                if value
            )
            details = [service_type]
            if location:
                details.append(location)
            if row.get("minimumAge") is not None:
                details.append(f"minimum age {row['minimumAge']}")
            if row.get("deliveryDays"):
                details.append(str(row["deliveryDays"]))
            if row.get("intakeRequired") is not None:
                details.append("intake required" if row["intakeRequired"] else "no intake required")
            lines.append(f"- {name} ({service_id}) — " + "; ".join(details))
            if service_id:
                related_ids.append(service_id)
        if len(rows) > 10:
            lines.append(f"- {len(rows) - 10} additional matching service record(s) are available.")
        lines.append(
            "Contact the program directly to verify current availability and enroll. "
            "The structured local records do not include phone numbers or websites; use the official sources below for current contact and application information."
        )
        return AgentResult(
            agent_name=self.name,
            status="success",
            summary="\n".join(lines),
            related_entity_ids=related_ids,
            confidence=1.0,
        )

    def _named_source_detail_result(
        self, query: str, results: dict[str, Any]
    ) -> AgentResult | None:
        text = query.casefold()
        if not ("more detail" in text or "provide detail" in text or "tell me more" in text):
            return None
        sources = self._rag_sources(results, query)
        named = [source for source in sources if source.source_name.casefold() in text]
        # A displayed citation marker can be followed by its title/source name. Match that
        # human-readable name; never assume SRC1 is stable across requests.
        if not named:
            named = [
                source for source in sources if source.title and source.title.casefold() in text
            ]
        if not named:
            return None
        source = named[0]
        summary = source.content.strip()
        if source.source_url:
            summary += f"\n\nOfficial source: {source.source_name} — {source.source_url}"
        return AgentResult(
            agent_name=self.name,
            status="success",
            summary=summary,
            confidence=1.0,
        )

    @staticmethod
    def _first_value(results: dict[str, Any], field: str) -> Any | None:
        for value in results.values():
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                if isinstance(row, dict) and row.get(field):
                    return row[field]
        return None

    def _rag_sources(self, results: dict[str, Any], query: str = "") -> list[RetrievedChunk]:
        """Build source citations only from server-returned RAG rows."""
        rag_tools = {
            "search_public_knowledge",
            "search_healthcare_knowledge",
            "search_medication_knowledge",
        }
        sources: list[RetrievedChunk] = []
        seen: set[str] = set()
        for tool_name, value in results.items():
            if tool_name not in rag_tools:
                continue
            rows = value if isinstance(value, list) else [value]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    chunk = RetrievedChunk.model_validate(row)
                except ValueError:
                    continue
                if chunk.category not in RAG_CATEGORIES[self.key] or chunk.chunk_id in seen:
                    continue
                if (
                    self.key == "meals"
                    and chunk.category == "benefits_financial"
                    and not re.search(
                        r"\b(?:benefit|financial|snap|commonhelp|food stamps?)\b",
                        query.casefold(),
                    )
                ):
                    continue
                if tool_name not in chunk.retrieved_by:
                    chunk.retrieved_by.append(tool_name)
                seen.add(chunk.chunk_id)
                sources.append(chunk)
        return sources

    @staticmethod
    def _discard_noop_actions(result: AgentResult) -> None:
        """Remove harmless LLM placeholders without weakening real action policies."""
        noop_names = {"none", "no_action", "no action"}
        retained = [
            action
            for action in result.proposed_actions
            if action.action_type.strip().casefold() not in noop_names
        ]
        if len(retained) != len(result.proposed_actions):
            result.warnings.append("An LLM no-action placeholder was ignored safely.")
        result.proposed_actions = retained

    def _explicit_write_request(self, query: str) -> bool:
        """Require explicit mutation language before accepting an LLM-proposed action."""
        patterns = {
            "healthcare": r"\b(?:book|schedule|make|reschedule|cancel)\b.*\bappointment\b",
            "transportation": (
                r"\b(?:book|schedule|arrange|reserve|request|need|want)\b"
                r".*\b(?:ride|transport)"
            ),
            "medication": r"\b(?:request|order|submit|renew)\b.*\brefill\b",
            "meals": r"\b(?:enroll|apply|sign\s*up|register|order)\b",
            "social": r"\b(?:register|sign\s*up|join|enroll)\b",
            "home_support": r"\b(?:request|apply|open|submit|schedule)\b",
            "case_status": r"$^",
        }
        return bool(re.search(patterns[self.key], query.casefold()))

    def _explicit_write_context(
        self, query: str, conversation_history: list[dict[str, str]]
    ) -> bool:
        """Preserve approval intent while the user answers required follow-up questions.

        A booking often spans several turns: the initial user request authorizes a
        proposal, while later turns only provide an address or accessibility answer.
        Those answers must not be mistaken for a new informational request. Only user
        messages are considered, and an actual write still requires human approval.
        """
        if self._explicit_write_request(query):
            return True
        recent_user_messages = [
            str(message.get("content", ""))
            for message in conversation_history[-6:]
            if message.get("role") == "user"
        ]
        return any(self._explicit_write_request(message) for message in recent_user_messages)

    async def _complete_appointment_parameters(
        self, proposed: Any, query: str, member: dict[str, Any]
    ) -> None:
        """Deterministically bind a local provider and open slot to an LLM proposal."""
        parameters = proposed.parameters
        parameters["reason"] = self._user_facing_appointment_reason(
            str(parameters.get("reason") or query)
        )

        provider_id = parameters.get("provider_id")
        availability_id = parameters.get("availability_id")
        if provider_id and availability_id:
            selected_slot = await self.gateway.call(
                "get_available_slot", availability_id=availability_id
            )
            slot_matches = (
                isinstance(selected_slot, dict)
                and selected_slot.get("providerId") == provider_id
                and selected_slot.get("status") == "available"
            )
            if not slot_matches:
                # The model can select IDs, but it cannot establish their relationship. Reject
                # an inconsistent pair and deterministically bind a real provider/slot pair.
                parameters.pop("provider_id", None)
                parameters.pop("availability_id", None)
                provider_id = None
                availability_id = None
        if availability_id and not provider_id:
            slot = await self.gateway.call("get_available_slot", availability_id=availability_id)
            if (
                isinstance(slot, dict)
                and slot.get("providerId")
                and slot.get("status") == "available"
            ):
                provider_id = slot["providerId"]
                parameters["provider_id"] = provider_id

        if provider_id and not availability_id:
            slots = await self.gateway.call("list_available_slots", provider_id=provider_id)
            available = self._first_available_slot(slots)
            if available:
                parameters["availability_id"] = available.get("availabilityId")
            return
        if provider_id:
            return

        specialty = self._specialty_for_query(query)
        providers = await self.gateway.call(
            "search_providers",
            specialty=specialty,
            county=member.get("county"),
            limit=10,
            include_public=False,
        )
        if (not isinstance(providers, list) or not providers) and specialty:
            providers = await self.gateway.call(
                "search_providers",
                specialty=None,
                county=member.get("county"),
                limit=10,
                include_public=False,
            )
        if not isinstance(providers, list):
            return
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            candidate_id = provider.get("providerId")
            if not candidate_id:
                continue
            slots = await self.gateway.call("list_available_slots", provider_id=candidate_id)
            available = self._first_available_slot(slots)
            if available and available.get("availabilityId"):
                parameters["provider_id"] = candidate_id
                parameters["availability_id"] = available["availabilityId"]
                return

    @staticmethod
    def _user_facing_appointment_reason(value: str) -> str:
        """Strip agent-only routing instructions before persisting appointment details."""
        reason = value.strip()
        marker = re.search(r"\bPrevious request:\s*", reason, flags=re.IGNORECASE)
        if marker:
            reason = reason[marker.end() :]
        reason = re.split(
            r"\s*\.?\s*Resolve\s+provider[_ ]id\b",
            reason,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        reason = re.sub(r"\.{2,}$", ".", reason).strip()
        return reason.rstrip(".") or "Doctor appointment"

    @staticmethod
    def _first_available_slot(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, list):
            return None
        return next(
            (
                row
                for row in value
                if isinstance(row, dict)
                and row.get("availabilityId")
                and row.get("status") == "available"
            ),
            None,
        )

    @staticmethod
    def _specialty_for_query(query: str) -> str | None:
        normalized = query.casefold()
        specialty_terms = {
            "orthopedics": (
                "knee",
                "shoulder",
                "hip",
                "joint",
                "bone",
                "leg",
                "orthopedic",
            ),
            "cardiology": ("heart", "cardiac", "cardiolog"),
            "neurology": ("neurolog", "migraine", "memory"),
            "ophthalmology": ("eye", "vision", "ophthalmolog"),
            "primary care": ("primary care", "general check", "checkup"),
        }
        return next(
            (
                specialty
                for specialty, terms in specialty_terms.items()
                if any(term in normalized for term in terms)
            ),
            None,
        )

    async def _complete_transportation_parameters(
        self,
        proposed: Any,
        query: str,
        member: dict[str, Any],
        recipient: dict[str, Any],
        case_id: str | None,
    ) -> None:
        """Bind transportation to the selected recipient's active-case appointment."""
        recipient_id = recipient.get("recipientId")
        appointments = await self.gateway.call("list_appointments", user_id=member["seniorId"])
        if not isinstance(appointments, list):
            return
        eligible = [
            row
            for row in appointments
            if isinstance(row, dict)
            and row.get("status") in {"scheduled", "pending_confirmation"}
            and (
                row.get("recipientId") == recipient_id
                or (
                    recipient.get("isAccountHolder")
                    and row.get("recipientId") in {None, recipient_id}
                )
            )
        ]
        requested_ids = self._appointment_ids(query)
        related_ids: set[str] = set()
        if case_id:
            case_value = await self.gateway.call(
                "get_case", user_id=member["seniorId"], case_id=case_id
            )
            if isinstance(case_value, dict) and isinstance(case_value.get("data"), dict):
                case_value = case_value["data"]
            if isinstance(case_value, dict):
                related_ids = set(case_value.get("relatedEntityIds") or [])
        appointment: dict[str, Any] | None = next(
            (row for row in eligible if row.get("appointmentId") in requested_ids),
            None,
        )
        if appointment is None:
            appointment = next(
                (row for row in reversed(eligible) if row.get("appointmentId") in related_ids),
                eligible[-1] if eligible else None,
            )
        if not appointment:
            return

        wheelchair = self._wheelchair_requirement(query)
        round_trip = self._round_trip_requirement(query)
        appointment_time = str(appointment.get("appointmentTime") or "")
        appointment_date = str(appointment.get("appointmentDate") or "")
        provider = await self.gateway.call(
            "get_provider", provider_id=appointment.get("providerId")
        )
        provider = provider if isinstance(provider, dict) else {}
        destination_address = self._provider_address(provider)
        pickup_address = self._pickup_address(query)
        if (
            not destination_address
            or not pickup_address
            or wheelchair is None
            or round_trip is None
        ):
            return
        plan = await self.gateway.call(
            "find_available_transportation",
            destination_address=destination_address,
            appointment_time=appointment_time,
            appointment_date=appointment_date,
            pickup_address=pickup_address,
            wheelchair_required=wheelchair,
        )
        if isinstance(plan, dict) and isinstance(plan.get("data"), dict):
            plan = plan["data"]
        if not isinstance(plan, dict) or not plan.get("available"):
            return
        proposed.parameters.update(
            {
                "appointment_id": appointment["appointmentId"],
                "service_id": plan["transportationServiceId"],
                "pickup_date": plan["pickupDate"],
                "pickup_time": plan["pickupTime"],
                "pickup_address": pickup_address,
                "destination_address": destination_address,
                "appointment_date": appointment_date,
                "appointment_time": appointment_time,
                "wheelchair_required": wheelchair,
                "vehicle_id": plan["vehicleId"],
                "estimated_travel_minutes": plan["estimatedTravelMinutes"],
                "accommodation": "wheelchair"
                if wheelchair
                else member.get("mobilityNeeds") or "none",
                "return_ride_required": round_trip,
            }
        )

    async def _enrich_transportation_description(
        self, proposed: Any, recipient: dict[str, Any]
    ) -> None:
        appointments = await self.gateway.call(
            "list_appointments", user_id=proposed.parameters["senior_id"]
        )
        appointment: dict[str, Any] = next(
            (
                row
                for row in appointments
                if row.get("appointmentId") == proposed.parameters["appointment_id"]
            ),
            {},
        )
        services = await self.gateway.call(
            "search_transportation_services", county=None, wheelchair_accessible=None
        )
        service: dict[str, Any] = next(
            (
                row
                for row in services
                if row.get("transportationServiceId") == proposed.parameters["service_id"]
            ),
            {},
        )
        provider = (
            await self.gateway.call("get_provider", provider_id=appointment.get("providerId"))
            if appointment.get("providerId")
            else {}
        )
        provider = provider if isinstance(provider, dict) else {}
        destination = ", ".join(
            str(value)
            for value in (
                provider.get("facilityName"),
                provider.get("city"),
                provider.get("state"),
                provider.get("zipCode"),
            )
            if value
        )
        recipient_name = (
            f"{recipient.get('firstName', '')} {recipient.get('lastName', '')}".strip()
            or "the selected care recipient"
        )
        trip_type = "round-trip" if proposed.parameters.get("return_ride_required") else "one-way"
        proposed.description = (
            f"{trip_type.title()} transportation for {recipient_name} with "
            f"{service.get('serviceName', proposed.parameters['service_id'])}, pickup from "
            f"{proposed.parameters['pickup_address']} on "
            f"{proposed.parameters['pickup_date']} at {proposed.parameters['pickup_time']} for "
            f"appointment {appointment.get('appointmentId', proposed.parameters['appointment_id'])} "
            f"at {appointment.get('appointmentTime', 'time not listed')}"
            f" at {proposed.parameters.get('destination_address') or destination}; vehicle "
            f"{proposed.parameters['vehicle_id']}; estimated travel time "
            f"{proposed.parameters['estimated_travel_minutes']} minutes; wheelchair required: "
            f"{proposed.parameters['wheelchair_required']}; accommodation: "
            f"{proposed.parameters['accommodation']}."
        )

    @staticmethod
    def _is_booking_request(query: str) -> bool:
        normalized = query.casefold()
        return any(term in normalized for term in ("book", "arrange", "schedule"))

    @classmethod
    def _is_appointment_transport_request(cls, query: str) -> bool:
        normalized = query.casefold()
        return bool(cls._appointment_ids(query)) or (
            "appointment" in normalized
            and (
                cls._is_booking_request(query)
                or any(term in normalized for term in ("transport", "ride", "pickup", "pick up"))
            )
        )

    @staticmethod
    def _appointment_ids(query: str) -> set[str]:
        return {value.upper() for value in re.findall(r"\bAPT[\w-]*\b", query, re.IGNORECASE)}

    @staticmethod
    def _pickup_address(query: str) -> str | None:
        patterns = (
            r"\bfrom\s+\[([^\]]+)\]",
            r"\bfrom\s+(\d+\s+.*?\b\d{5}(?:-\d{4})?)\b",
            r"\bpickup/home address(?: is|:)?\s+(\d+\s+.*?\b\d{5}(?:-\d{4})?)\b",
            r"\bpickup address(?: is|:)?\s+(.+)$",
            r"\brecipient(?:'s)?\s+(?:home|pickup) address(?: is|:)?\s+(.+)$",
            r"\bhome address(?: is|:)?\s+(.+)$",
            r"\bpick(?:\s+me|\s+him|\s+her|\s+them)?\s+up\s+(?:from|at)\s+(.+)$",
            r"\bfrom\s+(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, query.strip(), re.IGNORECASE)
            if not match:
                continue
            value = re.split(
                r"\s+(?:(?:to|for)\s+(?:appointment|the appointment|hospital|doctor)\b|"
                r"in\s+a\s+wheelchair\b|and\s+(?:drop|return)\b)",
                match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,?")
            if any(character.isdigit() for character in value) and len(value) >= 8:
                return value
        return None

    @staticmethod
    def _provider_address(provider: dict[str, Any]) -> str | None:
        value = ", ".join(
            str(item)
            for item in (
                provider.get("facilityName"),
                provider.get("addressLine1"),
                provider.get("addressLine2"),
                provider.get("city"),
                provider.get("state"),
                provider.get("zipCode"),
            )
            if item
        )
        return value or None

    @staticmethod
    def _wheelchair_requirement(query: str) -> bool | None:
        """Return an explicit trip-level wheelchair choice; absence remains unknown."""
        normalized = re.sub(r"\s+", " ", query.casefold()).strip()
        negative_patterns = (
            r"\bno\s+wheelchair(?:\s+assistance)?\b",
            r"\bwithout\s+(?:a\s+)?wheelchair(?:\s+assistance)?\b",
            r"\b(?:does\s+not|doesn't|do\s+not|don't|not)\s+(?:need|require)\s+"
            r"(?:a\s+)?wheelchair(?:\s+assistance)?\b",
            r"\bwheelchair(?:\s+assistance)?\s+(?:is\s+)?not\s+(?:needed|required)\b",
        )
        if any(re.search(pattern, normalized) for pattern in negative_patterns):
            return False
        if re.search(r"\bwheelchair(?:\s+assistance)?\b", normalized):
            return True
        return None

    @staticmethod
    def _round_trip_requirement(query: str) -> bool | None:
        """Return an explicit one-way/round-trip choice; absence remains unknown."""
        normalized = re.sub(r"\s+", " ", query.casefold()).strip()
        if re.search(r"\b(?:one[ -]way|no return(?: ride)?|drop[- ]?off only)\b", normalized):
            return False
        if any(
            phrase in normalized
            for phrase in (
                "round trip",
                "round-trip",
                "drop back",
                "return ride",
                "pickup and drop",
                "pick up and drop",
            )
        ):
            return True
        return None

    async def _transportation_appointment_confirmation(
        self,
        query: str,
        user_id: str,
        member: dict[str, Any],
        recipient: dict[str, Any],
        case_id: str | None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AgentResult | None:
        """Resolve the appointment and require a recipient pickup address."""
        appointments = await self.gateway.call("list_appointments", user_id=user_id)
        if not isinstance(appointments, list):
            appointments = []
        recipient_id = recipient.get("recipientId")
        eligible = [
            row
            for row in appointments
            if isinstance(row, dict)
            and row.get("status") in {"scheduled", "pending_confirmation"}
            and (
                row.get("recipientId") == recipient_id
                or (
                    recipient.get("isAccountHolder")
                    and row.get("recipientId") in {None, recipient_id}
                )
            )
        ]
        requested_ids = self._appointment_ids(query)
        eligible_ids = {str(row.get("appointmentId")) for row in eligible}
        recipient_name = (
            f"{recipient.get('firstName', '')} {recipient.get('lastName', '')}".strip()
            or "the selected care recipient"
        )
        if requested_ids and not requested_ids <= eligible_ids:
            return AgentResult(
                agent_name=self.name,
                status="needs_user_input",
                summary=(
                    f"The appointment tracking ID {', '.join(sorted(requested_ids))} is not an "
                    f"eligible scheduled appointment for {recipient_name}. Please choose one of: "
                    f"{self._appointment_choices(eligible)}"
                ),
                warnings=["No transportation action was proposed."],
                confidence=1.0,
            )
        if not eligible:
            return AgentResult(
                agent_name=self.name,
                status="needs_user_input",
                summary=(
                    f"I could not find an eligible scheduled appointment for {recipient_name}. "
                    "Please book or confirm the appointment before requesting transportation."
                ),
                warnings=["No transportation action was proposed."],
                confidence=1.0,
            )
        # The appointment ID is an explicit cross-domain confirmation. Never infer it, even when
        # only one appointment currently exists; the member may be referring to another booking.
        selected = next(
            (row for row in eligible if row.get("appointmentId") in requested_ids),
            None,
        )
        if selected is not None:
            pickup_address = self._pickup_address(query)
            wheelchair_required = self._wheelchair_requirement(query)
            round_trip_required = self._round_trip_requirement(query)
            if (
                pickup_address
                and wheelchair_required is not None
                and round_trip_required is not None
            ):
                action = self._transportation_action(
                    query, user_id, recipient_id, case_id, selected, pickup_address
                )
                await self._complete_transportation_parameters(
                    action, query, member, recipient, case_id
                )
                missing = sorted(
                    name
                    for name in REQUIRED_ACTION_PARAMETERS["book_dummy_ride"]
                    if action.parameters.get(name) in {None, ""}
                )
                if missing:
                    return AgentResult(
                        agent_name=self.name,
                        status="needs_user_input",
                        summary=(
                            "I found the appointment and pickup address, but could not find an "
                            "available transportation option that satisfies the trip requirements."
                        ),
                        warnings=[f"Missing transportation fields: {', '.join(missing)}."],
                        confidence=1.0,
                    )
                await self._enrich_transportation_description(action, recipient)
                result = await self._transportation_llm_decision(
                    query, recipient, selected, action, conversation_history or []
                )
                self.guardrail.validate_output(
                    self.key,
                    self.name,
                    result,
                    user_id,
                    WRITE_ACTIONS[self.key],
                    RAG_CATEGORIES[self.key],
                    recipient_id,
                )
                return result
            provider = await self.gateway.call(
                "get_provider", provider_id=selected.get("providerId")
            )
            destination = self._provider_address(provider if isinstance(provider, dict) else {})
            missing_questions: list[str] = []
            if not pickup_address:
                missing_questions.append("the recipient's full pickup/home address")
            if wheelchair_required is None:
                missing_questions.append(
                    "whether the recipient requires wheelchair assistance (yes or no)"
                )
            if round_trip_required is None:
                missing_questions.append("whether the ride is round trip (yes or no)")
            return AgentResult(
                agent_name=self.name,
                status="needs_user_input",
                summary=(
                    f"I found appointment {selected.get('appointmentId')} for {recipient_name} on "
                    f"{selected.get('appointmentDate')} at {selected.get('appointmentTime')}"
                    f"{f' at {destination}' if destination else ''}. Before I prepare the ride, "
                    f"please provide {' and '.join(missing_questions)}. Reply with, for example, "
                    f'"Book round-trip transportation for {selected.get("appointmentId")} from '
                    "123 Main Street, Richmond, VA 23220; wheelchair assistance: yes; "
                    'round trip: yes."'
                ),
                warnings=[
                    "Pickup address plus explicit wheelchair-assistance and round-trip choices "
                    "are required before transportation approval."
                ],
                confidence=1.0,
            )
        return AgentResult(
            agent_name=self.name,
            status="needs_user_input",
            summary=(
                f"Please confirm which appointment for {recipient_name} needs transportation. "
                f"Available appointments: {self._appointment_choices(eligible)}. Reply with, "
                'for example, "Book round-trip transportation for APT1021."'
            ),
            warnings=["Appointment tracking ID confirmation is required before approval."],
            confidence=1.0,
        )

    async def _transportation_llm_decision(
        self,
        query: str,
        recipient: dict[str, Any],
        appointment: dict[str, Any],
        trusted_action: ProposedAction,
        conversation_history: list[dict[str, str]],
    ) -> AgentResult:
        """Let the specialist explain/decide using a fully resolved, trusted trip context."""
        fallback = AgentResult(
            agent_name=self.name,
            status="success",
            case_id=trusted_action.case_id,
            summary=trusted_action.description,
            proposed_actions=[trusted_action],
            related_entity_ids=[str(appointment.get("appointmentId"))],
            confidence=1.0,
        )
        if not self.configured:
            return fallback
        payload = {
            "assignedObjective": query,
            "relevantConversationTurns": conversation_history[-5:],
            "taskSpecificSummary": "Validate and explain transportation for the referenced booking.",
            "verifiedFacts": {
                "selectedCareRecipient": recipient,
                "appointment": appointment,
                "resolvedTransportation": trusted_action.parameters,
            },
            "dependencyResults": {},
            "allowedTools": [],
            "constraints": {
                "useResolvedTransportationOnly": True,
                "humanApprovalRequired": True,
                "doNotInventOrReaskVerifiedFacts": True,
            },
            "approvalAction": trusted_action.model_dump(mode="json"),
            "responseSchema": AgentResult.model_json_schema(),
        }
        flow_event("llm", f"{self.name}_transport_decision", "input", payload)
        try:
            decision_model = self.model.with_structured_output(
                AgentResult, method="function_calling"
            )
            raw = await decision_model.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the TransportationAgent decision phase. All appointment, "
                            "recipient, address, accessibility, timing, service, and vehicle data "
                            "in the payload has already been retrieved and validated by MCP-backed "
                            "application code. Decide whether the requested local transportation "
                            "can be proposed. Do not ask for appointment date, time, destination, "
                            "provider, or recipient again. If the resolved transportation is "
                            "complete, return status success and explain the proposed trip. Do not "
                            "use the words dummy or simulated."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ]
            )
            decision = AgentResult.model_validate(raw)
            decision.agent_name = self.name
            decision.case_id = trusted_action.case_id
            # IDs and relational fields are never accepted from generated output. A positive LLM
            # decision receives the already validated local action verbatim.
            if decision.status in {"success", "partial"}:
                decision.proposed_actions = [trusted_action]
                decision.related_entity_ids = [str(appointment.get("appointmentId"))]
            else:
                decision.proposed_actions = []
            decision.retrieved_sources = []
            flow_event(
                "llm",
                f"{self.name}_transport_decision",
                "output",
                decision.model_dump(mode="json"),
            )
            return decision
        except Exception as exc:
            flow_event("llm", f"{self.name}_transport_decision", "error", exc)
            return fallback

    def _transportation_action(
        self,
        query: str,
        user_id: str,
        recipient_id: str | None,
        case_id: str | None,
        appointment: dict[str, Any],
        pickup_address: str,
    ) -> ProposedAction:
        wheelchair = self._wheelchair_requirement(query)
        round_trip = self._round_trip_requirement(query)
        return ProposedAction(
            action_id=f"ACT-{uuid4().hex[:12]}",
            action_type="book_dummy_ride",
            description="Transportation request",
            parameters={
                "senior_id": user_id,
                "recipient_id": recipient_id,
                "appointment_id": appointment.get("appointmentId"),
                "pickup_address": pickup_address,
                "wheelchair_required": wheelchair,
                "return_ride_required": round_trip,
            },
            agent_name=self.name,
            user_id=user_id,
            senior_id=user_id,
            recipient_id=recipient_id,
            case_id=case_id,
        )

    @staticmethod
    def _appointment_choices(appointments: list[dict[str, Any]]) -> str:
        return (
            "; ".join(
                f"{row.get('appointmentId')} on {row.get('appointmentDate')} at "
                f"{row.get('appointmentTime')} ({row.get('reason', 'reason not listed')})"
                for row in appointments
            )
            or "none"
        )

    async def _enrich_appointment_description(
        self, proposed: Any, recipient: dict[str, Any]
    ) -> None:
        provider = await self.gateway.call(
            "get_provider", provider_id=proposed.parameters["provider_id"]
        )
        slot = await self.gateway.call(
            "get_available_slot", availability_id=proposed.parameters["availability_id"]
        )
        if not isinstance(provider, dict) or not isinstance(slot, dict):
            raise RuntimeError("Selected appointment provider or availability is no longer valid")
        if slot.get("providerId") != proposed.parameters["provider_id"]:
            raise RuntimeError("Selected appointment slot does not belong to the provider")
        recipient_name = (
            f"{recipient.get('firstName', '')} {recipient.get('lastName', '')}".strip()
            or "the selected care recipient"
        )
        location = ", ".join(
            str(value)
            for value in (
                provider.get("facilityName"),
                provider.get("city"),
                provider.get("state"),
                provider.get("zipCode"),
            )
            if value
        )
        proposed.description = (
            f"Appointment for {recipient_name} with "
            f"{provider.get('providerName', proposed.parameters['provider_id'])} "
            f"({provider.get('specialty', 'specialty not listed')}) at {location or 'location not listed'} "
            f"on {slot.get('availableDate')} at {slot.get('availableTime')} for "
            f"{proposed.parameters['reason']}."
        )

    def _context_conflict(self, field: str) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            status="needs_user_input",
            summary=(
                "The proposed action conflicted with the validated request context. "
                "Please confirm who this request is for."
            ),
            warnings=[f"A conflicting {field} was rejected before approval."],
            confidence=0.0,
        )
