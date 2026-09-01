import asyncio
import json
from pathlib import Path

import pytest

from seniorcare_agents.agents.llm_specialist import (
    AGENT_NAMES,
    RAG_CATEGORIES,
    READ_TOOL_POLICIES,
    LangGraphSpecialist,
)
from seniorcare_agents.agents.orchestrator import SeniorCareOrchestratorAgent
from seniorcare_agents.api.app import contextualize_followup
from seniorcare_agents.evals import CodeBasedAgentEvaluator, HumanEvaluationStore
from seniorcare_agents.graph.router import route_intents
from seniorcare_agents.guardrails import (
    AgentGuardrailError,
    OrchestratorGuardrail,
    SpecialistGuardrail,
    ToolGuardrail,
)
from seniorcare_agents.mcp.gateway import MCPToolGateway
from seniorcare_agents.models import (
    AgentResult,
    ExecutionStage,
    OrchestratorPlan,
    OrchestratorResponse,
    ProposedAction,
    SpecialistPlan,
    ToolCallRecord,
)
from seniorcare_agents.services.session_store import AgentSession, ChatMessage
from seniorcare_runtime.config import RuntimeSettings


def safe_result(agent_name: str = "HealthcareAccessAgent") -> AgentResult:
    return AgentResult(agent_name=agent_name, status="success", summary="Found local records.")


def test_specialist_input_guardrail_validates_query_and_member():
    guardrail = SpecialistGuardrail()
    guardrail.validate_input("healthcare", "Find a doctor", "SEN1001")
    with pytest.raises(AgentGuardrailError):
        guardrail.validate_input("healthcare", "", "SEN1001")
    with pytest.raises(AgentGuardrailError):
        guardrail.validate_input("healthcare", "Find a doctor", "Robert")


def test_specialist_plan_guardrail_enforces_read_tool_policy():
    guardrail = SpecialistGuardrail()
    valid = SpecialistPlan(
        task_summary="Find a provider",
        selected_tools=["search_providers", "search_healthcare_knowledge"],
    )
    guardrail.validate_plan("healthcare", valid, READ_TOOL_POLICIES["healthcare"])
    invalid = SpecialistPlan(
        task_summary="Book immediately", selected_tools=["book_dummy_appointment"]
    )
    with pytest.raises(AgentGuardrailError, match="forbidden tools"):
        guardrail.validate_plan("healthcare", invalid, READ_TOOL_POLICIES["healthcare"])


def test_named_provider_booking_plan_resolves_name_through_mcp_search():
    specialist = LangGraphSpecialist(
        "healthcare",
        object(),
        object(),  # type: ignore[arg-type]
    )
    specialist._tools = {  # noqa: SLF001
        "search_providers": object(),
        "list_available_slots": object(),
        "get_provider": object(),
        "get_available_slot": object(),
    }
    plan = SpecialistPlan(
        task_summary="Book with Dr. Carter",
        selected_tools=["get_provider", "get_available_slot"],
        tool_arguments={
            "get_provider": {"provider_id": "Dr. Carter"},
            "get_available_slot": {"availability_id": "Dr. Carter"},
        },
        missing_information=["provider_id", "availability_id"],
    )

    normalized = specialist._normalize_healthcare_plan(  # noqa: SLF001
        plan,
        "Book a doctor appointment with Dr. Carter for knee pain",
    )

    assert normalized.selected_tools == ["search_providers", "list_available_slots"]
    assert normalized.tool_arguments["search_providers"]["specialty"] == "orthopedics"
    assert normalized.tool_arguments["list_available_slots"] == {}
    assert normalized.missing_information == []


def test_healthcare_fallback_plan_retrieves_provider_availability():
    specialist = LangGraphSpecialist(
        "healthcare",
        object(),
        object(),  # type: ignore[arg-type]
    )
    specialist._tools = {  # noqa: SLF001
        "search_providers": object(),
        "list_available_slots": object(),
        "search_healthcare_knowledge": object(),
    }
    plan = SpecialistPlan(
        task_summary="Find an orthopedic doctor",
        selected_tools=specialist._fallback_tools(),  # noqa: SLF001
    )

    normalized = specialist._normalize_healthcare_plan(  # noqa: SLF001
        plan, "Find an orthopedic doctor for knee pain"
    )

    assert "search_providers" in normalized.selected_tools
    assert "list_available_slots" in normalized.selected_tools
    assert normalized.tool_arguments["search_providers"]["specialty"] == "orthopedics"


def test_verified_provider_discovery_cannot_be_discarded_by_synthesis():
    specialist = LangGraphSpecialist(
        "healthcare",
        object(),
        object(),  # type: ignore[arg-type]
    )
    result = specialist._provider_discovery_result(  # noqa: SLF001
        {
            "search_providers": [
                {
                    "providerId": "PRV1003",
                    "providerName": "Dr. Carter",
                    "specialty": "Orthopedics",
                    "facilityName": "Central Virginia Orthopedics Center",
                    "city": "Henrico",
                    "state": "VA",
                    "zipCode": "23229",
                }
            ],
            "list_available_slots": [
                {
                    "availabilityId": "AVL1021",
                    "providerId": "PRV1003",
                    "availableDate": "2026-09-21",
                    "availableTime": "10:30",
                    "status": "available",
                }
            ],
        }
    )

    assert result.status == "success"
    assert "Dr. Carter" in result.summary
    assert "AVL1021" in result.summary
    assert result.related_entity_ids == ["PRV1003"]


def test_explicit_booking_recovers_grounded_proposal_when_llm_omits_it():
    specialist = LangGraphSpecialist(
        "healthcare",
        object(),
        object(),  # type: ignore[arg-type]
    )
    result = specialist._ensure_grounded_appointment_proposal(  # noqa: SLF001
        AgentResult(
            agent_name="HealthcareAccessAgent",
            status="needs_user_input",
            summary="What type of orthopedic care is needed?",
        ),
        {
            "query": "Please book a doctor appointment for my father's leg pain",
            "user_id": "SEN1022",
            "recipient_id": "SEN1022",
            "case_id": None,
            "retrieval_results": {
                "search_providers": [
                    {
                        "providerId": "PRV1003",
                        "providerName": "Dr. Carter",
                    }
                ],
                "list_available_slots": [
                    {
                        "availabilityId": "AVL1021",
                        "providerId": "PRV1003",
                        "availableDate": "2026-09-21",
                        "availableTime": "10:30",
                        "status": "available",
                    }
                ],
            },
        },
    )

    assert result.status == "success"
    assert len(result.proposed_actions) == 1
    action = result.proposed_actions[0]
    assert action.parameters["provider_id"] == "PRV1003"
    assert action.parameters["availability_id"] == "AVL1021"
    assert action.requires_approval is True


def test_provider_name_extraction_is_case_insensitive_and_canonical():
    assert (
        LangGraphSpecialist._provider_name_for_query(  # noqa: SLF001
            "Please book with dr. carter"
        )
        == "Dr. Carter"
    )


def test_provider_name_extraction_ignores_generic_doctor_phrase():
    assert (
        LangGraphSpecialist._provider_name_for_query(  # noqa: SLF001
            "Find an orthopedic doctor for my father's knee pain."
        )
        is None
    )


def test_internal_provider_resolution_instructions_are_not_persisted_as_reason():
    value = (
        "Book a doctor/provider appointment using the first verified pair. "
        "Previous request: Find an orthopedic doctor for my father's knee pain.. "
        "Resolve provider_id and availability_id through MCP records."
    )

    assert LangGraphSpecialist._user_facing_appointment_reason(value) == (  # noqa: SLF001
        "Find an orthopedic doctor for my father's knee pain"
    )


def test_mcp_gateway_restores_singleton_list_tool_contracts():
    appointment = {"appointmentId": "APT1021", "status": "scheduled"}

    assert MCPToolGateway._restore_list_contract(  # noqa: SLF001
        "list_appointments", appointment
    ) == [appointment]
    assert MCPToolGateway._restore_list_contract(  # noqa: SLF001
        "search_providers", appointment
    ) == [appointment]
    assert (
        MCPToolGateway._restore_list_contract(  # noqa: SLF001
            "get_member", appointment
        )
        == appointment
    )
    assert MCPToolGateway._restore_list_contract(  # noqa: SLF001
        "evaluate_risks", appointment
    ) == [appointment]
    assert MCPToolGateway._restore_list_contract(  # noqa: SLF001
        "list_knowledge_chunk_ids", "CHK1001"
    ) == ["CHK1001"]
    assert MCPToolGateway._restore_list_contract("list_cases", None) == []  # noqa: SLF001
    assert (
        LangGraphSpecialist._provider_name_for_query(  # noqa: SLF001
            "Please book with doctor Carter"
        )
        == "Dr. Carter"
    )


def test_transport_followup_carries_appointment_and_address_without_zip():
    session = AgentSession(session_id="SESSION-TRANSPORT", user_id="SEN1022")
    session.messages.extend(
        [
            ChatMessage(
                role="user",
                content="I need transportation for appointment APT1023",
            ),
            ChatMessage(
                role="assistant",
                content=(
                    "I found appointment APT1023. Please provide the full pickup/home "
                    "address, wheelchair assistance yes/no, and round trip yes/no."
                ),
            ),
        ]
    )

    contextual = contextualize_followup(
        "home address : 7743 Halifax Drive, Mechanicsville, VA and wheelchair "
        "assistance = yes and round trip= yes",
        session,
    )

    assert contextual == (
        "Book round-trip transportation for APT1023 from "
        "7743 Halifax Drive, Mechanicsville, VA; wheelchair assistance: yes; "
        "round trip: yes."
    )


def test_specialist_derives_citations_only_from_actual_rag_results():
    specialist = LangGraphSpecialist(
        "healthcare",
        object(),
        object(),  # type: ignore[arg-type]
    )
    sources = specialist._rag_sources(  # noqa: SLF001
        {
            "search_providers": [
                {
                    "providerId": "PRV1001",
                    "category": "Healthcare Provider",
                    "source_name": "Generated Provider Directory",
                }
            ],
            "search_healthcare_knowledge": [
                {
                    "chunk_id": "CHK1001",
                    "document_id": "DOC1001",
                    "content": "Official Medicare healthcare-access guidance.",
                    "category": "healthcare_access",
                    "source_name": "Medicare",
                }
            ],
        }
    )

    assert [source.chunk_id for source in sources] == ["CHK1001"]
    assert sources[0].retrieved_by == ["search_healthcare_knowledge"]


@pytest.mark.asyncio
async def test_appointment_binding_replaces_mismatched_provider_and_slot():
    class Gateway:
        async def call(self, tool_name, **arguments):
            if tool_name == "get_available_slot":
                return {
                    "availabilityId": arguments["availability_id"],
                    "providerId": "PRV9999",
                    "status": "available",
                }
            if tool_name == "search_providers":
                return [
                    {"providerId": "PRV1003"},
                    {"providerId": "PRV1013"},
                ]
            if tool_name == "list_available_slots":
                if arguments["provider_id"] == "PRV1003":
                    return []
                return [
                    {
                        "availabilityId": "AVL1013-OPEN",
                        "providerId": "PRV1013",
                        "status": "available",
                    }
                ]
            raise AssertionError(tool_name)

    specialist = LangGraphSpecialist(
        "healthcare",
        Gateway(),
        object(),  # type: ignore[arg-type]
    )
    action = ProposedAction(
        action_id="ACT1001",
        action_type="book_dummy_appointment",
        description="Proposed appointment",
        parameters={
            "provider_id": "PRV1003",
            "availability_id": "AVL-WRONG",
            "reason": "Knee pain",
        },
    )

    await specialist._complete_appointment_parameters(  # noqa: SLF001
        action, "Book an appointment for knee pain", {"county": "Henrico County"}
    )

    assert action.parameters["provider_id"] == "PRV1013"
    assert action.parameters["availability_id"] == "AVL1013-OPEN"


def test_orchestrator_guardrail_rejects_invalid_stages_and_invented_synthesis_ids():
    guardrail = OrchestratorGuardrail()
    available = {"healthcare": object(), "transportation": object()}
    invalid_plan = OrchestratorPlan(
        intents=["appointments", "transportation"],
        selected_agents=["healthcare", "transportation"],
        execution_stages=[ExecutionStage(stage=1, agents=["healthcare"])],
    )
    with pytest.raises(AgentGuardrailError, match="exactly once"):
        guardrail.validate_plan(invalid_plan, available)

    results = {"healthcare": safe_result("HealthcareAccessAgent")}
    invented = OrchestratorResponse(
        answer="An unsupported response",
        completed_agents=["healthcare"],
        citation_ids=["INVENTED-CITATION"],
        action_ids=["INVENTED-ACTION"],
    )
    with pytest.raises(AgentGuardrailError, match="action ID"):
        guardrail.validate_synthesis(invented, results)


def test_orchestrator_guardrail_rejects_internal_id_request_and_lost_provider_fact():
    guardrail = OrchestratorGuardrail()
    results = {
        "healthcare": AgentResult(
            agent_name="HealthcareAccessAgent",
            status="success",
            summary="Dr. Carter has an available orthopedic appointment.",
        )
    }
    asks_for_id = OrchestratorResponse(
        answer="Please provide the provider ID to continue.",
        completed_agents=["healthcare"],
    )
    with pytest.raises(AgentGuardrailError, match="internal provider"):
        guardrail.validate_synthesis(asks_for_id, results)

    loses_provider = OrchestratorResponse(
        answer="I found an available appointment.",
        completed_agents=["healthcare"],
    )
    with pytest.raises(AgentGuardrailError, match="provider name"):
        guardrail.validate_synthesis(loses_provider, results)


@pytest.mark.asyncio
async def test_orchestrator_llm_plans_agents_and_synthesizes_validated_results():
    class StructuredInvoker:
        def __init__(self, value):
            self.value = value

        async def ainvoke(self, _messages):
            return self.value

    class Model:
        def with_structured_output(self, schema, **kwargs):
            assert kwargs == {"method": "function_calling"}
            if schema is OrchestratorPlan:
                return StructuredInvoker(
                    OrchestratorPlan(
                        intents=["appointments", "transportation"],
                        selected_agents=["healthcare", "transportation"],
                        execution_stages=[
                            ExecutionStage(stage=1, agents=["healthcare", "transportation"])
                        ],
                        routing_summary={
                            "healthcare": "Appointment request",
                            "transportation": "Ride request",
                        },
                    )
                )
            return StructuredInvoker(
                OrchestratorResponse(
                    answer="I found appointment and transportation options.",
                    completed_agents=["healthcare", "transportation"],
                )
            )

    orchestrator = SeniorCareOrchestratorAgent(
        {"healthcare": object(), "transportation": object()},
        Model(),  # type: ignore[arg-type]
        configured=True,
    )
    plan = await orchestrator.plan(
        "Find a doctor and transportation",
        {"cases": []},
        {"recipientId": "SEN1001"},
    )
    response = await orchestrator.synthesize(
        "Find a doctor and transportation",
        {"recipientId": "SEN1001"},
        plan,
        {
            "healthcare": safe_result("HealthcareAccessAgent"),
            "transportation": safe_result("TransportationAgent"),
        },
        "fallback",
    )
    assert plan.selected_agents == ["healthcare", "transportation"]
    assert response.answer.startswith("I found")


@pytest.mark.asyncio
async def test_orchestrator_uses_safe_local_message_when_all_specialists_fail():
    orchestrator = SeniorCareOrchestratorAgent({"healthcare": object()})
    response = await orchestrator.synthesize(
        "Book a doctor appointment for my father",
        {"recipientId": "SEN1001"},
        OrchestratorPlan(
            intents=["appointments"],
            selected_agents=["healthcare"],
            execution_stages=[ExecutionStage(stage=1, agents=["healthcare"])],
        ),
        {
            "healthcare": AgentResult(
                agent_name="HealthcareAccessAgent",
                status="failed",
                summary="The local specialist failed.",
                confidence=0.0,
            )
        },
        "unused fallback",
    )

    assert "No local action was created" in response.answer
    assert "no external organization was contacted" in response.answer
    assert "contact a healthcare provider" not in response.answer.casefold()


def test_self_and_parent_appointment_queries_route_to_required_agents():
    parent_intents, parent_agents = route_intents(
        "Book a doctor appointment for my father for his knee pain"
    )
    self_intents, self_agents = route_intents(
        "Book a doctor appointment for my knee pain and arrange transport"
    )
    assert {"appointments", "provider_search"}.issubset(parent_intents)
    assert "healthcare" in parent_agents
    assert {"appointments", "provider_search", "transportation"}.issubset(self_intents)
    assert {"healthcare", "transportation"}.issubset(self_agents)


def test_existing_appointment_transportation_excludes_healthcare_creation_agent():
    plan = OrchestratorPlan(
        intents=["appointments", "transportation"],
        selected_agents=["healthcare", "transportation"],
        execution_stages=[ExecutionStage(stage=1, agents=["healthcare", "transportation"])],
    )

    enforced = SeniorCareOrchestratorAgent._enforce_cross_domain_routing(
        "Book wheelchair transportation for appointment APT1022", plan
    )

    assert enforced.selected_agents == ["transportation"]
    assert enforced.intents == ["transportation"]


def test_ambiguous_appointment_requires_domain_clarification():
    plan = OrchestratorPlan(
        intents=["appointments"],
        selected_agents=["healthcare"],
        execution_stages=[ExecutionStage(stage=1, agents=["healthcare"])],
    )

    ambiguous = SeniorCareOrchestratorAgent._enforce_appointment_clarification(
        "Can you help me with an appointment?", plan
    )
    explicit = SeniorCareOrchestratorAgent._enforce_appointment_clarification(
        "Can you help me with a doctor appointment for knee pain?",
        OrchestratorPlan(
            intents=["appointments"],
            selected_agents=["healthcare"],
            execution_stages=[ExecutionStage(stage=1, agents=["healthcare"])],
        ),
    )

    assert ambiguous.missing_information == [
        "Which type of appointment do you mean: a doctor/provider appointment, "
        "transportation for an existing appointment, or another service appointment?"
    ]
    assert explicit.missing_information == []


def test_actionable_doctor_request_does_not_require_type_of_care():
    plan = OrchestratorPlan(
        intents=["appointments", "provider_search"],
        selected_agents=["healthcare"],
        execution_stages=[ExecutionStage(stage=1, agents=["healthcare"])],
        missing_information=["What type of orthopedic care does your father need?"],
        routing_summary={"healthcare": "Find and book an appropriate provider."},
    )

    corrected = SeniorCareOrchestratorAgent._enforce_provider_search_clarification(
        "Please book a doctor appointment for my father's leg pain", plan
    )

    assert corrected.missing_information == []

    incorrect_llm_clarification = OrchestratorPlan(
        intents=["appointments", "transportation"],
        selected_agents=["healthcare", "transportation"],
        execution_stages=[ExecutionStage(stage=1, agents=["healthcare", "transportation"])],
        missing_information=[
            "Which type of appointment do you mean: a doctor/provider appointment, "
            "transportation for an existing appointment, or another service appointment?"
        ],
    )
    corrected = SeniorCareOrchestratorAgent._enforce_appointment_clarification(
        "Book transportation for APT1022", incorrect_llm_clarification
    )
    assert corrected.missing_information == []


def test_appointment_transport_follow_up_routes_only_to_transportation():
    intents, agents = route_intents(
        "Book transportation for pickup to the appointment and drop back home "
        "for the above booked appointment"
    )

    assert intents == ["transportation"]
    assert agents == ["transportation"]

    id_intents, id_agents = route_intents("APT1021")
    assert id_intents == ["transportation"]
    assert id_agents == ["transportation"]


def test_recipient_guardrail_distinguishes_self_parent_and_mismatch():
    guardrail = SpecialistGuardrail()
    self_member = {
        "accountRole": "self_care",
        "careRecipient": {"relationshipToAccountHolder": "self"},
    }
    representative = {
        "accountRole": "family_representative",
        "careRecipient": {"relationshipToAccountHolder": "father"},
    }
    assert guardrail.recipient_issue("Book for my knee", self_member) is None
    assert guardrail.recipient_issue("Book for my father", representative) is None
    assert guardrail.recipient_issue("Book for my father", self_member)
    assert guardrail.recipient_issue("Book for my mother", representative)
    assert guardrail.recipient_issue("Book for my knee", representative)


def test_recipient_guardrail_requires_selection_and_enforces_selected_person():
    guardrail = SpecialistGuardrail()
    member = {
        "careRecipients": [
            {
                "recipientId": "REC-FATHER",
                "relationshipToAccountHolder": "father",
                "isAccountHolder": False,
            },
            {
                "recipientId": "REC-MOTHER",
                "relationshipToAccountHolder": "mother",
                "isAccountHolder": False,
            },
        ]
    }
    assert guardrail.recipient_issue("Book an appointment", member)
    assert guardrail.recipient_issue("Book for my mother", member, "REC-FATHER")
    assert guardrail.recipient_issue("Book for my mother", member, "REC-MOTHER") is None
    assert guardrail.recipient_issue("Book an appointment", member, "REC-UNKNOWN")


def test_specialist_output_guardrail_rejects_llm_writes_and_clinical_claims():
    guardrail = SpecialistGuardrail()
    write_result = safe_result()
    write_result.tool_calls.append(
        ToolCallRecord(tool="book_dummy_appointment", operation="write", status="success")
    )
    with pytest.raises(AgentGuardrailError):
        guardrail.validate_output(
            "healthcare", "HealthcareAccessAgent", write_result, "SEN1001", set()
        )

    clinical = safe_result()
    clinical.summary = "You have arthritis and should arrange care."
    with pytest.raises(AgentGuardrailError):
        guardrail.validate_output("healthcare", "HealthcareAccessAgent", clinical, "SEN1001", set())


def test_specialist_output_guardrail_rejects_cross_member_action():
    result = safe_result()
    result.proposed_actions.append(
        ProposedAction(
            action_id="ACT-1",
            action_type="book_dummy_appointment",
            description="Simulation",
            parameters={},
            agent_name="HealthcareAccessAgent",
            user_id="SEN1002",
            senior_id="SEN1002",
        )
    )
    with pytest.raises(AgentGuardrailError):
        SpecialistGuardrail().validate_output(
            "healthcare",
            "HealthcareAccessAgent",
            result,
            "SEN1001",
            {"book_dummy_appointment"},
        )


@pytest.mark.asyncio
async def test_orchestrator_validates_selection_and_result_set():
    class FakeAgent:
        async def run(self, query: str, user_id: str, case_id: str | None):
            return safe_result()

    orchestrator = SeniorCareOrchestratorAgent({"healthcare": FakeAgent()})
    result = await orchestrator.run(["healthcare"], "Find a doctor", "SEN1001", None)
    assert set(result) == {"healthcare"}
    with pytest.raises(AgentGuardrailError):
        await orchestrator.run(["unknown"], "Find a doctor", "SEN1001", None)


def test_code_evaluator_scores_safe_contract_and_rejects_write():
    evaluator = CodeBasedAgentEvaluator()
    passed = evaluator.evaluate(safe_result(), expected_agent="HealthcareAccessAgent")
    assert passed.passed and passed.score == 1
    unsafe = safe_result()
    unsafe.tool_calls.append(ToolCallRecord(tool="write", operation="write", status="success"))
    failed = evaluator.evaluate(unsafe, expected_agent="HealthcareAccessAgent")
    assert not failed.passed and "no_llm_writes" in failed.failures


def test_code_evaluator_rejects_actions_outside_domain_policy():
    evaluator = CodeBasedAgentEvaluator()
    result = safe_result("MealsFoodAgent")
    result.proposed_actions.append(
        ProposedAction(
            action_id="ACT-FORBIDDEN-MEAL",
            action_type="enroll_dummy_meal_service",
            description="Forbidden Phase 1 meal enrollment",
            parameters={"senior_id": "SEN1001", "recipient_id": "SEN1001"},
            agent_name="MealsFoodAgent",
            user_id="SEN1001",
            senior_id="SEN1001",
            recipient_id="SEN1001",
        )
    )

    evaluation = evaluator.evaluate(
        result,
        expected_agent="MealsFoodAgent",
        allowed_action_types=set(),
    )

    assert not evaluation.passed
    assert "actions_allowlisted" in evaluation.failures


@pytest.mark.asyncio
async def test_tool_guardrail_rejects_removed_phase_one_write_before_mcp_call(tmp_path: Path):
    class GatewayThatMustNotRun:
        async def call(self, *_args, **_kwargs):
            raise AssertionError("MCP must not be called for a removed action")

    guardrail = ToolGuardrail(
        RuntimeSettings(project_root=tmp_path),
        GatewayThatMustNotRun(),  # type: ignore[arg-type]
    )
    action = ProposedAction(
        action_id="ACT-REMOVED-WRITE",
        action_type="request_dummy_refill",
        description="Removed Phase 1 write",
        parameters={"senior_id": "SEN1001", "recipient_id": "SEN1001"},
        agent_name="MedicationPharmacyAgent",
        user_id="SEN1001",
        senior_id="SEN1001",
        recipient_id="SEN1001",
    )

    with pytest.raises(PermissionError, match="not enabled for Phase 1"):
        await guardrail.validate(action, approved=True)


def test_human_eval_packet_and_summary(tmp_path: Path):
    path = tmp_path / "human.jsonl"
    store = HumanEvaluationStore()
    assert store.prepare(path, [{"evaluationId": "one", "response": "Safe response"}]) == 1
    row = json.loads(path.read_text())
    row.update(
        approved=True,
        reviewer="reviewer-1",
        ratings={name: 5 for name in store.DIMENSIONS},
    )
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary = store.summarize(path)
    assert summary["completed"] == 1
    assert summary["approvalRate"] == 1
    assert summary["averageRating"] == 5


def test_every_specialist_has_benchmark_coverage_and_read_only_tools():
    root = Path(__file__).parents[2]
    cases = json.loads((root / "evals/agent_benchmarks.json").read_text())
    covered = {case["agent"] for case in cases}
    assert covered == set(AGENT_NAMES)
    assert all(len([case for case in cases if case["agent"] == key]) >= 2 for key in AGENT_NAMES)
    assert all("book_dummy" not in tool for tools in READ_TOOL_POLICIES.values() for tool in tools)
    assert "search_providers" in READ_TOOL_POLICIES["healthcare"]
    assert "search_healthcare_knowledge" in READ_TOOL_POLICIES["healthcare"]
    assert "search_medication_references" in READ_TOOL_POLICIES["medication"]
    assert "search_medication_knowledge" in READ_TOOL_POLICIES["medication"]
    assert RAG_CATEGORIES["healthcare"] == {"healthcare_access"}
    assert RAG_CATEGORIES["medication"] == {"medication_reference"}
    recipient_cases = [case for case in cases if case.get("expectedRecipientMode")]
    assert {case["expectedRecipientMode"] for case in recipient_cases} == {
        "self",
        "family_representative",
    }
    assert any(case.get("expectedRecipientGuardrail") for case in cases)
    assert all(
        case.get("expectedTools")
        for case in cases
        if case["id"].endswith("_read") or case["id"] == "case_status"
    )


def test_specialist_discards_only_explicit_no_action_placeholders():
    result = AgentResult(
        agent_name="CaseStatusRiskAgent",
        status="success",
        summary="No case update is required.",
        proposed_actions=[
            ProposedAction(
                action_id="ACT-NONE",
                action_type="none",
                description="No action",
                parameters={},
            )
        ],
    )

    LangGraphSpecialist._discard_noop_actions(result)

    assert result.proposed_actions == []
    assert "no-action placeholder" in result.warnings[0]


@pytest.mark.asyncio
async def test_transportation_binds_selected_recipient_active_case_appointment():
    class Gateway:
        async def call(self, tool_name, **arguments):
            if tool_name == "list_appointments":
                return [
                    {
                        "appointmentId": "APT-OTHER",
                        "recipientId": "REC-OTHER",
                        "appointmentDate": "2026-09-10",
                        "appointmentTime": "09:00",
                        "status": "scheduled",
                    },
                    {
                        "appointmentId": "APT-FATHER",
                        "recipientId": "REC-FATHER",
                        "appointmentDate": "2026-09-12",
                        "appointmentTime": "11:30",
                        "providerId": "PRV-FATHER",
                        "status": "scheduled",
                    },
                ]
            if tool_name == "get_case":
                return {"data": {"relatedEntityIds": ["APT-FATHER"]}}
            if tool_name == "search_transportation_services":
                assert arguments["wheelchair_accessible"] is True
                return [{"transportationServiceId": "TRN1003"}]
            if tool_name == "get_provider":
                return {
                    "facilityName": "Orthopedics Center",
                    "addressLine1": "100 Medical Drive",
                    "city": "Midlothian",
                    "state": "VA",
                    "zipCode": "23113",
                }
            if tool_name == "find_available_transportation":
                assert arguments["appointment_time"] == "11:30"
                assert arguments["appointment_date"] == "2026-09-12"
                assert arguments["wheelchair_required"] is True
                return {
                    "data": {
                        "available": True,
                        "transportationServiceId": "TRN1003",
                        "vehicleId": "VEH-TRN1003-01",
                        "pickupDate": "2026-09-12",
                        "pickupTime": "10:51",
                        "estimatedTravelMinutes": 24,
                    }
                }
            raise AssertionError(tool_name)

    specialist = LangGraphSpecialist(
        "transportation",
        Gateway(),
        object(),
        configured=False,  # type: ignore[arg-type]
    )
    proposed = ProposedAction(
        action_id="ACT-RIDE",
        action_type="book_dummy_ride",
        description="Ride",
        parameters={},
    )

    await specialist._complete_transportation_parameters(
        proposed,
        "Pick up my dad from 123 Main Street, Richmond, VA 23220 in a wheelchair "
        "and drop back after the appointment",
        {"seniorId": "SEN1022", "county": "Henrico County"},
        {"recipientId": "REC-FATHER", "firstName": "Henry", "isAccountHolder": False},
        "CASE1003",
    )

    assert proposed.parameters == {
        "appointment_id": "APT-FATHER",
        "service_id": "TRN1003",
        "pickup_date": "2026-09-12",
        "pickup_time": "10:51",
        "pickup_address": "123 Main Street, Richmond, VA 23220",
        "destination_address": "Orthopedics Center, 100 Medical Drive, Midlothian, VA, 23113",
        "appointment_date": "2026-09-12",
        "appointment_time": "11:30",
        "wheelchair_required": True,
        "vehicle_id": "VEH-TRN1003-01",
        "estimated_travel_minutes": 24,
        "accommodation": "wheelchair",
        "return_ride_required": True,
    }


@pytest.mark.asyncio
async def test_transportation_resolves_single_appointment_and_requires_pickup_address():
    class Gateway:
        async def call(self, tool_name, **arguments):
            if tool_name == "get_provider":
                return {
                    "facilityName": "Orthopedics Center",
                    "addressLine1": "100 Medical Drive",
                    "city": "Midlothian",
                    "state": "VA",
                    "zipCode": "23113",
                }
            if tool_name == "get_case":
                return {"data": {"relatedEntityIds": ["APT-FATHER"]}}
            if tool_name == "find_available_transportation":
                return {
                    "available": True,
                    "transportationServiceId": "TRN1001",
                    "vehicleId": "VEH1001",
                    "pickupDate": "2026-09-12",
                    "pickupTime": "10:55",
                    "estimatedTravelMinutes": 20,
                }
            if tool_name == "search_transportation_services":
                return [
                    {
                        "transportationServiceId": "TRN1001",
                        "serviceName": "Care Ride",
                    }
                ]
            assert tool_name == "list_appointments"
            assert arguments == {"user_id": "SEN1022"}
            return [
                {
                    "appointmentId": "APT-OTHER",
                    "recipientId": "REC-OTHER",
                    "appointmentDate": "2026-09-10",
                    "appointmentTime": "09:00",
                    "reason": "checkup",
                    "status": "scheduled",
                },
                {
                    "appointmentId": "APT-FATHER",
                    "recipientId": "REC-FATHER",
                    "appointmentDate": "2026-09-12",
                    "appointmentTime": "11:30",
                    "providerId": "PRV-FATHER",
                    "reason": "knee pain",
                    "status": "scheduled",
                },
            ]

    specialist = LangGraphSpecialist(
        "transportation",
        Gateway(),
        object(),
        configured=False,  # type: ignore[arg-type]
    )
    recipient = {
        "recipientId": "REC-FATHER",
        "firstName": "Henry",
        "lastName": "Jones",
        "isAccountHolder": False,
    }

    missing = await specialist._transportation_appointment_confirmation(
        "Book transportation for my father",
        "SEN1022",
        {"seniorId": "SEN1022"},
        recipient,
        None,
    )
    assert missing is not None
    assert missing.status == "needs_user_input"
    assert "APT-FATHER" in missing.summary
    assert "APT-OTHER" not in missing.summary
    assert "confirm which appointment" in missing.summary

    missing_address = await specialist._transportation_appointment_confirmation(
        "Book transportation for APT-FATHER",
        "SEN1022",
        {"seniorId": "SEN1022"},
        recipient,
        None,
    )
    assert missing_address is not None
    assert "100 Medical Drive" in missing_address.summary
    assert "pickup/home address" in missing_address.summary
    assert "wheelchair assistance (yes or no)" in missing_address.summary

    missing_wheelchair_choice = await specialist._transportation_appointment_confirmation(
        "Book round-trip transportation for apt-father from 123 Main Street, Richmond, VA 23220",
        "SEN1022",
        {"seniorId": "SEN1022"},
        recipient,
        "CASE1001",
    )
    assert missing_wheelchair_choice is not None
    assert missing_wheelchair_choice.status == "needs_user_input"
    assert "wheelchair assistance (yes or no)" in missing_wheelchair_choice.summary
    assert not missing_wheelchair_choice.proposed_actions

    valid = await specialist._transportation_appointment_confirmation(
        "Book round-trip transportation for apt-father from 123 Main Street, Richmond, VA 23220; "
        "no wheelchair assistance",
        "SEN1022",
        {"seniorId": "SEN1022"},
        recipient,
        "CASE1001",
    )
    assert valid is not None
    assert valid.status == "success"
    assert valid.proposed_actions[0].parameters["appointment_id"] == "APT-FATHER"
    assert valid.proposed_actions[0].parameters["return_ride_required"] is True
    assert valid.proposed_actions[0].parameters["wheelchair_required"] is False

    mismatched = await specialist._transportation_appointment_confirmation(
        "Book transportation for APT-OTHER",
        "SEN1022",
        {"seniorId": "SEN1022"},
        recipient,
        None,
    )
    assert mismatched is not None
    assert mismatched.status == "needs_user_input"
    assert "not an eligible scheduled appointment" in mismatched.summary
    assert "APT-FATHER" in mismatched.summary


def test_address_followup_reuses_pending_transportation_appointment_context():
    session = AgentSession(
        session_id="SESSION-1",
        user_id="SEN1022",
        messages=[
            ChatMessage(role="user", content="Book transportation for APT1021"),
            ChatMessage(
                role="assistant",
                content="What is the recipient's full pickup/home address?",
            ),
        ],
    )

    contextual = contextualize_followup(
        "Recipient home address: 7743 Halifax Drive, Mechanicsville, VA 23116",
        session,
    )

    assert contextual == (
        "Book round-trip transportation for APT1021 from "
        "7743 Halifax Drive, Mechanicsville, VA 23116."
    )

    hyphenated = contextualize_followup(
        "I can provide the recipient address - 7743 Halifax Drive, Mechanicsville, "
        "VA - 23116. Doesn't the appointment already have provider details?",
        session,
    )
    assert hyphenated == (
        "Book round-trip transportation for APT1021 from "
        "7743 Halifax Drive, Mechanicsville, VA - 23116."
    )

    session.messages.append(ChatMessage(role="user", content=hyphenated))
    session.messages.append(
        ChatMessage(
            role="assistant",
            content="Does the recipient require wheelchair assistance? Please answer yes or no.",
        )
    )
    wheelchair_followup = contextualize_followup("Yes, wheelchair assistance is required.", session)
    assert wheelchair_followup == (
        "Book round-trip transportation for APT1021 from "
        "7743 Halifax Drive, Mechanicsville, VA - 23116; wheelchair assistance: yes."
    )

    new_domain_request = "I am looking for some meal assistance program for my dad"
    assert contextualize_followup(new_domain_request, session) == new_domain_request


def test_provider_followup_resolves_single_doctor_from_session_context():
    session = AgentSession(
        session_id="SESSION-DOCTOR",
        user_id="SEN1001",
        messages=[
            ChatMessage(
                role="user",
                content="Find an orthopedic doctor for my father's knee pain",
            ),
            ChatMessage(
                role="assistant",
                content="I found Dr. Carter at Central Virginia Orthopedics Center.",
            ),
        ],
    )

    contextual = contextualize_followup("Please book the appointment with this doctor", session)

    assert "doctor/provider appointment with Dr. Carter" in contextual
    assert "knee pain" in contextual
    assert "do not ask for the appointment type" in contextual


def test_named_provider_booking_followup_uses_preceding_provider_result():
    session = AgentSession(session_id="SESSION-2", user_id="SEN1001")
    session.messages.extend(
        [
            ChatMessage(
                role="user",
                content="Find an orthopedic doctor for my father's knee pain",
            ),
            ChatMessage(
                role="assistant",
                content=(
                    "I found Dr. Carter at Central Virginia Orthopedics Center. "
                    "An available slot is August 31, 2026 at 09:00."
                ),
            ),
        ]
    )

    contextual = contextualize_followup(
        "Great, please book the appointment with Dr. Carter", session
    )

    assert "doctor/provider appointment with Dr. Carter" in contextual
    assert "do not ask for the appointment type" in contextual


def test_implicit_booking_followup_uses_only_provider_in_preceding_result():
    session = AgentSession(session_id="SESSION-3", user_id="SEN1001")
    session.messages.extend(
        [
            ChatMessage(role="user", content="Find an orthopedic doctor nearby"),
            ChatMessage(
                role="assistant",
                content="I found Dr. Carter and an available orthopedic appointment.",
            ),
        ]
    )

    contextual = contextualize_followup("Please book the appointment", session)

    assert "doctor/provider appointment with Dr. Carter" in contextual


def test_booking_followup_reuses_prior_request_without_asking_for_internal_ids():
    session = AgentSession(session_id="SESSION-4", user_id="SEN1001")
    session.messages.extend(
        [
            ChatMessage(
                role="user",
                content="Find an orthopedic doctor for my father's knee pain.",
            ),
            ChatMessage(
                role="assistant",
                content="I found Dr. Carter and Dr. Morris with available appointments.",
            ),
        ]
    )
    query = "Please book the appointment"

    contextual = contextualize_followup(query, session)

    assert "first verified available provider-slot pair" in contextual
    assert "knee pain" in contextual
    assert "must never be requested from the user" in contextual


def test_specialist_recognizes_internal_id_requests_inside_questions():
    assert LangGraphSpecialist._mentions_system_field(  # noqa: SLF001
        "Please provide the provider ID of the doctor you want."
    )
    assert LangGraphSpecialist._mentions_system_field(  # noqa: SLF001
        "availability_id is required"
    )


def test_doctor_appointment_clarification_answer_recovers_prior_provider():
    session = AgentSession(session_id="SESSION-5", user_id="SEN1001")
    session.messages.extend(
        [
            ChatMessage(
                role="assistant",
                content="I found Dr. Carter and an available orthopedic appointment.",
            ),
            ChatMessage(
                role="user",
                content="Please book the appointment with Dr. Carter",
            ),
            ChatMessage(
                role="assistant",
                content=(
                    "Which type of appointment do you mean: a doctor/provider appointment, "
                    "transportation for an existing appointment, or another service appointment?"
                ),
            ),
        ]
    )

    contextual = contextualize_followup("doctor appointment", session)

    assert "doctor/provider appointment with Dr. Carter" in contextual


def test_named_doctor_is_not_forced_through_appointment_type_clarification():
    plan = OrchestratorPlan(
        intents=["healthcare"],
        selected_agents=["healthcare"],
        execution_stages=[ExecutionStage(stage=1, agents=["healthcare"])],
        missing_information=[
            "Which type of appointment do you mean: a doctor/provider appointment, "
            "transportation for an existing appointment, or another service appointment?"
        ],
        routing_summary={"healthcare": "Booking with the named provider."},
        confidence=0.9,
    )

    result = SeniorCareOrchestratorAgent._enforce_appointment_clarification(
        "Please book the appointment with Dr. Carter", plan
    )

    assert result.missing_information == []


def test_county_clarification_answer_continues_healthcare_request():
    session = AgentSession(session_id="SESSION-COUNTY", user_id="SEN1001")
    session.messages.extend(
        [
            ChatMessage(
                role="user",
                content="Find an orthopedic doctor for my father's knee pain.",
            ),
            ChatMessage(
                role="assistant",
                content="Please provide the county you would like me to search in.",
            ),
        ]
    )

    contextual = contextualize_followup("Hanover", session)

    assert "Find an orthopedic doctor" in contextual
    assert "Hanover County" in contextual
    assert "meal" not in contextual.casefold()


def test_orchestrator_rejects_meal_route_for_completed_orthopedic_request():
    plan = OrchestratorPlan(
        intents=["meal_assistance"],
        selected_agents=["meals"],
        execution_stages=[ExecutionStage(stage=1, agents=["meals"])],
        routing_summary={"meals": "Incorrect meal route."},
        confidence=0.9,
    )

    with pytest.raises(ValueError, match="Semantic routing conflict"):
        SeniorCareOrchestratorAgent._validate_semantic_consistency(
            "Continue: find an orthopedic doctor in Hanover County", plan
        )


def test_wheelchair_requirement_must_be_explicit():
    assert LangGraphSpecialist._wheelchair_requirement("Book a ride") is None
    assert LangGraphSpecialist._wheelchair_requirement("Wheelchair assistance is required") is True
    assert LangGraphSpecialist._wheelchair_requirement("No wheelchair assistance") is False


def test_pickup_address_parser_accepts_bracketed_transportation_reply():
    query = (
        "pickup/home address: 7743 Halifax Drive, Mechanicsville, VA 23116. "
        "Book round-trip transportation for APT1022 from "
        "[7743 Halifax Drive, Mechanicsville, VA 23116]."
    )
    assert LangGraphSpecialist._pickup_address(query) == (
        "7743 Halifax Drive, Mechanicsville, VA 23116"
    )


def test_golden_dataset_and_chunk_corpus_cover_recipient_and_rag_contracts():
    root = Path(__file__).parents[2]
    golden = json.loads((root / "evals/golden_questions.json").read_text())
    chunks = [
        json.loads(line)
        for line in (root / "data/processed/chunks.jsonl").read_text().splitlines()
        if line
    ]
    assert {row.get("recipientMode") for row in golden if row.get("recipientMode")} == {
        "self",
        "family_representative",
    }
    assert any(row.get("expectedGuardrail") == "recipient_mismatch" for row in golden)
    assert {row.get("expectedExecution") for row in golden if row.get("expectedExecution")} == {
        "parallel",
        "sequential",
    }
    assert {
        "discharge_coordination",
        "benefits_coordination",
        "caregiver_coordination",
        "tool_injection",
    } <= {row["id"] for row in golden}
    required_categories = set().union(*RAG_CATEGORIES.values())
    assert required_categories <= {row["category"] for row in chunks}


@pytest.mark.asyncio
async def test_specialist_compiles_once_and_reuses_stateless_graph():
    class Tool:
        def __init__(self, name, properties):
            self.name = name
            self.args_schema = {"type": "object", "properties": properties}

    class Gateway:
        def __init__(self):
            self.discovery_calls = 0

        async def get_tools(self, allowed):
            self.discovery_calls += 1
            schemas = {
                "search_providers": {
                    "specialty": {"type": ["string", "null"]},
                    "county": {"type": ["string", "null"]},
                    "limit": {"type": "integer"},
                    "include_public": {"type": "boolean"},
                }
            }
            return [Tool(name, schemas.get(name, {})) for name in allowed]

        async def call(self, tool_name, **arguments):
            if tool_name == "search_providers":
                assert arguments["specialty"] == "orthopedics"
                return [{"providerId": "PRV1001"}]
            if tool_name == "list_available_slots":
                return [
                    {
                        "availabilityId": "AVL1001",
                        "providerId": arguments["provider_id"],
                        "status": "available",
                    }
                ]
            if tool_name == "get_provider":
                return {
                    "providerId": arguments["provider_id"],
                    "providerName": "Dr. Morris",
                    "specialty": "Orthopedics",
                    "facilityName": "Richmond Clinic",
                    "city": "Richmond",
                    "state": "VA",
                    "zipCode": "23220",
                }
            if tool_name == "get_available_slot":
                return {
                    "availabilityId": arguments["availability_id"],
                    "providerId": "PRV1001",
                    "availableDate": "2026-09-12",
                    "availableTime": "11:30",
                    "status": "available",
                }
            assert tool_name == "get_member"
            return {
                "seniorId": arguments["user_id"],
                "accountRole": "self_care",
                "careRecipient": {
                    "firstName": "Test",
                    "relationshipToAccountHolder": "self",
                },
            }

    class StructuredInvoker:
        def __init__(self, model, schema):
            self.model = model
            self.schema = schema

        async def ainvoke(self, payload):
            self.model.invocations.append((self.schema, payload))
            if self.schema is SpecialistPlan:
                return SpecialistPlan(
                    task_summary="Find an orthopedic provider",
                    selected_tools=["search_providers"],
                    tool_arguments={"search_providers": {"specialty": "orthopedics"}},
                    # Internal MCP keys are not valid user-information requests.
                    missing_information=["provider_id", "availability_id"],
                )
            return AgentResult(
                agent_name="HealthcareAccessAgent",
                status="success",
                summary="Found local records.",
                proposed_actions=[
                    ProposedAction(
                        action_id="ACT-MODEL",
                        action_type="book_dummy_appointment",
                        description="Propose a local simulated appointment",
                        parameters={},
                    )
                ],
            )

    class Model:
        def __init__(self):
            self.invocations = []

        def with_structured_output(self, schema, **kwargs):
            assert kwargs == {"method": "function_calling"}
            return StructuredInvoker(self, schema)

    gateway = Gateway()
    model = Model()
    specialist = LangGraphSpecialist(
        "healthcare",
        gateway,
        model,  # type: ignore[arg-type]
        configured=True,  # type: ignore[arg-type]
    )

    initialized = await asyncio.gather(*(specialist.initialize() for _ in range(5)))
    subgraph_nodes = set(specialist._agent.get_graph().nodes)  # type: ignore[union-attr]
    first = await specialist.run("Book a doctor appointment for knee pain", "SEN1001", "CASE1001")
    second = await specialist.run("Book a doctor appointment for knee pain", "SEN1002", "CASE1002")

    assert all(initialized)
    assert gateway.discovery_calls == 1
    assert specialist.initialized is True
    assert {
        "specialist_plan_llm",
        "validate_tool_plan",
        "execute_mcp_reads",
        "validate_retrieval",
        "specialist_synthesis_llm",
        "validate_agent_result",
    } <= subgraph_nodes
    assert first.agent_name == second.agent_name == "HealthcareAccessAgent"
    assert first.proposed_actions[0].agent_name == "HealthcareAccessAgent"
    assert first.proposed_actions[0].user_id == "SEN1001"
    assert first.proposed_actions[0].senior_id == "SEN1001"
    assert first.proposed_actions[0].parameters["senior_id"] == "SEN1001"
    assert first.proposed_actions[0].parameters["provider_id"] == "PRV1001"
    assert first.proposed_actions[0].parameters["availability_id"] == "AVL1001"
    assert first.proposed_actions[0].parameters["reason"] == (
        "Book a doctor appointment for knee pain"
    )
    assert "Dr. Morris" in first.proposed_actions[0].description
    assert "Richmond Clinic" in first.proposed_actions[0].description
    assert "2026-09-12" in first.proposed_actions[0].description
    assert len(model.invocations) == 4
    assert [schema for schema, _ in model.invocations].count(SpecialistPlan) == 2
    synthesis_schemas = [schema for schema, _ in model.invocations if schema is not SpecialistPlan]
    assert len(synthesis_schemas) == 2
    assert all(issubclass(schema, AgentResult) for schema in synthesis_schemas)
    serialized = json.dumps(model.invocations, default=str)
    assert "SEN1001" in serialized and "SEN1002" in serialized
    assert "assignedObjective" in serialized
    assert "relevantConversationTurns" in serialized
    assert "taskSpecificSummary" in serialized
    assert "verifiedFacts" in serialized
    assert "dependencyResults" in serialized
    assert "allowedTools" in serialized
    assert "constraints" in serialized
    assert "responseSchema" in serialized
    assert "recipientResolutionStatus" in serialized
    assert "relationshipToAccountHolder" in serialized


@pytest.mark.asyncio
async def test_unconfigured_specialist_returns_structured_blocked_result():
    class Gateway:
        async def call(self, tool_name, **arguments):
            assert tool_name == "get_member"
            return {
                "seniorId": arguments["user_id"],
                "careRecipients": [
                    {
                        "recipientId": arguments["user_id"],
                        "relationshipToAccountHolder": "self",
                        "isAccountHolder": True,
                    }
                ],
            }

    specialist = LangGraphSpecialist(
        "healthcare",
        Gateway(),
        object(),
        configured=False,  # type: ignore[arg-type]
    )
    result = await specialist.run("Find a doctor", "SEN1001", recipient_id="SEN1001")
    assert result.status == "blocked"
    assert "LLM_MODEL" in result.summary
    assert not result.proposed_actions
