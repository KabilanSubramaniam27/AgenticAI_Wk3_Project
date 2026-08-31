import re
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from seniorcare_agents.agents.orchestrator import SeniorCareOrchestratorAgent
from seniorcare_agents.graph.approvals import ApprovalManager
from seniorcare_agents.graph.state import SeniorCareState
from seniorcare_agents.guardrails import InputGuardrail, OutputGuardrail
from seniorcare_agents.mcp import MCPToolGateway
from seniorcare_agents.models import AgentResult, ExecutionStage, OrchestratorPlan, ProposedAction
from seniorcare_agents.observability import ObservabilityService, flow_event
from seniorcare_agents.services import CitationService
from seniorcare_runtime.config import RuntimeSettings


class SeniorCareGraphBuilder:
    def __init__(
        self,
        settings: RuntimeSettings,
        orchestrator: SeniorCareOrchestratorAgent,
        gateway: MCPToolGateway,
    ):
        self.settings = settings
        self.orchestrator = orchestrator
        self.input = InputGuardrail()
        self.output = OutputGuardrail()
        self.mcp = gateway
        self.approvals = ApprovalManager(settings, self.mcp)
        self.citations = CitationService()
        self.events = ObservabilityService(settings.observability_path)

    def build(self):
        graph = StateGraph(SeniorCareState)
        graph.add_node("input_guardrails", self.input_node)
        graph.add_node("member_resolution", self.member_node)
        graph.add_node("orchestrator_plan_llm", self.router_node)
        graph.add_node("validate_orchestrator_plan", self.validate_orchestrator_plan_node)
        graph.add_node("execute_specialist_subgraphs", self.orchestrator_node)
        graph.add_node("orchestrator_synthesis_llm", self.merge_node)
        graph.add_node("output_guardrails", self.output_guardrails_node)
        graph.add_node("approval", self.approval_node)
        graph.add_edge(START, "input_guardrails")
        graph.add_conditional_edges(
            "input_guardrails", self.after_input, {"stop": END, "continue": "member_resolution"}
        )
        graph.add_conditional_edges(
            "member_resolution",
            self.after_member,
            {"stop": END, "continue": "orchestrator_plan_llm"},
        )
        graph.add_conditional_edges(
            "orchestrator_plan_llm",
            self.after_orchestrator_plan,
            {"clarify": END, "continue": "validate_orchestrator_plan"},
        )
        graph.add_edge("validate_orchestrator_plan", "execute_specialist_subgraphs")
        graph.add_edge("execute_specialist_subgraphs", "orchestrator_synthesis_llm")
        graph.add_edge("orchestrator_synthesis_llm", "output_guardrails")
        graph.add_conditional_edges(
            "output_guardrails", self.after_merge, {"approval": "approval", "done": END}
        )
        graph.add_edge("approval", END)
        return graph.compile(checkpointer=InMemorySaver())

    def input_node(self, state: SeniorCareState) -> dict:
        flow_event(
            "guardrail",
            "input_validation",
            "input",
            {"query": state.get("raw_user_query", ""), "userId": state.get("user_id")},
        )
        result = self.input.evaluate(state.get("raw_user_query", ""))
        response = None
        if result["emergency"]:
            response = "This may be urgent. Call 911 or your local emergency number now. I cannot diagnose or continue routine coordination for an emergency."
        elif not result["allowed"]:
            response = "I can only perform safe, simulation-only senior-care coordination and cannot bypass approval or reveal secrets."
        elif "medical_safety" in result["flags"]:
            response = "I cannot recommend medication or treatment changes. Please contact a clinician or pharmacist. I can help track a refill or appointment."
        request_id = state.get("request_id") or f"REQ-{uuid4().hex[:12]}"
        self.events.emit(
            "input_guardrail",
            requestId=request_id,
            flags=result["flags"],
            allowed=result["allowed"],
        )
        flow_event("guardrail", "input_validation", "output", result, request_id=request_id)
        return {
            "request_id": request_id,
            "normalized_query": " ".join(state.get("raw_user_query", "").split()),
            "safety_flags": result["flags"],
            "user_id": state.get("user_id")
            or (result["detectedUserIds"][0] if result["detectedUserIds"] else None),
            "final_response": response,
        }

    def after_input(self, state: SeniorCareState) -> str:
        return "stop" if state.get("final_response") else "continue"

    async def member_node(self, state: SeniorCareState) -> dict:
        user_id = state.get("user_id")
        if not user_id:
            return {
                "member_resolved": False,
                "final_response": "Please enter your SeniorCare User ID. If you are new, provide first name, last name, and date of birth to register.",
            }
        member = await self.mcp.call("get_member", user_id=user_id)
        if not member:
            return {
                "member_resolved": False,
                "final_response": "User ID was not found. Please retry or register as a new member.",
            }
        context = await self.mcp.call("get_member_context", user_id=user_id)
        member_profile = context["senior"]
        recipients = member_profile.get("careRecipients") or [
            member_profile.get("careRecipient")
            or {
                "recipientId": user_id,
                "firstName": member_profile.get("firstName"),
                "lastName": member_profile.get("lastName"),
                "relationshipToAccountHolder": "self",
                "isAccountHolder": True,
            }
        ]
        recipient_id = state.get("recipient_id")
        recipient = next(
            (value for value in recipients if value and value.get("recipientId") == recipient_id),
            recipients[0] if len(recipients) == 1 else None,
        )
        if not recipient:
            return {
                "member_resolved": False,
                "final_response": "Please select which registered care recipient this request is for.",
            }
        recipient.setdefault("recipientId", user_id)
        active_case_id = state.get("active_case_id")
        active_case = next(
            (value for value in context["cases"] if value.get("caseId") == active_case_id),
            None,
        )
        if active_case and active_case.get("recipientId") not in {
            None,
            recipient["recipientId"],
        }:
            active_case_id = None
        return {
            "senior_id": user_id,
            "recipient_id": recipient["recipientId"],
            "member_resolved": True,
            "member_context": context,
            "care_recipient": recipient,
            "caregiver_context": {"caregivers": context["caregivers"]},
            "existing_cases": context["cases"],
            "active_case_id": active_case_id,
        }

    def after_member(self, state: SeniorCareState) -> str:
        return "continue" if state.get("member_resolved") else "stop"

    async def router_node(self, state: SeniorCareState) -> dict:
        flow_event("router", "intent_routing", "input", state["normalized_query"])
        member_context = dict(state.get("member_context", {}))
        appointment_ids = {
            value.upper()
            for value in re.findall(r"\bAPT[\w-]*\b", state["normalized_query"], re.IGNORECASE)
        }
        referenced_appointments = [
            dict(value)
            for value in member_context.get("appointments", [])
            if str(value.get("appointmentId", "")).upper() in appointment_ids
        ]
        for appointment in referenced_appointments:
            provider_id = appointment.get("providerId")
            if provider_id:
                provider = await self.mcp.call("get_provider", provider_id=provider_id)
                if isinstance(provider, dict):
                    appointment["provider"] = provider
        member_context["referencedAppointments"] = referenced_appointments
        plan = await self.orchestrator.plan(
            state["normalized_query"],
            member_context,
            state.get("care_recipient", {}),
            state.get("conversation_history", []),
        )
        intents, agents = plan.intents, plan.selected_agents
        self.events.emit(
            "intent_routed",
            requestId=state["request_id"],
            userId=state.get("user_id"),
            intents=intents,
            agents=agents,
        )
        flow_event(
            "router",
            "intent_routing",
            "output",
            {"intents": intents, "agents": agents},
            request_id=state["request_id"],
        )
        return {
            "detected_intents": intents,
            "selected_agents": agents,
            "orchestrator_plan": plan.model_dump(mode="json"),
            "final_response": (
                " ".join(plan.missing_information) if plan.missing_information else None
            ),
        }

    def after_orchestrator_plan(self, state: SeniorCareState) -> str:
        plan = OrchestratorPlan.model_validate(state["orchestrator_plan"])
        return "clarify" if plan.missing_information else "continue"

    def validate_orchestrator_plan_node(self, state: SeniorCareState) -> dict:
        plan = OrchestratorPlan.model_validate(state["orchestrator_plan"])
        flow_event("guardrail", "orchestrator_plan_validation", "input", plan.model_dump())
        self.orchestrator.guardrail.validate_plan(plan, self.orchestrator.agents)
        flow_event("guardrail", "orchestrator_plan_validation", "output", {"valid": True})
        return {}

    async def orchestrator_node(self, state: SeniorCareState) -> dict:
        senior_id = state.get("senior_id")
        if not senior_id:
            return {"errors": [{"stage": "orchestrator", "message": "Member was not resolved"}]}
        results = await self.orchestrator.run(
            state["selected_agents"],
            state["normalized_query"],
            senior_id,
            state.get("active_case_id"),
            state.get("recipient_id"),
            [
                ExecutionStage.model_validate(value)
                for value in state.get("orchestrator_plan", {}).get("execution_stages", [])
            ],
            state.get("conversation_history", []),
            state.get("member_context", {}),
            state.get("care_recipient", {}),
            state.get("orchestrator_plan", {}).get("routing_summary", {}),
        )
        return {
            "agent_results": {key: value.model_dump(mode="json") for key, value in results.items()}
        }

    async def merge_node(self, state: SeniorCareState) -> dict:
        flow_event(
            "orchestrator",
            "coordination_merge",
            "input",
            {"agentResults": list(state.get("agent_results", {}))},
            request_id=state.get("request_id"),
        )
        results = [
            AgentResult.model_validate(value) for value in state.get("agent_results", {}).values()
        ]
        actions = [value for result in results for value in result.proposed_actions]
        chunks = [value for result in results for value in result.retrieved_sources]
        risks = [value for result in results for value in result.risks]
        citations = self.citations.from_chunks(chunks)
        lines = [result.summary for result in results]
        if "member_lookup" in state.get("detected_intents", []):
            member = state.get("member_context", {}).get("senior", {})
            recipient = state.get("care_recipient", {})
            lines.append(
                f"Welcome back, {member.get('firstName', 'member')}. Care recipient: "
                f"{recipient.get('firstName', 'member')} {recipient.get('lastName', '')}. "
                f"You have {len(state.get('existing_cases', []))} tracked case(s)."
            )
        needs_case = "case_create" in state.get("detected_intents", []) or (
            actions and not state.get("active_case_id")
        )
        senior_id = state.get("senior_id")
        if needs_case and senior_id and not actions:
            actions = [
                ProposedAction(
                    action_id=f"ACT-{uuid4().hex[:12]}",
                    action_type="create_case",
                    description="Create a local coordination case before executing domain actions",
                    parameters={
                        "user_id": senior_id,
                        "request": {
                            "case_type": "multi_domain_coordination"
                            if "multi_domain" in state.get("detected_intents", [])
                            else "care_coordination",
                            "title": "SeniorCare coordination request",
                            "description": state["normalized_query"],
                            "priority": "medium",
                            "recipient_id": state.get("recipient_id"),
                        },
                    },
                    agent_name="MemberCaseAgent",
                    user_id=senior_id,
                    senior_id=senior_id,
                    recipient_id=state.get("recipient_id"),
                )
            ]
            lines.append(
                "A coordination case must be approved and created before any domain write is performed."
            )
        elif needs_case and actions:
            lines.append(
                "Your single approval will create a local tracking case and save the requested "
                "local action to that case."
            )
        if risks:
            lines.append("Needs attention: " + "; ".join(risk.reason for risk in risks[:5]))
        if actions:
            lines.append(
                f"{len(actions)} local action(s) require your approval. No external organization has been contacted."
            )
        if citations:
            lines.append(
                "Sources: "
                + ", ".join(f"[{item.citation_id}] {item.source_name}" for item in citations)
            )
        deterministic_response = (
            "\n\n".join(lines).replace("dummy", "local").replace("Dummy", "Local")
        )
        synthesis = await self.orchestrator.synthesize(
            state["normalized_query"],
            state.get("care_recipient", {}),
            OrchestratorPlan.model_validate(state["orchestrator_plan"]),
            {
                key: AgentResult.model_validate(value)
                for key, value in state.get("agent_results", {}).items()
            },
            deterministic_response,
            state.get("conversation_history", []),
        )
        response = synthesis.answer
        # Approval and citation notices are authoritative application output. Restore
        # them if an otherwise valid synthesis omitted them.
        if actions and "approval" not in response.casefold():
            response += (
                "\n\nYour approval is required before any local action is saved. "
                "No external organization has been contacted."
            )
        if citations and "Sources:" not in response:
            response += "\n\nSources: " + ", ".join(
                f"[{item.citation_id}] {item.source_name}" for item in citations
            )
        self.events.emit(
            "coordination_merged",
            requestId=state["request_id"],
            userId=state.get("user_id"),
            caseId=state.get("active_case_id"),
            actionCount=len(actions),
            riskCount=len(risks),
            citationCount=len(citations),
            selectedAgents=state.get("selected_agents", []),
        )
        flow_event(
            "orchestrator",
            "coordination_merge",
            "output",
            {
                "response": response,
                "actionCount": len(actions),
                "riskCount": len(risks),
                "citationCount": len(citations),
            },
            request_id=state.get("request_id"),
        )
        return {
            "proposed_actions": [item.model_dump(mode="json") for item in actions],
            "retrieved_chunks": [item.model_dump(mode="json") for item in chunks],
            "risk_flags": [item.model_dump(mode="json") for item in risks],
            "citations": [item.model_dump(mode="json") for item in citations],
            "requires_human_approval": bool(actions),
            "final_response": response,
        }

    def output_guardrails_node(self, state: SeniorCareState) -> dict:
        results = [
            AgentResult.model_validate(value) for value in state.get("agent_results", {}).values()
        ]
        response = state.get("final_response") or ""
        flow_event("guardrail", "output_validation", "input", {"response": response})
        flags = self.output.validate(response, results)
        if flags:
            response = "I could not safely produce a grounded response."
        actions = [action for result in results for action in result.proposed_actions]
        self.approvals.register(actions)
        flow_event(
            "guardrail",
            "output_validation",
            "output",
            {"flags": flags, "actionCount": len(actions)},
        )
        return {
            "final_response": response,
            "safety_flags": [*state.get("safety_flags", []), *flags],
        }

    def after_merge(self, state: SeniorCareState) -> str:
        return "approval" if state.get("requires_human_approval") else "done"

    def approval_node(self, state: SeniorCareState) -> dict:
        decision = interrupt(
            {
                "type": "simulation_approval",
                "warning": "SIMULATION ONLY. This changes local demo data only.",
                "actions": state.get("proposed_actions", []),
            }
        )
        return {
            "approval_status": str(decision.get("status", "pending"))
            if isinstance(decision, dict)
            else str(decision)
        }
