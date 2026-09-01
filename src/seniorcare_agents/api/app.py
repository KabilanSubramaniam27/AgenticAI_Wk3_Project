import re
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from seniorcare_agents.api.normalization import unwrap_record_list
from seniorcare_agents.api.schemas import (
    CareRecipientRegistration,
    MemberRegistration,
    NewCaseRequest,
)
from seniorcare_agents.application import SeniorCareApplication, create_application
from seniorcare_agents.mcp import MCPToolError
from seniorcare_agents.observability import begin_request, end_request, flow_event
from seniorcare_agents.services.session_store import AgentSession


def contextualize_followup(query: str, session: AgentSession) -> str:
    """Resolve narrow follow-up references from the authenticated conversation."""
    if re.search(r"\bAPT[\w-]*\b", query, re.IGNORECASE) or not session.messages:
        return query
    lower_query = query.casefold()
    recent_messages = session.messages[-8:]
    recent = "\n".join(message.content for message in recent_messages)

    # Bind a short county answer to the immediately pending clarification. This is conversation
    # state resolution, not intent routing: the original request remains authoritative and the
    # orchestrator LLM still selects the specialist from the completed request.
    assistant_messages = [
        message.content for message in recent_messages if message.role == "assistant"
    ]
    previous_user_queries = [
        message.content for message in recent_messages if message.role == "user"
    ]
    preceding_assistant = assistant_messages[-1] if assistant_messages else ""
    county_reply = re.fullmatch(
        r"\s*(Richmond(?:\s+City)?|Henrico|Chesterfield|Hanover)(?:\s+County)?\s*[.!]?\s*",
        query,
        re.IGNORECASE,
    )
    if (
        county_reply
        and "county" in preceding_assistant.casefold()
        and any(term in preceding_assistant.casefold() for term in ("provide", "which", "what"))
    ):
        prior_request = previous_user_queries[-1] if previous_user_queries else ""
        county = county_reply.group(1).strip()
        if county.casefold() == "richmond":
            county = "Richmond City"
        elif not county.casefold().endswith(("county", "city")):
            county = f"{county} County"
        return (
            f"Continue the preceding request: {prior_request}. "
            f"The user answered the requested county clarification: {county}."
        )

    # A provider-search response grounds a direct booking follow-up such as "book with
    # Dr. Carter", "book this doctor", or simply "book the appointment" when that response
    # offered exactly one provider. MCP reads must still resolve and validate provider and
    # availability identifiers; conversation text never supplies trusted internal IDs.
    clarification_answer = lower_query.strip(" .?!") in {
        "doctor appointment",
        "doctor/provider appointment",
        "provider appointment",
    }
    provider_followup = (
        bool(re.search(r"\b(?:book|schedule|make)\b", lower_query) and "appointment" in lower_query)
        or clarification_answer
        or bool(
            re.search(r"\b(?:book|schedule)\b", lower_query)
            and re.search(r"\b(?:any|available)\s+(?:doctor|provider)\b", lower_query)
        )
    )
    if provider_followup:
        # Use the immediately preceding assistant result first so an older provider search cannot
        # hijack a new conversation topic.
        preceding_assistant = assistant_messages[-1] if assistant_messages else ""
        if (
            clarification_answer
            and "which type of appointment" in preceding_assistant.casefold()
            and len(assistant_messages) > 1
        ):
            preceding_assistant = assistant_messages[-2]
        provider_names = list(
            dict.fromkeys(
                match.strip()
                for match in re.findall(r"\bDr\.\s+[A-Z][A-Za-z'-]+", preceding_assistant)
            )
        )
        named_in_query = [
            provider
            for provider in provider_names
            if provider.casefold() in lower_query
            or re.search(
                rf"\bdoctor\s+{re.escape(provider.split()[-1])}\b",
                query,
                re.IGNORECASE,
            )
        ]
        selected_provider = (
            named_in_query[0]
            if len(named_in_query) == 1
            else provider_names[0]
            if len(provider_names) == 1
            else None
        )
        if selected_provider:
            prior_request = next(
                (
                    value
                    for value in reversed(previous_user_queries)
                    if not re.search(
                        r"\b(?:this|that)\s+(?:doctor|provider|physician)\b",
                        value,
                        re.IGNORECASE,
                    )
                ),
                "",
            )
            return (
                f"{query}. This is a doctor/provider appointment with {selected_provider}, "
                f"selected in the preceding response. Previous request: {prior_request}. "
                "Resolve provider and availability IDs using MCP records; do not ask for the "
                "appointment type."
            )
        prior_provider_request = next(
            (
                value
                for value in reversed(previous_user_queries)
                if re.search(r"\b(?:doctor|provider|physician|orthopedic)\b", value, re.I)
                and re.search(r"\b(?:find|search|available|knee|leg|shoulder|hip)\b", value, re.I)
            ),
            "",
        )
        if prior_provider_request:
            return (
                f"Book a doctor/provider appointment using the first verified available "
                f"provider-slot pair. Previous request: {prior_provider_request}. "
                "Resolve provider_id and availability_id through MCP records. These are internal "
                "identifiers and must never be requested from the user."
            )
    explicit_transport_request = any(
        term in lower_query
        for term in (
            "transport",
            "ride",
            "pickup",
            "pick up",
            "drop off",
            "drop back",
            "round trip",
            "round-trip",
            "wheelchair",
        )
    )
    address_reply = bool(
        re.search(
            r"(?:address\s*(?:(?:is\s*)?[:=\-]\s*)|^)\s*"
            r"\d+\s+[^;]+",
            query.strip(),
            re.IGNORECASE,
        )
    )
    wheelchair_choice_reply = bool(
        re.search(
            r"\b(?:yes|no)\b.{0,30}\bwheelchair\b|"
            r"\bwheelchair\b.{0,30}\b(?:yes|no|required|needed|not required|not needed)\b",
            lower_query,
        )
    )
    # Conversation history may supply missing values only for an actual continuation. Never
    # replace a new domain request merely because an older transportation turn contains an APT ID
    # and address.
    if not (explicit_transport_request or address_reply or wheelchair_choice_reply):
        return query
    if not any(term in recent.casefold() for term in ("transport", "pickup", "home address")):
        return query
    appointment_ids = re.findall(r"\bAPT[\w-]*\b", recent, re.IGNORECASE)
    address_match = re.search(
        r"(?:address\s*(?:(?:is\s*)?[:=\-]\s*)|^)\s*"
        r"(\d+\s+.*?\b\d{5}(?:-\d{4})?)\b",
        query.strip(),
        re.IGNORECASE,
    )
    if address_match is None:
        address_match = re.search(
            r"(?:address\s*(?:(?:is\s*)?[:=\-]\s*)|^)\s*"
            r"(\d+\s+.*?)(?=\s+and\s+(?:wheelchair|round\s+trip)|[;.!?]|$)",
            query.strip(),
            re.IGNORECASE,
        )
    if not appointment_ids:
        return query
    recent_address = re.findall(
        r"(?:from\s+|address\s*(?:(?:is\s*)?[:\-]\s*))"
        r"\[?(\d+\s+.*?\b\d{5}(?:-\d{4})?)\]?(?:[.;]|$)",
        recent,
        re.IGNORECASE,
    )
    resolved_address = address_match.group(1).strip() if address_match else None
    if resolved_address is None and recent_address:
        resolved_address = recent_address[-1].strip()
    wheelchair_choice: str | None = None
    if re.search(
        r"\b(?:no|without)\s+(?:a\s+)?wheelchair|"
        r"\b(?:do\s+not|don't|does\s+not|doesn't)\s+(?:need|require)\s+(?:a\s+)?wheelchair",
        lower_query,
    ):
        wheelchair_choice = "no"
    elif "wheelchair" in lower_query:
        wheelchair_choice = "yes"
    if not resolved_address:
        return query
    wheelchair = (
        f"; wheelchair assistance: {wheelchair_choice}" if wheelchair_choice is not None else ""
    )
    round_trip_choice: str | None = None
    if re.search(r"\bround\s*[- ]?trip\s*(?:=|:|is)?\s*yes\b", lower_query):
        round_trip_choice = "yes"
    elif re.search(r"\bround\s*[- ]?trip\s*(?:=|:|is)?\s*no\b", lower_query):
        round_trip_choice = "no"
    round_trip = f"; round trip: {round_trip_choice}" if round_trip_choice else ""
    return (
        f"Book round-trip transportation for {appointment_ids[-1].upper()} from "
        f"{resolved_address}{wheelchair}{round_trip}."
    )


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    user_id: str | None = None
    active_case_id: str | None = None
    thread_id: str | None = None
    recipient_id: str | None = None


class CaseStatusUpdate(BaseModel):
    status: str
    status_note: str


class ApprovalRequest(BaseModel):
    user_id: str


def create_api(application: SeniorCareApplication | None = None) -> FastAPI:
    runtime = application or create_application()

    @asynccontextmanager
    async def lifespan(_api: FastAPI):
        await runtime.initialize_agents()
        yield

    api = FastAPI(title="SeniorCare Connect AI", version="0.2.0", lifespan=lifespan)

    @api.middleware("http")
    async def terminal_request_trace(request: Request, call_next):
        token = begin_request(request.headers.get("x-request-id"))
        operation = f"{request.method} {request.url.path}"
        started = time.perf_counter()
        flow_event("api", operation, "input", {"query": dict(request.query_params)})
        try:
            response = await call_next(request)
            flow_event(
                "api",
                operation,
                "output",
                {"statusCode": response.status_code},
                duration_ms=(time.perf_counter() - started) * 1000,
                status="success" if response.status_code < 400 else "failed",
            )
            return response
        except Exception as exc:
            flow_event(
                "api",
                operation,
                "error",
                exc,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        finally:
            end_request(token)

    @api.exception_handler(MCPToolError)
    async def mcp_unavailable(_request: Request, exc: MCPToolError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc), "mcpAvailable": False},
        )

    @api.exception_handler(Exception)
    async def unexpected_backend_error(_request: Request, exc: Exception) -> JSONResponse:
        # Uvicorn still logs the traceback; do not expose stack traces or secrets to the browser.
        return JSONResponse(
            status_code=500,
            content={
                "detail": (
                    "The SeniorCare backend could not complete the request. "
                    f"Error type: {type(exc).__name__}. Check the seniorcare-api terminal."
                )
            },
        )

    @api.get("/health")
    async def health() -> dict:
        try:
            mcp_status = await runtime.mcp.call("server_status")
            mcp_tools = await runtime.mcp.list_tool_names()
        except Exception as exc:
            return {
                "status": "DEGRADED",
                "simulation": runtime.settings.simulation_mode,
                "externalMutationsAllowed": runtime.settings.allow_external_mutations,
                "mcp": {
                    "status": "UNAVAILABLE",
                    "url": runtime.settings.mcp_server_url,
                    "error": str(exc),
                },
            }
        return {
            "status": "OK",
            "simulation": runtime.settings.simulation_mode,
            "externalMutationsAllowed": runtime.settings.allow_external_mutations,
            "ragChunks": mcp_status.get("ragChunks", 0),
            "agents": runtime.agent_status(),
            "mcp": {
                "status": "OK",
                "server": "seniorcare-connect",
                "url": runtime.settings.mcp_server_url,
                "toolCount": len(mcp_tools),
            },
        }

    @api.get("/mcp/tools")
    async def mcp_tools() -> dict[str, Any]:
        """Expose MCP discovery information without executing a tool."""
        return {"server": "seniorcare-connect", "tools": await runtime.mcp.list_tool_names()}

    @api.post("/members/register")
    async def register(request: MemberRegistration) -> dict:
        return await runtime.mcp.call("register_member", **request.model_dump(mode="json"))

    @api.get("/members/{user_id}")
    async def member(user_id: str) -> dict:
        value = await runtime.mcp.call("get_member", user_id=user_id)
        if not value:
            raise HTTPException(404, "Member not found")
        return {
            "simulation": True,
            "externalActionPerformed": False,
            "data": {"recognized": True, "member": value},
        }

    @api.post("/members/{user_id}/care-recipients")
    async def add_care_recipient(user_id: str, request: CareRecipientRegistration) -> dict:
        return await runtime.mcp.call(
            "add_care_recipient", user_id=user_id, **request.model_dump(mode="json")
        )

    @api.get("/members/{user_id}/cases")
    async def cases(user_id: str) -> dict:
        member_value = await runtime.mcp.call("get_member", user_id=user_id)
        if not member_value:
            raise HTTPException(404, "Member not found")
        await runtime.mcp.call("close_due_cases", user_id=user_id)
        values = unwrap_record_list(
            await runtime.mcp.call("list_cases", user_id=user_id), allow_singleton=True
        )
        for case_value in values:
            related_ids = case_value.get("relatedEntityIds") or []
            case_value["relatedRecords"] = unwrap_record_list(
                await runtime.mcp.call(
                    "get_case_related_records", user_id=user_id, entity_ids=related_ids
                ),
                allow_singleton=True,
            )
        return {
            "simulation": True,
            "externalActionPerformed": False,
            "data": {"member": member_value, "cases": values},
        }

    @api.post("/members/{user_id}/cases")
    async def create_case(user_id: str, request: NewCaseRequest) -> dict:
        return await runtime.mcp.call(
            "create_case", user_id=user_id, request=request.model_dump(mode="json")
        )

    @api.get("/cases/{case_id}")
    async def case(case_id: str, user_id: str) -> dict:
        try:
            return await runtime.mcp.call("get_case", user_id=user_id, case_id=case_id)
        except Exception as exc:
            raise HTTPException(404, "Case not found") from exc

    @api.patch("/cases/{case_id}")
    async def update_case(case_id: str, user_id: str, request: CaseStatusUpdate) -> dict:
        allowed = {
            "open",
            "in_progress",
            "blocked",
            "waiting_for_user",
            "resolved",
            "closed",
            "cancelled",
        }
        if request.status not in allowed:
            raise HTTPException(422, "Invalid case status")
        return await runtime.mcp.call(
            "update_case_status",
            user_id=user_id,
            case_id=case_id,
            status=request.status,
            status_note=request.status_note,
        )

    @api.post("/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        if not request.user_id:
            thread_id = request.thread_id or f"SESSION-{uuid4().hex[:16]}"
            state: dict[str, Any] = {
                "raw_user_query": request.query,
                "user_id": None,
                "active_case_id": request.active_case_id,
                "recipient_id": request.recipient_id,
                "errors": [],
            }
            result = await runtime.graph.ainvoke(  # type: ignore[attr-defined]
                state,
                config={"configurable": {"thread_id": f"{thread_id}-{uuid4().hex[:8]}"}},
            )
            return {"sessionId": thread_id, "threadId": thread_id, **result}

        try:
            session = runtime.sessions.get_or_create(request.user_id, request.thread_id)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        superseded_action_ids = runtime.sessions.discard_pending_actions(session.session_id)
        discarded_action_ids = [
            action_id
            for action_id in superseded_action_ids
            if runtime.approvals.discard(action_id, request.user_id)
        ]
        if discarded_action_ids:
            flow_event(
                "session",
                "superseded_approvals",
                "output",
                {"sessionId": session.session_id, "actionIds": discarded_action_ids},
            )
        active_case_id = request.active_case_id or session.active_case_id
        recipient_id = request.recipient_id or session.recipient_id
        if recipient_id:
            runtime.sessions.set_recipient(session.session_id, recipient_id)
        contextual_query = contextualize_followup(request.query, session)
        if contextual_query != request.query:
            flow_event(
                "session",
                "followup_context",
                "output",
                {"originalQuery": request.query, "contextualQuery": contextual_query},
            )
        state = {
            "raw_user_query": contextual_query,
            "conversation_history": [
                {"role": message.role, "content": message.content} for message in session.messages
            ],
            "user_id": request.user_id,
            "active_case_id": active_case_id,
            "recipient_id": recipient_id,
            "errors": [],
        }
        result = await runtime.graph.ainvoke(  # type: ignore[attr-defined]
            state,
            config={"configurable": {"thread_id": f"{session.session_id}-{uuid4().hex[:8]}"}},
        )
        runtime.sessions.record_exchange(session.session_id, request.query, result)
        return {
            "sessionId": session.session_id,
            "threadId": session.session_id,
            "activeCaseId": result.get("active_case_id"),
            **result,
        }

    @api.get("/sessions/{session_id}")
    def session(session_id: str, user_id: str) -> dict[str, Any]:
        try:
            return runtime.sessions.snapshot(session_id, user_id)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @api.post("/actions/{action_id}/approve")
    async def approve(action_id: str, request: ApprovalRequest) -> dict[str, Any]:
        agent_session = runtime.sessions.find_by_action(action_id, request.user_id)
        raw_action = runtime.approvals.repo.get(action_id)
        try:
            executed = await runtime.approvals.approve(action_id, request.user_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (PermissionError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

        continuation: dict[str, Any] | None = None
        if agent_session:
            runtime.sessions.resolve_action(action_id, "executed")
            case_id = executed.get("case_id")
            if not case_id and raw_action and raw_action.get("action_type") == "create_case":
                result = executed.get("result")
                result_data = result.get("data") if isinstance(result, dict) else None
                if isinstance(result_data, dict):
                    case_id = result_data.get("caseId")
            if case_id:
                runtime.sessions.set_active_case(agent_session.session_id, str(case_id))
        return {
            "action": executed,
            "sessionId": agent_session.session_id if agent_session else None,
            "activeCaseId": (
                runtime.sessions.snapshot(agent_session.session_id, request.user_id).get(
                    "active_case_id"
                )
                if agent_session
                else None
            ),
            "continuation": continuation,
        }

    @api.post("/actions/{action_id}/reject")
    def reject(action_id: str, request: ApprovalRequest) -> dict[str, Any]:
        agent_session = runtime.sessions.find_by_action(action_id, request.user_id)
        try:
            rejected = runtime.approvals.reject(action_id, request.user_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if agent_session:
            runtime.sessions.resolve_action(action_id, "rejected")
        return {
            "action": rejected,
            "sessionId": agent_session.session_id if agent_session else None,
        }

    return api


app = create_api()


def run() -> None:
    uvicorn.run("seniorcare_agents.api.app:app", host="127.0.0.1", port=8000, reload=False)
