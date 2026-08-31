from typing import Any, TypedDict


class SeniorCareState(TypedDict, total=False):
    request_id: str
    raw_user_query: str
    conversation_history: list[dict[str, str]]
    normalized_query: str
    user_id: str | None
    senior_id: str | None
    recipient_id: str | None
    member_resolved: bool
    member_context: dict[str, Any]
    care_recipient: dict[str, Any]
    caregiver_context: dict[str, Any]
    active_case_id: str | None
    existing_cases: list[dict]
    detected_intents: list[str]
    selected_agents: list[str]
    orchestrator_plan: dict[str, Any]
    agent_results: dict[str, dict]
    operational_results: list[dict]
    retrieved_chunks: list[dict]
    proposed_actions: list[dict]
    risk_flags: list[dict]
    citations: list[dict]
    safety_flags: list[str]
    errors: list[dict]
    requires_human_approval: bool
    approval_status: str | None
    final_response: str | None
