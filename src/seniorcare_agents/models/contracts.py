from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    finding_type: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    description: str
    priority: Literal["low", "medium", "high", "urgent"] = "medium"
    rationale: str | None = None


class ProposedAction(BaseModel):
    action_id: str
    action_type: str
    description: str
    parameters: dict[str, Any]
    # LLMs may omit trusted bookkeeping fields; specialists bind and validate them
    # deterministically from the current authenticated invocation before registration.
    agent_name: str = ""
    user_id: str = ""
    senior_id: str = ""
    recipient_id: str | None = None
    case_id: str | None = None
    requires_approval: bool = True
    simulation: bool = True
    status: Literal["proposed", "approved", "rejected", "executed", "failed"] = "proposed"


class ToolCallRecord(BaseModel):
    tool: str
    operation: Literal["read", "write"]
    status: str
    entity_ids: list[str] = Field(default_factory=list)
    simulation: bool = True
    external_action_performed: bool = False


class RiskFlag(BaseModel):
    risk_id: str
    senior_id: str
    case_id: str | None = None
    category: str
    severity: Literal["low", "attention", "medium", "high", "critical"]
    reason: str
    related_entity_ids: list[str] = Field(default_factory=list)
    rule_id: str
    recommended_next_step: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    title: str | None = None
    section_title: str | None = None
    program_name: str | None = None
    category: str
    subcategory: str | None = None
    source_name: str
    source_url: str | None = None
    source_trust_tier: int | None = None
    state: str | None = None
    county: str | None = None
    city: str | None = None
    service_area: list[str] = Field(default_factory=list)
    last_verified: datetime | None = None
    last_updated_date: datetime | None = None
    page_number: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    vector_score: float | None = None
    vector_rank: int | None = None
    fusion_score: float | None = None
    fusion_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    retrieved_by: list[str] = Field(default_factory=list)
    stale: bool = False
    freshness_warning: str | None = None


class Citation(BaseModel):
    citation_id: str
    source_name: str
    title: str | None = None
    program_name: str | None = None
    source_url: str | None = None
    page_number: int | None = None
    last_verified: datetime | None = None


class AgentResult(BaseModel):
    agent_name: str
    status: Literal["success", "partial", "blocked", "failed", "needs_user_input"]
    case_id: str | None = None
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    proposed_actions: list[ProposedAction] = Field(default_factory=list)
    related_entity_ids: list[str] = Field(default_factory=list)
    retrieved_sources: list[RetrievedChunk] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    risks: list[RiskFlag] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0, le=1)


class ExecutionStage(BaseModel):
    stage: int = Field(ge=1)
    agents: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)


class OrchestratorPlan(BaseModel):
    intents: list[str] = Field(min_length=1)
    selected_agents: list[str] = Field(min_length=1)
    execution_stages: list[ExecutionStage] = Field(min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    routing_summary: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0, le=1)


class SpecialistPlan(BaseModel):
    task_summary: str
    selected_tools: list[str] = Field(default_factory=list)
    tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    retrieval_queries: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0, le=1)


class OrchestratorResponse(BaseModel):
    answer: str
    completed_agents: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    action_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0, le=1)
