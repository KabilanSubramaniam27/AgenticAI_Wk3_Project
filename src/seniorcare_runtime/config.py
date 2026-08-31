from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    project_root: Path = Path.cwd()
    simulation_mode: bool = True
    allow_external_mutations: bool = False
    require_approval_for_writes: bool = True
    rag_bm25_top_k: int = 20
    rag_vector_top_k: int = 20
    rag_fusion_top_k: int = 15
    rag_final_top_k: int = 6
    rrf_k: int = 60
    enable_reranker: bool = True
    reranker_provider: str = "local"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    stale_resource_days: int = 90
    stuck_case_days: int = 7
    mcp_server_url: str = "http://127.0.0.1:8001/mcp"
    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8001
    mcp_server_path: str = "/mcp"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = Field(
        default="", validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_API_KEY")
    )
    llm_base_url: str | None = None
    llm_temperature: float = 0.0
    mcp_read_max_attempts: int = 3
    mcp_read_retry_base_seconds: float = 0.25
    session_ttl_minutes: int = 1440

    @property
    def synthetic_dir(self) -> Path:
        return self.project_root / "data/synthetic-data"

    @property
    def audit_path(self) -> Path:
        return self.project_root / "data/runtime/audit.jsonl"

    @property
    def trace_path(self) -> Path:
        return self.project_root / "data/runtime/retrieval_traces.jsonl"

    @property
    def observability_path(self) -> Path:
        return self.project_root / "data/runtime/observability.jsonl"

    @property
    def session_state_path(self) -> Path:
        return self.project_root / "data/runtime/agent_sessions.json"

    @property
    def pending_actions_path(self) -> Path:
        return self.project_root / "data/runtime/pending_actions.json"

    def require_simulation(self) -> None:
        if not self.simulation_mode or self.allow_external_mutations:
            raise RuntimeError(
                "SeniorCare study tools require SIMULATION_MODE=true and "
                "ALLOW_EXTERNAL_MUTATIONS=false"
            )
