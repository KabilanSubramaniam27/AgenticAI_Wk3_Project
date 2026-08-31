import asyncio
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from seniorcare_agents.agents.llm_specialist import READ_TOOL_POLICIES, LangGraphSpecialist
from seniorcare_agents.agents.orchestrator import SeniorCareOrchestratorAgent
from seniorcare_agents.graph import SeniorCareGraphBuilder
from seniorcare_agents.graph.approvals import ApprovalManager
from seniorcare_agents.mcp import MCPToolGateway
from seniorcare_agents.services import PersistentAgentSessionStore
from seniorcare_runtime.config import RuntimeSettings


@dataclass
class SeniorCareApplication:
    settings: RuntimeSettings
    graph: object
    approvals: ApprovalManager
    sessions: PersistentAgentSessionStore
    mcp: MCPToolGateway
    agents: dict[str, object]
    orchestrator: SeniorCareOrchestratorAgent
    model: BaseChatModel

    async def initialize_agents(self) -> dict[str, dict[str, object]]:
        """Compile all configured specialist graphs once for reuse across requests."""
        specialists = {
            key: agent
            for key, agent in self.agents.items()
            if isinstance(agent, LangGraphSpecialist)
        }
        outcomes = await asyncio.gather(
            *(agent.initialize() for agent in specialists.values()), return_exceptions=True
        )
        report: dict[str, dict[str, object]] = {}
        for (key, agent), outcome in zip(specialists.items(), outcomes, strict=True):
            if isinstance(outcome, BaseException):
                report[key] = {"initialized": False, "error": str(outcome)}
            else:
                report[key] = {
                    "initialized": agent.initialized,
                    "error": agent.initialization_error,
                }
        return report

    def agent_status(self) -> dict[str, dict[str, object]]:
        return {
            key: {
                "initialized": agent.initialized,
                "error": agent.initialization_error,
            }
            for key, agent in self.agents.items()
            if isinstance(agent, LangGraphSpecialist)
        }

    def invalidate_agents(self) -> None:
        self.mcp.clear_cache()
        for agent in self.agents.values():
            if isinstance(agent, LangGraphSpecialist):
                agent.invalidate()


def create_application(
    settings: RuntimeSettings | None = None,
    gateway: MCPToolGateway | None = None,
    model: BaseChatModel | None = None,
) -> SeniorCareApplication:
    """Build the agent API process; all domain capabilities live in remote MCP."""
    runtime = settings or RuntimeSettings()
    mcp = gateway or MCPToolGateway(
        runtime.mcp_server_url,
        runtime.mcp_read_max_attempts,
        runtime.mcp_read_retry_base_seconds,
    )
    configured = model is not None or bool(runtime.llm_model and runtime.llm_api_key)
    llm = model or _build_model(runtime)
    agents: dict[str, object] = {
        key: LangGraphSpecialist(key, mcp, llm, configured) for key in READ_TOOL_POLICIES
    }
    orchestrator = SeniorCareOrchestratorAgent(agents, llm, configured)
    builder = SeniorCareGraphBuilder(
        runtime,
        orchestrator,
        mcp,
    )
    return SeniorCareApplication(
        settings=runtime,
        graph=builder.build(),
        approvals=builder.approvals,
        sessions=PersistentAgentSessionStore(
            runtime.session_state_path, ttl_minutes=runtime.session_ttl_minutes
        ),
        mcp=mcp,
        agents=agents,
        orchestrator=orchestrator,
        model=llm,
    )


def _build_model(settings: RuntimeSettings) -> BaseChatModel:
    provider = settings.llm_provider.casefold()
    if provider not in {"openai", "nebius", "openai_compatible"}:
        raise ValueError("LLM_PROVIDER must be 'openai', 'nebius', or 'openai_compatible'")
    if provider == "openai":
        return ChatOpenAI(
            model=settings.llm_model or "gpt-4o-mini",
            api_key=SecretStr(settings.llm_api_key or "configuration-required"),
            temperature=settings.llm_temperature,
        )
    return ChatOpenAI(
        model=settings.llm_model or "gpt-4o-mini",
        api_key=SecretStr(settings.llm_api_key or "configuration-required"),
        temperature=settings.llm_temperature,
        base_url=settings.llm_base_url or "https://api.tokenfactory.nebius.com/v1",
    )
