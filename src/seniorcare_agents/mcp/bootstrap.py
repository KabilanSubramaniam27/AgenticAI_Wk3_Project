from seniorcare_agents.mcp.server import create_seniorcare_mcp_server
from seniorcare_agents.retrieval import (
    ActianSemanticRetriever,
    BM25Retriever,
    CrossEncoderReranker,
    HybridRetriever,
)
from seniorcare_ingestion.config import Settings as IngestionSettings
from seniorcare_ingestion.embeddings import NebiusEmbeddingProvider
from seniorcare_ingestion.vectorstore import ActianVectorStore
from seniorcare_runtime.config import RuntimeSettings


def build_mcp_server(settings: RuntimeSettings | None = None):
    """Build the independently deployed MCP service and its retrieval dependencies."""
    runtime = settings or RuntimeSettings()
    ingestion = IngestionSettings(project_root=runtime.project_root)
    semantic = None
    if ingestion.nebius_api_key:
        semantic = ActianSemanticRetriever(
            runtime,
            NebiusEmbeddingProvider(ingestion),
            ActianVectorStore(ingestion),
        )
    retriever = HybridRetriever(
        runtime,
        BM25Retriever(runtime),
        semantic,
        CrossEncoderReranker(runtime.reranker_model),
    )
    return create_seniorcare_mcp_server(runtime, retriever)
