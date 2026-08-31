import asyncio

from seniorcare_agents.models import RetrievedChunk
from seniorcare_agents.retrieval.filters import matches_filters
from seniorcare_ingestion.embeddings import EmbeddingProvider
from seniorcare_ingestion.vectorstore import VectorStore
from seniorcare_runtime.config import RuntimeSettings


class ActianSemanticRetriever:
    def __init__(self, settings: RuntimeSettings, embedder: EmbeddingProvider, store: VectorStore):
        self.settings = settings
        self.embedder = embedder
        self.store = store

    async def retrieve(
        self,
        query: str,
        categories: list[str],
        geography: dict | None = None,
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        vector = await self.embedder.embed_query(query)
        limit = top_k or self.settings.rag_vector_top_k
        rows = await asyncio.to_thread(
            self.store.search, "seniorcare_knowledge", vector, max(limit * 4, 40), None
        )
        results: list[RetrievedChunk] = []
        for row in rows:
            metadata = row["metadata"]
            if not matches_filters(metadata, categories, geography, filters):
                continue
            results.append(
                RetrievedChunk.model_validate(
                    {
                        **metadata,
                        "last_verified": metadata.get("retrieved_at"),
                        "vector_score": row["score"],
                        "vector_rank": len(results) + 1,
                        "retrieved_by": ["vector"],
                    }
                )
            )
            if len(results) == limit:
                break
        return results
