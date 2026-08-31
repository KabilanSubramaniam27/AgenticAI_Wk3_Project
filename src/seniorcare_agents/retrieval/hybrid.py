import json
import time
from datetime import UTC, datetime, timedelta

from seniorcare_agents.models import RetrievedChunk
from seniorcare_agents.observability import flow_event
from seniorcare_agents.retrieval.bm25 import BM25Retriever
from seniorcare_agents.retrieval.reranker import Reranker
from seniorcare_agents.retrieval.rrf import reciprocal_rank_fusion
from seniorcare_agents.retrieval.semantic import ActianSemanticRetriever
from seniorcare_runtime.config import RuntimeSettings


class HybridRetriever:
    def __init__(
        self,
        settings: RuntimeSettings,
        bm25: BM25Retriever,
        semantic: ActianSemanticRetriever | None,
        reranker: Reranker,
    ):
        self.settings = settings
        self.bm25 = bm25
        self.semantic = semantic
        self.reranker = reranker

    async def retrieve(
        self,
        query: str,
        categories: list[str],
        geography: dict | None = None,
        filters: dict | None = None,
        top_k: int | None = None,
        agent: str = "unknown",
    ) -> list[RetrievedChunk]:
        flow_event(
            "rag",
            "hybrid_retrieval",
            "input",
            {
                "agent": agent,
                "query": query,
                "categories": categories,
                "geography": geography or {},
                "filters": filters or {},
                "topK": top_k or self.settings.rag_final_top_k,
            },
        )
        started = time.perf_counter()
        bm_started = time.perf_counter()
        lexical = self.bm25.retrieve(
            query.strip(), categories, geography, filters, self.settings.rag_bm25_top_k
        )
        bm_ms = (time.perf_counter() - bm_started) * 1000
        semantic_rows = []
        vector_error = None
        vector_started = time.perf_counter()
        if self.semantic:
            try:
                semantic_rows = await self.semantic.retrieve(
                    query.strip(), categories, geography, filters, self.settings.rag_vector_top_k
                )
            except Exception as exc:
                vector_error = type(exc).__name__
                flow_event("rag", "semantic_retrieval", "error", exc)
        vector_ms = (time.perf_counter() - vector_started) * 1000
        fused = reciprocal_rank_fusion([lexical, semantic_rows], self.settings.rrf_k)[
            : self.settings.rag_fusion_top_k
        ]
        rerank_started = time.perf_counter()
        final = await self.reranker.rerank(query, fused, top_k or self.settings.rag_final_top_k)
        stale_before = datetime.now(UTC) - timedelta(days=self.settings.stale_resource_days)
        for row in final:
            verified = row.last_verified
            if (
                verified
                and (verified if verified.tzinfo else verified.replace(tzinfo=UTC)) < stale_before
            ):
                row.stale = True
                row.freshness_warning = (
                    f"Source was retrieved more than {self.settings.stale_resource_days} days ago."
                )
        trace = {
            "timestamp": datetime.now(UTC).isoformat(),
            "agent": agent,
            "query": query,
            "categories": categories,
            "filters": filters or {},
            "geography": geography or {},
            "bm25ChunkIds": [row.chunk_id for row in lexical],
            "vectorChunkIds": [row.chunk_id for row in semantic_rows],
            "rrf": {row.chunk_id: row.fusion_score for row in fused},
            "rerank": {row.chunk_id: row.rerank_score for row in final},
            "finalChunkIds": [row.chunk_id for row in final],
            "vectorError": vector_error,
            "latencyMs": {
                "bm25": bm_ms,
                "vector": vector_ms,
                "rerank": (time.perf_counter() - rerank_started) * 1000,
                "total": (time.perf_counter() - started) * 1000,
            },
        }
        self.settings.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, default=str) + "\n")
        flow_event(
            "rag",
            "hybrid_retrieval",
            "output",
            {
                "chunkIds": [row.chunk_id for row in final],
                "resultCount": len(final),
                "vectorError": vector_error,
                "latencyMs": trace["latencyMs"],
            },
        )
        return final
