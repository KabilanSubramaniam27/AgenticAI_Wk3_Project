from abc import ABC, abstractmethod

from seniorcare_agents.models import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        import asyncio

        model = self._load()
        scores = await asyncio.to_thread(model.predict, [(query, row.content) for row in chunks])
        ordered = sorted(
            zip(chunks, map(float, scores), strict=True), key=lambda item: item[1], reverse=True
        )
        results = []
        for rank, (row, score) in enumerate(ordered[:top_k], 1):
            item = row.model_copy(deep=True)
            item.rerank_score = score
            item.rerank_rank = rank
            results.append(item)
        return results
