import asyncio
import logging
import time
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.processing import validate_vectors

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...


class NebiusEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings):
        if not settings.nebius_api_key:
            raise RuntimeError("NEBIUS_API_KEY is required")
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.nebius_api_key,
            base_url=settings.nebius_base_url,
            timeout=settings.http_timeout_seconds,
        )

    async def _batch(self, texts: list[str], number: int, total: int) -> list[list[float]]:
        for attempt in range(1, self.settings.http_max_retries + 1):
            started = time.perf_counter()
            try:
                logger.info(
                    "embeddings provider=nebius model=%s batch=%s/%s count=%s event=request_started",
                    self.settings.nebius_embedding_model,
                    number,
                    total,
                    len(texts),
                )
                response = await self.client.embeddings.create(
                    model=self.settings.nebius_embedding_model, input=texts
                )
                vectors = [
                    item.embedding for item in sorted(response.data, key=lambda item: item.index)
                ]
                validate_vectors(vectors, len(texts), self.settings.embedding_dimension)
                logger.info(
                    "embeddings provider=nebius vectors=%s dimension=%s elapsed_ms=%.1f event=request_completed",
                    len(vectors),
                    len(vectors[0]),
                    (time.perf_counter() - started) * 1000,
                )
                return vectors
            except Exception:
                if attempt == self.settings.http_max_retries:
                    raise
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("unreachable")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        size = self.settings.embedding_batch_size
        total = (len(texts) + size - 1) // size
        vectors = []
        for number, start in enumerate(range(0, len(texts), size), 1):
            vectors.extend(await self._batch(texts[start : start + size], number, total))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._batch([text], 1, 1))[0]
