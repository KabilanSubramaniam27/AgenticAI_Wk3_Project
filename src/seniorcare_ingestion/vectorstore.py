import hashlib
from abc import ABC, abstractmethod
from typing import Any

from seniorcare_ingestion.config import Settings


class VectorStore(ABC):
    @abstractmethod
    def health_check(self) -> bool: ...
    @abstractmethod
    def collection_exists(self, name: str) -> bool: ...
    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None: ...
    @abstractmethod
    def upsert(self, name: str, records: list[dict]) -> None: ...
    @abstractmethod
    def search(
        self, name: str, vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[dict]: ...
    @abstractmethod
    def delete(self, name: str, ids: list[str]) -> None: ...
    @abstractmethod
    def count(self, name: str) -> int: ...


class ActianVectorStore(VectorStore):
    """Thin adapter matching actian-vectorai-client used by the reference project.

    SDK imports are lazy so local processing and unit tests do not require a running DB.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def _client(self):
        from actian_vectorai import VectorAIClient

        kwargs: dict[str, Any] = {"url": self.settings.actian_vectorai_url}
        if self.settings.actian_vectorai_access_token:
            kwargs["access_token"] = self.settings.actian_vectorai_access_token
        return VectorAIClient(**kwargs)

    @staticmethod
    def _point_id(value: str) -> int:
        return int(hashlib.sha256(value.encode()).hexdigest()[:15], 16)

    def health_check(self) -> bool:
        with self._client() as client:
            client.collections.list()
        return True

    def collection_exists(self, name: str) -> bool:
        with self._client() as client:
            return client.collections.exists(name)

    def create_collection(self, name: str, dimension: int) -> None:
        from actian_vectorai import Distance, VectorParams  # type: ignore[attr-defined]

        with self._client() as client:
            client.collections.get_or_create(
                name=name, vectors_config=VectorParams(size=dimension, distance=Distance.Cosine)
            )

    def upsert(self, name: str, records: list[dict]) -> None:
        from actian_vectorai import PointStruct  # type: ignore[attr-defined]

        if not records:
            return
        self.create_collection(name, len(records[0]["values"]))
        with self._client() as client:
            points = [
                PointStruct(
                    id=self._point_id(row["id"]),
                    vector=row["values"],
                    payload={**row["metadata"], "chunk_id": row["id"]},
                )
                for row in records
            ]
            client.points.upsert(name, points=points)
            client.vde.flush(name)

    def search(
        self, name: str, vector: list[float], top_k: int, filters: dict | None = None
    ) -> list[dict]:
        with self._client() as client:
            results = client.points.search(name, vector=vector, limit=top_k, with_payload=True)
        rows = [
            {
                "id": item.payload.get("chunk_id", str(item.id)),
                "score": item.score,
                "metadata": item.payload,
            }
            for item in results or []
        ]
        return (
            rows
            if not filters
            else [
                row for row in rows if all(row["metadata"].get(k) == v for k, v in filters.items())
            ]
        )

    def delete(self, name: str, ids: list[str]) -> None:
        if ids:
            with self._client() as client:
                client.points.delete(name, ids=[self._point_id(value) for value in ids])

    def count(self, name: str) -> int:
        with self._client() as client:
            return client.points.count(name)
