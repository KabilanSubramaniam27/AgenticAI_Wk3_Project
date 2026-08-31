import json
import math
import re
from collections import Counter
from pathlib import Path

from seniorcare_agents.models import RetrievedChunk
from seniorcare_agents.retrieval.filters import matches_filters
from seniorcare_ingestion.utils import digest
from seniorcare_runtime.config import RuntimeSettings

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


class BM25Retriever:
    def __init__(self, settings: RuntimeSettings, chunks_path: Path | None = None):
        self.settings = settings
        self.path = chunks_path or settings.project_root / "data/processed/chunks.jsonl"
        self.rows = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokens = [tokenize(row["content"]) for row in self.rows]
        self.lengths = [len(values) for values in self.tokens]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.document_frequency = Counter(term for values in self.tokens for term in set(values))
        self.manifest_path = settings.project_root / "data/index/index_manifest.json"
        self._write_manifest()

    def _write_manifest(self) -> None:
        fingerprint = digest("".join(row["chunk_id"] + row["content_hash"] for row in self.rows))
        ingestion_manifest = self.settings.project_root / "data/manifests/ingestion_manifest.json"
        indexed_ids = set()
        if ingestion_manifest.exists():
            indexed_ids = set(
                json.loads(ingestion_manifest.read_text(encoding="utf-8")).get("chunks", {})
            )
        chunk_ids = [row["chunk_id"] for row in self.rows]
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "corpusFingerprint": fingerprint,
                    "chunkCount": len(self.rows),
                    "chunkIds": chunk_ids,
                    "actianCheckpointCount": len(indexed_ids),
                    "corpusMatchesActianCheckpoint": set(chunk_ids) == indexed_ids,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def retrieve(
        self,
        query: str,
        categories: list[str],
        geography: dict | None = None,
        filters: dict | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        query_terms = tokenize(query)
        count = len(self.rows)
        scored = []
        for index, (row, values) in enumerate(zip(self.rows, self.tokens, strict=True)):
            if not matches_filters(row, categories, geography, filters):
                continue
            frequencies = Counter(values)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                inverse = math.log(1 + (count - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * self.lengths[index] / max(1, self.average_length)
                )
                score += inverse * frequency * 2.5 / denominator
            if score:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for rank, (score, row) in enumerate(scored[: top_k or self.settings.rag_bm25_top_k], 1):
            results.append(
                RetrievedChunk.model_validate(
                    {
                        **row,
                        "last_verified": row.get("retrieved_at"),
                        "bm25_score": score,
                        "bm25_rank": rank,
                        "retrieved_by": ["bm25"],
                    }
                )
            )
        return results
