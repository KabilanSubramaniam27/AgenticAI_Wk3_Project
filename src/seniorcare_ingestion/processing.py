import math
import re
from collections import defaultdict
from typing import Any

from rapidfuzz.fuzz import ratio

from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.models import NormalizedDocument, RagChunk
from seniorcare_ingestion.utils import clean_text, digest, stable_id


def deduplicate(
    documents: list[NormalizedDocument], threshold: float = 96
) -> tuple[list[NormalizedDocument], dict[str, int]]:
    exact: set[str] = set()
    unique: list[NormalizedDocument] = []
    exact_removed = near_removed = 0
    by_source: dict[str, list[NormalizedDocument]] = defaultdict(list)
    for document in documents:
        if document.content_hash in exact:
            exact_removed += 1
            continue
        if any(
            ratio(document.content[:5000], other.content[:5000]) >= threshold
            for other in by_source[document.source_id]
        ):
            near_removed += 1
            continue
        exact.add(document.content_hash)
        unique.append(document)
        by_source[document.source_id].append(document)
    return unique, {
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
    }


class StructureAwareChunker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.encoding: Any | None = None
        try:
            import tiktoken

            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            pass

    def tokens(self, text: str) -> list:
        return self.encoding.encode(text) if self.encoding else text.split()

    def decode(self, values: list) -> str:
        return self.encoding.decode(values) if self.encoding else " ".join(values)

    def chunk(self, document: NormalizedDocument) -> list[RagChunk]:
        sections = re.split(
            r"(?=^#{1,3}\s+|^[A-Z][^\n]{2,80}\n[-=]{3,}$)", document.content, flags=re.MULTILINE
        )
        pieces: list[tuple[str | None, str]] = []
        for section in sections:
            section = clean_text(section)
            if not section:
                continue
            heading = (
                section.splitlines()[0].lstrip("# ") if len(section.splitlines()[0]) < 100 else None
            )
            values = self.tokens(section)
            target = self.settings.chunk_target_tokens
            overlap = self.settings.chunk_overlap_tokens
            for start in range(0, len(values), max(1, target - overlap)):
                text = self.decode(values[start : start + target]).strip()
                if (
                    len(self.tokens(text)) >= self.settings.min_chunk_tokens
                    and len(text) >= self.settings.min_chunk_characters
                ):
                    prefix = f"Title: {document.title}\nCategory: {document.category}\nLocation: {document.state or document.country}\n"
                    pieces.append(
                        (heading, prefix + (f"Section: {heading}\n" if heading else "") + text)
                    )
                if start + target >= len(values):
                    break
        count = len(pieces)
        chunks = []
        for index, (heading, content) in enumerate(pieces):
            content_hash = digest(clean_text(content).lower())
            chunks.append(
                RagChunk(
                    chunk_id=stable_id(document.document_id, index, content_hash),
                    document_id=document.document_id,
                    source_id=document.source_id,
                    source_name=document.source_name,
                    source_url=document.source_url,
                    organization=document.organization,
                    category=document.category,
                    subcategory=document.subcategory,
                    title=document.title,
                    section_title=heading,
                    content=content,
                    chunk_index=index,
                    chunk_count=count,
                    country=document.country,
                    state=document.state,
                    county=document.county,
                    city=document.city,
                    zip_codes=document.zip_codes,
                    service_area=document.service_area,
                    populations=document.populations,
                    tags=document.tags,
                    program_name=document.program_name,
                    authority_level=document.authority_level,
                    source_trust_tier=document.source_trust_tier,
                    effective_date=document.effective_date,
                    published_date=document.published_date,
                    last_updated_date=document.last_updated_date,
                    retrieved_at=document.retrieved_at,
                    page_number=document.page_number,
                    document_type=document.document_type,
                    language=document.language,
                    content_hash=content_hash,
                )
            )
        return chunks


def validate_vectors(
    vectors: list[list[float]], expected_count: int, dimension: int = 4096
) -> None:
    if len(vectors) != expected_count:
        raise ValueError(f"embedding count mismatch: expected {expected_count}, got {len(vectors)}")
    for vector in vectors:
        if not vector:
            raise ValueError("empty embedding")
        if len(vector) != dimension:
            raise ValueError(
                f"embedding dimension mismatch: expected {dimension}, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding contains NaN or Infinity")
