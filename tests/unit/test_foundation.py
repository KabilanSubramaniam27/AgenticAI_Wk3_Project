from datetime import UTC, datetime

import pytest

from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.geography import normalize_place
from seniorcare_ingestion.models import NormalizedDocument
from seniorcare_ingestion.processing import StructureAwareChunker, deduplicate, validate_vectors
from seniorcare_ingestion.utils import canonical_url, clean_text, stable_id


def document(content: str) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=stable_id("source", "title"),
        source_id="source",
        source_name="Official Source",
        source_url="https://cms.gov/a",
        organization="CMS",
        authority_level="federal",
        source_trust_tier=1,
        category="benefits_financial",
        title="Benefits",
        content=content,
        state="Virginia",
        document_type="html",
        retrieved_at=datetime.now(UTC),
        content_hash=stable_id(content),
    )


def test_url_canonicalization_and_stable_ids():
    assert (
        canonical_url("HTTPS://CMS.GOV/path/?utm_source=x&b=2&a=1#top")
        == "https://cms.gov/path?a=1&b=2"
    )
    assert stable_id("A", "B") == stable_id("a", "b")


def test_cleaning_and_geography():
    assert clean_text("A\u200b   B\n\n\nC") == "A B\n\nC"
    assert normalize_place("Henrico") == "Henrico County"
    assert normalize_place("VA") == "Virginia"


def test_exact_deduplication():
    first = document("Useful official benefit details " * 10)
    second = first.model_copy(update={"document_id": "other"})
    unique, counts = deduplicate([first, second])
    assert len(unique) == 1
    assert counts["exact_duplicates_removed"] == 1


def test_structure_aware_chunking_has_stable_ids(tmp_path):
    settings = Settings(
        project_root=tmp_path,
        chunk_target_tokens=30,
        chunk_overlap_tokens=5,
        min_chunk_tokens=5,
        min_chunk_characters=10,
    )
    chunks = StructureAwareChunker(settings).chunk(
        document("# Eligibility\n" + "official requirement details " * 40)
    )
    assert len(chunks) > 1
    assert (
        chunks[0].chunk_id
        == StructureAwareChunker(settings)
        .chunk(document("# Eligibility\n" + "official requirement details " * 40))[0]
        .chunk_id
    )
    assert "Title: Benefits" in chunks[0].content


def test_vector_validation_is_strict():
    validate_vectors([[0.0] * 4096], 1)
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_vectors([[0.0] * 10], 1)
    bad = [0.0] * 4096
    bad[3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_vectors([bad], 1)
