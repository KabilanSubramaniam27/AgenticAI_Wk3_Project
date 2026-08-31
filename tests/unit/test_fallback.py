from pathlib import Path

import pytest

from seniorcare_ingestion.collectors import (
    BaseCollector,
    CollectedArtifact,
    CollectionFailure,
    HttpCollector,
    collect_with_fallback,
)
from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.models import SourceConfig

SOURCE = SourceConfig.model_validate(
    {
        "source_id": "test",
        "source_name": "Test",
        "organization": "CMS",
        "category": "healthcare_access",
        "authority_level": "federal",
        "source_trust_tier": 1,
        "acquisition": {
            "preferred_method": "download",
            "fallback_methods": ["api"],
            "download": {"url": "https://cms.gov/file.zip"},
            "api": {"url": "https://data.cms.gov/api"},
        },
    }
)


class FakeCollector(BaseCollector):
    def __init__(self, failures: set[str]):
        super().__init__(Settings())
        self.failures = failures
        self.calls = []

    async def collect(self, source, method):
        self.calls.append(method)
        if method in self.failures:
            raise CollectionFailure(f"{method}_failed")
        from datetime import UTC, datetime

        return CollectedArtifact(
            source.source_id,
            method,
            Path("x"),
            "https://cms.gov",
            200,
            "application/json",
            datetime.now(UTC),
            "hash",
        )


@pytest.mark.asyncio
async def test_download_success_does_not_call_api():
    collector = FakeCollector(set())
    artifact, state = await collect_with_fallback(SOURCE, collector)
    assert (
        collector.calls == ["download"]
        and artifact.method == "download"
        and not state["fallbackUsed"]
    )


@pytest.mark.asyncio
async def test_download_failure_uses_api():
    collector = FakeCollector({"download"})
    artifact, state = await collect_with_fallback(SOURCE, collector)
    assert (
        collector.calls == ["download", "api"]
        and artifact.method == "api"
        and state["fallbackUsed"]
    )


@pytest.mark.asyncio
async def test_all_methods_fail():
    with pytest.raises(CollectionFailure):
        await collect_with_fallback(SOURCE, FakeCollector({"download", "api"}))


@pytest.mark.asyncio
async def test_api_pagination_honors_max_records(tmp_path, monkeypatch):
    import json

    import httpx

    configured = SourceConfig.model_validate(
        {
            "source_id": "paged",
            "source_name": "Paged",
            "organization": "FDA",
            "category": "medication_reference",
            "authority_level": "federal",
            "source_trust_tier": 1,
            "acquisition": {
                "preferred_method": "api",
                "api": {
                    "url": "https://api.fda.gov/drug/ndc.json",
                    "page_size": 2,
                    "max_records": 3,
                },
            },
        }
    )
    collector = HttpCollector(Settings(project_root=tmp_path, _env_file=None))
    calls = []

    async def request(url, headers, params=None):
        calls.append(params)
        offset = int((params or {}).get("offset", 0))
        records = [{"id": value} for value in range(offset, min(offset + 2, 4))]
        return httpx.Response(200, json={"results": records, "count": 4})

    monkeypatch.setattr(collector, "_request", request)
    artifact = await collector.collect(configured, "api")
    assert len(json.loads(artifact.path.read_text())["results"]) == 3
    assert calls == [{"limit": 2}, {"limit": 2, "offset": 2}]
