from datetime import UTC, datetime

from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.models import SourceConfig
from seniorcare_ingestion.pipeline import IngestionPipeline


def source(source_id: str, category: str = "medication_reference") -> SourceConfig:
    return SourceConfig.model_validate(
        {
            "source_id": source_id,
            "source_name": "Official Source",
            "organization": "Agency",
            "category": category,
            "authority_level": "federal",
            "source_trust_tier": 1,
            "acquisition": {
                "preferred_method": "api",
                "api": {"url": "https://api.fda.gov/drug/ndc.json"},
            },
        }
    )


def test_openfda_normalization_stays_structured(tmp_path):
    pipeline = object.__new__(IngestionPipeline)
    pipeline.settings = Settings(project_root=tmp_path, _env_file=None)
    pipeline._write_medications(
        [
            {
                "product_ndc": "0001-0001",
                "brand_name": "Example",
                "route": ["ORAL"],
                "active_ingredients": [{"name": "Ingredient"}],
            }
        ],
        source("openfda_ndc"),
        datetime.now(UTC),
        "https://api.fda.gov/drug/ndc.json",
    )
    output = (tmp_path / "data/normalized/medications.jsonl").read_text()
    assert '"product_ndc": "0001-0001"' in output
    assert '"substance_names": ["Ingredient"]' in output


def test_community_resource_extracts_geography_and_accessibility():
    config = source("grtc_care_paratransit", "transportation")
    resource = IngestionPipeline._community_resource(
        config,
        "CARE",
        "Eligibility for Henrico County wheelchair riders. Call (804) 555-1212. Riders 80 years or older may apply.",
        "https://example.gov/care",
        datetime.now(UTC),
    )
    assert resource.wheelchair_accessible is True
    assert resource.minimum_age == 80
    assert "Henrico County" in resource.service_area


def test_legacy_environment_names_are_supported(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "legacy-model")
    monkeypatch.setenv("ACTIAN_VECTOR_DB_HOST", "legacy:6574")
    settings = Settings(_env_file=None)
    assert settings.nebius_embedding_model == "legacy-model"
    assert settings.actian_vectorai_url == "legacy:6574"
