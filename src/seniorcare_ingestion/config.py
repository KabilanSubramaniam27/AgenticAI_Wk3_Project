from functools import lru_cache
from pathlib import Path
from typing import Annotated

import yaml  # type: ignore[import-untyped]
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore", populate_by_name=True
    )
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    nebius_api_key: str | None = None
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"
    nebius_embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-8B",
        validation_alias=AliasChoices("NEBIUS_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    embedding_dimension: int = 4096
    embedding_batch_size: int = 32
    actian_vectorai_url: str = Field(
        default="localhost:6574",
        validation_alias=AliasChoices("ACTIAN_VECTORAI_URL", "ACTIAN_VECTOR_DB_HOST"),
    )
    actian_vectorai_rest_url: str = "http://localhost:6573"
    actian_vectorai_collection: str = Field(
        default="seniorcare_knowledge",
        validation_alias=AliasChoices("ACTIAN_VECTORAI_COLLECTION", "ACTIAN_COLLECTION"),
    )
    actian_vectorai_access_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ACTIAN_VECTORAI_ACCESS_TOKEN", "ACTIAN_ACCESS_TOKEN"),
    )
    http_timeout_seconds: float = 30
    http_max_retries: int = 4
    scraper_user_agent: str = "SeniorCareResearchBot/0.1"
    chunk_target_tokens: int = 900
    chunk_overlap_tokens: int = 120
    min_chunk_tokens: int = 50
    min_chunk_characters: int = 200
    default_country: str = "US"
    default_state: str = "Virginia"
    supported_counties: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "Richmond City",
            "Henrico County",
            "Chesterfield County",
            "Hanover County",
        ]
    )
    vector_categories: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "healthcare_access",
            "transportation",
            "medication_reference",
            "discharge_support",
            "food_meals",
            "benefits_financial",
            "home_support",
            "caregiver_support",
            "social_wellbeing",
        ]
    )
    save_embeddings_debug: bool = False

    @field_validator("supported_counties", mode="before")
    @classmethod
    def split_counties(cls, value: object) -> object:
        return [part.strip() for part in value.split(",")] if isinstance(value, str) else value

    @field_validator("vector_categories", mode="before")
    @classmethod
    def split_vector_categories(cls, value: object) -> object:
        return (
            [part.strip() for part in value.split(",") if part.strip()]
            if isinstance(value, str)
            else value
        )

    @property
    def sources_path(self) -> Path:
        return self.project_root / "config/sources.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data/raw"

    @property
    def normalized_path(self) -> Path:
        return self.project_root / "data/normalized/documents.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.project_root / "data/processed/chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.project_root / "data/manifests/ingestion_manifest.json"

    def source_registry(self) -> dict:
        return yaml.safe_load(self.sources_path.read_text(encoding="utf-8"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
