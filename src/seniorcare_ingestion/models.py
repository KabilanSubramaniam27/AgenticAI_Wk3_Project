from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

Category = Literal[
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
Method = Literal["api", "download", "html", "pdf", "manual_seed"]


class Geography(BaseModel):
    country: str = "US"
    state: str | None = None


class MethodConfig(BaseModel):
    url: HttpUrl
    format: str | None = None
    params: dict[str, str] = Field(default_factory=dict)
    page_size: int | None = Field(default=None, ge=1)
    max_records: int | None = Field(default=None, ge=1)
    offset_parameter: str = "offset"


class AcquisitionConfig(BaseModel):
    preferred_method: Method
    fallback_methods: list[Method] = Field(default_factory=list)
    api: MethodConfig | None = None
    download: MethodConfig | None = None
    html: MethodConfig | None = None
    pdf: MethodConfig | None = None


class SourceConfig(BaseModel):
    source_id: str
    source_name: str
    organization: str
    category: Category
    subcategory: str | None = None
    authority_level: str
    source_trust_tier: int = Field(ge=1, le=4)
    enabled: bool = True
    acquisition: AcquisitionConfig
    geography: Geography = Field(default_factory=Geography)
    tags: list[str] = Field(default_factory=list)


class NormalizedDocument(BaseModel):
    document_id: str
    source_id: str
    source_name: str
    source_url: str | None = None
    organization: str | None = None
    authority_level: str | None = None
    source_trust_tier: int
    category: Category
    subcategory: str | None = None
    title: str
    content: str
    country: str = "US"
    state: str | None = None
    county: str | None = None
    city: str | None = None
    zip_codes: list[str] = Field(default_factory=list)
    service_area: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    program_name: str | None = None
    document_type: str
    published_date: date | None = None
    effective_date: date | None = None
    last_updated_date: date | None = None
    retrieved_at: datetime
    page_number: int | None = None
    language: str = "en"
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content cannot be empty")
        return value


class ProviderRecord(BaseModel):
    provider_id: str
    npi: str
    first_name: str | None = None
    last_name: str | None = None
    credential: str | None = None
    specialty: str | None = None
    provider_type: str | None = None
    organization_name: str | None = None
    address_line_1: str | None = None
    address_line_2: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None
    phone: str | None = None
    medicare_participation: str | None = None
    source_id: str
    source_url: str | None = None
    last_updated_date: date | None = None
    retrieved_at: datetime


class CommunityResource(BaseModel):
    resource_id: str
    resource_name: str
    program_name: str | None = None
    category: Category
    subcategory: str | None = None
    organization: str
    description: str | None = None
    country: str = "US"
    state: str | None = None
    county: str | None = None
    city: str | None = None
    zip_codes: list[str] = Field(default_factory=list)
    service_area: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    minimum_age: int | None = None
    eligibility_summary: str | None = None
    required_documents: list[str] = Field(default_factory=list)
    application_method: str | None = None
    application_instructions: str | None = None
    phone: str | None = None
    website: str | None = None
    wheelchair_accessible: bool | None = None
    medical_transportation: bool | None = None
    home_delivery: bool | None = None
    cost_summary: str | None = None
    operating_hours: str | None = None
    source_id: str
    source_url: str | None = None
    authority_level: str | None = None
    source_trust_tier: int
    effective_date: date | None = None
    last_verified: datetime


class MedicationReference(BaseModel):
    medication_id: str
    product_ndc: str
    brand_name: str | None = None
    generic_name: str | None = None
    manufacturer_name: str | None = None
    dosage_form: str | None = None
    route: list[str] = Field(default_factory=list)
    product_type: str | None = None
    substance_names: list[str] = Field(default_factory=list)
    source_id: str
    source_url: str
    retrieved_at: datetime


class RagChunk(BaseModel):
    chunk_id: str
    document_id: str
    source_id: str
    source_name: str
    source_url: str | None = None
    organization: str | None = None
    category: Category
    subcategory: str | None = None
    title: str
    section_title: str | None = None
    content: str
    chunk_index: int
    chunk_count: int
    country: str = "US"
    state: str | None = None
    county: str | None = None
    city: str | None = None
    zip_codes: list[str] = Field(default_factory=list)
    service_area: list[str] = Field(default_factory=list)
    populations: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    program_name: str | None = None
    authority_level: str | None = None
    source_trust_tier: int
    effective_date: date | None = None
    published_date: date | None = None
    last_updated_date: date | None = None
    retrieved_at: datetime
    page_number: int | None = None
    document_type: str
    language: str = "en"
    content_hash: str
    embedding_model: str | None = None
    embedding_dimension: int | None = None

    def payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
