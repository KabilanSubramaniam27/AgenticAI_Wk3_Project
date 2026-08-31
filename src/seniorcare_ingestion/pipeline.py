import json
import re
from datetime import datetime
from pathlib import Path

from seniorcare_ingestion.collectors import CollectedArtifact, HttpCollector, collect_with_fallback
from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.embeddings import EmbeddingProvider
from seniorcare_ingestion.manifest import Manifest
from seniorcare_ingestion.models import (
    CommunityResource,
    MedicationReference,
    NormalizedDocument,
    ProviderRecord,
    RagChunk,
    SourceConfig,
)
from seniorcare_ingestion.parsers import parse_html, parse_pdf, parse_structured
from seniorcare_ingestion.processing import StructureAwareChunker, deduplicate
from seniorcare_ingestion.registry import SourceRegistry
from seniorcare_ingestion.utils import clean_text, digest, read_jsonl, stable_id, write_jsonl
from seniorcare_ingestion.vectorstore import VectorStore


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
    ):
        self.settings = settings
        self.registry = SourceRegistry(settings)
        self.manifest = Manifest(settings)
        self.embedder = embedder
        self.store = store

    async def collect(
        self, source_id: str | None = None, force: bool = False
    ) -> list[CollectedArtifact]:
        artifacts = []
        selected = self.registry.enabled(source_id)
        if source_id and not selected:
            raise ValueError(f"Unknown or disabled source: {source_id}")
        for source in selected:
            previous = self.manifest.data["sources"].get(source.source_id, {})
            if (
                not force
                and previous.get("status") == "collected"
                and Path(previous.get("rawPath", "")).exists()
            ):
                continue
            try:
                artifact, strategy = await collect_with_fallback(
                    source, HttpCollector(self.settings)
                )
                artifacts.append(artifact)
                self.manifest.data["sources"][source.source_id] = {
                    "status": "collected",
                    **strategy,
                    "retrievedAt": artifact.retrieved_at.isoformat(),
                    "sha256": artifact.sha256,
                    "rawPath": str(artifact.path),
                }
            except Exception as exc:
                self.manifest.data["sources"][source.source_id] = {
                    "status": "failed",
                    "errors": [str(exc)],
                }
            self.manifest.save()
        return artifacts

    def _source(self, source_id: str) -> SourceConfig:
        return next(source for source in self.registry.sources if source.source_id == source_id)

    def normalize(self, source_id: str | None = None) -> list[NormalizedDocument]:
        documents: list[NormalizedDocument] = []
        resource_path = self.settings.project_root / "data/normalized/resources.jsonl"
        resources = [
            CommunityResource.model_validate(row)
            for row in read_jsonl(resource_path)
            if not source_id or row["source_id"] != source_id
        ]
        for sid, state in self.manifest.data["sources"].items():
            if (
                (source_id and sid != source_id)
                or state.get("status") == "failed"
                or not state.get("rawPath")
            ):
                continue
            source = self._source(sid)
            path = Path(state["rawPath"])
            retrieved = datetime.fromisoformat(state["retrievedAt"])
            method = state["actualMethodUsed"]
            url = str(getattr(source.acquisition, method).url)
            parsed: list[tuple[str, str, int | None]]
            if method == "pdf":
                parsed = [(title, text, page) for title, text, page in parse_pdf(path)]
            elif method == "html":
                title, text = parse_html(path.read_bytes())
                parsed = [(title, text, None)]
            else:
                rows = parse_structured(path)
                parsed = []
                if source.source_id == "cms_doctors":
                    self._write_providers(rows, source, retrieved, url)
                    continue
                if source.source_id.startswith("openfda_"):
                    self._write_medications(rows, source, retrieved, url)
                    continue
                parsed = [
                    (
                        source.source_name,
                        "\n".join(
                            f"{key}: {value}"
                            for key, value in row.items()
                            if value not in (None, "")
                        ),
                        None,
                    )
                    for row in rows
                ]
            for title, raw, page in parsed:
                content = clean_text(raw)
                if not content:
                    continue
                localities = [
                    tag for tag in source.tags if tag == "Richmond City" or tag.endswith(" County")
                ]
                counties = [value for value in localities if value.endswith(" County")]
                zip_codes = sorted(set(re.findall(r"\b(?:22|23|24)\d{3}(?:-\d{4})?\b", content)))
                populations = list(
                    dict.fromkeys(
                        [
                            "senior",
                            *[tag for tag in source.tags if tag in {"caregiver", "disability"}],
                        ]
                    )
                )
                document = NormalizedDocument(
                    document_id=stable_id(source.source_id, url, title, page or ""),
                    source_id=source.source_id,
                    source_name=source.source_name,
                    source_url=url,
                    organization=source.organization,
                    authority_level=source.authority_level,
                    source_trust_tier=source.source_trust_tier,
                    category=source.category,
                    subcategory=source.subcategory,
                    title=title,
                    content=content,
                    country=source.geography.country,
                    state=source.geography.state,
                    county=counties[0] if len(counties) == 1 else None,
                    city="Richmond" if localities == ["Richmond City"] else None,
                    zip_codes=zip_codes,
                    service_area=localities
                    or ([source.geography.state] if source.geography.state else []),
                    populations=populations,
                    tags=source.tags,
                    program_name=source.source_name,
                    document_type=method,
                    retrieved_at=retrieved,
                    page_number=page,
                    content_hash=digest(content.lower()),
                )
                documents.append(document)
                if method == "html":
                    resources.append(
                        self._community_resource(source, title, content, url, retrieved)
                    )
        existing = [
            NormalizedDocument.model_validate(row)
            for row in read_jsonl(self.settings.normalized_path)
            if not source_id or row["source_id"] != source_id
        ]
        combined, stats = deduplicate(documents + existing)
        write_jsonl(
            self.settings.normalized_path, [item.model_dump(mode="json") for item in combined]
        )
        write_jsonl(
            resource_path,
            [
                item.model_dump(mode="json")
                for item in {resource.resource_id: resource for resource in resources}.values()
            ],
        )
        self.manifest.data["deduplication"] = stats
        self.manifest.save()
        return combined

    def _write_providers(
        self, rows: list[dict], source: SourceConfig, retrieved: datetime, url: str
    ) -> None:
        providers = []
        for row in rows:
            lower = {str(key).lower().replace(" ", "_"): value for key, value in row.items()}
            npi = str(lower.get("npi") or lower.get("npi_number") or "").strip()
            state = str(lower.get("state") or lower.get("adr_ln_1_state") or "").upper()
            if not npi or state not in {"VA", "VIRGINIA"}:
                continue
            provider_id = stable_id(
                npi,
                lower.get("ind_enrl_id", ""),
                lower.get("org_pac_id", ""),
                lower.get("adrs_id", ""),
            )
            providers.append(
                ProviderRecord(
                    provider_id=provider_id,
                    npi=npi,
                    first_name=lower.get("provider_first_name")
                    or lower.get("first_name")
                    or lower.get("frst_nm"),
                    last_name=lower.get("provider_last_name")
                    or lower.get("last_name")
                    or lower.get("lst_nm"),
                    credential=lower.get("credential") or lower.get("cred"),
                    specialty=lower.get("specialty") or lower.get("pri_spec"),
                    provider_type=lower.get("provider_type"),
                    organization_name=lower.get("organization_name")
                    or lower.get("org_nm")
                    or lower.get("facility_name"),
                    address_line_1=lower.get("address_line_1") or lower.get("adr_ln_1"),
                    address_line_2=lower.get("address_line_2") or lower.get("adr_ln_2"),
                    city=lower.get("citytown") or lower.get("city") or lower.get("cty"),
                    state="Virginia",
                    zip_code=lower.get("zip_code") or lower.get("zip_cd"),
                    phone=lower.get("telephone_number")
                    or lower.get("phone")
                    or lower.get("phn_numbr"),
                    medicare_participation=lower.get("medicare_participation")
                    or lower.get("ind_assgn"),
                    source_id=source.source_id,
                    source_url=url,
                    retrieved_at=retrieved,
                )
            )
        write_jsonl(
            self.settings.project_root / "data/normalized/providers.jsonl",
            [item.model_dump(mode="json") for item in providers],
        )

    def _write_medications(
        self, rows: list[dict], source: SourceConfig, retrieved: datetime, url: str
    ) -> None:
        path = self.settings.project_root / "data/normalized/medications.jsonl"
        medications = [
            MedicationReference.model_validate(row)
            for row in read_jsonl(path)
            if row["source_id"] != source.source_id
        ]
        for row in rows:
            ndc = str(row.get("product_ndc") or "").strip()
            if not ndc:
                continue
            substances = [
                str(item["name"])
                for item in row.get("active_ingredients", [])
                if isinstance(item, dict) and item.get("name")
            ]
            route = row.get("route") or []
            medications.append(
                MedicationReference(
                    medication_id=stable_id(ndc),
                    product_ndc=ndc,
                    brand_name=row.get("brand_name"),
                    generic_name=row.get("generic_name"),
                    manufacturer_name=row.get("labeler_name"),
                    dosage_form=row.get("dosage_form"),
                    route=route if isinstance(route, list) else [str(route)],
                    product_type=row.get("product_type"),
                    substance_names=substances,
                    source_id=source.source_id,
                    source_url=url,
                    retrieved_at=retrieved,
                )
            )
        unique = {item.medication_id: item for item in medications}
        write_jsonl(path, [item.model_dump(mode="json") for item in unique.values()])

    @staticmethod
    def _community_resource(
        source: SourceConfig, title: str, content: str, url: str, retrieved: datetime
    ) -> CommunityResource:
        lower = content.lower()
        paragraphs = content.split("\n\n")
        phone = re.search(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}", content)
        zip_codes = sorted(set(re.findall(r"\b(?:22|23|24)\d{3}\b", content)))
        localities = [
            name
            for name in ("Richmond City", "Henrico County", "Chesterfield County", "Hanover County")
            if name.lower().split()[0] in lower
        ]
        eligibility = next((value for value in paragraphs if "eligib" in value.lower()), None)
        application = next(
            (
                value
                for value in paragraphs
                if any(
                    term in value.lower() for term in ("how to apply", "application", "reservation")
                )
            ),
            None,
        )
        return CommunityResource(
            resource_id=stable_id(source.source_id, url),
            resource_name=title,
            program_name=source.source_name,
            category=source.category,
            subcategory=source.subcategory,
            organization=source.organization,
            description=content[:1500],
            country=source.geography.country,
            state=source.geography.state,
            zip_codes=zip_codes,
            service_area=localities or ([source.geography.state] if source.geography.state else []),
            populations=list(
                dict.fromkeys(
                    ["senior", *[tag for tag in source.tags if tag in {"disability", "caregiver"}]]
                )
            ),
            minimum_age=80
            if source.source_id == "grtc_care_paratransit" and "80 years" in lower
            else None,
            eligibility_summary=eligibility[:2000] if eligibility else None,
            application_method="See official source" if application else None,
            application_instructions=application[:2000] if application else None,
            phone=phone.group(0) if phone else None,
            website=url,
            wheelchair_accessible=True if "wheelchair" in lower else None,
            medical_transportation=True
            if any(
                term in lower
                for term in ("medical transportation", "medicaid transportation", "nemt")
            )
            else None,
            home_delivery=True
            if "home delivered meal" in lower or "home-delivered meal" in lower
            else None,
            source_id=source.source_id,
            source_url=url,
            authority_level=source.authority_level,
            source_trust_tier=source.source_trust_tier,
            last_verified=retrieved,
        )

    def chunk(self) -> list[RagChunk]:
        documents = [
            NormalizedDocument.model_validate(row)
            for row in read_jsonl(self.settings.normalized_path)
            if row["category"] in self.settings.vector_categories
        ]
        chunker = StructureAwareChunker(self.settings)
        chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk IDs")
        write_jsonl(self.settings.chunks_path, [chunk.model_dump(mode="json") for chunk in chunks])
        return chunks

    @staticmethod
    def _index_hash(chunk: RagChunk) -> str:
        payload = chunk.model_dump(mode="json", exclude={"embedding_model", "embedding_dimension"})
        return digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    async def embed_and_index(self, resume: bool = False, dry_run: bool = False) -> dict:
        if not self.embedder or not self.store:
            raise RuntimeError("Nebius embedder and Actian store are required")
        chunks = [RagChunk.model_validate(row) for row in read_jsonl(self.settings.chunks_path)]
        known = self.manifest.data["chunks"]
        current_ids = {chunk.chunk_id for chunk in chunks}
        obsolete_ids = sorted(set(known) - current_ids)
        pending = [
            chunk
            for chunk in chunks
            if not resume or known.get(chunk.chunk_id) != self._index_hash(chunk)
        ]
        if dry_run:
            return {"pending": len(pending), "indexed": 0, "deleted": len(obsolete_ids)}
        if obsolete_ids:
            self.store.delete(self.settings.actian_vectorai_collection, obsolete_ids)
            for chunk_id in obsolete_ids:
                known.pop(chunk_id, None)
            self.manifest.save()
        indexed = 0
        size = self.settings.embedding_batch_size
        for start in range(0, len(pending), size):
            batch = pending[start : start + size]
            vectors = await self.embedder.embed_documents([chunk.content for chunk in batch])
            for chunk in batch:
                chunk.embedding_model = self.settings.nebius_embedding_model
                chunk.embedding_dimension = self.settings.embedding_dimension
            self.store.upsert(
                self.settings.actian_vectorai_collection,
                [
                    {"id": chunk.chunk_id, "values": vector, "metadata": chunk.payload()}
                    for chunk, vector in zip(batch, vectors, strict=True)
                ],
            )
            for chunk in batch:
                known[chunk.chunk_id] = self._index_hash(chunk)
            indexed += len(batch)
            self.manifest.save()
        return {"pending": len(pending), "indexed": indexed, "deleted": len(obsolete_ids)}

    async def ingest(
        self,
        source_id: str | None = None,
        resume: bool = False,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict:
        run_id = self.manifest.begin()
        artifacts = await self.collect(source_id, force)
        documents = self.normalize(source_id)
        chunks = self.chunk()
        index = (
            {"pending": len(chunks), "indexed": 0}
            if dry_run
            else await self.embed_and_index(resume)
        )
        report = {
            "run_id": run_id,
            "sources_attempted": len(self.registry.enabled(source_id)),
            "raw_artifacts": len(artifacts),
            "documents": len(documents),
            "chunks": len(chunks),
            **index,
        }
        self.manifest.data["runs"][run_id]["report"] = report
        self.manifest.finish(run_id)
        return report

    def validate(self) -> dict:
        documents = read_jsonl(self.settings.normalized_path)
        chunks = read_jsonl(self.settings.chunks_path)
        errors = []
        if len({d["document_id"] for d in documents}) != len(documents):
            errors.append("duplicate document IDs")
        if len({c["chunk_id"] for c in chunks}) != len(chunks):
            errors.append("duplicate chunk IDs")
        for chunk in chunks:
            for field in ("content", "source_id", "category", "retrieved_at", "content_hash"):
                if not chunk.get(field):
                    errors.append(f"{chunk.get('chunk_id')}: missing {field}")
            if chunk.get("category") not in self.settings.vector_categories:
                errors.append(f"{chunk.get('chunk_id')}: category is outside vector corpus")
        return {
            "valid": not errors,
            "documents": len(documents),
            "resources": len(
                read_jsonl(self.settings.project_root / "data/normalized/resources.jsonl")
            ),
            "providers": len(
                read_jsonl(self.settings.project_root / "data/normalized/providers.jsonl")
            ),
            "medications": len(
                read_jsonl(self.settings.project_root / "data/normalized/medications.jsonl")
            ),
            "chunks": len(chunks),
            "errors": errors,
        }

    def stats(self) -> dict:
        chunks = read_jsonl(self.settings.chunks_path)
        by_category: dict[str, int] = {}
        for chunk in chunks:
            by_category[chunk["category"]] = by_category.get(chunk["category"], 0) + 1
        counts = sorted(len(chunk["content"].split()) for chunk in chunks)
        return {
            "documents": len(read_jsonl(self.settings.normalized_path)),
            "resources": len(
                read_jsonl(self.settings.project_root / "data/normalized/resources.jsonl")
            ),
            "providers": len(
                read_jsonl(self.settings.project_root / "data/normalized/providers.jsonl")
            ),
            "medications": len(
                read_jsonl(self.settings.project_root / "data/normalized/medications.jsonl")
            ),
            "chunks": len(chunks),
            "chunks_by_category": by_category,
            "average_tokens_approx": round(sum(counts) / len(counts), 1) if counts else 0,
            "median_tokens_approx": counts[len(counts) // 2] if counts else 0,
            "p95_tokens_approx": counts[min(len(counts) - 1, int(len(counts) * 0.95))]
            if counts
            else 0,
        }
