import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from seniorcare_ingestion.config import get_settings
from seniorcare_ingestion.embeddings import NebiusEmbeddingProvider
from seniorcare_ingestion.pipeline import IngestionPipeline
from seniorcare_ingestion.registry import SourceRegistry
from seniorcare_ingestion.utils import configure_logging
from seniorcare_ingestion.vectorstore import ActianVectorStore

app = typer.Typer(help="SeniorCare Connect public knowledge ingestion")
sources_app = typer.Typer(help="Inspect configured sources")
app.add_typer(sources_app, name="sources")
console = Console()


def local_pipeline() -> IngestionPipeline:
    return IngestionPipeline(get_settings())


def live_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(
        settings, NebiusEmbeddingProvider(settings), ActianVectorStore(settings)
    )


def output(value: object) -> None:
    console.print_json(json.dumps(value, default=str))


@sources_app.command("list")
def sources_list() -> None:
    table = Table("ID", "Category", "Preferred", "Enabled")
    for source in SourceRegistry(get_settings()).sources:
        table.add_row(
            source.source_id,
            source.category,
            source.acquisition.preferred_method,
            str(source.enabled),
        )
    console.print(table)


@sources_app.command("stale")
def sources_stale(days: int = 30) -> None:
    pipeline = local_pipeline()
    cutoff = datetime.now(UTC) - timedelta(days=days)
    stale = []
    for source in pipeline.registry.sources:
        value = pipeline.manifest.data["sources"].get(source.source_id, {}).get("retrievedAt")
        if not value or datetime.fromisoformat(value) < cutoff:
            stale.append(source.source_id)
    output({"days": days, "stale_sources": stale})


@app.command()
def collect(source: str | None = None, force: bool = False) -> None:
    output([str(item.path) for item in asyncio.run(local_pipeline().collect(source, force))])


@app.command()
def normalize(source: str | None = None) -> None:
    output({"documents": len(local_pipeline().normalize(source))})


@app.command()
def clean() -> None:
    output({"documents": len(local_pipeline().normalize())})


@app.command()
def deduplicate() -> None:
    output({"documents": len(local_pipeline().normalize())})


@app.command()
def chunk() -> None:
    output({"chunks": len(local_pipeline().chunk())})


@app.command()
def embed(resume: bool = True, dry_run: bool = False) -> None:
    output(asyncio.run(live_pipeline().embed_and_index(resume, dry_run)))


@app.command()
def index(resume: bool = True, dry_run: bool = False) -> None:
    output(asyncio.run(live_pipeline().embed_and_index(resume, dry_run)))


@app.command()
def validate() -> None:
    output(local_pipeline().validate())


@app.command()
def stats() -> None:
    output(local_pipeline().stats())


@app.command()
def ingest(
    source: str | None = None, resume: bool = False, force: bool = False, dry_run: bool = False
) -> None:
    pipeline = local_pipeline() if dry_run else live_pipeline()
    output(asyncio.run(pipeline.ingest(source, resume, force, dry_run)))


@app.command()
def health(check_nebius: bool = False) -> None:
    settings = get_settings()
    report: dict[str, dict[str, Any]] = {
        "nebius": {
            "configured": bool(settings.nebius_api_key),
            "model": settings.nebius_embedding_model,
            "dimension": settings.embedding_dimension,
        },
        "actian": {
            "url": settings.actian_vectorai_url,
            "collection": settings.actian_vectorai_collection,
        },
    }
    try:
        report["actian"]["status"] = (
            "OK" if ActianVectorStore(settings).health_check() else "FAILED"
        )
    except Exception as exc:
        report["actian"].update(status="UNAVAILABLE", error=str(exc))
    if check_nebius and settings.nebius_api_key:
        try:
            asyncio.run(NebiusEmbeddingProvider(settings).embed_query("health check"))
            report["nebius"]["status"] = "OK"
        except Exception as exc:
            report["nebius"].update(status="FAILED", error=str(exc))
    else:
        report["nebius"]["status"] = "CONFIGURED" if settings.nebius_api_key else "NOT_CONFIGURED"
    output(report)


@app.command()
def search(
    query: str,
    top_k: int = 10,
    category: str | None = None,
    state: str | None = None,
    county: str | None = None,
) -> None:
    pipeline = live_pipeline()
    assert pipeline.embedder is not None and pipeline.store is not None
    vector = asyncio.run(pipeline.embedder.embed_query(query))
    filters = {
        k: v for k, v in {"category": category, "state": state, "county": county}.items() if v
    }
    output(
        pipeline.store.search(pipeline.settings.actian_vectorai_collection, vector, top_k, filters)
    )


def main() -> None:
    configure_logging()
    app()


if __name__ == "__main__":
    main()
