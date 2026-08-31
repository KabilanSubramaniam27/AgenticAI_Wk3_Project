import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib import robotparser

import httpx

from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.models import Method, SourceConfig
from seniorcare_ingestion.utils import digest

logger = logging.getLogger(__name__)
TRANSIENT = {408, 429, 500, 502, 503, 504}


@dataclass
class CollectedArtifact:
    source_id: str
    method: Method
    path: Path
    url: str
    status_code: int
    content_type: str
    retrieved_at: datetime
    sha256: str
    etag: str | None = None
    last_modified: str | None = None


class CollectionFailure(RuntimeError):
    pass


class BaseCollector(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    async def collect(self, source: SourceConfig, method: Method) -> CollectedArtifact: ...


class HttpCollector(BaseCollector):
    async def _request(
        self, url: str, headers: dict[str, str], params: dict | None = None
    ) -> httpx.Response:
        response = None
        for attempt in range(1, self.settings.http_max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=self.settings.http_timeout_seconds
                ) as client:
                    response = await client.get(url, headers=headers, params=params)
                if response.status_code not in TRANSIENT:
                    break
                retry_after = response.headers.get("Retry-After")
                await asyncio.sleep(
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(2 ** (attempt - 1), 8)
                )
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self.settings.http_max_retries:
                    raise CollectionFailure(str(exc)) from exc
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        assert response is not None
        if response.status_code >= 400:
            raise CollectionFailure(f"http_{response.status_code}")
        return response

    async def _robots_allowed(self, url: str) -> bool:
        parsed = httpx.URL(url)
        robots_url = str(parsed.copy_with(path="/robots.txt", query=None))
        try:
            async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
                response = await client.get(
                    robots_url, headers={"User-Agent": self.settings.scraper_user_agent}
                )
            if response.status_code >= 400:
                return True
            parser = robotparser.RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            return parser.can_fetch(self.settings.scraper_user_agent, url)
        except httpx.HTTPError:
            return True

    async def collect(self, source: SourceConfig, method: Method) -> CollectedArtifact:
        config = getattr(source.acquisition, method)
        if config is None:
            raise CollectionFailure(f"{method} is not configured")
        url = str(config.url)
        if method in {"html", "pdf"} and not await self._robots_allowed(url):
            raise CollectionFailure("robots_disallowed")
        headers = {"User-Agent": self.settings.scraper_user_agent, "Accept": "*/*"}
        initial_params = {
            **config.params,
            **({"limit": config.page_size} if config.page_size else {}),
        }
        response = await self._request(url, headers, initial_params)
        body = response.content
        if method == "api" and config.page_size:
            payload = response.json()
            records = list(payload.get("results", []))
            total = int(
                payload.get("count")
                or payload.get("meta", {}).get("results", {}).get("total")
                or len(records)
            )
            target = min(total, config.max_records) if config.max_records else total
            while len(records) < target:
                params = {
                    **config.params,
                    "limit": config.page_size,
                    config.offset_parameter: len(records),
                }
                page = await self._request(url, headers, params)
                page_records = page.json().get("results", [])
                if not page_records:
                    break
                records.extend(page_records[: target - len(records)])
                logger.info(
                    "source=%s stage=collect records=%s total=%s",
                    source.source_id,
                    len(records),
                    target,
                )
            body = json.dumps({"results": records, "count": total}, ensure_ascii=False).encode()
        retrieved = datetime.now(UTC)
        content_hash = digest(body.hex())
        suffix = {"html": ".html", "pdf": ".pdf", "api": ".json"}.get(
            method, Path(httpx.URL(url).path).suffix or ".bin"
        )
        folder = self.settings.raw_dir / source.source_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{retrieved:%Y%m%dT%H%M%SZ}_{content_hash[:12]}{suffix}"
        path.write_bytes(body)
        metadata = {
            "source_url": url,
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "retrieved_at": retrieved.isoformat(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "sha256": content_hash,
            "actual_method_used": method,
        }
        path.with_suffix(path.suffix + ".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return CollectedArtifact(
            source.source_id,
            method,
            path,
            url,
            response.status_code,
            metadata["content_type"],
            retrieved,
            content_hash,
            metadata["etag"],
            metadata["last_modified"],
        )


async def collect_with_fallback(
    source: SourceConfig, collector: BaseCollector
) -> tuple[CollectedArtifact, dict]:
    methods = [source.acquisition.preferred_method, *source.acquisition.fallback_methods]
    errors: list[str] = []
    for index, method in enumerate(methods):
        logger.info(
            "source=%s stage=collect method=%s fallback=%s event=started",
            source.source_id,
            method,
            index > 0,
        )
        try:
            artifact = await collector.collect(source, method)
            return artifact, {
                "preferredMethod": methods[0],
                "actualMethodUsed": method,
                "fallbackUsed": index > 0,
                "fallbackReason": errors[-1] if errors else None,
            }
        except CollectionFailure as exc:
            errors.append(str(exc))
            logger.warning(
                "source=%s stage=collect method=%s event=failed reason=%s",
                source.source_id,
                method,
                exc,
            )
    raise CollectionFailure("; ".join(errors))
