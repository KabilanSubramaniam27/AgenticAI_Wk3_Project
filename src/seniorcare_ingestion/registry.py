from urllib.parse import urlsplit

from seniorcare_ingestion.config import Settings
from seniorcare_ingestion.models import SourceConfig


class SourceRegistry:
    def __init__(self, settings: Settings):
        raw = settings.source_registry()
        self.allowed_domains = set(raw.get("allowed_domains", []))
        self.sources = [SourceConfig.model_validate(item) for item in raw.get("sources", [])]
        for source in self.sources:
            self._validate_domains(source)

    def _validate_domains(self, source: SourceConfig) -> None:
        for method in ("api", "download", "html", "pdf"):
            config = getattr(source.acquisition, method)
            hostname = urlsplit(str(config.url)).hostname if config else None
            allowed = hostname and any(
                hostname == domain or hostname.endswith(f".{domain}")
                for domain in self.allowed_domains
            )
            if config and not allowed:
                raise ValueError(f"Domain is not allowlisted for {source.source_id}: {config.url}")

    def enabled(self, source_id: str | None = None) -> list[SourceConfig]:
        return [
            source
            for source in self.sources
            if source.enabled and (not source_id or source.source_id == source_id)
        ]
