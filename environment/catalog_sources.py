"""Catalog ingestion sources for replay and live-data procurement pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

from environment.simulator import ProductCatalog


class CatalogSource(Protocol):
    """Minimal interface for loading a validated product catalog."""

    def load(self) -> ProductCatalog:
        """Return the latest catalog snapshot."""
        ...


@dataclass(frozen=True)
class CatalogLoadReport:
    """Small audit record for catalog ingestion jobs."""

    source: str
    num_items: int
    feature_columns: list[str]


@dataclass(frozen=True)
class CsvCatalogSource:
    """Load a catalog snapshot from a local or mounted CSV file."""

    path: str | Path

    def load(self) -> ProductCatalog:
        """Read and validate the configured CSV snapshot."""
        return ProductCatalog.from_csv(self.path)

    def describe(self) -> str:
        """Return a human-readable source identifier."""
        return str(self.path)


@dataclass(frozen=True)
class JsonCatalogSource:
    """Load a catalog snapshot from an HTTP JSON endpoint.

    The endpoint may return either a list of item records or an object with a
    top-level ``records`` list. Each record must follow the same numeric schema
    as ``data/products.csv``.
    """

    url: str
    timeout_seconds: float = 5.0
    headers: dict[str, str] | None = None

    def load(self) -> ProductCatalog:
        """Fetch, decode, and validate the configured JSON snapshot."""
        request = Request(self.url, headers=self.headers or {})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        records = payload.get("records", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("catalog JSON must be a list or an object with a records list")
        return ProductCatalog.from_records(records)

    def describe(self) -> str:
        """Return a human-readable source identifier."""
        return self.url


def load_catalog_with_report(source: CatalogSource) -> tuple[ProductCatalog, CatalogLoadReport]:
    """Load a catalog and return source metadata for audit logs."""
    catalog = source.load()
    describe = getattr(source, "describe", None)
    source_name = describe() if callable(describe) else source.__class__.__name__
    return catalog, CatalogLoadReport(
        source=source_name,
        num_items=len(catalog),
        feature_columns=list(catalog.feature_columns),
    )
