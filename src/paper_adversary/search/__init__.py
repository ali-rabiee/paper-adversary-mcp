"""Pluggable literature search: fan-out over scholarly APIs with caching, politeness and circuit breakers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable

import httpx2

from paper_adversary import __version__
from paper_adversary.search.backends import ArxivBackend, CrossrefBackend, OpenAlexBackend, SemanticScholarBackend
from paper_adversary.search.base import (
    Breaker,
    DiskCache,
    NotSupported,
    PaperRecord,
    ProviderUnavailable,
    RateGate,
    dumps,
    norm_title,
    records_from_json,
    records_to_json,
    strip_arxiv_version,
)

BackendFactory = Callable[[dict], object]

BACKENDS: dict[str, BackendFactory] = {
    "openalex": lambda opts: OpenAlexBackend(opts.get("contact_email")),
    "semantic_scholar": lambda opts: SemanticScholarBackend(opts.get("s2_api_key")),
    "arxiv": lambda opts: ArxivBackend(),
    "crossref": lambda opts: CrossrefBackend(opts.get("contact_email")),
}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Plug in another search provider (e.g. a local index or DBLP)."""
    BACKENDS[name] = factory


class Fetcher:
    """HTTP GET with a shared rate gate, bounded retries and a shared circuit breaker."""

    def __init__(self, client: httpx2.AsyncClient, gate: RateGate, breaker: Breaker, name: str,
                 block_statuses: tuple[int, ...] = (), max_retries: int = 2):
        self.client = client
        self.gate = gate
        self.breaker = breaker
        self.name = name
        self.block_statuses = block_statuses
        self.max_retries = max_retries

    async def _get(self, url: str, params: dict | None, headers: dict | None):
        if self.breaker.is_open(self.name):
            raise ProviderUnavailable(f"{self.name} is paused after an earlier block or failure")
        for attempt in range(self.max_retries + 1):
            await self.gate.wait()
            try:
                resp = await self.client.get(url, params=params, headers=headers)
            except (httpx2.TimeoutException, httpx2.RequestError) as exc:
                if attempt == self.max_retries:
                    raise ProviderUnavailable(f"{type(exc).__name__}: {exc}") from exc
                await asyncio.sleep(2 * 2**attempt)
                continue
            code = resp.status_code
            if code == 404:
                return None
            if code in self.block_statuses:
                self.breaker.trip(self.name, 7200, f"HTTP {code}")
                raise ProviderUnavailable(f"{self.name} answered HTTP {code}; pausing it for 2 h")
            if code == 429 or code >= 500:
                if attempt == self.max_retries:
                    self.breaker.trip(self.name, 600, f"HTTP {code}")
                    raise ProviderUnavailable(f"{self.name} kept answering HTTP {code}")
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 3 * 2**attempt
                await asyncio.sleep(min(delay, 60))
                continue
            if code >= 400:
                return None
            return resp
        return None

    async def get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        resp = await self._get(url, params, headers)
        if resp is None:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    async def get_text(self, url: str, params: dict | None = None, headers: dict | None = None):
        resp = await self._get(url, params, headers)
        return resp.text if resp is not None else None


# Which provider to trust for a field when several return the same paper. arXiv is the source of truth for
# its own abstracts (OpenAlex abstracts are sometimes attached to the wrong work); indexed venues beat "arXiv".
FIELD_PRIORITY = {
    "abstract": ("arxiv", "semantic_scholar", "crossref", "openalex"),
    "venue": ("semantic_scholar", "crossref", "openalex", "arxiv"),
}


def _rank(field: str, source: str) -> int:
    order = FIELD_PRIORITY.get(field, ())
    return order.index(source) if source in order else len(order)


def _weak_venue(venue: str | None) -> bool:
    return not venue or venue.lower().startswith("arxiv")


def merge_records(lists: list[list[PaperRecord]]) -> list[PaperRecord]:
    """Round-robin merge (keeps each provider's relevance order) with de-duplication."""
    merged: list[PaperRecord] = []
    alias: dict[str, int] = {}
    ranks: dict[int, dict[str, int]] = {}  # per merged record: rank of the source its abstract/venue came from

    def keys(r: PaperRecord) -> list[str]:
        out = []
        if r.doi:
            out.append("doi:" + r.doi.lower())
        if r.arxiv_id:
            out.append("arxiv:" + strip_arxiv_version(r.arxiv_id))
        if r.title:
            out.append("title:" + norm_title(r.title))
        return out

    depth = max((len(lst) for lst in lists), default=0)
    for rank in range(depth):
        for lst in lists:
            if rank >= len(lst):
                continue
            rec = lst[rank]
            idx = next((alias[k] for k in keys(rec) if k in alias), None)
            if idx is None:
                rec.sources = [rec.source]
                merged.append(rec)
                idx = len(merged) - 1
                ranks[idx] = {"abstract": _rank("abstract", rec.source), "venue": _rank("venue", rec.source)}
            else:
                base = merged[idx]
                for attr in ("doi", "arxiv_id", "url", "year", "citation_count"):
                    if getattr(base, attr) in (None, "") and getattr(rec, attr) not in (None, ""):
                        setattr(base, attr, getattr(rec, attr))
                r = ranks[idx]
                if rec.abstract and (not base.abstract or _rank("abstract", rec.source) < r["abstract"]):
                    base.abstract = rec.abstract
                    r["abstract"] = _rank("abstract", rec.source)
                if not _weak_venue(rec.venue) and (_weak_venue(base.venue) or _rank("venue", rec.source) < r["venue"]):
                    base.venue = rec.venue
                    r["venue"] = _rank("venue", rec.source)
                if not base.authors and rec.authors:
                    base.authors = rec.authors
                base.ids = {**rec.ids, **base.ids}
                if rec.source not in base.sources:
                    base.sources.append(rec.source)
            for k in keys(merged[idx]):
                alias.setdefault(k, idx)
    return merged


class LiteratureSearch:
    def __init__(self, providers: list[str], cache_root: Path, ttl_days: float = 14,
                 contact_email: str | None = None, transport: httpx2.AsyncBaseTransport | None = None):
        options = {"contact_email": contact_email or os.environ.get("PAPER_ADVERSARY_CONTACT_EMAIL") or None,
                   "s2_api_key": os.environ.get("S2_API_KEY") or None}
        unknown = [p for p in providers if p not in BACKENDS]
        if unknown:
            raise ValueError(f"unknown search providers {unknown}; known: {sorted(BACKENDS)}")
        self.backends = [BACKENDS[name](options) for name in providers]
        self.cache = DiskCache(cache_root / "search", ttl_days)
        self.breaker = Breaker(cache_root / "breaker")
        self.gates = {b.name: RateGate(cache_root / "ratelimit", b.name, b.min_interval_s) for b in self.backends}
        agent = f"paper-adversary-mcp/{__version__} (literature checks for pre-submission review)"
        if options["contact_email"]:
            agent += f"; mailto:{options['contact_email']}"
        self._client = httpx2.AsyncClient(timeout=httpx2.Timeout(30.0), follow_redirects=True,
                                          headers={"User-Agent": agent}, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LiteratureSearch":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _fetcher(self, backend) -> Fetcher:
        return Fetcher(self._client, self.gates[backend.name], self.breaker, backend.name,
                       getattr(backend, "block_statuses", ()))

    async def _call(self, backend, method: str, *args) -> tuple[list[PaperRecord], str]:
        key = dumps([method, *args])
        cached = self.cache.get(backend.name, key)
        if cached is not None:
            return records_from_json(cached), "cached"
        try:
            records = await getattr(backend, method)(self._fetcher(backend), *args)
        except NotSupported:
            return [], "not supported"
        except ProviderUnavailable as exc:
            return [], f"unavailable ({exc})"
        except Exception as exc:  # a broken provider must not take the agent down
            return [], f"error ({type(exc).__name__}: {exc})"
        self.cache.put(backend.name, key, records_to_json(records))
        return records, "ok"

    def _select(self, sources: list[str] | None):
        if not sources:
            return self.backends
        chosen = [b for b in self.backends if b.name in sources]
        return chosen or self.backends

    async def search(self, query: str, year_from: int | None = None, year_to: int | None = None,
                     limit: int = 10, sources: list[str] | None = None) -> dict:
        backends = self._select(sources)
        results = await asyncio.gather(*[self._call(b, "search", query, limit, year_from, year_to) for b in backends])
        merged = merge_records([recs for recs, _ in results])
        return {"records": merged[: max(limit, 1) * 2], "status": {b.name: st for b, (_, st) in zip(backends, results)}}

    async def lookup(self, identifier: str, sources: list[str] | None = None) -> dict:
        backends = self._select(sources)
        results = await asyncio.gather(*[self._call(b, "lookup", identifier) for b in backends])
        return {"records": merge_records([recs for recs, _ in results]),
                "status": {b.name: st for b, (_, st) in zip(backends, results)}}

    async def citing(self, identifier: str, limit: int = 15) -> dict:
        backends = [b for b in self.backends if b.name in ("openalex", "semantic_scholar")]
        results = await asyncio.gather(*[self._call(b, "citing", identifier, limit) for b in backends])
        return {"records": merge_records([recs for recs, _ in results])[: limit * 2],
                "status": {b.name: st for b, (_, st) in zip(backends, results)}}


__all__ = ["BACKENDS", "LiteratureSearch", "PaperRecord", "merge_records", "register_backend"]
