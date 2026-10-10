"""Pluggable literature search: fan-out over scholarly APIs with caching, politeness and circuit breakers."""

from __future__ import annotations

import asyncio
import email.utils
import os
import random
import time
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
    "openalex": lambda opts: OpenAlexBackend(opts.get("contact_email"), opts.get("openalex_api_key")),
    "semantic_scholar": lambda opts: SemanticScholarBackend(opts.get("s2_api_key")),
    "arxiv": lambda opts: ArxivBackend(),
    "crossref": lambda opts: CrossrefBackend(opts.get("contact_email")),
}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Plug in another search provider (e.g. a local index or DBLP)."""
    BACKENDS[name] = factory


def retry_after_seconds(value: str | None, now: float | None = None) -> float | None:
    """A Retry-After header in either form: delta seconds, or an HTTP date."""
    value = (value or "").strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


def budget_reset_seconds(headers) -> float | None:
    """Seconds until a spent daily request budget resets, from rate-limit headers; None when the headers do not say
    the budget is spent. OpenAlex (since February 2026) reports x-ratelimit-remaining (credits) and
    x-ratelimit-remaining-usd, and x-ratelimit-reset in seconds. The pause is capped at 25 h."""
    def number(name: str) -> float | None:
        try:
            return float(headers.get(name))
        except (TypeError, ValueError):
            return None

    remaining, remaining_usd = number("x-ratelimit-remaining"), number("x-ratelimit-remaining-usd")
    if not ((remaining is not None and remaining <= 0) or (remaining_usd is not None and remaining_usd <= 0)):
        return None
    reset = number("x-ratelimit-reset")
    return min(max(reset if reset is not None else 3600.0, 60.0), 25 * 3600.0)


class Fetcher:
    """HTTP GET with a shared rate gate, bounded jittered retries and a shared circuit breaker.

    The retry policy is the backend's (_policy): keyless Semantic Scholar backs off longer before it trips; Semantic
    Scholar and OpenAlex trip a short half-open breaker (one probe after the pause); everything else trips for 600 s
    (2 h on a block status). A 429 whose rate-limit headers say the daily budget is spent pauses until its reset."""

    def __init__(self, client: httpx2.AsyncClient, gate: RateGate, breaker: Breaker, name: str,
                 block_statuses: tuple[int, ...] = (), max_retries: int = 2, backoff_s: float = 3.0,
                 limit_trip_s: float = 600, half_open: bool = False):
        self.client = client
        self.gate = gate
        self.breaker = breaker
        self.name = name
        self.block_statuses = block_statuses
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.limit_trip_s = limit_trip_s  # after repeated HTTP 429
        self.half_open = half_open

    async def _get(self, url: str, params: dict | None, headers: dict | None):
        admitted = self.breaker.admit(self.name)
        if admitted is None:
            raise ProviderUnavailable(f"{self.name} is {self.breaker.describe(self.name)}")
        retries = 0 if admitted == "probe" else self.max_retries  # a probe is one request
        for attempt in range(retries + 1):
            await self.gate.wait()
            try:
                resp = await self.client.get(url, params=params, headers=headers)
            except (httpx2.TimeoutException, httpx2.RequestError) as exc:
                if attempt == retries:
                    if admitted == "probe":
                        self.breaker.trip(self.name, self.limit_trip_s, type(exc).__name__, half_open=True)
                    raise ProviderUnavailable(f"{type(exc).__name__}: {exc}") from exc
                await asyncio.sleep(2 * 2**attempt)
                continue
            code = resp.status_code
            if code in self.block_statuses:
                self.breaker.trip(self.name, 7200, f"HTTP {code}")
                raise ProviderUnavailable(f"{self.name} answered HTTP {code}; pausing it for 2 h")
            if code == 429 and (spent := budget_reset_seconds(resp.headers)) is not None:
                # a provider's daily budget is spent (OpenAlex's credit headers): retrying cannot help before the reset
                self.breaker.trip(self.name, spent, "daily budget spent (HTTP 429)")
                raise ProviderUnavailable(f"{self.name}'s daily budget is spent; paused until its reset in "
                                          f"{spent / 3600:.1f} h")
            if code == 429 or code >= 500:
                if attempt == retries:
                    rate = code == 429
                    self.breaker.trip(self.name, self.limit_trip_s if rate else 600, f"HTTP {code}",
                                      half_open=rate and self.half_open)
                    raise ProviderUnavailable(f"{self.name} kept answering HTTP {code}")
                wait = retry_after_seconds(resp.headers.get("retry-after"))
                if wait is None:
                    wait = self.backoff_s * 2**attempt * random.uniform(0.75, 1.25)
                await asyncio.sleep(min(wait, 60))
                continue
            if admitted == "probe":
                self.breaker.close(self.name)
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


def _policy(backend) -> dict:
    """A backend's retry and breaker policy (attributes a backend may set; the defaults are the general ones)."""
    return {"max_retries": getattr(backend, "max_retries", 2), "backoff_s": getattr(backend, "backoff_s", 3.0),
            "limit_trip_s": getattr(backend, "limit_trip_s", 600), "half_open": getattr(backend, "half_open", False)}


class LiteratureSearch:
    def __init__(self, providers: list[str], cache_root: Path, ttl_days: float = 14,
                 contact_email: str | None = None, transport: httpx2.AsyncBaseTransport | None = None):
        options = {"contact_email": contact_email or os.environ.get("PAPER_ADVERSARY_CONTACT_EMAIL") or None,
                   "s2_api_key": os.environ.get("S2_API_KEY") or None,
                   "openalex_api_key": os.environ.get("OPENALEX_API_KEY") or None}
        unknown = [p for p in providers if p not in BACKENDS]
        if unknown:
            raise ValueError(f"unknown search providers {unknown}; known: {sorted(BACKENDS)}")
        self.options = options
        self.cache_root = cache_root
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
                       getattr(backend, "block_statuses", ()), **_policy(backend))

    def fetcher(self, name: str, min_interval_s: float | None = None) -> Fetcher:
        """A Fetcher for one provider with that provider's policy, also for one this search does not query
        (full-text resolution asks OpenAlex and Semantic Scholar for open-access links)."""
        backend = next((b for b in self.backends if b.name == name), None) or BACKENDS[name](self.options)
        gate = self.gates.get(name) or RateGate(self.cache_root / "ratelimit", name,
                                                backend.min_interval_s if min_interval_s is None else min_interval_s)
        return Fetcher(self._client, gate, self.breaker, name, getattr(backend, "block_statuses", ()),
                       **_policy(backend))

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


LITERATURE_TOOLS = ("search_literature", "lookup_paper", "find_citing_papers")
DEGRADED_SHARE = 0.25  # a provider unavailable for this share of the calls that asked it is reported as degraded


def provider_health(rows: list[dict]) -> dict[str, dict]:
    """Per provider, from search_log rows of the literature tools (their 'status' per provider): the calls that
    asked it, how many it answered (fresh or cached), how many found it unavailable (blocked, rate limited, paused
    or failing), and when the first and last of those were. 'not supported' calls are not counted."""
    out: dict[str, dict] = {}
    for row in rows:
        if row.get("tool") not in LITERATURE_TOOLS or not isinstance(row.get("status"), dict):
            continue
        for name, status in row["status"].items():
            status = str(status)
            if status == "not supported":
                continue
            c = out.setdefault(name, {"calls": 0, "answered": 0, "unavailable": 0, "first_unavailable": None,
                                      "last_unavailable": None})
            c["calls"] += 1
            if status in ("ok", "cached"):
                c["answered"] += 1
                continue
            c["unavailable"] += 1
            at = row.get("at")
            if at:
                c["first_unavailable"] = min(x for x in (c["first_unavailable"], at) if x)
                c["last_unavailable"] = max(x for x in (c["last_unavailable"], at) if x)
    return out


__all__ = ["BACKENDS", "DEGRADED_SHARE", "LiteratureSearch", "PaperRecord", "merge_records", "provider_health",
           "register_backend", "retry_after_seconds"]
