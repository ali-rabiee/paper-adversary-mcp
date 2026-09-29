"""Common pieces of the pluggable literature-search subsystem.

A backend is any object with `name`, `min_interval_s` and async `search`,
`lookup` and (optionally) `citing` methods returning PaperRecord lists. Add a
provider by implementing SearchBackend and registering it in search/__init__.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from paper_adversary.util import FileLock, atomic_write_json, read_json


class ProviderUnavailable(RuntimeError):
    """The backend is blocked, rate limited beyond retry, or down. Skip it for a while."""


class NotSupported(RuntimeError):
    pass


@dataclass
class PaperRecord:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    source: str = ""
    ids: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def key(self) -> str:
        if self.doi:
            return "doi:" + self.doi.lower()
        if self.arxiv_id:
            return "arxiv:" + strip_arxiv_version(self.arxiv_id)
        return "title:" + norm_title(self.title)

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self, index: int | None = None, abstract_chars: int = 450) -> str:
        authors = ", ".join(self.authors[:4]) + (" et al." if len(self.authors) > 4 else "")
        ids = []
        if self.doi:
            ids.append(f"DOI {self.doi}")
        if self.arxiv_id:
            ids.append(f"arXiv {self.arxiv_id}")
        head = f"[{index}] " if index is not None else ""
        lines = [f"{head}{self.title} ({self.year or 'n.d.'})",
                 f"    {authors or 'authors unknown'}" + (f" — {self.venue}" if self.venue else "")]
        meta = "; ".join(ids + ([f"cited by {self.citation_count}"] if self.citation_count is not None else []))
        if meta:
            lines.append(f"    {meta}")
        if self.url:
            lines.append(f"    {self.url}")
        if self.abstract:
            abstract = re.sub(r"\s+", " ", self.abstract).strip()
            lines.append("    Abstract: " + (abstract[:abstract_chars] + ("…" if len(abstract) > abstract_chars else "")))
        if self.sources:
            lines.append(f"    (found via {', '.join(self.sources)})")
        return "\n".join(lines)


class SearchBackend(Protocol):
    name: str
    min_interval_s: float

    async def search(self, http, query: str, limit: int, year_from: int | None, year_to: int | None) -> list[PaperRecord]: ...

    async def lookup(self, http, identifier: str) -> list[PaperRecord]: ...

    async def citing(self, http, identifier: str, limit: int) -> list[PaperRecord]: ...


# ---------------------------------------------------------------- identifiers

_DOI = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(v\d+)?(?![\d])")
_ARXIV_OLD = re.compile(r"\b([a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?\b")


def strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip().lower().removeprefix("arxiv:"))


def parse_identifier(text: str) -> tuple[str, str]:
    """Classify an identifier as ('doi'|'arxiv'|'title', value)."""
    value = text.strip()
    low = value.lower()
    if low.startswith("doi:"):
        return "doi", value[4:].strip()
    if low.startswith("arxiv:"):
        return "arxiv", strip_arxiv_version(value[6:])
    m = _DOI.search(value)
    if m and ("doi.org" in low or low.startswith("10.")):
        doi = m.group(1).rstrip(".,;)")
        arx = re.match(r"10\.48550/arxiv\.(.+)", doi, re.I)
        if arx:
            return "arxiv", strip_arxiv_version(arx.group(1))
        return "doi", doi
    if "arxiv.org" in low:
        m = _ARXIV_NEW.search(value) or _ARXIV_OLD.search(value)
        if m:
            return "arxiv", strip_arxiv_version(m.group(1))
    m = _ARXIV_NEW.fullmatch(value) or _ARXIV_OLD.fullmatch(value)
    if m:
        return "arxiv", strip_arxiv_version(m.group(1))
    return "title", value


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


# ---------------------------------------------------------------- cross-process politeness


class RateGate:
    """Minimum spacing between requests to one provider, shared by every process on the machine."""

    def __init__(self, directory: Path, name: str, min_interval_s: float):
        self.lock = directory / f"{name}.lock"
        self.stamp = directory / f"{name}.last"
        self.min_interval_s = min_interval_s
        directory.mkdir(parents=True, exist_ok=True)

    def _wait_blocking(self) -> None:
        with FileLock(self.lock):
            try:
                last = float(self.stamp.read_text())
            except (FileNotFoundError, ValueError):
                last = 0.0
            delay = last + self.min_interval_s - time.time()
            if delay > 0:
                time.sleep(delay)
            self.stamp.write_text(f"{time.time():.3f}")

    async def wait(self) -> None:
        if self.min_interval_s > 0:
            await asyncio.get_running_loop().run_in_executor(None, self._wait_blocking)


class Breaker:
    """Shared 'provider is blocked until T' marker, so a block seen by one agent spares the others."""

    def __init__(self, directory: Path):
        self.dir = directory
        directory.mkdir(parents=True, exist_ok=True)

    def open_until(self, name: str) -> float:
        data = read_json(self.dir / f"{name}.json", {}) or {}
        return float(data.get("until", 0))

    def is_open(self, name: str) -> bool:
        return self.open_until(name) > time.time()

    def trip(self, name: str, seconds: float, reason: str) -> None:
        atomic_write_json(self.dir / f"{name}.json", {"until": time.time() + seconds, "reason": reason})


class DiskCache:
    def __init__(self, directory: Path, ttl_days: float):
        self.dir = directory
        self.ttl = ttl_days * 86400

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.dir / namespace / digest[:2] / f"{digest}.json"

    def get(self, namespace: str, key: str):
        path = self._path(namespace, key)
        data = read_json(path)
        if not data or time.time() - data.get("at", 0) > self.ttl:
            return None
        return data.get("value")

    def put(self, namespace: str, key: str, value) -> None:
        atomic_write_json(self._path(namespace, key), {"at": time.time(), "key": key, "value": value})


def records_to_json(records: list[PaperRecord]) -> list[dict]:
    return [r.to_dict() for r in records]


def records_from_json(items: list[dict]) -> list[PaperRecord]:
    return [PaperRecord(**item) for item in items]


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
