"""Open full texts of prior work, so quotes can be read and checked verbatim.

A reference (arXiv ID, DOI, or title) is resolved through the scholarly APIs, then fetched from, in order: the
arXiv HTML rendering (LaTeXML: clean text, math as LaTeX, paragraph anchors), the arXiv PDF, and open-access PDFs
that OpenAlex or Semantic Scholar list on an allowed host. Download URLs only ever come from that metadata, never
from an agent. Documents are size-capped, type-checked and extracted in a resource-limited child process
(search/extract.py). Results are cached machine-wide (content-addressed) and snapshotted into the run's prior/
folder, so every reader in a run sees the same bytes. A paper with no retrievable open text is reported as
`fulltext_unavailable` with the reason; the abstract is never passed off as the full text.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

import httpx2

from paper_adversary import __version__
from paper_adversary.search.base import (
    Breaker,
    DiskCache,
    PaperRecord,
    ProviderUnavailable,
    RateGate,
    parse_identifier,
    strip_arxiv_version,
)
from paper_adversary.util import FileLock, atomic_write_json, atomic_write_text, read_json, sha256_text, utcnow_iso

ALLOWED_HOSTS = (
    "arxiv.org", "export.arxiv.org", "openreview.net", "proceedings.mlr.press", "proceedings.neurips.cc",
    "papers.nips.cc", "aclanthology.org", "jmlr.org", "www.jmlr.org", "openaccess.thecvf.com", "ojs.aaai.org",
    "www.ijcai.org", "ijcai.org", "www.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "europepmc.org",
    "pdfs.semanticscholar.org",
)
ARXIV_HOSTS = ("arxiv.org", "export.arxiv.org")
ARXIV_API_INTERVAL = 3.5  # shared with the arXiv search backend's gate (one request per 3 s per arXiv's terms)
BLOCK_STATUSES = (403, 406, 429, 503)
MAX_REDIRECTS = 5
_VERSION = re.compile(r"/(?:html|pdf|abs)/([^/?#]+?)(v\d+)?(?:\.pdf)?/?(?:[?#]|$)")


@dataclass
class FullTextResult:
    status: str  # available | fulltext_unavailable | prior_not_found
    key: str | None = None  # arxiv:<id> or doi:<doi>
    reason: str | None = None  # why it is unavailable
    detail: str | None = None  # e.g. an open-access URL on a host that is not allowed
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    aliases: list[str] = field(default_factory=list)  # identifiers that resolved to this paper
    source: str | None = None  # arxiv_html | arxiv_pdf | oa_pdf | user_supplied
    url: str | None = None
    version: str | None = None
    sha256: str | None = None  # of text_md
    page_count: int | None = None
    abstract: str | None = None
    warnings: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    fetched_at: str | None = None
    text_md: str | None = None
    sections: list[dict] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.status == "available" and bool(self.text_md)

    def meta(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("text_md", "sections")}

    def label(self) -> str:
        if not self.available:
            return f"{self.status}" + (f" ({self.reason})" if self.reason else "")
        version = f" {self.version}" if self.version else ""
        return f"{self.source}{version}, {len(self.text_md or ''):,} characters"


def _host_ok(url: str, policy: str, extra: list[str]) -> bool:
    try:
        if not isinstance(url, str) or any(c in url for c in "\r\n\t\x7f ") or len(url) > 2000:
            return False
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.port not in (None, 443):
            return False
        host = parsed.hostname.lower()
    except ValueError:  # malformed metadata URLs (bad ports, brackets) are refused, never fatal
        return False
    allowed = ARXIV_HOSTS if policy == "arxiv_only" else ALLOWED_HOSTS + tuple(h.lower() for h in extra)
    return host in allowed


_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}|[a-z][a-z.\-]*(?:\.[A-Za-z]{2})?/\d{7}")
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")


def valid_arxiv_id(value: str | None) -> bool:
    return bool(value) and bool(_ARXIV_ID.fullmatch(strip_arxiv_version(str(value))))


def valid_doi(value: str | None) -> bool:
    return bool(value) and bool(_DOI.fullmatch(str(value))) and ".." not in str(value) and len(str(value)) < 200


def _document_warnings(out: dict) -> list[str]:
    """Extraction warnings that apply to a prior-work document (not the ones about the submission's PDF roles)."""
    return [w for w in out.get("warnings") or [] if "paper_format" not in w]


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def title_similarity(a: str, b: str) -> float:
    from paper_adversary.search.refcheck import title_similarity as sim

    return sim(a, b)


class FullTextStore:
    """Resolve, fetch, extract and cache open full texts; optionally snapshot them into a run's prior/ folder."""

    def __init__(self, cache_root: Path, cfg, providers: list[str] | None = None, prior_dir: Path | None = None,
                 transport: httpx2.AsyncBaseTransport | None = None, contact_email: str | None = None):
        self.cfg = cfg
        self.cache_root = cache_root
        self.root = cache_root / "fulltext"
        self.prior_dir = prior_dir
        self.providers = [p for p in (providers or ["openalex", "semantic_scholar", "crossref"]) if p != "arxiv"]
        self.transport = transport
        self.contact_email = contact_email
        self.breaker = Breaker(cache_root / "breaker")
        self.oa_cache = DiskCache(cache_root / "search", 14)
        self._search = None
        self._client: httpx2.AsyncClient | None = None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
        if self._search is not None:
            await self._search.aclose()

    async def __aenter__(self) -> "FullTextStore":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # ------------------------------------------------------------ cache

    def _index_path(self, key: str) -> Path:
        return self.root / "index" / f"{_digest(key)}.json"

    def _alias_path(self, identifier: str) -> Path:
        return self.root / "alias" / f"{_digest(identifier.strip().lower())}.json"

    def _load(self, key: str) -> FullTextResult | None:
        meta = read_json(self._index_path(key))
        if not meta:
            return None
        age_days = (time.time() - meta.get("_cached_at", 0)) / 86400
        pinned = meta.get("status") == "available" and meta.get("version") and str(key).startswith("arxiv:")
        ttl = self.cfg.ttl_days if meta.get("status") == "available" else self.cfg.negative_ttl_days
        if not pinned and age_days > ttl:
            return None
        result = FullTextResult(**{k: v for k, v in meta.items() if not k.startswith("_")})
        if result.status == "available":
            text_path = self.root / "text" / f"{result.sha256}.md"
            if not text_path.is_file():
                return None
            result.text_md = text_path.read_text(encoding="utf-8")
            result.sections = read_json(self.root / "text" / f"{result.sha256}.sections.json", []) or []
        return result

    def _save(self, result: FullTextResult, *identifiers: str) -> None:
        if result.key is None:
            return
        if result.available:
            atomic_write_text(self.root / "text" / f"{result.sha256}.md", result.text_md or "")
            atomic_write_json(self.root / "text" / f"{result.sha256}.sections.json", result.sections)
        if result.status != "fulltext_unavailable" or result.reason not in ("provider_paused", "offline_not_cached"):
            atomic_write_json(self._index_path(result.key), {**result.meta(), "_cached_at": time.time()})
        for ident in identifiers:
            if ident:
                atomic_write_json(self._alias_path(ident), {"key": result.key})

    def cached(self, identifier: str) -> FullTextResult | None:
        """What is known without any network access: the run snapshot first, then the machine cache."""
        if self.prior_dir is not None:
            snap = read_snapshot(self.prior_dir, identifier)
            if snap is not None:
                return snap
        key = (read_json(self._alias_path(identifier)) or {}).get("key")
        if key is None:
            kind, value = parse_identifier(identifier)
            key = {"arxiv": f"arxiv:{value}", "doi": f"doi:{value.lower()}"}.get(kind)
        return self._load(key) if key else None

    # ------------------------------------------------------------ public

    async def get(self, identifier: str, hint: dict | None = None) -> FullTextResult:
        """Full text of one paper, fetched if needed. Never raises for network or document problems."""
        hint = hint or {}
        hit = self.cached(identifier)
        if hit is not None and (hit.available or self.cfg.offline):
            return self._snapshot(hit, identifier)
        if self.cfg.offline or not self.cfg.enabled:
            reason = "offline_not_cached" if self.cfg.offline else "fulltext_disabled"
            return FullTextResult("fulltext_unavailable", reason=reason, title=hint.get("title"))
        try:
            key, record = await self._resolve(identifier, hint)
        except Exception as exc:  # resolution problems are reported, not raised
            return FullTextResult("fulltext_unavailable", reason="resolution_failed", detail=str(exc)[:200])
        if key is None:
            result = FullTextResult("prior_not_found", reason="no scholarly database knows this identifier",
                                    title=hint.get("title") or identifier, fetched_at=utcnow_iso())
            atomic_write_json(self._alias_path(identifier), {"key": None, "status": "prior_not_found"})
            return result
        known = self._load(key)
        if known is not None and (known.available or known.reason != "provider_paused"):
            self._save(known, identifier)
            return self._snapshot(known, identifier)
        try:
            result = await self._fetch(key, record)
        except Exception as exc:  # a document problem is reported, never raised into the worker
            result = FullTextResult("fulltext_unavailable", key=key, reason="fetch_failed",
                                    detail=f"{type(exc).__name__}: {exc}"[:200], fetched_at=utcnow_iso())
            return self._snapshot(result, identifier)
        self._save(result, identifier)
        return self._snapshot(result, identifier)

    async def add_user_file(self, identifier: str, path: Path, title: str | None = None,
                            submission_text: str | None = None) -> FullTextResult:
        """A PDF or HTML file the user supplies (e.g. a paywalled paper), extracted like a download.

        It goes into this run's snapshot only, never the machine-wide cache, and only if (a) its title matches
        the paper it is filed under, (b) it is not the submission itself, and (c) no downloaded text is there."""
        from paper_adversary.search.extract import run_extraction

        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        if any(part.startswith(".") for part in path.parts[1:]):
            raise ValueError(f"{path} is hidden or inside a hidden folder; copy it somewhere visible first")
        size_mb = path.stat().st_size / 1e6
        if size_mb > self.cfg.max_pdf_mb:
            raise ValueError(f"{path.name} is {size_mb:.0f} MB; the limit is {self.cfg.max_pdf_mb:g} MB")
        head = path.read_bytes()[:512].lstrip()
        if head.startswith(b"%PDF-"):
            kind = "pdf"
        elif head[:15].lower().startswith((b"<!doctype html", b"<html")):
            kind = "html"
        else:
            raise ValueError(f"{path.name} is neither a PDF nor an HTML page")
        kind_id, value = parse_identifier(identifier)
        if kind_id == "arxiv" and valid_arxiv_id(value):
            key = f"arxiv:{value}"
        elif kind_id == "doi" and valid_doi(value):
            key = f"doi:{value.lower()}"
        else:
            raise ValueError("identify the paper by its arXiv ID or DOI")
        given = " ".join(str(title or "").split())[:200].strip("\"'<>") or None
        expected = given
        if expected is None and not self.cfg.offline:
            try:
                _, record = await self._resolve(identifier, {})
                expected = record.title if record else None
            except Exception:
                expected = None
        out = await asyncio.to_thread(run_extraction, path, kind, max_pages=self.cfg.max_pages,
                                      timeout_s=self.cfg.extract_timeout_seconds)
        if out.get("status") != "ok":
            return FullTextResult("fulltext_unavailable", key=key, reason=out.get("reason"), detail=out.get("detail"))
        text = out.get("text_md") or ""
        found = " ".join(str(out.get("title") or "").split())[:200] or None
        if expected and found and title_similarity(expected, found) < 0.6:
            raise ValueError(f"the file's title is {found!r}, not {expected!r}; is it the right paper?")
        if not (expected or found):
            raise ValueError("cannot tell which paper this is; pass its title")
        if submission_text:
            from paper_adversary.isolation import _shingles

            mine, theirs = set(_shingles(submission_text)), set(_shingles(text))
            small = min(len(mine), len(theirs))
            if small and len(mine & theirs) / small >= 0.5:
                raise ValueError("this file is the submission itself, not prior work")
        if self.prior_dir is not None:
            existing = read_snapshot(self.prior_dir, key)
            if existing is not None and existing.available and existing.source != "user_supplied":
                raise ValueError(f"this run already has a downloaded text of {key} ({existing.source}); a supplied "
                                 "file never replaces it")
        result = FullTextResult("available", key=key, title=expected or found, source="user_supplied", url=None,
                                sha256=sha256_text(text), page_count=out.get("page_count"),
                                warnings=_document_warnings(out), fetched_at=utcnow_iso(), text_md=text,
                                sections=out.get("sections") or [])
        return self._snapshot(result, identifier)  # this run only: not the shared cache

    # ------------------------------------------------------------ resolution

    def _literature(self):
        if self._search is None:
            from paper_adversary.search import LiteratureSearch

            self._search = LiteratureSearch(self.providers or ["openalex"], self.cache_root, 14,
                                            self.contact_email, self.transport)
        return self._search

    async def _resolve(self, identifier: str, hint: dict) -> tuple[str | None, PaperRecord | None]:
        kind, value = parse_identifier(identifier)
        if kind == "title" and hint.get("arxiv_id"):
            kind, value = "arxiv", strip_arxiv_version(str(hint["arxiv_id"]))
        elif kind == "title" and hint.get("doi"):
            kind, value = "doi", str(hint["doi"])
        if (kind == "arxiv" and not valid_arxiv_id(value)) or (kind == "doi" and not valid_doi(value)):
            return None, None  # never put an unvalidated identifier into a URL
        search = self._literature()
        # A record's title comes from the scholarly APIs or, failing that, from the document itself, never from
        # the caller: the caller may be an agent, and the title reaches a blind verifier's prompt.
        if kind == "arxiv":
            found = await search.lookup(f"arXiv:{value}")
            record = next(iter(found["records"]), None) or PaperRecord(title="", arxiv_id=value)
            record.arxiv_id = record.arxiv_id or value
            return f"arxiv:{value}", record
        if kind == "doi":
            found = await search.lookup(value)
            record = next(iter(found["records"]), None) or PaperRecord(title="", doi=value)
            record.doi = record.doi or value
            return self._key(record), record
        found = await search.lookup(value[:300])
        scored = sorted(((title_similarity(value, r.title), r) for r in found["records"]), key=lambda x: -x[0])
        if not scored or scored[0][0] < 0.88:
            return None, None
        best = scored[0][1]
        year = hint.get("year")
        if isinstance(year, int) and best.year and abs(year - best.year) > 1:  # "Part I" vs "Part II" and kin
            return None, None
        return self._key(best), best

    @staticmethod
    def _key(record: PaperRecord) -> str | None:
        if record.arxiv_id and valid_arxiv_id(record.arxiv_id):
            return f"arxiv:{strip_arxiv_version(record.arxiv_id)}"
        if record.doi and valid_doi(record.doi):
            return f"doi:{record.doi.lower()}"
        return None

    async def _oa_urls(self, record: PaperRecord) -> list[str]:
        """Open-access PDF links that OpenAlex and Semantic Scholar list for a DOI (separately cached calls)."""
        if not valid_doi(record.doi) or self.cfg.host_policy == "arxiv_only":
            return []
        doi = quote(record.doi, safe="/:;()._-")
        cached = self.oa_cache.get("oa", record.doi.lower())
        if cached is not None:
            return cached
        from paper_adversary.search import Fetcher

        search = self._literature()
        urls: list[str] = []
        gate = search.gates.get("openalex") or RateGate(self.cache_root / "ratelimit", "openalex", 0.15)
        openalex = Fetcher(search._client, gate, search.breaker, "openalex")
        try:
            work = await openalex.get_json(f"https://api.openalex.org/works/doi:{doi}",
                                           params={"select": "id,doi,best_oa_location,locations,open_access"})
        except ProviderUnavailable:
            work = None
        if work:
            locs = [work.get("best_oa_location") or {}] + list(work.get("locations") or [])
            urls += [loc.get("pdf_url") for loc in locs if isinstance(loc, dict) and loc.get("pdf_url")]
        gate = search.gates.get("semantic_scholar") or RateGate(self.cache_root / "ratelimit", "semantic_scholar", 1.1)
        s2 = Fetcher(search._client, gate, search.breaker, "semantic_scholar")
        try:
            paper = await s2.get_json(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                                      params={"fields": "openAccessPdf,externalIds"})
        except ProviderUnavailable:
            paper = None
        if paper and isinstance(paper.get("openAccessPdf"), dict) and paper["openAccessPdf"].get("url"):
            urls.append(paper["openAccessPdf"]["url"])
        urls = list(dict.fromkeys(u for u in urls if isinstance(u, str) and u.startswith("https://")))
        self.oa_cache.put("oa", record.doi.lower(), urls)
        return urls

    # ------------------------------------------------------------ fetching

    def _http(self) -> httpx2.AsyncClient:
        if self._client is None:
            agent = f"paper-adversary-mcp/{__version__} (verifying quotations for a pre-submission review)"
            self._client = httpx2.AsyncClient(timeout=httpx2.Timeout(30.0, connect=10.0), follow_redirects=False,
                                              headers={"User-Agent": agent}, transport=self.transport)
        return self._client

    async def _fetch(self, key: str, record: PaperRecord) -> FullTextResult:
        from paper_adversary.search.extract import run_extraction

        base = FullTextResult("fulltext_unavailable", key=key, title=record.title or None, authors=record.authors[:8],
                              year=record.year, doi=record.doi, arxiv_id=record.arxiv_id, abstract=record.abstract,
                              fetched_at=utcnow_iso())
        candidates: list[tuple[str, str]] = []
        if record.arxiv_id and valid_arxiv_id(record.arxiv_id):
            arxiv = strip_arxiv_version(record.arxiv_id)
            candidates += [("arxiv_html", f"https://arxiv.org/html/{arxiv}"), ("arxiv_pdf", f"https://arxiv.org/pdf/{arxiv}")]
        oa = await self._oa_urls(record)
        refused = [u for u in oa if not _host_ok(u, self.cfg.host_policy, self.cfg.allow_hosts)]
        candidates += [("oa_pdf", u) for u in oa if u not in refused]
        if not candidates:
            base.reason = "host_not_allowed" if refused else "no_open_access_copy"
            base.detail = f"open-access copy on a host that is not allowed: {refused[0]}" if refused else None
            return base
        reasons = []
        for source, url in candidates:
            kind = "html" if source == "arxiv_html" else "pdf"
            body, final_url, reason = await self._download(url, kind)
            entry = {"source": source, "url": url, "final_url": final_url, "reason": reason}
            base.candidates.append(entry)
            if body is None:
                reasons.append(reason)
                continue
            with tempfile.TemporaryDirectory(prefix="pa-fulltext-") as tmp:
                path = Path(tmp) / f"document.{kind}"
                path.write_bytes(body)
                out = await asyncio.to_thread(run_extraction, path, kind, max_pages=self.cfg.max_pages,
                                              timeout_s=self.cfg.extract_timeout_seconds)
            if out.get("status") != "ok":
                entry["reason"] = out.get("reason") or "extraction_failed"
                reasons.append(entry["reason"])
                continue
            text = out.get("text_md") or ""
            got_title = out.get("title") or ""
            if record.title and got_title and title_similarity(record.title, got_title) < 0.6:
                entry["reason"] = "wrong_paper"
                entry["detail"] = f"the document's title is {got_title[:120]!r}"
                reasons.append("wrong_paper")
                continue
            version = None
            if source.startswith("arxiv"):
                m = _VERSION.search(final_url or url)
                version = m.group(2) if m else None
            return FullTextResult("available", key=key, title=record.title or got_title, authors=record.authors[:8],
                                  year=record.year, doi=record.doi, arxiv_id=record.arxiv_id, source=source,
                                  url=final_url or url, version=version,
                                  sha256=sha256_text(text), page_count=out.get("page_count"),
                                  abstract=record.abstract, warnings=_document_warnings(out),
                                  candidates=base.candidates, fetched_at=utcnow_iso(), text_md=text,
                                  sections=out.get("sections") or [])
        order = ("provider_paused", "wrong_paper", "too_large", "no_text_layer", "encrypted", "timeout", "crashed",
                 "extraction_failed", "conversion_failed", "not_pdf", "not_html", "download_failed", "not_found")
        base.reason = next((r for r in order if r in reasons), reasons[0] if reasons else "download_failed")
        if refused:
            base.detail = f"also listed on a host that is not allowed: {refused[0]}"
        return base

    async def _gate(self, host: str) -> str | None:
        """Wait for this host's turn; returns a breaker name, or None if the host is paused."""
        ratelimit = self.cache_root / "ratelimit"
        if host in ARXIV_HOSTS:
            name = "arxiv"
            if self.breaker.is_open(name):
                return None
            await RateGate(ratelimit, "arxiv_docs", self.cfg.arxiv_document_interval_seconds).wait()
            await RateGate(ratelimit, "arxiv", ARXIV_API_INTERVAL).wait()
            return name
        name = f"oa_{host}"
        if self.breaker.is_open(name):
            return None
        await RateGate(ratelimit, name, self.cfg.host_interval_seconds).wait()
        return name

    async def _download(self, url: str, kind: str) -> tuple[bytes | None, str, str | None]:
        """GET with redirects followed by hand (each hop host-checked), a streamed size cap and a type check."""
        cap = int((self.cfg.max_html_mb if kind == "html" else self.cfg.max_pdf_mb) * 1e6)
        client = self._http()
        for _ in range(MAX_REDIRECTS + 1):
            if not _host_ok(url, self.cfg.host_policy, self.cfg.allow_hosts):
                return None, url, "host_not_allowed"
            host = urlparse(url).hostname.lower()
            breaker = await self._gate(host)
            if breaker is None:
                return None, url, "provider_paused"
            try:
                client.build_request("GET", url)  # malformed URLs fail here, not deep inside the transfer
                async with client.stream("GET", url) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                        url = urljoin(url, resp.headers["location"])
                        continue
                    if resp.status_code in BLOCK_STATUSES:
                        self.breaker.trip(breaker, 7200 if host in ARXIV_HOSTS else 3600, f"HTTP {resp.status_code}")
                        return None, url, "provider_paused"
                    if resp.status_code >= 400:
                        return None, url, "not_found" if resp.status_code == 404 else "download_failed"
                    declared = int(resp.headers.get("content-length") or 0)
                    if declared > cap:
                        return None, url, "too_large"
                    ctype = (resp.headers.get("content-type") or "").lower()
                    chunks, size = [], 0
                    async for chunk in resp.aiter_bytes():
                        size += len(chunk)
                        if size > cap:
                            return None, url, "too_large"
                        chunks.append(chunk)
            except (httpx2.TimeoutException, httpx2.RequestError, httpx2.InvalidURL, ValueError):
                return None, url, "download_failed"
            body = b"".join(chunks)
            if kind == "pdf" and not body.lstrip()[:5] == b"%PDF-":
                return None, url, "not_pdf"
            if kind == "html" and "html" not in ctype and not body.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
                return None, url, "not_html"
            return body, url, None
        return None, url, "download_failed"

    # ------------------------------------------------------------ run snapshot

    def _snapshot(self, result: FullTextResult, identifier: str | None = None) -> FullTextResult:
        if identifier and identifier not in result.aliases:
            result.aliases.append(identifier)
        if self.prior_dir is not None and result.key is not None:
            write_snapshot(self.prior_dir, result)
        return result


# ---------------------------------------------------------------- per-run snapshot (prior/)


def write_snapshot(prior_dir: Path, result: FullTextResult) -> None:
    """Record a paper (and its text, if available) in the run's prior/ folder. Safe under concurrent writers."""
    prior_dir.mkdir(parents=True, exist_ok=True)
    if result.available:
        text_path = prior_dir / f"{result.sha256}.md"
        if not text_path.is_file():
            atomic_write_text(text_path, result.text_md or "")
            atomic_write_json(prior_dir / f"{result.sha256}.sections.json", result.sections)
    with FileLock(prior_dir / ".index.lock"):
        index = read_json(prior_dir / "index.json", {}) or {}
        current = index.get(result.key) or {}
        aliases = list(dict.fromkeys((current.get("aliases") or []) + result.aliases))
        if current.get("status") == "available" and not result.available:
            current["aliases"] = aliases  # never replace a text with a failure
            index[result.key] = current
        else:
            index[result.key] = {**result.meta(), "aliases": aliases, "recorded_at": utcnow_iso()}
        atomic_write_json(prior_dir / "index.json", index)


def snapshot_index(prior_dir: Path) -> dict:
    return read_json(prior_dir / "index.json", {}) or {}


def snapshot_key(prior_dir: Path, identifier: str, title: str | None = None) -> str | None:
    """The snapshot key for a reference given by arXiv ID, DOI or (closely matching) title."""
    index = snapshot_index(prior_dir)
    if identifier in index:
        return identifier
    wanted_alias = identifier.strip().lower()
    for key, meta in index.items():
        if wanted_alias in (a.strip().lower() for a in meta.get("aliases") or []):
            return key
    kind, value = parse_identifier(identifier)
    if kind == "arxiv":
        for key, meta in index.items():
            if key == f"arxiv:{value}" or strip_arxiv_version(str(meta.get("arxiv_id") or "")) == value:
                return key
    if kind == "doi":
        for key, meta in index.items():
            if key == f"doi:{value.lower()}" or str(meta.get("doi") or "").lower() == value.lower():
                return key
    wanted = title or (value if kind == "title" else None)
    if wanted:
        scored = sorted(((title_similarity(wanted, meta.get("title") or ""), key) for key, meta in index.items()),
                        key=lambda x: -x[0])
        if scored and scored[0][0] >= 0.88:
            return scored[0][1]
    return None


def read_snapshot(prior_dir: Path, identifier: str, title: str | None = None) -> FullTextResult | None:
    key = snapshot_key(prior_dir, identifier, title)
    if key is None:
        return None
    meta = snapshot_index(prior_dir).get(key) or {}
    known = {k: v for k, v in meta.items() if k in FullTextResult.__dataclass_fields__}
    result = FullTextResult(**known)
    if result.status == "available":
        path = prior_dir / f"{result.sha256}.md"
        if not path.is_file():
            return None
        result.text_md = path.read_text(encoding="utf-8")
        result.sections = read_json(prior_dir / f"{result.sha256}.sections.json", []) or []
    return result


def describe(result: FullTextResult) -> str:
    """A one-paragraph header for agents: what this text is and how to cite locations in it."""
    if not result.available:
        why = f" ({result.reason})" if result.reason else ""
        extra = f" {result.detail}." if result.detail else ""
        return (f"FULL TEXT UNAVAILABLE{why}: {result.title or result.key}.{extra} Do not quote this paper's body; "
                "say that you could not read it.")
    authors = ", ".join(result.authors[:4]) + (" et al." if len(result.authors) > 4 else "")
    return (f"{result.title or result.key} ({result.year or 'n.d.'}){' — ' + authors if authors else ''}\n"
            f"Key: {result.key}; source: {result.source}{' ' + result.version if result.version else ''}; "
            f"{result.url or ''}; text sha256 {str(result.sha256)[:12]}\n"
            "VERBATIM TEXT — quote only from this text, character for character.")


def dump(result: FullTextResult) -> str:
    return json.dumps(result.meta(), indent=2, default=str)
