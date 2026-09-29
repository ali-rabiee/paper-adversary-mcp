"""Scholarly search backends: OpenAlex, Semantic Scholar, arXiv, Crossref."""

from __future__ import annotations

import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

from paper_adversary.search.base import NotSupported, PaperRecord, parse_identifier, strip_arxiv_version


def _year_range(year_from: int | None, year_to: int | None) -> tuple[int | None, int | None]:
    if year_from and year_to and year_from > year_to:
        year_from, year_to = year_to, year_from
    return year_from, year_to


def _arxiv_from_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    m = re.match(r"10\.48550/arxiv\.(.+)", doi, re.I)
    return strip_arxiv_version(m.group(1)) if m else None


# ---------------------------------------------------------------- OpenAlex


class OpenAlexBackend:
    name = "openalex"
    min_interval_s = 0.15
    base = "https://api.openalex.org"
    select = ("id,doi,display_name,publication_year,authorships,primary_location,cited_by_count,"
              "abstract_inverted_index,locations")

    def __init__(self, contact_email: str | None = None):
        self.contact_email = contact_email

    def _params(self, **extra) -> dict:
        params = {"select": self.select, **extra}
        if self.contact_email:
            params["mailto"] = self.contact_email
        return params

    @staticmethod
    def _abstract(inverted: dict | None) -> str | None:
        if not inverted:
            return None
        positions = [(pos, word) for word, poss in inverted.items() for pos in poss]
        return " ".join(word for _, word in sorted(positions)) or None

    def _parse(self, w: dict) -> PaperRecord:
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        arxiv = _arxiv_from_doi(doi)
        for loc in w.get("locations") or []:
            url = (loc or {}).get("landing_page_url") or ""
            m = re.search(r"arxiv\.org/abs/([^\s?#]+)", url)
            if m and not arxiv:
                arxiv = strip_arxiv_version(m.group(1))
        primary = w.get("primary_location") or {}
        return PaperRecord(
            title=w.get("display_name") or "",
            authors=[a["author"]["display_name"] for a in w.get("authorships") or [] if a.get("author")],
            year=w.get("publication_year"),
            venue=(primary.get("source") or {}).get("display_name"),
            doi=None if arxiv and doi and doi.lower().startswith("10.48550") else doi,
            arxiv_id=arxiv,
            url=primary.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else w.get("id")),
            abstract=self._abstract(w.get("abstract_inverted_index")),
            citation_count=w.get("cited_by_count"),
            source=self.name,
            ids={"openalex": w.get("id")},
        )

    async def search(self, fetch, query, limit, year_from, year_to):
        year_from, year_to = _year_range(year_from, year_to)
        filters = []
        if year_from:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to:
            filters.append(f"to_publication_date:{year_to}-12-31")
        params = self._params(search=query, **{"per-page": limit})
        if filters:
            params["filter"] = ",".join(filters)
        data = await fetch.get_json(f"{self.base}/works", params=params)
        return [self._parse(w) for w in (data or {}).get("results", [])]

    async def lookup(self, fetch, identifier):
        kind, value = parse_identifier(identifier)
        if kind == "doi":
            data = await fetch.get_json(f"{self.base}/works/doi:{value}", params=self._params())
            return [self._parse(data)] if data else []
        if kind == "arxiv":
            data = await fetch.get_json(f"{self.base}/works/doi:10.48550/arXiv.{value}", params=self._params())
            return [self._parse(data)] if data else []
        cleaned = re.sub(r"[,:|!]", " ", value)
        data = await fetch.get_json(f"{self.base}/works",
                                    params=self._params(filter=f"title.search:{cleaned}", **{"per-page": 5}))
        return [self._parse(w) for w in (data or {}).get("results", [])]

    async def citing(self, fetch, identifier, limit):
        found = await self.lookup(fetch, identifier)
        if not found or not found[0].ids.get("openalex"):
            return []
        wid = found[0].ids["openalex"].rsplit("/", 1)[-1]
        data = await fetch.get_json(f"{self.base}/works", params=self._params(
            filter=f"cites:{wid}", sort="cited_by_count:desc", **{"per-page": limit}))
        return [self._parse(w) for w in (data or {}).get("results", [])]


# ---------------------------------------------------------------- Semantic Scholar


class SemanticScholarBackend:
    name = "semantic_scholar"
    base = "https://api.semanticscholar.org/graph/v1"
    fields = "title,authors,year,venue,externalIds,citationCount,abstract,url"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("S2_API_KEY") or None
        self.min_interval_s = 1.05 if self.api_key else 1.5

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def _parse(self, p: dict) -> PaperRecord:
        ext = p.get("externalIds") or {}
        doi = ext.get("DOI")
        arxiv = ext.get("ArXiv") or _arxiv_from_doi(doi)
        return PaperRecord(
            title=p.get("title") or "",
            authors=[a.get("name", "") for a in p.get("authors") or []],
            year=p.get("year"),
            venue=p.get("venue") or None,
            doi=None if doi and doi.lower().startswith("10.48550") else doi,
            arxiv_id=arxiv,
            url=p.get("url") or (f"https://arxiv.org/abs/{arxiv}" if arxiv else None),
            abstract=p.get("abstract"),
            citation_count=p.get("citationCount"),
            source=self.name,
            ids={"s2": p.get("paperId"), "corpus": ext.get("CorpusId")},
        )

    async def search(self, fetch, query, limit, year_from, year_to):
        year_from, year_to = _year_range(year_from, year_to)
        params = {"query": query, "limit": limit, "fields": self.fields}
        if year_from or year_to:
            params["year"] = f"{year_from or ''}-{year_to or ''}"
        data = await fetch.get_json(f"{self.base}/paper/search", params=params, headers=self._headers())
        return [self._parse(p) for p in (data or {}).get("data", []) or []]

    async def lookup(self, fetch, identifier):
        kind, value = parse_identifier(identifier)
        if kind in ("doi", "arxiv"):
            pid = f"DOI:{value}" if kind == "doi" else f"ARXIV:{value}"
            data = await fetch.get_json(f"{self.base}/paper/{urllib.parse.quote(pid, safe=':/')}",
                                        params={"fields": self.fields}, headers=self._headers())
            return [self._parse(data)] if data else []
        data = await fetch.get_json(f"{self.base}/paper/search/match", params={"query": value, "fields": self.fields},
                                    headers=self._headers())
        return [self._parse(p) for p in (data or {}).get("data", []) or []]

    async def citing(self, fetch, identifier, limit):
        found = await self.lookup(fetch, identifier)
        if not found or not found[0].ids.get("s2"):
            return []
        data = await fetch.get_json(f"{self.base}/paper/{found[0].ids['s2']}/citations",
                                    params={"fields": self.fields.replace(",abstract", ""), "limit": limit},
                                    headers=self._headers())
        return [self._parse(c.get("citingPaper") or {}) for c in (data or {}).get("data", []) or []
                if (c.get("citingPaper") or {}).get("title")]


# ---------------------------------------------------------------- arXiv


_ATOM = "{http://www.w3.org/2005/Atom}"
_ARX = "{http://arxiv.org/schemas/atom}"
_STOP = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "via", "by", "from", "is", "are",
         "we", "our", "using", "based", "towards", "toward"}


class ArxivBackend:
    """arXiv export API. It blocks heavy users for hours, so it gets a slow gate and a hard breaker."""

    name = "arxiv"
    min_interval_s = 3.5
    base = "https://export.arxiv.org/api/query"
    block_statuses = (403, 406, 429, 503)

    def _parse_feed(self, xml_text: str) -> list[PaperRecord]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        out = []
        for entry in root.findall(f"{_ATOM}entry"):
            raw_id = (entry.findtext(f"{_ATOM}id") or "").strip()
            m = re.search(r"arxiv\.org/abs/(.+)$", raw_id)
            if not m:
                continue
            arxiv = strip_arxiv_version(m.group(1))
            published = entry.findtext(f"{_ATOM}published") or ""
            out.append(PaperRecord(
                title=re.sub(r"\s+", " ", entry.findtext(f"{_ATOM}title") or "").strip(),
                authors=[a.findtext(f"{_ATOM}name") or "" for a in entry.findall(f"{_ATOM}author")],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue=(entry.findtext(f"{_ARX}journal_ref") or "arXiv").strip(),
                doi=(entry.findtext(f"{_ARX}doi") or None),
                arxiv_id=arxiv,
                url=f"https://arxiv.org/abs/{arxiv}",
                abstract=re.sub(r"\s+", " ", entry.findtext(f"{_ATOM}summary") or "").strip() or None,
                source=self.name,
                ids={"arxiv": arxiv},
            ))
        return out

    def _query(self, query: str, year_from: int | None, year_to: int | None) -> str:
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", query) if w.lower() not in _STOP][:8]
        q = " AND ".join(f"all:{w}" for w in words) or f'all:"{query}"'
        if year_from or year_to:
            q = f"({q}) AND submittedDate:[{year_from or 1990}01010000 TO {year_to or 2100}12312359]"
        return q

    async def search(self, fetch, query, limit, year_from, year_to):
        year_from, year_to = _year_range(year_from, year_to)
        text = await fetch.get_text(self.base, params={
            "search_query": self._query(query, year_from, year_to), "start": 0, "max_results": limit,
            "sortBy": "relevance", "sortOrder": "descending"})
        return self._parse_feed(text or "")

    async def lookup(self, fetch, identifier):
        kind, value = parse_identifier(identifier)
        if kind == "arxiv":
            text = await fetch.get_text(self.base, params={"id_list": value, "max_results": 1})
            return self._parse_feed(text or "")
        if kind == "doi":
            return []
        title = re.sub(r"[\"()]", " ", value)
        text = await fetch.get_text(self.base, params={"search_query": f'ti:"{title}"', "max_results": 5})
        return self._parse_feed(text or "")

    async def citing(self, fetch, identifier, limit):
        raise NotSupported("arXiv has no citation index")


# ---------------------------------------------------------------- Crossref


class CrossrefBackend:
    name = "crossref"
    base = "https://api.crossref.org/works"
    select = "DOI,title,author,issued,container-title,is-referenced-by-count,URL,abstract,type"

    def __init__(self, contact_email: str | None = None):
        self.contact_email = contact_email
        self.min_interval_s = 0.12 if contact_email else 0.3

    def _params(self, **extra) -> dict:
        params = {"select": self.select, **extra}
        if self.contact_email:
            params["mailto"] = self.contact_email
        return params

    def _parse(self, item: dict) -> PaperRecord:
        issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
        abstract = item.get("abstract")
        if abstract:
            abstract = re.sub(r"<[^>]+>", " ", abstract)
        authors = []
        for a in item.get("author") or []:
            name = " ".join(x for x in (a.get("given"), a.get("family")) if x) or a.get("name") or ""
            if name:
                authors.append(name)
        doi = item.get("DOI")
        return PaperRecord(
            title=((item.get("title") or [""])[0] or "").strip(),
            authors=authors,
            year=issued[0] if issued and isinstance(issued[0], int) else None,
            venue=((item.get("container-title") or [None])[0]),
            doi=doi,
            arxiv_id=_arxiv_from_doi(doi),
            url=item.get("URL"),
            abstract=re.sub(r"\s+", " ", abstract).strip() if abstract else None,
            citation_count=item.get("is-referenced-by-count"),
            source=self.name,
            ids={"crossref_type": item.get("type")},
        )

    async def search(self, fetch, query, limit, year_from, year_to):
        year_from, year_to = _year_range(year_from, year_to)
        params = self._params(**{"query.bibliographic": query, "rows": limit})
        filters = []
        if year_from:
            filters.append(f"from-pub-date:{year_from}")
        if year_to:
            filters.append(f"until-pub-date:{year_to}")
        if filters:
            params["filter"] = ",".join(filters)
        data = await fetch.get_json(self.base, params=params)
        return [self._parse(i) for i in ((data or {}).get("message") or {}).get("items", [])]

    async def lookup(self, fetch, identifier):
        kind, value = parse_identifier(identifier)
        if kind == "doi":
            data = await fetch.get_json(f"{self.base}/{urllib.parse.quote(value, safe='/:()._-;')}",
                                        params={"mailto": self.contact_email} if self.contact_email else None)
            item = (data or {}).get("message")
            return [self._parse(item)] if item else []
        if kind == "arxiv":
            return []
        data = await fetch.get_json(self.base, params=self._params(**{"query.bibliographic": value, "rows": 5}))
        return [self._parse(i) for i in ((data or {}).get("message") or {}).get("items", [])]

    async def citing(self, fetch, identifier, limit):
        raise NotSupported("Crossref does not expose citing works")
