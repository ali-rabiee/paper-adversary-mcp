"""Scholarly search backends, merging, and reference checking, against canned HTTP responses."""

import asyncio

import httpx2
import pytest

from paper_adversary.search import LiteratureSearch, merge_records
from paper_adversary.search.base import PaperRecord, parse_identifier
from paper_adversary.search.refcheck import check_references, title_similarity

OPENALEX_WORK = {
    "id": "https://openalex.org/W1", "doi": "https://doi.org/10.5555/ddpm", "display_name":
    "Denoising Diffusion Probabilistic Models", "publication_year": 2020,
    "authorships": [{"author": {"display_name": "Jonathan Ho"}}, {"author": {"display_name": "Ajay Jain"}}],
    "primary_location": {"source": {"display_name": "NeurIPS"}, "landing_page_url": "https://x/ddpm"},
    "cited_by_count": 9000, "abstract_inverted_index": {"We": [0], "present": [1], "diffusion": [2]},
    "locations": [{"landing_page_url": "https://arxiv.org/abs/2006.11239v2"}],
}
S2_PAPER = {"paperId": "abc", "title": "Denoising Diffusion Probabilistic Models", "year": 2020,
            "venue": "NeurIPS", "authors": [{"name": "Jonathan Ho"}], "citationCount": 9100,
            "externalIds": {"ArXiv": "2006.11239", "DOI": "10.5555/ddpm"}, "url": "https://s2/abc", "abstract": "We"}
ARXIV_FEED = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2207.12598v1</id><title>Classifier-Free Diffusion Guidance</title>
<summary>Guidance without a classifier.</summary><published>2022-07-26T00:00:00Z</published>
<author><name>Jonathan Ho</name></author><author><name>Tim Salimans</name></author></entry></feed>"""


def handler(request: httpx2.Request) -> httpx2.Response:
    url = str(request.url)
    if "api.openalex.org" in url:
        if "Nonexistent" in url or "nonexistent" in url:
            return httpx2.Response(200, json={"results": []})
        if "/works/doi:" in url:
            return httpx2.Response(200, json=OPENALEX_WORK)
        return httpx2.Response(200, json={"results": [OPENALEX_WORK]})
    if "semanticscholar" in url:
        if "Nonexistent" in url or "nonexistent" in url:
            return httpx2.Response(404, json={})
        if "/search" in url and "match" not in url:
            return httpx2.Response(200, json={"data": [S2_PAPER]})
        if "/search/match" in url:
            return httpx2.Response(200, json={"data": [S2_PAPER]})
        return httpx2.Response(200, json=S2_PAPER)
    if "arxiv.org" in url:
        return httpx2.Response(200, text=ARXIV_FEED)
    if "crossref" in url:
        return httpx2.Response(429, json={})
    return httpx2.Response(404)


@pytest.fixture
def search(tmp_path):
    s = LiteratureSearch(["openalex", "semantic_scholar", "arxiv", "crossref"], tmp_path / "cache",
                         transport=httpx2.MockTransport(handler))
    for gate in s.gates.values():
        gate.min_interval_s = 0
    yield s
    asyncio.run(s.aclose())


def test_search_merges_and_reports_provider_status(search, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    out = asyncio.run(search.search("diffusion models", limit=5))
    titles = [r.title for r in out["records"]]
    assert titles.count("Denoising Diffusion Probabilistic Models") == 1  # de-duplicated across providers
    ddpm = next(r for r in out["records"] if r.title.startswith("Denoising"))
    assert set(ddpm.sources) >= {"openalex", "semantic_scholar"}
    assert ddpm.arxiv_id == "2006.11239" and ddpm.doi == "10.5555/ddpm"
    assert "Classifier-Free Diffusion Guidance" in titles
    assert out["status"]["crossref"].startswith("unavailable")  # 429 after retries: reported, not fatal
    again = asyncio.run(search.search("diffusion models", limit=5))
    assert again["status"]["openalex"] == "cached"


async def _no_sleep(*_a, **_k):
    return None


def test_refcheck_statuses(search, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    refs = [
        {"title": "Denoising Diffusion Probabilistic Models", "year": 2020, "doi": "10.5555/ddpm"},
        {"title": "Denoising diffusion probabilistic model", "year": 2020},
        {"title": "A Nonexistent Paper About Nothing At All", "year": 2024},
        {"title": "Denoising Diffusion Probabilistic Models", "year": 2016, "arxiv_id": "2006.11239"},
    ]
    result = asyncio.run(check_references(refs, search))
    statuses = [i["status"] for i in result["items"]]
    assert statuses[0] == "verified"
    assert statuses[1] == "verified"  # near-identical title
    assert statuses[2] == "not_found"
    assert statuses[3] == "partial"  # found, but the cited year is wrong


def test_identifiers_and_similarity():
    assert parse_identifier("https://arxiv.org/abs/2106.09685v2") == ("arxiv", "2106.09685")
    assert parse_identifier("10.48550/arXiv.2106.09685") == ("arxiv", "2106.09685")
    assert parse_identifier("doi:10.1145/1234.5678") == ("doi", "10.1145/1234.5678")
    assert parse_identifier("Attention is all you need")[0] == "title"
    assert title_similarity("Attention Is All You Need", "attention is all you need.") > 0.95
    assert title_similarity("Attention Is All You Need", "Deep Residual Learning") < 0.5


def test_merge_keeps_provider_order():
    a = [PaperRecord("A", source="p1"), PaperRecord("B", source="p1")]
    b = [PaperRecord("C", source="p2"), PaperRecord("a", source="p2")]
    merged = merge_records([a, b])
    assert [r.title for r in merged] == ["A", "C", "B"]
    assert merged[0].sources == ["p1", "p2"]


def test_merge_prefers_trusted_fields():
    oa = PaperRecord("DDPM", arxiv_id="2006.11239", venue="arXiv (Cornell University)", abstract="WRONG genomics text",
                     source="openalex")
    ax = PaperRecord("DDPM", arxiv_id="2006.11239", venue="arXiv", abstract="We present diffusion models.", source="arxiv")
    s2 = PaperRecord("DDPM", arxiv_id="2006.11239", venue="NeurIPS", abstract="We present", source="semantic_scholar")
    merged = merge_records([[oa], [ax], [s2]])
    assert len(merged) == 1
    assert merged[0].abstract == "We present diffusion models."
    assert merged[0].venue == "NeurIPS"
