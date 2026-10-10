"""Scholarly search backends, merging, and reference checking, against canned HTTP responses."""

import asyncio

import httpx2
import pytest

from paper_adversary.util import atomic_write_json, read_json

from paper_adversary.search import LiteratureSearch, merge_records, provider_health, retry_after_seconds
from paper_adversary.search.base import Breaker, PaperRecord, parse_identifier
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



class S2Server:
    """Semantic Scholar answering HTTP 429 for its first `busy` requests, then normally."""

    def __init__(self, busy: int, retry_after: str | None = None):
        self.busy, self.calls, self.retry_after = busy, 0, retry_after

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.calls += 1
        if self.calls <= self.busy:
            return httpx2.Response(429, headers={"retry-after": self.retry_after} if self.retry_after else {})
        return httpx2.Response(200, json={"data": [S2_PAPER]})


def _s2(tmp_path, server, monkeypatch, key=None):
    if key:
        monkeypatch.setenv("S2_API_KEY", key)
    else:
        monkeypatch.delenv("S2_API_KEY", raising=False)
    s = LiteratureSearch(["semantic_scholar"], tmp_path / "cache", transport=httpx2.MockTransport(server))
    for gate in s.gates.values():
        gate.min_interval_s = 0
    return s


def test_keyless_semantic_scholar_backs_off_longer_before_pausing(tmp_path, monkeypatch):
    """K1 pilot (F29): without a key, three 429s within 9 s paused Semantic Scholar for 600 s, longer than a novelty
    refuter searches. Now a keyless 429 burst gets four jittered retries (about a minute) before any pause."""
    delays = []

    async def sleep(seconds, *_a, **_k):
        delays.append(seconds)
    monkeypatch.setattr("asyncio.sleep", sleep)
    server = S2Server(busy=4)
    search = _s2(tmp_path, server, monkeypatch)
    out = asyncio.run(search.search("diffusion models", limit=5))
    asyncio.run(search.aclose())
    assert out["status"]["semantic_scholar"] == "ok" and server.calls == 5
    assert len(delays) == 4 and 3 <= delays[0] <= 5 and 24 <= delays[3] <= 40 and sum(delays) >= 45
    assert not Breaker(tmp_path / "cache" / "breaker").is_open("semantic_scholar")


def test_keyless_semantic_scholar_pauses_briefly_then_lets_one_probe_through(tmp_path, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    server = S2Server(busy=5)
    search = _s2(tmp_path, server, monkeypatch)
    out = asyncio.run(search.search("diffusion models", limit=5))
    assert out["status"]["semantic_scholar"] == "unavailable (semantic_scholar kept answering HTTP 429)"
    breaker = Breaker(tmp_path / "cache" / "breaker")
    pause = breaker.open_until("semantic_scholar") - __import__("time").time()
    assert 60 <= pause <= 120 and read_json(tmp_path / "cache" / "breaker" / "semantic_scholar.json")["half_open"]
    paused = asyncio.run(search.search("another query", limit=5))["status"]["semantic_scholar"]
    assert paused.startswith("unavailable (semantic_scholar is paused for") and "more s" in paused
    data = read_json(tmp_path / "cache" / "breaker" / "semantic_scholar.json")  # the pause is over
    atomic_write_json(tmp_path / "cache" / "breaker" / "semantic_scholar.json", {**data, "until": 0})
    assert breaker.admit("semantic_scholar") == "probe" and breaker.admit("semantic_scholar") is None  # one probe
    atomic_write_json(tmp_path / "cache" / "breaker" / "semantic_scholar.json", {**data, "until": 0})
    probe = asyncio.run(search.search("a third query", limit=5))["status"]["semantic_scholar"]
    asyncio.run(search.aclose())
    assert probe == "ok" and breaker.admit("semantic_scholar") == "closed"  # the probe's success closes it


def test_a_failed_half_open_probe_makes_one_request_and_pauses_again(tmp_path, monkeypatch):
    """Review (G4): past the pause, the one probe request that a 429 answers re-trips the breaker, half-open, for
    another 60-120 s; nothing else gets through meanwhile."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    server = S2Server(busy=100)
    search = _s2(tmp_path, server, monkeypatch)
    asyncio.run(search.search("diffusion models", limit=5))
    path = tmp_path / "cache" / "breaker" / "semantic_scholar.json"
    atomic_write_json(path, {**read_json(path), "until": 0})  # the pause is over
    calls = server.calls
    probe = asyncio.run(search.search("a second query", limit=5))["status"]["semantic_scholar"]
    asyncio.run(search.aclose())
    assert probe.startswith("unavailable") and server.calls == calls + 1  # the probe alone, no retries
    breaker = Breaker(tmp_path / "cache" / "breaker")
    pause = breaker.open_until("semantic_scholar") - __import__("time").time()
    assert 60 <= pause <= 120 and read_json(path)["half_open"] and breaker.admit("semantic_scholar") is None


def test_semantic_scholar_with_a_key_pauses_briefly_then_probes(tmp_path, monkeypatch):
    """With a key, a 429 that outlasts 3 retries pauses Semantic Scholar for 120 s, then one probe goes through (in the
    K2 run on 2026-10-02 the earlier 600 s pause outlasted the whole novelty phase)."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    server = S2Server(busy=10)
    search = _s2(tmp_path, server, monkeypatch, key="a-key")
    asyncio.run(search.search("diffusion models", limit=5))
    asyncio.run(search.aclose())
    breaker = Breaker(tmp_path / "cache" / "breaker")
    left = breaker.open_until("semantic_scholar") - __import__("time").time()
    assert server.calls == 4 and 110 < left <= 120  # 1 + 3 retries, then a short pause
    assert breaker.admit("semantic_scholar") is None


def test_semantic_scholar_with_a_key_rides_out_a_single_429(tmp_path, monkeypatch):
    """With a key, a lone 429 near the 1 request/s limit (seen in a live test on 2026-10-02) is retried, not a pause."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    server = S2Server(busy=1)
    search = _s2(tmp_path, server, monkeypatch, key="a-key")
    asyncio.run(search.search("diffusion models", limit=5))
    asyncio.run(search.aclose())
    breaker = Breaker(tmp_path / "cache" / "breaker")
    assert server.calls == 2 and breaker.admit("semantic_scholar") is not None


def test_retry_after_is_honoured_in_both_forms(tmp_path, monkeypatch):
    assert retry_after_seconds("12") == 12.0 and retry_after_seconds("") is None and retry_after_seconds("soon") is None
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:30 GMT", now=1445412480.0) == 30.0  # an HTTP date
    delays = []

    async def sleep(seconds, *_a, **_k):
        delays.append(seconds)
    monkeypatch.setattr("asyncio.sleep", sleep)
    search = _s2(tmp_path, S2Server(busy=1, retry_after="Wed, 21 Oct 2099 07:28:00 GMT"), monkeypatch)
    asyncio.run(search.search("diffusion models", limit=5))
    asyncio.run(search.aclose())
    assert delays == [60]  # capped


def test_provider_health_counts_unavailable_calls_per_provider():
    rows = [{"tool": "search_literature", "at": "2026-10-02T03:14:30+00:00", "status": {
                "semantic_scholar": "unavailable (semantic_scholar kept answering HTTP 429)", "openalex": "ok"}},
            {"tool": "find_citing_papers", "at": "2026-10-02T03:15:00+00:00", "status": {
                "semantic_scholar": "unavailable (semantic_scholar is paused for 500 more s ...)",
                "arxiv": "not supported", "openalex": "cached"}},
            {"tool": "lookup_paper", "at": "2026-10-02T03:16:00+00:00", "status": {"semantic_scholar": "ok"}},
            {"tool": "read_prior_paper", "status": "fulltext_unavailable"}]
    health = provider_health(rows)
    assert health["semantic_scholar"] == {"calls": 3, "answered": 1, "unavailable": 2,
                                          "first_unavailable": "2026-10-02T03:14:30+00:00",
                                          "last_unavailable": "2026-10-02T03:15:00+00:00"}
    assert health["openalex"]["answered"] == 2 and "arxiv" not in health


class OpenAlexServer:
    """OpenAlex answering normally, or HTTP 429 with credit headers saying the daily budget is spent."""

    def __init__(self, spent: bool = False):
        self.spent, self.calls, self.params = spent, 0, []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.calls += 1
        self.params.append(dict(request.url.params))
        if self.spent:
            return httpx2.Response(429, headers={"x-ratelimit-remaining": "0", "x-ratelimit-remaining-usd": "0",
                                                 "x-ratelimit-reset": "7200"})
        return httpx2.Response(200, json={"results": [OPENALEX_WORK]})


def _openalex(tmp_path, server, monkeypatch, key=None):
    if key:
        monkeypatch.setenv("OPENALEX_API_KEY", key)
    else:
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    s = LiteratureSearch(["openalex"], tmp_path / "cache", transport=httpx2.MockTransport(server))
    for gate in s.gates.values():
        gate.min_interval_s = 0
    return s


def test_openalex_requests_carry_the_api_key(tmp_path, monkeypatch):
    """Since February 2026 OpenAlex bills a daily budget per key; requests carry OPENALEX_API_KEY as api_key."""
    server = OpenAlexServer()
    search = _openalex(tmp_path, server, monkeypatch, key="oa-key")
    asyncio.run(search.search("diffusion models", limit=5))
    asyncio.run(search.aclose())
    assert server.calls == 1 and server.params[0].get("api_key") == "oa-key"
    keyless = OpenAlexServer()
    search = _openalex(tmp_path / "b", keyless, monkeypatch)
    asyncio.run(search.search("diffusion models", limit=5))
    asyncio.run(search.aclose())
    assert "api_key" not in keyless.params[0]


def test_a_spent_daily_budget_pauses_until_its_reset_without_retrying(tmp_path, monkeypatch):
    """K2 run (2026-10-02): OpenAlex's keyless daily budget ran out. A 429 whose credit headers say the budget is
    spent is not retried; OpenAlex is paused until the reset the headers give."""
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    server = OpenAlexServer(spent=True)
    search = _openalex(tmp_path, server, monkeypatch)
    out = asyncio.run(search.search("diffusion models", limit=5))
    asyncio.run(search.aclose())
    breaker = Breaker(tmp_path / "cache" / "breaker")
    left = breaker.open_until("openalex") - __import__("time").time()
    assert server.calls == 1 and 7100 < left <= 7200
    assert "daily budget" in out["status"]["openalex"]
