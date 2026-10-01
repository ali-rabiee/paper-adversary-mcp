"""Full-text retrieval of prior work against canned HTTP responses: sources, host policy, caps, caching."""

import asyncio

import httpx2
import pymupdf
import pytest

from paper_adversary.config import FullTextConfig
from paper_adversary.search import fulltext
from paper_adversary.search.base import Breaker
from paper_adversary.search.extract import run_extraction
from paper_adversary.search.fulltext import (
    FullTextResult,
    FullTextStore,
    _host_ok,
    read_snapshot,
    snapshot_index,
    valid_arxiv_id,
    valid_doi,
    write_snapshot,
)

TITLE = "Rising Guidance Schedules for Diffusion Models"
BODY = ("We study classifier-free guidance whose weight increases monotonically over the sampling trajectory, "
        "and we show that a linear schedule from zero to the maximum weight improves sample quality.")


def pdf_bytes(title: str = TITLE, body: str = BODY) -> bytes:
    doc = pymupdf.open()
    for page_no in range(2):
        page = doc.new_page()
        if page_no == 0:
            page.insert_text((72, 80), title, fontsize=18)
        y = 130
        for line in (body * 3).split(". "):
            page.insert_text((72, y), line.strip()[:90] + ".", fontsize=10)
            y += 16
    doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


def work(doi: str, title: str = TITLE, oa: list[str] | None = None, arxiv: str | None = None) -> dict:
    locations = [{"pdf_url": u, "landing_page_url": u} for u in oa or []]
    if arxiv:
        locations.append({"landing_page_url": f"https://arxiv.org/abs/{arxiv}"})
    return {"id": "https://openalex.org/W9", "doi": f"https://doi.org/{doi}", "display_name": title,
            "publication_year": 2024, "authorships": [{"author": {"display_name": "A. Author"}}],
            "primary_location": {"source": {"display_name": "Venue"}}, "cited_by_count": 3,
            "abstract_inverted_index": {"Abstract": [0]}, "locations": locations,
            "best_oa_location": locations[0] if locations else None}


class Server:
    """A canned web: records every request, and answers from a route table."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.seen: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        url = str(request.url)
        self.seen.append(url)
        for prefix, answer in self.routes.items():
            if url.startswith(prefix):
                return answer(request) if callable(answer) else answer
        return httpx2.Response(404)


def make_store(tmp_path, server, prior=True, **cfg) -> FullTextStore:
    settings = FullTextConfig(arxiv_document_interval_seconds=0, host_interval_seconds=0, **cfg)
    return FullTextStore(tmp_path / "cache", settings, ["openalex"], prior_dir=tmp_path / "prior" if prior else None,
                         transport=httpx2.MockTransport(server))


@pytest.fixture(autouse=True)
def fast_gates(monkeypatch):
    monkeypatch.setattr(fulltext, "ARXIV_API_INTERVAL", 0)


def run(coro):
    return asyncio.run(coro)


async def _get(store, ident, hint=None):
    async with store:
        return await store.get(ident, hint)


def test_arxiv_pdf_when_html_is_missing_with_version_and_snapshot(tmp_path):
    server = Server({
        "https://api.openalex.org/works/doi:10.48550/arXiv.2401.00001": httpx2.Response(
            200, json=work("10.48550/arXiv.2401.00001", arxiv="2401.00001")),
        "https://arxiv.org/html/2401.00001": httpx2.Response(404),
        "https://arxiv.org/pdf/2401.00001v2": httpx2.Response(200, content=pdf_bytes(),
                                                               headers={"content-type": "application/pdf"}),
        "https://arxiv.org/pdf/2401.00001": httpx2.Response(302, headers={"location": "/pdf/2401.00001v2"}),
    })
    result = run(_get(make_store(tmp_path, server), "arXiv:2401.00001"))
    assert result.available and result.source == "arxiv_pdf" and result.version == "v2"
    assert "monotonically" in result.text_md and "<!-- page 2 -->" in result.text_md
    assert result.key == "arxiv:2401.00001"
    snap = read_snapshot(tmp_path / "prior", "2401.00001")
    assert snap is not None and snap.sha256 == result.sha256
    assert "arXiv:2401.00001" in snapshot_index(tmp_path / "prior")["arxiv:2401.00001"]["aliases"]
    # a second request is served from the cache without touching the network
    before = len(server.seen)
    again = run(_get(make_store(tmp_path, server, prior=False), "arXiv:2401.00001"))
    assert again.sha256 == result.sha256 and len(server.seen) == before


def test_open_access_pdf_only_from_allowed_hosts(tmp_path):
    allowed = "https://proceedings.mlr.press/v99/paper.pdf"
    server = Server({
        "https://api.openalex.org/works/doi:10.5555/ok": httpx2.Response(200, json=work("10.5555/ok", oa=[allowed])),
        "https://api.openalex.org/works/doi:10.5555/far": httpx2.Response(
            200, json=work("10.5555/far", oa=["https://files.example.com/paper.pdf"])),
        allowed: httpx2.Response(200, content=pdf_bytes(), headers={"content-type": "application/pdf"}),
        "https://files.example.com/": lambda r: pytest.fail("a disallowed host was contacted"),
    })
    ok = run(_get(make_store(tmp_path, server), "10.5555/ok"))
    assert ok.available and ok.source == "oa_pdf" and ok.key == "doi:10.5555/ok"
    far = run(_get(make_store(tmp_path, server), "10.5555/far"))
    assert far.status == "fulltext_unavailable" and far.reason == "host_not_allowed"
    assert "files.example.com" in far.detail and far.text_md is None
    assert far.abstract  # kept as metadata, never presented as the text


def test_redirects_are_host_checked_and_bodies_capped(tmp_path):
    server = Server({
        "https://api.openalex.org/works/doi:10.5555/hop": httpx2.Response(
            200, json=work("10.5555/hop", oa=["https://openreview.net/pdf?id=hop"])),
        "https://openreview.net/pdf?id=hop": httpx2.Response(302, headers={"location": "http://files.example.com/x"}),
        "https://api.openalex.org/works/doi:10.5555/big": httpx2.Response(
            200, json=work("10.5555/big", oa=["https://openreview.net/pdf?id=big"])),
        "https://openreview.net/pdf?id=big": httpx2.Response(
            200, content=b"%PDF-1.7\n" + b"0" * 3_000_000, headers={"content-type": "application/pdf"}),
        "https://api.openalex.org/works/doi:10.5555/html": httpx2.Response(
            200, json=work("10.5555/html", oa=["https://openreview.net/pdf?id=html"])),
        "https://openreview.net/pdf?id=html": httpx2.Response(200, text="<html>login</html>",
                                                              headers={"content-type": "text/html"}),
    })
    hop = run(_get(make_store(tmp_path, server), "10.5555/hop"))
    assert hop.reason in ("host_not_allowed", "download_failed") and hop.candidates[0]["reason"] == "host_not_allowed"
    big = run(_get(make_store(tmp_path, server, max_pdf_mb=1), "10.5555/big"))
    assert big.reason == "too_large"
    html = run(_get(make_store(tmp_path, server), "10.5555/html"))
    assert html.reason == "not_pdf"


def test_arxiv_block_pauses_arxiv_for_everyone(tmp_path):
    server = Server({
        "https://api.openalex.org/works/doi:10.48550/arXiv.2401.00002": httpx2.Response(
            200, json=work("10.48550/arXiv.2401.00002", arxiv="2401.00002")),
        "https://arxiv.org/": httpx2.Response(403),
    })
    result = run(_get(make_store(tmp_path, server), "arXiv:2401.00002"))
    assert result.status == "fulltext_unavailable" and result.reason == "provider_paused"
    assert Breaker(tmp_path / "cache" / "breaker").is_open("arxiv")  # the search backend shares this breaker
    calls = len(server.seen)
    again = run(_get(make_store(tmp_path, server), "arXiv:2401.00002"))
    assert again.reason == "provider_paused" and not any("arxiv.org/" in u for u in server.seen[calls:])


def test_wrong_paper_is_rejected(tmp_path):
    server = Server({
        "https://api.openalex.org/works/doi:10.5555/mix": httpx2.Response(
            200, json=work("10.5555/mix", oa=["https://aclanthology.org/mix.pdf"])),
        "https://aclanthology.org/mix.pdf": httpx2.Response(
            200, content=pdf_bytes(title="Completely Unrelated Survey of Graph Kernels"),
            headers={"content-type": "application/pdf"}),
    })
    result = run(_get(make_store(tmp_path, server), "10.5555/mix"))
    assert result.status == "fulltext_unavailable" and result.reason == "wrong_paper"


def test_offline_mode_never_touches_the_network(tmp_path):
    server = Server({"https://": lambda r: pytest.fail("network used while offline")})
    result = run(_get(make_store(tmp_path, server, offline=True), "arXiv:2401.00009"))
    assert result.status == "fulltext_unavailable" and result.reason == "offline_not_cached"


def test_user_supplied_files(tmp_path):
    store = make_store(tmp_path, Server({}))
    paper = tmp_path / "paywalled.pdf"
    paper.write_bytes(pdf_bytes())

    async def add(path):
        async with store:
            return await store.add_user_file("10.1109/paywalled", path, TITLE)

    result = run(add(paper))
    assert result.available and result.source == "user_supplied" and result.key == "doi:10.1109/paywalled"
    assert read_snapshot(tmp_path / "prior", "10.1109/paywalled").available
    hidden = tmp_path / ".secret" / "x.pdf"
    hidden.parent.mkdir()
    hidden.write_bytes(pdf_bytes())
    with pytest.raises(ValueError, match="hidden"):
        run(add(hidden))
    text = tmp_path / "notes.pdf"
    text.write_text("CLAUDE_CODE_OAUTH_TOKEN=whatever")
    with pytest.raises(ValueError, match="neither a PDF"):
        run(add(text))


def test_user_files_must_be_the_cited_paper_and_never_replace_a_download(tmp_path):
    store = make_store(tmp_path, Server({}), offline=True)
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(pdf_bytes())

    async def add(ident, title=None, submission=None):
        async with store:
            return await store.add_user_file(ident, paper, title, submission)

    with pytest.raises(ValueError, match="arXiv ID or DOI"):
        run(add("Rising Guidance Schedules for Diffusion Models", TITLE))
    with pytest.raises(ValueError, match="right paper"):
        run(add("10.1109/other", "A Survey of Graph Kernels for Chemistry"))
    own = run_extraction(paper, "pdf")["text_md"]  # the submission's own preprint, filed as prior work
    with pytest.raises(ValueError, match="submission itself"):
        run(add("10.1109/mine", TITLE, submission=own))
    write_snapshot(tmp_path / "prior", FullTextResult(
        "available", key="doi:10.1109/downloaded", title=TITLE, source="oa_pdf", sha256="ab" * 32, text_md=BODY,
        sections=[], aliases=["10.1109/downloaded"]))
    with pytest.raises(ValueError, match="never replaces"):
        run(add("10.1109/downloaded", TITLE))
    result = run(add("10.1109/fine", TITLE))
    assert result.available and result.url is None
    assert read_snapshot(tmp_path / "prior", "10.1109/fine").available
    machine_only = make_store(tmp_path, Server({}), prior=False, offline=True)
    assert machine_only.cached("10.1109/fine") is None  # the run snapshot only, never the shared cache


@pytest.mark.parametrize("url, ok", [
    ("https://arxiv.org/pdf/2401.00001", True),
    ("https://aclanthology.org/2024.acl-long.1.pdf", True),
    ("http://arxiv.org/pdf/2401.00001", False),
    ("https://arxiv.org:8443/pdf/2401.00001", False),
    ("https://arxiv.org/pdf/x\r\nHost: evil.example", False),
    ("https://[::1/pdf", False),
    ("https://evil.example/arxiv.org/pdf", False),
    ("https://arxiv.org.evil.example/pdf", False),
    ("file:///etc/passwd", False),
    (None, False),
])
def test_metadata_urls_are_validated(url, ok):
    assert _host_ok(url, "allowlist", []) is ok


def test_identifiers_are_validated():
    assert valid_arxiv_id("2401.00001v2") and valid_arxiv_id("hep-th/9901001")
    assert not valid_arxiv_id("../../etc/passwd") and not valid_arxiv_id("2401.00001/../x")
    assert valid_doi("10.1109/5.771073") and not valid_doi("10.1109/../../x") and not valid_doi("x" * 300)
