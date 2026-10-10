"""The verbatim prior-text tools agents get: scoped access, verbatim reads, quote checks, no report reads."""

import asyncio
import builtins
from pathlib import Path

from conftest import mock_override
from paper_adversary import service
from paper_adversary.search.fulltext import FullTextResult, write_snapshot
from paper_adversary.store import RunStore
from paper_adversary.tools_server import build_server
from paper_adversary.util import read_jsonl, runs_root, sha256_text

PRIOR = """# Analysis of Guidance Weight Schedulers

<!-- page 1 -->
## 1 Introduction

We find that monotonically increasing guidance schedules, including a simple linear ramp, improve sample quality.

<!-- page 2 -->
## 2 Method

The schedule is applied at sampling time and requires no retraining of the underlying diffusion model.
"""


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


def _server(store, mode, scope=None, passages=None):
    opts = {"providers": ["openalex"], "cache_root": str(runs_root() / ".cache"), "ttl_days": 14}
    return build_server(store.dir, "V9", store.dir / "logs" / "v9_tools.jsonl", False, False, opts, 1, mode, scope,
                        {"offline": True}, passages)


def _call(server, name, **args):
    return _text(asyncio.run(server.call_tool(name, args)))


def _run(runs_dir, sample_paper):
    info = service.create_run(str(sample_paper), config_override=mock_override())
    store = RunStore.open(info["run_id"])
    write_snapshot(store.prior_dir, FullTextResult("available", key="arxiv:2401.00001", title="Analysis of Guidance "
                                                   "Weight Schedulers", arxiv_id="2401.00001", source="arxiv_pdf",
                                                   version="v1", sha256=sha256_text(PRIOR), text_md=PRIOR,
                                                   aliases=["2401.00001"]))
    return store


def test_scoped_tools_only_serve_the_assigned_paper(runs_dir, sample_paper):
    store = _run(runs_dir, sample_paper)
    server = _server(store, "scoped", ["arxiv:2401.00001"])
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"read_prior_paper", "find_in_prior_paper", "check_quote"}
    out = _call(server, "read_prior_paper", identifier="2401.00001")
    assert "VERBATIM TEXT" in out and "monotonically increasing guidance schedules" in out
    assert "not one of the papers you were given" in _call(server, "read_prior_paper", identifier="1706.03762")
    page = _call(server, "read_prior_paper", identifier="arxiv:2401.00001", page=2)
    assert "requires no retraining" in page and "monotonically" not in page
    hits = _call(server, "find_in_prior_paper", identifier="2401.00001", query="linear ramp schedule")
    assert "linear ramp" in hits and "p. 1" in hits


def test_check_quote_against_prior_and_submission(runs_dir, sample_paper):
    store = _run(runs_dir, sample_paper)
    server = _server(store, "scoped", ["arxiv:2401.00001"])
    ok = _call(server, "check_quote", identifier="2401.00001",
               passage="monotonically increasing guidance schedules, including a simple linear ramp, improve sample")
    assert ok.startswith("VERIFIED")
    flipped = _call(server, "check_quote", identifier="2401.00001",
                    passage="monotonically decreasing guidance schedules, including a simple linear ramp, improve")
    assert "would not count as a verbatim quote" in flipped
    words = next(line for line in (store.source_dir / "extracted_text.md").read_text().splitlines()
                 if len(line.split()) > 14 and not line.startswith("#")).split()[:12]
    assert _call(server, "check_quote", passage=" ".join(words)).startswith("VERIFIED in the submission")


def test_tools_never_read_report_folders(runs_dir, sample_paper, monkeypatch):
    store = _run(runs_dir, sample_paper)
    server = _server(store, "open")
    opened: list[str] = []
    real_open, real_read = builtins.open, Path.read_text

    def spy_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def spy_read(self, *a, **k):
        opened.append(str(self))
        return real_read(self, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(Path, "read_text", spy_read)
    _call(server, "read_prior_paper", identifier="2401.00001")
    _call(server, "check_quote", passage="anything at all to look for in the submission text")
    _call(server, "read_prior_paper", identifier="arXiv:2499.99999")  # offline and not cached
    forbidden = ("/novelty/", "/rigor/", "/fit/", "/judges/", "/synthesis/", "/critic/", "/verify/", "/followup/")
    assert opened and not [p for p in opened if any(f in p for f in forbidden)]


def test_check_quote_tells_a_verifier_when_a_quote_lies_outside_its_passages(runs_dir, sample_paper):
    """K1 pilot (F5): 4 of 10 verifier overlap quotes were rejected for lying outside the passage under examination,
    while check_quote had called each of them verified (it matched the whole submission)."""
    store = _run(runs_dir, sample_paper)
    passage = ("We claim that a learned, instance-conditioned noise editor achieves the quality of per-query "
               "optimization at the cost of a single forward pass")
    server = _server(store, "scoped", ["arxiv:2401.00001"],
                     {"passages": [{"pid": "P1", "text": passage}], "max_words": 80, "accept_approximate_score": 1.01})
    inside = _call(server, "check_quote", passage="a learned, instance-conditioned noise editor achieves the quality")
    assert inside.startswith("VERIFIED in the submission") and "Inside your passage(s) P1" in inside
    outside = _call(server, "check_quote", passage="Frozen generative policies often need small corrections at "
                                                   "deployment time.")
    assert outside.startswith("VERIFIED in the submission") and "NOT INSIDE ANY OF YOUR PASSAGES (P1)" in outside
    prior = _call(server, "check_quote", identifier="2401.00001",
                  passage="monotonically increasing guidance schedules, including a simple linear ramp, improve sample")
    assert "YOUR PASSAGES" not in prior and "Inside your passage" not in prior  # prior-paper checks are unchanged


def test_every_prior_text_call_gets_a_search_log_row(runs_dir, sample_paper):
    """K1 pilot (F26): N1 made 11 find_in_prior_paper calls and its search log had 9; the missing ones had returned
    'FULL TEXT UNAVAILABLE'. Refusals and texts that are unavailable are logged too, with the usual arguments."""
    store = _run(runs_dir, sample_paper)
    log = store.dir / "logs" / "v9_tools.jsonl"
    scoped = _server(store, "scoped", ["arxiv:2401.00001"])
    for name, args in (("read_prior_paper", {"identifier": "1706.03762"}),
                       ("find_in_prior_paper", {"identifier": "1706.03762", "query": "attention"}),
                       ("check_quote", {"identifier": "1706.03762", "passage": "attention is all you need"}),
                       ("read_prior_paper", {"identifier": "2401.00001", "section": "S99"})):
        _call(scoped, name, **args)
    opened = _server(store, "open")  # offline: nothing can be fetched
    for name, args in (("find_in_prior_paper", {"identifier": "arXiv:2499.99999", "query": "schedule"}),
                       ("check_quote", {"identifier": "arXiv:2499.99999", "passage": "a quote to check here"})):
        assert "UNAVAILABLE" in _call(opened, name, **args).upper()
    rows = read_jsonl(log)
    assert [r["tool"] for r in rows] == ["read_prior_paper", "find_in_prior_paper", "check_quote", "read_prior_paper",
                                         "find_in_prior_paper", "check_quote"]
    assert [r["status"] for r in rows[:3]] == ["refused"] * 3 and "not one of the papers" in rows[0]["reason"]
    assert rows[1]["args"] == {"identifier": "1706.03762", "query": "attention"}
    assert rows[2]["args"] == {"identifier": "1706.03762", "chars": len("attention is all you need")}
    assert rows[3]["error"] == "no such section" and rows[3]["args"]["section"] == "S99"
    assert all(r["status"] != "available" for r in rows[4:]) and rows[4]["args"]["query"] == "schedule"
