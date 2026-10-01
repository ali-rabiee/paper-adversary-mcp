"""The verbatim prior-text tools agents get: scoped access, verbatim reads, quote checks, no report reads."""

import asyncio
import builtins
from pathlib import Path

from conftest import mock_override
from paper_adversary import service
from paper_adversary.search.fulltext import FullTextResult, write_snapshot
from paper_adversary.store import RunStore
from paper_adversary.tools_server import build_server
from paper_adversary.util import runs_root, sha256_text

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


def _server(store, mode, scope=None):
    opts = {"providers": ["openalex"], "cache_root": str(runs_root() / ".cache"), "ttl_days": 14}
    return build_server(store.dir, "V9", store.dir / "logs" / "v9_tools.jsonl", False, False, opts, 1, mode, scope,
                        {"offline": True})


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
