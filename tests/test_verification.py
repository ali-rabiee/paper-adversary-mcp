"""Verified novelty evidence end to end (mock provider, offline full texts): quote checks, blind verifiers,
the evidence gate, and the memo's unverified-threats section."""

import asyncio
import json
from pathlib import Path

import yaml

from conftest import mock_override
from paper_adversary import followup, gates, service
from paper_adversary.config import FullTextConfig
from paper_adversary.pipeline import Pipeline
from paper_adversary.reports import read_report
from paper_adversary.search.fulltext import FullTextResult, FullTextStore
from paper_adversary.store import RunStore
from paper_adversary.util import read_json, read_jsonl, runs_root, sha256_file, sha256_text
from paper_adversary.ingest import ingest_markdown
from paper_adversary.passages import SourceText
from paper_adversary.verification import _locate, pending_requests

ALL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]
TITLE = "Analysis of Guidance Weight Schedulers"
PRIOR_TEXT = """# Analysis of Guidance Weight Schedulers

## Abstract

<!-- anchor S1.p1 -->
Recent works vary the guidance weight throughout the diffusion process, and we study why this helps.

## 3 Schedulers

<!-- anchor S3.p1 -->
We find that monotonically increasing guidance schedules, including a simple linear ramp from zero to the maximum weight, consistently improve sample quality without any retraining of the model.

<!-- anchor S3.p2 -->
Decreasing schedules, by contrast, tend to reduce diversity and harm the quality of the generated samples overall.
"""
PRIOR_QUOTE = ("monotonically increasing guidance schedules, including a simple linear ramp from zero to the maximum "
               "weight, consistently improve sample quality")


def seed_prior(key="arxiv:2401.00001", ident="2401.00001", text=PRIOR_TEXT, title=TITLE):
    store = FullTextStore(runs_root() / ".cache", FullTextConfig(offline=True), ["openalex"])
    store._save(FullTextResult("available", key=key, title=title, arxiv_id=ident, source="arxiv_html", version="v2",
                               sha256=sha256_text(text), text_md=text, sections=[], fetched_at="2026-10-01"), ident)


def _create(sample_paper, **mock):
    info = service.create_run(str(sample_paper), venue="ICML 2027", config_override=mock_override(**mock))
    return RunStore.open(info["run_id"])


def _run(store, stages=ALL, **kw):
    pipeline = Pipeline(store)
    return asyncio.run(pipeline.run(stages, **kw)), pipeline


DECISIVE = {
    "N1": {"category": "already_done", "severity": "fatal", "basis": "full_text",
           "prior": {"title": TITLE, "arxiv_id": "2401.00001"}, "prior_passage": PRIOR_QUOTE},
    # the smoke run's N1-O1: a FATAL overlap claim made from the abstract alone, on a paper we cannot read
    "N2": {"category": "already_done", "severity": "fatal", "basis": "abstract_only",
           "prior": {"title": "Masked Diffusion Transformer", "arxiv_id": "2303.14389"}},
}


def test_verified_quotes_blind_verifier_and_the_evidence_gate(runs_dir, sample_paper):
    seed_prior()
    store = _create(sample_paper, decisive=DECISIVE)
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    agents = store.load_state()["agents"]

    ev1 = read_json(store.sidecar_path("N1", "novelty", ".evidence.json"))["objections"]["N1-O1"]
    assert ev1["decisive"] and ev1["status"] == "quotes_verified"
    assert ev1["pairs"][0]["prior_passage"]["status"] == "verified"
    ev2 = read_json(store.sidecar_path("N2", "novelty", ".evidence.json"))["objections"]["N2-O1"]
    assert ev2["status"] == "fulltext_unavailable"
    assert read_json(store.sidecar_path("N3", "novelty", ".evidence.json"))["objections"]["N3-O2"]["status"] == \
        "not_required"  # minor: not decisive

    # one blind verifier for the one readable prior paper; none for the unreadable one
    verifiers = sorted(a for a, v in agents.items() if v["role"] == "verifier")
    assert verifiers == ["V1"] and agents["V1"]["status"] == "complete"
    batch = store.load_state()["verification"]["batches"]["refuters"]
    assert batch["status"] == "complete" and batch["agents"] == ["V1"]
    results = read_json(store.role_dir("verifier") / "results.json")
    assert results["requests"]["N1-O1:r1"]["status"] == "verified"
    assert results["requests"]["N1-O1:r1"]["verdict"] == "anticipates_partially"
    assert results["requests"]["N2-O1:r1"]["status"] == "fulltext_unavailable"

    # blindness: the verifier saw neither any agent's report nor the objection's argument or IDs
    prompt = (store.agent_log_dir("V1") / "attempt-1" / "user_prompt.md").read_text()
    assert "CANARY_" not in prompt and "Mock argument" not in prompt and "N1-O1" not in prompt
    assert PRIOR_QUOTE in prompt and "<passages_under_examination>" in prompt
    _, v1 = read_report(store.report_path("V1", "verifier"))
    assert "Inputs seen: none" in v1

    # judges see the quote checks and the independent check; the gate labels what is not shown
    judge_prompt = (store.agent_log_dir("J1") / "attempt-1" / "user_prompt.md").read_text()
    assert "quotes VERIFIED" in judge_prompt and "<independent_verifications" in judge_prompt
    assert "V1 says anticipates_partially" in judge_prompt
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    by = {(v["judge"], tuple(v["objection_ids"])): v for v in gate["verdicts"]}
    assert by[("J1", ("N1-O1",))]["passes"]  # MAJOR: verified quotes + partial anticipation
    assert by[("J2", ("N1-O1",))]["label"].startswith("SCOPE")  # FATAL needs full anticipation
    assert by[("J1", ("N2-O1",))]["label"].startswith("UNVERIFIED: full text unavailable")
    assert {"N1-O1", "N2-O1"} <= set(gate["flagged_ids"])  # (N3-O1, N4-O1 cite an unreadable paper too)
    matrix = service.get_report(store.run_id, "matrix")
    assert "| Evidence |" in matrix and "quotes verified; V1: partially" in matrix

    # the memo files the flagged verdicts under "Unverified threats"; the placement check passes
    s1 = agents["S1"]
    assert s1["status"] == "complete" and not any("unverified" in w.lower() for w in s1["gate"]["warnings"])
    synthesis_prompt = (store.agent_log_dir("S1") / "attempt-1" / "user_prompt.md").read_text()
    assert "<evidence_gate" in synthesis_prompt


def test_misplaced_unverified_threats_are_sent_back(runs_dir, sample_paper):
    store = _create(sample_paper, decisive={"N2": DECISIVE["N2"]}, misplace={"S1": True})
    _run(store)
    s1 = store.load_state()["agents"]["S1"]
    assert s1["status"] == "complete"
    assert any("rewritten by a placement fix" in w and "N2-O1" in w for w in s1["gate"]["warnings"])
    assert read_json(store.gate_path("S1", "synthesis"))["placement_fix"]["outcome"] == "accepted"
    meta, body = read_report(store.report_path("S1", "synthesis"))
    assert "N2-O1" in gates.section_text(body, 6) and "N2-O1" not in gates.lead_ids(gates.section_text(body, 5))
    assert meta["orchestrator_edits"] and "CANARY_S1" in body  # the rest of the memo is untouched
    assert any("placement fix" in e.get("reason", "") for e in read_jsonl(store.dir / "archive" / "index.jsonl"))
    critic_prompt = (store.dir / "logs/agents/C1/attempt-1/user_prompt.md").read_text()
    assert "Mock (fixed)." in critic_prompt  # later stages read the fixed memo
    for mode in ("unchanged", "renamed", "invent", "fail:invalid_request"):
        store = _create(sample_paper, decisive={"N2": DECISIVE["N2"]}, misplace={"S1": True},
                        placement_fix={"S1": mode})
        result, _ = _run(store)
        s1 = store.load_state()["agents"]["S1"]
        assert s1["status"] == "quarantined" and "surviving criticisms" in s1["detail"], mode
        assert result["outcome"] == "quarantined", mode


def test_no_decisive_objections_skips_verification(runs_dir, sample_paper):
    store = _create(sample_paper, references={f"N{i}": [] for i in range(1, 5)})
    result, _ = _run(store)
    assert result["outcome"] == "complete"
    state = store.load_state()
    assert state["verification"]["batches"]["refuters"]["status"] == "skipped"
    assert not any(a["role"] == "verifier" for a in state["agents"].values())
    assert read_json(store.role_dir("judge") / "evidence_gate.json")["flagged_ids"] == []


def test_judges_alone_run_verification_first(runs_dir, sample_paper):
    seed_prior()
    store = _create(sample_paper, decisive={"N1": DECISIVE["N1"]})
    _run(store, ["intake", "novelty", "rigor", "fit"])
    assert "verifier" not in {a["role"] for a in store.load_state()["agents"].values()}
    result, _ = _run(store, ["judge"])
    assert result["outcome"] == "complete", result
    agents = store.load_state()["agents"]
    assert agents["V1"]["status"] == "complete" and agents["J1"]["status"] == "complete"


def test_verifier_disputes_and_unverifiable_quotes(runs_dir, sample_paper):
    seed_prior()
    store = _create(sample_paper, decisive={"N1": DECISIVE["N1"]}, verdict={"default": "does_not_anticipate"})
    _run(store)
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    labels = {v["label"] for v in gate["verdicts"] if v["objection_ids"] == ["N1-O1"]}
    assert labels == {"DISPUTED by the blind verifier"}


def test_old_runs_without_verification_still_work(runs_dir, sample_paper, tmp_path, monkeypatch):
    standing = tmp_path / "standing.yaml"  # turning the verifier off is a standing choice, never a per-run one
    standing.write_text("verifier:\n  enabled: false\n")
    monkeypatch.setenv("PAPER_ADVERSARY_CONFIG", str(standing))
    info = service.create_run(str(sample_paper), config_override=mock_override())
    store = RunStore.open(info["run_id"])
    assert "verification" not in store.load_state()
    result, _ = _run(store)
    assert result["outcome"] == "complete", result


def test_a_supplied_full_text_resolves_a_flag_on_recheck(runs_dir, sample_paper):
    store = _create(sample_paper, decisive={"N2": DECISIVE["N2"]})
    _run(store)
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert "N2-O1" in gate["flagged_ids"]
    # the user supplies the paywalled paper; a gate re-check then verifies it without rerunning the judges
    seed_prior(key="arxiv:2303.14389", ident="2303.14389", title="Masked Diffusion Transformer")
    batch_id, n = service.verification_batch(store.run_id, True, None)
    assert batch_id == "gate1" and n >= 1
    requested = read_json(store.role_dir("verifier") / "batches" / "gate1.requests.json")
    assert "N2-O1:r1" in {r["request_id"] for r in requested}
    judges_before = {a: v["attempts"] for a, v in store.load_state()["agents"].items() if v["role"] == "judge"}
    result, _ = _run(store, [f"verify:{batch_id}"])
    assert result["outcome"] == "complete", result
    state = store.load_state()
    assert state["verification"]["batches"]["gate1"]["status"] == "complete"
    assert {a: v["attempts"] for a, v in state["agents"].items() if v["role"] == "judge"} == judges_before
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    majors = [v for v in gate["verdicts"] if v["objection_ids"] == ["N2-O1"] and v["severity"] == "MAJOR_FIXABLE"]
    assert majors and all(v["passes"] for v in majors)  # verified partial anticipation shows a MAJOR verdict
    fatal = [v for v in gate["verdicts"] if v["objection_ids"] == ["N2-O1"] and v["severity"] == "FATAL"]
    assert fatal and all(v["label"].startswith("SCOPE") for v in fatal)


def test_a_cited_paper_with_another_title_is_a_mismatch(runs_dir, sample_paper):
    seed_prior()  # 2401.00001 is "Analysis of Guidance Weight Schedulers"
    wrong = {**DECISIVE["N1"], "prior": {"title": "Completely Different Work on Graph Kernels",
                                         "arxiv_id": "2401.00001"}}
    store = _create(sample_paper, decisive={"N1": wrong})
    _run(store)
    ev = read_json(store.sidecar_path("N1", "novelty", ".evidence.json"))["objections"]["N1-O1"]
    assert ev["status"] == "reference_mismatch"
    results = read_json(store.role_dir("verifier") / "results.json")
    assert results["requests"]["N1-O1:r1"]["status"] == "reference_mismatch"
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert any(v["label"].startswith("REFERENCE MISMATCH") for v in gate["verdicts"] if v["objection_ids"] == ["N1-O1"])


def test_the_submission_itself_is_not_prior_work(runs_dir, sample_paper):
    own = Path(sample_paper).read_text()
    seed_prior(key="arxiv:2409.99999", ident="2409.99999", text=own, title="Some Preprint Title")
    words = " ".join(next(line for line in own.splitlines() if len(line.split()) > 20).split()[:14])
    selfie = {"category": "already_done", "severity": "fatal", "basis": "full_text",
              "prior": {"title": "Some Preprint Title", "arxiv_id": "2409.99999"}, "prior_passage": words}
    store = _create(sample_paper, decisive={"N1": selfie})
    _run(store)
    ev = read_json(store.sidecar_path("N1", "novelty", ".evidence.json"))["objections"]["N1-O1"]
    assert ev["status"] == "prior_is_submission"
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert any("is the submission itself" in (v["label"] or "") for v in gate["verdicts"]
               if v["objection_ids"] == ["N1-O1"])


def _set_mock(store, **mock):
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"].update(mock)
    store.config_path.write_text(yaml.safe_dump(raw))


def test_a_rerun_refuter_gets_its_objections_verified_again(runs_dir, sample_paper):
    seed_prior()
    store = _create(sample_paper, decisive={"N1": DECISIVE["N1"]})
    _run(store)
    first = read_json(store.role_dir("verifier") / "results.json")["requests"]["N1-O1:r1"]
    # the rerun calls the overlap MAJOR: the same ID now names a different objection
    _set_mock(store, decisive={"N1": {**DECISIVE["N1"], "severity": "major"}})
    _run(store, ["novelty"], rerun=["N1"])
    assert [r.request_id for r in pending_requests(store, store.load_state())] == ["N1-O1:r1"]
    result, _ = _run(store, ["judge"])
    assert result["outcome"] == "complete", result
    state = store.load_state()
    assert state["verification"]["batches"]["refuters2"]["status"] == "complete"
    again = read_json(store.role_dir("verifier") / "results.json")["requests"]["N1-O1:r1"]
    assert again["batch"] == "refuters2" and again["origin_hash"] != first["origin_hash"]
    assert not pending_requests(store, state)


def test_a_released_verifier_counts(runs_dir, sample_paper):
    seed_prior()
    store = _create(sample_paper, decisive={"N1": DECISIVE["N1"]}, served_model={"V1": "claude-haiku-4-5"})
    _run(store)
    state = store.load_state()
    assert state["agents"]["V1"]["status"] == "quarantined"
    assert state["verification"]["batches"]["refuters"]["status"] == "incomplete"
    results = read_json(store.role_dir("verifier") / "results.json")
    assert results["requests"]["N1-O1:r1"]["status"] == "verifier_failed"
    service.release_quarantine(store.run_id, "V1", "a smoke test served haiku on purpose", "mcp")
    state = store.load_state()
    assert state["verification"]["batches"]["refuters"]["status"] == "complete"
    results = read_json(store.role_dir("verifier") / "results.json")
    assert results["requests"]["N1-O1:r1"]["status"] == "verified" and results["requests"]["N1-O1:r1"]["agent_id"] == "V1"
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert any(v["passes"] for v in gate["verdicts"] if v["objection_ids"] == ["N1-O1"])


IDEA = """# Rising Guidance

## Problem

Classifier-free guidance uses a constant guidance weight at every denoising step, which reduces diversity.

## Idea

We propose a guidance weight that increases linearly over the trajectory. We claim that:

1. We are the first to use a time-varying guidance schedule in diffusion models.
2. It needs no additional training and no additional compute.

## Related work

Earlier schedules exist, and we are the first to use a time-varying guidance schedule in diffusion models is false.
"""


def _locate_in(doc, quote, location):
    result = ingest_markdown(doc, "markdown")
    sections = [vars(s) if not isinstance(s, dict) else s for s in result.sections]
    return _locate(quote, location, SourceText(result.text_md, sections), result.text_md, sections)


def test_claims_are_located_without_quote_marks_or_section_numbers():
    """A real refuter's claim_targeted: the claim's own words, then where it is, unquoted (smoke run 002)."""
    span, where = _locate_in(IDEA, None, "We are the first to use a time-varying guidance schedule in diffusion "
                                         "models (Idea, claim 1)")
    assert span == "We are the first to use a time-varying guidance schedule in diffusion models" and "Idea" in where
    span, where = _locate_in(IDEA, None, "Idea, claim 2")  # a named section, when the words are not given
    assert "no additional training" in span and "Idea" in where
    assert _locate_in(IDEA, None, "Section 9") == (None, None)
    span, where = _locate_in(IDEA, None, "the guidance schedule (Related work)")  # never the related work
    assert (span, where) == (None, None)


# ---------------------------------------------------------------- K1 pilot audit: selection, budgets, passages

S_INTRO = ("We claim that a learned, instance-conditioned noise editor achieves the quality of per-query optimization "
           "at the cost of a single forward pass.")
S_METHOD = "The loss combines a task term for requested attributes S and a preservation term for protected attributes P."
S_THEORY = ("If g is L-Lipschitz and the denoiser is contractive, the edited sample stays within epsilon of the "
            "original on protected attributes.")
S_RELATED = "Noise optimization methods such as INSPO optimize the initial noise per query."


def _paper(n):
    return f"Guidance Schedulers Paper Number {n}", f"2401.0000{n}"


def _seed(*ns):
    for n in ns:
        title, ident = _paper(n)
        seed_prior(key=f"arxiv:{ident}", ident=ident, text=PRIOR_TEXT.replace(TITLE, title), title=title)


def _claim(*ns, passages, severity="major", pairs=()):
    """A prior-work objection citing these papers, quoting passages[i] of the submission against paper i+1."""
    return {"category": "already_done", "severity": severity, "basis": "full_text", "confidence": "high",
            "priors": [{"title": _paper(n)[0], "arxiv_id": _paper(n)[1]} for n in ns], "prior_passage": PRIOR_QUOTE,
            "submission_passage": passages[0],
            "pairs": [{"reference": i, "submission_passage": p} for i, p in enumerate(passages[1:], start=2)]
            + list(pairs)}


def _create_with(sample_paper, extra=None, trusted=None, **mock):
    from paper_adversary.config import deep_merge

    info = service.create_run(str(sample_paper), config_override=deep_merge(mock_override(**mock), extra or {}),
                              trusted_config=trusted)
    return RunStore.open(info["run_id"])


BASE = ["intake", "novelty", "rigor", "fit", "judge"]


def test_every_objection_gets_a_check_before_any_gets_a_second(runs_dir, sample_paper):
    """K1 pilot (F1, F23): N1-O2's two equal-priority papers took two of three slots by request-ID order, and
    N2-O2 went unchecked behind a reason ('more central, severe or uncertain threats came first') that was false."""
    _seed(1, 2, 3, 4)
    store = _create_with(sample_paper, extra={"verifier": {"max_papers_per_run": 2}},
                         decisive={"N1": _claim(1, 2, passages=[S_INTRO, S_METHOD]),
                                   "N2": _claim(3, passages=[S_THEORY]), "N3": _claim(4, passages=[S_METHOD])})
    _run(store, BASE)
    reqs = read_json(store.role_dir("verifier") / "results.json")["requests"]
    assert reqs["N1-O1:r1"]["status"] == "verified" and reqs["N2-O1:r1"]["status"] == "verified"
    second = reqs["N1-O1:r2"]  # sorts before N2-O1:r1, but its objection already has a check
    assert second["status"] == "not_run" and "at most 2 prior paper(s)" in second["reason"]
    assert "every objection it serves (N1-O1) already has an independent check (T1)" in second["reason"]
    tied = reqs["N3-O1:r1"]  # equal on every key with N2-O1:r1: the reason says so instead of inventing a rank
    assert tied["status"] == "not_run" and "tied with N2-O1:r1 on coverage, centrality, severity and doubt" in \
        tied["reason"] and "more central, severe or uncertain" not in tied["reason"]
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert any("verification cap" in (v["label"] or "") for v in gate["verdicts"] if v["objection_ids"] == ["N3-O1"])

    store = _create_with(sample_paper, extra={"verifier": {"max_papers_per_run": 1}},
                         decisive={"N1": _claim(1, passages=[S_INTRO]), "N2": _claim(3, passages=[S_THEORY],
                                                                                    severity="fatal")})
    _run(store, BASE)
    reqs = read_json(store.role_dir("verifier") / "results.json")["requests"]
    assert reqs["N2-O1:r1"]["status"] == "verified"
    assert "ranked below N2-O1:r1, rated more severe (fatal against major)" in reqs["N1-O1:r1"]["reason"]


def test_second_passages_and_references_past_the_limit(runs_dir, sample_paper):
    """F4: other submission claims quoted against the same paper are checked too (one more per reference, in the
    same task); F8: a fourth reference is recorded as not run instead of vanishing."""
    _seed(1)
    n1 = _claim(1, 5, 6, 7, passages=[S_INTRO], pairs=[{"reference": 1, "submission_passage": S_THEORY},
                                                       {"reference": 1, "submission_passage": S_INTRO}])
    n2 = _claim(1, passages=[S_THEORY], pairs=[{"reference": 1, "submission_passage": S_RELATED}])
    store = _create_with(sample_paper, decisive={"N1": n1, "N2": n2})
    _run(store, BASE)
    results = read_json(store.role_dir("verifier") / "results.json")
    reqs = results["requests"]
    assert "N1-O1:r1.p3" not in reqs  # the repeated passage is not a second check
    task = results["tasks"][reqs["N1-O1:r1"]["task_id"]]
    assert [p["text"] for p in read_json(store.role_dir("verifier") / "tasks" / "T1.json")["passages"]] == \
        [S_INTRO.rstrip("."), S_THEORY.rstrip(".")]  # each request's own passage first
    assert sorted(task["requests"]) == ["N1-O1:r1", "N1-O1:r1.p2", "N2-O1:r1"]
    assert reqs["N1-O1:r1.p2"]["passage_id"] == reqs["N2-O1:r1"]["passage_id"] == "P2"
    assert {reqs[r]["status"] for r in task["requests"]} == {"verified"}
    assert [a for a, v in store.load_state()["agents"].items() if v["role"] == "verifier"] == ["V1"]
    related = reqs["N2-O1:r1.p2"]  # a further passage is taken verbatim only, and never from related work
    assert related["status"] == "claim_not_located" and related["not_located"] == ["overlap_evidence"]
    fourth = reqs["N1-O1:r4"]
    assert fourth["status"] == "not_run" and "per-objection reference limit" in fourth["reason"]
    assert fourth.get("prior_key") is None  # never fetched or planned
    assert not pending_requests(store, store.load_state())
    batch_id, _ = service.verification_batch(store.run_id, True, None)  # a gate re-check keeps the limit
    requested = {r["request_id"] for r in read_json(store.role_dir("verifier") / "batches"
                                                    / f"{batch_id}.requests.json")}
    assert "N1-O1:r2" in requested and "N1-O1:r4" not in requested
    judge_prompt = (store.agent_log_dir("J1") / "attempt-1" / "user_prompt.md").read_text()
    assert "[submission passage at" in judge_prompt
    # the judges are told which paper went unchecked (it was "N1-O1 vs ?": a limited request is never fetched)
    assert "- N1-O1 vs Guidance Schedulers Paper Number 7: not independently checked — not_run (per-objection " \
           "reference limit" in judge_prompt
    # the verifier's quote tool knows its passages (F5), so it can say when a quote lies outside them
    from paper_adversary.context import ContextBuilder
    from paper_adversary.pipeline import spec_from_state
    from paper_adversary.registry import ModelRegistry

    state = store.load_state()
    ctx = ContextBuilder(store, store.load_config(), ModelRegistry.load()).build(
        spec_from_state(state["agents"]["V1"]), state)
    args = ctx.tool_server["args"]
    shown = json.loads(args[args.index("--passages") + 1])
    assert [p["pid"] for p in shown["passages"]] == ["P1", "P2"] and shown["max_words"] == 80


def test_runs_planned_before_second_passages_grow_no_new_batch(runs_dir, sample_paper):
    """Review (G1): on a copy of the K1 pilot, pending_requests suddenly listed N1-O1:r1.p2, N1-O2:r4, ... (requests
    the run's batch was planned without), so the next judge job planned a refuters2 batch that could only be
    capped, and rewriting the judges' view made them stale. An objection whose first request is recorded for its
    current content does not grow such requests; a rerun objection still gets all of them."""
    from paper_adversary.util import atomic_write_json

    _seed(1)
    n1 = _claim(1, 5, 6, 7, passages=[S_INTRO], pairs=[{"reference": 1, "submission_passage": S_THEORY}])
    store = _create_with(sample_paper, decisive={"N1": n1})
    _run(store, BASE)
    path = store.role_dir("verifier") / "results.json"
    results = read_json(path)
    for rid in ("N1-O1:r1.p2", "N1-O1:r4"):  # as a batch planned before 2026-10-02 recorded it
        results["requests"].pop(rid)
    atomic_write_json(path, results)
    state = store.load_state()
    assert pending_requests(store, state) == [] and not Pipeline(store)._verification_pending()
    result, _ = _run(store, ["judge"], rerun=["J1"])
    assert result["outcome"] == "complete" and "refuters2" not in store.load_state()["verification"]["batches"]
    _set_mock(store, decisive={"N1": {**n1, "severity": "fatal"}})  # a rerun objection is new content
    _run(store, ["novelty"], rerun=["N1"])
    assert {r.request_id for r in pending_requests(store, store.load_state())} >= {"N1-O1:r1", "N1-O1:r1.p2",
                                                                                   "N1-O1:r4"}


def test_verification_markdown_shows_every_accepted_overlap():
    """F3: the block sliced the first two pairs before filtering, so an accepted pair after a rejected one never
    reached judges or memos. F21: the header says what verifiers actually saw."""
    from paper_adversary.verification import verification_markdown

    def pair(i, ok):
        return {"prior_passage": f"prior sentence number {i}", "prior_location_found": f"p. {i}", "accepted": ok,
                "submission_passage": f"submission words {i}"}
    rec = {"origin_ids": ["N1-O1"], "prior_title": "DeCurvEd", "status": "verified", "agent_id": "V1",
           "verdict": "anticipates_partially", "confidence": "high", "passage_location": "S01 Abstract",
           "overlap": [pair(1, False), pair(2, True), pair(3, False), pair(4, True)]}
    text = verification_markdown({"requests": {"N1-O1:r1": rec}})
    assert "prior sentence number 2" in text and "prior sentence number 4" in text
    assert "prior sentence number 1" not in text and "prior sentence number 3" not in text
    assert 'on the submission\'s "submission words 4"' in text and "[submission passage at S01 Abstract]" in text
    assert "each saw the whole submission" in text and "never the objection's or item's argument" in text
    rec["overlap"] = [pair(i, True) for i in range(1, 6)]
    text = verification_markdown({"requests": {"N1-O1:r1": rec}})
    assert "prior sentence number 3" in text and "prior sentence number 4" not in text
    assert "(+2 more verbatim-verified overlap pairs)" in text


def _gate(members_evidence, requests, severity="MAJOR_FIXABLE", declared=None, independent=True):
    """evaluate() on one judgment over hand-made quote checks and verification records."""
    from paper_adversary.evidence_gate import evaluate

    ids = list(members_evidence)
    objections = {oid: {"id": oid, "category": "already_done", "references": [{"title": "X"}]} for oid in ids}
    judgments = {"J2": [{"objection_ids": ids, "title": "grouped", "severity": severity,
                         "evidence_status": declared}]}
    return evaluate(judgments, objections, members_evidence, {"requests": requests}, independent, hashes={})


CAPPED = {"N2-O2:r1": {"origin_ids": ["N2-O2"], "status": "not_run",
                       "reason": "verification cap: at most 3 prior paper(s) are verified per run"}}


def test_a_truthful_quotes_only_declaration_is_not_called_overstated():
    """K1 pilot (F7, F11): J1-J3 declared 'verified_quotes_only' for verified quotes without a blind check, which is
    what that status means, and the gate called it unsupported (printing 'quotes quotes verified')."""
    from paper_adversary.evidence_gate import gate_markdown

    evidence = {"N2-O2": {"status": "quotes_verified", "decisive": True},
                "N1-O5": {"status": "not_required", "decisive": False, "severity_estimate": "minor"}}
    gate = _gate(evidence, CAPPED, declared="verified_quotes_only")
    row = gate["verdicts"][0]
    assert not row["passes"] and not row["overstated"]
    text = gate_markdown(gate)
    assert "quotes quotes" not in text and "N2-O2: quote check: quotes verified" in text
    row = _gate(evidence, CAPPED, declared="verified_independent")["verdicts"][0]
    assert row["overstated"] and "declared 'verified_independent'; the record shows no blind verifier" in \
        row["overstated_detail"]
    row = _gate({"N1-O5": {"status": "no_evidence"}}, {}, declared="verified_quotes_only")["verdicts"][0]
    assert row["overstated"] and "no member's quotes verified (N1-O5: no evidence)" in row["overstated_detail"]
    # independent of whether the gate requires the blind check: a quotes-only pass hides no false claim
    row = _gate({"N2-O2": {"status": "quotes_verified"}}, {}, declared="verified_independent",
                independent=False)["verdicts"][0]
    assert row["passes"] and row["overstated"]
    assert "J2 declared 'verified_independent'" in gate_markdown(_gate({"N2-O2": {"status": "quotes_verified"}}, {},
                                                                         declared="verified_independent",
                                                                         independent=False))


def test_each_member_says_why_it_has_no_independent_check():
    """K1 pilot (F6, F10, F25): N1-O5 was never requested (its refuter rated it minor, the judges escalated it),
    yet its group was labelled 'the run's verification cap was reached', which was true of N2-O2 only."""
    evidence = {"N2-O2": {"status": "quotes_verified", "decisive": True},
                "N1-O5": {"status": "quotes_verified", "decisive": False, "severity_estimate": "minor"}}
    gate = _gate(evidence, CAPPED)
    row = gate["verdicts"][0]
    reasons = {m["id"]: m["reason"] for m in row["members"]}
    assert reasons == {"N2-O2": "cap reached",
                       "N1-O5": "not requested: the refuter rated it minor; judges escalated it"}
    assert row["label"] == ("UNVERIFIED: not independently checked (N2-O2: cap reached; N1-O5: not requested: the "
                            "refuter rated it minor; judges escalated it)")
    assert gate["never_requested"] == ["N1-O5"]
    only_capped = _gate({"N2-O2": evidence["N2-O2"]}, CAPPED)["verdicts"][0]
    assert only_capped["label"] == "UNVERIFIED: not independently checked (the run's verification cap was reached)"


def test_a_serious_verdict_on_prior_work_is_gated_through_whichever_member_carries_it():
    """Review (G2): a judgment that relies on prior work, grouping a prior-work objection it rates lower with a MAJOR
    one, produced no gate record at all (before member_severity it was checked through the prior-work member).
    Its serious member is now held to the evidence rule, the lower prior-work member is listed apart, and a member
    that is not a prior-work objection is not called escalated nor sent to the automatic gate batch."""
    from paper_adversary.evidence_gate import evaluate
    from paper_adversary.reports import judge_judgments

    objections = {"N1-O3": {"id": "N1-O3", "category": "already_done", "references": [{"title": "P"}]},
                  "R1-O2": {"id": "R1-O2", "category": "unsupported_claim"},
                  "N1-O6": {"id": "N1-O6", "category": "incremental", "severity_estimate": "major"}}
    evidence = {"N1-O6": {"status": "not_required", "decisive": False, "severity_estimate": "major"}}

    def gate(*judgments):
        return evaluate({"J1": judge_judgments({"judgments": list(judgments)})}, objections, evidence,
                        {"requests": {}}, True, hashes={})
    grouped = {"objection_ids": ["N1-O3", "R1-O2"], "severity": "MAJOR_BUT_FIXABLE",
               "member_severity": {"N1-O3": "MINOR"}, "relies_on_prior_work": True}
    row, = gate(grouped)["verdicts"]
    assert [m["id"] for m in row["members"]] == ["R1-O2"] and row["lower_members"] == [{"id": "N1-O3",
                                                                                        "severity": "MINOR"}]
    assert not row["passes"] and row["label"].startswith("UNVERIFIED")
    assert row["members"][0]["reason"] == "not requested: it cites no prior paper to check"
    result = gate({**grouped, "relies_on_prior_work": False})  # nothing says it rests on prior work: not gated
    assert result["verdicts"] == [] and result["flagged_ids"] == []
    alone = gate({"objection_ids": ["N1-O6"], "severity": "FATAL", "relies_on_prior_work": True})
    member = alone["verdicts"][0]["members"][0]  # a novelty objection rated major that cites no paper
    assert member["reason"] == "not requested: it cites no prior paper to check" and not member["prior_work"]
    assert alone["flagged_ids"] == ["N1-O6"] and alone["never_requested"] == []


ESCALATED = {"minor_prior": {"N1": _claim(1, passages=[S_THEORY], severity="minor")},
             "rate": {"J1": {"N1-O2": "MAJOR_FIXABLE"}}}  # N1 rates N1-O2 minor; judge J1 rates it MAJOR


def test_an_escalated_objection_is_quote_checked_and_can_be_rechecked(runs_dir, sample_paper):
    """K1 pilot (F6, F10, F25): objections the refuter rated minor were never quote-checked, the gate blamed the
    cap, and run_verification(from_gate=true) could not reach them."""
    _seed(1)
    store = _create_with(sample_paper, **ESCALATED)
    _run(store)
    ev = read_json(store.sidecar_path("N1", "novelty", ".evidence.json"))["objections"]["N1-O2"]
    assert not ev["decisive"] and ev["status"] == "quotes_verified"  # checked although minor
    assert "N1-O2:r1" not in read_json(store.role_dir("verifier") / "results.json")["requests"]
    judge_prompt = (store.agent_log_dir("J1") / "attempt-1" / "user_prompt.md").read_text()
    assert "N1-O2 [already_done, minor; below major, checked because it quotes prior work]" in judge_prompt
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    row = next(v for v in gate["verdicts"] if v["objection_ids"] == ["N1-O2"])
    assert "N1-O2" in gate["flagged_ids"] and "N1-O2" in gate["never_requested"]
    assert row["label"] == ("UNVERIFIED: not independently checked (N1-O2: not requested: the refuter rated it "
                            "minor; judges escalated it)")
    line = next(ln for ln in service.get_report(store.run_id, "matrix").splitlines() if ln.startswith("| N1-O2 "))
    assert "quotes verified; not blind-checked (not requested: the refuter rated it minor" in line  # F13
    assert line.rstrip(" |").endswith("not shown (evidence gate)")
    assert "auto_gate" not in store.load_state()["verification"]  # auto_gate_batch is off here
    batch_id, _ = service.verification_batch(store.run_id, True, None)
    requested = {r["request_id"] for r in read_json(store.role_dir("verifier") / "batches"
                                                    / f"{batch_id}.requests.json")}
    assert "N1-O2:r1" in requested
    result, _ = _run(store, [f"verify:{batch_id}"])
    assert result["outcome"] == "complete", result
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert "N1-O2" not in gate["flagged_ids"]
    assert next(v for v in gate["verdicts"] if v["objection_ids"] == ["N1-O2"])["passes"]


def test_the_automatic_gate_batch_checks_escalations_without_rerunning_judges(runs_dir, sample_paper):
    """verifier.auto_gate_batch (stage2_split): after the judges, one gate batch verifies the objections they
    escalated, within its own budget; the gate is recomputed, the judges neither rerun nor turn stale, and an
    interrupted batch is finished before the memo on resume, never planned twice."""
    from paper_adversary.pipeline import stale_agents

    _seed(1)
    store = _create_with(sample_paper, trusted={"verifier": {"auto_gate_batch": True}},
                         fail={"V1": ["plan_limit", "plan_limit"]}, **ESCALATED)
    result, _ = _run(store)
    state = store.load_state()
    assert result["outcome"] == "incomplete" and state["verification"]["auto_gate"] == "gate1"
    assert state["verification"]["batches"]["gate1"]["status"] == "planned"
    assert state["agents"]["V1"]["status"] == "interrupted" and state["agents"]["S1"]["status"] == "pending"
    _set_mock(store, fail={})
    stages = service.resume_stages(state, "critic", stale_agents(store, state))
    result, _ = _run(store, stages, refresh_stale=True)
    assert result["outcome"] == "complete", result
    state = store.load_state()
    batches = state["verification"]["batches"]
    assert batches["gate1"]["status"] == "complete" and "automatic" in batches["gate1"]["reason"]
    assert not any(k.startswith("gate") and k != "gate1" for k in batches)
    assert read_json(store.role_dir("verifier") / "results.json")["requests"]["N1-O2:r1"]["status"] == "verified"
    assert "N1-O2" not in read_json(store.role_dir("judge") / "evidence_gate.json")["flagged_ids"]
    assert {a: v["attempts"] for a, v in state["agents"].items() if v["role"] == "judge"} == \
        {"J1": 1, "J2": 1, "J3": 1}
    assert not stale_agents(store, state)  # the judges read the refuters' view; the memo came after the batch
    memo_prompt = (store.agent_log_dir("S1") / "attempt-1" / "user_prompt.md").read_text()
    assert "N1-O2 vs Guidance Schedulers Paper Number 1" in memo_prompt
    assert any(e["event"] == "auto_gate_batch" for e in read_jsonl(store.events_path))
    _run(store, ["synthesis"], rerun=["S1"])  # once per run
    assert not any(k.startswith("gate") and k != "gate1" for k in store.load_state()["verification"]["batches"])


def test_a_failed_gate_verifier_is_collected_once_it_reruns(runs_dir, sample_paper):
    """Review (G1): a gate verifier that failed for good left gate1 'incomplete'; resume_run reran it, but nothing
    collected its verdict, so the gate kept flagging the objection and a later re-check would have paid for a
    second verifier. A batch recorded incomplete is finished again once its verifiers ran, by the verifier stage,
    or by the automatic step of a later job (never twice in the job where they failed)."""
    from paper_adversary.pipeline import stale_agents

    _seed(1)
    results = lambda st: read_json(st.role_dir("verifier") / "results.json")["requests"]  # noqa: E731
    flagged = lambda st: read_json(st.role_dir("judge") / "evidence_gate.json")["flagged_ids"]  # noqa: E731
    for how in ("resume", "memo"):
        store = _create_with(sample_paper, trusted={"verifier": {"auto_gate_batch": True}},
                             fail={"V1": ["overloaded"] * 3}, **ESCALATED)
        result, pipeline = _run(store)
        state = store.load_state()
        assert result["outcome"] == "incomplete" and "V1 (failed)" in result["message"], result
        assert pipeline.provider.attempts["V1"] == 3  # not retried again before the memo of the same job
        assert state["verification"]["batches"]["gate1"]["status"] == "incomplete"
        assert state["agents"]["S1"]["status"] == "complete" and "N1-O2" in flagged(store)
        assert results(store)["N1-O2:r1"]["status"] == "verifier_failed"
        if how == "memo":  # still failing: the verifier stage reruns it, and the step before the memo does not
            _, pipeline = _run(store, ["verifier", "synthesis"])
            assert pipeline.provider.attempts["V1"] == 3
            assert store.load_state()["verification"]["batches"]["gate1"]["status"] == "incomplete"
        _set_mock(store, fail={})
        if how == "resume":
            stages = service.resume_stages(state, "critic", stale_agents(store, state))
            assert stages == ["verifier"]
            result, _ = _run(store, stages, refresh_stale=True)
        else:  # a later job without the verifier stage: its automatic step reruns the failed verifier
            result, _ = _run(store, ["synthesis"], rerun=["S1"])
        assert result["outcome"] == "complete", (how, result)
        state = store.load_state()
        assert state["verification"]["batches"]["gate1"]["status"] == "complete"
        assert results(store)["N1-O2:r1"]["status"] == "verified" and "N1-O2" not in flagged(store), how
        assert not any(k.startswith("gate") and k != "gate1" for k in state["verification"]["batches"])
        if how == "resume":  # the memo read the old gate; the next resume refreshes it
            assert "S1" in stale_agents(store, state)
        else:
            assert "N1-O2 vs Guidance Schedulers Paper Number 1" in \
                (store.agent_log_dir("S1") / "attempt-2" / "user_prompt.md").read_text()


def test_a_gate_recheck_that_finds_nothing_new_leaves_the_judges_current(runs_dir, sample_paper, monkeypatch):
    """Review (G1): before a gate batch, every refuter with an objection whose quote check could not succeed
    (unavailable text, quotes not found) is checked again, in case a full text was supplied. Its record was
    rewritten even when nothing changed, with a new timestamp, so every judge turned stale for nothing."""
    from paper_adversary.pipeline import stale_agents

    _seed(1)
    wrong = {**_claim(1, passages=[S_INTRO], severity="minor"), "prior_passage": "words this paper never says at "
             "any point in its text, about guidance or about anything else"}
    store = _create_with(sample_paper, minor_prior={**ESCALATED["minor_prior"], "N2": wrong}, rate=ESCALATED["rate"])
    _run(store)
    assert read_json(store.sidecar_path("N2", "novelty", ".evidence.json"))["objections"]["N2-O2"]["status"] == \
        "quotes_not_found"
    assert not stale_agents(store, store.load_state())
    before = {aid: sha256_file(store.sidecar_path(aid, "novelty", ".evidence.json")) for aid in ("N1", "N2")}
    monkeypatch.setattr("paper_adversary.evidence.utcnow_iso", lambda: "2099-01-01T00:00:00+00:00")  # time passes
    batch_id, _ = service.verification_batch(store.run_id, True, None)
    result, _ = _run(store, [f"verify:{batch_id}"])
    assert result["outcome"] == "complete", result
    assert {aid: sha256_file(store.sidecar_path(aid, "novelty", ".evidence.json")) for aid in ("N1", "N2")} == before
    stale = stale_agents(store, store.load_state())
    assert not [a for a in stale if a.startswith("J")], stale  # the memo, which reads the gate, may be


def test_references_past_the_limit_are_recorded_in_a_gate_batch(runs_dir, sample_paper):
    """Review (G1): an escalated objection was never in a refuters' batch, and the gate batches dropped its
    references past the third, so they were recorded nowhere. They are now recorded as not run (never fetched)."""
    _seed(1)
    store = _create_with(sample_paper, trusted={"verifier": {"auto_gate_batch": True}}, rate=ESCALATED["rate"],
                         minor_prior={"N1": _claim(1, 5, 6, 7, passages=[S_THEORY], severity="minor")})
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    requested = [r["request_id"] for r in read_json(store.role_dir("verifier") / "batches" / "gate1.requests.json")]
    assert requested == ["N1-O2:r1", "N1-O2:r2", "N1-O2:r3", "N1-O2:r4"]
    fourth = read_json(store.role_dir("verifier") / "results.json")["requests"]["N1-O2:r4"]
    assert fourth["status"] == "not_run" and "per-objection reference limit" in fourth["reason"]
    assert fourth.get("prior_key") is None and fourth["prior_title"] == "Guidance Schedulers Paper Number 7"
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    member = next(m for v in gate["verdicts"] if v["objection_ids"] == ["N1-O2"] for m in v["members"])
    assert member["limited"] == 1
    assert len(store.load_state()["verification"]["batches"]["gate1"]["agents"]) == 1  # paper 1 only


def test_authors_take_turns_and_papers_serving_more_objections_come_first(runs_dir, sample_paper):
    """Review (G1): two keys of the coverage-greedy selection had no test. On equal coverage, centrality, severity
    and doubt, an author without a task yet goes before a second task of another author; and a paper serving two
    unchecked objections goes before one serving a single one."""
    _seed(1, 2, 3, 4)
    major = lambda n: _claim(n, passages=[S_METHOD], severity="major")  # noqa: E731
    store = _create_with(sample_paper, extra={"verifier": {"max_papers_per_run": 2}},
                         decisive={"N1": major(1), "N2": major(3)}, minor_prior={"N1": major(2)})
    _run(store, BASE)
    reqs = read_json(store.role_dir("verifier") / "results.json")["requests"]
    assert reqs["N1-O1:r1"]["status"] == reqs["N2-O1:r1"]["status"] == "verified"
    second = reqs["N1-O2:r1"]  # sorts before N2-O1:r1, but N1 already has a task
    assert second["status"] == "not_run" and "tied with N2-O1:r1" in second["reason"] and \
        "authors take turns" in second["reason"]

    store = _create_with(sample_paper, extra={"verifier": {"max_papers_per_run": 1}},
                         decisive={"N1": major(2), "N2": major(1), "N3": major(1)})
    _run(store, BASE)
    reqs = read_json(store.role_dir("verifier") / "results.json")["requests"]
    assert reqs["N2-O1:r1"]["status"] == reqs["N3-O1:r1"]["status"] == "verified"  # one paper, two objections
    assert reqs["N1-O1:r1"]["status"] == "not_run" and \
        "ranked below N2-O1:r1, which serves more objections without an independent check" in reqs["N1-O1:r1"]["reason"]


def _k1_judgments():
    """The K1 pilot's grouping pattern (F9): J1 and J3 put minor framing objections into MAJOR groups; J2 rules on
    them alone. J3 (judge_v3 style) gives N2-O3 its own, lower severity."""
    from paper_adversary.reports import judge_judgments

    def j(ids, sev, member=None):
        return {"objection_ids": ids, "title": "t", "severity": sev, "confidence": "medium",
                "member_severity": member or {}}
    raw = {"J1": [j(["N2-O2", "N1-O5"], "MAJOR_FIXABLE"), j(["N2-O3"], "MINOR"), j(["N1-O3"], "MAJOR_FIXABLE")],
           "J2": [j(["N2-O2"], "MAJOR_FIXABLE"), j(["N1-O5"], "MINOR"), j(["N2-O3"], "MINOR"), j(["N1-O3"], "MINOR")],
           "J3": [j(["N1-O5", "N2-O2", "N2-O3"], "MAJOR_FIXABLE", {"N2-O3": "MINOR"}), j(["N1-O3"], "MAJOR_FIXABLE")]}
    return {jid: judge_judgments({"judgments": items}) for jid, items in raw.items()}


K1_OBJECTIONS = {oid: {"id": oid, "source": oid[:2], "title": f"objection {oid}", "category": "framing_exists",
                       "references": [{"title": "X"}]} for oid in ("N1-O3", "N1-O5", "N2-O2", "N2-O3")}


def test_grouped_members_are_held_to_their_own_severity():
    """K1 pilot (F9): the gate flagged N1-O3, N1-O5, N2-O2 and N2-O3, three of them only because a judge filed them
    under a MAJOR group. Each member is now checked at its own severity: one rated lower in member_severity is not
    flagged, and flags that rest only on grouped verdicts are named."""
    from paper_adversary.evidence_gate import evaluate, gate_markdown

    evidence = {"N2-O2": {"status": "quotes_verified", "decisive": True}}
    gate = evaluate(_k1_judgments(), K1_OBJECTIONS, evidence, {"requests": CAPPED}, True, hashes={})
    assert gate["flagged_ids"] == ["N1-O3", "N1-O5", "N2-O2"]  # N2-O3: every judge's own severity is MINOR
    assert gate["flagged_grouped_only"] == ["N1-O5"]  # J2, ruling on it alone, rated it MINOR
    j3 = next(v for v in gate["verdicts"] if v["judge"] == "J3" and len(v["objection_ids"]) == 3)
    assert [m["id"] for m in j3["members"]] == ["N1-O5", "N2-O2"]
    assert j3["lower_members"] == [{"id": "N2-O3", "severity": "MINOR"}]
    text = gate_markdown(gate)
    assert "N2-O3: cited in this judgment, but J3's own severity for it is MINOR" in text
    assert "Flagged only through grouped verdicts" in text and "lower: N1-O5" in text


def test_the_matrix_tabulates_own_severities_marks_groups_and_flags_rows():
    """F9, F13, F22: per-member severities with their group recorded, grouped cells marked, gate-flagged rows
    flagged in the table, and one row per declared rubric test."""
    from paper_adversary.reports import build_judgment_matrix, matrix_markdown

    objections = {"N1": [K1_OBJECTIONS["N1-O3"], K1_OBJECTIONS["N1-O5"]],
                  "N2": [K1_OBJECTIONS["N2-O2"], K1_OBJECTIONS["N2-O3"]]}
    tests = [{"id": "RT1", "text": "amortizing method X with a network is incremental"},
             {"id": "RT2", "text": "a trivially implied theorem target does not count"}]
    rubric = {"J1": [{"test_id": "RT1", "applies": "yes", "severity": "FATAL"}],
              "J2": [{"test_id": "RT1", "applies": "no", "severity": None}]}
    matrix = build_judgment_matrix(objections, _k1_judgments(), rubric_tests=tests, rubric=rubric)
    rows = {r["id"]: r for r in matrix["rows"]}
    n15 = rows["N1-O5"]["verdicts"]
    assert (n15["J1"]["severity"], n15["J1"]["grouped"], n15["J1"]["group"]) == ("MAJOR_FIXABLE", True, "J1#1")
    assert n15["J2"]["severity"] == "MINOR" and not n15["J2"]["grouped"]
    n23 = rows["N2-O3"]["verdicts"]["J3"]
    assert (n23["severity"], n23["group_severity"], n23["grouped"]) == ("MINOR", "MAJOR_FIXABLE", False)
    assert rows["N2-O3"]["max_severity"] == "MINOR"  # no longer lifted by J3's group
    matrix["evidence_gate"] = {"flagged_ids": ["N1-O5"]}
    text = matrix_markdown(matrix)
    line = next(ln for ln in text.splitlines() if ln.startswith("| N1-O5 "))
    assert "MAJOR BUT FIXABLE (medium, grouped)" in line and "not shown (evidence gate)" in line
    assert "Cells marked 'grouped'" in text
    assert "| RT1 | amortizing method X with a network is incremental | yes, FATAL | no | — |" in text
    assert matrix["rubric_tests"][1]["not_ruled_by"] == ["J1", "J2", "J3"]


def test_the_evidence_column_shows_capped_and_never_requested_states():
    """K1 pilot (F13): the matrix showed N2-O2 as 'quotes verified' although its only request was capped, and '—'
    for flagged objections nobody was asked to check."""
    from paper_adversary.evidence_gate import member_status
    from paper_adversary.pipeline import _evidence_cell

    verification = {"requests": {
        **CAPPED,
        "N1-O2:r1": {"origin_ids": ["N1-O2"], "status": "verified", "verdict": "anticipates_partially",
                     "agent_id": "V2"},
        "N1-O2:r3": {"origin_ids": ["N1-O2"], "status": "not_run", "reason": "verification cap: at most 3"},
        "N1-O2:r4": {"origin_ids": ["N1-O2"], "status": "not_run", "limited": True,
                     "reason": "per-objection reference limit"}}}
    evidence = {"N2-O2": {"status": "quotes_verified", "decisive": True},
                "N1-O2": {"status": "quotes_verified", "decisive": True},
                "N1-O5": {"status": "not_required", "decisive": False, "severity_estimate": "minor"}}
    cell = {oid: _evidence_cell(member_status(oid, evidence, verification, {}), verification)
            for oid in ("N2-O2", "N1-O2", "N1-O5")}
    assert cell["N2-O2"] == "quotes verified; not blind-checked (cap reached)"
    assert cell["N1-O2"] == "quotes verified; V2: partially; 1 more not run (cap); 1 beyond the per-objection limit"
    assert cell["N1-O5"] == ("no quote check required; not blind-checked (not requested: the refuter rated it minor; "
                             "judges escalated it)")


def test_a_judge_v3_member_severity_reaches_the_matrix(runs_dir, sample_paper):
    """F9 end to end: judge_v3's member_severity survives the gate and the sidecar, and the matrix tabulates each
    member at its own severity. judge_v3 tells the judge that a member it leaves out of member_severity counts at
    the judgment's severity, so that is the judge's own rating of it: not marked 'grouped', and not discounted as
    grouped-only when another judge rates it lower (judge_v2 groups still are: test_grouped_members_...)."""
    from paper_adversary.reports import is_grouped

    group = [{"ids": ["N1-O1", "R1-O1"], "severity": "MAJOR_FIXABLE", "member_severity": {"R1-O1": "MINOR"}}]
    store = _create_with(sample_paper, extra={"judge": {"prompt": "judge_v3"}}, group={"J1": group},
                         rate={"J2": {"N1-O1": "MINOR"}})
    _run(store, BASE)
    j1 = store.load_state()["agents"]["J1"]
    assert j1["status"] == "complete" and not any("member_severity" in w for w in j1["gate"].get("warnings") or [])
    rows = {r["id"]: r for r in read_json(store.role_dir("judge") / "judgment_matrix.json")["rows"]}
    assert (rows["R1-O1"]["verdicts"]["J1"]["severity"], rows["R1-O1"]["verdicts"]["J1"]["grouped"]) == ("MINOR", False)
    assert (rows["N1-O1"]["verdicts"]["J1"]["severity"], rows["N1-O1"]["verdicts"]["J1"]["grouped"]) == \
        ("MAJOR_FIXABLE", False)
    assert rows["N1-O1"]["verdicts"]["J1"]["group_ids"] == ["N1-O1", "R1-O1"]  # the group is still recorded
    line = next(ln for ln in service.get_report(store.run_id, "matrix").splitlines() if ln.startswith("| N1-O1 "))
    assert "MAJOR BUT FIXABLE (medium)" in line and "grouped" not in line
    major = {x["id"]: x for x in followup.open_issues(store, store.load_state())["major"]}
    assert not major["N1-O1"]["grouped_only"]  # J1 vs J2 is a plain disagreement on the group's strongest member
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert "N1-O1" not in gate["flagged_grouped_only"]
    v2 = {"objection_ids": ["N1-O1", "R1-O1"], "member_severity": {"R1-O1": "MINOR"}}
    assert is_grouped(v2, "N1-O1") and not is_grouped({**v2, "own_severities": True}, "N1-O1")
