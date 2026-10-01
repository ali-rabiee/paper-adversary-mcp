"""Verified novelty evidence end to end (mock provider, offline full texts): quote checks, blind verifiers,
the evidence gate, and the memo's unverified-threats section."""

import asyncio
from pathlib import Path

import yaml

from conftest import mock_override
from paper_adversary import gates, service
from paper_adversary.config import FullTextConfig
from paper_adversary.pipeline import Pipeline
from paper_adversary.reports import read_report
from paper_adversary.search.fulltext import FullTextResult, FullTextStore
from paper_adversary.store import RunStore
from paper_adversary.util import read_json, read_jsonl, runs_root, sha256_text
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
