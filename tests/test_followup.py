"""Follow-up rounds after the completeness critic (mock provider): triage, blind verification, adjudication,
the revised memo, the re-check critic, the stop rule, isolation inside rounds, resume."""

import asyncio
import re

import pytest
import yaml

from conftest import mock_override
from paper_adversary import followup, gates, service
from paper_adversary.config import ConfigError, deep_merge, load_config
from paper_adversary.isolation import IsolationGuard, IsolationViolation, position
from paper_adversary.pipeline import Pipeline
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore
from paper_adversary.util import read_json, read_jsonl, sha256_file
from test_verification import S_INTRO, S_THEORY, _claim, _paper, _seed

FULL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic", "followup"]


def item(i, kind="new_issue", severity="MAJOR_FIXABLE", **extra):
    return {"id": f"I{i}", "type": kind, "title": f"critic item {i} about {kind}", "location": "Sec. 3",
            "argument": f"argument {i}", "severity_estimate": severity, "confidence": "medium", "claim_ids": [],
            "objection_ids": [], "judge_ids": [], "memo_sections": [], "candidate_references": [], **extra}


ITEMS = [item(1), item(2, "synthesis_flaw"), item(3, severity="MINOR"),
         item(4, "minority_critique", objection_ids=["Z9-O1"])]


def _create(sample_paper, extra=None, trusted=None, **mock):
    info = service.create_run(str(sample_paper), config_override=deep_merge(mock_override(**mock), extra or {}),
                              trusted_config=trusted)
    return RunStore.open(info["run_id"])


def _run(store, stages=FULL, **kw):
    pipeline = Pipeline(store)
    return asyncio.run(pipeline.run(stages, **kw)), pipeline


def _seen(store, aid, role):
    _, body = read_report(store.report_path(aid, role))
    m = re.search(r"Inputs seen: (.*)", body)
    return set() if not m or m.group(1).strip() == "none" else {x.strip() for x in m.group(1).split(",")}


def test_a_round_end_to_end(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": ITEMS})
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    # no new items, but the mock's J2 rates every first objection FATAL: open issues, not ready
    assert "follow-up: review_saturated_with_open_issues" in result["message"], result
    state = store.load_state()
    agents = state["agents"]
    for aid, role in (("A1", "adjudicator"), ("A2", "adjudicator"), ("S2", "revision"), ("C2", "recheck")):
        assert agents[aid]["status"] == "complete" and agents[aid]["round"] == 1, aid
        assert store.report_path(aid, role).is_file()
    rnd = state["followup"]["rounds"]["1"]
    assert rnd["status"] == "complete" and rnd["outcome"] == followup.SATURATED and rnd["memo_before"] == "S1"
    issues = read_json(store.dir / "followup/round-1/round.json")["open_issues"]
    assert {x["id"] for x in issues["blocking"]["fatal"]} >= {"R1-O1", "F1-O1"} and not issues["blocking"].get("reraised")
    items = read_json(store.dir / "followup/round-1/items.json")["items"]
    assert [(i["id"], i["route"]) for i in items] == [("C1-I1", "adjudicate"), ("C1-I2", "revision"),
                                                     ("C1-I3", "noted"), ("C1-I4", "invalid")]
    assert (store.dir / "followup/round-1/followup_matrix.md").is_file()
    assert (store.dir / "followup/round-1/round.json").is_file()
    # the revised memo is current; the first memo stays in place, recorded as superseded
    assert state["followup"]["current_memo"] == "S2" and agents["S1"]["superseded_by"] == "S2"
    assert (store.dir / "synthesis/memo.md").is_file() and (store.dir / "synthesis/memo_S2.md").is_file()
    assert any(e.get("agent_id") == "S1" and e.get("kept_in_place")
               for e in read_jsonl(store.dir / "archive" / "index.jsonl"))
    assert service.get_report(store.run_id, "memo").find("(S2)") > 0
    dispositions = read_json(store.sidecar_path("S2", "revision", ".json"))["data"]["item_dispositions"]
    assert {d["item_id"]: d["disposition"] for d in dispositions} == {
        "C1-I1": "incorporated", "C1-I2": "incorporated", "C1-I3": "noted", "C1-I4": "invalid"}
    status = service.status_text(store.run_id)
    assert "Follow-up round 1: complete" in status and "Current memo: S2" in status
    # K1 pilot (F18): one line per follow-up agent, and the superseded memo says so
    assert "  A1     adjudicator: complete" in status and "  A2     adjudicator: complete" in status
    assert "  S2     revised memo: complete" in status and "  C2     re-check critic: complete" in status
    synthesis = next(ln for ln in status.splitlines() if ln.startswith("Synthesis: "))
    assert synthesis.startswith("Synthesis: complete") and synthesis.endswith("(superseded by S2)")
    assert "synthesis/memo.md (S1, superseded by S2)" in status


def test_who_sees_what_inside_a_round(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": ITEMS})
    _run(store)
    a1, a2 = _seen(store, "A1", "adjudicator"), _seen(store, "A2", "adjudicator")
    assert {"CANARY_N1", "CANARY_J1", "CANARY_S1", "CANARY_C1"} <= a1
    assert "CANARY_A2" not in a1 and "CANARY_A1" not in a2  # adjudicators are blind to each other
    assert "CANARY_S2" not in a1 and "CANARY_C2" not in a1
    assert {"CANARY_A1", "CANARY_A2", "CANARY_C1"} <= _seen(store, "S2", "revision")
    assert {"CANARY_S2", "CANARY_A1", "CANARY_A2"} <= _seen(store, "C2", "recheck")
    guard = IsolationGuard(store)
    _, a2_body = read_report(store.report_path("A2", "adjudicator"))
    with pytest.raises(IsolationViolation, match="A2"):  # its marker gives it away
        guard.check_prompt("adjudicator", "A1", a2_body, pos=position("adjudicator", 1))
    # the position rule: same or later steps are forbidden, earlier steps are not
    assert "A2" in guard.forbidden("adjudicator", "A1", position("adjudicator", 1))
    for_revision = guard.forbidden("revision", "S9", position("revision", 1))
    assert "A2" not in for_revision and "C1" not in for_revision and {"S2", "C2"} <= set(for_revision)
    for_rerun_c1 = guard.forbidden("critic", "C1", position("critic", 0))
    assert {"A1", "A2", "S2", "C2"} <= set(for_rerun_c1)  # a rerun of the base critic never sees the follow-up


def test_unverified_prior_work_item_must_be_filed_as_such(runs_dir, sample_paper):
    novelty = item(5, "novelty_to_verify", candidate_references=[{"title": "Some Earlier Paper",
                                                                  "arxiv_id": "2201.00001"}])
    store = _create(sample_paper, items={"C1": [novelty]})
    _run(store)
    matrix = read_json(store.dir / "followup/round-1/followup_matrix.json")
    assert matrix["gate"]["flagged_ids"] == ["C1-I5"]  # no full text, so no independent check
    disp = read_json(store.sidecar_path("S2", "revision", ".json"))["data"]["item_dispositions"]
    assert disp == [{"item_id": "C1-I5", "disposition": "unverified_threat", "memo_sections": [12], "note": "mock"}]
    results = read_json(store.role_dir("verifier") / "results.json")
    assert results["requests"]["C1-I5:r1"]["status"] == "fulltext_unavailable"

    store = _create(sample_paper, items={"C1": [novelty]}, misplace={"S2": True})
    result, _ = _run(store)
    state = store.load_state()
    assert state["agents"]["S2"]["status"] == "quarantined"
    assert "not filed as unverified threats: C1-I5" in state["agents"]["S2"]["detail"]
    assert state["followup"]["current_memo"] == "S1" and state["followup"]["rounds"]["1"]["status"] == "incomplete"
    assert result["outcome"] == "incomplete"


def test_new_items_start_another_round_up_to_the_cap(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": [item(1)], "C2": [item(7, title="a different gap entirely")]})
    result, _ = _run(store)
    state = store.load_state()
    rounds = state["followup"]["rounds"]
    assert rounds["1"]["outcome"] == "another_round" and rounds["2"]["outcome"] in (followup.SATURATED,
                                                                                   "max_rounds_reached")
    assert {a: state["agents"][a]["round"] for a in ("A3", "A4", "S3", "C3")} == {"A3": 2, "A4": 2, "S3": 2,
                                                                                 "C3": 2}
    assert state["followup"]["current_memo"] == "S3"
    items2 = read_json(store.dir / "followup/round-2/items.json")
    assert items2["critic"] == "C2" and items2["items"][0]["id"] == "C2-I7"
    # F20: a manifest lists the memo, critic and adjudications the agent was shown, nothing else, each once
    shown = {"A3": {"S2", "C2"}, "S3": {"S2", "C2", "A3", "A4"}, "C3": {"S3", "C2", "A3", "A4"}}
    for aid, expected in shown.items():
        manifest = read_json(store.sidecar_path(aid, state["agents"][aid]["role"], ".context.json"))["manifest"]
        got = {m["agent_id"] for m in manifest if m["kind"] in ("synthesis", "revision", "critic", "recheck",
                                                                "adjudication")}
        assert got == expected, (aid, got)
        assert len(manifest) == len({tuple(sorted(m.items())) for m in manifest}), aid

    capped = _create(sample_paper, extra={"followup": {"max_rounds": 1}},
                     items={"C1": [item(1)], "C2": [item(7, title="a different gap entirely")]})
    _run(capped)
    rounds = capped.load_state()["followup"]["rounds"]
    assert list(rounds) == ["1"] and rounds["1"]["outcome"] == "max_rounds_reached"


def test_repeats_start_no_round_but_stay_open(runs_dir, sample_paper):
    store = _create(sample_paper, extra={"judge": {"agents": 1}},  # J1 rates nothing FATAL
                    items={"C1": [item(1)], "C2": [item(1, repeats_item="C1-I1")]})
    _run(store)
    result = read_json(store.dir / "followup/round-1/round.json")
    assert result["outcome"] == followup.SATURATED and result["disputed_repeats"][0]["repeats"] == "C1-I1"
    dispute = result["open_issues"]["blocking"]["reraised"][0]
    assert dispute["earlier_rulings"] == {"A1": "MAJOR_FIXABLE", "A2": "MAJOR_FIXABLE"}
    assert dispute["earlier_disposition"] == "incorporated"
    assert len(store.load_state()["followup"]["rounds"]) == 1  # no new round for a re-raised item
    assert "re-raised by the re-check critic: C2-I1 (repeats C1-I1" in service.status_text(store.run_id)


def test_ready_for_the_next_gate_when_nothing_serious_is_open(runs_dir, sample_paper):
    store = _create(sample_paper, extra={"judge": {"agents": 1}}, items={"C1": [item(1)]})
    result, _ = _run(store)
    assert "follow-up: ready_for_next_gate" in result["message"], result
    issues = read_json(store.dir / "followup/round-1/round.json")["open_issues"]
    assert not issues["blocking"] and issues["major"]  # fixable issues are listed, they do not block
    assert "ready for the next gate" in service.status_text(store.run_id)


class _Files:
    def __init__(self, root):
        self.dir = root

    def role_dir(self, role):
        return self.dir / {"judge": "judges"}[role]


def test_open_issues_rules(tmp_path):
    from paper_adversary.util import atomic_write_json

    store = _Files(tmp_path)
    atomic_write_json(tmp_path / "judges/judgment_matrix.json", {"rows": [
        {"id": "N1-O1", "title": "a", "verdicts": {"J1": {"severity": "FATAL"}, "J2": {"severity": "MAJOR_FIXABLE"}}},
        {"id": "N1-O2", "title": "b", "verdicts": {"J1": {"severity": "FATAL"}}},
        {"id": "R1-O1", "title": "c", "verdicts": {"J1": {"severity": "MAJOR_FIXABLE"}}}]})
    atomic_write_json(tmp_path / "judges/evidence_gate.json", {"flagged_ids": ["N1-O2"]})
    state = {"agents": {"A1": {"status": "complete"}, "A2": {"status": "complete"}},
             "followup": {"rounds": {"1": {"adjudicators": ["A1", "A2"]}}}}

    def round_with(updates):
        atomic_write_json(tmp_path / "followup/round-1/followup_matrix.json", {
            "rows": [{"id": "C1-I1", "title": "d", "verdicts": {"A1": {"severity": "MINOR"},
                                                                "A2": {"severity": "MAJOR_FIXABLE"}}}],
            "gate": {"flagged_ids": []}, "objection_updates": updates})
        return followup.open_issues(store, state)

    issues = round_with([{"adjudicator": "A1", "objection_id": "N1-O1", "severity": "MAJOR_FIXABLE"}])
    assert [x["id"] for x in issues["blocking"]["fatal"]] == ["N1-O1"]  # one adjudicator cannot overturn it
    assert [x["id"] for x in issues["blocking"]["unverified_fatal"]] == ["N1-O2"]
    both = [{"adjudicator": a, "objection_id": "N1-O1", "severity": "MAJOR_FIXABLE"} for a in ("A1", "A2")]
    issues = round_with(both)
    assert "fatal" not in issues["blocking"] or not issues["blocking"]["fatal"]
    assert {x["id"] for x in issues["major"]} == {"N1-O1", "R1-O1", "C1-I1"}
    issues = round_with([{"adjudicator": "A2", "objection_id": "R1-O1", "severity": "FATAL"}])
    assert {x["id"] for x in issues["blocking"]["fatal"]} == {"N1-O1", "R1-O1"}  # an adjudicator's FATAL counts too
    atomic_write_json(tmp_path / "judges/evidence_gate.json", {"flagged_ids": []})
    atomic_write_json(tmp_path / "judges/judgment_matrix.json", {"rows": [
        {"id": "R1-O1", "title": "c", "verdicts": {"J1": {"severity": "MAJOR_FIXABLE"}}}]})
    assert round_with([])["blocking"] == {}  # only MAJOR BUT FIXABLE left: ready
    assert round_with([])["major"] and followup.open_issues(store, state, [{"id": "C2-I1"}])["blocking"]["reraised"]


def test_an_interrupted_round_resumes(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": [item(1)]}, fail={"S2": ["invalid_request"]})
    result, _ = _run(store)
    state = store.load_state()
    assert result["outcome"] == "incomplete" and state["followup"]["rounds"]["1"]["status"] == "incomplete"
    attempts_a1 = state["agents"]["A1"]["attempts"]
    assert "followup" in service.resume_stages(state, "followup")
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"]["fail"] = {}
    store.config_path.write_text(yaml.safe_dump(raw))
    result, _ = _run(store, ["followup"])
    state = store.load_state()
    assert state["followup"]["rounds"]["1"]["status"] == "complete" and state["agents"]["S2"]["status"] == "complete"
    assert state["agents"]["A1"]["attempts"] == attempts_a1  # finished adjudicators are not rerun


def test_follow_up_needs_a_structured_critic_and_something_to_do(runs_dir, sample_paper):
    store = _create(sample_paper)  # the default mock critic raises one MINOR item
    result, _ = _run(store)
    rnd = store.load_state()["followup"]["rounds"]["1"]
    assert rnd["outcome"] == followup.SATURATED and rnd["reason"] == "nothing_to_follow_up"  # J2's FATALs stand
    assert not any(a["role"] == "adjudicator" for a in store.load_state()["agents"].values())
    old = _create(sample_paper, extra={"critic": {"prompt": "critic_v1"}})
    result, _ = _run(old)
    assert "wrote no structured items" in result["message"] or "follow-up" in result["message"]
    assert "followup" not in old.load_state() or not old.load_state()["followup"].get("rounds")


def test_dry_run_and_per_run_limits(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": ITEMS})
    _run(store, ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"])
    plan = service.followup_plan(store.run_id)
    assert plan.startswith("Follow-up round 1") and "adjudicators claude-fable-5-1 (max), claude-fable-5-1 (max)" in plan
    before = sha256_file(store.state_path)
    assert sha256_file(store.state_path) == before
    with pytest.raises(ConfigError, match="max_rounds"):
        load_config({"followup": {"max_rounds": 5}})
    with pytest.raises(ConfigError, match="adjudicators"):
        load_config({"followup": {"adjudicators": 4}})
    cfg, _ = load_config({"followup": {"max_rounds": 1, "auto_continue": False}})
    assert cfg.followup.max_rounds == 1 and not cfg.followup.auto_continue
    with pytest.raises(ConfigError, match="verifier.enabled"):  # the evidence gate's independent check stays
        load_config({"verifier": {"enabled": False}})
    with pytest.raises(ConfigError, match="evidence"):
        load_config({"evidence": {"accept_approximate_score": 0.9}})
    cfg, _ = load_config({"verifier": {"max_agents_per_batch": 2}})
    assert cfg.verifier.max_agents_per_batch == 2


def _set_mock(store, **mock):
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"].update(mock)
    store.config_path.write_text(yaml.safe_dump(raw))


def test_follow_up_ids_never_collide_with_base_agents(runs_dir, sample_paper):
    store = _create(sample_paper, trusted={"synthesis": {"agents": 2}}, items={"C1": [item(1)]})
    _run(store)
    state = store.load_state()
    rnd = state["followup"]["rounds"]["1"]
    assert state["agents"]["S2"]["role"] == "synthesis" and not state["agents"]["S2"].get("round")
    assert (rnd["revision"], rnd["recheck"]) == ("S3", "C2") and state["agents"]["S3"]["role"] == "revision"
    assert state["followup"]["current_memo"] == "S3"


def test_rerunning_the_base_memo_restarts_the_follow_up(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": [item(1)]})
    _run(store)
    _run(store, ["synthesis"], rerun=["S1"])
    state = store.load_state()
    assert state["followup"]["stale"] == "S1 was rerun" and state["followup"]["current_memo"] == "S1"
    result, _ = _run(store, ["followup"])  # C1 still describes the old memo
    assert result["outcome"] == "incomplete" and "C1 reviewed an earlier version of memo S1" in result["message"]
    state = store.load_state()
    old = ("A1", "A2", "S2", "C2")
    assert {a: state["agents"][a]["status"] for a in old} == dict.fromkeys(old, "superseded")
    assert "superseded_by" not in state["agents"]["S1"] and not state["followup"]["rounds"]
    assert state["followup_history"][0]["rounds"]["1"]["revision"] == "S2"
    assert list((store.dir / "archive").glob("*/followup/round-1/items.json"))  # archived, never deleted
    _run(store, ["critic"], rerun=["C1"])
    result, _ = _run(store, ["followup"])
    state = store.load_state()
    rnd = state["followup"]["rounds"]["1"]
    assert result["outcome"] == "complete", result
    assert (rnd["adjudicators"], rnd["revision"], rnd["recheck"]) == (["A3", "A4"], "S3", "C3")
    assert state["followup"]["current_memo"] == "S3" and state["agents"]["S1"]["superseded_by"] == "S3"
    assert len(state["followup_history"]) == 1
    assert ("Earlier follow-up agents (no current round lists them): A1 (adjudicator, superseded), A2 (adjudicator, "
            "superseded), C2 (re-check critic, superseded), S2 (revised memo, superseded)") in \
        service.status_text(store.run_id)


def test_rerunning_one_adjudicator_reopens_its_round(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": ITEMS})
    _run(store)
    attempts = {a: v["attempts"] for a, v in store.load_state()["agents"].items()}
    ids = service.resolve_followup_rerun(store.load_state(), ["a1"])
    assert ids == ["A1"]
    with pytest.raises(ValueError, match="not an agent of a current follow-up round"):
        service.resolve_followup_rerun(store.load_state(), ["J1"])
    result, _ = _run(store, ["followup"], rerun=ids)
    state = store.load_state()
    assert result["outcome"] == "complete" and state["agents"]["A1"]["attempts"] == attempts["A1"] + 1
    assert state["agents"]["A2"]["attempts"] == attempts["A2"]  # the other adjudicator is not rerun
    assert state["followup"]["rounds"]["1"]["status"] == "complete" and len(state["followup"]["rounds"]) == 1


def test_a_released_revision_becomes_the_current_memo(runs_dir, sample_paper):
    novelty = item(5, "novelty_to_verify", candidate_references=[{"title": "Some Earlier Paper",
                                                                  "arxiv_id": "2201.00001"}])
    store = _create(sample_paper, items={"C1": [novelty]}, misplace={"S2": True})
    _run(store)
    assert store.load_state()["agents"]["S2"]["status"] == "quarantined"
    service.release_quarantine(store.run_id, "S2", "the threat is named in section 5 on purpose here", "mcp")
    state = store.load_state()
    assert state["followup"]["current_memo"] == "S2" and state["agents"]["S1"]["superseded_by"] == "S2"
    assert service.get_report(store.run_id, "memo").find("(S2)") > 0


def test_an_adjudicator_that_skips_an_item_gets_a_supplement(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": ITEMS}, skip_item={"A1": "C1-I1"})
    _run(store)
    a1 = store.load_state()["agents"]["A1"]
    assert a1["status"] == "complete" and any("C1-I1" in w for w in a1["gate"]["warnings"])
    rulings = read_json(store.sidecar_path("A1", "adjudicator", ".json"))["data"]["rulings"]
    assert any(r["item_ids"] == ["C1-I1"] and r.get("origin") == "supplement" for r in rulings)
    prompt = next((store.dir / "logs/agents/A1").glob("attempt-1/supplement-1/user_prompt.md")).read_text()
    assert "<critic_report" in prompt and "<refuter_reports" not in prompt and "<judge_reports" not in prompt
    assert "CANARY_A2" not in prompt  # still blind to the other adjudicator


def test_a_revised_memo_that_misfiles_a_threat_is_fixed(runs_dir, sample_paper):
    novelty = item(5, "novelty_to_verify", candidate_references=[{"title": "Some Earlier Paper",
                                                                  "arxiv_id": "2201.00001"}])
    store = _create(sample_paper, items={"C1": [novelty]}, misfile={"S2": True})
    _run(store)
    state = store.load_state()
    assert state["agents"]["S2"]["status"] == "complete" and state["followup"]["current_memo"] == "S2"
    fix = read_json(store.gate_path("S2", "revision"))["placement_fix"]
    assert fix["outcome"] == "accepted" and "C1-I5" in fix["flagged"]
    _, body = read_report(store.report_path("S2", "revision"))
    assert "C1-I5" not in gates.lead_ids(gates.section_text(body, 5)) and "item_dispositions" in body


LEDGER = [{"id": "C1", "claim": "Mock claim", "location": "Abstract", "type": "empirical"},
          {"id": "C2", "claim": S_THEORY, "location": "Sec. 4", "type": "theoretical"}]


def test_critic_items_quote_the_submission_and_have_their_own_budget(runs_dir, sample_paper):
    """K1 pilot: every critic request ended claim_not_located (no submission words, only a free-text location),
    and the base batch had used the run-wide caps, so neither a round nor a gate re-check could start a verifier."""
    _seed(1, 2, 3)
    ref = {"title": _paper(3)[0], "arxiv_id": _paper(3)[1]}
    items = [item(1, "novelty_to_verify", candidate_references=[ref], submission_quote=S_INTRO),
             item(2, "novelty_to_verify", candidate_references=[ref], claim_ids=["C2"]),
             item(3, "novelty_to_verify", candidate_references=[ref], claim_ids=["c2"],
                  submission_quote="words that the submission never says about anything at all"),
             # as critic_v2 tells the critic to write them (review G3: they became 'LEDGERCLAIML2', never found)
             item(4, "novelty_to_verify", candidate_references=[ref], claim_ids=["ledger claim L2"]),
             item(5, "novelty_to_verify", candidate_references=[ref], claim_ids="ledger claim C2")]
    store = _create(sample_paper, extra={"verifier": {"max_papers_per_run": 1, "max_agents_per_run": 1}},
                    decisive={"N1": _claim(1, passages=[S_INTRO]), "N2": _claim(2, passages=[S_THEORY])},
                    items={"C1": items}, ledger=LEDGER)
    _run(store)
    reqs = read_json(store.role_dir("verifier") / "results.json")["requests"]
    capped = reqs["N1-O1:r1"]  # the refuters' budget went to N2-O1, which restates ledger claim C2
    assert capped["status"] == "not_run" and "ranked below N2-O1:r1, a threat to a central claim" in capped["reason"]
    critic = ("C1-I1:r1", "C1-I2:r1", "C1-I3:r1")
    assert [reqs[r]["located_by"] for r in critic] == ["submission_quote", "ledger claim C2", "ledger claim C2"]
    assert reqs["C1-I3:r1"]["not_located"] == ["submission_quote"]  # a quote not in the paper falls back
    assert [reqs[r]["located_by"] for r in ("C1-I4:r1", "C1-I5:r1")] == ["ledger claim L2", "ledger claim C2"]
    stored = {i["id"]: i["claim_ids"] for i in read_json(store.dir / "followup/round-1/items.json")["items"]}
    assert (stored["C1-I4"], stored["C1-I5"]) == (["L2"], ["C2"])
    assert {reqs[r]["status"] for r in critic} == {"verified"}  # the round's own budget
    batch = store.load_state()["verification"]["batches"]["critic1"]
    assert batch["status"] == "complete" and len(batch["agents"]) == 1
    for aid, role in (("A1", "adjudicator"), ("S2", "revision"), ("C2", "recheck")):  # F20: shown once
        user = (store.agent_log_dir(aid) / "attempt-1" / "user_prompt.md").read_text()
        for rid in ("C1-I1", "C1-I2", "C1-I3"):
            assert user.count(f"- {rid} vs ") == 1 and f"- {rid} vs " in user.split("<followup_verifications", 1)[1]
        manifest = read_json(store.sidecar_path(aid, role, ".context.json"))["manifest"]
        assert [m["path"] for m in manifest].count("verify/results.json") == 1, aid
    status = service.status_text(store.run_id)  # the round's verifier is listed with its round (F18)
    vid = batch["agents"][0]
    assert f"  {vid:<7}blind verifier: complete" in status
    base = status.split("Blind verifiers:", 1)[1].split("\n\n", 1)[0].split("Blind verification", 1)[0]
    assert f"  {vid} " not in base
    batch_id, _ = service.verification_batch(store.run_id, True, None)  # and a gate re-check has its own
    result, _ = _run(store, [f"verify:{batch_id}"])
    assert result["outcome"] == "complete", result
    assert read_json(store.role_dir("verifier") / "results.json")["requests"]["N1-O1:r1"]["status"] == "verified"
    zero = _create(sample_paper, extra={"verifier": {"max_papers_per_user_batch": 0}})
    with pytest.raises(ValueError, match="verifier.max_papers_per_user_batch is 0"):  # refused before any fetch
        service.verification_batch(zero.run_id, True, None)


def test_verification_requests_carry_the_quote_and_the_ledger_fallback():
    record = {"critic": "C1", "items": [
        {**item(1, "novelty_to_verify", claim_ids=["C9", "C2"], submission_quote=" the exact words "),
         "id": "C1-I1", "verify": True, "references": [{"title": "Prior", "arxiv_id": "2401.00003"}],
         "severity": "MAJOR_FIXABLE"}]}
    req, = followup.verification_requests(record, {"C2": "ledger claim two", "C3": "unused"})
    assert (req.claim_quote, req.quote_label) == ("the exact words", "submission_quote")
    assert req.alt_quotes == (("ledger claim C2", "ledger claim two"),) and req.claim_location == "Sec. 3"
    bare = {**record["items"][0]}
    for key in ("submission_quote", "claim_ids"):
        bare.pop(key)
    assert followup._item_core(bare) == {k: bare.get(k) for k in ("id", "title", "type", "location",
                                                                   "candidate_references", "severity_estimate")}
    # critic_v2 items cite claim_ids without a submission_quote: they keep the hash (and verdicts) they had
    assert followup._item_core({**bare, "claim_ids": ["C2"]}) == followup._item_core(bare)
    assert followup._item_core(record["items"][0])["claim_ids"] == ["C9", "C2"]


LEDGER_L = {"synthesis": {"prompt": "synthesis_v4"}, "critic": {"prompt": "critic_v3"},
            "followup": {"recheck": {"prompt": "critic_recheck_v2"}}}  # the prompts that cite L IDs


def test_claims_ledger_ids_never_collide_with_agent_ids(runs_dir, sample_paper):
    """K1 pilot (F19): the ledger numbered claims C1..C61, the critics' own IDs. On the prompts that cite L IDs
    (synthesis_v4, critic_v3, critic_recheck_v2), ledger IDs are L1, L2, ... by position, whatever the intake model
    wrote; the IDs it wrote (as earlier runs' ledgers show them) still resolve. Runs on the older prompts keep the
    intake's IDs, so their memo and critic cite what their ledger shows (review G3); a mix shows both."""
    from paper_adversary.util import atomic_write_json
    from paper_adversary.verification import ledger_claim_map, ledger_claims

    store = _create(sample_paper, extra=LEDGER_L, ledger=LEDGER)
    _run(store, ["intake"])
    ledger = (store.source_dir / "claims_ledger.md").read_text()
    assert "- **L1** (empirical) [Abstract]: Mock claim" in ledger and "(L1, L2, ...)" in ledger
    assert "**C1**" not in ledger and "**C2**" not in ledger
    routine = RunStore.open(service.create_run(str(sample_paper), config_override=mock_override(
        "routine", ledger=LEDGER, items={"C1": []}))["run_id"])
    _run(routine, ["intake", "novelty", "rigor", "fit", "judge", "synthesis"])
    shown = (routine.source_dir / "claims_ledger.md").read_text()
    assert "- **C1** (empirical) [Abstract]: Mock claim" in shown and "**L1**" not in shown  # as before L IDs
    s1 = (routine.agent_log_dir("S1") / "attempt-1" / "system_prompt.md").read_text()
    assert "ledger ID (C1, C2, ...)" in s1 and "- **C2** " in \
        (routine.agent_log_dir("S1") / "attempt-1" / "user_prompt.md").read_text()
    mixed = _create(sample_paper, extra={"synthesis": {"prompt": "synthesis_v4"}}, ledger=LEDGER)
    _run(mixed, ["intake"])
    assert "- **L2** = **C2** (theoretical) [Sec. 4]: " in (mixed.source_dir / "claims_ledger.md").read_text()
    claims = ledger_claim_map(store)
    assert claims["L2"] == claims["C2"] == S_THEORY and ledger_claims(store) == ["Mock claim", S_THEORY]
    side_path = store.sidecar_path("INTAKE", "intake", ".json")
    side = read_json(side_path)
    side["data"]["claims"] = [{"id": "L2", "claim": "first"}, {"id": "L1", "claim": "second"}]
    atomic_write_json(side_path, side)
    assert ledger_claim_map(store) == {"L1": "first", "L2": "second"}  # an L ID always means its position
    record = {"critic": "C1", "items": [
        {**item(1, "novelty_to_verify", claim_ids=["ledger claim L2"]), "id": "C1-I1", "verify": True,
         "references": [{"title": "Prior", "arxiv_id": "2401.00003"}]},
        {**item(2, "novelty_to_verify", claim_ids=["C2", "XC1"]), "id": "C1-I2", "verify": True,
         "references": [{"title": "Prior", "arxiv_id": "2401.00003"}]}]}
    new, old = followup.verification_requests(record, {"L2": "two", "C2": "two (as the run's ledger showed it)"})
    assert new.alt_quotes == (("ledger claim L2", "two"),)
    assert old.alt_quotes == (("ledger claim C2", "two (as the run's ledger showed it)"),)


def _k1_open_issues_store(tmp_path, updates=None):
    """judges/, followup/round-1/ files shaped like the K1 pilot's (F9, F12, F14): J1-J3 group the novelty overlap,
    the two fit/rigor pairs and some minor framing objections; C1-I3 and C1-I16 repeat F1-O2 and F1-O7."""
    from paper_adversary.reports import build_judgment_matrix, judge_judgments
    from paper_adversary.util import atomic_write_json

    def j(ids, sev="MAJOR_FIXABLE"):
        return {"objection_ids": ids, "title": "t", "severity": sev, "confidence": "medium"}
    nov = ["N1-O1", "N1-O2", "N1-O3", "N2-O1"]
    raw = {"J1": [j(nov), j(["N2-O2", "N1-O5"]), j(["R1-O4", "F1-O3"]), j(["R1-O5", "F1-O4"]), j(["F1-O2"]),
                  j(["N2-O3"], "MINOR"), j(["R1-O8"], "MINOR"), j(["F1-O7"], "MINOR")],
           "J2": [j(["N1-O1", "N1-O2", "N2-O1"]), j(["N2-O2"]), j(["R1-O4", "F1-O3"]), j(["R1-O5", "F1-O4"]),
                  j(["F1-O2"]), j(["N1-O3"], "MINOR"), j(["N1-O5"], "MINOR"), j(["N2-O3"], "MINOR"),
                  j(["R1-O8"], "MINOR"), j(["F1-O7"], "MINOR")],
           "J3": [j(nov), j(["N1-O5", "N2-O2", "N2-O3"]), j(["R1-O4", "F1-O3", "F1-O7"]), j(["R1-O5", "F1-O4", "R1-O8"]),
                  j(["F1-O2"])]}
    ids = sorted({i for items in raw.values() for x in items for i in x["objection_ids"]})
    objections = {"X": [{"id": i, "source": i[:2], "title": f"objection {i}"} for i in ids]}
    matrix = build_judgment_matrix(objections, {k: judge_judgments({"judgments": v}) for k, v in raw.items()})
    atomic_write_json(tmp_path / "judges/judgment_matrix.json", matrix)
    atomic_write_json(tmp_path / "judges/evidence_gate.json", {"flagged_ids": ["N1-O3", "N1-O5", "N2-O2", "N2-O3"]})
    both = {"A1": {"severity": "MAJOR_FIXABLE"}, "A2": {"severity": "MAJOR_FIXABLE"}}
    link = lambda is_new, covered, rerated=(): {"is_new": is_new, "covered_by": list(covered),  # noqa: E731
                                                "rerated": list(rerated)}
    atomic_write_json(tmp_path / "followup/round-1/followup_matrix.json", {
        "rows": [{"id": i, "title": f"item {i}", "verdicts": both} for i in ("C1-I1", "C1-I2", "C1-I3", "C1-I16")],
        "gate": {"flagged_ids": []},
        "objection_updates": [{"adjudicator": a, "item_ids": ["C1-I16"], "objection_id": "F1-O7",
                               "severity": "MAJOR_BUT_FIXABLE"} for a in ("A1", "A2")] + list(updates or []),
        "item_links": {"C1-I1": {a: link("yes", []) for a in ("A1", "A2")},
                       "C1-I2": {"A1": link("partially", ["F1-O8", "R1-O4", "F1-O3"]), "A2": link("yes", ["F1-O8"])},
                       "C1-I3": {a: link("partially", ["F1-O2"]) for a in ("A1", "A2")},
                       "C1-I16": {"A1": link("partially", ["F1-O7", "R1-O1", "F1-O1", "R1-O4"], ["F1-O7"]),
                                  "A2": link("partially", ["F1-O7", "R1-O4", "F1-O1", "F1-O3"], ["F1-O7"])}}})
    state = {"agents": {"A1": {"status": "complete"}, "A2": {"status": "complete"}},
             "followup": {"rounds": {"1": {"adjudicators": ["A1", "A2"]}}}}
    return _Files(tmp_path), state


def test_open_issues_count_issues_not_ids(tmp_path):
    """K1 pilot (F12, F14, F9): 'open MAJOR BUT FIXABLE: 23' counted objection IDs. IDs every judge ruled on in one
    judgment are one issue; an item every adjudicator said repeats one objection joins it; flagged and unflagged IDs
    never merge; IDs that are MAJOR only through grouped verdicts are counted apart."""
    store, state = _k1_open_issues_store(tmp_path)
    issues = followup.open_issues(store, state)
    assert len(issues["major"]) == 18 and not issues["blocking"]
    groups = {tuple(g["ids"]): g for g in issues["issues"]["major"]}
    assert ("N1-O1", "N1-O2", "N2-O1") in groups and ("F1-O3", "R1-O4") in groups and ("F1-O4", "R1-O5") in groups
    assert ("C1-I3", "F1-O2") in groups and ("C1-I16", "F1-O7") in groups  # repeats merged with their objection
    assert ("C1-I1",) in groups and ("C1-I2",) in groups  # A2 ruled C1-I2 new: not merged
    only = sorted(g["ids"][0] for g in issues["issues"]["major"] if g["grouped_only"])
    assert only == ["N1-O3", "N1-O5", "N2-O3", "R1-O8"]  # judges ruling on them alone said MINOR
    entry = {x["id"]: x for x in issues["major"]}
    assert entry["F1-O7"]["by"] == ["A1", "A2", "J3"] and not entry["F1-O7"]["grouped_only"]  # adjudicators agree
    assert entry["N2-O2"]["unverified"] and not entry["N2-O2"]["grouped_only"]  # J2 rated it MAJOR on its own
    assert entry["C1-I3"]["issue"] == "C1-I3" and entry["F1-O2"]["issue"] == "C1-I3"
    lines = followup.open_issue_lines(issues)
    assert lines == ["open MAJOR BUT FIXABLE (not blocking): 8 issue(s) over 14 ID(s); unverified prior work among "
                     "them: N2-O2",
                     "MAJOR BUT FIXABLE only through grouped verdicts, which judges ruling on them alone rated lower "
                     "(not counted above): N1-O3, N1-O5, N2-O3, R1-O8"]
    old = {"major": issues["major"]}  # a round.json written before issues were counted
    assert followup.open_issue_lines(old) == ["open MAJOR BUT FIXABLE (not blocking): 18 IDs"]


def test_every_adjudicator_re_rating_overturns_a_major_verdict(tmp_path):
    """K1 pilot (F15): C1-I11 disputed the grouped MAJOR on N2-O3, but nothing could carry a correction into the
    open issues. When every adjudicator of the round re-rates an objection, the more severe judge verdicts go."""
    down = [{"adjudicator": a, "item_ids": ["C1-I11"], "objection_id": "N2-O3", "severity": "MINOR"}
            for a in ("A1", "A2")]
    store, state = _k1_open_issues_store(tmp_path, down)
    issues = followup.open_issues(store, state)
    assert "N2-O3" not in {x["id"] for x in issues["major"]}
    assert issues["overturned"]["N2-O3"] == {"round": 1, "by": ["A1", "A2"], "verdicts": {"J3": "MAJOR_FIXABLE"}}
    store, state = _k1_open_issues_store(tmp_path, down[:1])  # one adjudicator alone overturns nothing
    issues = followup.open_issues(store, state)
    assert "N2-O3" in {x["id"] for x in issues["major"]} and "N2-O3" not in issues["overturned"]


def test_a_dispute_of_a_serious_verdict_is_adjudicated_and_can_correct_it(runs_dir, sample_paper):
    """K1 pilot (F15): a judging_flaw item the critic rated MINOR was 'noted', so its correction of a computed verdict
    could never be ruled on. It is now adjudicated, and the adjudicators' unanimous re-rating reaches the open issues;
    a critique that disputes nothing serious stays noted."""
    items = [item(1, "judging_flaw", severity="MINOR", objection_ids=["R1-O1"]),
             item(2, "minority_critique", severity="MINOR", objection_ids=["R1-O2"])]
    lower = [{"objection_id": "R1-O1", "severity": "MINOR", "rationale": "grouped inflation"}]
    store = _create(sample_paper, items={"C1": items}, ruling={"default": "MINOR"},
                    updates={"A1": {"C1-I1": lower}, "A2": {"C1-I1": lower}})
    _run(store)
    record = read_json(store.dir / "followup/round-1/items.json")
    routes = {i["id"]: (i["route"], i.get("route_reason")) for i in record["items"]}
    assert routes["C1-I1"] == ("adjudicate", "disputes the computed verdicts on R1-O1")  # J2 FATAL, J1 MAJOR
    assert routes["C1-I2"] == ("noted", None)  # R1-O2 has no serious verdict
    assert "routed to adjudication because it disputes the computed verdicts on R1-O1" in \
        (store.dir / "followup/round-1/items.md").read_text()
    issues = read_json(store.dir / "followup/round-1/round.json")["open_issues"]
    assert issues["overturned"]["R1-O1"]["verdicts"] == {"J1": "MAJOR_FIXABLE", "J2": "FATAL"}
    assert "R1-O1" not in {x["id"] for x in issues["blocking"].get("fatal") or []} | {x["id"] for x in issues["major"]}
    assert {x["id"] for x in issues["blocking"]["fatal"]} >= {"F1-O1", "R2-O1"}  # J2's other FATALs stand
    assert "Re-rated by every adjudicator of this round" in (store.dir / "followup/round-1/followup_matrix.md").read_text()


def test_recheck_items_at_the_round_cap_are_recorded_and_block_ready(runs_dir, sample_paper):
    """K1 pilot (F16, F17): at the round cap the re-check critic's two MAJOR items were left out of the open issues,
    and its six other items (two naming prior work) were recorded nowhere. Items below min_severity are recorded
    and never block, whichever critic raised them."""
    novelty = item(2, "novelty_to_verify", severity="MINOR", title="iterative amortized inference overlap",
                   candidate_references=[{"title": "Iterative Amortized Inference"}])
    c2 = [item(1, title="route class disconnects the fibre"), novelty, item(3, severity="MINOR", title="notation")]
    store = _create(sample_paper, extra={"followup": {"max_rounds": 1, "recheck": {"prompt": "critic_recheck_v2"}},
                                         "judge": {"agents": 1}, "critic": {"prompt": "critic_v3"}},
                    items={"C1": [item(1)], "C2": c2})
    _run(store)
    record = read_json(store.dir / "followup/round-1/round.json")
    assert record["outcome"] == "max_rounds_reached" and [x["id"] for x in record["new_items"]] == ["C2-I1"]
    assert [x["id"] for x in record["below_floor"]] == ["C2-I2", "C2-I3"]
    assert record["below_floor"][0]["references"] == ["Iterative Amortized Inference"]
    assert [x["id"] for x in record["open_issues"]["blocking"]["unadjudicated"]] == ["C2-I1"]
    status = service.status_text(store.run_id)
    assert ("Re-check C2: 3 item(s): 1 new at or above MAJOR BUT FIXABLE (C2-I1); 2 below MAJOR BUT FIXABLE, "
            "recorded only (C2-I2, C2-I3); suspected prior work never verified: C2-I2") in status
    assert "new items of the re-check critic that no round adjudicated: C2-I1 (MAJOR BUT FIXABLE)" in status
    assert "Suspected prior work in re-check items was never verified: C2-I2" in status
    # the critics are told what their severity does (F17): rendered with the run's own threshold
    c1 = (store.agent_log_dir("C1") / "attempt-1" / "system_prompt.md").read_text()
    assert "Items you rate MAJOR BUT FIXABLE or above are ruled on by independent adjudicators" in c1
    assert "{{" not in c1

    # Below the floor, a re-check item is only recorded, as the base critic's noted items are: with a FATAL floor
    # its MAJOR items neither start a round nor block 'ready', a re-raise of an adjudicated item included (it was
    # listed as "new ... that no round adjudicated").
    reraise = item(6, severity="MAJOR_FIXABLE", title="the same point again", repeats_item="C1-I1")
    fatal = _create(sample_paper, extra={"followup": {"max_rounds": 1, "min_severity": "FATAL",
                                                      "recheck": {"prompt": "critic_recheck_v2"}},
                                         "judge": {"agents": 1}},
                    items={"C1": [item(1, severity="FATAL")], "C2": [item(5, title="a different gap entirely"),
                                                                      reraise]})
    _run(fatal)
    record = read_json(fatal.dir / "followup/round-1/round.json")
    assert record["new_items"] == [] and record["outcome"] == followup.READY  # MAJOR, below a FATAL floor
    assert [(x["id"], x["repeats_item"]) for x in record["below_floor"]] == [("C2-I5", None), ("C2-I6", "C1-I1")]
    assert not record["open_issues"]["blocking"]
    assert "no round adjudicated" not in service.status_text(fatal.run_id)
    c2_prompt = (fatal.agent_log_dir("C2") / "attempt-1" / "system_prompt.md").read_text()
    assert "noted items (below FATAL) were not" in c2_prompt
    assert "items below FATAL are only recorded" in c2_prompt


def test_rubric_tests_are_ruled_on_by_every_judge_and_fatal_rulings_block(runs_dir, sample_paper, tmp_path):
    """K1 pilot (F22): the rubric's own FATAL tests were never raised by a refuter, so no judge ruled on them. A
    rubric may now declare tests; judge_v3 rules on each, coverage is checked (a supplement, then quarantine), the
    matrix tabulates the rulings, and a FATAL ruling blocks 'ready'. The judge's tool sentence matches its tools."""
    rubric = tmp_path / "rubric_with_tests.md"
    rubric.write_text("---\nname: with_tests\ntests:\n  - {id: RT1, text: 'Amortizing method X with a network is "
                      "incremental: FATAL'}\n  - {id: RT2, text: 'A theorem implied by first-order arguments does not "
                      "count'}\n---\nJudge as an expert reviewer would.\n")
    v3 = {"judge": {"prompt": "judge_v3"}}
    info = service.create_run(str(sample_paper), rubric=str(rubric), config_override=mock_override())
    assert any("declares tests, but judge prompt 'judge_v2'" in w for w in info["warnings"])
    store = _create_rubric(sample_paper, rubric, v3, items={"C1": [item(1)]}, rubric={"J1": {"RT1": "FATAL"},
                                                                                     "J2": {"RT2": "skip"}})
    _run(store)
    state = store.load_state()
    assert {a: state["agents"][a]["status"] for a in ("J1", "J2", "J3")} == dict.fromkeys(("J1", "J2", "J3"),
                                                                                         "complete")
    j1_user = (store.agent_log_dir("J1") / "attempt-1" / "user_prompt.md").read_text()
    assert "<rubric_tests" in j1_user and "- RT1: Amortizing method X with a network is incremental: FATAL" in j1_user
    j1_system = (store.agent_log_dir("J1") / "attempt-1" / "system_prompt.md").read_text()
    assert "You have no tools: everything you need is in this message" in j1_system and "paper.pdf" not in j1_system
    assert any("ruled on in a coverage supplement: RT2" in w for w in state["agents"]["J2"]["gate"]["warnings"])
    sup = next((store.dir / "logs/agents/J2").glob("attempt-1/supplement-1/user_prompt.md")).read_text()
    assert "- RT2: rubric test, not ruled on in your report" in sup and "<rubric_tests" in sup
    side = read_json(store.sidecar_path("J2", "judge", ".json"))
    assert {r["test_id"]: (r["applies"], r["severity"]) for r in side["rubric_rulings"]} == {
        "RT1": ("no", None), "RT2": ("partly", "MINOR")}
    matrix = service.get_report(store.run_id, "matrix")
    assert "## Rubric tests" in matrix and "| RT1 | Amortizing method X with a network is incremental: FATAL | " \
                                          "yes, FATAL | no | no |" in matrix
    assert "## Rubric tests" in (store.agent_log_dir("S1") / "attempt-1" / "user_prompt.md").read_text()
    fatal = read_json(store.dir / "followup/round-1/round.json")["open_issues"]["blocking"]["fatal"]
    rt1 = next(x for x in fatal if x["id"] == "RT1")
    assert rt1["by"] == ["J1"] and rt1["kind"] == "rubric_test"
    assert "RT1 (J1; rubric test)" in service.status_text(store.run_id)

    bad = _create_rubric(sample_paper, rubric, v3, rubric={"J1": {"RT1": "invalid"}}, supplement={"J1": "skip_tests"})
    _run(bad, ["intake", "novelty", "rigor", "fit", "judge"])
    j1 = bad.load_state()["agents"]["J1"]
    assert j1["status"] == "quarantined" and "rubric tests: invalid ruling: RT1" in j1["detail"]

    plain = _create(sample_paper, extra=v3)  # a rubric without tests: judge_v3 behaves as before
    _run(plain, ["intake", "novelty", "rigor", "fit", "judge"])
    assert "<rubric_tests" not in (plain.agent_log_dir("J1") / "attempt-1" / "user_prompt.md").read_text()
    assert not any(c["check"] == "rubric_tests" for c in read_json(plain.gate_path("J1", "judge"))["checks"])


def test_a_fatal_rubric_ruling_can_be_disputed_and_overturned(runs_dir, sample_paper, tmp_path):
    """Review (G2): a FATAL rubric-test ruling blocks 'ready' like a FATAL judgment, but an item disputing it was
    only 'noted' (disputes were read from objection rows alone), and adjudicator_v2 never said a test can be
    re-rated. It is now adjudicated whatever its rating, and the adjudicators' re-ratings overturn the ruling."""
    rubric = tmp_path / "rubric_with_tests.md"
    rubric.write_text("---\nname: with_tests\ntests:\n  - {id: RT1, text: 'Amortizing method X with a network is "
                      "incremental: FATAL'}\n---\nJudge as an expert reviewer would.\n")
    lower = [{"objection_id": "RT1", "severity": "NOT_CONVINCING", "rationale": "the method is not X"}]
    store = _create_rubric(sample_paper, rubric, {"judge": {"prompt": "judge_v3"},
                                                  "followup": {"adjudicator": {"prompt": "adjudicator_v2"}}},
                           rubric={"J1": {"RT1": "FATAL"}}, ruling={"default": "MINOR"},
                           items={"C1": [item(1, "judging_flaw", severity="MINOR", objection_ids=["RT1"])]},
                           updates={"A1": {"C1-I1": lower}, "A2": {"C1-I1": lower}})
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    it, = read_json(store.dir / "followup/round-1/items.json")["items"]
    assert (it["route"], it["route_reason"]) == ("adjudicate", "disputes the computed verdicts on RT1")
    assert it["cited_verdicts"] == {"RT1": {"J1": "FATAL"}} and it["cited_tests"] == ["RT1"]
    assert "cites RT1 (rubric test): J1 FATAL" in (store.dir / "followup/round-1/items.md").read_text()
    a1 = (store.agent_log_dir("A1") / "attempt-1" / "system_prompt.md").read_text()
    assert "A disputed rubric test (an ID such as RT1" in a1
    issues = read_json(store.dir / "followup/round-1/round.json")["open_issues"]
    assert issues["overturned"]["RT1"]["verdicts"] == {"J1": "FATAL"}
    assert "RT1" not in {x["id"] for x in (issues["blocking"].get("fatal") or [])}
    rows = {"N1-O1": {"verdicts": {"J1": {"severity": "MINOR"}}}}
    matrix = {"rubric_tests": [{"id": "RT2", "rulings": {"J1": {"applies": "no", "severity": None}}}]}
    flaw = {**item(2, "judging_flaw", objection_ids=["RT2"]), "severity": "MINOR"}
    assert followup._disputed(flaw, {**rows, **followup.rubric_test_rows(matrix)}) == []  # ruled not to apply


def test_refuters_on_the_new_prompts_read_the_rubric(runs_dir, sample_paper, tmp_path):
    """K1 pilot (F22): the rubric never reached the refuters, so none raised its FATAL tests. On novelty_v3, rigor_v2
    and fit_v2 every refuter gets the run's rubric, its tests included, passes the isolation guard and records the
    rubric; refuters on the earlier prompts still see only the paper."""
    rubric = tmp_path / "rubric_with_tests.md"
    rubric.write_text("---\nname: with_tests\ntests:\n  - {id: RT1, text: 'Amortizing method X with a network is "
                      "incremental: FATAL'}\n---\nJudge as an expert reviewer would.\n")
    new = {"novelty": {"prompt": "novelty_v3"}, "rigor": {"prompt": "rigor_v2"}, "fit": {"prompt": "fit_v2"}}
    store = _create_rubric(sample_paper, rubric, new)
    result, _ = _run(store, ["intake", "novelty", "rigor", "fit", "judge"])
    assert result["outcome"] == "complete", result
    agents = store.load_state()["agents"]
    refuters = {aid: a["role"] for aid, a in agents.items() if a["role"] in ("novelty", "rigor", "fit")}
    assert len(refuters) == 10
    for aid, role in refuters.items():
        assert agents[aid]["status"] == "complete", aid
        user = (store.agent_log_dir(aid) / "attempt-1" / "user_prompt.md").read_text()
        assert "<rubric source=" in user and "Amortizing method X with a network is incremental: FATAL" in user
        assert "CANARY_" not in user.split("</submission>", 1)[1]  # still no other agent's output
        kinds = {m["kind"] for m in read_json(store.sidecar_path(aid, role, ".context.json"))["manifest"]}
        assert kinds == {"paper", "rubric"}, aid
        assert read_report(store.report_path(aid, role))[0]["rubric"], aid
    assert "You have no tools" in (store.agent_log_dir("R1") / "attempt-1" / "system_prompt.md").read_text()
    old = _create_rubric(sample_paper, rubric)
    _run(old, ["intake", "novelty"])
    assert "<rubric" not in (old.agent_log_dir("N1") / "attempt-1" / "user_prompt.md").read_text()


def _create_rubric(sample_paper, rubric_file, extra=None, **mock):
    info = service.create_run(str(sample_paper), rubric=str(rubric_file),
                              config_override=deep_merge(mock_override(**mock), extra or {}))
    return RunStore.open(info["run_id"])


def test_which_items_dispute_a_computed_verdict():
    rows = {"N1-O1": {"verdicts": {"J1": {"severity": "MAJOR_FIXABLE"}, "J2": {"severity": "MINOR"}}},
            "R1-O1": {"verdicts": {"J1": {"severity": "MINOR"}}}}
    flaw = {**item(1, "judging_flaw", objection_ids=["N1-O1", "R1-O1"]), "severity": "MAJOR_FIXABLE"}  # as triaged
    assert followup._disputed(flaw, rows) == ["N1-O1"]  # a judging flaw may dispute the evidence status too
    assert followup._disputed({**flaw, "type": "minority_critique"}, rows) == []  # agrees with the MAJOR
    assert followup._disputed({**flaw, "type": "minority_critique", "severity": "FATAL"}, rows) == ["N1-O1"]
    assert followup._disputed({**flaw, "type": "new_issue"}, rows) == []
    assert followup._disputed({**flaw, "objection_ids": ["R1-O1"]}, rows) == []  # nothing serious to dispute
