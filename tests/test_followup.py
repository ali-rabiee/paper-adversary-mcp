"""Follow-up rounds after the completeness critic (mock provider): triage, blind verification, adjudication,
the revised memo, the re-check critic, the stop rule, isolation inside rounds, resume."""

import asyncio
import re

import pytest
import yaml

from conftest import mock_override
from paper_adversary import service
from paper_adversary.config import ConfigError, deep_merge, load_config
from paper_adversary.isolation import IsolationGuard, IsolationViolation, position
from paper_adversary.pipeline import Pipeline
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore
from paper_adversary.util import read_json, read_jsonl, sha256_file

FULL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic", "followup"]


def item(i, kind="new_issue", severity="MAJOR_FIXABLE", **extra):
    return {"id": f"I{i}", "type": kind, "title": f"critic item {i} about {kind}", "location": "Sec. 3",
            "argument": f"argument {i}", "severity_estimate": severity, "confidence": "medium", "claim_ids": [],
            "objection_ids": [], "judge_ids": [], "memo_sections": [], "candidate_references": [], **extra}


ITEMS = [item(1), item(2, "synthesis_flaw"), item(3, severity="MINOR"),
         item(4, "minority_critique", objection_ids=["Z9-O1"])]


def _create(sample_paper, extra=None, **mock):
    info = service.create_run(str(sample_paper), config_override=deep_merge(mock_override(**mock), extra or {}))
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
    assert result["outcome"] == "complete" and "follow-up: converged" in result["message"], result
    state = store.load_state()
    agents = state["agents"]
    for aid, role in (("A1", "adjudicator"), ("A2", "adjudicator"), ("S2", "revision"), ("C2", "recheck")):
        assert agents[aid]["status"] == "complete" and agents[aid]["round"] == 1, aid
        assert store.report_path(aid, role).is_file()
    rnd = state["followup"]["rounds"]["1"]
    assert rnd["status"] == "complete" and rnd["outcome"] == "converged" and rnd["memo_before"] == "S1"
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
    assert rounds["1"]["outcome"] == "another_round" and rounds["2"]["outcome"] in ("converged",
                                                                                   "max_rounds_reached")
    assert {a: state["agents"][a]["round"] for a in ("A3", "A4", "S3", "C3")} == {"A3": 2, "A4": 2, "S3": 2,
                                                                                 "C3": 2}
    assert state["followup"]["current_memo"] == "S3"
    items2 = read_json(store.dir / "followup/round-2/items.json")
    assert items2["critic"] == "C2" and items2["items"][0]["id"] == "C2-I7"

    capped = _create(sample_paper, extra={"followup": {"max_rounds": 1}},
                     items={"C1": [item(1)], "C2": [item(7, title="a different gap entirely")]})
    _run(capped)
    rounds = capped.load_state()["followup"]["rounds"]
    assert list(rounds) == ["1"] and rounds["1"]["outcome"] == "max_rounds_reached"


def test_repeats_do_not_count_as_new(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": [item(1)], "C2": [item(1, repeats_item="C1-I1")]})
    _run(store)
    result = read_json(store.dir / "followup/round-1/round.json")
    assert result["outcome"] == "converged" and result["disputed_repeats"][0]["repeats"] == "C1-I1"


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
    assert store.load_state()["followup"]["rounds"]["1"]["outcome"] == "nothing_to_follow_up"
    assert not any(a["role"] == "adjudicator" for a in store.load_state()["agents"].values())
    old = _create(sample_paper, extra={"critic": {"prompt": "critic_v1"}})
    result, _ = _run(old)
    assert "wrote no structured items" in result["message"] or "follow-up" in result["message"]
    assert "followup" not in old.load_state() or not old.load_state()["followup"].get("rounds")


def test_dry_run_and_per_run_limits(runs_dir, sample_paper):
    store = _create(sample_paper, items={"C1": ITEMS})
    _run(store, ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"])
    plan = service.followup_plan(store.run_id)
    assert plan.startswith("Follow-up round 1") and "2 × claude-fable-5-1" in plan
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
    store = _create(sample_paper, extra={"synthesis": {"agents": 2}}, items={"C1": [item(1)]})
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
