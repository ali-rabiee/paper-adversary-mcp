"""An offline Stage-2 run end to end (mock provider), on the stage2_split preset as the Stage-2 batch uses it, with
the features the K1 audit fixes and their review touched working together: a rubric that declares tests, a judge_v3
grouping with member_severity, an objection the judges escalated (checked by the automatic gate batch), a critic's
novelty_to_verify item that quotes the submission, a follow-up round, and Fable's model-specific weekly limit
stopping only Fable agents while the Opus agents go on (in the refuter stage and inside the follow-up round)."""

import asyncio
import datetime as dt

import yaml

from conftest import mock_override
from paper_adversary import followup, service
from paper_adversary.pipeline import Pipeline, stale_agents
from paper_adversary.store import RunStore
from paper_adversary.util import read_json, read_jsonl, utcnow
from test_followup import item
from test_verification import S_INTRO, S_METHOD, S_THEORY, _claim, _paper, _seed

OPUS, FABLE = "claude-opus-5-5", "claude-fable-5-1"
RUBRIC = ("---\nname: stage2_with_tests\ntests:\n"
          "  - {id: RT1, text: 'Amortizing an existing per-query method with a network is incremental: FATAL'}\n"
          "  - {id: RT2, text: 'A theorem target implied by first-order null-space arguments does not count'}\n"
          "---\nThe submission is a concept at the design stage. End with 'Recommendation: <tier>'.\n")


def _run(store, stages, **kw):
    pipeline = Pipeline(store)
    return asyncio.run(pipeline.run(stages, **kw)), pipeline


def _set_mock(store, **mock):
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"].update(mock)
    store.config_path.write_text(yaml.safe_dump(raw))


def _events(store, name):
    return [e for e in read_jsonl(store.events_path) if e["event"] == name]


def _fable_limit(reset):
    return {"type": "seven_day_overage_included", "scope": "model", "reset_at": reset}


def test_a_stage2_run_end_to_end(runs_dir, sample_paper, tmp_path):
    _seed(1, 2, 3)
    rubric = tmp_path / "rubric_stage2.md"
    rubric.write_text(RUBRIC)
    reset = (utcnow() + dt.timedelta(days=5)).replace(microsecond=0).isoformat()
    novelty_item = item(1, "novelty_to_verify", title="the editor amortizes a known noise optimizer",
                        candidate_references=[{"title": _paper(3)[0], "arxiv_id": _paper(3)[1]}],
                        submission_quote=S_METHOD, claim_ids=["ledger claim L1"])
    mock = mock_override(
        "stage2_split",
        decisive={"N2": _claim(2, passages=[S_INTRO])},  # a decisive overlap, verified in the refuters' batch
        minor_prior={"N1": _claim(1, passages=[S_THEORY], severity="minor")},  # rated minor by its refuter ...
        rate={"J1": {"N1-O2": "MAJOR_FIXABLE"}},  # ... and MAJOR by a judge: the automatic gate batch checks it
        group={"J1": [{"ids": ["N1-O1", "R1-O1"], "severity": "MAJOR_FIXABLE",
                       "member_severity": {"R1-O1": "MINOR"}}]},
        rubric={"J1": {"RT1": "MAJOR_FIXABLE"}},
        items={"C1": [novelty_item, item(2, severity="MAJOR_FIXABLE", title="no ablation of the preservation term")]},
        fail={"R1": ["plan_limit"]}, limit={"R1": _fable_limit(reset)})
    info = service.create_run(str(sample_paper), rubric=str(rubric), config_override=mock,
                              trusted_config={"presets": {"selectable": ["routine", "stage2_split"]}})
    store = RunStore.open(info["run_id"])
    stages = service.full_review_stages(store, info)
    assert stages[-1] == "followup"

    # job 1: Fable's weekly window is full until days from now; only the Fable rigor refuter stops
    result, _ = _run(store, stages)
    agents = store.load_state()["agents"]
    assert result["outcome"] == "incomplete", result
    assert result["message"].startswith("stopped at Fable 5.1's model-specific plan limit (claude-fable-5-1; "
                                        f"seven_day_overage_included), which resets {reset}")
    assert {a: agents[a]["status"] for a in ("INTAKE", "N1", "N2", "F1")} == dict.fromkeys(
        ("INTAKE", "N1", "N2", "F1"), "complete")  # all on Opus 5.5
    assert {agents[a]["model"] for a in ("INTAKE", "N1", "N2", "F1")} == {OPUS}
    r1 = agents["R1"]
    assert r1["status"] == "interrupted" and r1["model"] == FABLE and reset in r1["detail"]
    assert agents["J1"]["status"] == agents["J2"]["status"] == "pending"
    assert len(_events(store, "plan_limit_skip")) == 1 and not (store.load_state().get("plan_limit") or {}).get("waits")
    assert "Next: Stopped at a plan limit" in service.status_text(store.run_id)

    # job 2, after the reset: everything up to the follow-up round, where Fable's limit is hit again (by A2)
    _set_mock(store, fail={"A2": ["plan_limit"]}, limit={"A2": _fable_limit(reset)})
    state = store.load_state()
    resume = service.resume_stages(state, "followup", stale_agents(store, state))
    assert resume == ["rigor", "judge", "synthesis", "critic", "followup"]  # judges add the verifiers
    result, _ = _run(store, resume, refresh_stale=True)
    assert result["outcome"] == "incomplete" and "follow-up" in result["message"], result
    state = store.load_state()
    agents = state["agents"]
    base = ("INTAKE", "N1", "N2", "R1", "F1", "J1", "J2", "S1", "C1")
    assert all(agents[a]["status"] == "complete" for a in base), {a: agents[a]["status"] for a in base}

    # the refuters' batch: N2's decisive overlap, on the other model than its Opus author
    batches = state["verification"]["batches"]
    reqs = read_json(store.role_dir("verifier") / "results.json")["requests"]
    tasks = read_json(store.role_dir("verifier") / "results.json")["tasks"]
    assert batches["refuters"]["status"] == "complete" and reqs["N2-O1:r1"]["status"] == "verified"
    assert tasks[reqs["N2-O1:r1"]["task_id"]]["verifier_model"] == "fable-5.1"

    # rubric tests: both judges rule on each; the refuters read the rubric, not its instructions for the judges
    matrix = read_json(store.role_dir("judge") / "judgment_matrix.json")
    tests = {t["id"]: t for t in matrix["rubric_tests"]}
    assert tests["RT1"]["rulings"]["J1"]["severity"] == "MAJOR_FIXABLE" and tests["RT1"]["rulings"]["J2"]["applies"] == "no"
    assert not tests["RT2"]["not_ruled_by"]
    r1_prompt = store.agent_log_dir("R1") / "attempt-2"
    assert "RT1" in (r1_prompt / "user_prompt.md").read_text()
    assert "Instructions in the rubric about the judges' report" in (r1_prompt / "system_prompt.md").read_text()

    # J1's judge_v3 group: each member at its own severity; the strongest one is J1's own rating, not 'grouped'
    rows = {r["id"]: r for r in matrix["rows"]}
    assert (rows["R1-O1"]["verdicts"]["J1"]["severity"], rows["R1-O1"]["verdicts"]["J1"]["grouped"]) == ("MINOR", False)
    assert (rows["N1-O1"]["verdicts"]["J1"]["severity"], rows["N1-O1"]["verdicts"]["J1"]["grouped"]) == \
        ("MAJOR_FIXABLE", False)

    # the automatic gate batch: N1-O2, escalated by J1, checked after the judges without rerunning them
    gate1 = batches["gate1"]
    assert state["verification"]["auto_gate"] == "gate1" and gate1["status"] == "complete"
    assert "automatic" in gate1["reason"] and reqs["N1-O2:r1"]["status"] == "verified"
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert "N1-O2" not in gate["flagged_ids"] and not gate["never_requested"]
    assert {agents[j]["attempts"] for j in ("J1", "J2")} == {1}
    assert "N1-O2 vs Guidance Schedulers Paper Number 1" in \
        (store.agent_log_dir("S1") / "attempt-1" / "user_prompt.md").read_text()

    # the memo and the critics cite the ledger by L IDs, as the ledger shows them
    assert "- **L1** " in (store.source_dir / "claims_ledger.md").read_text()

    # follow-up round 1: the critic's quoted item is verified on the other model than the Fable critic
    assert reqs["C1-I1:r1"]["located_by"] == "submission_quote" and reqs["C1-I1:r1"]["status"] == "verified"
    assert batches["critic1"]["status"] == "complete"
    assert tasks[reqs["C1-I1:r1"]["task_id"]]["verifier_model"] == "opus-5.5"
    items = {i["id"]: i for i in read_json(store.dir / "followup/round-1/items.json")["items"]}
    assert items["C1-I1"]["claim_ids"] == ["L1"] and items["C1-I1"]["route"] == "adjudicate"
    rnd = state["followup"]["rounds"]["1"]
    a1, a2 = rnd["adjudicators"]
    assert (agents[a1]["model"], agents[a1]["status"]) == (OPUS, "complete")  # Opus went on
    assert (agents[a2]["model"], agents[a2]["status"]) == (FABLE, "interrupted") and reset in agents[a2]["detail"]
    assert rnd["status"] != "complete" and agents[rnd["revision"]]["status"] != "complete"

    # job 3, after the reset: the round is finished, nothing earlier is redone
    _set_mock(store, fail={})
    state = store.load_state()
    resume = service.resume_stages(state, "followup", stale_agents(store, state))
    assert resume == ["followup"]
    result, _ = _run(store, resume, refresh_stale=True)
    assert result["outcome"] == "complete", result
    state = store.load_state()
    agents = state["agents"]
    rnd = state["followup"]["rounds"]["1"]
    assert rnd["status"] == "complete" and all(agents[a]["status"] == "complete"
                                               for a in (a1, a2, rnd["revision"], rnd["recheck"]))
    assert agents[a1]["attempts"] == 1 and state["followup"]["current_memo"] == rnd["revision"]
    record = read_json(store.dir / "followup/round-1/round.json")
    assert record["outcome"] in (followup.READY, followup.SATURATED)
    status = service.status_text(store.run_id)
    assert "Follow-up round 1: complete" in status and "Blind verification (gate1): complete" in status
    assert not stale_agents(store, state)
