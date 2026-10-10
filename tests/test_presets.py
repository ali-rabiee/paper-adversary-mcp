"""Presets and mixed-model panels (mock provider, offline): preset contents and validation, per-run limits,
saved assignments, cross-model verifier routing and caps, peer isolation with mixed models (repair, supplement
and follow-up calls included), stale reruns on resume, the plan-limit wait cap, and runs saved before presets."""

import asyncio
import shutil

import pytest
import yaml

from conftest import mock_override
from test_verification import PRIOR_TEXT, seed_prior
from paper_adversary import config as config_module
from paper_adversary import service
from paper_adversary.config import ConfigError, check_config, deep_merge, load_config, plan_agents
from paper_adversary.pipeline import Pipeline, stale_agents
from paper_adversary.registry import ModelRegistry
from paper_adversary.store import RunStore
from paper_adversary.util import config_dir, read_json, read_jsonl
from paper_adversary.verification import is_central, priority

REG = ModelRegistry.load()
OPUS, FABLE = "claude-opus-5-5", "claude-fable-5-1"


def _create(sample_paper, preset="routine", extra=None, trusted=None, **mock):
    info = service.create_run(str(sample_paper), config_override=deep_merge(mock_override(preset, **mock), extra or {}),
                              trusted_config=trusted)
    return RunStore.open(info["run_id"])


def _run(store, stages=("intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic", "followup"), **kw):
    pipeline = Pipeline(store)
    return asyncio.run(pipeline.run(list(stages), **kw)), pipeline


def _models(cfg):
    return {s.agent_id: (s.model_id, s.effort, (s.lens or {}).get("id")) for s in plan_agents(cfg, REG)}


def _set_mock(store, **mock):
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"].update(mock)
    store.config_path.write_text(yaml.safe_dump(raw))


# ---------------------------------------------------------------- presets


def test_routine_is_the_default_and_matches_the_specified_panel():
    cfg, _ = load_config()
    assert cfg.preset == "routine" and [i for i in check_config(cfg, REG) if i.level == "error"] == []
    assert _models(cfg) == {
        "INTAKE": (OPUS, "medium", None),
        "N1": (OPUS, "high", "direct_and_recent"), "N2": (FABLE, "high", "equivalents_and_gaps"),
        "R1": (FABLE, "xhigh", "theory"), "R2": (OPUS, "high", "empirical_and_claims"),
        "F1": (OPUS, "medium", "fit_overall"),
        "J1": (OPUS, "high", None), "J2": (FABLE, "high", None),
        "S1": (OPUS, "high", None), "C1": (FABLE, "high", None)}
    assert (cfg.gates.repair.model, cfg.gates.repair.effort) == ("opus-5.5", "low")
    v, f = cfg.verifier, cfg.followup
    assert v.routing == "cross_model" and v.model_for("opus-5.5") == "fable-5.1" and v.model_for("fable-5.1") == "opus-5.5"
    assert v.max_papers_per_run == 3 and f.max_rounds == 1 and f.min_severity == "MAJOR_FIXABLE"
    assert [f.adjudicator.for_agent(k)[:2] for k in (1, 2)] == [("opus-5.5", "high"), ("fable-5.1", "high")]
    assert (f.revision.model, f.revision.effort, f.recheck.model, f.recheck.effort) == ("opus-5.5", "high",
                                                                                        "fable-5.1", "high")


def test_milestone_widens_routine_without_max_effort():
    cfg, raw = load_config(preset="milestone")
    models = _models(cfg)
    novelty = [models[f"N{i}"][2] for i in range(1, 5)]
    rigor = [models[f"R{i}"][2] for i in range(1, 4)]
    assert len(set(novelty)) == 4 and len(set(rigor)) == 3  # distinct lenses, no redundant assignment
    assert {models[f"N{i}"][0] for i in range(1, 5)} == {OPUS, FABLE}  # still mixed
    assert models["J1"][1] == models["J2"][1] == models["C1"][1] == "xhigh"
    assert cfg.verifier.max_papers_per_run == 6 and cfg.followup.max_rounds == 2
    for name in ("routine", "milestone"):  # no effort: max anywhere (roles, agents, verifier, follow-up, repair)
        assert config_module._max_efforts(load_config(preset=name)[1]) == [], name
    assert config_module._max_efforts(load_config(preset="legacy_thorough")[1])  # the old split does use max


@pytest.mark.parametrize("trusted, message", [
    ({"judge": {"per_agent": {3: {"model": "opus-5.5"}}}}, "there is no agent 3"),
    ({"judge": {"per_agent": {1: {"model": "gpt-9"}}}}, "unknown model alias"),
    ({"rigor": {"per_agent": {2: {"effort": "extreme"}}}}, "effort 'extreme'"),
    ({"novelty": {"per_agent": {1: {"lens": "astrology"}}}}, "lens 'astrology'"),
    ({"novelty": {"per_agent": {2: {"lens": "direct_and_recent"}}}}, "assigned to more than one agent"),
    ({"verifier": {"counterpart": {"opus-5.5": "gpt-9"}}}, "unknown model alias 'gpt-9'"),
])
def test_per_agent_overrides_are_validated(trusted, message):
    cfg, _ = load_config(None, trusted)
    errors = [str(i) for i in check_config(cfg, REG) if i.level == "error"]
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize("override, message", [
    ({"judge": {"per_agent": {1: {"effort": "max"}}}}, "judge.per_agent.1.effort (max effort)"),
    ({"followup": {"revision": {"effort": "max"}}}, "followup.revision.effort (max effort)"),
    ({"gates": {"repair": {"effort": "max"}}}, "gates.repair.effort (max effort)"),
    ({"novelty": {"agents": 3}}, "novelty.agents (above the configured 2)"),
    ({"verifier": {"max_papers_per_run": 4}}, "verifier.max_papers_per_run"),
    ({"verifier": {"max_papers_per_followup_round": 3}}, "verifier.max_papers_per_followup_round"),
    ({"verifier": {"max_agents_per_user_batch": 5}}, "verifier.max_agents_per_user_batch"),
    ({"verifier": {"auto_gate_batch": True}}, "verifier.auto_gate_batch (turning it on)"),
    ({"plan_limit": {"max_waits_per_run": 2}}, "plan_limit.max_waits_per_run"),
    ({"plan_limit": {"model_window_reserve": 0}}, "plan_limit.model_window_reserve"),
    ({"presets": {"selectable": ["mine"]}}, "presets"),
])
def test_model_chosen_overrides_cannot_raise_limits_or_enable_max(override, message):
    with pytest.raises(ConfigError, match=message.replace("(", r"\(").replace(")", r"\)")):
        load_config(override)


def test_stage2_split_uses_the_audited_prompts_and_budgets():
    cfg, _ = load_config(None, {"preset": "stage2_split"})
    assert [i for i in check_config(cfg, REG) if i.level == "error"] == []
    v = cfg.verifier
    assert (v.prompt, cfg.critic.prompt, cfg.followup.recheck.prompt) == ("verifier_v2", "critic_v3",
                                                                          "critic_recheck_v2")
    assert (cfg.judge.prompt, cfg.followup.adjudicator.prompt) == ("judge_v3", "adjudicator_v2")
    assert (cfg.novelty.prompt, cfg.rigor.prompt, cfg.fit.prompt, cfg.synthesis.prompt) == (
        "novelty_v3", "rigor_v2", "fit_v2", "synthesis_v4")
    # the user's 'lighter Fable panel' (2026-10-02): two Fable max judges, one adjudicator on each model, a high re-check
    assert (cfg.judge.agents, cfg.judge.model, cfg.judge.effort) == (2, "fable-5.1", "max")
    fu = cfg.followup
    assert [fu.adjudicator.for_agent(i)[:2] for i in (1, 2)] == [("opus-5.5", "high"), ("fable-5.1", "max")]
    assert (fu.recheck.model, fu.recheck.effort, fu.revision.model, fu.revision.effort) == (
        "fable-5.1", "high", "fable-5.1", "max")
    assert [(r.model, r.effort, r.agents) for r in (cfg.novelty, cfg.rigor, cfg.fit, cfg.synthesis, cfg.critic)] == [
        ("opus-5.5", "high", 2), ("fable-5.1", "max", 1), ("opus-5.5", "high", 1), ("fable-5.1", "max", 1),
        ("fable-5.1", "xhigh", 1)]
    assert (v.model_for("opus-5.5"), v.effort) == ("fable-5.1", "high")
    assert v.auto_gate_batch and (v.max_papers_per_run, v.max_papers_per_followup_round, v.max_papers_per_user_batch,
                                  v.max_agents_per_followup_round, v.max_agents_per_user_batch) == (3, 2, 2, 2, 2)
    assert cfg.plan_limit.model_window_reserve == 0.15  # the K1 pilot's share of Fable's weekly window
    routine = load_config()[0]  # the other presets keep their prompt versions; the new budgets are shared defaults
    assert routine.plan_limit.model_window_reserve == 0  # the window guard is off elsewhere
    assert (routine.verifier.prompt, routine.critic.prompt, routine.followup.recheck.prompt) == (
        "verifier_v1", "critic_v2", "critic_recheck_v1")
    assert (routine.judge.prompt, routine.followup.adjudicator.prompt) == ("judge_v2", "adjudicator_v1")
    assert (routine.novelty.prompt, routine.rigor.prompt, routine.fit.prompt, routine.synthesis.prompt) == (
        "novelty_v2", "rigor_v1", "fit_v1", "synthesis_v3")
    assert not routine.verifier.auto_gate_batch and routine.verifier.max_papers_per_followup_round == 2
    assert load_config({"verifier": {"max_papers_per_user_batch": 1}})[0].verifier.max_papers_per_user_batch == 1


def test_stage2_split_runs_end_to_end_on_its_own_prompts(runs_dir, sample_paper, tmp_path):
    """The preset as the Stage-2 runs use it (trusted, with a rubric that declares tests): every selected prompt
    renders and passes its gates, two judges rule on the rubric tests, the refuters read the rubric, the memo cites
    L IDs, and the follow-up panel is one adjudicator per model with a high-effort re-check."""
    rubric = tmp_path / "rubric_with_tests.md"
    rubric.write_text("---\nname: with_tests\ntests:\n  - {id: RT1, text: 'Amortizing method X with a network is "
                      "incremental: FATAL'}\n---\nThe submission is a concept at the design stage.\n")
    major = {"id": "I1", "type": "new_issue", "title": "an untested gap", "location": "Sec. 3", "argument": "a",
             "severity_estimate": "MAJOR_FIXABLE", "confidence": "medium", "claim_ids": ["L1"],
             "objection_ids": [], "judge_ids": [], "memo_sections": [], "candidate_references": []}
    info = service.create_run(str(sample_paper), rubric=str(rubric),
                              config_override=mock_override("stage2_split", items={"C1": [major]}),
                              trusted_config={"presets": {"selectable": ["routine", "stage2_split"]}})
    store = RunStore.open(info["run_id"])
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    agents = store.load_state()["agents"]
    assert {aid for aid, a in agents.items() if a["role"] == "judge"} == {"J1", "J2"}
    assert all(a["status"] == "complete" for a in agents.values() if a["role"] != "verifier"), {
        aid: a["status"] for aid, a in agents.items()}
    panel = {aid: (agents[aid]["model"], agents[aid]["effort"]) for aid in ("A1", "A2", "S2", "C2")}
    assert panel == {"A1": (OPUS, "high"), "A2": (FABLE, "max"), "S2": (FABLE, "max"), "C2": (FABLE, "high")}
    prompts = {aid: a["prompt"] for aid, a in agents.items() if a["role"] != "verifier"}
    assert prompts == {"INTAKE": "intake_v1", "N1": "novelty_v3", "N2": "novelty_v3", "R1": "rigor_v2",
                       "F1": "fit_v2", "J1": "judge_v3", "J2": "judge_v3", "S1": "synthesis_v4", "C1": "critic_v3",
                       "A1": "adjudicator_v2", "A2": "adjudicator_v2", "S2": "synthesis_revision_v2",
                       "C2": "critic_recheck_v2"}
    log = lambda aid, name: (store.agent_log_dir(aid) / "attempt-1" / name).read_text()  # noqa: E731
    for aid in ("N1", "R1", "F1"):
        assert "<rubric source=" in log(aid, "user_prompt.md") and "RT1" in log(aid, "user_prompt.md")
    assert "<rubric_tests" in log("J2", "user_prompt.md") and "(L1, L2, ...)" in log("S1", "system_prompt.md")
    assert "- **L1** " in log("S1", "user_prompt.md")


def test_presets_are_picked_by_name_and_trusted_settings_may_raise():
    assert load_config({"preset": "legacy_thorough"})[0].novelty.agents == 4
    assert load_config(None, None, "milestone")[0].novelty.agents == 4
    with pytest.raises(ConfigError, match="unknown preset 'galaxy'"):
        load_config(None, None, "galaxy")
    with pytest.raises(ConfigError, match="file stems"):
        load_config(None, None, "../default")
    with pytest.raises(ConfigError, match="cannot be chosen per run"):
        load_config({"preset": "milestone"}, {"presets": {"selectable": ["routine"]}})
    cfg, _ = load_config(None, {"rigor": {"per_agent": {1: {"effort": "max"}}}, "novelty": {"agents": 3}})
    assert cfg.rigor.for_agent(1)[1] == "max" and cfg.novelty.agents == 3  # the user's own settings may
    lowered, _ = load_config({"judge": {"per_agent": {2: {"model": "opus-5.5"}}}, "novelty": {"agents": 1}})
    assert lowered.judge.for_agent(2)[0] == "opus-5.5" and lowered.novelty.agents == 1


def test_a_run_keeps_its_assignment_when_the_preset_changes(runs_dir, sample_paper, tmp_path, monkeypatch):
    store = _create(sample_paper, items={"C1": []})
    assignment = read_json(store.dir / "assignment.json")
    assert assignment["preset"] == "routine" and store.load_metadata()["preset"] == "routine"
    assert {a["agent_id"]: a["model"] for a in assignment["agents"]}["N2"] == FABLE
    # someone edits the routine preset afterwards: this run must not notice
    other = tmp_path / "config"
    shutil.copytree(config_dir(), other)
    routine = yaml.safe_load((other / "presets" / "routine.yaml").read_text())
    routine["novelty"]["per_agent"][2]["model"] = "opus-5.5"
    routine["followup"]["revision"]["model"] = "fable-5.1"
    (other / "presets" / "routine.yaml").write_text(yaml.safe_dump(routine))
    monkeypatch.setattr(config_module, "config_dir", lambda: other)
    assert load_config()[0].novelty.for_agent(2)[0] == "opus-5.5"  # new runs would see the change
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    used = {u["agent_id"]: u["model_requested"] for u in read_jsonl(store.dir / "logs" / "usage.jsonl")}
    assert used["N2"] == FABLE and used["J2"] == FABLE  # the saved assignment, not the edited preset


# ---------------------------------------------------------------- full runs


@pytest.mark.parametrize("preset", ["routine", "milestone"])
def test_full_mock_runs_complete_with_mixed_models_and_blind_peers(runs_dir, sample_paper, preset):
    store = _create(sample_paper, preset)
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    state = store.load_state()
    assert all(a["status"] == "complete" for a in state["agents"].values() if not a.get("round"))
    by_agent = {}
    for u in read_jsonl(store.dir / "logs" / "usage.jsonl"):
        by_agent.setdefault(u["agent_id"], set()).add((u["model_requested"], u["effort"]))
    assert by_agent["N1"] == {(OPUS, "high")} and by_agent["N2"] == {(FABLE, "high")}
    assert by_agent["J2"] == {(FABLE, "xhigh" if preset == "milestone" else "high")}

    def seen(aid, role):
        return (store.dir / "logs" / "agents" / aid / "attempt-1" / "user_prompt.md").read_text()
    refuters = [a for a, v in state["agents"].items() if v["role"] == "novelty"]
    for a in refuters:  # refuters never see each other
        assert not any(f"CANARY_{b}" in seen(a, "novelty") for b in refuters if b != a)
    assert "CANARY_J2" not in seen("J1", "judge") and "CANARY_J1" not in seen("J2", "judge")


def test_mixed_panels_keep_peers_blind_in_repair_supplement_and_follow_up_calls(runs_dir, sample_paper):
    item = {"id": "I1", "type": "new_issue", "title": "an unmeasured claim", "location": "Sec. 3",
            "argument": "a", "severity_estimate": "MAJOR_FIXABLE", "confidence": "medium", "claim_ids": [],
            "objection_ids": [], "judge_ids": [], "memo_sections": [], "candidate_references": []}
    store = _create(sample_paper, omit_json=["N1"], coverage={"J1": 0.5}, items={"C1": [item]})
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    logs = store.dir / "logs" / "agents"
    repair = next(logs.glob("N1/attempt-1/repair-1/user_prompt.md")).read_text()
    supplement = next(logs.glob("J1/attempt-1/supplement-1/user_prompt.md")).read_text()
    assert "CANARY_N2" not in repair and "CANARY_J2" not in supplement
    assert "CANARY_A2" not in (logs / "A1/attempt-1/user_prompt.md").read_text()
    assert "CANARY_A1" not in (logs / "A2/attempt-1/user_prompt.md").read_text()
    usage = read_jsonl(store.dir / "logs" / "usage.jsonl")
    calls = {(u["agent_id"], u["stage"]): (u["model_requested"], u["effort"]) for u in usage}
    assert calls[("N1", "repair")] == (OPUS, "low")  # format repair: Opus 5.5 low
    assert calls[("J1", "supplement")] == (OPUS, "high")  # the judge's own model
    assert calls[("A1", "round1:adjudicator")] == (OPUS, "high") and calls[("A2", "round1:adjudicator")] == (FABLE, "high")
    assert calls[("S2", "round1:revision")] == (OPUS, "high") and calls[("C2", "round1:recheck")] == (FABLE, "high")


# ---------------------------------------------------------------- cross-model verification


def _prior(n):
    return f"arxiv:2401.0000{n}", f"2401.0000{n}", f"Guidance Schedulers Paper Number {n}"


def _decisive(n, severity="fatal"):
    key, ident, title = _prior(n)
    return {"category": "already_done", "severity": severity, "basis": "full_text",
            "prior": {"title": title, "arxiv_id": ident},
            "prior_passage": "monotonically increasing guidance schedules, including a simple linear ramp from zero "
                             "to the maximum weight, consistently improve sample quality"}


def test_verifiers_run_on_the_other_model_and_the_paper_cap_holds(runs_dir, sample_paper):
    for n in range(1, 5):
        key, ident, title = _prior(n)
        seed_prior(key=key, ident=ident, text=PRIOR_TEXT.replace("Analysis of Guidance Weight Schedulers", title),
                   title=title)
    # four refuters (trusted: more than routine's two); N1 and N2 cite paper 1, so it gets one verifier
    store = _create(sample_paper, trusted={"novelty": {"agents": 4}},
                    decisive={"N1": _decisive(1), "N2": _decisive(1, "major"), "N3": _decisive(3),
                              "N4": _decisive(4, "major")})
    _run(store, ("intake", "novelty", "rigor", "fit", "judge"))
    state = store.load_state()
    models = {a: v["model_alias"] for a, v in state["agents"].items()}
    assert (models["N1"], models["N2"], models["N3"], models["N4"]) == ("opus-5.5", "fable-5.1", "opus-5.5", "opus-5.5")
    results = read_json(store.role_dir("verifier") / "results.json")
    tasks = {t["prior_key"]: t for t in results["tasks"].values()}
    assert set(tasks) == {"arxiv:2401.00001", "arxiv:2401.00003", "arxiv:2401.00004"}  # one task per paper
    one = tasks["arxiv:2401.00001"]
    assert one["routed_by"] == "N1-O1:r1" and one["verifier_model"] == "fable-5.1"  # N1 (fatal) outranks N2
    assert one["same_model_authors"] == ["N2"] and sorted(one["requests"]) == ["N1-O1:r1", "N2-O1:r1"]
    assert tasks["arxiv:2401.00003"]["verifier_model"] == "fable-5.1"
    verifiers = {a: v["model_alias"] for a, v in state["agents"].items() if v["role"] == "verifier"}
    assert sorted(verifiers.values()) == ["fable-5.1", "fable-5.1", "fable-5.1"]

    capped = _create(sample_paper, trusted={"novelty": {"agents": 4}},
                     decisive={"N1": _decisive(1), "N2": _decisive(2), "N3": _decisive(3),
                               "N4": _decisive(4, "major")})
    _run(capped, ("intake", "novelty", "rigor", "fit", "judge"))
    results = read_json(capped.role_dir("verifier") / "results.json")
    assert len({t["prior_key"] for t in results["tasks"].values()}) == 3  # routine verifies at most 3 papers
    dropped = results["requests"]["N4-O1:r1"]  # the least severe threat waits
    assert dropped["status"] == "not_run" and "at most 3 prior paper" in dropped["reason"]
    assert results["tasks"][next(t for t, v in results["tasks"].items()
                                 if v["prior_key"] == "arxiv:2401.00002")]["verifier_model"] == "opus-5.5"
    gate = read_json(capped.role_dir("judge") / "evidence_gate.json")
    labels = {tuple(v["objection_ids"]): v["label"] for v in gate["verdicts"] if not v["passes"]}
    assert any("verification cap" in (lab or "") for ids, lab in labels.items() if "N4-O1" in ids)


def test_priority_ranks_central_then_severity_then_doubt():
    base = {"central": False, "severity": "major", "confidence": "high", "basis": "full_text"}
    reqs = [{**base, "request_id": "A"}, {**base, "request_id": "B", "severity": "fatal"},
            {**base, "request_id": "C", "confidence": "low"}, {**base, "request_id": "D", "central": True},
            {**base, "request_id": "E", "basis": "abstract_only"}]
    assert [r["request_id"] for r in sorted(reqs, key=priority)] == ["D", "B", "C", "E", "A"]
    claims = ["We are the first to use a time-varying guidance schedule in diffusion models"]
    assert is_central("1. We are the first to use a time-varying guidance schedule in diffusion models.", "S02 Idea",
                      claims, set())
    assert not is_central("We sample 10,000 images with a pretrained model.", "S03 Planned", claims, set())
    assert is_central("anything at all", "S01 Abstract", [], {"S01"})


# ---------------------------------------------------------------- resume, stale outputs, plan limits


def test_resume_reruns_stale_agents_and_everything_after_them(runs_dir, sample_paper):
    store = _create(sample_paper)
    _run(store)
    before = {a: v["attempts"] for a, v in store.load_state()["agents"].items()}
    _run(store, ("novelty",), rerun=["N2"])  # an upstream output changes
    state = store.load_state()
    assert {"J1", "J2"} <= set(stale_agents(store, state))
    stages = service.resume_stages(state, "followup", stale_agents(store, state))
    result, pipeline = _run(store, stages, refresh_stale=True)
    assert result["outcome"] == "complete", result
    after = store.load_state()["agents"]
    reran = {a for a in ("N1", "R1", "R2", "F1", "J1", "J2", "S1", "C1") if pipeline.provider.attempts.get(a)}
    assert reran == {"J1", "J2", "S1", "C1"}  # unchanged outputs are reused
    assert not stale_agents(store, store.load_state())
    assert any(e["event"] == "stale_rerun" for e in read_jsonl(store.events_path))
    assert after["N1"]["attempts"] == before["N1"]


def test_plan_limit_waits_are_capped_per_run(runs_dir, sample_paper):
    store = _create(sample_paper, fail={"F1": ["plan_limit", "plan_limit"]})
    result, _ = _run(store, ("intake", "novelty", "rigor", "fit"))
    state = store.load_state()
    assert result["outcome"] == "incomplete" and "automatic wait(s)" in result["message"], result
    assert state["agents"]["F1"]["status"] == "interrupted" and state["plan_limit"]["waits"] == 1
    _set_mock(store, fail={})
    result, _ = _run(store, ("intake", "novelty", "rigor", "fit"))
    assert result["outcome"] == "complete" and store.load_state()["agents"]["F1"]["status"] == "complete"
    pipeline = Pipeline(_create(sample_paper))
    assert pipeline._may_wait("2026-10-02T00:00:00Z") and pipeline._may_wait("2026-10-02T00:00:00Z")
    assert not pipeline._may_wait("2026-10-02T05:00:00Z")  # one window per run; the same window counts once


# ---------------------------------------------------------------- runs saved before presets


def test_runs_saved_before_presets_still_resume(runs_dir, sample_paper):
    store = _create(sample_paper, "legacy_thorough")
    raw = yaml.safe_load(store.config_path.read_text())
    for key in ("preset", "presets"):
        raw.pop(key, None)
    for section in raw.values():
        if isinstance(section, dict):
            section.pop("per_agent", None)
            for key in ("routing", "counterpart", "max_papers_per_run", "max_waits_per_run"):
                section.pop(key, None)
    raw["followup"]["adjudicator"].pop("per_agent", None)
    store.config_path.write_text(yaml.safe_dump(raw))
    (store.dir / "assignment.json").unlink()
    cfg = store.load_config()
    assert cfg.preset is None and cfg.verifier.routing == "fixed" and cfg.plan_limit.max_waits_per_run is None
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    assert "Follow-up round 1" in service.status_text(store.run_id)


def test_gates_still_hold_under_routine(runs_dir, sample_paper):
    store = _create(sample_paper, served_model={"R2": "claude-haiku-4-5"})
    result, _ = _run(store, ("intake", "novelty", "rigor", "fit", "judge"))
    agents = store.load_state()["agents"]
    assert agents["R2"]["status"] == "quarantined" and result["outcome"] == "quarantined"
    assert agents["J1"]["status"] == "pending"  # judges wait for a quarantined refuter
    store = _create(sample_paper, decisive={"N2": {"category": "already_done", "severity": "fatal",
                                                   "basis": "abstract_only",
                                                   "prior": {"title": "Masked Diffusion Transformer",
                                                             "arxiv_id": "2303.14389"}}},
                    misplace={"S1": True})
    result, _ = _run(store)
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    assert "N2-O1" in gate["flagged_ids"]
    assert read_json(store.gate_path("S1", "synthesis"))["placement_fix"]["outcome"] == "accepted"
