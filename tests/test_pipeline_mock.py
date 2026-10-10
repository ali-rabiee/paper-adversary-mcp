"""End-to-end pipeline runs with the offline mock provider."""

import asyncio
import datetime as dt
import json
import re

import pytest
import yaml

from conftest import mock_override
from paper_adversary import service
from paper_adversary.config import deep_merge
from paper_adversary.context import ContextBuilder
from paper_adversary.isolation import IsolationGuard, IsolationViolation, position
from paper_adversary.pipeline import Pipeline, spec_from_state
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore
from paper_adversary.util import atomic_write_json, read_json, read_jsonl, utcnow

ALL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]
OPUS, FABLE = "claude-opus-5-5", "claude-fable-5-1"


def _create(sample_paper, **mock_options):
    info = service.create_run(str(sample_paper), venue="ICML 2027", field="robot learning",
                              config_override=mock_override(**mock_options))
    return RunStore.open(info["run_id"]), info


def _run(store, stages=ALL, **kw):
    return asyncio.run(Pipeline(store).run(stages, **kw))


def _seen(store, aid, role):
    _, body = read_report(store.report_path(aid, role))
    m = re.search(r"Inputs seen: (.*)", body)
    return set() if not m or m.group(1).strip() == "none" else {x.strip() for x in m.group(1).split(",")}


def test_full_run_completes_with_metadata(runs_dir, sample_paper):
    store, info = _create(sample_paper)
    result = _run(store)
    assert result["outcome"] == "complete", result
    state = store.load_state()
    assert all(a["status"] == "complete" for a in state["agents"].values())
    # storage layout from the spec
    for rel in ("metadata.json", "config.yaml", "source/extracted_text.md", "source/paper.md", "novelty/N4.md",
                "rigor/R3.md", "fit/F3.md", "judges/J3.md", "synthesis/memo.md", "critic/completeness.md",
                "logs/events.jsonl", "logs/api_usage.json", "judges/judgment_matrix.md", "source/claims_ledger.md"):
        assert (store.dir / rel).is_file(), rel
    meta, body = read_report(store.dir / "rigor/R2.md")
    for key in ("model", "effort", "timestamp", "prompt_version", "run_id", "role", "prompt_sha256"):
        assert meta.get(key), key
    assert meta["role"] == "rigor" and meta["prompt_version"] == "rigor_v1" and meta["effort"] == "max"
    assert meta["isolation_audit"] == "pass"
    assert meta["lens"] == "Empirical methodology"


def test_isolation_is_visible_in_outputs(runs_dir, sample_paper):
    store, _ = _create(sample_paper)
    _run(store)
    refuters = [f"{p}{i}" for p, n in (("N", 4), ("R", 3), ("F", 3)) for i in range(1, n + 1)]
    role = {"N": "novelty", "R": "rigor", "F": "fit"}
    for aid in refuters:
        assert _seen(store, aid, role[aid[0]]) == set(), f"{aid} saw other agents' outputs"
    assert _seen(store, "INTAKE", "intake") == set()
    refuter_canaries = {f"CANARY_{a}" for a in refuters}
    for j in ("J1", "J2", "J3"):
        seen = _seen(store, j, "judge")
        assert seen == refuter_canaries, f"{j} saw {seen ^ refuter_canaries}"
    synth = _seen(store, "S1", "synthesis")
    assert synth == refuter_canaries | {"CANARY_J1", "CANARY_J2", "CANARY_J3"}
    critic = _seen(store, "C1", "critic")
    assert critic == synth | {"CANARY_S1"}
    # the claims ledger (derived from the intake pass) reaches synthesis and critic only
    def prompt(aid):
        return (store.dir / "logs" / "agents" / aid / "attempt-1" / "user_prompt.md").read_text()
    for aid in ("S1", "C1"):
        assert "Mock claim" in prompt(aid)
    for aid in ("N1", "R1", "F1", "J1"):
        assert "Mock claim" not in prompt(aid) and "claims_ledger" not in prompt(aid)


def test_judgment_matrix_flags(runs_dir, sample_paper):
    store, _ = _create(sample_paper)
    _run(store)
    matrix = service.get_report(store.run_id, "matrix")
    assert "contested" in matrix  # J1 MAJOR vs J2 FATAL/NOT CONVINCING on O2 objections
    assert "not classified by" not in matrix  # J3 skipped the second objections; a supplement ruled on them
    assert "N1-O1" in matrix and "F3-O2" in matrix


def test_status_and_reports(runs_dir, sample_paper):
    store, _ = _create(sample_paper)
    _run(store)
    status = service.status_text(store.run_id)
    assert "Novelty:" in status and "N4" in status and "Synthesis: complete" in status
    assert "Completeness critic: complete" in status
    assert "Failed calls and retries: none" in status
    assert "Isolation audit: all" in status
    one = service.get_report(store.run_id, "novelty", "N2")
    assert "CANARY_N2" in one
    summary = service.get_report(store.run_id, "rigor")
    assert "R1" in summary and "R3" in summary and "CANARY" not in summary
    assert "S00" in service.get_report(store.run_id, "sections")
    cost = service.cost_text(store.run_id)
    assert "returned no usage" in cost  # the mock reports no usage and nothing is invented


def test_retries_then_success(runs_dir, sample_paper):
    store, _ = _create(sample_paper, fail={"N2": ["overloaded", "network"]})
    result = _run(store, ["novelty"])
    assert result["outcome"] == "complete"
    a = store.load_state()["agents"]["N2"]
    assert a["status"] == "complete" and a["attempts"] == 3
    assert [f["kind"] for f in a["failures"]] == ["overloaded", "network"]
    assert "N2: 2 failed call(s)" in service.status_text(store.run_id)


def test_failure_blocks_judges_then_resume(runs_dir, sample_paper):
    store, _ = _create(sample_paper, fail={"F1": ["model_unavailable"]})
    result = _run(store)
    assert result["outcome"] == "incomplete"
    state = store.load_state()
    assert state["agents"]["F1"]["status"] == "failed"
    assert state["agents"]["J1"]["status"] == "pending"
    before = {aid: a["attempts"] for aid, a in state["agents"].items()}
    # fix the cause (drop the injected failure) and resume
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"]["fail"] = {}
    store.config_path.write_text(yaml.safe_dump(raw))
    assert _run(store)["outcome"] == "complete"
    after = store.load_state()["agents"]
    assert after["F1"]["attempts"] == before["F1"] + 1
    for aid in ("N1", "R2", "F2", "INTAKE"):
        assert after[aid]["attempts"] == before[aid], f"{aid} was re-run"


def test_rerun_archives_and_marks_stale(runs_dir, sample_paper):
    store, _ = _create(sample_paper)
    _run(store)
    old = (store.dir / "novelty/N1.md").read_text()
    _run(store, ["novelty"], rerun=["N1"])
    assert list((store.dir / "archive").rglob("N1.md")), "old report not archived"
    assert (store.dir / "novelty/N1.md").read_text() != old
    status = service.status_text(store.run_id)
    assert "Stale outputs" in status and "J1" in status


def test_plan_limit_is_waited_out(runs_dir, sample_paper):
    store, _ = _create(sample_paper, fail={"R1": ["plan_limit"]})
    result = _run(store, ["rigor"])
    assert result["outcome"] == "complete"
    events = [e["event"] for e in __import__("paper_adversary.util", fromlist=["read_jsonl"]).read_jsonl(store.events_path)]
    assert "plan_limit_wait" in events
    assert store.load_state()["agents"]["R1"]["status"] == "complete"


def test_auth_failure_stops_the_run(runs_dir, sample_paper):
    store, _ = _create(sample_paper, fail={"N1": ["auth"]})
    result = _run(store)
    assert result["outcome"] == "failed" and "auth" in result["message"]
    agents = store.load_state()["agents"]
    assert agents["N1"]["status"] == "failed"
    assert agents["J1"]["status"] == "pending"


def test_judges_require_refuters(runs_dir, sample_paper):
    store, _ = _create(sample_paper)
    result = _run(store, ["judge"])
    assert result["outcome"] == "blocked" and "refuters not complete" in result["message"]


def test_missing_structured_block_is_reported(runs_dir, sample_paper):
    store, _ = _create(sample_paper, omit_json=["J2"])
    _run(store)
    meta, _ = read_report(store.dir / "judges/J2.md")
    assert meta["structured_block"] != "ok"  # as written
    j2 = store.load_state()["agents"]["J2"]
    assert j2["status"] == "complete" and j2["structured"] == "derived:headings"  # recovered from its headings
    assert "structured data: derived:headings" in service.get_report(store.run_id, "gates", "J2")
    matrix = service.get_report(store.run_id, "matrix")
    assert "J2" in matrix.splitlines()[4]  # J2's verdicts are tabulated


# ---------------------------------------------------------------- plan limits (K1 audit F27)


def _future(days: float = 7) -> str:
    return (utcnow() + dt.timedelta(days=days)).replace(microsecond=0).isoformat()


def _fable_limit(reset: str | None) -> dict:
    """Fable's own weekly window is full (rateLimitType seven_day_overage_included), as on 2026-09-29."""
    return {"type": "seven_day_overage_included", "scope": "model", "reset_at": reset}


def _set_mock(store, **mock):
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"].update(mock)
    store.config_path.write_text(yaml.safe_dump(raw))


def _events(store, name):
    return [e for e in read_jsonl(store.events_path) if e["event"] == name]


def test_three_fable_judges_at_the_fable_limit_are_interrupted_without_using_the_wait(runs_dir, sample_paper):
    """K1 audit (F27): the Fable weekly window resets days away, beyond max_wait_hours. The judges are interrupted
    (not failed) with the model, limit type and reset; no wait slot is used; one plan_limit_skip; resumable."""
    reset = _future()
    store, _ = _create(sample_paper, fail={j: ["plan_limit"] for j in ("J1", "J2", "J3")},
                       limit={FABLE: _fable_limit(reset)})
    result = _run(store)
    state = store.load_state()
    assert result["outcome"] == "incomplete", result
    assert "Fable 5.1's model-specific plan limit" in result["message"] and reset in result["message"]
    for j in ("J1", "J2", "J3"):
        a = state["agents"][j]
        assert a["status"] == "interrupted", (j, a)
        assert FABLE in a["detail"] and "seven_day_overage_included" in a["detail"] and reset in a["detail"]
    assert (state.get("plan_limit") or {}).get("waits", 0) == 0  # checked before a wait slot is reserved
    skips = _events(store, "plan_limit_skip")
    assert len(skips) == 1 and (skips[0]["model"], skips[0]["scope"], skips[0]["reset_at"]) == (FABLE, "model", reset)
    assert not _events(store, "plan_limit_wait") and not _events(store, "plan_limit_stop")
    assert state["agents"]["S1"]["status"] == "pending"
    status = service.status_text(store.run_id)
    assert "Fable 5.1 (model-specific) weekly window 100% used" in status
    assert "Next: Stopped at a plan limit" in status and "resume_run continues after the reset" in status
    _set_mock(store, fail={})  # after the reset
    assert _run(store)["outcome"] == "complete"
    assert all(a["status"] == "complete" for a in store.load_state()["agents"].values())


def test_a_fable_limit_leaves_agents_on_other_models_running(runs_dir, sample_paper):
    """Only that model's agents stop: the Opus refuters finish, and the judges, which need every refuter, wait."""
    info = service.create_run(str(sample_paper), config_override=mock_override(
        "routine", fail={"N2": ["plan_limit"], "R1": ["plan_limit"]}, limit={FABLE: _fable_limit(_future())}))
    store = RunStore.open(info["run_id"])
    result = _run(store, ["intake", "novelty", "rigor", "fit", "judge"])
    agents = store.load_state()["agents"]
    assert {a: agents[a]["status"] for a in ("INTAKE", "N1", "R2", "F1")} == dict.fromkeys(
        ("INTAKE", "N1", "R2", "F1"), "complete")  # Opus
    assert agents["N2"]["status"] == agents["R1"]["status"] == "interrupted"  # Fable
    assert agents["J1"]["status"] == agents["J2"]["status"] == "pending"
    assert result["outcome"] == "incomplete" and "stopped before verifier: N2 (interrupted)" in result["message"]
    assert result["message"].startswith("stopped at Fable 5.1's model-specific plan limit (claude-fable-5-1; "
                                        "seven_day_overage_included), which resets 20")


def test_an_account_limit_beyond_max_wait_hours_stops_the_job_resumably(runs_dir, sample_paper):
    reset = _future(2)
    info = service.create_run(str(sample_paper), config_override=mock_override(
        "routine", fail={"F1": ["plan_limit"]}, limit={"F1": {"type": "seven_day", "scope": "account",
                                                              "reset_at": reset}}))
    store = RunStore.open(info["run_id"])
    result = _run(store, ["intake", "novelty", "rigor", "fit"])
    state = store.load_state()
    assert result["outcome"] == "incomplete" and "the plan's usage limit" in result["message"], result
    f1 = state["agents"]["F1"]
    assert f1["status"] == "interrupted" and "seven_day" in f1["detail"] and reset in f1["detail"]
    assert not [a for a, v in state["agents"].items() if v["status"] == "failed"]
    assert (state.get("plan_limit") or {}).get("waits", 0) == 0
    assert [e["scope"] for e in _events(store, "plan_limit_skip")] == ["account"]


def test_a_plan_limit_in_a_gate_call_names_the_model_limit_and_reset(runs_dir, sample_paper):
    """Review (G4): a coverage supplement (or repair, or placement fix) stopped at a plan limit left the agent's
    detail at 'stopped at a plan limit', without the model, the limit type or the reset that F27 asks for."""
    reset = _future()
    store, _ = _create(sample_paper, coverage={"J1": 0.5}, supplement={"J1": "fail:plan_limit"},
                       limit={"J1-supplement": _fable_limit(reset)})
    result = _run(store, ["intake", "novelty", "rigor", "fit", "judge"])
    j1 = store.load_state()["agents"]["J1"]
    assert result["outcome"] == "incomplete" and j1["status"] == "gating", (result, j1)
    assert "plan limit: stopped at Fable 5.1's model-specific plan limit (claude-fable-5-1; " \
           f"seven_day_overage_included), which resets {reset}" in j1["detail"]
    _set_mock(store, supplement={})  # after the reset, the resume finishes the gate
    assert _run(store, ["judge"])["outcome"] == "complete"
    assert store.load_state()["agents"]["J1"]["status"] == "complete"


def test_parallel_hits_without_a_reset_time_use_one_wait(runs_dir, sample_paper, monkeypatch):
    """Unknown resets are keyed by model and limit type: three judges hitting one limit count once (before, each
    hit recorded its own 'unknown@<now>' window, so the second hit stopped the whole job).

    The window lasts as long as the wait. The shared mock config waits about 1 s, so under a loaded full suite a
    judge whose hit was recorded more than 1 s after the first one opened a second window and the test failed
    now and then. Here the wait is a minute long (as a real wait is 30 minutes) and sleeping is a no-op."""
    async def _no_sleep(self, seconds):
        return None

    monkeypatch.setattr(Pipeline, "_sleep", _no_sleep)
    override = mock_override(fail={j: ["plan_limit"] for j in ("J1", "J2", "J3")}, together=["J1", "J2", "J3"],
                             limit={FABLE: _fable_limit(None)})  # at one moment, whatever the load
    override["plan_limit"] = {**override["plan_limit"], "default_wait_minutes": 1}
    info = service.create_run(str(sample_paper), venue="ICML 2027", field="robot learning", config_override=override)
    store = RunStore.open(info["run_id"])
    result = _run(store)
    state = store.load_state()
    assert result["outcome"] == "complete", result
    assert state["plan_limit"]["waits"] == 1 and len(state["plan_limit"]["windows"]) == 1
    assert state["plan_limit"]["windows"][0].startswith(f"unknown:{FABLE}:seven_day_overage_included@")
    assert len(_events(store, "plan_limit_wait")) == 3


def test_the_window_guard_stops_a_run_before_its_first_fable_agent(runs_dir, sample_paper):
    """plan_limit.model_window_reserve: a run that would need 15% of Fable's weekly window, which an earlier run
    in the same runs directory saw at 90%, stops before spending anything; a run already using Fable goes on."""
    resets = int((utcnow() + dt.timedelta(days=5)).timestamp())
    seen = {"status": "allowed", "rateLimitType": "five_hour", "unifiedWindows": {
        "five_hour": {"utilization": 0.2, "resetsAt": resets - 400000},
        "seven_day_overage_included": {"utilization": 0.9, "resetsAt": resets}}}
    trusted = {"plan_limit": {"model_window_reserve": 0.15}}
    first = RunStore.open(service.create_run(str(sample_paper), config_override=mock_override(
        "routine", rate_limit={FABLE: seen}), trusted_config=trusted)["run_id"])
    assert _run(first, ["intake", "novelty", "rigor", "fit"])["outcome"] == "complete"
    shared = read_json(runs_dir / ".cache" / "plan_windows.json")["models"][FABLE]["seven_day_overage_included"]
    assert (shared["utilization"], shared["resetsAt"]) == (0.9, resets)
    assert _run(first, ["judge"])["outcome"] == "complete"  # its Fable agents already started: no guard
    second = RunStore.open(service.create_run(str(sample_paper), config_override=mock_override("routine"),
                                              trusted_config=trusted)["run_id"])
    pipeline = Pipeline(second)
    result = asyncio.run(pipeline.run(ALL))
    assert result["outcome"] == "incomplete", result
    assert "Fable 5.1's model-specific weekly window" in result["message"] and "90%" in result["message"]
    assert "plan_limit.model_window_reserve" in result["message"]
    assert pipeline.provider.attempts == {}  # nothing was spent, on any model
    assert {a["status"] for a in second.load_state()["agents"].values()} == {"pending"}
    assert len(_events(second, "plan_limit_guard")) == 1
    shared_path = runs_dir / ".cache" / "plan_windows.json"
    data = read_json(shared_path)
    data["models"][FABLE]["seven_day_overage_included"]["resetsAt"] = int(utcnow().timestamp()) - 60  # it reset
    atomic_write_json(shared_path, data)
    assert _run(second, ["intake", "novelty", "rigor", "fit"])["outcome"] == "complete"


def test_the_window_guard_uses_what_this_preflight_just_probed(runs_dir, sample_paper):
    """Review (G4): the guard ran before the job's own probes, so the first run in a runs directory without
    plan_windows.json started every Fable agent although its probe had just reported the window 95% used."""
    resets = int((utcnow() + dt.timedelta(days=5)).timestamp())
    seen = {"status": "allowed", "unifiedWindows": {"seven_day_overage_included": {"utilization": 0.95,
                                                                                    "resetsAt": resets}}}
    override = deep_merge(mock_override("routine", rate_limit={FABLE: seen}), {"provider": {"preflight": "probe"}})
    store = RunStore.open(service.create_run(str(sample_paper), config_override=override, trusted_config={
        "plan_limit": {"model_window_reserve": 0.15}})["run_id"])
    assert not (runs_dir / ".cache" / "plan_windows.json").exists()
    pipeline = Pipeline(store)
    result = asyncio.run(pipeline.run(["intake", "novelty", "rigor", "fit", "judge"]))
    assert result["outcome"] == "incomplete" and "95% used" in result["message"], result
    assert pipeline.provider.attempts == {}  # stopped before any agent, on any model
    assert len(_events(store, "plan_limit_guard")) == 1


def test_validate_probes_label_and_record_the_fable_window(runs_dir):
    """`validate --probe` before a batch: the Fable window it sees is labelled with its model and kept for the
    next run's window guard."""
    seen = {"status": "allowed", "unifiedWindows": {"seven_day_overage_included": {"utilization": 0.9,
                                                                                    "resetsAt": 4102444800}}}
    out = asyncio.run(service.validate_text({"provider": {"type": "mock", "mock": {"rate_limit": {FABLE: seen}}}},
                                            probe_models=True))
    assert "Fable 5.1 (model-specific) weekly window 90% used" in out
    shared = read_json(runs_dir / ".cache" / "plan_windows.json")["models"]
    assert shared[FABLE]["seven_day_overage_included"]["utilization"] == 0.9 and OPUS not in shared


# ---------------------------------------------------------------- hidden turns (K1 audit F28)


def test_hidden_turns_are_warned_blocked_and_counted(runs_dir, sample_paper):
    """A report from the final turn alone: intact (S1, warned) or without its leading section (C1, quarantined).
    The hidden turns' output tokens are reported in the usage summary and get_run_cost."""
    cut = [{"stop_reason": "max_tokens", "output_tokens": 128000}]
    store, _ = _create(sample_paper, synthetic_usage=True, turns={"S1": cut, "C1": cut}, drop_sections={"C1": [1]})
    result = _run(store)
    agents = store.load_state()["agents"]
    assert agents["S1"]["status"] == "complete" and result["outcome"] == "quarantined", result
    assert any("1 hidden turn(s) before the final message used 128,000 output tokens" in w
               for w in agents["S1"]["warnings"])
    check = next(c for c in read_json(store.gate_path("S1", "synthesis"))["checks"] if c["check"] == "hidden_turns")
    assert check["result"] == "warn" and "intact" in check["message"]
    assert agents["C1"]["status"] == "quarantined"
    assert any("leading section '1. Newly discovered issues'" in r for r in agents["C1"]["gate"]["reasons"])
    total = read_json(store.usage_summary_path)["total"]
    assert (total["hidden_turns"], total["hidden_output_tokens"]) == (2, 256000)
    cost = service.cost_text(store.run_id)
    assert "Hidden turns: 2 turn(s) before agents' final messages used 256.0k output tokens" in cost
    assert "S1 attempt 1: 1 turn(s), 128.0k output tokens" in cost


def test_hidden_turns_block_only_a_report_without_its_structured_block(runs_dir, sample_paper):
    """Review (G4): with hidden turns, a structured block that was there but invalid (repaired, or rebuilt) was
    treated like a missing one and quarantined the agent. A block at the end of the final turn shows that turn
    holds the report's end; only a report with no block at all may have lost its end to a discarded turn."""
    turns = [{"stop_reason": "end_turn", "output_tokens": 9000}]
    store, _ = _create(sample_paper, turns={"INTAKE": turns}, json={"INTAKE": "malformed"})
    _run(store, ["intake"])
    intake = store.load_state()["agents"]["INTAKE"]
    check = next(c for c in read_json(store.gate_path("INTAKE", "intake"))["checks"] if c["check"] == "hidden_turns")
    assert intake["status"] == "complete" and check["result"] == "warn" and "intact" in check["message"]
    assert intake["gate"]["structured"] == "repaired:syntax"
    store, _ = _create(sample_paper, turns={"INTAKE": turns}, json={"INTAKE": "omit"})
    _run(store, ["intake"])
    intake = store.load_state()["agents"]["INTAKE"]
    assert intake["status"] == "quarantined" and any("lacks its structured block" in r
                                                     for r in intake["gate"]["reasons"])


def test_agents_with_the_pdf_are_told_so_in_their_tool_sentence(runs_dir, sample_paper, tmp_path):
    """Review (G2): {{tool_access}} is rendered twice in _assemble, the second time after read_pdf joins the tools;
    only the tool-less branch was tested end to end. A judge_v3 judge reading a PDF submission, and its coverage
    supplement, are told about ./paper.pdf and the Read tool, never that they have no tools."""
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "Schedules for Diffusion Models", fontsize=18, fontname="hebo")
    for i in range(12):
        page.insert_text((72, 100 + 14 * i), "The finite horizon analysis covers every schedule this paper "
                                             f"considers in detail, line {i}.", fontsize=10)
    doc.save(pdf)
    override = deep_merge(mock_override(coverage={"J1": 0.5}), {"judge": {"prompt": "judge_v3", "paper_format": "pdf"}})
    store = RunStore.open(service.create_run(str(pdf), config_override=override)["run_id"])
    assert _run(store, ["intake", "novelty", "rigor", "fit", "judge"])["outcome"] == "complete"
    j1 = store.agent_log_dir("J1") / "attempt-1"
    for path in (j1 / "system_prompt.md", j1 / "supplement-1" / "system_prompt.md"):
        system = path.read_text()
        assert "./paper.pdf" in system and "Read tool" in system and "You have no tools" not in system, path
        assert "{{" not in system
    assert any("coverage supplement" in w for w in store.load_state()["agents"]["J1"]["gate"]["warnings"])


# ---------------------------------------------------------------- search health (K1 audit F24, F29)


def _search_rows(attempt: int, unavailable: int, answered: int) -> list[dict]:
    down = {"openalex": "ok", "semantic_scholar": "unavailable (semantic_scholar kept answering HTTP 429)",
            "arxiv": "cached", "crossref": "ok"}
    up = {**down, "semantic_scholar": "ok"}
    return [{"at": f"2026-10-02T03:{14 + i // 10:02d}:{i % 60:02d}+00:00", "agent_id": "N", "attempt": attempt,
             "tool": "search_literature", "args": {"query": f"q{i}"}, "results": [],
             "status": down if i < unavailable else up} for i in range(unavailable + answered)]


def test_search_health_is_recorded_and_told_to_judges_memo_and_critic(runs_dir, sample_paper, monkeypatch):
    """K1 pilot: Semantic Scholar (no key) was unavailable for 37 of 43 literature calls, and only the agents' own
    prose said so. The run now records it, warns before it, and tells the judges, the memo and the critic."""
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setenv("OPENALEX_API_KEY", "an-openalex-key")  # this test is about Semantic Scholar's warning
    store, _ = _create(sample_paper)
    assert _run(store, ["intake", "novelty", "rigor", "fit"])["outcome"] == "complete"
    warnings = store.load_state()["preflight"]["warnings"]
    assert len(warnings) == 1 and "S2_API_KEY is not set" in warnings[0]["message"]
    assert [e["setup"] for e in _events(store, "preflight_warning")] == ["search"]
    for aid, rows in (("N1", _search_rows(1, 20, 3)), ("N2", _search_rows(1, 17, 3))):
        path = store.sidecar_path(aid, "novelty", ".search_log.jsonl")
        other = _search_rows(2, 5, 0)  # an attempt that did not write the report: not counted
        path.write_text("".join(json.dumps(r) + "\n" for r in rows + other))
    assert _run(store, ["judge", "synthesis", "critic"])["outcome"] == "complete"
    health = store.load_state()["search_health"]
    s2 = health["providers"]["semantic_scholar"]
    assert (s2["calls"], s2["answered"], s2["unavailable"], health["degraded"]) == (43, 6, 37, ["semantic_scholar"])
    assert health["providers"]["arxiv"]["unavailable"] == 0 and len(_events(store, "search_degraded")) == 1
    note = "semantic_scholar was unavailable for 37 of 43 calls (86%; N1 20 of 23, N2 17 of 20)"
    assert note in health["note"]
    for aid, role in (("J1", "judge"), ("J3", "judge"), ("S1", "synthesis"), ("C1", "critic")):
        prompt = (store.agent_log_dir(aid) / "attempt-1" / "user_prompt.md").read_text()
        assert "<search_coverage source=" in prompt and note in prompt, aid
    status = service.status_text(store.run_id)
    assert "Literature search (refuters' logs): semantic_scholar unavailable for 37 of 43 calls (86%" in status
    assert "DEGRADED" in status and re.search(r"Preflight warning \(\d{4}-\d\d-\d\d \d\d:\d\d\): semantic_scholar "
                                              "is a search provider but S2_API_KEY is not set", status)
    assert _run(store, ["judge"], rerun=["J2"])["outcome"] == "complete"
    assert len(_events(store, "search_degraded")) == 1  # unchanged: not emitted again
    # A judge that quotes the line must not make it look like a leak to another judge: the line is allowed text.
    guard = IsolationGuard(store)
    guard.register("J9", "judge", "<!-- aid:J9:0123456789ab -->", health["note"], set(), position("judge"))
    state = store.load_state()
    ctx = ContextBuilder(store, store.load_config(), Pipeline(store).registry).build(
        spec_from_state(state["agents"]["J2"]), state)
    prompt = ctx.system_prompt + "\n" + ctx.user_text
    guard.check_prompt("judge", "J2", prompt, ctx.allowed_text, position("judge"))
    with pytest.raises(IsolationViolation):  # without the allowance it would be flagged
        guard.check_prompt("judge", "J2", prompt, "", position("judge"))
    monkeypatch.setenv("S2_API_KEY", "a-key")
    keyed, _ = _create(sample_paper)
    _run(keyed, ["novelty"])
    assert not (keyed.load_state().get("preflight") or {}).get("warnings")
    # review (G4): with the key now set, the next preflight (here a judge's) drops the stale warning
    assert _run(store, ["judge"], rerun=["J2"])["outcome"] == "complete"
    assert not store.load_state()["preflight"]["warnings"]
    assert "Preflight warning" not in service.status_text(store.run_id)


def test_a_missing_openalex_key_is_warned_about_and_dropped_once_set(runs_dir, sample_paper, monkeypatch):
    """K2 run (2026-10-02): without a key OpenAlex's daily budget (about 100 searches) ran out after about two reviews
    and OpenAlex answered HTTP 429 to 26 of 27 calls. A run without OPENALEX_API_KEY now warns before its literature
    agents start, next to the Semantic Scholar warning, and the warning goes once the key is set."""
    monkeypatch.setenv("S2_API_KEY", "an-s2-key")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    store, _ = _create(sample_paper)
    assert _run(store, ["intake", "novelty", "rigor", "fit"])["outcome"] == "complete"
    warnings = store.load_state()["preflight"]["warnings"]
    assert len(warnings) == 1 and "OPENALEX_API_KEY is not set" in warnings[0]["message"]
    monkeypatch.setenv("OPENALEX_API_KEY", "an-openalex-key")
    assert _run(store, ["judge"])["outcome"] == "complete"
    assert not store.load_state()["preflight"]["warnings"]


# ---------------------------------------------------------------- preflight (K1 audit F30)


def test_preflight_merges_models_and_probes_across_calls(runs_dir, sample_paper):
    """The verifier and follow-up preflights cover fewer models; they used to replace the run's preflight record."""
    seen = {"status": "allowed", "unifiedWindows": {"seven_day_overage_included": {"utilization": 0.2,
                                                                                    "resetsAt": 4102444800}}}
    override = deep_merge(mock_override(context_window={OPUS: 1000000, FABLE: 900000}, rate_limit={FABLE: seen}),
                          {"provider": {"preflight": "probe"}})
    store = RunStore.open(service.create_run(str(sample_paper), config_override=override)["run_id"])
    pipeline = Pipeline(store)
    state = store.load_state()
    asyncio.run(pipeline.preflight([], [spec_from_state(state["agents"]["N1"])]))  # Opus, with search tools
    asyncio.run(pipeline.preflight([], [spec_from_state(state["agents"]["J1"])]))  # Fable, no tools
    pf = store.load_state()["preflight"]
    assert {m: p["context_window"] for m, p in pf["models"].items()} == {OPUS: 1000000, FABLE: 900000}
    assert sorted(k.split(":")[1] for k in pf["probes"]) == [FABLE, OPUS]
    windows = store.load_state()["plan_usage"]["model_windows"]  # probes report plan windows too
    assert windows[FABLE]["seven_day_overage_included"]["utilization"] == 0.2
    # review (G4): a fresh probe without a context window keeps the one an earlier probe reported, and a model
    # whose latest probe failed (transiently: a warning, the run goes on) leaves no stale ok entry behind
    (runs_dir / ".cache" / "model_probes.json").unlink()  # past the 24 h cache
    pipeline.provider.options.update(context_window={}, probe_fail={FABLE: "overloaded"})
    asyncio.run(pipeline.preflight([], [spec_from_state(state["agents"][a]) for a in ("N1", "J1")]))
    pf = store.load_state()["preflight"]
    assert list(pf["models"]) == [OPUS] and pf["models"][OPUS]["context_window"] == 1000000
    assert [e["kind"] for e in _events(store, "preflight_warning") if e.get("setup") != "search"] == ["overloaded"]
