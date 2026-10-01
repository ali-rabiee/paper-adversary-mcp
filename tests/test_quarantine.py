"""End-to-end completion gates with the mock provider: quarantine, automatic reruns, repairs, release, resume."""

import asyncio
import re

import pytest
import yaml

from conftest import mock_override
from paper_adversary import service
from paper_adversary.config import deep_merge
from paper_adversary.pipeline import Pipeline
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore
from paper_adversary.util import FileLock, read_json, read_jsonl
from paper_adversary.worker import WorkerBusy

ALL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]
PASSWD = {"name": "Read", "input": {"file_path": "/etc/passwd"}, "result": "root:x:0:0:root"}


def _create(sample_paper, extra=None, **mock):
    override = deep_merge(mock_override(**mock), extra or {})
    info = service.create_run(str(sample_paper), config_override=override)
    return RunStore.open(info["run_id"])


def _run(store, stages=ALL, **kw):
    pipeline = Pipeline(store)
    result = asyncio.run(pipeline.run(stages, **kw))
    return result, pipeline


def _agents(store):
    return store.load_state()["agents"]


def _set_mock(store, **mock):
    raw = yaml.safe_load(store.config_path.read_text())
    raw["provider"]["mock"].update(mock)
    store.config_path.write_text(yaml.safe_dump(raw))


def test_failed_audit_reruns_once_then_completes(runs_dir, sample_paper):
    store = _create(sample_paper, tool_calls={"R1": {"attempts": [[PASSWD], []]}})
    result, _ = _run(store)
    assert result["outcome"] == "complete", result
    r1 = _agents(store)["R1"]
    assert r1["status"] == "complete" and r1["auto_reruns"] == 1 and r1["attempts"] == 2
    assert any(e["event"] == "agent_auto_rerun" for e in read_jsonl(store.events_path))
    index = read_jsonl(store.dir / "archive" / "index.jsonl")
    assert any(e["agent_id"] == "R1" and "automatic rerun" in e["reason"] for e in index)


def test_repeated_audit_failure_quarantines_and_blocks_judges(runs_dir, sample_paper):
    store = _create(sample_paper, tool_calls={"R1": [PASSWD]})
    result, _ = _run(store)
    assert result["outcome"] == "quarantined" and "R1 (quarantined: isolation audit failed" in result["message"]
    agents = _agents(store)
    assert agents["R1"]["status"] == "quarantined" and agents["R1"]["gate"]["classes"] == ["isolation"]
    assert agents["J1"]["status"] == "pending"
    assert (store.dir / "rigor" / "R1.md").is_file()  # kept on disk
    # with allow_incomplete the judges run, but never see R1
    result, _ = _run(store, ["judge", "synthesis", "critic"], allow_incomplete=True)
    assert result["outcome"] == "complete", result
    for jid in ("J1", "J2", "J3"):
        _, body = read_report(store.report_path(jid, "judge"))
        assert "CANARY_R1" not in body
        prompt = (store.agent_log_dir(jid) / "attempt-1" / "user_prompt.md").read_text()
        assert "R1 (quarantined: isolation audit failed" in prompt and "CANARY_R1" not in prompt
    matrix = service.get_report(store.run_id, "matrix")
    assert "Not included (no usable output): R1 (quarantined" in matrix
    status = service.status_text(store.run_id)
    assert "R1: QUARANTINED (isolation)" in status


def test_unverifiable_audit_and_the_per_job_rerun_cap(runs_dir, sample_paper):
    store = _create(sample_paper, transcript={"N1": {"attempts": ["missing"]}, "N2": {"attempts": ["no_init"]},
                                              "N3": {"attempts": ["empty"]}})
    _run(store, ["novelty"])
    agents = _agents(store)
    statuses = sorted(agents[a]["status"] for a in ("N1", "N2", "N3"))
    assert statuses == ["complete", "complete", "quarantined"]  # only two automatic reruns per job
    held = next(a for a in ("N1", "N2", "N3") if agents[a]["status"] == "quarantined")
    assert agents[held]["gate"]["reasons"][0].startswith("isolation could not be verified")
    assert agents["N4"]["status"] == "complete"


def test_leaked_marker_is_caught(runs_dir, sample_paper):
    store = _create(sample_paper, leak_marker_of={"R2": "N1"})
    _run(store, ["novelty"])
    _run(store, ["rigor"])
    r2 = _agents(store)["R2"]
    assert r2["status"] == "quarantined" and "marker of N1" in r2["detail"]


def test_bad_json_is_rebuilt_from_headings_without_a_model_call(runs_dir, sample_paper):
    store = _create(sample_paper, json={"R1": "malformed", "F1": "schema_invalid", "J2": "trailing_comma"})
    result, pipeline = _run(store)
    assert result["outcome"] == "complete", result
    agents = _agents(store)
    assert agents["R1"]["structured"] == "derived:headings" and agents["R1"]["objections"] == 2
    assert agents["F1"]["structured"] == "derived:headings"
    assert agents["J2"]["structured"] == "ok:lenient"
    assert pipeline.provider.repair_calls == 0
    meta, _ = read_report(store.report_path("R1", "rigor"))
    assert meta["structured_block"].startswith("invalid JSON")  # as written
    side = read_json(store.sidecar_path("R1", "rigor", ".json"))
    assert side["structured"]["source"] == "derived:headings" and side["objections"][0]["id"] == "R1-O1"


def test_missing_novelty_block_is_repaired_and_validated(runs_dir, sample_paper):
    store = _create(sample_paper, omit_json=["N1"], json={"N2": "malformed"}, repair={"N3": "invent_reference"},
                    extra={"novelty": {"agents": 4}})
    _set_mock(store, json={"N2": "malformed", "N3": "omit"})
    result, pipeline = _run(store, ["novelty"])
    agents = _agents(store)
    assert pipeline.provider.repair_calls == 3
    assert agents["N1"]["structured"] == "repaired:transcribe"
    refs = read_json(store.sidecar_path("N1", "novelty", ".json"))["objections"][0]["references"]
    assert refs and refs[0]["title"] == "Attention Is All You Need"
    assert agents["N2"]["structured"] == "repaired:syntax"
    gate = read_json(store.gate_path("N3", "novelty"))
    assert gate["repair"]["outcome"] == "rejected" and "does not appear" in gate["repair"]["rejections"][0]
    assert agents["N3"]["structured"] == "derived:headings" and agents["N3"]["status"] == "complete"
    usage = [r for r in read_jsonl(store.usage_path) if r.get("stage") == "repair"]
    assert len(usage) == 3 and all(r["effort"] == "low" for r in usage)


def test_intake_without_usable_data_is_quarantined_and_synthesis_says_so(runs_dir, sample_paper):
    store = _create(sample_paper, omit_json=["INTAKE"], repair={"INTAKE": "fail:invalid_request"})
    result, _ = _run(store)
    agents = _agents(store)
    assert agents["INTAKE"]["status"] == "quarantined"
    assert agents["S1"]["status"] == "complete" and not (store.source_dir / "claims_ledger.md").exists()
    prompt = (store.agent_log_dir("S1") / "attempt-1" / "user_prompt.md").read_text()
    assert "claims ledger, because INTAKE (quarantined" in prompt


def test_truncated_memo_is_quarantined_and_blocks_the_critic(runs_dir, sample_paper):
    store = _create(sample_paper, truncate={"S1": True}, stop_reason={"S1": "max_tokens"})
    result, _ = _run(store)
    agents = _agents(store)
    assert agents["S1"]["status"] == "quarantined"
    assert any("output-token limit" in r for r in agents["S1"]["gate"]["reasons"])
    result, _ = _run(store, ["critic"], allow_incomplete=True)
    assert result["outcome"] == "blocked" and agents["C1"]["status"] == "pending"


def test_memo_section_rules(runs_dir, sample_paper):
    store = _create(sample_paper, drop_sections={"S1": [8]})
    _run(store)
    s1 = _agents(store)["S1"]
    assert s1["status"] == "complete" and any("missing sections 8" in w for w in s1["gate"]["warnings"])
    store = _create(sample_paper, drop_sections={"S1": [13]})  # synthesis_v3 ends with section 13
    _run(store)
    assert _agents(store)["S1"]["status"] == "quarantined"


def test_substitution_quarantines_and_coverage_gaps_get_a_supplement(runs_dir, sample_paper):
    store = _create(sample_paper, served_model={"F1": "claude-haiku-4-5"}, coverage={"J1": 0.9})
    _run(store, allow_incomplete=True)
    agents = _agents(store)
    assert agents["F1"]["status"] == "quarantined" and agents["F1"]["gate"]["classes"] == ["integrity"]
    assert agents["F1"].get("auto_reruns") in (None, 0)  # only isolation failures rerun automatically
    j1 = agents["J1"]
    assert j1["status"] == "complete" and any("coverage supplement" in w for w in j1["gate"]["warnings"])
    gate = read_json(store.gate_path("J1", "judge"))
    asked = gate["supplement"]["ids"]
    assert gate["supplement"]["outcome"] == "accepted" and asked == ["R3-O1", "R3-O2"]  # the last 10% it skipped
    sup_log = next((store.dir / "logs/agents/J1").glob("attempt-1/supplement-1/user_prompt.md")).read_text()
    assert re.findall(r'<report agent_id="([NRF]\d+)"', sup_log) == ["R3"]  # only the report that raised them
    judgments = read_json(store.sidecar_path("J1", "judge", ".json"))["data"]["judgments"]
    assert {i for j in judgments for i in j["objection_ids"]} >= set(asked)
    _, body = read_report(store.report_path("J1", "judge"))
    assert "Coverage supplement (requested by the orchestrator)" in body
    assert "R3-O2" in service.get_report(store.run_id, "matrix") and "not classified by J1" not in \
        service.get_report(store.run_id, "matrix")
    usage = read_jsonl(store.dir / "logs" / "usage.jsonl")
    assert any(u["stage"] == "supplement" and u["agent_id"] == "J1" for u in usage)


def test_a_supplement_that_leaves_gaps_quarantines(runs_dir, sample_paper):
    store = _create(sample_paper, coverage={"J1": 0.9}, supplement={"J1": "skip_one", "J3": "fail:invalid_request"})
    _run(store, ["intake", "novelty", "rigor", "fit", "judge"])
    agents = _agents(store)
    assert agents["J1"]["status"] == "quarantined" and "not classified (R3-O1)" in agents["J1"]["detail"]
    assert agents["J3"]["status"] == "quarantined"  # its supplement call was refused: the gaps stay
    assert read_json(store.gate_path("J3", "judge"))["supplement"]["outcome"] == "failed"


def test_two_verdicts_for_one_objection_are_resolved(runs_dir, sample_paper):
    store = _create(sample_paper, conflict={"J2": "N1-O1"})
    _run(store, ["intake", "novelty", "rigor", "fit", "judge"])
    gate = read_json(store.gate_path("J2", "judge"))
    assert gate["supplement"]["conflicts"] == {"N1-O1": ["FATAL", "MAJOR_FIXABLE"]}
    judgments = read_json(store.sidecar_path("J2", "judge", ".json"))["data"]["judgments"]
    verdicts = [j["severity"] for j in judgments if "N1-O1" in j["objection_ids"]]
    assert verdicts == ["MINOR"] and _agents(store)["J2"]["status"] == "complete"


def test_interrupted_gate_resumes_without_rerunning_the_agent(runs_dir, sample_paper):
    store = _create(sample_paper, omit_json=["N1"], repair={"N1": "fail:cancelled"})
    _run(store, ["novelty"])
    n1 = _agents(store)["N1"]
    assert n1["status"] == "gating" and n1["attempts"] == 1
    assert "novelty" in service.resume_stages(store.load_state(), "refuters")
    _set_mock(store, repair={})
    result, pipeline = _run(store, ["novelty"])
    n1 = _agents(store)["N1"]
    assert n1["status"] == "complete" and n1["attempts"] == 1 and n1["structured"] == "repaired:transcribe"
    assert pipeline.provider.attempts.get("N1") is None  # the agent itself never ran again


def test_rerun_clears_the_gate_and_archives_its_record(runs_dir, sample_paper):
    store = _create(sample_paper, served_model={"F2": "claude-haiku-4-5"})
    _run(store, ["fit"])
    assert _agents(store)["F2"]["status"] == "quarantined"
    _set_mock(store, served_model={})
    _run(store, ["fit"], rerun=["F2"])
    f2 = _agents(store)["F2"]
    assert f2["status"] == "complete" and f2["gate"]["verdict"] in ("pass", "warn")
    archived = list((store.dir / "archive").rglob("F2.gate.json"))
    assert archived and read_json(archived[0])["verdict"] == "quarantine"


def test_release_rules(runs_dir, sample_paper):
    store = _create(sample_paper, tool_calls={"R1": [PASSWD]}, served_model={"F1": "claude-haiku-4-5"})
    _run(store, allow_incomplete=True)
    agents = _agents(store)
    assert agents["R1"]["status"] == agents["F1"]["status"] == "quarantined"
    with pytest.raises(PermissionError, match="paper-adversary release"):
        service.release_quarantine(store.run_id, "R1", "the path was a test fixture", "mcp")
    with pytest.raises(ValueError, match="reason"):
        service.release_quarantine(store.run_id, "F1", "ok", "mcp")
    lock = FileLock(store.lock_path)
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(WorkerBusy):
            service.release_quarantine(store.run_id, "F1", "haiku answered a smoke test on purpose", "mcp")
    finally:
        lock.release()
    text = service.release_quarantine(store.run_id, "F1", "haiku answered a smoke test on purpose", "mcp")
    assert "Released F1" in text and "stale" in text  # the judges ran without F1
    stale = service.stale_agents(store, store.load_state())
    assert all(any("F1 became available" in r for r in stale[j]) for j in ("J1", "J2", "J3"))
    assert service.release_quarantine(store.run_id, "R1", "the read path was inside a test fixture", "cli")
    agents = _agents(store)
    assert agents["R1"]["status"] == "complete" and agents["R1"]["release"]["via"] == "cli"
    assert "released by cli" in service.status_text(store.run_id)
    assert any(e["event"] == "quarantine_released" for e in read_jsonl(store.events_path))
    gates_report = service.get_report(store.run_id, "gates", "R1")
    assert "isolation_audit" in gates_report and "released" in gates_report


def test_gates_report_and_banner(runs_dir, sample_paper):
    store = _create(sample_paper, served_model={"F3": "claude-haiku-4-5"})
    _run(store, ["fit"])
    report = service.get_report(store.run_id, "fit", "F3")
    assert report.startswith("[F3 is QUARANTINED") and "No later stage reads this report" in report
    assert re.search(r"BLOCK\s+model \[integrity\]", service.get_report(store.run_id, "gates", "F3"))


def test_crash_windows_inside_the_gate(runs_dir, sample_paper):
    store = _create(sample_paper)
    _run(store, ["fit"])
    attempts = {a: v["attempts"] for a, v in _agents(store).items()}
    # 1. decided but not applied: the recorded decision is applied, the agent is not run again
    store.update_state(lambda st: st["agents"]["F1"].update(status="gating"))
    # 2. an automatic rerun archived the output, then the process died: the agent runs afresh
    store.update_state(lambda st: st["agents"]["F2"].update(status="gating"))
    store.archive_agent_outputs("F2", "fit", "test: simulated crash after archiving")
    result, pipeline = _run(store, ["fit"])
    agents = _agents(store)
    assert result["outcome"] == "complete", result
    assert agents["F1"]["status"] == "complete" and pipeline.provider.attempts.get("F1") is None
    assert agents["F2"]["status"] == "complete" and pipeline.provider.attempts.get("F2") == 1
    assert agents["F1"]["attempts"] == attempts["F1"]


def test_an_accepted_supplement_is_reused_after_a_crash(runs_dir, sample_paper):
    store = _create(sample_paper, coverage={"J1": 0.9})
    _, pipeline = _run(store, ["intake", "novelty", "rigor", "fit", "judge"])
    assert pipeline.provider.attempts.get("J1-supplement") == 1
    # a crash right after the supplement was accepted: the report lacks it and the gate is undecided
    archived = sorted((store.dir / "archive").rglob("J1.md"))[0]
    store.report_path("J1", "judge").write_text(archived.read_text())
    gate = read_json(store.gate_path("J1", "judge"))
    gate.update(verdict="pending", decision=None)
    store.gate_path("J1", "judge").write_text(__import__("json").dumps(gate))
    store.update_state(lambda st: st["agents"]["J1"].update(status="gating"))
    result, pipeline = _run(store, ["judge"])
    assert result["outcome"] == "complete", result
    assert pipeline.provider.attempts.get("J1-supplement") is None  # not paid for twice
    _, body = read_report(store.report_path("J1", "judge"))
    assert body.count("Coverage supplement (requested by the orchestrator)") == 1
    assert _agents(store)["J1"]["status"] == "complete"
