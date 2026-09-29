"""The isolation policy and guard reject forbidden inputs regardless of how they got in."""

import asyncio

import pytest

from conftest import mock_override
from paper_adversary import service
from paper_adversary.config import AgentSpec
from paper_adversary.context import ContextBuilder
from paper_adversary.isolation import ROLE_VISIBILITY, ArtifactAccess, IsolationGuard, IsolationViolation
from paper_adversary.pipeline import Pipeline
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore


@pytest.fixture
def finished_run(runs_dir, sample_paper):
    info = service.create_run(str(sample_paper), config_override=mock_override())
    store = RunStore.open(info["run_id"])
    asyncio.run(Pipeline(store).run(["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]))
    return store


def test_policy_table_matches_the_spec():
    for role in ("novelty", "rigor", "fit"):
        assert ROLE_VISIBILITY[role] == {"paper"}
    assert {"novelty", "rigor", "fit", "rubric"} <= ROLE_VISIBILITY["judge"]
    assert "judge" not in ROLE_VISIBILITY["judge"]
    assert {"judge", "novelty"} <= ROLE_VISIBILITY["synthesis"] and "critic" not in ROLE_VISIBILITY["synthesis"]
    assert "synthesis" in ROLE_VISIBILITY["critic"]


def test_access_refuses_forbidden_kinds(finished_run):
    state = finished_run.load_state()
    refuter = ArtifactAccess(finished_run, "novelty", "N1")
    for kind in ("novelty", "rigor", "fit", "judge", "synthesis", "profile"):
        with pytest.raises(IsolationViolation):
            refuter.reports(kind, state)
    judge = ArtifactAccess(finished_run, "judge", "J1")
    assert len(judge.reports("rigor", state)) == 3
    with pytest.raises(IsolationViolation):
        judge.reports("judge", state)


def test_guard_catches_a_leaked_report(finished_run):
    guard = IsolationGuard(finished_run)
    _, leaked = read_report(finished_run.dir / "novelty" / "N2.md")
    # a refuter prompt that somehow contains another refuter's report
    with pytest.raises(IsolationViolation, match="N2"):
        guard.check_prompt("novelty", "N1", "paper text\n" + leaked)
    # without the marker line, the distinctive 8-gram fingerprints still catch it
    body = "\n".join(line for line in leaked.splitlines() if not line.startswith("<!--"))
    with pytest.raises(IsolationViolation):
        guard.check_prompt("rigor", "R1", body)
    # judges may see refuters but never another judge
    _, judge_report = read_report(finished_run.dir / "judges" / "J1.md")
    guard.check_prompt("judge", "J2", leaked)
    with pytest.raises(IsolationViolation, match="J1"):
        guard.check_prompt("judge", "J2", judge_report)


def test_built_contexts_contain_only_allowed_inputs(finished_run):
    state = finished_run.load_state()
    builder = ContextBuilder(finished_run, finished_run.load_config(), __import__(
        "paper_adversary.registry", fromlist=["ModelRegistry"]).ModelRegistry.load())
    spec = AgentSpec(agent_id="N9", role="novelty", index=9, model_alias="opus-5.5", model_id="claude-opus-5-5",
                     effort="high", prompt_name="novelty_v1", tools=["literature"])
    ctx = builder.build(spec, state)
    kinds = {m["kind"] for m in ctx.manifest}
    assert kinds == {"paper"}
    assert "CANARY_" not in ctx.user_text and "<refuter_reports>" not in ctx.user_text
    judge = AgentSpec(agent_id="J9", role="judge", index=9, model_alias="fable-5.1", model_id="claude-fable-5-1",
                      effort="max", prompt_name="judge_v1")
    jctx = builder.build(judge, state)
    assert {m["kind"] for m in jctx.manifest} == {"paper", "novelty", "rigor", "fit", "rubric"}
    assert "CANARY_J1" not in jctx.user_text and "CANARY_N1" in jctx.user_text


def test_audit_flags_reads_outside_the_sandbox(tmp_path, finished_run):
    guard = IsolationGuard(finished_run)
    transcript = tmp_path / "t.jsonl"
    transcript.write_text('{"kind": "tool_use", "name": "Read", "input": {"file_path": "/etc/passwd"}}\n'
                          '{"kind": "tool_use", "name": "Read", "input": {"file_path": "paper.pdf"}}\n')
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    result = guard.audit("rigor", "R9", transcript, sandbox, "report")
    assert result["status"] == "fail" and any("/etc/passwd" in f for f in result["findings"])
    ok = tmp_path / "ok.jsonl"
    ok.write_text('{"kind": "tool_use", "name": "Read", "input": {"file_path": "paper.pdf"}}\n')
    assert guard.audit("rigor", "R9", ok, sandbox, "report")["status"] == "pass"
