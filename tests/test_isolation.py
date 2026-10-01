"""The isolation policy and guard reject forbidden inputs regardless of how they got in."""

import asyncio
import json

import pytest

from conftest import FAKE_SECRETS, mock_override
from paper_adversary import service
from paper_adversary.config import AgentSpec
from paper_adversary.context import ContextBuilder
from paper_adversary.isolation import ROLE_VISIBILITY, ArtifactAccess, IsolationGuard, IsolationViolation, find_secrets
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
    assert {m["kind"] for m in jctx.manifest} == {"paper", "novelty", "rigor", "fit", "rubric", "evidence",
                                                  "verification"}
    assert "CANARY_J1" not in jctx.user_text and "CANARY_N1" in jctx.user_text


def _transcript(path, *events, init=True, result=True, denials=()):
    lines = ([{"kind": "init"}] if init else []) + list(events) + (
        [{"kind": "result", "permission_denials": [{"tool_use_id": d} for d in denials]}] if result else [])
    path.write_text("".join(json.dumps(e) + "\n" for e in lines))
    return path


def test_audit_flags_reads_outside_the_sandbox(tmp_path, finished_run):
    guard = IsolationGuard(finished_run)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    read_passwd = {"kind": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/etc/passwd"}}
    read_pdf = {"kind": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "paper.pdf"}}
    got = {"kind": "tool_result", "tool_use_id": "t1", "is_error": False, "content": "root:x:0:0"}
    result = guard.audit("rigor", "R9", _transcript(tmp_path / "a.jsonl", read_passwd, got, read_pdf), sandbox,
                         "report")
    assert result["status"] == "fail" and any("/etc/passwd" in f for f in result["findings"])
    ok = _transcript(tmp_path / "ok.jsonl", read_pdf)
    assert guard.audit("rigor", "R9", ok, sandbox, "report")["status"] == "pass"
    # a refused attempt is recorded but does not fail the audit
    refused = guard.audit("rigor", "R9", _transcript(tmp_path / "d.jsonl", read_passwd, denials=("t1",)), sandbox,
                          "report")
    assert refused["status"] == "pass" and refused["denied"]


def test_audit_without_evidence_is_unverifiable(tmp_path, finished_run):
    guard = IsolationGuard(finished_run)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    assert guard.audit("rigor", "R9", tmp_path / "missing.jsonl", sandbox, "r")["status"] == "unverifiable"
    assert guard.audit("rigor", "R9", None, sandbox, "r")["status"] == "unverifiable"
    (tmp_path / "empty.jsonl").write_text("")
    assert guard.audit("rigor", "R9", tmp_path / "empty.jsonl", sandbox, "r")["status"] == "unverifiable"
    no_init = _transcript(tmp_path / "n.jsonl", init=False)
    assert "no init event" in " ".join(guard.audit("rigor", "R9", no_init, sandbox, "r")["unverifiable"])
    unpaired = {"kind": "tool_use", "id": "t9", "name": "Read", "input": {"file_path": "~/.ssh/id_rsa"}}
    out = guard.audit("rigor", "R9", _transcript(tmp_path / "u.jsonl", unpaired), sandbox, "r")
    assert out["status"] == "unverifiable"
    out = guard.audit("rigor", "R9", _transcript(tmp_path / "s.jsonl", unpaired), None, "r")
    assert out["status"] == "unverifiable" and "sandbox is unknown" in out["unverifiable"][0]


def test_quarantined_reports_are_forbidden_for_every_role(finished_run):
    guard = IsolationGuard(finished_run)
    _, body = read_report(finished_run.dir / "novelty" / "N2.md")
    guard.check_prompt("judge", "J9", body)  # judges may normally read refuter reports
    guard.set_quarantined("N2", True)
    with pytest.raises(IsolationViolation, match="N2"):
        guard.check_prompt("judge", "J9", body)
    guard.set_quarantined("N2", False)
    guard.check_prompt("judge", "J9", body)


def test_secret_scan_refuses_credentials(monkeypatch):
    IsolationGuard.check_secrets("N1", "an ordinary prompt about sk-learn and AKIA-style naming")
    with pytest.raises(IsolationViolation, match="Anthropic"):
        IsolationGuard.check_secrets("N1", "token " + FAKE_SECRETS["anthropic"])
    monkeypatch.setenv("S2_API_KEY", "abcdefghijklmnop1234")
    with pytest.raises(IsolationViolation, match="S2_API_KEY"):
        IsolationGuard.check_secrets("N1", "rubric text abcdefghijklmnop1234 more text")


@pytest.mark.parametrize("text", [
    "export ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXX",
    'set ANTHROPIC_API_KEY="your-api-key-here" first',
    "the documented key AKIAIOSFODNN7EXAMPLE",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----",
    "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
])
def test_secret_scan_ignores_documentation_placeholders(text):
    """Papers about LLM tooling print placeholders; refusing them would make such papers unreviewable."""
    assert find_secrets(text) == []


@pytest.mark.parametrize("text, label", [
    ("S2_API_KEY=" + FAKE_SECRETS["s2"], "credential assignment"),
    (FAKE_SECRETS["pem_header"] + "\n" + FAKE_SECRETS["pem_body"] + "\n", "private key"),
    (FAKE_SECRETS["rsa_header"] + "\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,AB\n\n"
     + FAKE_SECRETS["pem_body"] + "\n", "private key"),
    (FAKE_SECRETS["aws"], "AWS"),
    (FAKE_SECRETS["github"], "GitHub"),
    ("a placeholder sk-ant-api03-XXXXXXXXXXXXXXXXXXXX then " + FAKE_SECRETS["anthropic_api"], "Anthropic"),
])
def test_secret_scan_still_catches_real_looking_credentials(text, label):
    assert any(label in f for f in find_secrets(text)), find_secrets(text)


def test_secret_scan_is_linear_on_hostile_text():
    import time

    for text in ("-----BEGIN " + "A" * 200_000, FAKE_SECRETS["pem_header"] + " " * 500_000, "sk-ant-" * 50_000,
                 FAKE_SECRETS["pem_header"] + "\n" + "a:\n" * 100_000):
        t0 = time.perf_counter()
        find_secrets(text)
        assert time.perf_counter() - t0 < 1.0
