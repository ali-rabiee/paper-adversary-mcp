"""End-to-end pipeline runs with the offline mock provider."""

import asyncio
import re

import yaml

from conftest import mock_override
from paper_adversary import service
from paper_adversary.pipeline import Pipeline
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore

ALL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]


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
    assert "not classified by J3" in matrix  # J3 skips second objections
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
    assert meta["structured_block"] != "ok"
    assert "J2" in service.get_report(store.run_id, "matrix").split("could not be parsed")[-1]
