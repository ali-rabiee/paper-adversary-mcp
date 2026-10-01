"""Detached workers (launch, cancel, resume) and the MCP tool surface."""

import asyncio
import json
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from conftest import mock_override
from paper_adversary import server, service
from paper_adversary.store import RunStore
from paper_adversary.util import lock_is_held
from paper_adversary.worker import WorkerBusy, cancel, launch

ALL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]


def _wait_idle(store, timeout=120):
    deadline = time.monotonic() + timeout
    while lock_is_held(store.lock_path):
        assert time.monotonic() < deadline, "worker did not finish"
        time.sleep(0.2)


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "text", None))


def call(name, **args):
    return _text(asyncio.run(server.mcp.call_tool(name, args)))


def test_detached_worker_runs_full_pipeline(runs_dir, sample_paper):
    info = service.create_run(str(sample_paper), config_override=mock_override())
    store = RunStore.open(info["run_id"])
    started = launch(store, ALL)
    assert started["started"]
    _wait_idle(store)
    state = store.load_state()
    assert all(a["status"] == "complete" for a in state["agents"].values()), service.status_text(store.run_id)
    worker = json.loads(store.worker_path.read_text())
    assert worker["status"] == "exited" and worker["outcome"] == "complete"


def test_cancel_then_resume(runs_dir, sample_paper):
    info = service.create_run(str(sample_paper), config_override=mock_override(delay_seconds=3))
    store = RunStore.open(info["run_id"])
    launch(store, ALL)
    time.sleep(1.5)
    with pytest.raises(WorkerBusy):
        launch(store, ALL)
    result = cancel(store)
    assert result["cancelled"]
    assert not lock_is_held(store.lock_path)
    state = store.load_state()
    assert any(a["status"] == "interrupted" for a in state["agents"].values())
    assert not any(a["status"] == "complete" for a in state["agents"].values() if a["role"] == "judge")
    # resume finishes the job without redoing completed agents
    done_before = {aid for aid, a in state["agents"].items() if a["status"] == "complete"}
    raw = store.config_path.read_text().replace("delay_seconds: 3", "delay_seconds: 0.01")
    store.config_path.write_text(raw)
    launch(store, ALL)
    _wait_idle(store)
    after = store.load_state()["agents"]
    assert all(a["status"] == "complete" for a in after.values())
    for aid in done_before:
        assert after[aid]["attempts"] == state["agents"][aid]["attempts"]


def test_mcp_tools_end_to_end(runs_dir, sample_paper):
    tools = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"create_review_run", "run_refuters", "run_judges", "run_synthesis", "run_completeness_critic",
            "run_full_review", "get_run_status", "get_report", "get_run_cost", "resume_run", "cancel_run",
            "list_runs", "validate_config", "release_quarantine", "run_verification", "add_prior_fulltext",
            "run_followup"} <= tools

    created = call("create_review_run", paper_path=str(sample_paper), venue="ICML 2027",
                   config_override=mock_override())
    run_id = created.split("Created run ")[1].split()[0]
    assert "Planned agents" in created and "N4" in created

    with pytest.raises(ToolError, match="refuters not complete"):
        asyncio.run(server.mcp.call_tool("run_judges", {"run_id": run_id}))

    status = call("run_refuters", run_id=run_id, wait_seconds=60)
    assert "Novelty:" in status
    store = RunStore.open(run_id)
    _wait_idle(store)
    call("run_judges", run_id=run_id, wait_seconds=60)
    _wait_idle(store)
    call("run_synthesis", run_id=run_id, wait_seconds=60)
    _wait_idle(store)
    final = call("run_completeness_critic", run_id=run_id, wait_seconds=60)
    _wait_idle(store)
    final = call("get_run_status", run_id=run_id)
    assert "Completeness critic: complete" in final and "Review complete" in final
    assert "Mock" in call("get_report", run_id=run_id, report_type="synthesis")
    assert "Judgment matrix" in call("get_report", run_id=run_id, report_type="matrix")
    assert "Phase" in call("get_run_cost", run_id=run_id)
    assert run_id in call("list_runs")
    assert "Nothing to resume" in call("resume_run", run_id=run_id)


def test_run_full_review_tool(runs_dir, sample_paper):
    out = call("run_full_review", paper_path=str(sample_paper), config=mock_override(), wait_seconds=90)
    run_id = out.split("Created run ")[1].split()[0]
    _wait_idle(RunStore.open(run_id))
    assert "Review complete" in call("get_run_status", run_id=run_id)


def test_tool_errors_are_clean(runs_dir):
    with pytest.raises(ToolError, match="no run 'nope'"):
        asyncio.run(server.mcp.call_tool("get_run_status", {"run_id": "nope"}))
    with pytest.raises(ToolError, match="no paper given"):
        asyncio.run(server.mcp.call_tool("create_review_run", {}))
    with pytest.raises(ToolError, match="effort"):
        asyncio.run(server.mcp.call_tool("create_review_run", {
            "paper_text": "x " * 100, "config_override": {"rigor": {"effort": "extreme"}}}))


def test_validate_config_tool(runs_dir):
    out = call("validate_config", config_override={"provider": {"type": "mock"}})
    assert "Config: OK" in out and "claude-fable-5-1" in out and "claude-opus-5-5" in out
