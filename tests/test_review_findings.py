"""Regression tests for the issues found by the blind code review."""

import asyncio

import pytest

from conftest import mock_override
from paper_adversary import service
from paper_adversary.config import ConfigError, check_config, deep_merge, load_config
from paper_adversary.ingest import ingest
from paper_adversary.pipeline import Pipeline
from paper_adversary.registry import ModelRegistry
from paper_adversary.store import RunStore

ALL = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]


def _create(sample_paper, extra=None, **mock_options):
    override = deep_merge(mock_override(**mock_options), extra or {})
    info = service.create_run(str(sample_paper), config_override=override)
    return RunStore.open(info["run_id"])


def _run(store, stages=ALL, **kw):
    return asyncio.run(Pipeline(store).run(stages, **kw))


def test_quoting_judges_do_not_trip_the_guard(runs_dir, sample_paper):
    """Judges quote refuters; later judges, reruns and the critic must still start (finding 2)."""
    store = _create(sample_paper, {"concurrency": {"max_parallel_agents": 1}}, quote_inputs=True)
    assert _run(store)["outcome"] == "complete"
    for rerun in (["J2"], ["C1"], ["S1"]):
        stage = {"J": "judge", "C": "critic", "S": "synthesis"}[rerun[0][0]]
        result = _run(store, [stage], rerun=rerun)
        assert result["outcome"] == "complete", result
    agents = store.load_state()["agents"]
    assert all(a["isolation_audit"] == "pass" for a in agents.values())


def test_agent_is_complete_before_reference_checking(runs_dir, sample_paper, monkeypatch):
    """A slow or failing reference check must not leave a finished agent unfinished (finding 3)."""
    store = _create(sample_paper, {"search": {"refcheck": True}})

    async def broken(*_a, **_k):
        raise RuntimeError("scholarly APIs down")

    monkeypatch.setattr("paper_adversary.search.refcheck.check_references", broken)
    assert _run(store, ["novelty"])["outcome"] == "complete"
    agents = store.load_state()["agents"]
    assert agents["N1"]["status"] == "complete" and "error" in agents["N1"]["refcheck"]
    assert not store.sidecar_path("N1", "novelty", ".refcheck.json").exists()

    async def works(refs, search, max_refs=60, cancel=None):
        return {"checked": len(refs), "skipped": 0, "counts": {"verified": len(refs)}, "items": []}

    monkeypatch.setattr("paper_adversary.search.refcheck.check_references", works)
    _run(store, ["rigor", "fit"])
    assert _run(store, ["judge"])["outcome"] == "complete"  # the judge stage backfills missing checks first
    assert store.sidecar_path("N1", "novelty", ".refcheck.json").exists()


def test_protected_provider_keys_cannot_be_set_per_run():
    for key, value in (("allow_api_key", True), ("extra_args", ["--add-dir", "/"]), ("safe_mode", False),
                       ("env_passthrough", ["ANTHROPIC_API_KEY"]), ("claude_binary", "/tmp/evil")):
        with pytest.raises(ConfigError, match=key):
            load_config({"provider": {key: value}})


def test_isolation_breaking_flags_are_rejected(monkeypatch, tmp_path):
    user_file = tmp_path / "mine.yaml"
    user_file.write_text("provider:\n  extra_args: ['--add-dir', '/home']\n")
    monkeypatch.setenv("PAPER_ADVERSARY_CONFIG", str(user_file))
    cfg, _ = load_config()
    errors = [str(i) for i in check_config(cfg, ModelRegistry.load()) if i.level == "error"]
    assert any("--add-dir" in e for e in errors)


def test_api_key_login_is_refused_before_any_agent(runs_dir, sample_paper):
    store = _create(sample_paper, {"provider": {"preflight": "auth_only"}}, auth_method="api_key")
    result = _run(store)
    assert result["outcome"] == "failed" and "billing_guard" in result["message"]
    assert all(a["attempts"] == 0 for a in store.load_state()["agents"].values())


def test_intake_output_never_reaches_refuters(runs_dir, sample_paper):
    """The orchestrator's reading of the paper stays out of refuter and judge headers (finding 6)."""
    store = _create(sample_paper)
    _run(store, ["intake"])
    assert store.load_metadata()["intake"]["field"] == "machine learning"
    _run(store, ["novelty", "rigor", "fit", "judge"])
    for aid in ("N1", "R1", "J1"):
        prompt = (store.dir / "logs" / "agents" / aid / "attempt-1" / "user_prompt.md").read_text()
        assert "Research field: not specified" in prompt and "machine learning" not in prompt


def test_headings_without_a_title_still_split(runs_dir):
    """Plain text (or a PDF without a detected title) keeps top-level units (finding 7)."""
    filler = "Words about the work. " * 400
    text = "\n\n".join(["Abstract", "We study X.", "1 Introduction", filler, "2 Method", "Key idea. " + filler,
                         "3 Experiments", filler, "References", "[1] A. B. Something. 2020."])
    r = ingest(paper_text=text)
    tops = [s.title for s in r.sections if s.level == 1]
    assert tops == ["Abstract", "1 Introduction", "2 Method", "3 Experiments", "References"]
    from paper_adversary.budget import plan_document

    plan = plan_document(r.text_md, r.sections, int(len(r.text_md) * 0.3 * 0.5), ["abstract", "method"], 0.3)
    assert plan.mode == "sectioned" and "Key idea." in plan.text


def test_cancel_during_plan_limit_wait_is_an_interruption(runs_dir, sample_paper):
    store = _create(sample_paper, {"plan_limit": {"default_wait_minutes": 5, "max_wait_hours": 2}},
                    fail={"R1": ["plan_limit"]})

    async def go():
        pipeline = Pipeline(store)
        task = asyncio.create_task(pipeline.run(["rigor"]))
        await asyncio.sleep(1.0)
        pipeline.cancel.set()
        return await task

    assert asyncio.run(go())["outcome"] == "cancelled"
    assert store.load_state()["agents"]["R1"]["status"] == "interrupted"


@pytest.mark.parametrize("text, codes, status, expected", [
    # verbatim messages returned for this account on 2026-09-29
    ("You've reached your Fable limit. Switch to another model, or manage usage credits at "
     "claude.ai/settings/usage?from=cc_cli_limit_message, to continue.", ["rate_limit"], 429, "plan_limit"),
    ("API Error: 400 Claude Code 2.1.266 does not support this model; version 2.1.280 or newer is required. "
     "Run 'claude update', or update the Claude desktop app, then try again.", ["invalid_request"], 400,
     "model_unavailable"),
    ("Failed to authenticate. API Error: 401 OAuth access token is invalid.", ["authentication_failed"], 401,
     "auth"),
])
def test_real_account_messages(text, codes, status, expected):
    from paper_adversary.providers.claude_code import classify_error

    assert classify_error([text], codes, status, None)[0].value == expected


def test_waiting_agents_free_their_slot(runs_dir, sample_paper):
    """An agent waiting out a plan limit must not block agents on other models (one slot here)."""
    from paper_adversary.util import read_jsonl

    store = _create(sample_paper, {"concurrency": {"max_parallel_agents": 1},
                                   "plan_limit": {"default_wait_minutes": 0.05}}, fail={"R1": ["plan_limit"]})
    assert _run(store, ["rigor", "fit"])["outcome"] == "complete"
    events = read_jsonl(store.events_path)
    wait_at = next(i for i, e in enumerate(events) if e["event"] == "plan_limit_wait")
    r1_done = next(i for i, e in enumerate(events) if e["event"] == "agent_complete" and e["agent_id"] == "R1")
    others = [e["agent_id"] for e in events[wait_at:r1_done] if e["event"] == "agent_complete"]
    assert others, "no other agent ran while R1 waited for the plan limit"
