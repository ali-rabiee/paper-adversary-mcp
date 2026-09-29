"""Configuration loading, validation against the model registry, and agent planning."""

import pytest

from paper_adversary.config import ConfigError, check_config, load_config, plan_agents
from paper_adversary.registry import ModelRegistry


def test_defaults_match_the_requested_split():
    cfg, _ = load_config()
    reg = ModelRegistry.load()
    assert check_config(cfg, reg) == [] or all(i.level == "warning" for i in check_config(cfg, reg))
    specs = plan_agents(cfg, reg)
    by_role = {}
    for s in specs:
        by_role.setdefault(s.role, []).append(s)
    assert [len(by_role[r]) for r in ("novelty", "rigor", "fit", "judge", "synthesis", "critic")] == [4, 3, 3, 3, 1, 1]
    assert {(s.model_id, s.effort) for s in by_role["rigor"]} == {("claude-fable-5-1", "max")}
    assert {(s.model_id, s.effort) for s in by_role["novelty"]} == {("claude-opus-5-5", "high")}
    assert by_role["critic"][0].effort == "xhigh"
    assert [s.agent_id for s in by_role["novelty"]] == ["N1", "N2", "N3", "N4"]
    assert len({s.lens["id"] for s in by_role["novelty"]}) == 4  # distinct primary focus per agent
    assert all(s.lens is None for s in by_role["judge"])


def test_override_merges_deeply():
    cfg, raw = load_config({"novelty": {"agents": 6}, "rigor": {"effort": "high"}})
    assert cfg.novelty.agents == 6 and cfg.novelty.model == "opus-5.5" and cfg.rigor.effort == "high"
    specs = plan_agents(cfg, ModelRegistry.load())
    n = [s for s in specs if s.role == "novelty"]
    assert "independent pass 2" in n[4].lens["title"]  # lenses cycle when agents outnumber them


def test_yaml_string_override():
    cfg, _ = load_config("fit:\n  agents: 1\n")
    assert cfg.fit.agents == 1


@pytest.mark.parametrize("override, message", [
    ({"rigor": {"effort": "extreme"}}, "effort 'extreme'"),
    ({"judge": {"model": "gpt-5"}}, "unknown model alias"),
    ({"judge": {"tools": ["web_search"]}}, "only refuters may use search tools"),
    ({"synthesis": {"prompt": "synthesis_v9"}}, "not found"),
])
def test_bad_configs_are_rejected(override, message):
    cfg, _ = load_config(override)
    errors = [str(i) for i in check_config(cfg, ModelRegistry.load()) if i.level == "error"]
    assert any(message in e for e in errors), errors


def test_unknown_keys_fail_fast():
    with pytest.raises(ConfigError, match="novelty.agentz"):
        load_config({"novelty": {"agentz": 3}})


def test_full_model_ids_are_accepted_with_a_warning():
    cfg, _ = load_config({"fit": {"model": "claude-sonnet-9"}})
    issues = check_config(cfg, ModelRegistry.load())
    assert any(i.level == "warning" and "claude-sonnet-9" in i.message for i in issues)
