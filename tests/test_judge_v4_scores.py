"""judge_v4 and the icml_judge_matrix preset (mock provider, offline): judge_v4 is judge_v3 verbatim plus a score
block; the preset's six judges form a model x effort matrix; a judge_v4 run passes its gates, and each judge's
`scores` reach its saved structured output (also through a coverage supplement), while judge_v3 judges are asked
for none."""

import asyncio
import json

import pytest

from conftest import mock_override
from paper_adversary import gates, service
from paper_adversary.config import ConfigError, check_config, load_config, plan_agents
from paper_adversary.pipeline import Pipeline
from paper_adversary.prompts import load_prompt
from paper_adversary.providers import mock as mock_module
from paper_adversary.registry import ModelRegistry
from paper_adversary.store import RunStore
from paper_adversary.util import read_json, read_jsonl

REG = ModelRegistry.load()
OPUS, FABLE = "claude-opus-5-5", "claude-fable-5-1"
MATRIX = {"J1": (OPUS, "high"), "J2": (OPUS, "xhigh"), "J3": (OPUS, "max"),
          "J4": (FABLE, "high"), "J5": (FABLE, "xhigh"), "J6": (FABLE, "max")}
SCORE_KEYS = ["originality", "theory", "empirical_soundness", "significance", "presentation", "overall", "confidence",
              "p_accept", "p_accept_rationale"]
TRUSTED = {"presets": {"selectable": ["routine", "icml_judge_matrix"]}}
RUBRIC = "---\nname: matrix_with_tests\ntests:\n" + "".join(
    f"  - {{id: RT{i}, text: 'mock rubric test {i}'}}\n" for i in range(1, 7)) + "---\nJudge for ICML.\n"


def _scores(aid: str) -> dict:
    """Distinct scores per judge, so a mix-up between judges would show."""
    n = int(aid[1:])
    return {"originality": 1 + n % 4, "theory": 1 + (n + 1) % 4, "empirical_soundness": 1 + (n + 2) % 4,
            "significance": 1 + (n + 3) % 4, "presentation": 4, "overall": 1 + n % 5, "confidence": 1 + (n + 2) % 5,
            "p_accept": round(n / 10, 2), "p_accept_rationale": f"{aid}: mock rationale"}


@pytest.fixture
def scoring_mock(monkeypatch):
    """The mock judge, answering a prompt that asks for scores (judge_v4) as such a judge would: a Scores section
    after its rubric tests and a scores object in its JSON block. Judges whose prompt asks for none are unchanged."""
    original = mock_module._render

    def render(req, *args, **kwargs):
        text = original(req, *args, **kwargs)
        if req.role != "judge" or '"scores": {' not in req.system_prompt:
            return text
        head, fence, block = text.rpartition("```json\n")
        data = json.loads(block.removesuffix("\n```"))
        data["scores"] = _scores(req.agent_id)
        section = "## Scores\n\n" + "".join(f"- {k}: {v}\n" for k, v in data["scores"].items()) + "\n"
        return head + section + fence + json.dumps(data, indent=2) + "\n```"

    monkeypatch.setattr(mock_module, "_render", render)


def _run(store, stages):
    return asyncio.run(Pipeline(store).run(list(stages)))


# ---------------------------------------------------------------- the prompt


def test_judge_v4_is_judge_v3_verbatim_plus_a_score_block():
    v3, v4 = load_prompt("judge_v3"), load_prompt("judge_v4")
    assert (v4.meta["role"], v4.meta["version"]) == ("judge", 4)
    assert v4.meta["rubric_tests"] is True and v4.meta["member_severity"] is True
    assert v4.meta["description"].startswith(v3.meta["description"])
    # every line of v3's template is in v4, in order: v4 only adds lines (the JSON block gains its scores object)
    # and appends sentences to v3's closing paragraph
    added = iter(v4.body.splitlines())
    assert all(any(other == line or (line and other.startswith(line + " ")) for other in added)
               for line in v3.body.splitlines())
    assert len(v4.body.splitlines()) > len(v3.body.splitlines())
    assert v4.variables == v3.variables
    assert "## Scores" in v4.body.split("# Output format", 1)[1] and "scores" not in v3.body
    out = v4.body.split("# Output format", 1)[1]
    assert out.index("## Rubric tests") < out.index("## Scores") < out.index("## Disagreements between refuters")
    shape3, shape4 = json.loads(gates.json_shape(v3.body)), json.loads(gates.json_shape(v4.body))
    assert set(shape4) == set(shape3) | {"scores"} and list(shape4["scores"]) == SCORE_KEYS
    # the gates read the prompt the same way: same leading section, no numbered sections
    assert gates.leading_section(v4.body) == gates.leading_section(v3.body) == "Overall assessment"
    assert gates.required_sections(v4.body) == gates.required_sections(v3.body) == []


# ---------------------------------------------------------------- the preset


def test_icml_judge_matrix_is_trusted_and_assigns_the_panel():
    with pytest.raises(ConfigError, match="cannot be chosen per run"):
        load_config({"preset": "icml_judge_matrix"})  # max effort: trusted settings only
    cfg, _ = load_config(None, {"preset": "icml_judge_matrix"})
    assert cfg.preset == "icml_judge_matrix" and [i for i in check_config(cfg, REG) if i.level == "error"] == []
    plan = {s.agent_id: (s.model_id, s.effort, (s.lens or {}).get("id"), s.prompt_name) for s in plan_agents(cfg, REG)}
    assert plan == {
        "INTAKE": (OPUS, "high", None, "intake_v1"),
        "N1": (OPUS, "high", "direct_and_recent", "novelty_v3"),
        "N2": (FABLE, "high", "equivalents_and_gaps", "novelty_v3"),
        "R1": (FABLE, "xhigh", "theory", "rigor_v2"), "R2": (OPUS, "high", "empirical_and_claims", "rigor_v2"),
        "F1": (OPUS, "high", "fit_overall", "fit_v2"),
        **{aid: (model, effort, None, "judge_v4") for aid, (model, effort) in MATRIX.items()},
        "S1": (FABLE, "max", None, "synthesis_v4"), "C1": (FABLE, "xhigh", None, "critic_v3")}
    assert cfg.novelty.tools == ["web_search", "web_fetch", "literature", "prior_text"]
    assert [r.timeout_minutes for r in (cfg.novelty, cfg.rigor, cfg.fit, cfg.judge)] == [90, 150, 90, 180]
    stage2, _ = load_config(None, {"preset": "stage2_split"})
    assert cfg.verifier == stage2.verifier  # the blind verifiers exactly as in stage2_split
    assert not cfg.followup.auto_start and not cfg.followup.auto_continue and cfg.followup.max_rounds == 0
    assert cfg.plan_limit.model_window_reserve == 0.25
    with pytest.raises(ConfigError, match="followup.max_rounds"):  # a per-run override cannot allow a round
        load_config({"followup": {"max_rounds": 1}}, {"preset": "icml_judge_matrix"})


# ---------------------------------------------------------------- end to end


def test_a_judge_v4_matrix_run_passes_its_gates_and_keeps_the_scores(runs_dir, sample_paper, tmp_path, scoring_mock):
    rubric = tmp_path / "rubric_matrix.md"
    rubric.write_text(RUBRIC)
    info = service.create_run(str(sample_paper), venue="ICML 2027", rubric=str(rubric), submission_type="paper",
                              config_override=mock_override("icml_judge_matrix"), trusted_config=TRUSTED)
    store = RunStore.open(info["run_id"])
    stages = service.full_review_stages(store, info)
    assert "followup" not in stages  # followup.auto_start: false
    result = _run(store, stages)
    assert result["outcome"] == "complete", result
    agents = store.load_state()["agents"]
    judges = sorted(aid for aid, a in agents.items() if a["role"] == "judge")
    assert judges == sorted(MATRIX)
    assert all(a["status"] == "complete" for a in agents.values() if a["role"] != "verifier"), {
        aid: a["status"] for aid, a in agents.items()}
    assert not any(a.get("round") for a in agents.values())  # no follow-up round ran

    # each judge ran on its own cell of the matrix (plan, saved assignment, and what the provider was asked)
    assert {aid: (agents[aid]["model"], agents[aid]["effort"]) for aid in judges} == MATRIX
    saved = {a["agent_id"]: (a["model"], a["effort"]) for a in read_json(store.dir / "assignment.json")["agents"]}
    assert {aid: saved[aid] for aid in judges} == MATRIX
    calls = {(u["agent_id"], u["stage"]): (u["model_requested"], u["effort"])
             for u in read_jsonl(store.dir / "logs" / "usage.jsonl")}
    assert {aid: calls[(aid, "judge")] for aid in judges} == MATRIX
    assert calls[("J3", "supplement")] == MATRIX["J3"]  # the mock's J3 leaves objections out: a supplement

    for aid in judges:
        gate = read_json(store.gate_path(aid, "judge"))
        assert gate["verdict"] in (gates.PASS, gates.WARN) and agents[aid]["gate"]["reasons"] == []
        assert agents[aid]["structured"] == "ok"
        side = read_json(store.sidecar_path(aid, "judge", ".json"))
        assert side["data"]["scores"] == _scores(aid), aid  # kept as written, also after J3's supplement merge
        assert len(side["rubric_rulings"]) == 6
        report = store.report_path(aid, "judge").read_text()
        assert "## Scores" in report and '"scores": {' in report
        system = (store.agent_log_dir(aid) / "attempt-1" / "system_prompt.md").read_text()
        assert "# Scores" in system and "would be accepted at ICML 2027" in system
        user = (store.agent_log_dir(aid) / "attempt-1" / "user_prompt.md").read_text()
        assert not any(f"CANARY_{other}" in user for other in judges if other != aid)  # judges stay blind
    assert "ruled on in a coverage supplement" in " ".join(agents["J3"]["gate"]["warnings"])

    matrix = read_json(store.role_dir("judge") / "judgment_matrix.json")
    assert [t["id"] for t in matrix["rubric_tests"]] == [f"RT{i}" for i in range(1, 7)]
    assert all(not t["not_ruled_by"] for t in matrix["rubric_tests"])
    assert all(set(row["verdicts"]) == set(judges) for row in matrix["rows"])

    # a follow-up asked for anyway (as `resume --through followup` would) runs nothing: max_rounds is 0
    result = _run(store, ["followup"])
    assert result["outcome"] == "complete" and result["message"] == "follow-up: disabled", result
    assert not any(a.get("round") for a in store.load_state()["agents"].values())


def test_judge_v3_judges_are_asked_for_no_scores(runs_dir, sample_paper, tmp_path, scoring_mock):
    """The same scoring mock on stage2_split (judge_v3): no judge is asked for scores, so none are saved."""
    rubric = tmp_path / "rubric_matrix.md"
    rubric.write_text(RUBRIC)
    info = service.create_run(str(sample_paper), rubric=str(rubric),
                              config_override=mock_override("stage2_split", items={"C1": []}),
                              trusted_config={"presets": {"selectable": ["routine", "stage2_split"]}})
    store = RunStore.open(info["run_id"])
    result = _run(store, ("intake", "novelty", "rigor", "fit", "judge"))
    assert result["outcome"] == "complete", result
    for aid in ("J1", "J2"):
        assert store.load_state()["agents"][aid]["status"] == "complete"
        assert "scores" not in read_json(store.sidecar_path(aid, "judge", ".json"))["data"]
        assert "## Scores" not in store.report_path(aid, "judge").read_text()
        assert '"scores"' not in (store.agent_log_dir(aid) / "attempt-1" / "system_prompt.md").read_text()
