"""The offline evaluation harness: case validation, paired runs on identical evidence, and the report
(mock provider; nothing here is evidence about any preset)."""

import asyncio
import shutil
from pathlib import Path

import pytest
import yaml

from conftest import mock_override
from paper_adversary import evaluation
from paper_adversary.config import load_config
from paper_adversary.context import ContextBuilder
from paper_adversary.pipeline import Pipeline, spec_from_state
from paper_adversary.registry import ModelRegistry
from paper_adversary.store import RunStore
from paper_adversary.util import read_json

FIXTURE = Path(__file__).parent / "fixtures" / "eval_synthetic"
TEMPLATE = Path(__file__).parents[1] / "eval" / "case_template"
PASSAGE = ("monotonically increasing guidance schedules, including a simple linear ramp from zero to the maximum "
           "weight, consistently improve sample quality")


def _decisive(ident, title, passage=PASSAGE):
    return {"category": "already_done", "severity": "fatal", "basis": "full_text",
            "prior": {"title": title, "arxiv_id": ident}, "prior_passage": passage}


def _case(tmp_path, **changes):
    root = tmp_path / "case"
    shutil.copytree(FIXTURE, root)
    data = yaml.safe_load((root / "case.yaml").read_text())
    data.update(changes)
    (root / "case.yaml").write_text(yaml.safe_dump(data))
    return root


def test_cases_are_validated_and_templates_never_run(runs_dir, tmp_path):
    assert evaluation.validate_case(TEMPLATE)[0] == []
    with pytest.raises(evaluation.CaseError, match="template"):
        evaluation.prepare(TEMPLATE, ["routine"])
    assert evaluation.validate_case(FIXTURE)[0] == []
    bad = _case(tmp_path, category="vibes", status="adjudicated", submission="missing.md")
    errors, _ = evaluation.validate_case(bad)
    assert any("category" in e for e in errors) and any("missing.md" in e for e in errors)
    assert any("adjudicated_by" in e for e in errors)  # an adjudicated case names its adjudicator
    data = yaml.safe_load((bad / "case.yaml").read_text())
    data["ground_truth"]["serious_issues"][0]["prior_work"] = ["a title, not an ID"]
    (bad / "case.yaml").write_text(yaml.safe_dump(data))
    assert any("by arXiv ID or DOI" in e for e in evaluation.validate_case(bad)[0])


def test_a_pair_runs_on_identical_evidence_and_is_reported(runs_dir, tmp_path):
    override = mock_override(decisive={"N1": _decisive("2401.00001", "Analysis of Guidance Weight Schedulers"),
                                       "N2": _decisive("2401.00002", "A Different Paper About Guidance Weights",
                                                       "a learned guidance network predicts a per-sample weight from "
                                                       "the prompt embedding, which improves alignment")})
    pair = evaluation.prepare(FIXTURE, ["routine", "legacy_thorough"], "reasoning", override)
    runs = [RunStore.open(r["run_id"]) for r in pair["runs"]]
    assert [s.load_metadata()["preset"] for s in runs] == ["routine", "legacy_thorough"]
    for store in runs:  # prepared, not started; search off; the same frozen evidence in both
        state = store.load_state()
        assert all(a["status"] == "pending" for a in state["agents"].values())
        cfg = store.load_config()
        assert cfg.novelty.tools == ["prior_text"] and not cfg.search.refcheck and cfg.search.fulltext.offline
        spec = spec_from_state(state["agents"]["N1"])
        ctx = ContextBuilder(store, cfg, ModelRegistry.load()).build(spec, state)
        assert "<evidence_pack" in ctx.user_text and "arxiv:2401.00002" in ctx.user_text
        args = ctx.tool_server["args"]
        assert args[args.index("--prior-text") + 1] == "scoped" and "arxiv:2401.00001" in args[args.index("--prior-keys") + 1]
    for store in runs:
        result = asyncio.run(Pipeline(store).run(["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]))
        assert result["outcome"] == "complete", result
    report = evaluation.report(pair["pair_dir"])
    assert report["identical_evidence"] and len(report["runs"]) == 2
    routine = report["runs"][0]
    gt = routine["ground_truth"]
    assert gt["status"] == "synthetic" and not gt["counts_as_evidence"]
    assert [f["id"] for f in gt["prior_work_found"]] == ["GT1"] and not gt["prior_work_missed"]
    assert [f["id"] for f in gt["false_fatal_prior_work"]] == ["NF1"]  # J2 called the different paper FATAL
    assert gt["technical_issues_to_score_by_hand"] == ["GT2"]
    assert routine["tokens"] and routine["calls"] > 0 and routine["citations"]["outside_the_evidence_pack"] == []
    md = (Path(pair["pair_dir"]) / "comparison.md").read_text()
    assert "SYNTHETIC case" in md and "routine (" in md and "legacy_thorough (" in md
    sheet = Path(pair["pair_dir"]) / "manual_scores.yaml"
    scores = yaml.safe_load(sheet.read_text())
    first = report["runs"][0]["run_id"]
    scores["scored_by"] = "test"
    scores["runs"][first]["missed_serious_issues"] = ["GT2"]
    filled = tmp_path / "scores.yaml"
    filled.write_text(yaml.safe_dump(scores))
    again = evaluation.report(pair["pair_dir"], filled)
    assert again["runs"][0]["manual"]["missed_serious_issues"] == ["GT2"]
    assert read_json(Path(pair["pair_dir"]) / "comparison.json")["runs"][0]["manual"]["scored_by"] == "test"


def test_discovery_mode_keeps_search_and_seeds_nothing(runs_dir):
    pair = evaluation.prepare(FIXTURE, ["routine"], "discovery", mock_override())
    store = RunStore.open(pair["runs"][0]["run_id"])
    cfg = store.load_config()
    assert "web_search" in cfg.novelty.tools and pair["evidence"] == []
    assert not (store.prior_dir / "index.json").exists()


def test_the_preset_argument_wins_over_an_override_naming_another(runs_dir):
    cfg, raw = load_config({"preset": "legacy_thorough"}, None, "routine")
    assert cfg.preset == "routine" and raw["preset"] == "routine" and cfg.novelty.agents == 2
