"""Offline evaluation harness: paired comparison of presets on the same inputs, scored against cases a person
adjudicated.

A case (eval/README.md) is a submission, a fixed evidence pack of prior-paper full texts, and ground truth
written by a human adjudicator. `prepare` creates one run per preset with identical inputs: the same submission
and, in reasoning mode, the same frozen evidence snapshot with search switched off, so only the reviewers'
reasoning differs. Discovery mode leaves search on and is scored separately, never mixed with reasoning results.
Nothing is launched: the runs are started by the user. `report` reads the finished runs and records what can be
computed deterministically (missed serious prior-work issues, false fatal prior-work accusations, evidence
support and citation correctness, tokens, API-equivalent cost, elapsed time, repair and supplement calls) and
writes a scoring sheet for what a program cannot judge (technical issues). Template and synthetic cases never
yield quality claims, and one pair of runs is never evidence that one preset is better or cheaper.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import yaml

from paper_adversary.registry import ModelRegistry
from paper_adversary.reports import SEVERITY_RANK, normalize_severity
from paper_adversary.search.base import parse_identifier
from paper_adversary.search.fulltext import FullTextResult, write_snapshot
from paper_adversary.store import RunStore
from paper_adversary.usage import summarize
from paper_adversary.util import atomic_write_json, atomic_write_text, read_json, read_jsonl, runs_root, \
    sha256_text, utcnow, utcnow_iso

CATEGORIES = ("genuine_overlap", "different_prior_work", "valid_objection", "incorrect_objection")
STATUSES = ("template", "synthetic", "adjudicated")
MODES = ("reasoning", "discovery")
# Reasoning mode: the evidence pack is all the prior work there is. No search, no new downloads, no lookups.
REASONING_SETTINGS = {"novelty": {"tools": ["prior_text"]},
                      "search": {"refcheck": False, "fulltext": {"offline": True}}}


class CaseError(ValueError):
    pass


def evidence_key(identifier: str) -> str | None:
    """'arXiv:2404.13040v2' -> 'arxiv:2404.13040'; a DOI -> 'doi:<lowercase>'; anything else -> None."""
    kind, value = parse_identifier(str(identifier or ""))
    if kind == "arxiv":
        return f"arxiv:{value}"
    if kind == "doi":
        return f"doi:{value.lower()}"
    return None


# ---------------------------------------------------------------- cases


def load_case(case_dir: str | Path) -> tuple[Path, dict]:
    root = Path(case_dir).expanduser().resolve()
    path = root / "case.yaml" if root.is_dir() else root
    if not path.is_file():
        raise CaseError(f"no case.yaml at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise CaseError(f"{path} is not a YAML mapping")
    return path.parent, data


def validate_case(case_dir: str | Path) -> tuple[list[str], list[str]]:
    """(errors, warnings). A template case is valid but not runnable; only adjudicated cases count as evidence."""
    root, case = load_case(case_dir)
    errors, warnings = [], []
    for key in ("id", "category", "status", "submission"):
        if not case.get(key):
            errors.append(f"'{key}' is required")
    if case.get("category") and case["category"] not in CATEGORIES:
        errors.append(f"category must be one of {', '.join(CATEGORIES)}")
    status = case.get("status")
    if status and status not in STATUSES:
        errors.append(f"status must be one of {', '.join(STATUSES)}")
    if status == "template":
        warnings.append("a template: fill in the submission, evidence and ground truth before running it")
        return errors, warnings
    if case.get("submission") and not (root / case["submission"]).is_file():
        errors.append(f"submission file {case['submission']} not found")
    keys = []
    for i, ev in enumerate(case.get("evidence") or [], start=1):
        key = evidence_key((ev or {}).get("identifier"))
        if not key:
            errors.append(f"evidence {i}: identifier must be an arXiv ID or DOI")
        keys.append(key)
        if not (root / str((ev or {}).get("file") or "")).is_file():
            errors.append(f"evidence {i}: file {ev.get('file')} not found")
    if len(set(keys)) != len(keys):
        errors.append("an evidence identifier is listed twice")
    gt = case.get("ground_truth") or {}
    if status == "adjudicated":
        for key in ("adjudicated_by", "adjudicated_on"):
            if not gt.get(key):
                errors.append(f"ground_truth.{key} is required for an adjudicated case")
    if status == "synthetic":
        warnings.append("synthetic: for testing the harness only; its results are never evidence about quality")
    for i, issue in enumerate(gt.get("serious_issues") or [], start=1):
        if issue.get("kind") not in ("prior_work", "technical"):
            errors.append(f"serious issue {i}: kind must be prior_work or technical")
        if normalize_severity(issue.get("severity")) not in ("FATAL", "MAJOR_FIXABLE"):
            errors.append(f"serious issue {i}: severity must be FATAL or MAJOR_FIXABLE")
        if issue.get("kind") == "prior_work" and not [evidence_key(x) for x in issue.get("prior_work") or []
                                                      if evidence_key(x)]:
            errors.append(f"serious issue {i}: a prior_work issue names the anticipating paper(s) by arXiv ID or DOI")
    for i, item in enumerate(gt.get("not_fatal") or [], start=1):
        if not item.get("description"):
            errors.append(f"not_fatal {i}: a description is required")
    return errors, warnings


def _evidence_results(root: Path, case: dict) -> list[FullTextResult]:
    """The evidence pack as full-text results, extracted once so every run reads the same bytes."""
    from paper_adversary.search.extract import run_extraction

    out = []
    for ev in case.get("evidence") or []:
        path = root / ev["file"]
        key = evidence_key(ev["identifier"])
        if path.suffix.lower() in (".md", ".txt", ".markdown"):
            text, sections, title = path.read_text(encoding="utf-8"), [], ev.get("title")
        else:
            got = run_extraction(path, "pdf" if path.suffix.lower() == ".pdf" else "html")
            if got.get("status") != "ok":
                raise CaseError(f"cannot extract {path.name}: {got.get('reason')} {got.get('detail') or ''}".strip())
            text, sections, title = got["text_md"], got.get("sections") or [], ev.get("title") or got.get("title")
        kind, value = parse_identifier(ev["identifier"])
        out.append(FullTextResult("available", key=key, title=title, source="evaluation_evidence",
                                  sha256=sha256_text(text), text_md=text, sections=sections,
                                  fetched_at=utcnow_iso(), aliases=[ev["identifier"], value],
                                  arxiv_id=value if kind == "arxiv" else None, doi=value if kind == "doi" else None))
    return out


def prepare(case_dir: str | Path, presets: list[str], mode: str = "reasoning", config_override=None) -> dict:
    """Create one run per preset for this case, with identical inputs. Launches nothing; returns the pair."""
    from paper_adversary import service

    if mode not in MODES:
        raise CaseError(f"mode must be one of {', '.join(MODES)}")
    errors, _ = validate_case(case_dir)
    root, case = load_case(case_dir)
    if errors:
        raise CaseError("the case is invalid:\n" + "\n".join(f"  - {e}" for e in errors))
    if case["status"] == "template":
        raise CaseError("a template case cannot be run; supply a submission, evidence and ground truth")
    evidence = _evidence_results(root, case) if mode == "reasoning" else []
    stamp = utcnow().strftime("%Y%m%dT%H%M%S")
    pair_dir = runs_root() / "eval" / case["id"] / f"{stamp}-{mode}"
    runs = []
    for preset in presets:
        info = service.create_run(
            str(root / case["submission"]), title=case.get("title"), venue=case.get("venue"),
            submission_type=case.get("submission_type") or "auto", preset=preset, config_override=config_override,
            trusted_config=REASONING_SETTINGS if mode == "reasoning" else None,
            evaluation={"case": case["id"], "status": case["status"], "mode": mode, "pair": str(pair_dir),
                        "evidence": [{"key": r.key, "title": r.title, "sha256": r.sha256} for r in evidence]})
        store = RunStore.open(info["run_id"])
        for result in evidence:  # the same frozen snapshot in every run of the pair
            write_snapshot(store.prior_dir, result)
        runs.append({"preset": preset, "run_id": info["run_id"]})
    pair = {"case": case["id"], "case_dir": str(root), "status": case["status"], "category": case["category"],
            "mode": mode, "created_at": utcnow_iso(), "runs": runs,
            "evidence": [{"key": r.key, "sha256": r.sha256} for r in evidence]}
    atomic_write_json(pair_dir / "pair.json", pair)
    return {**pair, "pair_dir": str(pair_dir)}


# ---------------------------------------------------------------- measurements


def _cited_keys(store: RunStore, state: dict) -> dict[str, set[str]]:
    """Objection or item ID -> the prior papers it cites (normalized keys)."""
    out: dict[str, set[str]] = {}
    for aid, a in state["agents"].items():
        if a["role"] == "novelty":
            side = read_json(store.sidecar_path(aid, "novelty", ".json"), {}) or {}
            for obj in side.get("objections") or []:
                keys = {evidence_key(r.get("arxiv_id") or r.get("doi") or "") for r in obj.get("references") or []
                        if isinstance(r, dict)}
                out.setdefault(obj["id"], set()).update(k for k in keys if k)
            ev = read_json(store.sidecar_path(aid, "novelty", ".evidence.json"), {}) or {}
            for oid, entry in (ev.get("objections") or {}).items():
                out.setdefault(oid, set()).update(r["key"] for r in entry.get("references") or [] if r.get("key"))
    for rdir in sorted((store.dir / "followup").glob("round-*")):
        for it in (read_json(rdir / "items.json", {}) or {}).get("items") or []:
            keys = {evidence_key(r.get("arxiv_id") or r.get("doi") or "") for r in it.get("references") or []
                    if isinstance(r, dict)}
            out.setdefault(it["id"], set()).update(k for k in keys if k)
    return out


def _verdicts(store: RunStore) -> dict[str, list[str]]:
    """Objection or item ID -> every severity a judge or adjudicator gave it."""
    out: dict[str, list[str]] = {}
    sources = [store.role_dir("judge") / "judgment_matrix.json"] + sorted(
        (store.dir / "followup").glob("round-*/followup_matrix.json"))
    for path in sources:
        for row in (read_json(path, {}) or {}).get("rows") or []:
            out.setdefault(row["id"], []).extend(v.get("severity") for v in (row.get("verdicts") or {}).values()
                                                 if v.get("severity"))
    return out


def _elapsed_seconds(store: RunStore) -> float:
    """Wall-clock time of the run's jobs (from each job_started to its job_finished)."""
    total, start = 0.0, None
    for e in read_jsonl(store.events_path):
        at = dt.datetime.fromisoformat(e["at"]) if e.get("at") else None
        if e.get("event") == "job_started":
            start = at
        elif e.get("event") == "job_finished" and start and at:
            total += (at - start).total_seconds()
            start = None
    return total


def measure(run_id: str, case: dict) -> dict:
    """Everything recorded for one run of a pair."""
    store = RunStore.open(run_id)
    state = store.load_state()
    usage = read_jsonl(store.dir / "logs" / "usage.jsonl")
    summary = summarize(usage, ModelRegistry.load())
    total = summary["total"]
    stages = [u.get("stage") or "" for u in usage]
    verdicts = _verdicts(store)
    cited = _cited_keys(store, state)
    gate = read_json(store.role_dir("judge") / "evidence_gate.json", {}) or {}
    flagged = set(gate.get("flagged_ids") or [])
    for path in sorted((store.dir / "followup").glob("round-*/followup_matrix.json")):
        flagged |= set(((read_json(path, {}) or {}).get("gate") or {}).get("flagged_ids") or [])
    pack = {e["key"] for e in (store.load_metadata().get("evaluation") or {}).get("evidence") or []}
    serious = {oid for oid, sevs in verdicts.items() if any(SEVERITY_RANK.get(s, -1) >= SEVERITY_RANK["MAJOR_FIXABLE"]
                                                            for s in sevs)}
    fatal = {oid for oid, sevs in verdicts.items() if "FATAL" in sevs}
    prior_serious = sorted(oid for oid in serious if cited.get(oid))
    evidence_status: dict[str, int] = {}
    for aid, a in state["agents"].items():
        if a["role"] == "novelty":
            ev = read_json(store.sidecar_path(aid, "novelty", ".evidence.json"), {}) or {}
            for entry in (ev.get("objections") or {}).values():
                if entry.get("decisive"):
                    evidence_status[entry.get("status") or "?"] = evidence_status.get(entry.get("status") or "?", 0) + 1
    results = read_json(store.role_dir("verifier") / "results.json", {}) or {}
    independent: dict[str, int] = {}
    for r in (results.get("requests") or {}).values():
        independent[r.get("status") or "?"] = independent.get(r.get("status") or "?", 0) + 1
    citations = sorted({k for oid in prior_serious for k in cited[oid]})
    rounds = (state.get("followup") or {}).get("rounds") or {}
    last = rounds.get(str(len(rounds))) if rounds else None
    out = {
        "run_id": run_id, "preset": store.load_metadata().get("preset"),
        "outcome": (state.get("last_job") or {}).get("outcome"),
        "review_label": (last or {}).get("outcome"),
        "agents": len([a for a in state["agents"].values() if a["status"] == "complete"]),
        "tokens": {k: total.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                 "cache_read_input_tokens", "output_tokens")},
        "api_equivalent_cost_usd": round(total.get("reported_cost_usd", 0.0), 4),
        "estimated_cost_usd": round(total.get("estimated_cost_usd", 0.0), 4),
        "calls_without_usage": total.get("calls_without_usage", 0),
        "elapsed_seconds": round(_elapsed_seconds(store), 1),
        "agent_seconds": round(total.get("agent_seconds", 0.0), 1),
        "calls": len(usage), "failed_calls": sum(1 for u in usage if u.get("outcome") != "complete"),
        "repair_calls": sum(1 for s in stages if s.endswith("repair")),
        "supplement_calls": sum(1 for s in stages if s.endswith("supplement")),
        "placement_fix_calls": sum(1 for s in stages if s.endswith("placement_fix")),
        "serious_findings": sorted(serious), "fatal_findings": sorted(fatal),
        "prior_work_serious": {oid: {"cites": sorted(cited[oid]), "shown": oid not in flagged}
                               for oid in prior_serious},
        "evidence": {"decisive_objections_by_quote_check": evidence_status,
                     "independent_checks_by_status": independent,
                     "serious_prior_work_verdicts": len(prior_serious),
                     "shown": sum(1 for oid in prior_serious if oid not in flagged),
                     "unverified": sum(1 for oid in prior_serious if oid in flagged)},
        "citations": {"cited_by_serious_prior_work_verdicts": citations,
                      "outside_the_evidence_pack": sorted(k for k in citations if pack and k not in pack)},
    }
    out["ground_truth"] = _against_ground_truth(case, verdicts, cited, flagged, serious, fatal)
    return out


def _against_ground_truth(case: dict, verdicts, cited, flagged, serious, fatal) -> dict:
    """What can be scored without judgment: prior-work issues by the papers they cite. Technical issues and
    incorrect objections need a person (the scoring sheet)."""
    gt = case.get("ground_truth") or {}
    status = case.get("status")
    found, missed = [], []
    for issue in gt.get("serious_issues") or []:
        if issue.get("kind") != "prior_work":
            continue
        keys = {evidence_key(x) for x in issue.get("prior_work") or []} - {None}
        hits = sorted(oid for oid in serious if cited.get(oid, set()) & keys)
        (found if hits else missed).append({"id": issue.get("id"), "objections": hits,
                                            "shown": [o for o in hits if o not in flagged]})
    false_fatal = []
    for item in gt.get("not_fatal") or []:
        keys = {evidence_key(x) for x in item.get("prior_work") or []} - {None}
        hits = sorted(oid for oid in fatal if keys and cited.get(oid, set()) & keys)
        if hits:
            false_fatal.append({"id": item.get("id"), "objections": hits})
    technical = [i.get("id") for i in gt.get("serious_issues") or [] if i.get("kind") == "technical"]
    return {"status": status, "counts_as_evidence": status == "adjudicated",
            "prior_work_found": found, "prior_work_missed": missed, "false_fatal_prior_work": false_fatal,
            "technical_issues_to_score_by_hand": technical}


# ---------------------------------------------------------------- the report


def report(pair_dir: str | Path, scores: str | Path | None = None) -> dict:
    """Measure every run of a pair, fold in a filled scoring sheet if given, and write comparison.json/.md."""
    pair_dir = Path(pair_dir).expanduser().resolve()
    pair = read_json(pair_dir / "pair.json")
    if not pair:
        raise CaseError(f"no pair.json in {pair_dir}")
    _, case = load_case(pair["case_dir"])
    runs = [measure(r["run_id"], case) for r in pair["runs"]]
    sheet = pair_dir / "manual_scores.yaml"
    if not sheet.is_file():
        atomic_write_text(sheet, _scoring_sheet(case, runs))
    manual = yaml.safe_load(Path(scores).read_text(encoding="utf-8")) if scores else None
    if manual:
        for run in runs:
            got = (manual.get("runs") or {}).get(run["run_id"]) or {}
            run["manual"] = {"missed_serious_issues": got.get("missed_serious_issues"),
                             "false_fatal_accusations": got.get("false_fatal_accusations"),
                             "scored_by": manual.get("scored_by")}
    snapshots = {r["run_id"]: _snapshot_digest(r["run_id"]) for r in pair["runs"]}
    result = {"pair": pair, "runs": runs, "identical_evidence": len(set(snapshots.values())) <= 1,
              "snapshots": snapshots, "written_at": utcnow_iso()}
    atomic_write_json(pair_dir / "comparison.json", result)
    atomic_write_text(pair_dir / "comparison.md", comparison_markdown(result))
    return result


def _snapshot_digest(run_id: str) -> str:
    store = RunStore.open(run_id)
    index = read_json(store.prior_dir / "index.json", {}) or {}
    return sha256_text(json.dumps(sorted((k, v.get("sha256")) for k, v in index.items()
                                         if v.get("source") == "evaluation_evidence")))


def _scoring_sheet(case: dict, runs: list[dict]) -> str:
    """A YAML sheet for the adjudicator: which serious findings match a ground-truth issue, which are false."""
    gt = case.get("ground_truth") or {}
    lines = ["# Manual scoring sheet. Fill it in, then run: paper-adversary eval report <pair dir> --scores <this file>",
             "# For each run: list the ground-truth issue IDs it missed and the finding IDs that are false FATAL",
             "# accusations. Ground-truth issues:"]
    for issue in gt.get("serious_issues") or []:
        lines.append(f"#   {issue.get('id')} [{issue.get('kind')}, {issue.get('severity')}]: "
                     f"{' '.join(str(issue.get('description') or '').split())[:150]}")
    lines += ["scored_by: null", "runs:"]
    for run in runs:
        lines += [f"  {run['run_id']}:   # {run['preset']}; serious findings: {', '.join(run['serious_findings']) or 'none'}"
                  f"; FATAL: {', '.join(run['fatal_findings']) or 'none'}",
                  "    missed_serious_issues: []      # ground-truth IDs, e.g. [GT2]",
                  "    false_fatal_accusations: []    # finding IDs, e.g. [R1-O3]"]
    return "\n".join(lines) + "\n"


def comparison_markdown(result: dict) -> str:
    pair, runs = result["pair"], result["runs"]
    status = pair["status"]
    banner = {"adjudicated": "Adjudicated case: these are measurements on one case, not a verdict on the presets.",
              "synthetic": "SYNTHETIC case: it tests the harness only. Nothing here is evidence about quality or cost."
              }.get(status, "Not an adjudicated case: nothing here is evidence about quality.")
    lines = [f"# Evaluation: {pair['case']} ({pair['category']}, {pair['mode']} mode)", "", banner, "",
             "Inputs: identical submission" + (", identical evidence snapshot (search off)" if pair["mode"] == "reasoning"
                                                else " (search on: discovery results may differ between runs)")
             + ("" if result["identical_evidence"] else " — WARNING: the evidence snapshots differ"), ""]
    head = "| | " + " | ".join(f"{r['preset']} ({r['run_id']})" for r in runs) + " |"
    lines += [head, "|---|" + "---|" * len(runs)]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(str(fn(r)) for r in runs) + " |")
    row("Job outcome / review label", lambda r: f"{r['outcome']} / {r['review_label'] or '—'}")
    row("Serious findings (FATAL)", lambda r: f"{len(r['serious_findings'])} ({len(r['fatal_findings'])})")
    row("Prior-work serious verdicts: shown / unverified",
        lambda r: f"{r['evidence']['shown']} / {r['evidence']['unverified']}")
    row("Citations outside the evidence pack", lambda r: ", ".join(r["citations"]["outside_the_evidence_pack"]) or "none")
    row("Ground-truth prior work missed", lambda r: ", ".join(x["id"] for x in r["ground_truth"]["prior_work_missed"])
        or "none")
    row("False FATAL on prior work", lambda r: ", ".join(x["id"] for x in r["ground_truth"]["false_fatal_prior_work"])
        or "none")
    row("Missed serious issues (manual)", lambda r: (r.get("manual") or {}).get("missed_serious_issues", "not scored"))
    row("False FATAL accusations (manual)", lambda r: (r.get("manual") or {}).get("false_fatal_accusations", "not scored"))
    row("Tokens in / cache write / cache read / out",
        lambda r: " / ".join(f"{r['tokens'][k]:,}" for k in ("input_tokens", "cache_creation_input_tokens",
                                                               "cache_read_input_tokens", "output_tokens")))
    row("API-equivalent cost (reported, $)", lambda r: f"{r['api_equivalent_cost_usd']:.2f}"
        + (f" + {r['calls_without_usage']} call(s) without usage" if r["calls_without_usage"] else ""))
    row("Elapsed (wall) / agent time (s)", lambda r: f"{r['elapsed_seconds']:.0f} / {r['agent_seconds']:.0f}")
    row("Calls (failed) / repairs / supplements / placement fixes",
        lambda r: f"{r['calls']} ({r['failed_calls']}) / {r['repair_calls']} / {r['supplement_calls']} / "
                  f"{r['placement_fix_calls']}")
    lines += ["", "The API-equivalent cost is what Claude Code reports per call; on a subscription nothing is billed "
              "per token, and plan usage limits are not dollars. Do not conclude that one preset is better or "
              "cheaper from one case: compare across many adjudicated cases."]
    return "\n".join(lines) + "\n"
