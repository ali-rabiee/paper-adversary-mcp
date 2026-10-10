"""Follow-up rounds: what happens to the completeness critic's findings.

Round r takes the items of its critic (C1 for round 1, then the previous round's re-check critic) through:

  1. triage (deterministic, free): validate every item, route it (adjudicate, revision only, noted, invalid; an
     item disputing a serious verdict on an objection it cites is adjudicated whatever its rating), and attach the
     judges' verdicts on objections it cites;
  2. verification: items naming suspected prior work go to blind verifiers through verification.py, which see
     the submission, the passage at stake (the item's submission_quote, else a ledger claim it cites, else its
     location) and the prior paper, never the critic's prose;
  3. adjudication: independent adjudicators rule on every routed item with the judges' four severities; a
     follow-up matrix and the same evidence gate as the judges' are computed from their rulings;
  4. a revised memo S<r+1> that disposes of every item (the gate quarantines it if one is missing or an
     unverified prior-work item is not filed as an unverified threat), which becomes the current memo;
  5. a fresh re-check critic C<r+1>, whose new items decide whether another round runs.

Items the re-check critic re-raises (it may only do so when the handling was plainly wrong or new evidence
exists) never start a round, since the adjudicators already ruled on them; they stay open. Every re-check item is
recorded in round.json, below the floor included (those are only recorded). A review that stops without new
items is labelled ready_for_next_gate only when nothing serious is open: no re-raised item, no FATAL verdict (or
rubric test ruled FATAL) that adjudicators did not overturn, no unverified FATAL prior-work threat. Otherwise it is
review_saturated_with_open_issues, with the list (open MAJOR BUT FIXABLE issues are listed, counted as issues
rather than IDs, but do not block). At the round cap, new MAJOR or FATAL items that no round adjudicated are open
issues too.

Nothing is overwritten: each round's records live in followup/round-<r>/, and superseded memos stay in place.
"""

from __future__ import annotations

import json
import re

from paper_adversary.config import AgentSpec, PipelineConfig
from paper_adversary.evidence_gate import evaluate as evaluate_evidence
from paper_adversary.gates import BLOCK, ITEM_TYPES, PASS, Check
from paper_adversary.reports import (
    SEVERITY_LABEL,
    SEVERITY_RANK,
    build_judgment_matrix,
    matrix_markdown,
    normalize_severity,
    read_report,
)
from paper_adversary.search.refcheck import title_similarity
from paper_adversary.util import atomic_write_json, atomic_write_text, read_json, sha256_text, utcnow_iso
from paper_adversary.verification import VerificationRequest, origin_hash

READY = "ready_for_next_gate"
SATURATED = "review_saturated_with_open_issues"
REPEAT_SIMILARITY = 0.75
RUBBER_STAMP = 0.9  # an adjudicator agreeing with the critic (or rejecting) on this share of 5+ items is flagged


def round_dir(store, r: int):
    return store.dir / "followup" / f"round-{r}"


def round_critic(state: dict, r: int) -> tuple[str, str]:
    """The critic whose items round r handles: (agent ID, role) — C1 for round 1, then the previous round's
    re-check critic, as recorded when that round was planned."""
    if r == 1:
        return "C1", "critic"
    prev = (((state.get("followup") or {}).get("rounds") or {}).get(str(r - 1)) or {})
    return prev.get("recheck") or f"C{r}", "recheck"


def next_free(state: dict, prefix: str) -> int:
    """The next unused number for agent IDs with this prefix (base agents may already use S2, C2, ...)."""
    used = [int(m.group(1)) for aid in state.get("agents", {}) if (m := re.fullmatch(rf"{prefix}(\d+)", aid))]
    return max(used, default=0) + 1


def current_memo(state: dict) -> str:
    return (state.get("followup") or {}).get("current_memo") or "S1"


def _id_list(value) -> list[str]:
    """IDs from a list, or from a string an agent wrote instead of a list ("N1-O1, N2-O3")."""
    if isinstance(value, str):
        value = re.split(r"[,;\s]+", value)
    if not isinstance(value, list):
        return []
    return [re.sub(r"\s+", "", str(x)).upper() for x in value if x and isinstance(x, (str, int))]


_LEDGER_ID = re.compile(r"(?<![\w-])([LC]\d+)(?![\w-])", re.I)


def _claim_ids(value) -> list[str]:
    """Claims-ledger IDs (L3, or C3 as older ledgers and prompts write them) from claim_ids as a critic wrote them:
    a list or a string, an entry possibly in words ("ledger claim L3", as critic_v2 asks). Read before any
    whitespace is removed, which would make 'ledger claim L3' one token."""
    entries = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    return list(dict.fromkeys(m.upper() for x in entries if isinstance(x, (str, int))
                              for m in _LEDGER_ID.findall(str(x))))


def critic_items(data: dict | None, agent_id: str) -> list[dict]:
    """Items from a critic's structured block, with run-unique IDs such as 'C1-I3'."""
    out = []
    items = (data or {}).get("items")
    for i, item in enumerate(items if isinstance(items, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        local = str(item.get("id") or f"I{i}").strip().upper().split("-")[-1]
        refs = item.get("candidate_references")
        out.append({**item, "id": f"{agent_id}-{local}", "source": agent_id,
                    "type": str(item.get("type") or "").strip().lower(),
                    "objection_ids": _id_list(item.get("objection_ids")),
                    "judge_ids": _id_list(item.get("judge_ids")), "claim_ids": _claim_ids(item.get("claim_ids")),
                    "candidate_references": [r for r in refs if isinstance(r, dict)] if isinstance(refs, list) else [],
                    "severity": normalize_severity(item.get("severity_estimate"))})
    return out


def _rank(severity: str | None) -> int:
    return SEVERITY_RANK.get(severity or "", -1)


# ---------------------------------------------------------------- 1. triage


DISPUTE_TYPES = ("judging_flaw", "minority_critique")


def _disputed(item: dict, rows: dict[str, dict]) -> list[str]:
    """The cited objections (or rubric tests, whose rulings count like judgments) whose computed verdicts an item
    disputes. Those with a FATAL or MAJOR BUT FIXABLE verdict feed the open issues and the evidence gate: a
    judging_flaw item disputes any such verdict it cites (its severity or its evidence status), a
    minority_critique item one whose highest verdict differs from the item's own severity estimate."""
    if item.get("type") not in DISPUTE_TYPES:
        return []
    out = []
    for oid in item.get("objection_ids") or []:
        sevs = [v.get("severity") for v in ((rows.get(oid) or {}).get("verdicts") or {}).values()]
        top = max((_rank(s) for s in sevs), default=-1)
        if top >= _rank("MAJOR_FIXABLE") and (item.get("type") == "judging_flaw" or top != _rank(item.get("severity"))):
            out.append(oid)
    return out


def triage(store, cfg: PipelineConfig, state: dict, r: int) -> dict:
    """Validate and route the round's items; writes followup/round-<r>/items.json (once) and items.md. Items at
    or above min_severity are adjudicated, and so is an item that disputes a serious computed verdict on an
    objection or rubric test it cites (_disputed), whatever its own rating, so that the adjudicators' re-ratings
    can correct it."""
    path = round_dir(store, r) / "items.json"
    existing = read_json(path)
    if existing:
        return existing
    critic_id, role = round_critic(state, r)
    side = read_json(store.sidecar_path(critic_id, role, ".json"), {}) or {}
    _, body = read_report(store.report_path(critic_id, role))
    matrix = read_json(store.role_dir("judge") / "judgment_matrix.json", {}) or {}
    rows = {row["id"]: row for row in matrix.get("rows") or []}
    tests = rubric_test_rows(matrix)  # rubric tests may be cited, and disputed, like objections
    ruled = {**rows, **tests}
    judges = set(matrix.get("judges") or [])
    floor = _rank(cfg.followup.min_severity)
    items = []
    for item in critic_items(side.get("data"), critic_id):
        problems = []
        if item.get("type") not in ITEM_TYPES:
            problems.append(f"unknown type {item.get('type')!r}")
        unknown = [o for o in item.get("objection_ids") or [] if o not in ruled]
        if unknown:
            problems.append(f"cites objection IDs that do not exist: {', '.join(unknown)}")
        bad_judges = [j for j in item.get("judge_ids") or [] if j not in judges]
        if bad_judges:
            problems.append(f"cites judges that do not exist: {', '.join(bad_judges)}")
        refs = [ref for ref in item.get("candidate_references") or [] if isinstance(ref, dict)
                and (ref.get("title") or ref.get("doi") or ref.get("arxiv_id"))]
        disputed = [] if problems else _disputed(item, ruled)
        reason = None
        if problems:
            route = "invalid"
        elif item.get("type") == "synthesis_flaw":
            route = "revision"
        elif _rank(item.get("severity")) >= floor:
            route = "adjudicate"
        elif disputed:
            route, reason = "adjudicate", f"disputes the computed verdicts on {', '.join(disputed)}"
        else:
            route = "noted"
        cited = {o: {j: v.get("severity") for j, v in ruled[o].get("verdicts", {}).items()}
                 for o in item.get("objection_ids") or [] if o in ruled}
        hints = sorted(((title_similarity(item.get("title") or "", row.get("title") or ""), oid)
                        for oid, row in rows.items()), reverse=True)
        items.append({**item, "route": route, "route_reason": reason, "problems": problems,
                      "verify": bool(refs) and route != "invalid", "references": refs, "cited_verdicts": cited,
                      "cited_tests": [o for o in item.get("objection_ids") or [] if o in tests],
                      "possibly_covered_by": [f"{oid} ({s:.2f})" for s, oid in hints[:2] if s >= 0.6]})
    record = {"round": r, "critic": critic_id, "critic_sha256": sha256_text(body), "created_at": utcnow_iso(),
              "items": items}
    atomic_write_json(path, record)
    atomic_write_text(round_dir(store, r) / "items.md", items_markdown(record))
    return record


def rubric_test_rows(matrix: dict) -> dict[str, dict]:
    """The judgment matrix's rubric tests as rows like an objection's: each judge's ruling that the test applies,
    with its severity (a test ruled not to apply has no verdict)."""
    return {t["id"]: {"id": t["id"], "rubric_test": True, "verdicts": {
        j: {"severity": r.get("severity")} for j, r in (t.get("rulings") or {}).items()
        if r.get("applies") in ("yes", "partly") and r.get("severity")}} for t in matrix.get("rubric_tests") or []}


def items_markdown(record: dict) -> str:
    lines = [f"Completeness-critique items of {record['critic']} (follow-up round {record['round']}), as triaged by "
             "the orchestrator. Route 'adjudicate' items need a ruling; 'revision' items are memo problems for the "
             "revised memo; 'noted' items are below the adjudication threshold; 'invalid' items are malformed. An "
             "item that disputes a FATAL or MAJOR BUT FIXABLE verdict on an objection it cites is adjudicated "
             "whatever its own rating; re-rate those objections in objection_updates."]
    for it in record["items"]:
        lines.append(f"- {it['id']} [{it.get('type')}; critic: {it.get('severity') or '?'}; route: {it['route']}] "
                     f"{it.get('title') or ''} — {it.get('location') or 'no location'}")
        if it.get("route_reason"):
            lines.append(f"  routed to adjudication because it {it['route_reason']}")
        if it.get("argument"):
            lines.append(f"  {' '.join(str(it['argument']).split())[:1200]}")
        for oid, verdicts in it.get("cited_verdicts", {}).items():
            test = oid in (it.get("cited_tests") or [])
            lines.append(f"  cites {oid}{' (rubric test)' if test else ''}: " + (
                ", ".join(f"{j} {SEVERITY_LABEL.get(s, s)}" for j, s in sorted(verdicts.items()))
                or ("no judge ruled that it applies" if test else "no judge classified it")))
        if it.get("references"):
            lines.append("  suspected prior work: " + "; ".join(
                f"{ref.get('title') or ref.get('arxiv_id') or ref.get('doi')}" for ref in it["references"]))
        if it.get("possibly_covered_by"):
            lines.append(f"  possibly covered by: {', '.join(it['possibly_covered_by'])}")
        if it.get("problems"):
            lines.append(f"  problems: {'; '.join(it['problems'])}")
    return "\n".join(lines) + "\n"


def _item_core(it: dict) -> dict:
    """The fields that identify a critic item's content (for verification.origin_hash). The passage-locating
    fields (submission_quote, and with it the cited ledger claims) count only for an item that gives a
    submission_quote (critic_v3, critic_recheck_v2), so items written before submission_quote existed, which also
    cite claim_ids, keep their hashes and their earlier verdicts."""
    core = {k: it.get(k) for k in ("id", "title", "type", "location", "candidate_references", "severity_estimate")}
    if it.get("submission_quote"):
        core.update({k: it[k] for k in ("submission_quote", "claim_ids") if it.get(k)})
    return core


def routable(record: dict) -> list[dict]:
    return [it for it in record["items"] if it["route"] in ("adjudicate", "revision") or it["verify"]]


def verification_requests(record: dict, claims: dict[str, str] | None = None) -> list[VerificationRequest]:
    """Blind-verification requests for the round's items that name suspected prior work. The passage at stake is
    the item's submission_quote (the submission's own words, critic_v3); if that is missing or not found
    verbatim, the first claims-ledger claim it cites (`claims`: ledger ID -> claim) that is found; then its
    location. plan_batch records which one located it."""
    out = []
    for it in record["items"]:
        if not it["verify"]:
            continue
        quote = str(it.get("submission_quote") or "").strip() or None
        # ledger IDs are L<n>; C<n> is what ledgers showed before (and what intake_v1 writes): both are matched,
        # also as items triaged before critic_items kept only the IDs stored them ("LEDGERCLAIMC2")
        cited = [m.group(1) for cid in it.get("claim_ids") or []
                 if (m := re.search(r"(?:^|LEDGERCLAIM|CLAIM)([LC]\d+)$", re.sub(r"\s+", "", str(cid)).upper()))]
        ledger = tuple((f"ledger claim {cid}", (claims or {})[cid]) for cid in dict.fromkeys(cited)
                       if (claims or {}).get(cid))
        for k, ref in enumerate(it["references"][:3], start=1):
            ident = str(ref.get("arxiv_id") or ref.get("doi") or ref.get("title") or "").strip()
            out.append(VerificationRequest(
                request_id=f"{it['id']}:r{k}", origin="critic", origin_ids=(it["id"],),
                prior={"identifier": ident, "title": ref.get("title"), "doi": ref.get("doi"),
                       "arxiv_id": ref.get("arxiv_id"), "year": ref.get("year")},
                claim_quote=quote, quote_label="submission_quote", alt_quotes=ledger,
                claim_location=it.get("location"), note=it.get("argument"),
                origin_hash=origin_hash(_item_core(it)), author=record.get("critic"), severity=it.get("severity"),
                confidence=it.get("confidence")))
    return out


# ---------------------------------------------------------------- 2. planning


def plan_agents(cfg: PipelineConfig, registry, state: dict, r: int) -> list[AgentSpec]:
    """Adjudicators A<n> (numbered across rounds), the revised memo S<r+1> and the re-check critic C<r+1>."""
    fc = cfg.followup
    first = next_free(state, "A")
    specs = []

    def spec(aid: str, role: str, index: int) -> AgentSpec:
        rc = cfg.role(role)
        alias, effort, _ = rc.for_agent(index)  # adjudicator k of the round: its own model (mixed panels)
        return AgentSpec(agent_id=aid, role=role, index=index, model_alias=alias,
                         model_id=registry.resolve(alias).id, effort=effort, prompt_name=rc.prompt,
                         tools=[], paper_format=rc.paper_format, timeout_s=rc.timeout_minutes * 60,
                         max_turns=rc.max_turns, round=r)

    for k in range(fc.adjudicators):
        specs.append(spec(f"A{first + k}", "adjudicator", k + 1))
    specs.append(spec(f"S{next_free(state, 'S')}", "revision", 1))
    specs.append(spec(f"C{next_free(state, 'C')}", "recheck", 1))
    return specs


# ---------------------------------------------------------------- 3. the follow-up matrix


def build_followup_matrix(store, cfg: PipelineConfig, state: dict, r: int) -> dict:
    """Items against the adjudicators' rulings, never averaged, plus the evidence gate for prior-work items."""
    record = read_json(round_dir(store, r) / "items.json", {}) or {}
    items = [it for it in record.get("items") or [] if it["route"] == "adjudicate"]
    as_objections = {record.get("critic", "C?"): [
        {"id": it["id"], "source": it["source"], "title": it.get("title"), "severity_estimate": it.get("severity"),
         "category": "already_done" if it["verify"] else None, "references": it.get("references")}
        for it in items]}
    rulings: dict[str, list[dict]] = {}
    for aid, a in sorted(state["agents"].items()):
        if a["role"] == "adjudicator" and int(a.get("round") or 0) == r and a["status"] == "complete":
            data = (read_json(store.sidecar_path(aid, "adjudicator", ".json"), {}) or {}).get("data") or {}
            rulings[aid] = [{**x, "objection_ids": [str(i).upper() for i in x.get("item_ids") or []],
                             "severity": normalize_severity(x.get("severity")),
                             "relies_on_prior_work": None}
                            for x in data.get("rulings") or [] if isinstance(x, dict)]
    matrix = build_judgment_matrix(as_objections, rulings)
    known = {o["id"]: o for objs in as_objections.values() for o in objs}
    verification = read_json(store.role_dir("verifier") / "results.json", {}) or {}
    hashes = {it["id"]: origin_hash(_item_core(it)) for it in items}
    gate = evaluate_evidence(rulings, known, {}, verification, True, hashes)
    matrix["gate"] = {"flagged_ids": gate["flagged_ids"], "verdicts": gate["verdicts"]}
    matrix["rubber_stamp"] = _rubber_stamps(items, rulings)
    re_rulings = [{"adjudicator": aid, "item_ids": x["objection_ids"], **u} for aid, rs in rulings.items() for x in rs
                  for u in x.get("objection_updates") or [] if isinstance(u, dict)]
    matrix["objection_updates"] = re_rulings
    # what each adjudicator said an item repeats (for counting open issues, see _item_targets)
    matrix["item_links"] = {iid: {aid: {"is_new": str(x.get("is_new") or "").strip().lower(),
                                        "covered_by": _id_list(x.get("already_covered_by")),
                                        "rerated": _id_list([u.get("objection_id") for u in x.get("objection_updates")
                                                             or [] if isinstance(u, dict)])}
                                  for aid, rs in rulings.items() for x in rs if iid in x["objection_ids"]}
                            for iid in (it["id"] for it in items)}
    atomic_write_json(round_dir(store, r) / "followup_matrix.json", matrix)
    text = matrix_markdown(matrix, title=f"Follow-up matrix (round {r})", followup=True)
    text += "\nFollow-up gate: not shown: " + (", ".join(gate["flagged_ids"]) or "none") + "\n"
    for v in gate["verdicts"]:
        if not v["passes"]:
            text += f"- {v['judge']} {SEVERITY_LABEL.get(v['severity'], v['severity'])} on {', '.join(v['objection_ids'])}: {v['label']}\n"
    for stamp in matrix["rubber_stamp"]:
        text += f"Possible rubber-stamping: {stamp}\n"
    if re_rulings:
        text += "Base objections re-rated by adjudicators: " + "; ".join(
            f"{u['adjudicator']} {u.get('objection_id')} -> {u.get('severity')}" for u in re_rulings) + "\n"
        everyone = sorted(rulings)
        by: dict[str, dict[str, str]] = {}
        for u in re_rulings:
            oid, sev = str(u.get("objection_id") or "").strip().upper(), normalize_severity(u.get("severity"))
            if oid and sev:
                by.setdefault(oid, {})[u["adjudicator"]] = sev
        unanimous = {oid: v for oid, v in by.items() if everyone and set(v) >= set(everyone)}
        if unanimous:
            text += ("Re-rated by every adjudicator of this round (in the open issues, their ratings replace the "
                     "judges' more severe verdicts): " + "; ".join(
                         f"{oid} -> {'/'.join(SEVERITY_LABEL.get(s, s) for s in v.values())}"
                         for oid, v in sorted(unanimous.items())) + "\n")
    atomic_write_text(round_dir(store, r) / "followup_matrix.md", text)
    return matrix


def _rubber_stamps(items: list[dict], rulings: dict[str, list[dict]]) -> list[str]:
    critic = {it["id"]: it.get("severity") for it in items}
    out = []
    for aid, rs in rulings.items():
        pairs = [(critic.get(i), x.get("severity")) for x in rs for i in x["objection_ids"] if i in critic]
        if len(pairs) < 5:
            continue
        same = sum(1 for c, s in pairs if c == s) / len(pairs)
        rejected = sum(1 for _, s in pairs if s == "NOT_CONVINCING") / len(pairs)
        if same >= RUBBER_STAMP:
            out.append(f"{aid} matched the critic's severity on {same:.0%} of {len(pairs)} items")
        if rejected >= RUBBER_STAMP:
            out.append(f"{aid} rejected {rejected:.0%} of {len(pairs)} items")
    return out


# ---------------------------------------------------------------- 4. the revised memo's dispositions


def check_dispositions(store, state: dict, r: int, data: dict | None) -> list[Check]:
    """Blocking checks for a revised memo: every item disposed of; unverified prior-work items filed as such."""
    record = read_json(round_dir(store, r) / "items.json", {}) or {}
    matrix = read_json(round_dir(store, r) / "followup_matrix.json", {}) or {}
    flagged = set((matrix.get("gate") or {}).get("flagged_ids") or [])
    disp = {str(d.get("item_id") or "").upper(): d.get("disposition")
            for d in (data or {}).get("item_dispositions") or [] if isinstance(d, dict)}
    expected = [it["id"] for it in record.get("items") or [] if it["route"] != "invalid"]
    missing = [i for i in expected if i not in disp]
    misfiled = sorted(i for i in flagged if disp.get(i) not in (None, "unverified_threat"))
    checks = [Check("dispositions", BLOCK if missing else PASS, "quality",
                    f"items without a disposition: {', '.join(missing)}" if missing else "",
                    {"missing": missing, "expected": len(expected)}),
              Check("unverified_items", BLOCK if misfiled else PASS, "quality",
                    f"unverified prior-work items not filed as unverified threats: {', '.join(misfiled)}"
                    if misfiled else "", {"misfiled": misfiled})]
    return checks


# ---------------------------------------------------------------- 5. the stop rule


def stop_rule(store, cfg: PipelineConfig, state: dict, r: int) -> dict:
    """After the re-check critic: another_round or max_rounds_reached while it raises new items at or above the
    floor; otherwise ready_for_next_gate or review_saturated_with_open_issues (see the module docstring). Every
    re-check item is recorded in round.json: new, re-raised, below the floor, or of an unknown type. New items (at
    or above the floor) rated MAJOR BUT FIXABLE or FATAL that no round adjudicated, i.e. at the round cap, are open
    issues that block 'ready'; items below the floor never block, as the base critic's noted items do not."""
    critic_id = (((state.get("followup") or {}).get("rounds") or {}).get(str(r)) or {}).get("recheck") or f"C{r + 1}"
    side = read_json(store.sidecar_path(critic_id, "recheck", ".json"), {}) or {}
    earlier = []  # only items that were actually adjudicated or disposed of count as already handled
    for k in range(1, r + 1):
        earlier += [{**it, "round": k} for it in (read_json(round_dir(store, k) / "items.json", {}) or {}).get("items")
                    or [] if it.get("route") in ("adjudicate", "revision")]
    floor = _rank(cfg.followup.min_severity)
    new, repeats, below, invalid = [], [], [], []
    for it in critic_items(side.get("data"), critic_id):
        rec = {"id": it["id"], "type": it.get("type"), "severity": it.get("severity"), "title": it.get("title"),
               "references": [str(ref.get("title") or ref.get("arxiv_id") or ref.get("doi"))
                              for ref in it.get("candidate_references") or []]}
        if it.get("type") not in ITEM_TYPES:
            invalid.append(rec)
            continue
        if _rank(it.get("severity")) < floor:
            below.append({**rec, "repeats_item": str(it.get("repeats_item") or "").strip().upper() or None})
            continue
        twin = _repeat_of(it, earlier)
        (repeats if twin else new).append({**rec, "repeats": twin})
    serious = [x for x in new if _rank(x["severity"]) >= _rank("MAJOR_FIXABLE")]
    issues = open_issues(store, state, _disputes(store, state, repeats, earlier), serious)
    if new:
        outcome = "max_rounds_reached" if r >= cfg.followup.max_rounds else "another_round"
    else:
        outcome = SATURATED if issues["blocking"] else READY
    result = {"round": r, "recheck": critic_id, "outcome": outcome, "min_severity": cfg.followup.min_severity,
              "new_items": new, "disputed_repeats": repeats, "below_floor": below, "invalid_items": invalid,
              "open_issues": issues, "decided_at": utcnow_iso()}
    atomic_write_json(round_dir(store, r) / "round.json", result)
    return result


def _disputes(store, state: dict, repeats: list[dict], earlier: list[dict]) -> list[dict]:
    """Re-raised items with what happened to the item they repeat: the adjudicators' rulings and the memo's
    disposition."""
    by_id = {it["id"]: it for it in earlier}
    rounds = (state.get("followup") or {}).get("rounds") or {}
    out = []
    for rep in repeats:
        old = by_id.get(rep["repeats"]) or {}
        k = old.get("round")
        rows = (read_json(round_dir(store, k) / "followup_matrix.json", {}) or {}).get("rows") or [] if k else []
        row = next((x for x in rows if x.get("id") == old.get("id")), {})
        revision = (rounds.get(str(k)) or {}).get("revision") if k else None
        disp = ((read_json(store.sidecar_path(revision, "revision", ".json"), {}) or {}).get("data") or {}) \
            if revision else {}
        disposition = next((d.get("disposition") for d in disp.get("item_dispositions") or []
                            if isinstance(d, dict) and str(d.get("item_id") or "").upper() == old.get("id")), None)
        out.append({**rep, "earlier_rulings": {a: v.get("severity") for a, v in (row.get("verdicts") or {}).items()},
                    "earlier_disposition": disposition})
    return out


def open_issues(store, state: dict, disputes: list[dict] | None = None,
                unadjudicated: list[dict] | None = None) -> dict:
    """What keeps a finished review from being ready for the next gate ("blocking": re-raised items, new MAJOR or
    FATAL items no round adjudicated (`unadjudicated`), FATAL verdicts that adjudicators did not overturn,
    unverified FATAL prior-work threats) and what is open but fixable ("major": MAJOR BUT FIXABLE verdicts, listed
    only). Verdicts come from the judges' matrix (each objection at its own severity, and the rubric tests ruled to
    apply), the adjudicators' rulings on follow-up items, and their re-ratings of base objections: the latest round
    that re-rated an objection decides, and when every one of its adjudicators re-rated it, the verdicts more
    severe than their highest rating no longer count (`overturned`).

    Entries stay per ID; `issues` counts issues: IDs that every judge ruled on in one and the same judgment are
    one issue, and so is a follow-up item with the one base objection that every adjudicator said it repeats
    (_item_targets); an issue never mixes IDs the evidence gates flagged with IDs they did not. An entry is
    `grouped_only` when every vote for its severity comes from a grouped judgment that gave it no severity of its
    own while every judge that ruled on it on its own rated it lower."""
    rounds = (state.get("followup") or {}).get("rounds") or {}
    flagged = set((read_json(store.role_dir("judge") / "evidence_gate.json", {}) or {}).get("flagged_ids") or [])
    verdicts: dict[str, dict[str, str]] = {}  # objection, item or rubric-test ID -> {judge or adjudicator: severity}
    grouped: dict[str, set[str]] = {}  # ID -> judges whose verdict is only a group's severity
    keys: dict[str, tuple] = {}  # objection ID -> the judgment each judge put it in
    titles: dict[str, str] = {}
    tests: set[str] = set()
    matrix = read_json(store.role_dir("judge") / "judgment_matrix.json", {}) or {}
    for row in matrix.get("rows") or []:
        verdicts[row["id"]] = {j: v.get("severity") for j, v in (row.get("verdicts") or {}).items()}
        grouped[row["id"]] = {j for j, v in (row.get("verdicts") or {}).items() if v.get("grouped")}
        keys[row["id"]] = tuple(sorted((j, v.get("group") or row["id"]) for j, v in (row.get("verdicts") or {}).items()))
        titles[row["id"]] = row.get("title") or ""
    for t in matrix.get("rubric_tests") or []:  # a declared rubric test that judges ruled to apply
        verdicts[t["id"]] = {j: r.get("severity") for j, r in (t.get("rulings") or {}).items()
                             if r.get("applies") in ("yes", "partly") and r.get("severity")}
        titles[t["id"]], tests = f"rubric test: {t.get('text') or ''}", tests | {t["id"]}
    rerated: dict[str, tuple[int, dict[str, str]]] = {}
    targets: dict[str, str] = {}  # follow-up item -> the base objection it repeats
    for k in sorted(int(x) for x in rounds):
        fm = read_json(round_dir(store, k) / "followup_matrix.json", {}) or {}
        flagged |= set((fm.get("gate") or {}).get("flagged_ids") or [])
        for row in fm.get("rows") or []:
            verdicts[row["id"]] = {a: v.get("severity") for a, v in (row.get("verdicts") or {}).items()}
            titles[row["id"]] = row.get("title") or ""
        for u in fm.get("objection_updates") or []:
            oid, sev = str(u.get("objection_id") or "").strip().upper(), normalize_severity(u.get("severity"))
            if oid and sev and u.get("adjudicator"):
                if rerated.get(oid, (0, {}))[0] < k:
                    rerated[oid] = (k, {})
                rerated[oid][1][u["adjudicator"]] = sev
        targets.update(_item_targets(fm, _panel(state, rounds, k)))
    overturned: dict[str, dict] = {}
    for oid, (k, by) in rerated.items():
        panel = _panel(state, rounds, k)
        current = dict(verdicts.get(oid) or {})
        if panel and all(by.get(a) for a in panel):  # every adjudicator re-rated it: the more severe verdicts go
            top = max(_rank(s) for s in by.values())
            gone = {j: s for j, s in current.items() if _rank(s) > top}
            if gone:
                overturned[oid] = {"round": k, "by": sorted(by), "verdicts": gone}
            current = {j: s for j, s in current.items() if _rank(s) <= top}
        current.update(by)
        verdicts[oid] = current

    def grouped_only(oid: str, voters: list[str]) -> bool:
        own = [s for j, s in verdicts[oid].items() if j not in grouped.get(oid, set())]
        return bool(voters) and all(j in grouped.get(oid, set()) for j in voters) and bool(own) \
            and all(_rank(s) < _rank(verdicts[oid][voters[0]]) for s in own)

    fatal, unverified, major = [], [], []
    for oid in sorted(verdicts):
        by = sorted(j for j, sev in verdicts[oid].items() if sev == "FATAL")
        entry = {"id": oid, "title": titles.get(oid, "")[:120], "by": by}
        if oid in tests:
            entry["kind"] = "rubric_test"
        if by:
            (unverified if oid in flagged else fatal).append({**entry, "grouped_only": grouped_only(oid, by)})
        else:
            majors = sorted(j for j, sev in verdicts[oid].items() if sev == "MAJOR_FIXABLE")
            if majors:
                major.append({**entry, "by": majors, "unverified": oid in flagged,
                              "grouped_only": grouped_only(oid, majors)})
    blocking = {"reraised": list(disputes or []), "unadjudicated": list(unadjudicated or []), "fatal": fatal,
                "unverified_fatal": unverified}
    return {"blocking": {k: v for k, v in blocking.items() if v}, "major": major, "issues": {"fatal": _issues(fatal + unverified, keys, targets, flagged),
                                       "major": _issues(major, keys, targets, flagged)},
            "overturned": overturned}


def _panel(state: dict, rounds: dict, k: int) -> list[str]:
    """The adjudicators of round k that completed."""
    return [a for a in (rounds.get(str(k)) or {}).get("adjudicators") or []
            if (state["agents"].get(a) or {}).get("status") == "complete"]


def _item_targets(fm: dict, panel: list[str]) -> dict[str, str]:
    """Follow-up items that repeat one base objection: every adjudicator of the round ruled it not (or only
    partly) new, and the one objection all of them name, by citing it as the only one the item repeats
    (already_covered_by) or by re-rating it for the item, is the same."""
    out = {}
    for iid, by in (fm.get("item_links") or {}).items():
        if not panel or any(a not in by or by[a].get("is_new") not in ("no", "partially", "partly") for a in panel):
            continue
        named = [set(by[a].get("rerated") or []) | (set(by[a]["covered_by"]) if len(by[a].get("covered_by") or []) == 1
                                                    else set()) for a in panel]
        common = set.intersection(*named) if named else set()
        if len(common) == 1:
            out[iid] = common.pop()
    return out


def _issues(entries: list[dict], keys: dict[str, tuple], targets: dict[str, str], flagged: set[str]) -> list[dict]:
    """Open entries grouped into issues (see open_issues); each entry gets the issue's first ID as `issue`."""
    listed = {e["id"] for e in entries}
    groups: dict[tuple, list[dict]] = {}
    for e in entries:
        oid = e["id"]
        base = targets.get(oid) if targets.get(oid) in listed and (targets[oid] in flagged) == (oid in flagged) \
            else oid
        key = ("judgment", keys[base], base in flagged) if keys.get(base) else ("own", base)
        groups.setdefault(key, []).append(e)
    out = []
    for members in sorted(groups.values(), key=lambda m: min(x["id"] for x in m)):
        ids = sorted(x["id"] for x in members)
        for x in members:
            x["issue"] = ids[0]
        out.append({"ids": ids, "unverified": all(x["id"] in flagged for x in members),
                    "grouped_only": all(x.get("grouped_only") for x in members)})
    return out


def _sev_label(severity: str | None) -> str:
    return SEVERITY_LABEL.get(severity or "", severity or "?")


def open_issue_lines(issues: dict) -> list[str]:
    """Short status lines for open issues: issues and IDs counted apart, verdicts that are serious only through
    grouping listed separately."""
    b = issues.get("blocking") or {}
    lines = []
    if b.get("reraised"):
        lines.append("re-raised by the re-check critic: " + ", ".join(
            f"{d['id']} (repeats {d['repeats']}; earlier ruled "
            + (", ".join(f"{a} {SEVERITY_LABEL.get(s, s)}" for a, s in (d.get("earlier_rulings") or {}).items())
               or "—") + f", {d.get('earlier_disposition') or 'no disposition'})" for d in b["reraised"]))
    if b.get("unadjudicated"):
        lines.append("new items of the re-check critic that no round adjudicated: " + ", ".join(
            f"{x['id']} ({_sev_label(x.get('severity'))})" for x in b["unadjudicated"]))
    if b.get("fatal"):
        lines.append("FATAL verdicts standing: " + ", ".join(
            f"{x['id']} ({'/'.join(x['by'])}{'; rubric test' if x.get('kind') == 'rubric_test' else ''}"
            f"{'; only through grouped verdicts' if x.get('grouped_only') else ''})" for x in b["fatal"]))
    if b.get("unverified_fatal"):
        lines.append("unverified FATAL prior-work threats: " + ", ".join(x["id"] for x in b["unverified_fatal"]))
    major = issues.get("major") or []
    groups = (issues.get("issues") or {}).get("major")
    if major and groups is None:  # recorded before issues were counted
        lines.append(f"open MAJOR BUT FIXABLE (not blocking): {len(major)} IDs")
    elif major:
        main = [g for g in groups if not g["grouped_only"]]
        only = [i for g in groups if g["grouped_only"] for i in g["ids"]]
        unverified = [i for g in main if g["unverified"] for i in g["ids"]]
        if main:
            lines.append(f"open MAJOR BUT FIXABLE (not blocking): {len(main)} issue(s) over "
                         f"{sum(len(g['ids']) for g in main)} ID(s)"
                         + (f"; unverified prior work among them: {', '.join(unverified)}" if unverified else ""))
        if only:
            lines.append("MAJOR BUT FIXABLE only through grouped verdicts, which judges ruling on them alone rated "
                         f"lower (not counted above): {', '.join(only)}")
    return lines


def _repeat_of(item: dict, earlier: list[dict]) -> str | None:
    claimed = str(item.get("repeats_item") or "").strip().upper()
    if claimed and any(old["id"] == claimed for old in earlier):  # only a real, handled item counts
        return claimed
    for old in earlier:
        if old.get("type") != item.get("type"):
            continue
        if title_similarity(item.get("title") or "", old.get("title") or "") >= REPEAT_SIMILARITY:
            return old["id"]
        ids = set(item.get("objection_ids") or [])
        if ids and ids == set(old.get("objection_ids") or []):
            return old["id"]
    return None


ROUND_ROLES = {"verifier": "blind verifier", "adjudicator": "adjudicator", "revision": "revised memo",
               "recheck": "re-check critic"}


def round_agents(state: dict, r: int) -> list[str]:
    """Round r's agents in order: its verifiers (batch critic<r>), adjudicators, revised memo and re-check critic."""
    rnd = ((state.get("followup") or {}).get("rounds") or {}).get(str(r)) or {}
    batch = (((state.get("verification") or {}).get("batches") or {}).get(f"critic{r}") or {})
    ids = list(batch.get("agents") or []) + list(rnd.get("adjudicators") or []) + [rnd.get("revision"),
                                                                                  rnd.get("recheck")]
    return [a for a in dict.fromkeys(ids) if a and a in state.get("agents", {})]


def round_summary(store, state: dict, describe=None) -> list[str]:
    """Status lines for the follow-up rounds; with `describe` (agent ID -> status text), one line per agent of
    each round, and the follow-up agents no current round lists (superseded by a restart, or orphaned)."""
    fu = state.get("followup") or {}
    agents = state.get("agents", {})
    lines, listed = [], set()
    for r, rnd in sorted((fu.get("rounds") or {}).items(), key=lambda kv: int(kv[0])):
        items = (read_json(round_dir(store, int(r)) / "items.json", {}) or {}).get("items") or []
        routes: dict[str, int] = {}
        for it in items:
            routes[it["route"]] = routes.get(it["route"], 0) + 1
        bits = ", ".join(f"{v} {k}" for k, v in sorted(routes.items())) or "no items"
        lines.append(f"Follow-up round {r}: {rnd.get('status')} — items of {rnd.get('critic')}: {bits}"
                     + (f"; outcome {rnd['outcome']}" if rnd.get("outcome") else ""))
        if describe is not None:
            for aid in round_agents(state, int(r)):
                listed.add(aid)
                lines.append(f"  {aid:<7}{ROUND_ROLES.get(agents[aid]['role'], agents[aid]['role'])}: "
                             f"{describe(aid)}")
        record = read_json(round_dir(store, int(r)) / "round.json", {}) or {}
        if record.get("recheck") and "below_floor" in record:
            lines.append("  " + recheck_line(record))
        if rnd.get("outcome") in (READY, SATURATED, "max_rounds_reached"):
            issues = record.get("open_issues") or rnd.get("open_issues") or {}
            lines += [f"  Open: {line}" for line in open_issue_lines(issues)]
    if describe is not None:
        rest = sorted((aid for aid, a in agents.items() if aid not in listed and int(a.get("round") or 0)
                       and (a["role"] in ROUND_ROLES)), key=lambda x: (len(x), x))
        if rest:
            lines.append("Earlier follow-up agents (no current round lists them): " + ", ".join(
                f"{aid} ({ROUND_ROLES[agents[aid]['role']]}, {agents[aid]['status']})" for aid in rest))
    if fu.get("current_memo") and fu["current_memo"] != "S1":
        lines.append(f"Current memo: {fu['current_memo']} (revised; earlier memos are kept)")
    return lines


def recheck_line(record: dict) -> str:
    """Every item of a round's re-check critic, as round.json records it."""
    floor = _sev_label(record.get("min_severity"))
    new, below = record.get("new_items") or [], record.get("below_floor") or []
    repeats, invalid = record.get("disputed_repeats") or [], record.get("invalid_items") or []
    n = len(new) + len(below) + len(repeats) + len(invalid)
    bits = [f"{len(new)} new at or above {floor}" + (f" ({', '.join(x['id'] for x in new)})" if new else "")]
    if repeats:
        bits.append(f"{len(repeats)} re-raised ({', '.join(x['id'] for x in repeats)})")
    if below:
        bits.append(f"{len(below)} below {floor}, recorded only ({', '.join(x['id'] for x in below)})")
    if invalid:
        bits.append(f"{len(invalid)} of an unknown type ({', '.join(x['id'] for x in invalid)})")
    unchecked = [] if record.get("outcome") == "another_round" else \
        [x["id"] for x in new + below if x.get("type") == "novelty_to_verify"]  # no further round verifies them
    return (f"Re-check {record['recheck']}: {n} item(s): " + "; ".join(bits)
            + (f"; suspected prior work never verified: {', '.join(unchecked)}" if unchecked else ""))


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
