"""Follow-up rounds: what happens to the completeness critic's findings.

Round r takes the items of its critic (C1 for round 1, then the previous round's re-check critic) through:

  1. triage (deterministic, free): validate every item, route it (adjudicate, revision only, noted, invalid),
     and attach the judges' verdicts on objections it cites;
  2. verification: items naming suspected prior work go to blind verifiers through verification.py, which see
     only the submission's passage and the prior paper, never the critic's prose;
  3. adjudication: independent adjudicators rule on every routed item with the judges' four severities; a
     follow-up matrix and the same evidence gate as the judges' are computed from their rulings;
  4. a revised memo S<r+1> that disposes of every item (the gate quarantines it if one is missing or an
     unverified prior-work item is not filed as an unverified threat), which becomes the current memo;
  5. a fresh re-check critic C<r+1>, whose new items decide whether another round runs.

Items the re-check critic re-raises (it may only do so when the handling was plainly wrong or new evidence
exists) never start a round, since the adjudicators already ruled on them; they stay open. A review that stops
without new items is labelled ready_for_next_gate only when nothing serious is open: no re-raised item, no FATAL
verdict that adjudicators did not overturn, no unverified FATAL prior-work threat. Otherwise it is
review_saturated_with_open_issues, with the list (open MAJOR BUT FIXABLE issues are listed but do not block).

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
                    "judge_ids": _id_list(item.get("judge_ids")), "claim_ids": _id_list(item.get("claim_ids")),
                    "candidate_references": [r for r in refs if isinstance(r, dict)] if isinstance(refs, list) else [],
                    "severity": normalize_severity(item.get("severity_estimate"))})
    return out


def _rank(severity: str | None) -> int:
    return SEVERITY_RANK.get(severity or "", -1)


# ---------------------------------------------------------------- 1. triage


def triage(store, cfg: PipelineConfig, state: dict, r: int) -> dict:
    """Validate and route the round's items; writes followup/round-<r>/items.json (once) and items.md."""
    path = round_dir(store, r) / "items.json"
    existing = read_json(path)
    if existing:
        return existing
    critic_id, role = round_critic(state, r)
    side = read_json(store.sidecar_path(critic_id, role, ".json"), {}) or {}
    _, body = read_report(store.report_path(critic_id, role))
    matrix = read_json(store.role_dir("judge") / "judgment_matrix.json", {}) or {}
    rows = {row["id"]: row for row in matrix.get("rows") or []}
    judges = set(matrix.get("judges") or [])
    floor = _rank(cfg.followup.min_severity)
    items = []
    for item in critic_items(side.get("data"), critic_id):
        problems = []
        if item.get("type") not in ITEM_TYPES:
            problems.append(f"unknown type {item.get('type')!r}")
        unknown = [o for o in item.get("objection_ids") or [] if o not in rows]
        if unknown:
            problems.append(f"cites objection IDs that do not exist: {', '.join(unknown)}")
        bad_judges = [j for j in item.get("judge_ids") or [] if j not in judges]
        if bad_judges:
            problems.append(f"cites judges that do not exist: {', '.join(bad_judges)}")
        refs = [ref for ref in item.get("candidate_references") or [] if isinstance(ref, dict)
                and (ref.get("title") or ref.get("doi") or ref.get("arxiv_id"))]
        if problems:
            route = "invalid"
        elif item.get("type") == "synthesis_flaw":
            route = "revision"
        elif _rank(item.get("severity")) < floor:
            route = "noted"
        else:
            route = "adjudicate"
        cited = {o: {j: v.get("severity") for j, v in (rows.get(o) or {}).get("verdicts", {}).items()}
                 for o in item.get("objection_ids") or [] if o in rows}
        hints = sorted(((title_similarity(item.get("title") or "", row.get("title") or ""), oid)
                        for oid, row in rows.items()), reverse=True)
        items.append({**item, "route": route, "problems": problems, "verify": bool(refs) and route != "invalid",
                      "references": refs, "cited_verdicts": cited,
                      "possibly_covered_by": [f"{oid} ({s:.2f})" for s, oid in hints[:2] if s >= 0.6]})
    record = {"round": r, "critic": critic_id, "critic_sha256": sha256_text(body), "created_at": utcnow_iso(),
              "items": items}
    atomic_write_json(path, record)
    atomic_write_text(round_dir(store, r) / "items.md", items_markdown(record))
    return record


def items_markdown(record: dict) -> str:
    lines = [f"Completeness-critique items of {record['critic']} (follow-up round {record['round']}), as triaged by "
             "the orchestrator. Route 'adjudicate' items need a ruling; 'revision' items are memo problems for the "
             "revised memo; 'noted' items are below the adjudication threshold; 'invalid' items are malformed."]
    for it in record["items"]:
        lines.append(f"- {it['id']} [{it.get('type')}; critic: {it.get('severity') or '?'}; route: {it['route']}] "
                     f"{it.get('title') or ''} — {it.get('location') or 'no location'}")
        if it.get("argument"):
            lines.append(f"  {' '.join(str(it['argument']).split())[:1200]}")
        for oid, verdicts in it.get("cited_verdicts", {}).items():
            lines.append(f"  cites {oid}: " + (", ".join(f"{j} {SEVERITY_LABEL.get(s, s)}" for j, s in
                                                        sorted(verdicts.items())) or "no judge classified it"))
        if it.get("references"):
            lines.append("  suspected prior work: " + "; ".join(
                f"{ref.get('title') or ref.get('arxiv_id') or ref.get('doi')}" for ref in it["references"]))
        if it.get("possibly_covered_by"):
            lines.append(f"  possibly covered by: {', '.join(it['possibly_covered_by'])}")
        if it.get("problems"):
            lines.append(f"  problems: {'; '.join(it['problems'])}")
    return "\n".join(lines) + "\n"


def _item_core(it: dict) -> dict:
    """The fields that identify a critic item's content (for verification.origin_hash)."""
    return {k: it.get(k) for k in ("id", "title", "type", "location", "candidate_references", "severity_estimate")}


def routable(record: dict) -> list[dict]:
    return [it for it in record["items"] if it["route"] in ("adjudicate", "revision") or it["verify"]]


def verification_requests(record: dict) -> list[VerificationRequest]:
    out = []
    for it in record["items"]:
        if not it["verify"]:
            continue
        for k, ref in enumerate(it["references"][:3], start=1):
            ident = str(ref.get("arxiv_id") or ref.get("doi") or ref.get("title") or "").strip()
            out.append(VerificationRequest(
                request_id=f"{it['id']}:r{k}", origin="critic", origin_ids=(it["id"],),
                prior={"identifier": ident, "title": ref.get("title"), "doi": ref.get("doi"),
                       "arxiv_id": ref.get("arxiv_id"), "year": ref.get("year")},
                claim_quote=None, claim_location=it.get("location"), note=it.get("argument"),
                origin_hash=origin_hash(_item_core(it))))
    return out


# ---------------------------------------------------------------- 2. planning


def plan_agents(cfg: PipelineConfig, registry, state: dict, r: int) -> list[AgentSpec]:
    """Adjudicators A<n> (numbered across rounds), the revised memo S<r+1> and the re-check critic C<r+1>."""
    fc = cfg.followup
    first = next_free(state, "A")
    specs = []

    def spec(aid: str, role: str, index: int) -> AgentSpec:
        rc = cfg.role(role)
        return AgentSpec(agent_id=aid, role=role, index=index, model_alias=rc.model,
                         model_id=registry.resolve(rc.model).id, effort=rc.effort, prompt_name=rc.prompt,
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
    re_rulings = [{"adjudicator": aid, **u} for aid, rs in rulings.items() for x in rs
                  for u in x.get("objection_updates") or [] if isinstance(u, dict)]
    matrix["objection_updates"] = re_rulings
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
    floor; otherwise ready_for_next_gate or review_saturated_with_open_issues (see the module docstring)."""
    critic_id = (((state.get("followup") or {}).get("rounds") or {}).get(str(r)) or {}).get("recheck") or f"C{r + 1}"
    side = read_json(store.sidecar_path(critic_id, "recheck", ".json"), {}) or {}
    earlier = []  # only items that were actually adjudicated or disposed of count as already handled
    for k in range(1, r + 1):
        earlier += [{**it, "round": k} for it in (read_json(round_dir(store, k) / "items.json", {}) or {}).get("items")
                    or [] if it.get("route") in ("adjudicate", "revision")]
    floor = _rank(cfg.followup.min_severity)
    new, repeats = [], []
    for it in critic_items(side.get("data"), critic_id):
        if _rank(it.get("severity")) < floor or it.get("type") not in ITEM_TYPES:
            continue
        twin = _repeat_of(it, earlier)
        (repeats if twin else new).append({"id": it["id"], "type": it.get("type"), "severity": it.get("severity"),
                                            "title": it.get("title"), "repeats": twin})
    issues = open_issues(store, state, _disputes(store, state, repeats, earlier))
    if new:
        outcome = "max_rounds_reached" if r >= cfg.followup.max_rounds else "another_round"
    else:
        outcome = SATURATED if issues["blocking"] else READY
    result = {"round": r, "recheck": critic_id, "outcome": outcome, "new_items": new, "disputed_repeats": repeats,
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


def open_issues(store, state: dict, disputes: list[dict] | None = None) -> dict:
    """What keeps a finished review from being ready for the next gate ("blocking": re-raised items, FATAL
    verdicts that adjudicators did not overturn, unverified FATAL prior-work threats) and what is open but
    fixable ("major": MAJOR BUT FIXABLE verdicts, listed only). Verdicts come from the judges' matrix, the
    adjudicators' rulings on follow-up items, and their re-ratings of base objections (the latest round that
    re-rated an objection decides; it overturns a FATAL only if all its adjudicators rated it lower)."""
    rounds = (state.get("followup") or {}).get("rounds") or {}
    flagged = set((read_json(store.role_dir("judge") / "evidence_gate.json", {}) or {}).get("flagged_ids") or [])
    verdicts: dict[str, dict[str, str]] = {}  # objection or item ID -> {judge or adjudicator: severity}
    titles: dict[str, str] = {}
    for row in (read_json(store.role_dir("judge") / "judgment_matrix.json", {}) or {}).get("rows") or []:
        verdicts[row["id"]] = {j: v.get("severity") for j, v in (row.get("verdicts") or {}).items()}
        titles[row["id"]] = row.get("title") or ""
    rerated: dict[str, tuple[int, dict[str, str]]] = {}
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
    for oid, (k, by) in rerated.items():
        panel = [a for a in (rounds.get(str(k)) or {}).get("adjudicators") or []
                 if (state["agents"].get(a) or {}).get("status") == "complete"]
        current = dict(verdicts.get(oid) or {})
        if panel and all(by.get(a) not in (None, "FATAL") for a in panel):  # every adjudicator rated it lower
            current = {j: s for j, s in current.items() if s != "FATAL"}
        current.update(by)
        verdicts[oid] = current
    fatal, unverified, major = [], [], []
    for oid in sorted(verdicts):
        by = sorted(j for j, sev in verdicts[oid].items() if sev == "FATAL")
        entry = {"id": oid, "title": titles.get(oid, "")[:120], "by": by}
        if by:
            (unverified if oid in flagged else fatal).append(entry)
        else:
            majors = sorted(j for j, sev in verdicts[oid].items() if sev == "MAJOR_FIXABLE")
            if majors:
                major.append({**entry, "by": majors, "unverified": oid in flagged})
    blocking = {"reraised": list(disputes or []), "fatal": fatal, "unverified_fatal": unverified}
    return {"blocking": blocking if any(blocking.values()) else {}, "major": major}


def open_issue_lines(issues: dict) -> list[str]:
    """Short status lines for open issues."""
    b = issues.get("blocking") or {}
    lines = []
    if b.get("reraised"):
        lines.append("re-raised by the re-check critic: " + ", ".join(
            f"{d['id']} (repeats {d['repeats']}; earlier ruled "
            + (", ".join(f"{a} {SEVERITY_LABEL.get(s, s)}" for a, s in (d.get("earlier_rulings") or {}).items())
               or "—") + f", {d.get('earlier_disposition') or 'no disposition'})" for d in b["reraised"]))
    if b.get("fatal"):
        lines.append("FATAL verdicts standing: " + ", ".join(f"{x['id']} ({'/'.join(x['by'])})" for x in b["fatal"]))
    if b.get("unverified_fatal"):
        lines.append("unverified FATAL prior-work threats: " + ", ".join(x["id"] for x in b["unverified_fatal"]))
    if issues.get("major"):
        lines.append(f"open MAJOR BUT FIXABLE (not blocking): {len(issues['major'])}")
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


def round_summary(store, state: dict) -> list[str]:
    """Status lines for the follow-up rounds."""
    fu = state.get("followup") or {}
    lines = []
    for r, rnd in sorted((fu.get("rounds") or {}).items(), key=lambda kv: int(kv[0])):
        items = (read_json(round_dir(store, int(r)) / "items.json", {}) or {}).get("items") or []
        routes: dict[str, int] = {}
        for it in items:
            routes[it["route"]] = routes.get(it["route"], 0) + 1
        bits = ", ".join(f"{v} {k}" for k, v in sorted(routes.items())) or "no items"
        lines.append(f"Follow-up round {r}: {rnd.get('status')} — items of {rnd.get('critic')}: {bits}"
                     + (f"; outcome {rnd['outcome']}" if rnd.get("outcome") else ""))
        if rnd.get("outcome") in (READY, SATURATED, "max_rounds_reached"):
            record = read_json(round_dir(store, int(r)) / "round.json", {}) or {}
            issues = record.get("open_issues") or rnd.get("open_issues") or {}
            lines += [f"  Open: {line}" for line in open_issue_lines(issues)]
    if fu.get("current_memo") and fu["current_memo"] != "S1":
        lines.append(f"Current memo: {fu['current_memo']} (revised; earlier memos are kept)")
    return lines


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
