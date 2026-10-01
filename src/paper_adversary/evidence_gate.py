"""The evidence gate: which FATAL / MAJOR verdicts on prior work are actually shown.

Judges keep the four severities; a severity is what an objection would cost the paper if it is true. This module
decides, deterministically and per judge, whether a serious verdict that rests on prior work is *shown*. By default
(evidence.require_independent_check) that takes a blind verifier, who never saw the refuter's argument, finding
that the prior work anticipates the claim, with its own overlap quotes found verbatim in both texts
(verification.py; FATAL needs full anticipation, MAJOR at least partial). With the check switched off, the
refuter's own quotes, found verbatim in the prior paper and in the submission (evidence.py), suffice. Verdicts
that are not shown are labelled, never downgraded; the synthesis memo lists them under "Unverified threats —
check before acting" instead of "Criticisms that survived judging".
"""

from __future__ import annotations

from paper_adversary.reports import is_prior_work_objection, normalize_evidence_status

SERIOUS = ("FATAL", "MAJOR_FIXABLE")
QUOTED = ("quotes_verified", "quotes_approximate")
ANTICIPATES = {"FATAL": ("anticipates_fully",), "MAJOR_FIXABLE": ("anticipates_fully", "anticipates_partially")}


def _requests_for(oid: str, verification: dict, hashes: dict[str, str] | None) -> dict[str, dict]:
    """Results requested for this objection's current content (see verification.origin_hash)."""
    from paper_adversary.verification import current

    return {rid: r for rid, r in (verification.get("requests") or {}).items()
            if oid in (r.get("origin_ids") or []) and current(r, oid, hashes)}


def member_status(oid: str, evidence: dict, verification: dict, hashes: dict[str, str] | None = None) -> dict:
    """What is known about one prior-work objection: its quote check and its independent checks."""
    ev = evidence.get(oid) or {}
    mine = _requests_for(oid, verification, hashes)
    results = list(mine.values())
    verdicts = [r.get("verdict") for r in results if r.get("status") in ("verified", "disputed", "cannot_tell")]
    return {"id": oid, "quotes": ev.get("status") or "no_evidence", "basis": ev.get("evidence_basis"),
            "unavailable": [r.get("reason") for r in ev.get("references") or []
                            if str(r.get("full_text", "")).startswith("unavailable")],
            "not_found": bool(ev.get("references")) and all(r.get("full_text") == "not_found"
                                                             for r in ev.get("references") or []),
            "verdicts": verdicts, "verifiers": [r.get("agent_id") for r in results
                                                if r.get("agent_id") and r.get("status") in ("verified", "disputed",
                                                                                             "cannot_tell")],
            "requests": list(mine),
            "pending": [r.get("status") for r in results if r.get("status") in ("pending", "not_run",
                                                                                  "verifier_failed")]}


def _truthy(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"))


def _member_passes(m: dict, severity: str, require_independent: bool) -> bool:
    # an anticipation verdict only survives verification.collect when the verifier's quotes check out
    independent = any(v in ANTICIPATES[severity] for v in m["verdicts"])
    return independent if require_independent else (independent or m["quotes"] in QUOTED)


def _label(members: list[dict], severity: str, require_independent: bool) -> str:
    verdicts = [v for m in members for v in m["verdicts"]]
    if verdicts and all(v == "does_not_anticipate" for v in verdicts):
        return "DISPUTED by the blind verifier"
    if severity == "FATAL" and "anticipates_partially" in verdicts and "anticipates_fully" not in verdicts:
        return "SCOPE: only partial anticipation is verified"
    if members and all(m["not_found"] for m in members):
        return "REFERENCE NOT FOUND"
    if any(m["quotes"] == "prior_is_submission" for m in members):
        return "INVALID: the cited prior paper is the submission itself"
    if any(m["quotes"] == "reference_mismatch" for m in members):
        return "REFERENCE MISMATCH: the identifier resolves to a different paper than the one cited"
    if any(m["quotes"] in QUOTED for m in members):
        if "cannot_tell" in verdicts:
            return "UNCLEAR: the blind verifier could not tell"
        if require_independent:
            return "UNVERIFIED: not independently checked"
    if any(m["unavailable"] or m["quotes"] == "fulltext_unavailable" for m in members):
        reasons = sorted({r for m in members for r in m["unavailable"] if r})
        return "UNVERIFIED: full text unavailable" + (f" ({', '.join(reasons)})" if reasons else "")
    if any(m["basis"] == "abstract_only" for m in members):
        return "UNVERIFIED: only the abstract was read"
    if any(m["quotes"] == "quotes_not_found" for m in members):
        return "UNVERIFIED: the quoted passages were not found in the texts"
    return "UNVERIFIED: no full-text evidence"


def evaluate(judgments: dict[str, list[dict]], objections: dict[str, dict], evidence: dict, verification: dict,
             require_independent: bool = True, hashes: dict[str, str] | None = None) -> dict:
    """judgments: judge -> its judgments; objections: objection ID -> refuter objection (with category);
    evidence: objection ID -> quote-check record; verification: verify/results.json. Verdicts requested for an
    earlier version of an objection (before its refuter was rerun) are ignored."""
    from paper_adversary.verification import origin_hash

    hashes = hashes if hashes is not None else {oid: origin_hash(o) for oid, o in objections.items()}
    out: list[dict] = []
    for jid, items in sorted(judgments.items()):
        for j in items:
            severity = j.get("severity")
            ids = j.get("objection_ids") or []
            prior = [oid for oid in ids if is_prior_work_objection(objections.get(oid) or {})]
            relies = _truthy(j.get("relies_on_prior_work")) or bool(prior)
            if severity not in SERIOUS or not relies:
                continue
            members = [member_status(oid, evidence, verification, hashes) for oid in (prior or ids)]
            passes = any(_member_passes(m, severity, require_independent) for m in members)
            declared = normalize_evidence_status(j.get("evidence_status"))
            record = {"judge": jid, "objection_ids": ids, "title": j.get("title"), "severity": severity,
                      "passes": passes, "label": None if passes else _label(members, severity, require_independent),
                      "members": members, "judge_declared": declared,
                      "overstated": declared in ("verified_independent", "verified_quotes_only") and not passes}
            if passes:
                weak = [m["id"] for m in members if not _member_passes(m, severity, require_independent)]
                if weak:
                    record["note"] = f"shown through {', '.join(m['id'] for m in members if m['id'] not in weak)}; " \
                                     f"{', '.join(weak)} not shown — do not cite them as established"
            out.append(record)
    # per objection: a member that is not shown is flagged even when another member carries its judgment
    flagged = sorted({m["id"] for r in out for m in r["members"]
                      if not _member_passes(m, r["severity"], require_independent)})
    follow_up = sorted({rid for r in out if not r["passes"] for m in r["members"]
                        if m["quotes"] in QUOTED and not m["verdicts"] for rid in m["requests"]})
    return {"require_independent_check": require_independent, "verdicts": out, "flagged_ids": flagged,
            "overstated": [r for r in out if r["overstated"]], "follow_up_requests": follow_up}


def gate_markdown(gate: dict) -> str:
    rows = [r for r in gate.get("verdicts") or [] if not r["passes"]]
    rule = ("a blind verifier, who never saw the objection, found that the prior work anticipates the claim, with "
            "its own quotes verified verbatim in both texts (fully for FATAL, at least partially for MAJOR)"
            if gate.get("require_independent_check") else
            "the overlap is quoted verbatim from the prior paper's full text and the submission")
    lines = ["# Evidence gate", "", "Serious verdicts (FATAL or MAJOR BUT FIXABLE) that rest on prior work count as "
             f"shown only when {rule}. Severities are unchanged.", ""]
    if not rows:
        lines.append("Every serious verdict on prior work is shown.")
    for r in rows:
        sev = r["severity"].replace("MAJOR_FIXABLE", "MAJOR BUT FIXABLE")
        lines.append(f"- {r['judge']} {sev} on {', '.join(r['objection_ids'])} ({r.get('title') or ''}): {r['label']}")
        for m in r["members"]:
            checks = ", ".join(f"{a}: {v}" for a, v in zip(m["verifiers"], m["verdicts"])) or "no independent check"
            lines.append(f"    - {m['id']}: quotes {m['quotes'].replace('_', ' ')}; {checks}")
        if r["overstated"]:
            lines.append(f"    - {r['judge']} declared '{r['judge_declared']}', which the checks do not support")
    shown = [r for r in gate.get("verdicts") or [] if r["passes"] and r.get("note")]
    for r in shown:
        lines.append(f"- {r['judge']} on {', '.join(r['objection_ids'])}: shown; {r['note']}")
    return "\n".join(lines) + "\n"
