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

from paper_adversary.reports import is_grouped, is_prior_work_objection, member_severity, normalize_evidence_status

SERIOUS = ("FATAL", "MAJOR_FIXABLE")
QUOTED = ("quotes_verified", "quotes_approximate")
ANTICIPATES = {"FATAL": ("anticipates_fully",), "MAJOR_FIXABLE": ("anticipates_fully", "anticipates_partially")}


def _requests_for(oid: str, verification: dict, hashes: dict[str, str] | None) -> dict[str, dict]:
    """Results requested for this objection's current content (see verification.origin_hash)."""
    from paper_adversary.verification import current

    return {rid: r for rid, r in (verification.get("requests") or {}).items()
            if oid in (r.get("origin_ids") or []) and current(r, oid, hashes)}


def member_status(oid: str, evidence: dict, verification: dict, hashes: dict[str, str] | None = None,
                  obj: dict | None = None) -> dict:
    """What is known about one member of a serious verdict on prior work (`obj`: its refuter objection, when known):
    its quote check, its independent checks and, when it has none, why (never requested, capped, unavailable, not
    a prior-work objection, ...)."""
    ev = evidence.get(oid) or {}
    mine = _requests_for(oid, verification, hashes)
    results = list(mine.values())
    done = [r for r in results if r.get("status") in ("verified", "disputed", "cannot_tell")]
    verdicts = [r.get("verdict") for r in done]
    # requests still without a verdict; references past the per-objection limit are counted, not waited for
    waiting = [r for r in results if r.get("status") not in ("verified", "disputed", "cannot_tell")
               and not r.get("limited")]
    limited = sum(1 for r in results if r.get("limited") and r.get("status") == "not_run")
    capped = any(r.get("status") == "not_run" and "cap" in str(r.get("reason") or "") for r in waiting)
    return {"id": oid, "quotes": ev.get("status") or "no_evidence", "basis": ev.get("evidence_basis"),
            "unavailable": [r.get("reason") for r in ev.get("references") or []
                            if str(r.get("full_text", "")).startswith("unavailable")],
            "not_found": bool(ev.get("references")) and all(r.get("full_text") == "not_found"
                                                             for r in ev.get("references") or []),
            "verdicts": verdicts, "verifiers": [r.get("agent_id") for r in done if r.get("agent_id")],
            "requests": list(mine),
            "pending": [r.get("status") for r in results if r.get("status") in ("pending", "not_run",
                                                                                  "verifier_failed")],
            "capped": capped, "limited": limited, "prior_work": None if obj is None else is_prior_work_objection(obj),
            "reason": None if verdicts else _why_unchecked(ev, results, waiting, limited, obj)}


_UNCHECKED = {"pending": "verification pending", "verifier_failed": "the verifier failed",
              "claim_not_located": "the claim was not located in the submission",
              "prior_not_found": "the cited paper was not found",
              "reference_mismatch": "the identifier resolves to another paper",
              "prior_is_submission": "the cited paper is the submission itself"}


def _why_unchecked(ev: dict, results: list[dict], waiting: list[dict], limited: int, obj: dict | None = None) -> str:
    """Why a member has no blind verdict, from its own records (not from another member's)."""
    if not results:
        if obj is not None and not is_prior_work_objection(obj):  # in a judgment that says it relies on prior work
            return ("not requested: not a prior-work objection" if obj.get("references")
                    else "not requested: it cites no prior paper to check")
        if ev and not ev.get("decisive"):
            return f"not requested: the refuter rated it {ev.get('severity_estimate') or 'below major'}; judges " \
                   "escalated it"
        return "not requested"
    out = []
    for r in waiting:
        status = r.get("status")
        if status == "not_run":
            text = "cap reached" if "cap" in str(r.get("reason") or "") else f"not run ({r.get('reason')})"
        elif status == "fulltext_unavailable":
            text = "full text unavailable" + (f": {r['reason']}" if r.get("reason") else "")
        else:
            text = _UNCHECKED.get(status, str(status))
        if text not in out:
            out.append(text)
    if limited:
        out.append(f"{limited} reference(s) beyond the per-objection limit not checked")
    return "; ".join(out[:4]) or "no independent check"


def _truthy(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"))


def _member_passes(m: dict, severity: str, require_independent: bool) -> bool:
    # an anticipation verdict only survives verification.collect when the verifier's quotes check out
    independent = any(v in ANTICIPATES[severity] for v in m["verdicts"])
    return independent if require_independent else (independent or m["quotes"] in QUOTED)


def _unchecked(members: list[dict]) -> str:
    """Per member without a blind verdict: why ('N2-O2: cap reached; N1-O5: not requested: ...')."""
    return "; ".join(f"{m['id']}: {m['reason']}" for m in members if not m["verdicts"] and m.get("reason"))


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
    if not verdicts and members and all(m.get("capped") for m in members):
        return "UNVERIFIED: not independently checked (the run's verification cap was reached)"
    if not verdicts and any(m.get("capped") for m in members):  # some capped, others never requested etc.
        return f"UNVERIFIED: not independently checked ({_unchecked(members)})"
    if any(m["quotes"] in QUOTED for m in members):
        if "cannot_tell" in verdicts:
            return "UNCLEAR: the blind verifier could not tell"
        if require_independent:
            why = _unchecked(members)
            return "UNVERIFIED: not independently checked" + (f" ({why})" if why else "")
    if any(m["unavailable"] or m["quotes"] == "fulltext_unavailable" for m in members):
        reasons = sorted({r for m in members for r in m["unavailable"] if r})
        return "UNVERIFIED: full text unavailable" + (f" ({', '.join(reasons)})" if reasons else "")
    if any(m["basis"] == "abstract_only" for m in members):
        return "UNVERIFIED: only the abstract was read"
    if any(m["quotes"] == "quotes_not_found" for m in members):
        return "UNVERIFIED: the quoted passages were not found in the texts"
    return "UNVERIFIED: no full-text evidence"


def _misdeclared(declared: str | None, members: list[dict], severity: str) -> str | None:
    """Whether a judge's declared evidence status claims more than the record shows, compared with what that
    declaration itself claims (whatever the gate requires, and whether or not the verdict passes)."""
    if declared == "verified_independent" and not any(v in ANTICIPATES[m.get("severity") or severity] for m in members
                                                       for v in m["verdicts"]):
        need = "full" if severity == "FATAL" else "at least partial"
        found = ", ".join(f"{m['id']}: {', '.join(m['verdicts'])}" for m in members if m["verdicts"])
        return (f"declared 'verified_independent'; the record shows no blind verifier finding {need} anticipation "
                "with verified quotes" + (f" ({found})" if found else " (no blind verdict)"))
    if declared == "verified_quotes_only" and not any(m["quotes"] in QUOTED for m in members):
        checks = ", ".join(f"{m['id']}: {m['quotes'].replace('_', ' ')}" for m in members)
        return f"declared 'verified_quotes_only'; the record shows no member's quotes verified ({checks})"
    return None


def evaluate(judgments: dict[str, list[dict]], objections: dict[str, dict], evidence: dict, verification: dict,
             require_independent: bool = True, hashes: dict[str, str] | None = None) -> dict:
    """judgments: judge -> its judgments; objections: objection ID -> refuter objection (with category);
    evidence: objection ID -> quote-check record; verification: verify/results.json. Verdicts requested for an
    earlier version of an objection (before its refuter was rerun) are ignored.

    Each member of a grouped judgment is held to its own severity (member_severity, judge_v3; else the
    judgment's). A member whose own severity is below MAJOR is not checked as part of the serious verdict and
    not flagged; it is listed apart (`lower_members`). A judgment that says it relies on prior work while no
    prior-work member carries its severity (it has none, or rates each lower) has its serious members checked. A
    flagged ID whose every failing verdict is grouped (the judgment gave it no severity of its own) while judges
    that ruled on it alone rated it lower is listed in `flagged_grouped_only`."""
    from paper_adversary.verification import origin_hash

    hashes = hashes if hashes is not None else {oid: origin_hash(o) for oid, o in objections.items()}
    out: list[dict] = []
    alone: dict[str, list[str]] = {}  # objection ID -> its severities where a judge ruled on it on its own
    for jid, items in sorted(judgments.items()):
        for j in items:
            ids = j.get("objection_ids") or []
            own = {oid: member_severity(j, oid) for oid in ids}
            for oid in ids:
                if not is_grouped(j, oid):
                    alone.setdefault(oid, []).append(own[oid])
            prior = [oid for oid in ids if is_prior_work_objection(objections.get(oid) or {})]
            cited = prior or ids
            serious = [oid for oid in prior if own[oid] in SERIOUS]
            if not serious and _truthy(j.get("relies_on_prior_work")):
                # the judge says the verdict rests on prior work, but no prior-work member carries it (there is none,
                # or each is rated lower): its serious members are held to the evidence rule instead
                serious = [oid for oid in ids if own[oid] in SERIOUS]
            if not serious:
                continue
            severity = "FATAL" if any(own[oid] == "FATAL" for oid in serious) else "MAJOR_FIXABLE"
            members = [{**member_status(oid, evidence, verification, hashes, objections.get(oid)),
                        "severity": own[oid], "grouped": is_grouped(j, oid)} for oid in serious]
            passes = any(_member_passes(m, m["severity"], require_independent) for m in members)
            declared = normalize_evidence_status(j.get("evidence_status"))
            mismatch = _misdeclared(declared, members, severity)
            record = {"judge": jid, "objection_ids": ids, "title": j.get("title"), "severity": severity,
                      "passes": passes, "label": None if passes else _label(members, severity, require_independent),
                      "members": members, "judge_declared": declared, "overstated": mismatch is not None,
                      "overstated_detail": mismatch,
                      "lower_members": [{"id": oid, "severity": own[oid]} for oid in cited if oid not in serious]}
            if passes:
                weak = [m["id"] for m in members if not _member_passes(m, m["severity"], require_independent)]
                if weak:
                    record["note"] = f"shown through {', '.join(m['id'] for m in members if m['id'] not in weak)}; " \
                                     f"{', '.join(weak)} not shown — do not cite them as established"
            out.append(record)
    # per objection: a member that is not shown is flagged even when another member carries its judgment
    failing = [(r, m) for r in out for m in r["members"] if not _member_passes(m, m["severity"], require_independent)]
    flagged = sorted({m["id"] for _, m in failing})
    grouped_only = sorted(oid for oid in flagged
                          if all(m["grouped"] for _, m in failing if m["id"] == oid) and alone.get(oid)
                          and all(s not in SERIOUS for s in alone[oid]))
    follow_up = sorted({rid for r in out if not r["passes"] for m in r["members"]
                        if m["quotes"] in QUOTED and not m["verdicts"] for rid in m["requests"]})
    # flagged prior-work objections that no verification batch was ever asked about (e.g. escalated by the judges)
    never = sorted({m["id"] for r in out for m in r["members"] if m["id"] in flagged and not m["requests"]
                    and m.get("prior_work") is not False})
    return {"require_independent_check": require_independent, "verdicts": out, "flagged_ids": flagged,
            "flagged_grouped_only": grouped_only,
            "overstated": [r for r in out if r["overstated"]], "follow_up_requests": follow_up,
            "never_requested": never}


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
            checks = ", ".join(f"{a}: {v}" for a, v in zip(m["verifiers"], m["verdicts"])) or \
                f"no independent check ({m.get('reason') or 'none'})"
            own = f" (its own severity: {_sev(m['severity'])})" if m.get("severity") not in (None, r["severity"]) \
                else ""
            lines.append(f"    - {m['id']}{own}: quote check: {m['quotes'].replace('_', ' ')}; {checks}")
        lines += _lower_lines(r)
        if r["overstated"]:
            lines.append(f"    - {r['judge']} {_overstated_text(r)}")
    shown = [r for r in gate.get("verdicts") or [] if r["passes"] and (r.get("note") or r["overstated"]
                                                                        or r.get("lower_members"))]
    for r in shown:
        lines.append(f"- {r['judge']} on {', '.join(r['objection_ids'])}: shown"
                     + (f"; {r['note']}" if r.get("note") else ""))
        lines += _lower_lines(r)
        if r["overstated"]:
            lines.append(f"    - {r['judge']} {_overstated_text(r)}")
    if gate.get("flagged_grouped_only"):
        lines += ["", "Flagged only through grouped verdicts (judgments that gave them no severity of their own), "
                  "while judges that ruled on them alone rated them lower: " + ", ".join(gate["flagged_grouped_only"])]
    return "\n".join(lines) + "\n"


def _sev(severity: str | None) -> str:
    return str(severity or "?").replace("MAJOR_FIXABLE", "MAJOR BUT FIXABLE").replace("_", " ")


def _lower_lines(r: dict) -> list[str]:
    """Members a serious judgment cites whose own severity (member_severity) is below MAJOR: not checked as part
    of it and not flagged."""
    return [f"    - {x['id']}: cited in this judgment, but {r['judge']}'s own severity for it is {_sev(x['severity'])}; "
            "not checked as a serious verdict" for x in r.get("lower_members") or []]


def _overstated_text(r: dict) -> str:
    return r.get("overstated_detail") or f"declared '{r['judge_declared']}', which the checks do not support"
