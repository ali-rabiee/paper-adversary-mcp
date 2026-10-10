"""Report files (Markdown + YAML front matter), structured-block parsing, and the judgment matrix."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from paper_adversary.util import atomic_write_text

SEVERITIES = ("FATAL", "MAJOR_FIXABLE", "MINOR", "NOT_CONVINCING")
SEVERITY_LABEL = {"FATAL": "FATAL", "MAJOR_FIXABLE": "MAJOR BUT FIXABLE", "MINOR": "MINOR",
                  "NOT_CONVINCING": "NOT CONVINCING"}
SEVERITY_RANK = {"FATAL": 3, "MAJOR_FIXABLE": 2, "MINOR": 1, "NOT_CONVINCING": 0}



# ---------------------------------------------------------------- files


def write_report(path: Path, meta: dict, body: str, marker: str) -> None:
    front = yaml.safe_dump(json.loads(json.dumps(meta, default=str)), sort_keys=False, allow_unicode=True)
    atomic_write_text(path, f"---\n{front}---\n{marker}\n\n{body.strip()}\n")


def read_report(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            meta = yaml.safe_load(text[4:end]) or {}
            return meta, text[end + 5 :].lstrip("\n")
    return {}, text


def strip_marker(body: str) -> str:
    return re.sub(r"^<!-- aid:[A-Z0-9]+:[0-9a-f]{12} -->\n+", "", body)


# ---------------------------------------------------------------- structured blocks


def normalize_severity(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = re.sub(r"[^A-Z]+", "_", value.upper()).strip("_")
    key = {"MAJOR_BUT_FIXABLE": "MAJOR_FIXABLE", "MAJOR": "MAJOR_FIXABLE", "NOT_CONVINCED": "NOT_CONVINCING",
           "UNCONVINCING": "NOT_CONVINCING", "REJECTED": "NOT_CONVINCING"}.get(key, key)
    return key if key in SEVERITY_RANK else None


def refuter_objections(data: dict | None, agent_id: str) -> list[dict]:
    """Objections from a refuter's block, with run-unique IDs such as 'N2-O3'."""
    if not data:
        return []
    out = []
    for i, obj in enumerate(data.get("objections") or [], start=1):
        if not isinstance(obj, dict):
            continue
        local = str(obj.get("id") or f"O{i}").strip()
        local = re.sub(r"^[A-Z]\d+-", "", local)
        if not re.fullmatch(r"O\d+", local):
            local = f"O{i}"
        out.append({**obj, "id": f"{agent_id}-{local}", "source": agent_id})
    return out


def judge_judgments(data: dict | None) -> list[dict]:
    """A judge's judgments with normalized IDs and severities. `member_severity` (judge_v3) keeps only entries for
    the judgment's own IDs with a valid severity (gates.validate_block warns about the others)."""
    if not data:
        return []
    out = []
    for j in data.get("judgments") or []:
        if not isinstance(j, dict):
            continue
        ids = j.get("objection_ids") or j.get("objection_id") or []
        if isinstance(ids, str):
            ids = [ids]
        ids = [re.sub(r"\s+", "", str(x)).upper() for x in ids if x]
        raw = j.get("member_severity")
        member = {}
        for key, value in (raw.items() if isinstance(raw, dict) else []):
            oid, sev = re.sub(r"\s+", "", str(key)).upper(), normalize_severity(value)
            if oid in ids and sev:
                member[oid] = sev
        out.append({**j, "objection_ids": ids, "severity": normalize_severity(j.get("severity")),
                    "member_severity": member})
    return out


def member_severity(judgment: dict, oid: str) -> str | None:
    """One objection's own severity in a judgment: its member_severity entry, else the judgment's severity (which
    judge_v3 defines as its strongest member's)."""
    return (judgment.get("member_severity") or {}).get(oid) or judgment.get("severity")


def is_grouped(judgment: dict, oid: str) -> bool:
    """Whether this objection's severity in the judgment is only the group's: the judgment covers several
    objections and gives this one no severity of its own. Under a judge prompt that defines member_severity
    (judge_v3, front matter `member_severity: true`, marked `own_severities` by build_matrix), a member left out of
    member_severity is rated at the judgment's severity by the judge's own contract, so no member is only grouped."""
    return len(judgment.get("objection_ids") or []) > 1 and oid not in (judgment.get("member_severity") or {}) \
        and not judgment.get("own_severities")


RUBRIC_APPLIES = {"yes": "yes", "no": "no", "partly": "partly", "partially": "partly", "partial": "partly"}


def rubric_ruling(entry: object) -> tuple[dict | None, str | None]:
    """(normalized ruling, problem) for one entry of a judge's rubric_rulings (judge_v3). A test that applies (yes
    or partly) needs a severity; one that does not apply has none."""
    if not isinstance(entry, dict):
        return None, "a ruling is not an object"
    tid = re.sub(r"\s+", "", str(entry.get("test_id") or entry.get("id") or "")).upper()
    if not tid:
        return None, "a ruling has no test_id"
    applies = RUBRIC_APPLIES.get(re.sub(r"[^a-z]+", "", str(entry.get("applies") or "").lower()))
    rec = {"test_id": tid, "applies": applies, "severity": None,
           "rationale": " ".join(str(entry.get("rationale") or "").split())}
    if applies is None:
        return rec, f"{tid}: applies {entry.get('applies')!r} is not yes, no or partly"
    if applies != "no":
        rec["severity"] = normalize_severity(entry.get("severity"))
        if rec["severity"] is None:
            return rec, f"{tid}: applies '{applies}' but severity {entry.get('severity')!r} is not one of the four"
    return rec, None


def rubric_rulings(data: dict | None) -> list[dict]:
    """A judge's valid rubric rulings, normalized (invalid ones are what the gate's rubric check reports)."""
    raw = (data or {}).get("rubric_rulings")
    out = []
    for entry in raw if isinstance(raw, list) else []:
        rec, problem = rubric_ruling(entry)
        if rec is not None and problem is None:
            out.append(rec)
    return out


# Novelty objections that claim the prior work already contains something. These must be shown with verbatim
# full-text evidence before a FATAL or MAJOR verdict on them counts as established (evidence_gate.py).
PRIOR_WORK_CATEGORIES = {"already_done", "partially_anticipated", "framing_exists", "gap_not_real", "concurrent_work"}
EVIDENCE_STATUSES = ("verified_independent", "verified_quotes_only", "disputed", "abstract_only",
                     "fulltext_unavailable", "unverified", "not_applicable")


_CATEGORY_ALIASES = {"anticipated": "partially_anticipated", "prior_work": "already_done",
                     "already_exists": "already_done", "not_novel": "already_done", "exists": "already_done",
                     "concurrent": "concurrent_work", "framing": "framing_exists", "no_gap": "gap_not_real"}


def normalize_category(value: object) -> str:
    key = re.sub(r"[^a-z]+", "_", str(value or "").lower()).strip("_") if isinstance(value, str) else ""
    return _CATEGORY_ALIASES.get(key, key)


def is_prior_work_objection(obj: dict) -> bool:
    """Whether an objection rests on prior work. Exact labels are not trusted: any objection that cites
    references is treated as one (except a pure missing-citation note), so a relabelled overlap claim
    cannot slip past the evidence gate."""
    category = normalize_category(obj.get("category"))
    refs = obj.get("references")
    return category in PRIOR_WORK_CATEGORIES or (bool(refs) and isinstance(refs, list)
                                                 and category != "missing_citation")


def is_decisive(obj: dict) -> bool:
    severity = normalize_category(obj.get("severity_estimate"))
    return is_prior_work_objection(obj) and severity in {"fatal", "major", "major_but_fixable"}


def normalize_evidence_status(value: object) -> str | None:
    key = re.sub(r"[^a-z]+", "_", str(value or "").lower()).strip("_")
    return key if key in EVIDENCE_STATUSES else None


def references_from(data: dict | None) -> list[dict]:
    refs: list[dict] = []
    if not data:
        return refs
    for obj in data.get("objections") or []:
        for ref in (obj.get("references") or []) if isinstance(obj, dict) else []:
            if isinstance(ref, dict) and (ref.get("title") or ref.get("doi") or ref.get("arxiv_id")):
                refs.append({**ref, "objection": obj.get("id")})
    for ref in data.get("prior_work") or data.get("closest_prior_work") or []:
        if isinstance(ref, dict) and (ref.get("title") or ref.get("doi") or ref.get("arxiv_id")):
            refs.append({**ref, "objection": None})
    return refs


# ---------------------------------------------------------------- judgment matrix


def build_judgment_matrix(objections: dict[str, list[dict]], judgments: dict[str, list[dict]],
                          excluded: dict[str, str] | None = None, rubric_tests: list[dict] | None = None,
                          rubric: dict[str, list[dict]] | None = None) -> dict:
    """Cross-tabulate every refuter objection against every judge's verdict.

    Verdicts are kept per judge, never averaged, each at the objection's own severity (member_severity of a
    grouped judgment, else the judgment's); a verdict records its judgment (`group`, `group_ids`,
    `group_severity`) and whether its severity is only the group's (`grouped`). Flags mark objections no judge
    classified, objections some judges skipped, and contested objections.
    `excluded` names refuters and judges left out (failed or quarantined) with a
    short label; judge citations of an excluded refuter's IDs are listed apart.
    `rubric_tests` (the rubric's declared tests) and `rubric` (judge -> its normalized rubric rulings) add one
    row per test with each judge's ruling.
    """
    excluded = excluded or {}
    known = {o["id"]: o for objs in objections.values() for o in objs}
    rows: dict[str, dict] = {
        oid: {"id": oid, "source": o["source"], "title": o.get("title") or o.get("objection") or "",
              "refuter_severity": o.get("severity_estimate"), "verdicts": {}}
        for oid, o in known.items()
    }
    unknown_refs: dict[str, list[str]] = {}
    excluded_refs: dict[str, list[str]] = {}
    double: dict[str, list[str]] = {}
    for jid, items in judgments.items():
        for k, item in enumerate(items, start=1):
            for oid in item["objection_ids"]:
                if oid not in rows:
                    target = excluded_refs if oid.split("-")[0] in excluded else unknown_refs
                    target.setdefault(jid, []).append(oid)
                    continue
                if jid in rows[oid]["verdicts"]:  # keep the first verdict and say so; never overwrite silently
                    double.setdefault(jid, []).append(oid)
                    continue
                rows[oid]["verdicts"][jid] = {"severity": member_severity(item, oid),
                                              "confidence": item.get("confidence"),
                                              "resolvable": item.get("resolvable_with_more_evidence"),
                                              "group": f"{jid}#{k}", "group_ids": list(item["objection_ids"]),
                                              "group_severity": item.get("severity"),
                                              "grouped": is_grouped(item, oid)}
    judges = sorted(judgments)
    for row in rows.values():
        ranks = [SEVERITY_RANK[v["severity"]] for v in row["verdicts"].values() if v.get("severity")]
        row["unclassified_by"] = [j for j in judges if j not in row["verdicts"]]
        row["contested"] = bool(ranks) and (max(ranks) - min(ranks) >= 2)
        row["unanimous"] = len(ranks) == len(judges) and len(set(ranks)) == 1 and bool(judges)
        row["max_severity"] = next((s for s, r in SEVERITY_RANK.items() if ranks and r == max(ranks)), None)
    ordered = sorted(rows.values(), key=lambda r: (-(SEVERITY_RANK.get(r["max_severity"] or "", -1)), r["id"]))
    matrix = {
        "judges": judges,
        "rows": ordered,
        "never_classified": [r["id"] for r in ordered if judges and len(r["unclassified_by"]) == len(judges)],
        "partially_classified": [r["id"] for r in ordered if 0 < len(r["unclassified_by"]) < len(judges)],
        "contested": [r["id"] for r in ordered if r["contested"]],
        "unknown_ids_cited_by_judges": unknown_refs,
        "ids_of_excluded_refuters_cited_by_judges": excluded_refs,
        "classified_twice": double,
        "excluded": excluded,
        "judges_without_structured_output": [],
        "refuters_without_structured_data": [],
    }
    if rubric_tests:
        table = []
        for t in rubric_tests:
            rulings = {}
            for jid in judges:
                r = next((x for x in (rubric or {}).get(jid) or [] if x.get("test_id") == t["id"]), None)
                if r:
                    rulings[jid] = {"applies": r.get("applies"), "severity": r.get("severity"),
                                    "rationale": str(r.get("rationale") or "")[:400]}
            table.append({"id": t["id"], "text": t["text"], "rulings": rulings,
                          "not_ruled_by": [j for j in judges if j not in rulings]})
        matrix["rubric_tests"] = table
    return matrix


def _flagged(matrix: dict) -> set[str]:
    """IDs the evidence gate of this matrix found not shown (base matrix: 'evidence_gate'; follow-up: 'gate')."""
    return set((matrix.get("evidence_gate") or matrix.get("gate") or {}).get("flagged_ids") or [])


def matrix_markdown(matrix: dict, title: str = "Judgment matrix", followup: bool = False) -> str:
    """The base judgment matrix, or (followup=True) a round's critic items against the adjudicators' rulings."""
    judges = matrix["judges"]
    evidence = any(r.get("evidence") for r in matrix["rows"])
    row = "Item" if followup else "Objection"
    head = f"| {row} | Source | Title | " + " | ".join(judges) + (" | Evidence" if evidence else "") + " | Flags |"
    sep = "|" + "---|" * (4 + len(judges) + (1 if evidence else 0))
    note = ("Computed by the orchestrator from the critic's items and the adjudicators' structured blocks. "
            "Each cell is one adjudicator's own ruling; nothing is averaged.") if followup else (
        "Computed by the orchestrator from the refuters' and judges' structured blocks. "
        "Each cell is one judge's own verdict; nothing is averaged.")
    lines = [f"# {title}", "", note, "", head, sep]
    flagged = _flagged(matrix)
    grouped = False
    for r in matrix["rows"]:
        cells = []
        for j in judges:
            v = r["verdicts"].get(j)
            grouped = grouped or bool(v and v.get("grouped"))
            cells.append(f"{SEVERITY_LABEL.get(v['severity'], v['severity'] or '?')} ({v.get('confidence') or '?'}"
                         + (", grouped)" if v.get("grouped") else ")") if v else "—")
        flags = []
        if r["contested"]:
            flags.append("contested")
        if r["unclassified_by"] and judges:
            flags.append("not classified by " + ",".join(r["unclassified_by"]))
        if r["id"] in flagged:
            flags.append("not shown (evidence gate)")
        title = (r["title"] or "").replace("|", "/")[:90]
        ev = f" | {(r.get('evidence') or '—').replace('|', '/')}" if evidence else ""
        lines.append(f"| {r['id']} | {r['source']} | {title} | " + " | ".join(cells) + ev + f" | {'; '.join(flags)} |")
    never = "Items no adjudicator ruled on" if followup else "Objections no judge classified"
    lines += ["", f"{never}: {', '.join(matrix['never_classified']) or 'none'}",
              f"Contested (verdicts two or more levels apart): {', '.join(matrix['contested']) or 'none'}"]
    if grouped:
        lines.append("Cells marked 'grouped': the severity of a " + ("ruling" if followup else "judgment")
                     + " that covers several IDs and gave this one no severity of its own; it is not necessarily that "
                     + ("adjudicator's" if followup else "judge's") + " view of this ID alone.")
    if matrix["unknown_ids_cited_by_judges"]:
        lines.append("Judge citations of objection IDs that do not exist: "
                     + "; ".join(f"{j}: {', '.join(v)}" for j, v in matrix["unknown_ids_cited_by_judges"].items()))
    gate = matrix.get("evidence_gate")
    if gate is not None:
        lines.append("Evidence gate (judges/evidence_gate.md): serious verdicts on prior work that are not shown: "
                     + (", ".join(gate.get("flagged_ids") or []) or "none"))
    if matrix.get("classified_twice"):
        lines.append("Objections a judge classified more than once (the first verdict is shown): "
                     + "; ".join(f"{j}: {', '.join(v)}" for j, v in matrix["classified_twice"].items()))
    if matrix.get("judges_without_structured_output"):
        lines.append("Judges without structured data, not tabulated (read their reports directly): "
                     + ", ".join(matrix["judges_without_structured_output"]))
    if matrix.get("refuters_without_structured_data"):
        lines.append("Refuters released without structured data, so their objections are not tabulated: "
                     + ", ".join(matrix["refuters_without_structured_data"]))
    if matrix.get("excluded"):
        lines.append("Not included (no usable output): " + "; ".join(matrix["excluded"].values()))
    if matrix.get("ids_of_excluded_refuters_cited_by_judges"):
        lines.append("Judge citations of objections from excluded refuters: "
                     + "; ".join(f"{j}: {', '.join(v)}" for j, v in
                                 matrix["ids_of_excluded_refuters_cited_by_judges"].items()))
    if matrix.get("rubric_tests"):
        lines += ["", "## Rubric tests", "",
                  "Each test the rubric declares, as each judge ruled on it whether or not a refuter raised it "
                  "(applies; severity if it applies). A test that applies with FATAL severity counts like a FATAL "
                  "judgment. An empty cell: the judge gave no ruling (judge prompts before judge_v3 give none).", "",
                  "| Test | Text | " + " | ".join(judges) + " |", "|" + "---|" * (2 + len(judges))]
        for t in matrix["rubric_tests"]:
            cells = []
            for j in judges:
                r = t["rulings"].get(j)
                cells.append("—" if not r else r["applies"] if not r.get("severity")
                             else f"{r['applies']}, {SEVERITY_LABEL.get(r['severity'], r['severity'])}")
            lines.append(f"| {t['id']} | {t['text'].replace('|', '/')[:160]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
