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

_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\n(.*?)\n```", re.S)


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


def extract_json_block(text: str) -> tuple[dict | None, str | None]:
    """Return the last fenced JSON object in the text, or (None, reason)."""
    blocks = _FENCE.findall(text)
    if not blocks:
        return None, "no fenced JSON block found"
    last_error = "unparseable JSON"
    for raw in reversed(blocks):
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON: {exc}"
            continue
        if isinstance(data, dict):
            return data, None
    return None, last_error


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
        out.append({**j, "objection_ids": ids, "severity": normalize_severity(j.get("severity"))})
    return out


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


def build_judgment_matrix(objections: dict[str, list[dict]], judgments: dict[str, list[dict]]) -> dict:
    """Cross-tabulate every refuter objection against every judge's verdict.

    Verdicts are kept per judge, never averaged. Flags mark objections no judge
    classified, objections some judges skipped, and contested objections.
    """
    known = {o["id"]: o for objs in objections.values() for o in objs}
    rows: dict[str, dict] = {
        oid: {"id": oid, "source": o["source"], "title": o.get("title") or o.get("objection") or "",
              "refuter_severity": o.get("severity_estimate"), "verdicts": {}}
        for oid, o in known.items()
    }
    unknown_refs: dict[str, list[str]] = {}
    for jid, items in judgments.items():
        for item in items:
            for oid in item["objection_ids"]:
                if oid not in rows:
                    unknown_refs.setdefault(jid, []).append(oid)
                    continue
                rows[oid]["verdicts"][jid] = {"severity": item.get("severity"),
                                              "confidence": item.get("confidence"),
                                              "resolvable": item.get("resolvable_with_more_evidence")}
    judges = sorted(judgments)
    for row in rows.values():
        ranks = [SEVERITY_RANK[v["severity"]] for v in row["verdicts"].values() if v.get("severity")]
        row["unclassified_by"] = [j for j in judges if j not in row["verdicts"]]
        row["contested"] = bool(ranks) and (max(ranks) - min(ranks) >= 2)
        row["unanimous"] = len(ranks) == len(judges) and len(set(ranks)) == 1 and bool(judges)
        row["max_severity"] = next((s for s, r in SEVERITY_RANK.items() if ranks and r == max(ranks)), None)
    ordered = sorted(rows.values(), key=lambda r: (-(SEVERITY_RANK.get(r["max_severity"] or "", -1)), r["id"]))
    return {
        "judges": judges,
        "rows": ordered,
        "never_classified": [r["id"] for r in ordered if judges and len(r["unclassified_by"]) == len(judges)],
        "partially_classified": [r["id"] for r in ordered if 0 < len(r["unclassified_by"]) < len(judges)],
        "contested": [r["id"] for r in ordered if r["contested"]],
        "unknown_ids_cited_by_judges": unknown_refs,
        "judges_without_structured_output": [],
    }


def matrix_markdown(matrix: dict) -> str:
    judges = matrix["judges"]
    head = "| Objection | Source | Title | " + " | ".join(judges) + " | Flags |"
    sep = "|" + "---|" * (4 + len(judges))
    lines = [
        "# Judgment matrix",
        "",
        "Computed by the orchestrator from the refuters' and judges' structured blocks. "
        "Each cell is one judge's own verdict; nothing is averaged.",
        "",
        head,
        sep,
    ]
    for r in matrix["rows"]:
        cells = []
        for j in judges:
            v = r["verdicts"].get(j)
            cells.append(f"{SEVERITY_LABEL.get(v['severity'], v['severity'] or '?')} ({v.get('confidence') or '?'})"
                         if v else "—")
        flags = []
        if r["contested"]:
            flags.append("contested")
        if r["unclassified_by"] and judges:
            flags.append("not classified by " + ",".join(r["unclassified_by"]))
        title = (r["title"] or "").replace("|", "/")[:90]
        lines.append(f"| {r['id']} | {r['source']} | {title} | " + " | ".join(cells) + f" | {'; '.join(flags)} |")
    lines += ["", f"Objections no judge classified: {', '.join(matrix['never_classified']) or 'none'}",
              f"Contested (verdicts two or more levels apart): {', '.join(matrix['contested']) or 'none'}"]
    if matrix["unknown_ids_cited_by_judges"]:
        lines.append("Judge citations of objection IDs that do not exist: "
                     + "; ".join(f"{j}: {', '.join(v)}" for j, v in matrix["unknown_ids_cited_by_judges"].items()))
    if matrix.get("judges_without_structured_output"):
        lines.append("Judges whose structured block could not be parsed (read their reports directly): "
                     + ", ".join(matrix["judges_without_structured_output"]))
    return "\n".join(lines) + "\n"
