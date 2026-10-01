"""Completion gates: the checks an agent's output must pass before any later stage may read it.

Pure functions over the agent's text, its transcript audit and its provider result: no file I/O and no model
calls. The pipeline saves an output first, runs these checks, and marks the agent `complete` only when the gate
passes (or a person releases it with a recorded reason). A blocking check quarantines the output instead: it
stays on disk, and no later stage reads it.

Structured blocks are recovered before anything is thrown away, cheapest first: strict parse of the last fenced
block, a lenient parse, then the report's own headings (which carry what the judgment matrix needs). Only intake
and novelty reports, whose blocks hold data the headings lack, go to a model repair call, and that call's output
is validated so that it can transcribe but never add content.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from paper_adversary.registry import substituted_models
from paper_adversary.reports import normalize_severity

REFUTER_ROLES = ("novelty", "rigor", "fit")
PASS, WARN, BLOCK = "pass", "warn", "block"

CATEGORIES = {
    "novelty": {"already_done", "partially_anticipated", "incremental", "framing_exists", "gap_not_real",
                "missing_citation", "concurrent_work", "integrity"},
    "rigor": {"math_error", "invalid_assumption", "counterexample", "leakage", "contamination", "control",
              "identifiability", "causal_claim", "evaluation_flaw", "statistics", "failure_case", "missing_ablation",
              "claim_not_established", "integrity"},
    "fit": {"claim_test_mismatch", "benchmark", "compute", "cost", "baseline_fairness", "feasibility",
            "reproducibility", "data", "engineering_assumption", "deployment", "hidden_cost", "integrity"},
}
SEVERITY_ESTIMATES = {"fatal", "major", "minor"}
VERIFIER_VERDICTS = ("anticipates_fully", "anticipates_partially", "does_not_anticipate", "cannot_tell")
ITEM_TYPES = ("new_issue", "minority_critique", "judging_flaw", "synthesis_flaw", "novelty_to_verify")
DISPOSITIONS = ("incorporated", "rejected", "needs_evidence", "unverified_threat", "noted", "invalid")
_ITEM_ID = re.compile(r"^C\d+-I\d+$")
CONFIDENCE = {"high", "medium", "low"}
RESOLVABLE = {"yes", "no", "partially"}

# Parsing runs on untrusted model output, so everything below is line-based and linear in the input size
# (a regex like ```(.*?)``` is quadratic on thousands of unclosed fences).
_FENCE_OPEN = re.compile(r"[ \t]*```[ \t]*([A-Za-z0-9_+-]*)[ \t]*$")
_FENCE_CLOSE = re.compile(r"[ \t]*```[ \t]*$")
_COMMA_CLOSE = re.compile(r"\s*[}\]]")
_LOCAL_ID = re.compile(r"^(?:[A-Z]+\d+-)?(O\d+)$")
_OBJECTION_ID = re.compile(r"^[A-Z]+\d+-O\d+$")
_OBJ_HEAD = re.compile(r"#{3,4}[ \t]+(O\d+)[ \t]*[:.\u2014\u2013-](.*)")
_JUDGE_HEAD = re.compile(r"#{3,4}[ \t]+(FATAL|MAJOR[ _]BUT[ _]FIXABLE|MAJOR|MINOR|NOT[ _]CONVINCING)[ \t]*"
                         r"[:\u2014\u2013-](.*)", re.I)
_NUM_SECTION = re.compile(r"##[ \t]+(\d+)\.(.*)")


@dataclass
class Check:
    check: str
    result: str  # pass | warn | block
    category: str  # isolation | integrity | quality
    message: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Structured:
    data: dict | None
    source: str  # ok | ok:lenient | derived:headings | repaired:syntax | repaired:transcribe | none | n/a
    as_written: str  # "ok", or why the block as the agent wrote it was unusable
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repair_mode: str | None = None  # "syntax" or "transcribe" when a model repair is the next step
    broken_block: str | None = None

    @property
    def usable(self) -> bool:
        return self.data is not None


def verdict(checks: list[Check]) -> str:
    results = {c.result for c in checks}
    return BLOCK if BLOCK in results else (WARN if WARN in results else PASS)


# ---------------------------------------------------------------- JSON parsing


class _Fence:
    """One fenced block: group(1) is its language tag, group(2) its body (like the regex match it replaced)."""
    __slots__ = ("lang", "body", "start", "end")

    def __init__(self, lang: str, body: str, start: int, end: int):
        self.lang, self.body, self.start, self.end = lang, body, start, end

    def group(self, n: int) -> str:
        return self.lang if n == 1 else self.body


def _lines(text: str):
    """(start offset, end offset, line) for every line, in one pass."""
    pos = 0
    while True:
        nl = text.find("\n", pos)
        end = len(text) if nl == -1 else nl
        yield pos, end, text[pos:end]
        if nl == -1:
            return
        pos = nl + 1


def _fences(text: str) -> list[_Fence]:
    text = text.replace("\r\n", "\n")
    out: list[_Fence] = []
    opened: tuple[int, str, int] | None = None  # (fence start, language, body start)
    for start, end, line in _lines(text):
        if opened is None:
            m = _FENCE_OPEN.match(line)
            if m:
                opened = (start, m.group(1), end + 1)
        elif _FENCE_CLOSE.match(line):
            body_start = opened[2]
            out.append(_Fence(opened[1], text[body_start: max(body_start, start - 1)], opened[0], end))
            opened = None
    return out


def parse_json_strict(text: str) -> tuple[dict | None, str | None]:
    """The report's last fenced block, which must be a JSON object. Earlier blocks are never used: one of them
    could be a snippet quoted from the paper."""
    blocks = _fences(text)
    if not blocks:
        return None, "no fenced JSON block found"
    lang, body = blocks[-1].group(1).lower(), blocks[-1].group(2).strip()
    if lang not in ("", "json"):
        return None, f"the last fenced block is '{lang}', not JSON"
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "the JSON block is not an object"
    return data, None


def _top_level_objects(text: str) -> list[str]:
    out, depth, start, in_str, esc = [], 0, 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"' and depth:
            in_str = True
        elif ch == "{":
            if not depth:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if not depth:
                out.append(text[start : i + 1])
    return out


def _strip_trailing_commas(raw: str) -> str:
    out, in_str, esc = [], False, False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "," and _COMMA_CLOSE.match(raw, i + 1):
            continue
        out.append(ch)
    return "".join(out)


def parse_json_lenient(text: str) -> tuple[dict | None, str | None]:
    """Fence variants, an unfenced final object, trailing commas."""
    text = text.replace("\r\n", "\n")
    candidates = []
    blocks = _fences(text)
    if blocks:
        candidates.append(blocks[-1].group(2))
    objects = _top_level_objects(text)
    if objects:
        candidates.append(objects[-1])
    for raw in candidates:
        for variant in (raw, _strip_trailing_commas(raw)):
            try:
                data = json.loads(variant)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data, None
    return None, "no parseable JSON object"


def json_shape(prompt_body: str) -> str:
    """The JSON shape an agent was asked for: the last ```json fence in its prompt."""
    blocks = [m for m in _fences(prompt_body) if m.group(1).lower() == "json"]
    return blocks[-1].group(2).strip() if blocks else ""


# ---------------------------------------------------------------- schemas


def _norm_ids(value) -> list[str]:
    ids = value if isinstance(value, list) else [value] if value else []
    return [re.sub(r"\s+", "", str(x)).upper() for x in ids if x]


def validate_block(role: str, data: dict) -> tuple[list[str], list[str]]:
    """(hard errors, warnings). Hard rules cover only fields the pipeline consumes. Never raises: a value of an
    unexpected type is an error to report, not a crash."""
    try:
        return _validate_block(role, data)
    except Exception as exc:  # e.g. a list where a string belongs
        return [f"the structured block has an unexpected shape ({type(exc).__name__}: {exc})"[:300]], []


def _text(value) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _validate_block(role: str, data: dict) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    if role in REFUTER_ROLES:
        objs = data.get("objections")
        if not isinstance(objs, list):
            return ["'objections' is missing or not a list"], soft
        seen: set[str] = set()
        for i, obj in enumerate(objs, start=1):
            if not isinstance(obj, dict):
                hard.append(f"objection {i} is not an object")
                continue
            raw = str(obj.get("id") or "").strip().upper()
            m = _LOCAL_ID.match(raw)
            if not m:
                hard.append(f"objection {i}: id {raw or '(none)'!r} is not O<n>")
                continue
            if m.group(1) in seen:
                hard.append(f"objection id {m.group(1)} is used twice")
            seen.add(m.group(1))
            if not str(obj.get("title") or "").strip():
                soft.append(f"{m.group(1)} has no title")
            if _text(obj.get("severity_estimate")) not in SEVERITY_ESTIMATES:
                soft.append(f"{m.group(1)}: severity_estimate {obj.get('severity_estimate')!r} is not fatal/major/minor")
            if _text(obj.get("confidence")) not in CONFIDENCE:
                soft.append(f"{m.group(1)}: confidence {obj.get('confidence')!r} is not high/medium/low")
            if _text(obj.get("category")) not in CATEGORIES[role]:
                soft.append(f"{m.group(1)}: unknown category {obj.get('category')!r}")
            refs = obj.get("references")
            if refs is not None and not isinstance(refs, list):
                soft.append(f"{m.group(1)}: references is not a list")
            for ref in refs if isinstance(refs, list) else []:
                if not (isinstance(ref, dict) and (ref.get("title") or ref.get("doi") or ref.get("arxiv_id"))):
                    soft.append(f"{m.group(1)}: a reference has no title, DOI or arXiv ID")
            pairs = obj.get("overlap_evidence")
            if pairs is not None and not isinstance(pairs, list):
                soft.append(f"{m.group(1)}: overlap_evidence is not a list")
    elif role == "judge":
        items = data.get("judgments")
        if not isinstance(items, list):
            return ["'judgments' is missing or not a list"], soft
        for i, j in enumerate(items, start=1):
            if not isinstance(j, dict):
                hard.append(f"judgment {i} is not an object")
                continue
            ids = _norm_ids(j.get("objection_ids") or j.get("objection_id"))
            if not ids:
                hard.append(f"judgment {i} names no objection IDs")
            bad = [x for x in ids if not _OBJECTION_ID.match(x)]
            if bad:
                hard.append(f"judgment {i}: {', '.join(bad)} are not objection IDs like N1-O3")
            if normalize_severity(j.get("severity")) is None:
                hard.append(f"judgment {i}: severity {j.get('severity')!r} is not one of the four classes")
            if _text(j.get("confidence")) not in CONFIDENCE:
                soft.append(f"judgment {i}: confidence {j.get('confidence')!r} is not high/medium/low")
            if _text(j.get("resolvable_with_more_evidence")) not in RESOLVABLE:
                soft.append(f"judgment {i}: resolvable_with_more_evidence is not yes/no/partially")
    elif role in ("critic", "recheck"):
        items = data.get("items")
        if not isinstance(items, list):
            return ["'items' is missing or not a list"], soft
        seen: set[str] = set()
        for i, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                hard.append(f"item {i} is not an object")
                continue
            local = str(item.get("id") or "").strip().upper()
            local = re.sub(r"^C\d+-", "", local)
            if not re.fullmatch(r"I\d+", local):
                hard.append(f"item {i}: id {item.get('id')!r} is not I<n>")
            elif local in seen:
                hard.append(f"item id {local} is used twice")
            seen.add(local)
            if _text(item.get("type")) not in ITEM_TYPES:
                hard.append(f"item {i}: type {item.get('type')!r} is not one of {ITEM_TYPES}")
            for key in ("objection_ids", "judge_ids", "claim_ids", "candidate_references"):
                if item.get(key) is not None and not isinstance(item.get(key), (list, str)):
                    soft.append(f"item {i}: {key} is not a list")
            if normalize_severity(item.get("severity_estimate")) is None:
                soft.append(f"item {i}: severity_estimate {item.get('severity_estimate')!r} is not a severity")
            if item.get("type") == "novelty_to_verify" and not item.get("candidate_references"):
                soft.append(f"item {i}: a novelty_to_verify item names no candidate reference")
    elif role == "adjudicator":
        rulings = data.get("rulings")
        if not isinstance(rulings, list):
            return ["'rulings' is missing or not a list"], soft
        for i, r in enumerate(rulings, start=1):
            if not isinstance(r, dict):
                hard.append(f"ruling {i} is not an object")
                continue
            ids = _norm_ids(r.get("item_ids") or r.get("item_id"))
            if not ids or [x for x in ids if not _ITEM_ID.match(x)]:
                hard.append(f"ruling {i}: item_ids must be item IDs like C1-I3")
            if normalize_severity(r.get("severity")) is None:
                hard.append(f"ruling {i}: severity {r.get('severity')!r} is not one of the four classes")
    elif role == "revision":
        items = data.get("item_dispositions")
        if not isinstance(items, list):
            return ["'item_dispositions' is missing or not a list"], soft
        for i, d in enumerate(items, start=1):
            if not isinstance(d, dict) or not _ITEM_ID.match(str(d.get("item_id") or "").strip().upper()):
                hard.append(f"disposition {i} has no item_id like C1-I3")
            elif _text(d.get("disposition")) not in DISPOSITIONS:
                hard.append(f"{d['item_id']}: disposition {d.get('disposition')!r} is not one of {DISPOSITIONS}")
    elif role == "verifier":
        items = data.get("passages")
        if not isinstance(items, list) or not items:
            return ["'passages' is missing or empty"], soft
        for i, item in enumerate(items, start=1):
            if not isinstance(item, dict) or not re.fullmatch(r"P\d+", str(item.get("passage_id") or "")):
                hard.append(f"passage {i} has no passage_id like P1")
            elif _text(item.get("verdict")) not in VERIFIER_VERDICTS:
                hard.append(f"{item['passage_id']}: verdict {item.get('verdict')!r} is not one of {VERIFIER_VERDICTS}")
    elif role == "intake":
        claims = data.get("claims")
        if not isinstance(claims, list) or not claims:
            return ["'claims' is missing or empty"], soft
        for i, c in enumerate(claims, start=1):
            if not (isinstance(c, dict) and c.get("id") and c.get("claim")):
                hard.append(f"claim {i} needs an id and a claim")
    return hard, soft


# ---------------------------------------------------------------- headings


def _strip_fenced(text: str) -> str:
    """The text with fenced blocks blanked out (line count kept), so headings inside them are ignored."""
    text = text.replace("\r\n", "\n")
    parts, pos = [], 0
    for f in _fences(text):
        parts += [text[pos: f.start], "\n" * text.count("\n", f.start, f.end)]
        pos = f.end
    parts.append(text[pos:])
    return "".join(parts)


def objection_headings(text: str) -> list[dict]:
    out = []
    for _, _, line in _lines(_strip_fenced(text)):
        m = _OBJ_HEAD.match(line)
        if m and m.group(2).strip():
            out.append({"id": m.group(1).upper(), "title": m.group(2).strip()})
    return out


def judgment_headings(text: str) -> list[dict]:
    out = []
    for _, _, line in _lines(_strip_fenced(text)):
        m = _JUDGE_HEAD.match(line)
        rest = m.group(2).strip() if m else ""
        if not rest.endswith(")") or "(" not in rest:
            continue
        cut = rest.rfind("(")  # IDs are the last parenthesised group; titles may contain parentheses
        ids = [x for x in _norm_ids(re.split(r"[,;]", rest[cut + 1: -1])) if _OBJECTION_ID.match(x)]
        if ids:
            out.append({"objection_ids": ids, "title": rest[:cut].strip(), "severity": normalize_severity(m.group(1))})
    return out


def numbered_sections(text: str) -> list[tuple[int, str]]:
    return _numbered(_strip_fenced(text))


def _numbered(text: str) -> list[tuple[int, str]]:
    out = []
    for _, _, line in _lines(text):
        m = _NUM_SECTION.match(line)
        if m and m.group(2).strip():
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def derive_from_headings(role: str, text: str) -> dict | None:
    """What the matrix needs (IDs, titles, judge severities), rebuilt from the report's own headings."""
    if role in REFUTER_ROLES:
        objs = objection_headings(text)
        return {"objections": [{"id": o["id"], "title": o["title"], "origin": "heading"} for o in objs]} if objs else None
    if role == "judge":
        js = judgment_headings(text)
        return {"judgments": [{**j, "origin": "heading"} for j in js]} if js else None
    return None


def reconcile_objections(data: dict, text: str) -> list[str]:
    """Judges cite the IDs they read in the Markdown, so every ID from either source is kept (in place).
    Titles may legitimately differ between a heading and the block; only IDs are compared."""
    heads = {h["id"]: h["title"] for h in objection_headings(text)}
    objs = data.get("objections") or []
    block_ids = {_LOCAL_ID.match(str(o.get("id") or "").strip().upper()).group(1) for o in objs
                 if isinstance(o, dict) and _LOCAL_ID.match(str(o.get("id") or "").strip().upper())}
    warnings = []
    only_heads = [h for h in heads if h not in block_ids]
    only_block = sorted(block_ids - set(heads)) if heads else []
    for oid in only_heads:
        objs.append({"id": oid, "title": heads[oid], "origin": "heading"})
    for o in objs:
        if isinstance(o, dict) and not str(o.get("title") or "").strip():
            local = _LOCAL_ID.match(str(o.get("id") or "").strip().upper())
            if local and local.group(1) in heads:
                o["title"] = heads[local.group(1)]
    if only_heads:
        warnings.append(f"objections in the headings but not the JSON block (added): {', '.join(only_heads)}")
    if only_block:
        warnings.append(f"objections in the JSON block without a heading: {', '.join(only_block)}")
    data["objections"] = objs
    return warnings


def compare_judgments(data: dict, text: str) -> list[str]:
    heads = judgment_headings(text)
    if not heads:
        return []
    by_id = {oid: j.get("severity") for j in heads for oid in j["objection_ids"]}
    block = {oid: normalize_severity(j.get("severity")) for j in data.get("judgments") or [] if isinstance(j, dict)
             for oid in _norm_ids(j.get("objection_ids") or j.get("objection_id"))}
    warnings = []
    if set(by_id) - set(block):
        warnings.append(f"judged in the headings but not the JSON block: {', '.join(sorted(set(by_id) - set(block)))}")
    differ = sorted(oid for oid in set(by_id) & set(block) if by_id[oid] != block[oid])
    if differ:
        warnings.append(f"heading and JSON severities differ for {', '.join(differ)} (the JSON block is used)")
    return warnings


# ---------------------------------------------------------------- structured-block recovery


def parse_structured(role: str, text: str, model_repair_first: bool) -> Structured:
    """Strict, then lenient, then (repair or) headings. `model_repair_first` marks roles whose blocks hold data
    the headings lack (intake claims, novelty references): those get a model repair before any derivation."""
    data, err = parse_json_strict(text)
    as_written = "ok"
    if data is not None:
        hard, soft = validate_block(role, data)
        if not hard:
            return _finish(role, text, data, "ok", as_written, soft)
        as_written = "invalid: " + "; ".join(hard[:3])
    else:
        as_written = err or "unparseable"
    lenient, _ = parse_json_lenient(text)
    if lenient is not None:
        hard, soft = validate_block(role, lenient)
        if not hard:
            return _finish(role, text, lenient, "ok:lenient", as_written, soft)
    blocks = _fences(text)
    broken = blocks[-1].group(2) if blocks else None
    if model_repair_first:
        return Structured(None, "none", as_written, [as_written], [], repair_mode="syntax" if broken else "transcribe",
                          broken_block=broken)
    return fallback(role, text, as_written)


def fallback(role: str, text: str, as_written: str, note: str | None = None) -> Structured:
    derived = derive_from_headings(role, text)
    if derived is not None:
        warning = "the structured block was unusable; IDs, titles and severities were rebuilt from the headings"
        return Structured(derived, "derived:headings", as_written, [], [warning] + ([note] if note else []))
    return Structured(None, "none", as_written, [f"structured block unusable ({as_written}) and no headings to "
                                                  "rebuild it from"], [])


def _finish(role: str, text: str, data: dict, source: str, as_written: str, warnings: list[str]) -> Structured:
    if role in REFUTER_ROLES:
        warnings = warnings + reconcile_objections(data, text)
    elif role == "judge":
        warnings = warnings + compare_judgments(data, text)
    return Structured(data, source, as_written, [], warnings)


def accept_repair(role: str, text: str, as_written: str, data: dict, mode: str, warnings: list[str]) -> Structured:
    return _finish(role, text, data, f"repaired:{mode}", as_written,
                   [f"the structured block was rebuilt by a format-repair call ({mode})"] + warnings)


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _squash(text: str) -> str:
    text = text.replace('\\"', '"').replace("\\n", " ").replace("\\t", " ")
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_repair(role: str, mode: str, repaired_text: str, report_text: str,
                    broken_block: str | None) -> tuple[dict | None, list[str], list[str]]:
    """(data, rejections, warnings). A repair may transcribe or fix syntax, never add content."""
    data, err = parse_json_strict(repaired_text)
    if data is None:
        return None, [f"repair output: {err}"], []
    hard, soft = validate_block(role, data)
    if hard:
        return None, [f"repair output: {h}" for h in hard], []
    rejections: list[str] = []
    if mode == "syntax" and broken_block:
        # Every value must already be in the agent's own output (the broken block or, for entries it lost to a
        # cut-off, the report text around it).
        source = _squash(report_text + "\n" + broken_block)
        invented = [s for s in _strings(data) if len(s.strip()) > 3 and _squash(s) not in source]
        if invented:
            rejections.append(f"repair output contains text that is not in the agent's report: {invented[0][:80]!r}")
    if role in REFUTER_ROLES:
        heads = {h["id"]: h["title"] for h in objection_headings(report_text)}
        got = {}
        for o in data.get("objections") or []:
            m = _LOCAL_ID.match(str(o.get("id") or "").strip().upper())
            if m:
                got[m.group(1)] = str(o.get("title") or "")
        if heads and set(got) != set(heads):
            rejections.append(f"repair output objection IDs {sorted(got)} differ from the headings {sorted(heads)}")
        if mode == "transcribe":
            for oid, title in got.items():
                if oid in heads and _squash(title) != _squash(heads[oid]):
                    rejections.append(f"{oid}: title {title!r} is not the heading's title")
            report = _squash(report_text)
            for o in data.get("objections") or []:
                for ref in o.get("references") or []:
                    for key in ("title", "doi", "arxiv_id"):
                        value = str(ref.get(key) or "").strip()
                        if value and _squash(value) not in report:
                            rejections.append(f"{o.get('id')}: reference {key} {value!r} does not appear in the report")
    return (None if rejections else data), rejections, soft


def repair_user_text(role: str, report_text: str, shape: str, mode: str, problem: str,
                     broken_block: str | None) -> str:
    """The format-repair call's input: the agent's own report and nothing else it did not write."""
    if role in REFUTER_ROLES:
        heads = [f"{h['id']}: {h['title']}" for h in objection_headings(report_text)]
    elif role == "judge":
        heads = [f"{j['severity']}: {j['title']} ({', '.join(j['objection_ids'])})" for j in judgment_headings(report_text)]
    else:
        heads = []
    task = ("Fix the JSON block at the end of the report: repair its syntax, and if it was cut off, complete the "
            "missing entries from the report itself, changing no existing content."
            if mode == "syntax" and broken_block else
            "Transcribe the report into the required JSON shape, copying IDs, titles and references verbatim.")
    return (f"<report>\n{report_text.strip()}\n</report>\n\n<required_shape>\n{shape}\n</required_shape>\n\n"
            f"<headings>\n{chr(10).join(heads) or '(none found)'}\n</headings>\n\n"
            f"<problem>\n{problem}\n</problem>\n\n<assignment>\n{task} Output exactly one fenced JSON block and "
            "nothing else.\n</assignment>")


# ---------------------------------------------------------------- other checks


def check_audit(audit: dict, unverifiable_policy: str) -> Check:
    status = audit.get("status")
    if status == "fail":
        return Check("isolation_audit", BLOCK, "isolation", "isolation audit failed: " + "; ".join(
            audit.get("findings", [])[:3]), {"findings": audit.get("findings", [])})
    if status == "unverifiable":
        result = BLOCK if unverifiable_policy == "quarantine" else WARN
        return Check("isolation_audit", result, "isolation", "isolation could not be verified: " + "; ".join(
            audit.get("unverifiable", [])[:3]), {"unverifiable": audit.get("unverifiable", [])})
    if audit.get("denied"):
        return Check("isolation_audit", WARN, "isolation", "the sandbox refused tool calls (possible prompt "
                     "injection in the submission): " + "; ".join(audit["denied"][:3]), {"denied": audit["denied"]})
    return Check("isolation_audit", PASS, "isolation")


def check_substitution(requested: str, served: list[str], policy: str) -> Check:
    other = substituted_models(requested, served)
    if not other:
        return Check("model", PASS, "integrity")
    return Check("model", BLOCK if policy == "block" else WARN, "integrity",
                 f"requested {requested} but turns were served by {', '.join(other)}", {"served": served})


def check_truncation(stop_reason: str | None, block_intact: bool, policy: str) -> Check:
    if stop_reason != "max_tokens":
        return Check("truncation", PASS, "quality")
    if block_intact:
        return Check("truncation", WARN, "quality", "the final message hit the output-token limit, but its "
                     "structured block is complete")
    return Check("truncation", BLOCK if policy == "block" else WARN, "quality",
                 "the report was cut off at the output-token limit")


def check_structured(st: Structured) -> Check:
    if st.source == "n/a":
        return Check("structured_block", PASS, "quality")
    if not st.usable:
        return Check("structured_block", BLOCK, "quality", "structured block unusable: " + "; ".join(st.errors[:2]),
                     {"as_written": st.as_written})
    result = PASS if st.source == "ok" and not st.warnings else WARN
    bits = ([] if st.source == "ok" else [f"structured data source: {st.source} (as written: {st.as_written})"]) \
        + [str(w) for w in st.warnings[:3]]
    message = "; ".join(bits)
    return Check("structured_block", result, "quality", message,
                 {"source": st.source, "as_written": st.as_written, "warnings": st.warnings})


def check_coverage(judgments: list[dict], expected: set[str], min_fraction: float) -> Check:
    counts: dict[str, int] = {}
    for j in judgments:
        for oid in j.get("objection_ids") or []:
            counts[oid] = counts.get(oid, 0) + 1
    covered = set(counts) & expected
    missing = sorted(expected - covered)
    unknown = sorted(set(counts) - expected)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    fraction = len(covered) / len(expected) if expected else 1.0
    details = {"expected": len(expected), "classified": len(covered), "fraction": round(fraction, 3),
               "missing": missing, "unknown": unknown, "duplicates": duplicates}
    if expected and fraction < min_fraction:
        return Check("judge_coverage", BLOCK, "quality", f"classified only {len(covered)} of {len(expected)} "
                     f"objections (minimum {min_fraction:.0%})", details)
    if missing or unknown or duplicates:
        bits = ([f"{len(missing)} not classified"] if missing else []) + \
               ([f"unknown IDs {', '.join(unknown[:5])}"] if unknown else []) + \
               ([f"classified twice: {', '.join(sorted(duplicates)[:5])}"] if duplicates else [])
        return Check("judge_coverage", WARN, "quality", "; ".join(bits), details)
    return Check("judge_coverage", PASS, "quality", "", details)


def required_sections(prompt_body: str) -> list[tuple[int, str]]:
    """The numbered sections an agent's prompt asks for (from its Output format part)."""
    part = prompt_body.split("# Output format", 1)[-1]
    return _numbered(part)


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def check_sections(text: str, required: list[tuple[int, str]], mode: str) -> Check:
    if not required:
        return Check("sections", PASS, "quality")
    found = dict(numbered_sections(text))
    missing = [n for n, _ in required if n not in found]
    renamed = [n for n, t in required if n in found and _title_key(found[n]) != _title_key(t)]
    last = max(n for n, _ in required)
    details = {"required": len(required), "missing": missing, "renamed": renamed}
    if mode == "block_any" and missing:
        result = BLOCK
    elif mode == "block_incomplete" and (not found or last not in found):
        result = BLOCK
    else:
        result = WARN if missing or renamed else PASS
    message = ""
    if missing:
        message = f"missing sections {', '.join(map(str, missing))}" + (" (the report ends early)"
                                                                          if last in missing else "")
    elif renamed:
        message = f"sections {', '.join(map(str, renamed))} have different titles than the prompt asks for"
    return Check("sections", result, "quality", message, details)


def section_text(text: str, number: int) -> str:
    """The body of the memo section numbered `number` ('## 6. ...'), up to the next numbered section."""
    body = _strip_fenced(text)
    m = re.search(rf"^##[ \t]+{number}\.[^\n]*\n(.*?)(?=^##[ \t]+\d+\.|\Z)", body, re.M | re.S)
    return m.group(1) if m else ""


def check_unverified_placement(text: str, required: list[tuple[int, str]], flagged: list[str],
                               block: bool) -> Check:
    """Gated (unverified) objections must sit under 'Unverified threats', not 'Criticisms that survived judging'."""
    survived = next((n for n, t in required if _title_key(t).startswith("criticisms that survived")), None)
    unverified = next((n for n, t in required if _title_key(t).startswith("unverified threats")), None)
    if not flagged or survived is None or unverified is None:
        return Check("unverified_placement", PASS, "quality")
    ids = set(flagged)
    misplaced = sorted(i for i in ids if re.search(rf"\b{re.escape(i)}\b", section_text(text, survived)))
    absent = sorted(i for i in ids if not re.search(rf"\b{re.escape(i)}\b", section_text(text, unverified)))
    if not misplaced and not absent:
        return Check("unverified_placement", PASS, "quality")
    bits = ([f"unverified objections listed as surviving criticisms: {', '.join(misplaced)}"] if misplaced else []) + \
           ([f"unverified objections missing from 'Unverified threats': {', '.join(absent)}"] if absent else [])
    return Check("unverified_placement", BLOCK if block and misplaced else WARN, "quality", "; ".join(bits),
                 {"misplaced": misplaced, "absent": absent})


def unavailable_label(agent_id: str, entry: dict) -> str:
    """How a missing input is named to downstream agents: never its content, only why it is missing."""
    status = entry.get("status", "?")
    if status == "quarantined":
        reason = ((entry.get("gate") or {}).get("reasons") or ["failed its checks"])[0]
        return f"{agent_id} (quarantined: {reason[:120]})"
    return f"{agent_id} ({status})"
