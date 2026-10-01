"""Context isolation, enforced in code rather than by prompting.

Three layers:

1. Access policy. Context builders read prior outputs only through
   ArtifactAccess, which refuses any artifact kind the role may not see
   (ROLE_VISIBILITY). The policy lives here, not in the config, so a config
   change cannot loosen it.
2. Pre-flight guard. Before an agent starts, IsolationGuard scans the fully
   assembled prompt for fingerprints of every existing artifact the role must
   not see (an embedded marker plus distinctive word 8-grams). A hit aborts the
   agent with IsolationViolation.
3. Transcript audit. After an agent finishes, its transcript is checked for
   file reads outside its sandbox and for forbidden fingerprints in tool
   results. The verdict is stored in the agent's metadata.

On top of this, every agent is a separate process in an empty sandbox folder
with no shell and no file access outside that folder (see providers/claude_code).
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from paper_adversary.util import atomic_write_json, read_json, read_jsonl, sha256_text, utcnow

# kind -> which roles may read it. "paper", "rubric" and "prior_text" (third-party full texts) are inputs, not
# agent outputs; "evidence" (quote checks) and "vtask" (a verifier's passages) are computed by the orchestrator.
ROLE_VISIBILITY: dict[str, frozenset[str]] = {
    "intake": frozenset({"paper"}),
    "novelty": frozenset({"paper"}),
    "rigor": frozenset({"paper"}),
    "fit": frozenset({"paper"}),
    "verifier": frozenset({"paper", "prior_text", "vtask"}),
    "judge": frozenset({"paper", "rubric", "novelty", "rigor", "fit", "refcheck", "evidence", "verification",
                        "prior_text"}),
    "synthesis": frozenset({"paper", "rubric", "novelty", "rigor", "fit", "refcheck", "evidence", "verification",
                            "prior_text", "judge", "matrix", "profile"}),
    "critic": frozenset({"paper", "rubric", "novelty", "rigor", "fit", "refcheck", "evidence", "verification",
                         "prior_text", "judge", "matrix", "profile", "synthesis"}),
}

# Follow-up rounds: adjudicators, the revised memo and the re-check critic see the whole review plus the round's
# own artifacts ("followup" = the orchestrator's triage and follow-up matrix). Who sees whom inside a round is
# decided by position (below): only strictly earlier steps.
_FOLLOWUP_KINDS = frozenset({"critic", "recheck", "adjudication", "revision", "followup"})
for _role in ("adjudicator", "revision", "recheck"):
    ROLE_VISIBILITY[_role] = ROLE_VISIBILITY["critic"] | _FOLLOWUP_KINDS

# Agent outputs and the artifact kind they produce.
OUTPUT_KIND = {"intake": "profile", "novelty": "novelty", "rigor": "rigor", "fit": "fit", "verifier": "verification",
               "judge": "judge", "synthesis": "synthesis", "critic": "critic", "adjudicator": "adjudication",
               "revision": "revision", "recheck": "recheck"}
KIND_ROLE = {"profile": "intake", "verification": "verifier", "adjudication": "adjudicator"}

# Order of steps: an artifact is readable only if it comes strictly before the reader. Base review is round 0.
BASE_POS = {"intake": 0, "novelty": 0, "rigor": 0, "fit": 0, "verifier": 1, "judge": 2, "synthesis": 3, "critic": 4}
FOLLOWUP_POS = {"verifier": 0, "adjudicator": 1, "revision": 2, "recheck": 3}


def position(role: str, rnd: int = 0) -> tuple[int, int]:
    return (rnd, (BASE_POS if rnd == 0 else FOLLOWUP_POS).get(role, 0))

MARKER_RE = re.compile(r"<!-- aid:([A-Z0-9]+):([0-9a-f]{12}) -->")
_WORD = re.compile(r"[a-z0-9]+")
SHINGLE = 8
MAX_SHINGLES = 24
HIT_THRESHOLD = 3


class IsolationViolation(RuntimeError):
    pass


# Credentials must never reach an agent's prompt (and from there a model, a report or a log). Papers about LLM
# tooling print placeholders ("sk-ant-api03-XXXX...", AKIAIOSFODNN7EXAMPLE, "-----BEGIN PRIVATE KEY-----" alone),
# so a match counts only if the value looks random; the values this process actually holds always count.
def _random_looking(value: str, need_digit: bool = True) -> bool:
    value = value.strip("\"'`")
    if len(value) < 16 or len(set(value)) < 10 or "example" in value.lower():
        return False
    if re.search(r"(.)\1{5}", value):  # XXXXXX, 000000: a placeholder
        return False
    return any(c.isalpha() for c in value) and (any(c.isdigit() for c in value) or not need_digit)


_SECRET_PATTERNS = (
    ("an Anthropic API key or OAuth token", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),
     lambda m: _random_looking(re.sub(r"^sk-ant-(?:[a-z]{3,6}\d{2}-)?", "", m.group(0)))),
    ("a credential assignment", re.compile(r"\b(?:CLAUDE_CODE_OAUTH_TOKEN|ANTHROPIC_API_KEY|ANTHROPIC_AUTH_TOKEN|"
                                           r"S2_API_KEY|OPENALEX_API_KEY)[ \t]*[=:][ \t]*(\S{8,})"),
     lambda m: _random_looking(m.group(1))),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----\s*(?:[A-Za-z-]{1,40}:[^\n]{0,300}\n\s*)*"
                                 r"[A-Za-z0-9+/]{40,}"), None),
    ("an AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), lambda m: _random_looking(m.group(0), need_digit=False)),
    ("a GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}"),
     lambda m: _random_looking(re.sub(r"^(?:gh[pous]_|github_pat_)", "", m.group(0)))),
)
SECRET_ENV_VARS = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "S2_API_KEY",
                   "OPENALEX_API_KEY")


def find_secrets(text: str) -> list[str]:
    """Labels of anything in `text` that looks like a credential, including the values this process holds."""
    found = [label for label, rx, real in _SECRET_PATTERNS
             if any(real is None or real(m) for m in rx.finditer(text))]
    for var in SECRET_ENV_VARS:
        value = (os.environ.get(var) or "").strip()
        if len(value) >= 12 and value in text:
            found.append(f"the value of {var}")
    return found


def new_marker(agent_id: str) -> str:
    return f"<!-- aid:{agent_id}:{secrets.token_hex(6)} -->"


def _shingles(text: str) -> list[str]:
    words = _WORD.findall(text.lower())
    return [" ".join(words[i : i + SHINGLE]) for i in range(0, max(0, len(words) - SHINGLE + 1))]


def distinctive_shingles(body: str, exclude: set[str]) -> list[str]:
    """Sample 8-grams unique to this report (not in the paper or prompt templates)."""
    candidates = []
    seen: set[str] = set()
    for sh in _shingles(body):
        if sh in exclude or sh in seen:
            continue
        seen.add(sh)
        candidates.append(sh)
    if len(candidates) <= MAX_SHINGLES:
        return candidates
    step = len(candidates) / MAX_SHINGLES
    return [candidates[int(i * step)] for i in range(MAX_SHINGLES)]


@dataclass
class ReportRecord:
    agent_id: str
    role: str
    path: Path
    meta: dict
    body: str
    sha256: str


class ArtifactAccess:
    """Policy-checked reader handed to context builders. Records a manifest of what was read."""

    def __init__(self, store, role: str, agent_id: str, pos: tuple[int, int] | None = None):
        if role not in ROLE_VISIBILITY:
            raise IsolationViolation(f"unknown role '{role}'")
        self.store = store
        self.role = role
        self.agent_id = agent_id
        self.visible = ROLE_VISIBILITY[role]
        self.pos = pos  # the reader's (round, step); reports from the same or a later step stay hidden
        self.manifest: list[dict] = []
        self.texts: list[str] = []  # everything handed out through this reader (used by the guard's allowance)

    def _require(self, kind: str) -> None:
        if kind not in self.visible:
            raise IsolationViolation(f"{self.agent_id} ({self.role}) may not read '{kind}' artifacts")

    def note_input(self, kind: str, path: Path | None, sha256: str, agent_id: str | None = None,
                   hash_mode: str = "text") -> None:
        """hash_mode says how sha256 was computed: 'text' (whole file), 'body' (report minus front matter), 'file' (bytes)."""
        self._require(kind)
        self.manifest.append({"kind": kind, "agent_id": agent_id, "path": self.store.rel(path) if path else None,
                              "sha256": sha256, "hash_mode": hash_mode})

    def allow_text(self, text: str) -> None:
        """Register a rendering derived from artifacts already read through this accessor."""
        self.texts.append(text)

    def reports(self, kind: str, state: dict) -> list[ReportRecord]:
        """Completed reports of one kind, in agent order."""
        from paper_adversary.reports import read_report

        self._require(kind)
        role = KIND_ROLE.get(kind, kind)
        out = []
        for aid, a in sorted(state.get("agents", {}).items(), key=lambda kv: _agent_sort_key(kv[0])):
            if a.get("role") != role or a.get("status") != "complete":
                continue
            if aid == self.agent_id:
                continue
            if self.pos is not None and position(role, int(a.get("round") or 0)) >= tuple(self.pos):
                continue
            path = self.store.report_path(aid, role)
            if not path.is_file():
                continue
            meta, body = read_report(path)
            rec = ReportRecord(aid, role, path, meta, body, sha256_text(body))
            self.note_input(kind, path, rec.sha256, aid, hash_mode="body")
            self.texts.append(body)
            out.append(rec)
        return out

    def sidecar(self, kind: str, agent_id: str, role: str, suffix: str) -> dict | None:
        self._require(kind)
        path = self.store.sidecar_path(agent_id, role, suffix)
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        self.note_input(kind, path, sha256_text(text), agent_id)
        self.texts.append(text)
        return json.loads(text)

    def file(self, kind: str, path: Path) -> str:
        self._require(kind)
        text = path.read_text(encoding="utf-8")
        self.note_input(kind, path, sha256_text(text))
        self.texts.append(text)
        return text


def _agent_sort_key(agent_id: str) -> tuple:
    m = re.match(r"([A-Z]+)(\d*)", agent_id)
    return (m.group(1), int(m.group(2) or 0)) if m else (agent_id, 0)


class IsolationGuard:
    """Fingerprint registry plus the pre-flight check and the post-run audit."""

    def __init__(self, store):
        self.store = store
        self.path = store.dir / "logs" / "fingerprints.json"

    def _load(self) -> dict:
        return read_json(self.path, {}) or {}

    def register(self, agent_id: str, role: str, marker: str, body: str, exclude: set[str],
                 pos: tuple[int, int] | None = None) -> None:
        data = self._load()
        data[agent_id] = {"kind": OUTPUT_KIND[role], "marker": marker, "position": list(pos or position(role)),
                          "shingles": distinctive_shingles(body, exclude)}
        atomic_write_json(self.path, data)

    def forbidden(self, role: str, agent_id: str, pos: tuple[int, int] | None = None) -> dict[str, dict]:
        """Fingerprints of every artifact whose kind this role may not see (an agent's own earlier reports
        are always of such a kind: refuters see no reports, judges no judge reports, and so on), every
        quarantined report, which no role may see, and, given the reader's position, everything from the same or
        a later step (so two adjudicators never see each other, and a rerun never sees what came after it)."""
        visible = ROLE_VISIBILITY[role]
        out = {}
        for aid, fp in self._load().items():
            fp_pos = tuple(fp.get("position") or position(KIND_ROLE.get(fp["kind"], fp["kind"])))
            if fp["kind"] not in visible or fp.get("quarantined") or (pos is not None and fp_pos >= tuple(pos)):
                out[aid] = fp
        return out

    def set_quarantined(self, agent_id: str, flag: bool) -> None:
        data = self._load()
        if agent_id in data:
            data[agent_id]["quarantined"] = flag
            atomic_write_json(self.path, data)

    def archive(self, agent_id: str) -> None:
        """Keep a superseded report's fingerprint (still forbidden by kind) under an archived key."""
        data = self._load()
        fp = data.pop(agent_id, None)
        if fp is not None:
            data[f"{agent_id}@{utcnow().strftime('%Y%m%dT%H%M%S%fZ')}"] = fp
            atomic_write_json(self.path, data)

    def _hits(self, text: str, forbidden: dict[str, dict], allowed: set[str] | None = None,
              min_fraction: float = 0.0) -> list[dict]:
        """Forbidden artifacts whose marker, or enough of whose distinctive 8-grams, occur in `text`.

        8-grams that also occur in content the agent may legitimately see (`allowed`) are not evidence:
        a judge quoting a refuter shares phrases with every other judge's legitimate inputs.
        """
        found = []
        markers = {m.group(0) for m in MARKER_RE.finditer(text)}
        text_shingles = set(_shingles(text)) if forbidden else set()
        allowed = allowed or set()
        for aid, fp in forbidden.items():
            marker_hit = fp["marker"] in markers
            effective = [sh for sh in fp["shingles"] if sh not in allowed]
            threshold = max(HIT_THRESHOLD, math.ceil(min_fraction * len(effective)))
            shingle_hits = sum(1 for sh in effective if sh in text_shingles)
            if marker_hit or (effective and shingle_hits >= threshold):
                found.append({"artifact": aid, "kind": fp["kind"], "marker": marker_hit, "shingle_hits": shingle_hits})
        return found

    @staticmethod
    def check_secrets(agent_id: str, prompt_text: str) -> None:
        found = find_secrets(prompt_text)
        if found:
            raise IsolationViolation(f"refusing to start {agent_id}: its prompt contains {', '.join(found)}")

    def check_prompt(self, role: str, agent_id: str, prompt_text: str, allowed_text: str = "",
                     pos: tuple[int, int] | None = None) -> None:
        hits = self._hits(prompt_text, self.forbidden(role, agent_id, pos), set(_shingles(allowed_text)))
        if hits:
            names = ", ".join(f"{h['artifact']} ({h['kind']})" for h in hits)
            raise IsolationViolation(f"refusing to start {agent_id}: its prompt contains content from {names}")

    def audit(self, role: str, agent_id: str, transcript: Path | None, sandbox: Path | None, report_text: str,
              required_events: tuple[str, ...] = ("init", "result"), pos: tuple[int, int] | None = None) -> dict:
        """Post-run check of what the agent actually touched.

        "fail": evidence of contamination (a file outside the sandbox was read, forbidden content arrived in a
        tool result, or the report carries another report's marker). "unverifiable": the transcript cannot show
        that isolation held (missing, empty, or without the init/result events). Attempts the sandbox refused
        are recorded as "denied" without failing the audit.
        """
        findings: list[str] = []
        unverifiable: list[str] = []
        denied: list[str] = []
        forbidden = self.forbidden(role, agent_id, pos)
        events = read_jsonl(transcript) if transcript is not None else []
        if transcript is None or not transcript.is_file():
            unverifiable.append("no transcript was recorded")
        elif not events:
            unverifiable.append("the transcript is empty")
        else:
            kinds = {ev.get("kind") for ev in events}
            unverifiable += [f"the transcript has no {k} event" for k in required_events if k not in kinds]
        results = {ev.get("tool_use_id"): ev for ev in events if ev.get("kind") == "tool_result"}
        refused = {d.get("tool_use_id") for ev in events if ev.get("kind") == "result"
                   for d in ev.get("permission_denials") or [] if isinstance(d, dict)}
        tool_results: list[str] = []
        for ev in events:
            if ev.get("kind") == "tool_use":
                name = ev.get("name", "")
                inp = ev.get("input") or {}
                for key in ("file_path", "path", "notebook_path"):
                    value = inp.get(key) if isinstance(inp, dict) else None
                    if not value:
                        continue
                    if sandbox is None:
                        unverifiable.append(f"{name} used a file path ({value}) but the sandbox is unknown")
                    elif not _inside(Path(str(value)).expanduser(), sandbox):
                        res = results.get(ev.get("id"))
                        if ev.get("id") in refused or (res is not None and res.get("is_error")):
                            denied.append(f"{name} tried {value} outside its sandbox and was refused")
                        elif res is None:
                            unverifiable.append(f"{name} touched {value} outside its sandbox; no result was recorded")
                        else:
                            findings.append(f"{name} read a path outside its sandbox: {value}")
            elif ev.get("kind") == "tool_result":
                tool_results.append(str(ev.get("content", "")))
        # Search results can legitimately repeat a sentence another refuter quoted from the same abstract,
        # so tool results only count when they carry most of a report's fingerprint (or its marker).
        for hit in self._hits("\n".join(tool_results), forbidden, min_fraction=0.5):
            findings.append(f"a tool result contained content from {hit['artifact']} ({hit['kind']})")
        forbidden_markers = {fp["marker"]: aid for aid, fp in forbidden.items()}
        for m in MARKER_RE.finditer(report_text):
            if m.group(0) in forbidden_markers:
                findings.append(f"the report contains the marker of {forbidden_markers[m.group(0)]}")
        status = "fail" if findings else ("unverifiable" if unverifiable else "pass")
        return {"status": status, "findings": findings, "unverifiable": unverifiable, "denied": denied,
                "events": len(events), "checked_against": sorted(forbidden)}


def _inside(path: Path, root: Path) -> bool:
    try:
        path = path if path.is_absolute() else root / path
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def input_changed(store, entry: dict) -> bool:
    """True if an input recorded in a context manifest no longer matches what is on disk."""
    from paper_adversary.reports import read_report
    from paper_adversary.util import sha256_file

    if not entry.get("path"):
        return False
    path = store.dir / entry["path"]
    if not path.is_file():
        return True
    mode = entry.get("hash_mode", "text")
    if mode == "file":
        current = sha256_file(path)
    elif mode == "body":
        current = sha256_text(read_report(path)[1])
    else:
        current = sha256_text(path.read_text(encoding="utf-8"))
    return current != entry.get("sha256")
