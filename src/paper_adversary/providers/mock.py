"""Offline provider for tests and dry runs. Makes no model calls and invents no usage.

Every report carries a canary (CANARY_<agent id>) and lists the canaries it saw
in its own prompt, so tests can prove from the outputs alone which artifacts
each agent was given. Failures can be injected per agent and attempt, and so can
every defect the completion gates check for (see `_opt`).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from pathlib import Path

from paper_adversary.providers.base import AgentRequest, AgentResult, ErrorKind, ProviderError
from paper_adversary.reports import SEVERITY_LABEL
from paper_adversary.util import append_jsonl, atomic_write_text, read_json, utcnow

CANARY = re.compile(r"CANARY_([A-Z]+[0-9]*)")
SYNTHESIS_SECTIONS = ["Executive summary", "Strongest novelty threats", "Strongest rigor threats",
                      "Strongest experimental / feasibility threats", "Criticisms that survived judging",
                      "Criticisms that were rejected", "Unresolved disagreements", "Claims that should be weakened",
                      "Experiments or analyses that should be added", "Prior work that must be discussed",
                      "Recommended paper changes", "Remaining submission risk"]
CRITIC_SECTIONS = ["Newly discovered issues", "Overlooked minority critiques", "Weaknesses in the judging process",
                   "Weaknesses in the synthesis", "Recommended final checks"]
DEFAULT_ITEMS = [{"id": "I1", "type": "new_issue", "title": "Notation used before it is defined",
                  "location": "Sec. 3", "argument": "mock", "severity_estimate": "MINOR", "confidence": "medium",
                  "claim_ids": [], "objection_ids": [], "memo_sections": [], "candidate_references": []}]


def head_of(text: str) -> str:
    return text.split("\n## ", 1)[0] + "\n"


class MockProvider:
    """Options (all optional; per-agent values apply to every attempt unless given as {"attempts": [...]}):

    delay_seconds, synthetic_usage, quote_inputs, reset_at, retry_after_s
    fail            {aid: [error kind or None per attempt]}
    omit_json       [aid, ...]                     (same as json: omit)
    json            {aid: omit|malformed|trailing_comma|unfenced|schema_invalid|id_mismatch|quoted_snippet}
    headings        {aid: False}                   objections/judgments without ### headings
    transcript      {aid: missing|empty|no_init|no_result}
    tool_calls      {aid: [{name, input, result, is_error, denied}]}
    leak_marker_of  {aid: other aid}               the report carries another report's marker
    stop_reason     {aid: "max_tokens"}
    truncate        {aid: True}                    cut the report in half (with stop_reason max_tokens)
    served_model    {aid: model id}
    drop_sections   {aid: [section numbers]}       synthesis/critic sections left out
    coverage        {aid: fraction}                share of objections a judge classifies
    conflict        {aid: objection ID}            a judge classifies this ID a second time, MAJOR BUT FIXABLE
    repair          {aid: valid|invent_objection|invent_reference|change_title|fail:<kind>}
    decisive        {aid: {category, severity, basis, prior: {title, arxiv_id, doi}, prior_passage,
                           submission_passage}}    makes a novelty refuter's O1 a decisive prior-work objection
    verdict         {aid or "default": verdict}    what verifiers answer (default anticipates_partially)
    misplace        {aid: True}                    a memo lists unverified threats as surviving criticisms
    misfile         {aid: True}                    a revised memo also leads a section-5 entry with them
    items           {aid: [item, ...]}             a critic's (critic_v2) or re-check's items (default: one MINOR)
    ruling          {aid or "default": severity}   what adjudicators rule (default MAJOR_FIXABLE)
    skip_item       {aid: item id}                 an adjudicator or revision leaves this item out
    references      {aid: [ref, ...]}              a novelty refuter's references (default: one well-known paper)
    supplement      {aid: valid|skip_one|invalid|extra|fail:<kind>}   how a coverage supplement answers
    placement_fix   {aid: valid|unchanged|renamed|invent|fail:<kind>} how a memo placement fix answers
    """

    name = "mock"

    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.attempts: dict[str, int] = {}
        self.repair_calls = 0

    def _opt(self, name: str, aid: str, attempt: int, default=None):
        value = (self.options.get(name) or {}).get(aid, default) if isinstance(self.options.get(name), dict) \
            else default
        if isinstance(value, dict) and "attempts" in value:
            seq = value["attempts"]
            return seq[attempt - 1] if attempt <= len(seq) else default
        return value

    def cleanup(self, result: AgentResult) -> None:
        if result.sandbox_dir is not None:
            shutil.rmtree(result.sandbox_dir, ignore_errors=True)

    async def auth_status(self) -> dict:
        return {"ok": True, "loggedIn": True, "authMethod": self.options.get("auth_method", "mock"),
                "subscriptionType": "mock"}

    async def probe(self, model: str, log_dir: Path, tools=(), tool_server: dict | None = None) -> dict:
        return {"ok": True, "model_requested": model, "tools": sorted(tools), "served_models": [model],
                "context_window": None, "checked_at": utcnow().isoformat(timespec="seconds"), "mock": True,
                "substituted": []}

    async def run(self, req: AgentRequest, cancel: asyncio.Event | None = None) -> AgentResult:
        attempt = self.attempts.get(req.agent_id, 0) + 1
        self.attempts[req.agent_id] = attempt
        aid = req.agent_id
        req.log_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(req.log_dir / "system_prompt.md", req.system_prompt)
        atomic_write_text(req.log_dir / "user_prompt.md", req.user_text)

        delay = float(self.options.get("delay_seconds", 0.01))
        waited = 0.0
        while waited < delay:
            if cancel is not None and cancel.is_set():
                raise ProviderError(ErrorKind.CANCELLED, "cancelled")
            await asyncio.sleep(min(0.05, delay))
            waited += 0.05

        planned = (self.options.get("fail") or {}).get(aid) or []
        if attempt <= len(planned) and planned[attempt - 1]:
            kind = ErrorKind(planned[attempt - 1])
            raise ProviderError(kind, f"injected {kind.value} (attempt {attempt})",
                                reset_at=self.options.get("reset_at"), retry_after_s=self.options.get("retry_after_s"))

        if req.role == "repair":
            self.repair_calls += 1
            text = self._repair(req, attempt)
        elif req.role == "supplement":
            text = self._supplement(req, attempt)
        elif req.role == "placement_fix":
            text = self._placement_fix(req, attempt)
        else:
            seen = sorted({f"CANARY_{m}" for m in CANARY.findall(req.system_prompt + req.user_text)})
            json_mode = self._opt("json", aid, attempt) or ("omit" if aid in (self.options.get("omit_json") or [])
                                                             else None)
            text = _render(req, seen, json_mode, self._opt("headings", aid, attempt, True),
                           self._opt("coverage", aid, attempt), self._opt("decisive", aid, attempt),
                           self._opt("verdict", aid, attempt) or (self.options.get("verdict") or {}).get("default"),
                           bool(self._opt("misplace", aid, attempt)), self._opt("references", aid, attempt),
                           self._opt("conflict", aid, attempt))
            if req.role in ("critic", "recheck") and '"items": [' in req.system_prompt and json_mode != "omit":
                text += "\n" + _block({"summary": "mock", "items": self._opt("items", aid, attempt, DEFAULT_ITEMS),
                                        "checks_without_findings": []}, json_mode)
            elif req.role == "adjudicator":
                text = _adjudicator(req, head_of(text), self._opt("ruling", aid, attempt) or
                                    (self.options.get("ruling") or {}).get("default") or "MAJOR_FIXABLE",
                                    self._opt("skip_item", aid, attempt))
            elif req.role == "revision":
                text = _revision(req, head_of(text), bool(self._opt("misplace", aid, attempt)),
                                 bool(self._opt("misfile", aid, attempt)),
                                 self._opt("skip_item", aid, attempt))
            if self.options.get("quote_inputs"):
                text += _quotes(req.user_text)
            for n in self._opt("drop_sections", aid, attempt) or []:
                text = re.sub(rf"^## {n}\. .*?(?=^## |\Z)", "", text, flags=re.M | re.S)
            leak = self._opt("leak_marker_of", aid, attempt)
            if leak:
                fingerprints = read_json(req.log_dir.parents[2] / "fingerprints.json", {}) or {}
                text += "\n\n" + (fingerprints.get(leak) or {}).get("marker", "") + "\n"
            if self._opt("truncate", aid, attempt):
                text = text[: len(text) // 2]

        sandbox = Path(tempfile.mkdtemp(prefix="pa-mock-sandbox-"))
        served = self._opt("served_model", aid, attempt) or req.model
        stop_reason = self._opt("stop_reason", aid, attempt) or "end_turn"
        transcript = req.log_dir / "transcript.jsonl"
        self._transcript(transcript, req, attempt, text, served, stop_reason)
        usage = None
        if self.options.get("synthetic_usage"):
            usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0,
                     "cache_creation_input_tokens": 0, "server_tool_use": {"web_search_requests": 0}}
        return AgentResult(text=text, stop_reason=stop_reason, usage=usage,
                           model_usage={served: {"inputTokens": 1000, "outputTokens": 500}} if usage else None,
                           reported_cost_usd=None, duration_ms=int(delay * 1000), num_turns=1,
                           served_models=[served], transcript_path=transcript, sandbox_dir=sandbox,
                           provider="mock", runtime={"mock": True})

    def _transcript(self, path: Path, req: AgentRequest, attempt: int, text: str, served: str,
                    stop_reason: str) -> None:
        mode = self._opt("transcript", req.agent_id, attempt)
        if mode == "missing":
            return
        path.touch()
        if mode == "empty":
            return
        if mode != "no_init":
            append_jsonl(path, {"kind": "init", "model": served, "tools": [], "apiKeySource": "none"})
        denials = []
        for i, call in enumerate(self._opt("tool_calls", req.agent_id, attempt) or [], start=1):
            tid = f"toolu_mock_{i}"
            append_jsonl(path, {"kind": "tool_use", "id": tid, "name": call.get("name", "Read"),
                                "input": call.get("input") or {}})
            if call.get("denied"):
                denials.append({"tool_name": call.get("name", "Read"), "tool_use_id": tid})
            if "result" in call:
                append_jsonl(path, {"kind": "tool_result", "tool_use_id": tid, "is_error": bool(call.get("is_error")),
                                    "content": str(call["result"])})
        append_jsonl(path, {"kind": "assistant_text", "text": text[:2000], "model": served})
        if mode != "no_result":
            append_jsonl(path, {"kind": "result", "subtype": "success", "stop_reason": stop_reason,
                                "permission_denials": denials})

    def _repair(self, req: AgentRequest, attempt: int) -> str:
        """Act as the format-repair model: transcribe the report's headings (and any references in its prose)."""
        base = req.agent_id.removesuffix("-repair")
        mode = str(self._opt("repair", base, attempt) or "valid")
        if mode.startswith("fail:"):
            raise ProviderError(ErrorKind(mode.split(":", 1)[1]), f"injected repair failure ({mode})")
        m = re.search(r"<report>\n(.*?)\n</report>", req.user_text, re.S)
        report = m.group(1) if m else ""
        claims = re.findall(r"^- (C\d+): (.+?)(?: \((.+?)\))?$", report, re.M)
        if claims:
            data: dict = {"claims": [{"id": c, "claim": t, "location": loc or None} for c, t, loc in claims]}
        else:
            objections = []
            for oid, title, body in re.findall(r"^### (O\d+): (.+?)\n(.*?)(?=^### |\Z)", report, re.M | re.S):
                refs = [{"title": t, "arxiv_id": a} for t, a in
                        re.findall(r'"([^"]+)" \(\d{4}, arXiv (\d{4}\.\d{4,5})\)', body)]
                objections.append({"id": oid, "title": title.strip(), "severity_estimate": "major",
                                   "confidence": "medium", "category": None, "references": refs})
            if mode == "invent_objection":
                objections.append({"id": "O99", "title": "An objection the report never made"})
            if mode == "invent_reference" and objections:
                objections[0]["references"].append({"title": "A Paper That Does Not Exist", "arxiv_id": None})
            if mode == "change_title" and objections:
                objections[0]["title"] += " (reworded)"
            data = {"objections": objections}
        return "```json\n" + json.dumps(data, indent=2) + "\n```"


    def _supplement(self, req: AgentRequest, attempt: int) -> str:
        """Act as a judge's or adjudicator's coverage supplement: rule on exactly the IDs it was asked about."""
        base = req.agent_id.removesuffix("-supplement")
        mode = str(self._opt("supplement", base, attempt) or "valid")
        if mode.startswith("fail:"):
            raise ProviderError(ErrorKind(mode.split(":", 1)[1]), f"injected supplement failure ({mode})")
        asked = re.findall(r"^- ([A-Z]{1,2}\d+-[OI]\d+): (?:not classified|you gave it)", req.user_text, re.M)
        if mode == "skip_one":
            asked = asked[1:]
        if mode == "extra":
            asked = asked + ["Z9-O9" if not base.startswith("A") else "C9-I9"]
        shown = sorted(set(re.findall(r'<report agent_id="([NRF]\d+)"', req.user_text)))
        if base.startswith("A"):
            entries = [{"item_ids": [i], "title": f"ruling on {i}", "severity": "MINOR", "confidence": "medium",
                        "is_new": "yes", "already_covered_by": [], "objection_updates": [],
                        "evidence_status": "not_applicable", "supporting_evidence": "mock", "rationale": "mock",
                        "resolvable_with_more_evidence": "yes", "what_would_resolve": "mock"} for i in asked]
            data: dict = {"rulings": entries}
        else:
            entries = [{"objection_ids": [i], "title": f"{i} (supplement)", "severity": "MINOR",
                        "confidence": "medium", "refuter_sources": [i.split("-")[0]], "supporting_evidence": "mock",
                        "resolvable_with_more_evidence": "yes", "what_would_resolve": "mock"} for i in asked]
            data = {"judgments": entries, "refuter_disagreements": []}
        if mode == "invalid":
            data = {"judgments": [{"objection_ids": asked[:1], "severity": "SEVERE"}]} if not base.startswith("A") \
                else {"rulings": [{"item_ids": asked[:1], "severity": "SEVERE"}]}
        heads = "\n".join(f"### MINOR: ruling ({i})\nMock supplement ruling.\n" for i in asked)
        return f"Supplement saw reports of: {', '.join(shown) or 'none'}\n\n{heads}\n" + _block(data)

    def _placement_fix(self, req: AgentRequest, attempt: int) -> str:
        """Act as a memo's placement fix: file the named IDs under the unverified-threats section."""
        base = req.agent_id.removesuffix("-placement_fix")
        mode = str(self._opt("placement_fix", base, attempt) or "valid")
        if mode.startswith("fail:"):
            raise ProviderError(ErrorKind(mode.split(":", 1)[1]), f"injected placement-fix failure ({mode})")
        heads = re.findall(r"`## (\d+)\. ([^`]+)`", req.system_prompt)
        m = re.search(r"may not lead an entry of section \d+: ([^\n]+)", req.user_text)
        flagged = [x.strip() for x in m.group(1).split(",")] if m else []
        (five, five_title), (six, six_title) = heads[:2]
        if mode == "renamed":
            six_title = "Threats"
        survived = "Mock (fixed)."
        if mode == "unchanged":
            survived = "\n".join(f"- **{i}** — survived judging." for i in flagged)
        if mode == "invent":
            survived = "- **Z9-O9** — an objection the memo never cited."
        unverified = "\n".join(f"- **{i}** — unverified; check the full text first." for i in flagged)
        return f"## {five}. {five_title}\n\n{survived}\n\n## {six}. {six_title}\n\n{unverified}\n"


def _block(data: dict, mode: str | None = None) -> str:
    raw = json.dumps(data, indent=2)
    if mode == "malformed":
        return "```json\n" + raw[: len(raw) // 2] + "\n```"
    if mode == "trailing_comma":
        return "```json\n" + raw.replace("\n  ]", ",\n  ]", 1) + "\n```"
    if mode == "unfenced":
        return raw
    if mode == "quoted_snippet":  # a valid JSON snippet earlier in the report, then a broken final block
        return "```json\n{\"objections\": []}\n```\n\nFinal block:\n\n```json\n" + raw[: len(raw) // 2] + "\n```"
    return "```json\n" + raw + "\n```"


def _render(req: AgentRequest, seen: list[str], json_mode: str | None, headings: bool,
            coverage: float | None, decisive: dict | None = None, verdict: str | None = None,
            misplace: bool = False, references: list | None = None, conflict: str | None = None) -> str:
    aid = req.agent_id
    head = f"# Mock {req.role} report ({aid})\n\nCANARY_{aid}\n\nInputs seen: {', '.join(seen) or 'none'}\n"
    if req.role == "verifier":
        return head + "\n" + _verifier(req, verdict or "anticipates_partially")
    if req.role in ("novelty", "rigor", "fit"):
        objections = []
        for i, (sev, conf) in enumerate((("major", "medium"), ("minor", "low")), start=1):
            obj = {"id": f"O{i}", "title": f"{aid} objection {i}", "claim_targeted": "Section 1, main claim",
                   "argument": f"Mock argument {i} from {aid}.", "evidence": [f"evidence {i}"],
                   "severity_estimate": sev, "confidence": conf, "resolvable_by": "an extra experiment"}
            if req.role == "novelty":
                obj["references"] = list(references) if references is not None else [
                    {"title": "Attention Is All You Need", "authors": "Vaswani et al.", "year": 2017,
                     "arxiv_id": "1706.03762"}]
            objections.append(obj)
        if decisive and req.role == "novelty":
            prior = decisive.get("prior") or {}
            objections[0].update(category=decisive.get("category", "already_done"),
                                 severity_estimate=decisive.get("severity", "fatal"),
                                 evidence_basis=decisive.get("basis", "full_text"),
                                 references=[{"title": prior.get("title"), "arxiv_id": prior.get("arxiv_id"),
                                              "doi": prior.get("doi"), "year": prior.get("year")}])
            if decisive.get("prior_passage"):
                objections[0]["overlap_evidence"] = [{
                    "reference": 1, "prior_passage": decisive["prior_passage"], "prior_location": "p. 1",
                    "submission_passage": decisive.get("submission_passage") or _submission_words(req.user_text),
                    "submission_location": "Sec. 1", "relation": "same_method"}]
        parts = []
        for o in objections:
            prose = o["argument"]
            if req.role == "novelty" and references is None:
                prose += ' Evidence: Vaswani et al., "Attention Is All You Need" (2017, arXiv 1706.03762).'
            parts.append((f"### {o['id']}: {o['title']}\n" if headings else f"**{o['id']} {o['title']}**\n") + prose
                         + "\n")
        body = head + "\n## Objections\n\n" + "\n".join(parts)
        data = {"summary_verdict": "mock verdict", "objections": objections, "areas_checked_without_findings": []}
        if json_mode == "schema_invalid":
            data = {"summary_verdict": "mock verdict", "objections": [{"id": "first", "title": "bad id"}]}
        if json_mode == "id_mismatch":
            data["objections"].append({**objections[0], "id": "O7", "title": "only in the JSON"})
        return body + ("" if json_mode == "omit" else "\n" + _block(data, json_mode))
    if req.role == "judge":
        refuters = sorted(set(re.findall(r'<report agent_id="([NRF]\d+)"', req.user_text)))
        index = int(re.sub(r"\D", "", aid) or 1)
        palette = [("MAJOR_FIXABLE", "MINOR"), ("FATAL", "NOT_CONVINCING"), ("MINOR", "MINOR")]
        sev1, sev2 = palette[(index - 1) % len(palette)]
        judgments = []
        for rid in refuters:
            judgments.append({"objection_ids": [f"{rid}-O1"], "title": f"{rid} objection 1", "severity": sev1,
                              "confidence": "medium", "refuter_sources": [rid], "supporting_evidence": "mock",
                              "resolvable_with_more_evidence": "yes", "what_would_resolve": "mock"})
            if index != 3:  # the third judge skips the second objections, to exercise coverage checks
                judgments.append({"objection_ids": [f"{rid}-O2"], "title": f"{rid} objection 2", "severity": sev2,
                                  "confidence": "low", "refuter_sources": [rid], "supporting_evidence": "mock",
                                  "resolvable_with_more_evidence": "partially", "what_would_resolve": "mock"})
        if coverage is not None:
            judgments = judgments[: max(0, round(len(judgments) * coverage))]
        if conflict:
            judgments.append({"objection_ids": [conflict], "title": f"{conflict} again", "severity": "MAJOR_FIXABLE",
                              "confidence": "low", "refuter_sources": [conflict.split("-")[0]],
                              "supporting_evidence": "mock", "resolvable_with_more_evidence": "yes",
                              "what_would_resolve": "mock"})
        lines = []
        for j in judgments:
            label = SEVERITY_LABEL[j["severity"]]
            lines.append((f"### {label}: {j['title']} ({', '.join(j['objection_ids'])})\n" if headings
                          else f"**{label}** {j['title']}\n") + "Mock judgment.\n")
        data = {"judgments": judgments, "refuter_disagreements": []}
        body = head + "\n## Judgments\n\n" + "\n".join(lines)
        return body + ("" if json_mode == "omit" else "\n" + _block(data, json_mode))
    if req.role in ("synthesis", "critic", "recheck", "revision", "adjudicator"):
        sections = _prompt_sections(req.system_prompt) or list(enumerate(
            SYNTHESIS_SECTIONS if req.role == "synthesis" else CRITIC_SECTIONS, 1))
        ids = _gated_ids(req.user_text)
        out = []
        for n, title in sections:
            body = "Mock."
            if title.lower().startswith("unverified threats") and ids and not misplace:
                body = f"- **{', '.join(ids)}** — unverified; check the full text first."
            if title.lower().startswith("criticisms that survived") and ids and misplace:
                body = f"- **{', '.join(ids)}** — survived judging."
            out.append(f"## {n}. {title}\n\n{body}\n")
        return head + "\n" + "\n".join(out)
    if req.role == "intake":
        data = {"title": "Mock Title", "field": "machine learning", "submission_type": "paper",
                "venue_guess": None, "claims": [{"id": "C1", "claim": "Mock claim", "location": "Abstract",
                                                 "type": "empirical"}]}
        body = head + "\n## Claims\n\n- C1: Mock claim (Abstract)\n"
        return body + ("" if json_mode == "omit" else "\n" + _block(data, json_mode))
    return head


def _triage(user_text: str) -> list[tuple[str, str]]:
    """(item ID, route) pairs from the orchestrator's <item_triage> block."""
    m = re.search(r"<item_triage[^>]*>\n(.*?)</item_triage>", user_text, re.S)
    return re.findall(r"^- (C\d+-I\d+) \[[^\]]*?route: (\w+)", m.group(1) if m else "", re.M)


def _adjudicator(req: AgentRequest, head: str, severity: str, skip: str | None) -> str:
    routed = [i for i, route in _triage(req.user_text) if route == "adjudicate" and i != skip]
    rulings = [{"item_ids": [i], "title": f"ruling on {i}", "severity": severity, "confidence": "medium",
                "is_new": "yes", "already_covered_by": [], "objection_updates": [], "evidence_status": "not_applicable",
                "supporting_evidence": "mock", "rationale": "mock", "resolvable_with_more_evidence": "yes",
                "what_would_resolve": {"kind": "analysis", "description": "mock"}, "dimension": "soundness"}
               for i in routed]
    body = "".join(f"### {SEVERITY_LABEL[severity]}: ruling on {i} ({i})\nMock.\n\n" for i in routed)
    return head + "\n## Rulings\n\n" + body + _block({"overall": "mock", "rulings": rulings})


def _gated_ids(user_text: str) -> list[str]:
    """IDs the evidence gates in an agent's prompt list as not shown (base and follow-up gates)."""
    groups = re.findall(r"^- [JA]\d+ \S+(?: BUT FIXABLE)? on ([A-Z0-9, -]+?)(?: \(|:)", user_text, re.M)
    return sorted({x.strip() for group in groups for x in group.split(",") if x.strip()})


def _revision(req: AgentRequest, head: str, misplace: bool, misfile: bool, skip: str | None) -> str:
    gate = re.search(r"Follow-up gate: not shown: ([^\n]+)", req.user_text)
    flagged = {x.strip() for x in gate.group(1).split(",") if x.strip() != "none"} if gate else set()
    shown_in_six = sorted(set(_gated_ids(req.user_text)) | flagged)
    disp = {"adjudicate": "incorporated", "revision": "incorporated", "noted": "noted", "invalid": "invalid"}
    items = []
    for item_id, route in _triage(req.user_text):
        if item_id == skip:
            continue
        d = "unverified_threat" if item_id in flagged and not misplace else disp.get(route, "incorporated")
        items.append({"item_id": item_id, "disposition": d, "memo_sections": [12], "note": "mock"})
    sections = _prompt_sections(req.system_prompt)
    def body(t: str) -> str:
        if t.lower().startswith("unverified") and shown_in_six and not misplace:
            return f"- **{', '.join(shown_in_six)}** — unverified; check first."
        if t.lower().startswith("criticisms that survived") and shown_in_six and misfile:
            return f"- **{', '.join(shown_in_six)}** — survived judging."
        return "Mock."
    memo = "\n".join(f"## {n}. {t}\n\n{body(t)}\n" for n, t in sections)
    return head + "\n" + memo + "\n" + _block({"item_dispositions": items, "objection_changes": [],
                                                "remaining_risk": "medium"})


def _prompt_sections(system_prompt: str) -> list[tuple[int, str]]:
    part = system_prompt.split("# Output format", 1)[-1]
    return [(int(n), title.strip()) for n, title in re.findall(r"^##[ \t]+(\d+)\.[ \t]*(.+?)[ \t]*$", part, re.M)]


def _submission_words(user_text: str, n: int = 12) -> str:
    """Twelve consecutive words from a plain paragraph of the submission, verbatim."""
    m = re.search(r"<submission[^>]*>\n(.*?)</submission>", user_text, re.S)
    for line in (m.group(1) if m else "").splitlines():
        words = line.split()
        if len(words) >= n and not line.startswith(("#", "<", ">")):
            return " ".join(words[:n])
    return ""


def _verifier(req: AgentRequest, verdict: str) -> str:
    """Answer every passage under examination; overlap quotes are copied verbatim from the two texts."""
    passages = re.findall(r"^(P\d+) \[", req.user_text, re.M)
    m = re.search(r"<prior_paper[^>]*>\n(.*?)</prior_paper>", req.user_text, re.S)
    prior_line = next((ln for ln in (m.group(1) if m else "").splitlines()
                       if len(ln.split()) >= 15 and not ln.startswith(("#", "<", ">"))), "")
    items = []
    for pid in passages:
        span = re.search(rf"^{pid} \[[^\]]*\]:\n(.+)$", req.user_text, re.M)
        overlap = ([{"prior_passage": " ".join(prior_line.split()[:15]), "prior_location": "p. 1",
                     "submission_passage": " ".join(span.group(1).split()[:10]) if span else "",
                     "explanation": "mock overlap"}] if verdict.startswith("anticipates") else [])
        items.append({"passage_id": pid, "verdict": verdict, "confidence": "medium", "overlap": overlap,
                      "differences": [{"prior_passage": None, "submission_passage": "", "explanation": "mock"}],
                      "rationale": "mock"})
    body = "".join(f"### {i['passage_id']}: {i['verdict']}\nMock comparison.\n\n" for i in items)
    return body + _block({"text_quality": "ok", "passages": items, "summary": "mock"})


def _quotes(prompt: str) -> str:
    """Quote 40 words from each report in the prompt, as real judges and memos quote refuters."""
    out = []
    for m in re.finditer(r'<report agent_id="([A-Z]+\d+)"[^>]*>\n(.*?)</report>', prompt, re.S):
        words = re.sub(r"<!--.*?-->|CANARY_\w+|Inputs seen:[^\n]*", " ", m.group(2)).split()
        out.append(f'> {m.group(1)} wrote: "{" ".join(words[5:45])}"')
    return ("\n\nQuoted evidence:\n" + "\n".join(out) + "\n") if out else ""
