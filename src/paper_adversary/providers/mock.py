"""Offline provider for tests and dry runs. Makes no model calls and invents no usage.

Every report carries a canary (CANARY_<agent id>) and lists the canaries it saw
in its own prompt, so tests can prove from the outputs alone which artifacts
each agent was given. Failures can be injected per agent and attempt.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from paper_adversary.providers.base import AgentRequest, AgentResult, ErrorKind, ProviderError
from paper_adversary.util import append_jsonl, atomic_write_text, utcnow

CANARY = re.compile(r"CANARY_([A-Z]+[0-9]*)")


class MockProvider:
    name = "mock"

    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.attempts: dict[str, int] = {}

    def cleanup(self, result: AgentResult) -> None:
        return None

    async def auth_status(self) -> dict:
        return {"ok": True, "loggedIn": True, "authMethod": self.options.get("auth_method", "mock"),
                "subscriptionType": "mock"}

    async def probe(self, model: str, log_dir: Path, tools=(), tool_server: dict | None = None) -> dict:
        return {"ok": True, "model_requested": model, "tools": sorted(tools), "served_models": [model],
                "context_window": None, "checked_at": utcnow().isoformat(timespec="seconds"), "mock": True}

    async def run(self, req: AgentRequest, cancel: asyncio.Event | None = None) -> AgentResult:
        attempt = self.attempts.get(req.agent_id, 0) + 1
        self.attempts[req.agent_id] = attempt
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

        planned = (self.options.get("fail") or {}).get(req.agent_id) or []
        if attempt <= len(planned) and planned[attempt - 1]:
            kind = ErrorKind(planned[attempt - 1])
            reset = self.options.get("reset_at")
            raise ProviderError(kind, f"injected {kind.value} (attempt {attempt})", reset_at=reset,
                                retry_after_s=self.options.get("retry_after_s"))

        seen = sorted({f"CANARY_{m}" for m in CANARY.findall(req.system_prompt + req.user_text)})
        text = _render(req, seen, omit_json=req.agent_id in (self.options.get("omit_json") or []))
        if self.options.get("quote_inputs"):
            text += _quotes(req.user_text)
        transcript = req.log_dir / "transcript.jsonl"
        append_jsonl(transcript, {"kind": "assistant_text", "text": text[:2000]})
        usage = None
        if self.options.get("synthetic_usage"):
            usage = {"input_tokens": 1000, "output_tokens": 500, "cache_read_input_tokens": 0,
                     "cache_creation_input_tokens": 0, "server_tool_use": {"web_search_requests": 0}}
        return AgentResult(text=text, stop_reason="end_turn", usage=usage,
                           model_usage={req.model: {"inputTokens": 1000, "outputTokens": 500}} if usage else None,
                           reported_cost_usd=None, duration_ms=int(delay * 1000), num_turns=1,
                           served_models=[req.model], transcript_path=transcript, provider="mock",
                           runtime={"mock": True})


def _block(data: dict) -> str:
    return "```json\n" + json.dumps(data, indent=2) + "\n```"


def _render(req: AgentRequest, seen: list[str], omit_json: bool) -> str:
    aid = req.agent_id
    head = f"# Mock {req.role} report ({aid})\n\nCANARY_{aid}\n\nInputs seen: {', '.join(seen) or 'none'}\n"
    if req.role in ("novelty", "rigor", "fit"):
        objections = []
        for i, (sev, conf) in enumerate((("major", "medium"), ("minor", "low")), start=1):
            obj = {"id": f"O{i}", "title": f"{aid} objection {i}", "claim_targeted": "Section 1, main claim",
                   "argument": f"Mock argument {i} from {aid}.", "evidence": [f"evidence {i}"],
                   "severity_estimate": sev, "confidence": conf, "resolvable_by": "an extra experiment"}
            if req.role == "novelty":
                obj["references"] = [{"title": "Attention Is All You Need", "authors": "Vaswani et al.",
                                      "year": 2017, "arxiv_id": "1706.03762"}]
            objections.append(obj)
        body = head + "\n## Objections\n\n" + "\n".join(f"### {o['id']}: {o['title']}\n{o['argument']}\n"
                                                        for o in objections)
        data = {"summary_verdict": "mock verdict", "objections": objections, "areas_checked_without_findings": []}
        return body + ("" if omit_json else "\n" + _block(data))
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
        data = {"judgments": judgments, "refuter_disagreements": []}
        return head + "\n## Judgments\n\nMock judgments.\n" + ("" if omit_json else "\n" + _block(data))
    if req.role == "synthesis":
        sections = ["Executive summary", "Strongest novelty threats", "Strongest rigor threats",
                    "Strongest experimental / feasibility threats", "Criticisms that survived judging",
                    "Criticisms that were rejected", "Unresolved disagreements", "Claims that should be weakened",
                    "Experiments or analyses to add", "Prior work that must be discussed",
                    "Recommended paper changes", "Remaining submission risk"]
        return head + "\n" + "\n".join(f"## {i}. {s}\n\nMock.\n" for i, s in enumerate(sections, 1))
    if req.role == "critic":
        sections = ["Newly discovered issues", "Overlooked minority critiques", "Weaknesses in the judging process",
                    "Weaknesses in the synthesis", "Recommended final checks"]
        return head + "\n" + "\n".join(f"## {i}. {s}\n\nMock.\n" for i, s in enumerate(sections, 1))
    if req.role == "intake":
        data = {"title": "Mock Title", "field": "machine learning", "submission_type": "paper",
                "venue_guess": None, "claims": [{"id": "C1", "claim": "Mock claim", "location": "Abstract",
                                                 "type": "empirical"}]}
        return head + "\n" + _block(data)
    return head


def _quotes(prompt: str) -> str:
    """Quote 40 words from each report in the prompt, as real judges and memos quote refuters."""
    out = []
    for m in re.finditer(r'<report agent_id="([A-Z]+\d+)"[^>]*>\n(.*?)</report>', prompt, re.S):
        words = re.sub(r"<!--.*?-->|CANARY_\w+|Inputs seen:[^\n]*", " ", m.group(2)).split()
        out.append(f'> {m.group(1)} wrote: "{" ".join(words[5:45])}"')
    return ("\n\nQuoted evidence:\n" + "\n".join(out) + "\n") if out else ""
