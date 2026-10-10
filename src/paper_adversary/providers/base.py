"""Provider interface: how one isolated agent call is executed."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol


class ErrorKind(str, Enum):
    AUTH = "auth"
    MODEL_UNAVAILABLE = "model_unavailable"
    PLAN_LIMIT = "plan_limit"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    SERVER = "server_error"
    NETWORK = "network"
    TIMEOUT = "timeout"
    IDLE_TIMEOUT = "idle_timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    REFUSAL = "refusal"
    INVALID_REQUEST = "invalid_request"
    TOOL_SETUP = "tool_setup"
    EMPTY_OUTPUT = "empty_output"
    MAX_TURNS = "max_turns"
    BILLING_GUARD = "billing_guard"
    ISOLATION = "isolation"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


RETRYABLE = {
    ErrorKind.RATE_LIMIT, ErrorKind.OVERLOADED, ErrorKind.SERVER, ErrorKind.NETWORK, ErrorKind.TIMEOUT,
    ErrorKind.IDLE_TIMEOUT, ErrorKind.EMPTY_OUTPUT, ErrorKind.TOOL_SETUP, ErrorKind.UNKNOWN,
}
# Failures that make every other agent fail too; the pipeline stops scheduling.
RUN_FATAL = {ErrorKind.AUTH, ErrorKind.BILLING_GUARD}
# Plan windows that limit one model, not the account. Claude Code reports seven_day_overage_included only in the
# events of Fable calls, and a rejection in it refuses only that model's calls (K1 pilot, 2026-09-29 probe).
MODEL_WINDOWS = {"seven_day_overage_included": "weekly"}


class ProviderError(Exception):
    def __init__(self, kind: ErrorKind, message: str, *, retry_after_s: float | None = None,
                 reset_at: str | None = None, usage: dict | None = None, detail: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retry_after_s = retry_after_s
        self.reset_at = reset_at
        self.usage = usage
        self.detail = detail or {}

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE

    def __str__(self) -> str:
        return f"{self.kind.value}: {self.message}"


@dataclass
class AgentRequest:
    run_id: str
    agent_id: str
    role: str
    model: str
    effort: str
    system_prompt: str
    user_text: str
    log_dir: Path
    tools: list[str] = field(default_factory=list)  # web_search, web_fetch, literature, paper_sections, read_pdf
    pdf_path: Path | None = None
    tool_server: dict | None = None  # {"args": [...], "env": {...}} for the literature/section MCP server
    timeout_s: float = 3600
    idle_timeout_s: float = 1500
    max_turns: int | None = None
    max_output_tokens: int | None = None


@dataclass
class AgentResult:
    text: str
    stop_reason: str | None = None
    usage: dict | None = None
    model_usage: dict | None = None
    reported_cost_usd: float | None = None
    duration_ms: int | None = None
    duration_api_ms: int | None = None
    num_turns: int | None = None
    served_models: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    transcript_path: Path | None = None
    sandbox_dir: Path | None = None
    session_id: str | None = None
    provider: str = ""
    runtime: dict = field(default_factory=dict)  # e.g. claude_code_version, apiKeySource
    turns: list[dict] = field(default_factory=list)  # per model turn: stop_reason, output_tokens (stream events)
    hidden_turns: dict | None = None  # turns before the final message whose text the report lacks (hidden_turns())


def hidden_turns(turns: list[dict], num_turns: int | None, usage: dict | None, tool_less: bool) -> dict | None:
    """Turns before the final message whose text the result does not contain: for an agent without tools every
    earlier turn (it has nothing to call, so a second turn means the first was discarded); for any agent an
    earlier turn that stopped at the output-token limit. From per-turn stream events when there are any, else
    (older Claude Code) from num_turns and the usage's per-iteration output tokens, which list only the last turn.
    A streamed message that never ended (no stop reason: a request Claude Code retried after a stream error) is no
    turn; for an agent without tools the count is reconciled with the num_turns Claude Code reports: one turn and
    no earlier output-token stop means nothing is hidden, more turns than the stream showed count as hidden.
    None when there are none. In the K1 pilot, J1's hidden first turn used 128,000 output tokens."""
    earlier = [t for t in turns[:-1] if t.get("stop_reason")]
    cut = [t for t in earlier if t.get("stop_reason") == "max_tokens"]
    reported = int(num_turns) if isinstance(num_turns, (int, float)) else None
    if turns:
        hidden = earlier if tool_less else cut
        if tool_less and reported is not None and reported <= 1:
            hidden = cut  # Claude Code counted one turn: the other messages were retries of the same request
        unseen = max(0, reported - 1 - len(hidden)) if tool_less and reported is not None else 0
        if not hidden and not unseen:
            return None
        tokens = sum(int(t.get("output_tokens") or 0) for t in hidden)
        if unseen:  # turns the stream did not show: the usage's figure covers every turn before the last
            tokens = _usage_tokens(usage) or tokens or None
        return {"count": len(hidden) + unseen, "output_tokens": tokens, "max_tokens_stops": len(cut),
                "turns": [int(t.get("turn") or 0) for t in hidden], "source": "stream+num_turns" if unseen else "stream"}
    if not tool_less or int(num_turns or 0) <= 1:
        return None
    return {"count": int(num_turns) - 1, "output_tokens": _usage_tokens(usage), "max_tokens_stops": None,
            "turns": [], "source": "usage"}


def _usage_tokens(usage: dict | None) -> int | None:
    """Output tokens of the turns before the last, from the usage: its total less the per-iteration tokens, which
    list only the last turn."""
    usage = usage or {}
    iterations = [i for i in usage.get("iterations") or [] if isinstance(i, dict)]
    total = usage.get("output_tokens")
    return (int(total) - sum(int(i.get("output_tokens") or 0) for i in iterations)
            if iterations and isinstance(total, (int, float)) else None)


def hidden_turns_warning(hidden: dict) -> str:
    tokens = hidden.get("output_tokens")
    stops = hidden.get("max_tokens_stops")
    return (f"{hidden['count']} hidden turn(s) before the final message"
            + (f" used {tokens:,} output tokens" if tokens is not None else "")
            + (f"; {stops} of them stopped at the output-token limit" if stops else "")
            + "; only the final turn's text is in the report")


class Provider(Protocol):
    name: str

    async def run(self, req: AgentRequest, cancel: asyncio.Event | None = None) -> AgentResult: ...

    async def auth_status(self) -> dict: ...

    async def probe(self, model: str, log_dir: Path, tools=(), tool_server: dict | None = None) -> dict: ...

    def cleanup(self, result: AgentResult) -> None: ...
