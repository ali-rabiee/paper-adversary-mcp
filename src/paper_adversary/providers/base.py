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


class Provider(Protocol):
    name: str

    async def run(self, req: AgentRequest, cancel: asyncio.Event | None = None) -> AgentResult: ...

    async def auth_status(self) -> dict: ...

    async def probe(self, model: str, log_dir: Path, tools=(), tool_server: dict | None = None) -> dict: ...

    def cleanup(self, result: AgentResult) -> None: ...
