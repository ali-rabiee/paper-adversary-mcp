"""Agent execution backends."""

from __future__ import annotations

from paper_adversary.config import ProviderConfig
from paper_adversary.providers.base import (
    RETRYABLE,
    RUN_FATAL,
    AgentRequest,
    AgentResult,
    ErrorKind,
    Provider,
    ProviderError,
)


def make_provider(cfg: ProviderConfig) -> Provider:
    if cfg.type == "mock":
        from paper_adversary.providers.mock import MockProvider

        return MockProvider(cfg.mock)
    from paper_adversary.providers.claude_code import ClaudeCodeProvider

    return ClaudeCodeProvider(cfg)


__all__ = ["AgentRequest", "AgentResult", "ErrorKind", "Provider", "ProviderError", "RETRYABLE", "RUN_FATAL",
           "make_provider"]
