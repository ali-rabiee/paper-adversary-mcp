"""Model registry: aliases -> model IDs, limits and API-equivalent prices (config/models.yaml)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from paper_adversary.util import config_dir


@dataclass
class ModelInfo:
    alias: str
    id: str
    display_name: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    efforts: list[str] = field(default_factory=list)
    pricing: dict[str, float] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    known: bool = True  # False when the name is not in models.yaml


def normalize_model_id(model_id: str) -> str:
    """Strip Claude Code decorations such as '[1m]' so usage keys map back to registry IDs."""
    return re.sub(r"\[[^\]]*\]$", "", model_id.strip())


def substituted_models(requested: str, served: list[str]) -> list[str]:
    """Models that served turns although another model was requested (date suffixes like -20251001 are fine)."""
    want = normalize_model_id(requested)
    return [m for m in served if not normalize_model_id(m).startswith(want)]


class ModelRegistry:
    def __init__(self, data: dict):
        self.pricing_source: str = data.get("pricing_source", "")
        self.web_search_usd_per_1k: float | None = data.get("web_search_usd_per_1k")
        self.efforts: list[str] = list(data.get("efforts") or ["low", "medium", "high", "xhigh", "max"])
        self.models: dict[str, ModelInfo] = {}
        for alias, spec in (data.get("models") or {}).items():
            self.models[alias] = ModelInfo(
                alias=alias,
                id=spec["id"],
                display_name=spec.get("display_name", ""),
                context_window=spec.get("context_window"),
                max_output_tokens=spec.get("max_output_tokens"),
                efforts=list(spec.get("efforts") or []),
                pricing=dict(spec.get("pricing_usd_per_mtok") or {}),
                aliases=list(spec.get("aliases") or []),
            )

    @classmethod
    def load(cls, path: Path | None = None) -> "ModelRegistry":
        path = path or config_dir() / "models.yaml"
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def resolve(self, name: str) -> ModelInfo:
        """Resolve an alias or a full model ID. Unknown names pass through as unregistered IDs."""
        key = name.strip()
        if key in self.models:
            return self.models[key]
        found = self.by_id(key)
        if found:
            return found
        return ModelInfo(alias=key, id=key, known=False)

    def by_id(self, model_id: str) -> ModelInfo | None:
        mid = normalize_model_id(model_id)
        for info in self.models.values():
            if mid == info.id or mid in info.aliases:
                return info
        # dated snapshots, e.g. claude-haiku-4-5-20251001
        base = re.sub(r"-\d{8}$", "", mid)
        for info in self.models.values():
            if base == info.id:
                return info
        return None
