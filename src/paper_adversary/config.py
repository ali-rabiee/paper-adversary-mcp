"""Pipeline configuration: loading, merging, validation and agent planning."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paper_adversary.registry import ModelInfo, ModelRegistry
from paper_adversary.util import config_dir

REFUTER_ROLES = ("novelty", "rigor", "fit")
REVIEW_ROLES = ("novelty", "rigor", "fit", "judge", "synthesis", "critic")
ROLE_PREFIX = {"novelty": "N", "rigor": "R", "fit": "F", "judge": "J", "synthesis": "S", "critic": "C"}
INTAKE_ID = "INTAKE"
STAGE_ORDER = ("intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic")
STAGE_LABEL = {
    "intake": "Intake (orchestrator)",
    "novelty": "Novelty",
    "rigor": "Rigor",
    "fit": "Fit / feasibility",
    "judge": "Judges",
    "synthesis": "Synthesis",
    "critic": "Completeness critic",
}

ToolName = Literal["web_search", "web_fetch", "literature"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleConfig(_Strict):
    model: str
    effort: str
    agents: int = Field(1, ge=1, le=12)
    prompt: str
    lenses: bool = False
    tools: list[ToolName] = Field(default_factory=list)
    paper_format: Literal["text", "pdf"] = "text"
    timeout_minutes: float = Field(60, gt=0)
    max_turns: int | None = Field(None, ge=1)


class OrchestratorConfig(_Strict):
    enabled: bool = True
    model: str
    effort: str
    prompt: str = "intake_v1"
    timeout_minutes: float = Field(30, gt=0)


class ProviderConfig(_Strict):
    type: Literal["claude_code", "mock"] = "claude_code"
    claude_binary: str | None = None
    allow_api_key: bool = False
    safe_mode: bool = True
    pdf_delivery: Literal["read_tool"] = "read_tool"
    preflight: Literal["probe", "auth_only", "off"] = "probe"
    probe_cache_hours: float = 24
    idle_timeout_minutes: float = Field(25, gt=0)
    env_passthrough: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)
    mock: dict[str, Any] = Field(default_factory=dict)  # failure injection for tests


class ConcurrencyConfig(_Strict):
    max_parallel_agents: int = Field(3, ge=1, le=32)
    stagger_seconds: float = Field(3, ge=0)


class RetryConfig(_Strict):
    max_attempts: int = Field(4, ge=1, le=20)
    base_delay_seconds: float = Field(30, ge=0)
    max_delay_seconds: float = Field(900, ge=0)
    jitter: float = Field(0.25, ge=0, le=1)


class PlanLimitConfig(_Strict):
    policy: Literal["wait", "fail"] = "wait"
    max_wait_hours: float = Field(6, ge=0)
    default_wait_minutes: float = Field(30, ge=0)


class SearchConfig(_Strict):
    providers: list[Literal["openalex", "semantic_scholar", "arxiv", "crossref"]] = Field(
        default_factory=lambda: ["openalex", "semantic_scholar", "arxiv", "crossref"]
    )
    max_results: int = Field(10, ge=1, le=50)
    cache_ttl_days: float = 14
    refcheck: bool = True
    refcheck_max_refs: int = 60
    web_search: bool = True
    web_fetch: bool = True


class BudgetConfig(_Strict):
    chars_per_token: float = Field(3.2, gt=0.5)
    safety_margin_tokens: int = Field(12000, ge=0)
    output_reserve_tokens: int = Field(64000, ge=1000)
    section_priority: dict[str, list[str]] = Field(default_factory=dict)


class PipelineConfig(_Strict):
    orchestrator: OrchestratorConfig
    novelty: RoleConfig
    rigor: RoleConfig
    fit: RoleConfig
    judge: RoleConfig
    synthesis: RoleConfig
    critic: RoleConfig
    rubric: str = "default_v1"
    lens_set: str = "lenses_v1"
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    plan_limit: PlanLimitConfig = Field(default_factory=PlanLimitConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    def role(self, role: str) -> RoleConfig:
        return getattr(self, role)


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def parse_override(override: Any) -> dict:
    """Accept a dict, a JSON/YAML string, or a path to a YAML/JSON file."""
    if override is None or override == "":
        return {}
    if isinstance(override, dict):
        return override
    if isinstance(override, (str, Path)):
        text = str(override).strip()
        candidate = Path(text).expanduser()
        if "\n" not in text and len(text) < 1024 and candidate.suffix in {".yaml", ".yml", ".json"}:
            if not candidate.exists():
                raise ConfigError(f"config file not found: {candidate}")
            text = candidate.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"config override is neither valid YAML nor JSON: {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError("config override must be a mapping")
        return data
    raise ConfigError(f"unsupported config override type: {type(override).__name__}")


# Settings that decide what an agent process can reach or who pays for it. Per-run overrides arrive through
# MCP tool arguments chosen by a model (which reads untrusted paper text), so only files the user edits
# (config/default.yaml, $PAPER_ADVERSARY_CONFIG) may set them.
PROTECTED_PROVIDER_KEYS = ("allow_api_key", "extra_args", "env_passthrough", "claude_binary", "safe_mode")

# `claude` flags that would widen an agent's access or replace the sandbox; never allowed in extra_args.
FORBIDDEN_EXTRA_ARGS = (
    "--add-dir", "--dangerously-skip-permissions", "--allow-dangerously-skip-permissions", "--permission-mode",
    "--permission-prompts", "--settings", "--setting-sources", "--mcp-config", "--strict-mcp-config", "--tools",
    "--allowedTools", "--allowed-tools", "--disallowedTools", "--disallowed-tools", "--system-prompt",
    "--system-prompt-file", "--append-system-prompt", "--append-system-prompt-file", "--agents", "--agent",
    "--plugin-dir", "--plugin-url", "--continue", "-c", "--resume", "-r", "--fork-session", "--restricted",
    "--safe-mode", "--bare", "--file", "--chrome", "--ide", "--model", "--effort",
)


def load_raw_config(override: Any = None) -> dict:
    raw = yaml.safe_load((config_dir() / "default.yaml").read_text(encoding="utf-8")) or {}
    user_file = os.environ.get("PAPER_ADVERSARY_CONFIG")
    if user_file:
        raw = deep_merge(raw, parse_override(Path(user_file)))
    per_run = parse_override(override)
    locked = [k for k in PROTECTED_PROVIDER_KEYS if k in (per_run.get("provider") or {})]
    if locked:
        raise ConfigError(f"provider.{', provider.'.join(locked)} cannot be set per run; set it in "
                          "config/default.yaml or the file named by $PAPER_ADVERSARY_CONFIG")
    return deep_merge(raw, per_run)


def validate_config(raw: dict) -> PipelineConfig:
    try:
        return PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("invalid pipeline config:\n" + "\n".join(lines)) from exc


def load_config(override: Any = None) -> tuple[PipelineConfig, dict]:
    raw = load_raw_config(override)
    return validate_config(raw), raw


# ---------------------------------------------------------------- resolution


@dataclass
class ConfigIssue:
    level: Literal["error", "warning"]
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


def check_model_and_effort(
    where: str, model_name: str, effort: str, registry: ModelRegistry
) -> tuple[ModelInfo, list[ConfigIssue]]:
    issues: list[ConfigIssue] = []
    info = registry.resolve(model_name)
    if not info.known:
        if model_name.startswith("claude-"):
            issues.append(ConfigIssue("warning", where, f"model '{model_name}' is not in config/models.yaml; "
                                      "context window and prices are unknown until the preflight probe runs"))
        else:
            issues.append(ConfigIssue("error", where, f"unknown model alias '{model_name}' "
                                      f"(known: {', '.join(sorted(registry.models))})"))
    if effort not in registry.efforts:
        issues.append(ConfigIssue("error", where, f"effort '{effort}' is not one of {registry.efforts}"))
    elif info.known and info.efforts and effort not in info.efforts:
        issues.append(ConfigIssue("error", where, f"{info.id} does not support effort '{effort}' "
                                  f"(supported: {info.efforts})"))
    elif info.known and not info.efforts:
        issues.append(ConfigIssue("error", where, f"{info.id} has no effort control"))
    return info, issues


def check_config(cfg: PipelineConfig, registry: ModelRegistry) -> list[ConfigIssue]:
    from paper_adversary.prompts import prompt_exists, rubric_exists

    issues: list[ConfigIssue] = []
    if cfg.orchestrator.enabled:
        _, found = check_model_and_effort("orchestrator", cfg.orchestrator.model, cfg.orchestrator.effort, registry)
        issues += found
        if not prompt_exists(cfg.orchestrator.prompt):
            issues.append(ConfigIssue("error", "orchestrator", f"prompt '{cfg.orchestrator.prompt}' not found"))
    for role in REVIEW_ROLES:
        rc = cfg.role(role)
        _, found = check_model_and_effort(role, rc.model, rc.effort, registry)
        issues += found
        if not prompt_exists(rc.prompt):
            issues.append(ConfigIssue("error", role, f"prompt '{rc.prompt}' not found in prompts/"))
        if rc.tools and role not in REFUTER_ROLES:
            issues.append(ConfigIssue("error", role, "only refuters may use search tools"))
    if not rubric_exists(cfg.rubric):
        issues.append(ConfigIssue("warning", "rubric", f"rubric '{cfg.rubric}' not found; judges will get none"))
    if cfg.provider.allow_api_key:
        issues.append(ConfigIssue("warning", "provider", "allow_api_key is on: agents may bill ANTHROPIC_API_KEY"))
    bad = sorted({a.split("=", 1)[0] for a in cfg.provider.extra_args if a.split("=", 1)[0] in FORBIDDEN_EXTRA_ARGS})
    if bad:
        issues.append(ConfigIssue("error", "provider.extra_args", f"{', '.join(bad)} would weaken agent isolation "
                                  "or override what the pipeline controls"))
    return issues


@dataclass
class AgentSpec:
    agent_id: str
    role: str  # intake | novelty | rigor | fit | judge | synthesis | critic
    index: int  # 1-based within the role
    model_alias: str
    model_id: str
    effort: str
    prompt_name: str
    tools: list[str] = field(default_factory=list)
    paper_format: str = "text"
    timeout_s: float = 3600
    max_turns: int | None = None
    lens: dict | None = None

    @property
    def stage(self) -> str:
        return self.role

    def state_entry(self) -> dict:
        """Everything needed to rebuild this spec later (stored in state.json at run creation)."""
        return {**self.public(), "index": self.index, "timeout_s": self.timeout_s, "max_turns": self.max_turns,
                "lens_detail": self.lens}

    def public(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "model": self.model_id,
            "model_alias": self.model_alias,
            "effort": self.effort,
            "prompt": self.prompt_name,
            "tools": self.tools,
            "paper_format": self.paper_format,
            "lens": (self.lens or {}).get("title"),
        }


def plan_agents(cfg: PipelineConfig, registry: ModelRegistry) -> list[AgentSpec]:
    from paper_adversary.prompts import assign_lens

    specs: list[AgentSpec] = []
    if cfg.orchestrator.enabled:
        info = registry.resolve(cfg.orchestrator.model)
        specs.append(AgentSpec(
            agent_id=INTAKE_ID, role="intake", index=1, model_alias=cfg.orchestrator.model, model_id=info.id,
            effort=cfg.orchestrator.effort, prompt_name=cfg.orchestrator.prompt,
            timeout_s=cfg.orchestrator.timeout_minutes * 60,
        ))
    for role in REVIEW_ROLES:
        rc = cfg.role(role)
        info = registry.resolve(rc.model)
        for i in range(1, rc.agents + 1):
            specs.append(AgentSpec(
                agent_id=f"{ROLE_PREFIX[role]}{i}", role=role, index=i, model_alias=rc.model, model_id=info.id,
                effort=rc.effort, prompt_name=rc.prompt, tools=list(rc.tools), paper_format=rc.paper_format,
                timeout_s=rc.timeout_minutes * 60, max_turns=rc.max_turns,
                lens=assign_lens(role, i, rc.agents, cfg.lens_set) if rc.lenses else None,
            ))
    return specs


def dump_yaml(data: Any) -> str:
    return yaml.safe_dump(json.loads(json.dumps(data, default=str)), sort_keys=False, allow_unicode=True)
