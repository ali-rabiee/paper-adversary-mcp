"""Usage and cost accounting by phase.

Only numbers the provider returned are counted. A call that returned no usage
(e.g. killed on timeout) is listed as missing, never estimated. Two cost
figures are shown side by side: the API-equivalent cost Claude Code reports
itself, and an estimate from the prices in config/models.yaml. On a Claude
plan neither is billed per token.
"""

from __future__ import annotations

from paper_adversary.config import STAGE_LABEL, STAGE_ORDER
from paper_adversary.registry import ModelRegistry
from paper_adversary.util import fmt_duration

_TOKEN_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def _bucket() -> dict:
    return {"calls": 0, "failed_calls": 0, "calls_without_usage": 0, "input_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0,
            "web_search_requests": 0, "web_fetch_requests": 0, "reported_cost_usd": 0.0,
            "reported_cost_missing": 0, "estimated_cost_usd": 0.0, "estimate_incomplete": False,
            "agent_seconds": 0.0, "api_seconds": 0.0, "models": set(), "missing": []}


def _mu(entry: dict, *keys: str) -> int:
    for key in keys:
        if isinstance(entry.get(key), (int, float)):
            return int(entry[key])
    return 0


def estimate_cost(rec: dict, registry: ModelRegistry) -> tuple[float | None, list[str]]:
    """Estimate from per-model usage (modelUsage) and the registry's prices."""
    usage = rec.get("usage") or {}
    model_usage = rec.get("model_usage") or {}
    unknown: list[str] = []
    split_1h = 0.0
    creation = usage.get("cache_creation") or {}
    total_creation = (creation.get("ephemeral_1h_input_tokens") or 0) + (creation.get("ephemeral_5m_input_tokens") or 0)
    if total_creation:
        split_1h = (creation.get("ephemeral_1h_input_tokens") or 0) / total_creation
    entries: list[tuple[str, dict]] = []
    if model_usage:
        entries = list(model_usage.items())
    elif usage:
        entries = [(rec.get("model_requested") or "", {
            "inputTokens": usage.get("input_tokens"), "outputTokens": usage.get("output_tokens"),
            "cacheReadInputTokens": usage.get("cache_read_input_tokens"),
            "cacheCreationInputTokens": usage.get("cache_creation_input_tokens"),
            "webSearchRequests": (usage.get("server_tool_use") or {}).get("web_search_requests")})]
    if not entries:
        return None, unknown
    total = 0.0
    for model_id, mu in entries:
        info = registry.by_id(model_id)
        if not info or not info.pricing:
            unknown.append(model_id or "?")
            continue
        p = info.pricing
        cache_write = _mu(mu, "cacheCreationInputTokens")
        total += (_mu(mu, "inputTokens") * p.get("input", 0)
                  + _mu(mu, "outputTokens") * p.get("output", 0)
                  + _mu(mu, "cacheReadInputTokens") * p.get("cache_read", 0)
                  + cache_write * (1 - split_1h) * p.get("cache_write_5m", 0)
                  + cache_write * split_1h * p.get("cache_write_1h", 0)) / 1_000_000
        if registry.web_search_usd_per_1k:
            total += _mu(mu, "webSearchRequests") * registry.web_search_usd_per_1k / 1000
    return total, unknown


def summarize(records: list[dict], registry: ModelRegistry) -> dict:
    phases: dict[str, dict] = {}
    for rec in records:
        stage = rec.get("stage") or "other"
        b = phases.setdefault(stage, _bucket())
        b["calls"] += 1
        if rec.get("outcome") != "complete":
            b["failed_calls"] += 1
        if rec.get("model_requested"):
            b["models"].add(rec["model_requested"])
        usage = rec.get("usage")
        if rec.get("duration_ms"):
            b["agent_seconds"] += rec["duration_ms"] / 1000
        if rec.get("duration_api_ms"):
            b["api_seconds"] += rec["duration_api_ms"] / 1000
        if not usage:
            b["calls_without_usage"] += 1
            b["missing"].append(f"{rec.get('agent_id')} attempt {rec.get('attempt')} ({rec.get('outcome')})")
            continue
        for key in _TOKEN_KEYS:
            b[key] += int(usage.get(key) or 0)
        stu = usage.get("server_tool_use") or {}
        b["web_search_requests"] += int(stu.get("web_search_requests") or 0)
        b["web_fetch_requests"] += int(stu.get("web_fetch_requests") or 0)
        if rec.get("reported_cost_usd") is not None:
            b["reported_cost_usd"] += float(rec["reported_cost_usd"])
        else:
            b["reported_cost_missing"] += 1
        est, unknown = estimate_cost(rec, registry)
        if est is None or unknown:
            b["estimate_incomplete"] = True
        b["estimated_cost_usd"] += est or 0.0
    total = _bucket()
    for b in phases.values():
        for key, val in b.items():
            if key == "models":
                total["models"] |= val
            elif key == "missing":
                total["missing"] += val
            elif key == "estimate_incomplete":
                total[key] = total[key] or val
            else:
                total[key] += val
    for b in (*phases.values(), total):
        b["models"] = sorted(b["models"])
    ordered = {s: phases[s] for s in STAGE_ORDER if s in phases}
    ordered.update({s: b for s, b in phases.items() if s not in ordered})
    return {"phases": ordered, "total": total, "pricing_source": registry.pricing_source}


def _k(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def phase_label(stage: str) -> str:
    """'round1:adjudicator' -> 'Round 1 · Adjudicators'; base stages by their label."""
    if stage.startswith("round") and ":" in stage:
        rnd, role = stage.split(":", 1)
        return f"Round {rnd.removeprefix('round')} · {STAGE_LABEL.get(role, role)}"
    return STAGE_LABEL.get(stage, stage)


def cost_markdown(run_id: str, summary: dict, provider: str) -> str:
    rows = [
        f"Run: {run_id} — usage and cost by phase",
        "",
        ("Agents run on your Claude plan through Claude Code, so nothing below was billed per token. "
         "'Reported' is the API-equivalent cost Claude Code computed; 'estimate' applies the prices in "
         "config/models.yaml to the same token counts. Both cover only calls that returned usage.")
        if provider == "claude_code" else f"Provider: {provider}.",
        "",
        "| Phase | Calls (failed) | Input | Cache write | Cache read | Output | Web searches | Reported $ | "
        "Estimate $ | Agent time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def line(name: str, b: dict) -> str:
        rep = f"{b['reported_cost_usd']:.2f}" + ("*" if b["reported_cost_missing"] else "")
        est = f"{b['estimated_cost_usd']:.2f}" + ("*" if b["estimate_incomplete"] else "")
        return (f"| {name} | {b['calls']} ({b['failed_calls']}) | {_k(b['input_tokens'])} | "
                f"{_k(b['cache_creation_input_tokens'])} | {_k(b['cache_read_input_tokens'])} | "
                f"{_k(b['output_tokens'])} | {b['web_search_requests']} | {rep} | {est} | "
                f"{fmt_duration(b['agent_seconds'])} |")

    for stage, b in summary["phases"].items():
        rows.append(line(phase_label(stage), b))
    rows.append(line("**Total**", summary["total"]))
    total = summary["total"]
    notes = []
    if total["calls_without_usage"]:
        notes.append(f"{total['calls_without_usage']} call(s) returned no usage and are not counted: "
                     + "; ".join(total["missing"][:12]))
    if total["reported_cost_missing"]:
        notes.append("* some calls had no reported cost")
    if total["estimate_incomplete"]:
        notes.append("* the estimate skips models without prices in config/models.yaml")
    notes.append(f"Prices: {summary.get('pricing_source') or 'config/models.yaml'}")
    return "\n".join(rows + [""] + notes) + "\n"
