"""Operations shared by the MCP server and the CLI."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from paper_adversary import __version__
from paper_adversary.config import (
    INTAKE_ID,
    REFUTER_ROLES,
    STAGE_LABEL,
    STAGE_ORDER,
    ConfigError,
    check_config,
    load_config,
    plan_agents,
)
from paper_adversary.ingest import ingest
from paper_adversary.isolation import input_changed
from paper_adversary.pipeline import check_preconditions
from paper_adversary.prompts import load_prompt, load_rubric
from paper_adversary.providers import make_provider
from paper_adversary.registry import ModelRegistry
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore, list_runs
from paper_adversary.usage import cost_markdown, summarize
from paper_adversary.util import (
    atomic_write_json,
    atomic_write_text,
    fmt_duration,
    local_hm,
    parse_iso,
    read_json,
    read_jsonl,
    runs_root,
    sha256_file,
    sha256_text,
    utcnow,
    utcnow_iso,
)
from paper_adversary.worker import worker_info

ROLE_ALIASES = {
    "novelty": "novelty", "rigor": "rigor", "fit": "fit", "feasibility": "fit",
    "judge": "judge", "judges": "judge", "synthesis": "synthesis", "memo": "synthesis",
    "critic": "critic", "completeness": "critic", "completeness_critic": "critic",
    "intake": "intake", "profile": "intake", "orchestrator": "intake",
}


# ---------------------------------------------------------------- create


def create_run(paper_path: str | None = None, paper_text: str | None = None, title: str | None = None,
               venue: str | None = None, field: str | None = None, rubric: str | None = None,
               config_override=None, submission_type: str = "auto") -> dict:
    cfg, raw = load_config(config_override)
    registry = ModelRegistry.load()
    issues = check_config(cfg, registry)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        raise ConfigError("config problems:\n" + "\n".join(f"  {e}" for e in errors))
    if rubric:
        label, text = load_rubric(rubric)
        if not text:
            raise ConfigError(f"rubric '{rubric}' is empty or not found")
    result = ingest(paper_path, paper_text, submission_type)

    root = runs_root()
    root.mkdir(parents=True, exist_ok=True)
    detected_title = result.title
    kind = result.submission_type
    run_id = RunStore.new_run_id(root, title or detected_title, kind)
    specs = plan_agents(cfg, registry)
    prompts = {}
    for name in sorted({s.prompt_name for s in specs}):
        tpl = load_prompt(name)
        prompts[name] = tpl.sha256
    source: dict = {
        "format": result.source_format,
        "original_path": result.source_path,
        "pages": result.page_count,
        "chars": len(result.text_md),
        "words": len(result.text_md.split()),
        "est_tokens": int(len(result.text_md) / cfg.budget.chars_per_token),
        "sections": len(result.sections),
        "references": len(result.references),
    }
    metadata = {
        "run_id": run_id,
        "created_at": utcnow_iso(),
        "title": title or detected_title,
        "title_source": "user" if title else ("detected" if detected_title else None),
        "venue": venue,
        "field": field,
        "rubric": rubric or cfg.rubric,
        "submission_type": kind,
        "detected": {"title": detected_title, "abstract": (result.abstract or "")[:1500] or None},
        "source": source,
        "warnings": result.warnings,
        "prompts": prompts,
        "provider": cfg.provider.type,
        "package_version": __version__,
    }
    store = RunStore.create(root, run_id, metadata, raw)
    src = store.source_dir
    atomic_write_text(src / "extracted_text.md", result.text_md)
    atomic_write_json(src / "sections.json", result.to_index())
    atomic_write_json(src / "references.json", result.references)
    if paper_text and not paper_path:
        atomic_write_text(src / "paper_input.txt", paper_text)
    if result.source_path:
        original = Path(result.source_path)
        dest = src / ("paper" + original.suffix.lower())
        dest.write_bytes(original.read_bytes())
        source["sha256"] = sha256_file(dest)
    else:
        source["sha256"] = sha256_text(paper_text or "")
    metadata["source"] = source
    store.save_metadata(metadata)
    store.init_state([s.state_entry() for s in specs])
    store.event("run_created", title=metadata["title"], agents=len(specs), source_format=result.source_format)
    return {
        "run_id": run_id,
        "run_dir": str(store.dir),
        "metadata": {k: metadata[k] for k in ("title", "title_source", "venue", "field", "submission_type", "rubric")}
        | {"source": source, "abstract": metadata["detected"]["abstract"]},
        "warnings": result.warnings + [str(i) for i in issues if i.level == "warning"],
        "agents": [s.public() for s in specs],
        "prompt_versions": prompts,
    }


def format_created(info: dict) -> str:
    md = info["metadata"]
    src = md["source"]
    lines = [
        f"Created run {info['run_id']}",
        f"Folder: {info['run_dir']}",
        "",
        f"Title: {md.get('title') or '(not detected)'}" + (f"  [{md['title_source']}]" if md.get("title_source") else ""),
        f"Type: {md['submission_type']}   Venue: {md.get('venue') or 'not set'}   Field: {md.get('field') or 'not set'}",
        f"Source: {src['format']}, {src.get('pages') or '?'} pages, {src['words']:,} words (~{src['est_tokens']:,} "
        f"tokens est.), {src['sections']} sections, {src['references']} references parsed",
        f"Rubric: {md.get('rubric')}",
        "",
        "Planned agents:",
    ]
    by_role: dict[str, list[dict]] = {}
    for a in info["agents"]:
        by_role.setdefault(a["role"], []).append(a)
    for role in STAGE_ORDER:
        agents = by_role.get(role)
        if not agents:
            continue
        a0 = agents[0]
        tools = f", tools: {', '.join(a0['tools'])}" if a0["tools"] else ""
        fmt = ", gets the PDF" if a0["paper_format"] == "pdf" and src["format"] == "pdf" else ""
        lenses = [a["lens"] for a in agents if a.get("lens")]
        lines.append(f"  {STAGE_LABEL[role]}: {len(agents)} × {a0['model']} (effort {a0['effort']}), prompt "
                     f"{a0['prompt']}{tools}{fmt}")
        for a, lens in zip(agents, lenses):
            lines.append(f"      {a['agent_id']}: {lens}")
    if info["warnings"]:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in info["warnings"]]
    return "\n".join(lines)


# ---------------------------------------------------------------- status


def _agent_desc(a: dict, run_state: str) -> str:
    status = a["status"]
    if status in ("running", "retrying", "waiting_plan_limit") and run_state != "running":
        status = "interrupted"
    desc = status
    if status == "complete":
        bits = [f"{a.get('model')}/{a.get('effort')}"]
        if a.get("duration_s"):
            bits.append(fmt_duration(a["duration_s"]))
        if a.get("objections") is not None:
            bits.append(f"{a['objections']} objections")
        if a.get("judgments") is not None:
            bits.append(f"{a['judgments']} judgments")
        rc = a.get("refcheck") or {}
        if rc.get("checked"):
            bits.append(f"refs verified {rc.get('verified', 0)}/{rc['checked']}")
        if a.get("structured") is False and a["role"] not in ("synthesis", "critic"):
            bits.append("no structured block")
        desc += "  " + ", ".join(bits)
    elif status == "running":
        started = parse_iso(a.get("started_at"))
        elapsed = f", {fmt_duration((utcnow() - started).total_seconds())} elapsed" if started else ""
        desc += f"  attempt {a.get('attempts', 1)}{elapsed}"
    elif status in ("retrying", "waiting_plan_limit", "failed", "interrupted") and a.get("detail"):
        desc += f"  {a['detail'][:220]}"
    return desc


def run_state_label(state: dict, worker: dict) -> str:
    if worker.get("running"):
        return "running"
    agents = state.get("agents", {}).values()
    if agents and all(a["status"] == "complete" for a in agents):
        return "complete"
    return "idle"


def stale_agents(store: RunStore, state: dict) -> dict[str, list[str]]:
    stale: dict[str, list[str]] = {}
    for aid, a in state.get("agents", {}).items():
        if a["status"] != "complete":
            continue
        ctx = read_json(store.sidecar_path(aid, a["role"], ".context.json"), {}) or {}
        changed = [m.get("agent_id") or m.get("path") for m in ctx.get("manifest", []) if input_changed(store, m)]
        if changed:
            stale[aid] = [str(c) for c in changed]
    return stale


def plan_usage_line(info: dict | None) -> str:
    """Plan windows as Claude Code last reported them (rate_limit_event)."""
    if not info:
        return ""
    import datetime as _dt

    parts = []
    for name, window in (info.get("unifiedWindows") or {}).items():
        if not isinstance(window, dict):
            continue
        used = window.get("utilization")
        resets = window.get("resetsAt")
        when = (_dt.datetime.fromtimestamp(resets, _dt.timezone.utc).astimezone().strftime("%a %H:%M")
                if isinstance(resets, (int, float)) else "?")
        pct = f"{used * 100:.0f}%" if isinstance(used, (int, float)) else "?"
        parts.append(f"{name.replace('_', '-')} {pct} used (resets {when})")
    overage = "on" if info.get("isUsingOverage") or info.get("overageStatus") == "allowed" else "off"
    status = "" if info.get("status") in (None, "allowed") else f"; status {info.get('status')}"
    return (f"Plan usage (as of {local_hm(info.get('observed_at'))}): " + ", ".join(parts)
            + f"; paid overage {overage}{status}") if parts else ""


def next_step(state: dict, label: str) -> str:
    if label == "running":
        return "Wait, or poll get_run_status(run_id, wait_seconds=50). cancel_run stops the worker."
    agents = state["agents"]
    for stage, tool in (("novelty", "run_refuters"), ("rigor", "run_refuters"), ("fit", "run_refuters"),
                        ("judge", "run_judges"), ("synthesis", "run_synthesis"),
                        ("critic", "run_completeness_critic")):
        members = [a for a in agents.values() if a["role"] == stage]
        if members and any(a["status"] != "complete" for a in members):
            blockers = check_preconditions(state, stage)
            if blockers:
                return f"Blocked: {'; '.join(blockers)}. resume_run(run_id) retries unfinished agents."
            return f"Call {tool}(run_id) or resume_run(run_id) to continue."
    return ("Review complete. Read get_report(run_id, 'synthesis') and get_report(run_id, 'critic'); "
            "the judgment matrix is get_report(run_id, 'matrix').")


def status_text(run: str) -> str:
    store = RunStore.open(run)
    state = store.load_state()
    meta = store.load_metadata()
    worker = worker_info(store)
    label = run_state_label(state, worker)
    title = meta.get("title") or (meta.get("intake") or {}).get("title")
    lines = [f"Run: {store.run_id}" + (f' — "{title}"' if title else "")]
    if label == "running":
        lines.append(f"State: running (worker pid {worker.get('pid')}, started {local_hm(worker.get('started_at'))}, "
                     f"stages {', '.join(worker.get('stages') or [])})")
    else:
        last = state.get("last_job") or {}
        tail = f" — last job {last.get('outcome')}" + (f": {last['message']}" if last.get("message") else "") \
            if last else ""
        if worker.get("status") == "exited" and worker.get("outcome") == "crashed":
            tail = f" — worker crashed: {worker.get('message')}"
        lines.append(f"State: {label}{tail}")
    auth = (state.get("preflight") or {}).get("auth") or {}
    if auth:
        lines.append(f"Provider: {meta.get('provider')} ({auth.get('authMethod') or '?'} login)")
    plan = plan_usage_line(state.get("plan_usage"))
    if plan:
        lines.append(plan)
    lines.append("")
    agents = state.get("agents", {})
    for role in STAGE_ORDER:
        members = sorted((aid for aid, a in agents.items() if a["role"] == role),
                         key=lambda x: (len(x), x))
        if not members:
            continue
        if role in ("synthesis", "critic", "intake") and len(members) == 1:
            lines.append(f"{STAGE_LABEL[role]}: {_agent_desc(agents[members[0]], label)}")
            continue
        lines.append(f"{STAGE_LABEL[role]}:")
        lines += [f"  {aid:<7}{_agent_desc(agents[aid], label)}" for aid in members]
    failures = [(aid, f) for aid, a in agents.items() for f in a.get("failures") or []]
    lines.append("")
    if failures:
        lines.append("Failed calls and retries:")
        per: dict[str, list[dict]] = {}
        for aid, f in failures:
            per.setdefault(aid, []).append(f)
        for aid, fs in sorted(per.items()):
            kinds = ", ".join(f"{f['kind']} ({local_hm(f['at'])})" for f in fs[-4:])
            lines.append(f"  {aid}: {len(fs)} failed call(s), last: {kinds}")
    else:
        lines.append("Failed calls and retries: none")
    warnings = [(aid, w) for aid, a in agents.items() for w in a.get("warnings") or []]
    if warnings:
        lines.append("Warnings:")
        lines += [f"  {aid}: {w}" for aid, w in warnings[:12]]
    audits = {aid: a.get("isolation_audit") for aid, a in agents.items() if a.get("isolation_audit")}
    bad = [aid for aid, v in audits.items() if v != "pass"]
    lines.append(f"Isolation audit: {'all ' + str(len(audits)) + ' completed agents pass' if not bad else 'FAILED for ' + ', '.join(bad)}"
                 if audits else "Isolation audit: no completed agents yet")
    stale = stale_agents(store, state)
    if stale:
        lines.append("Stale outputs (an input changed after they ran; rerun them to refresh): "
                     + "; ".join(f"{aid} (inputs {', '.join(v[:4])})" for aid, v in stale.items()))
    lines += ["", f"Artifacts: {store.dir}"]
    for role in STAGE_ORDER:
        for aid in sorted(a for a, v in agents.items() if v["role"] == role and v["status"] == "complete"):
            lines.append(f"  {store.rel(store.report_path(aid, role))}")
        if role == "judge" and (store.role_dir("judge") / "judgment_matrix.md").exists():
            lines.append("  judges/judgment_matrix.md")
    lines += ["  logs/events.jsonl, logs/usage.jsonl, logs/api_usage.json", "", "Next: " + next_step(state, label)]
    return "\n".join(lines)


# ---------------------------------------------------------------- reports


def _section(body: str, heading: str) -> str:
    """Text of a '## heading' section (first paragraph), or ''."""
    import re

    m = re.search(rf"^##\s+(?:\d+\.\s*)?{re.escape(heading)}\s*$\n+(.+?)(?:\n\n|\n#|\Z)", body, re.M | re.S)
    return m.group(1).strip() if m else ""


def _page(text: str, offset: int, max_chars: int, label: str) -> str:
    total = len(text)
    offset = max(0, min(offset, total))
    chunk = text[offset : offset + max_chars]
    end = offset + len(chunk)
    if offset == 0 and end >= total:
        return chunk
    more = f"; call again with offset={end} for more" if end < total else ""
    return f"{chunk}\n\n[{label}: characters {offset}-{end} of {total}{more}]"


def get_report(run: str, report_type: str, agent_id: str | None = None, max_chars: int = 40000,
               offset: int = 0) -> str:
    store = RunStore.open(run)
    state = store.load_state()
    agents = state.get("agents", {})
    rtype = report_type.strip().lower()
    aid = agent_id.strip().upper() if agent_id else None

    if rtype in ("index", "list", "artifacts"):
        files = sorted(p for p in store.dir.rglob("*") if p.is_file() and "archive" not in p.parts
                       and p.name not in ("run.lock",))
        return "\n".join(store.rel(p) for p in files)
    if rtype in ("matrix", "judgment_matrix"):
        path = store.role_dir("judge") / "judgment_matrix.md"
        return path.read_text(encoding="utf-8") if path.exists() else "No judgment matrix yet (judges not complete)."
    if rtype == "metadata":
        return json.dumps(store.load_metadata(), indent=2)
    if rtype == "config":
        return store.config_path.read_text(encoding="utf-8")
    if rtype in ("claims", "ledger", "claims_ledger"):
        path = store.source_dir / "claims_ledger.md"
        return path.read_text(encoding="utf-8") if path.exists() else "No claims ledger (intake disabled or not run)."
    if rtype in ("paper", "source", "text"):
        return _page((store.source_dir / "extracted_text.md").read_text(encoding="utf-8"), offset, max_chars,
                     "extracted text")
    if rtype in ("sections", "toc"):
        index = read_json(store.source_dir / "sections.json", {}) or {}
        return "\n".join(f"{s['id']} [{s['kind']}] {'  ' * max(0, s['level'] - 1)}{s['title']} "
                         f"({s['chars']:,} chars" + (f", pp. {s['page_start']}-{s['page_end']})" if s.get("page_start") else ")")
                         for s in index.get("sections", [])) or "No sections detected."
    if rtype == "references":
        refs = read_json(store.source_dir / "references.json", []) or []
        return _page("\n".join(f"[{r['n']}] {r['text']}" for r in refs) or "No references parsed.", offset,
                     max_chars, "references")
    if rtype == "events":
        events = read_jsonl(store.events_path)[-60:]
        return "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
    if rtype in ("usage", "cost"):
        return cost_text(run)
    if rtype == "refcheck":
        targets = [aid] if aid else sorted(a for a, v in agents.items() if v["role"] == "novelty")
        out = []
        from paper_adversary.search.refcheck import refcheck_markdown

        for t in targets:
            data = read_json(store.sidecar_path(t, "novelty", ".refcheck.json"))
            out.append(refcheck_markdown(data, t) if data else f"{t}: no reference check yet")
        return "\n\n".join(out)
    if rtype in ("context", "manifest", "transcript", "prompt", "search_log"):
        if not aid or aid not in agents:
            raise ValueError(f"report_type '{rtype}' needs agent_id (one of {', '.join(sorted(agents))})")
        role = agents[aid]["role"]
        if rtype in ("context", "manifest"):
            data = read_json(store.sidecar_path(aid, role, ".context.json"))
            return json.dumps(data, indent=2) if data else f"No context manifest for {aid} yet."
        if rtype == "search_log":
            rows = read_jsonl(store.sidecar_path(aid, role, ".search_log.jsonl"))
            return "\n".join(f"{r['at']} {r['tool']} {json.dumps(r['args'])} -> {len(r.get('results', []))} results"
                             for r in rows) or f"No search calls logged for {aid}."
        attempts = sorted(store.agent_log_dir(aid).glob("attempt-*"), key=lambda p: int(p.name.split("-")[1]))
        if not attempts:
            return f"No attempts logged for {aid}."
        last = attempts[-1]
        if rtype == "prompt":
            system = (last / "system_prompt.md").read_text(encoding="utf-8") if (last / "system_prompt.md").exists() else ""
            user = (last / "user_prompt.md").read_text(encoding="utf-8") if (last / "user_prompt.md").exists() else ""
            return _page(f"=== SYSTEM ({last.name}) ===\n{system}\n\n=== USER ===\n{user}", offset, max_chars, "prompt")
        rows = read_jsonl(last / "transcript.jsonl")
        lines = []
        for r in rows:
            k = r.get("kind")
            if k == "tool_use":
                lines.append(f"TOOL {r.get('name')}: {json.dumps(r.get('input'), ensure_ascii=False)[:300]}")
            elif k == "tool_result":
                lines.append(f"  -> {str(r.get('content'))[:200].replace(chr(10), ' ')}")
            elif k == "assistant_text":
                lines.append(f"TEXT: {str(r.get('text'))[:300].replace(chr(10), ' ')}")
            elif k in ("init", "result"):
                lines.append(f"{k.upper()}: {json.dumps({x: r.get(x) for x in ('model', 'tools', 'apiKeySource', 'subtype', 'is_error', 'num_turns', 'total_cost_usd') if x in r})}")
        return _page("\n".join(lines) or "Empty transcript.", offset, max_chars, f"transcript {last.name}")

    role = ROLE_ALIASES.get(rtype)
    if role is None:
        raise ValueError(f"unknown report_type '{report_type}'. Use one of: novelty, rigor, fit, judge, synthesis, "
                         "critic, intake, matrix, claims, metadata, config, paper, sections, references, refcheck, "
                         "usage, events, context, transcript, prompt, search_log, index")
    members = sorted((a for a, v in agents.items() if v["role"] == role), key=lambda x: (len(x), x))
    if aid is None and len(members) == 1:
        aid = members[0]
    if aid is None:
        lines = [f"{STAGE_LABEL[role]} reports (pass agent_id to read one):"]
        for m in members:
            path = store.report_path(m, role)
            if not path.exists():
                lines.append(f"  {m}: {agents[m]['status']}")
                continue
            meta, body = read_report(path)
            data = (read_json(store.sidecar_path(m, role, ".json"), {}) or {}).get("data") or {}
            verdict = data.get("summary_verdict") or data.get("overall") or _section(body, "Verdict") \
                or _section(body, "Overall assessment") or ""
            counts = ", ".join(f"{meta[k]} {k}" for k in ("objections", "judgments") if meta.get(k) is not None)
            lens = f" [{meta['lens']}]" if meta.get("lens") else ""
            lines.append(f"  {m}{lens}: {agents[m]['status']}" + (f", {counts}" if counts else "")
                         + (f" — {' '.join(verdict.split())[:220]}" if verdict else ""))
        return "\n".join(lines)
    if aid not in agents or agents[aid]["role"] != role:
        raise ValueError(f"no {role} agent '{aid}' (have: {', '.join(members)})")
    path = store.report_path(aid, role)
    if not path.exists():
        return f"{aid} has no report yet (status: {agents[aid]['status']})."
    return _page(path.read_text(encoding="utf-8"), offset, max_chars, f"{aid} report")


# ---------------------------------------------------------------- cost, list, validate


def cost_text(run: str) -> str:
    store = RunStore.open(run)
    summary = summarize(read_jsonl(store.usage_path), ModelRegistry.load())
    atomic_write_json(store.usage_summary_path, summary)
    return cost_markdown(store.run_id, summary, store.load_metadata().get("provider", "?"))


def list_runs_text(limit: int = 20) -> str:
    runs = list_runs()[:limit]
    if not runs:
        return f"No runs yet in {runs_root()}."
    lines = [f"Runs in {runs_root()} (newest first):"]
    for s in runs:
        state = s.load_state()
        meta = s.load_metadata()
        label = run_state_label(state, worker_info(s))
        agents = state.get("agents", {}).values()
        done = sum(1 for a in agents if a["status"] == "complete")
        lines.append(f"  {s.run_id}: {label}, {done}/{len(agents)} agents complete — "
                     f"{(meta.get('title') or '')[:70]} ({local_hm(state.get('created_at'))})")
    return "\n".join(lines)


def claude_version(binary: str) -> str | None:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30,
                             stdin=subprocess.DEVNULL)
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


async def validate_text(config_override=None, probe_models: bool = False) -> str:
    lines = []
    try:
        cfg, raw = load_config(config_override)
    except ConfigError as exc:
        return f"Config: INVALID\n{exc}"
    registry = ModelRegistry.load()
    issues = check_config(cfg, registry)
    errors = [i for i in issues if i.level == "error"]
    lines.append("Config: " + ("OK" if not errors else f"{len(errors)} error(s)"))
    lines += [f"  {i}" for i in issues]
    specs = plan_agents(cfg, registry)
    lines.append("")
    lines.append("Resolved roles (alias -> model ID, effort):")
    seen = set()
    for s in specs:
        if s.role in seen:
            continue
        seen.add(s.role)
        n = sum(1 for x in specs if x.role == s.role)
        info = registry.resolve(s.model_id)
        lines.append(f"  {STAGE_LABEL[s.role]:<22} {n} × {s.model_alias} -> {s.model_id} ({s.effort})"
                     + ("" if info.known else "  [not in models.yaml]"))
    lines.append(f"Total agents per full run: {len(specs)}; parallel limit {cfg.concurrency.max_parallel_agents}")
    provider = make_provider(cfg.provider)
    lines.append("")
    if cfg.provider.type == "claude_code":
        try:
            binary = provider.binary()
            version = await asyncio.to_thread(claude_version, binary)
            lines.append(f"Claude Code CLI: {binary} ({version or 'version unknown'})")
        except Exception as exc:
            lines.append(f"Claude Code CLI: NOT FOUND ({exc})")
            return "\n".join(lines)
    auth = await provider.auth_status()
    lines.append(f"Auth: {'logged in' if auth.get('ok') else 'NOT logged in'}"
                 f" (method {auth.get('authMethod')}, plan {auth.get('subscriptionType')}); "
                 f"CLAUDE_CODE_OAUTH_TOKEN set: {'yes' if auth.get('oauth_token_in_env') else 'no'}")
    if auth.get("api_key_in_server_env") and not cfg.provider.allow_api_key:
        lines.append("  ANTHROPIC_API_KEY is set in the server environment; it is stripped from agent processes, "
                     "so agents still use your plan.")
    lines.append("  Note: 'logged in' only means credentials exist; an expired login shows up in the probe below.")
    if probe_models:
        from paper_adversary.context import tool_server_spec

        lines.append("")
        lines.append("Probes (one tiny low-effort call per model and tool setup, on your plan):")
        probe_root = runs_root() / ".cache" / "probe_logs"
        last_plan: dict | None = None
        setups: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        for s in specs:
            tools = [t for t in s.tools if not (t == "web_search" and not cfg.search.web_search)
                     and not (t == "web_fetch" and not cfg.search.web_fetch)]
            if s.paper_format == "pdf":
                tools.append("read_pdf")
            setups.setdefault((s.model_id, tuple(sorted(tools))), []).append(s.role)
        for (model, tools), roles in sorted(setups.items()):
            label = f"{model} [{', '.join(tools) or 'no tools'}] for {', '.join(sorted(set(roles)))}"
            log_dir = probe_root / f"{model}-{'+'.join(tools) or 'no-tools'}"
            server = (tool_server_spec(probe_root, cfg, "PROBE", log_dir / "tool_calls.jsonl", True, False)
                      if "literature" in tools else None)
            res = await provider.probe(model, log_dir, list(tools), server)
            plan_info = (res.get("runtime") or {}).get("rate_limit") or res.get("rate_limit")
            if plan_info:
                last_plan = plan_info
            if res.get("ok"):
                ctx = f", context window {res['context_window']:,}" if res.get("context_window") else ""
                lines.append(f"  OK      {label} (served by {', '.join(res.get('served_models') or ['?'])}{ctx})")
                for w in res.get("warnings") or []:
                    lines.append(f"          warning: {w}")
            else:
                lines.append(f"  FAILED  {label} — {res.get('error_kind')}: {res.get('error')}")
        if last_plan:
            lines.append(plan_usage_line({**last_plan, "observed_at": utcnow_iso()}))
    else:
        lines.append("Model availability is checked by a probe when a run starts (validate_config(probe_models=True) "
                     "checks now).")
    return "\n".join(lines)


def stages_for_phase(phase: str, state: dict | None = None) -> list[str]:
    phase = (phase or "all").strip().lower()
    if phase == "all":
        stages = ["novelty", "rigor", "fit"]
        if state is None or INTAKE_ID in state.get("agents", {}):
            stages.insert(0, "intake")
        return stages
    if phase in REFUTER_ROLES:
        return [phase]
    raise ValueError("phase must be one of: novelty, rigor, fit, all")


def resolve_rerun(state: dict, rerun_agents: list[str] | None, stages: list[str]) -> list[str]:
    ids = [a.strip().upper() for a in rerun_agents or [] if a and a.strip()]
    agents = state.get("agents", {})
    for aid in ids:
        if aid not in agents:
            raise ValueError(f"unknown agent '{aid}' (have: {', '.join(sorted(agents))})")
        if agents[aid]["role"] not in stages:
            raise ValueError(f"agent {aid} ({agents[aid]['role']}) is not part of stages {', '.join(stages)}")
    return ids

