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
from paper_adversary.isolation import IsolationGuard, input_changed
from paper_adversary import followup
from paper_adversary.pipeline import (
    RUNNABLE,
    apply_intake,
    batch_of,
    build_matrix,
    check_preconditions,
    finish_batch,
    supersede_memo,
)
from paper_adversary.reports import SEVERITY_RANK
from paper_adversary.prompts import load_prompt, load_rubric
from paper_adversary.providers import make_provider
from paper_adversary.registry import ModelRegistry
from paper_adversary.reports import read_report
from paper_adversary.store import RunStore, list_runs
from paper_adversary.usage import cost_markdown, summarize
from paper_adversary.util import (
    FileLock,
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
from paper_adversary.worker import WorkerBusy, worker_info

ROLE_ALIASES = {
    "novelty": "novelty", "rigor": "rigor", "fit": "fit", "feasibility": "fit",
    "judge": "judge", "judges": "judge", "synthesis": "synthesis",
    "adjudicator": "adjudicator", "adjudicators": "adjudicator", "revision": "revision", "revisions": "revision",
    "recheck": "recheck", "re-check": "recheck",
    "critic": "critic", "completeness": "critic", "completeness_critic": "critic",
    "intake": "intake", "profile": "intake", "orchestrator": "intake",
    "verifier": "verifier", "verifiers": "verifier",
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
    rubric_label, rubric_text = None, None
    if rubric:
        rubric_label, rubric_text = load_rubric(rubric)  # validated once; the run keeps its own copy
        if not rubric_text:
            raise ConfigError(f"rubric '{rubric}' is empty or not found")
    result = ingest(paper_path, paper_text, submission_type)

    root = runs_root()
    root.mkdir(parents=True, exist_ok=True)
    detected_title = result.title
    kind = result.submission_type
    run_id = RunStore.new_run_id(root, title or detected_title, kind)
    specs = plan_agents(cfg, registry)
    prompts = {}
    names = {s.prompt_name for s in specs} | {cfg.gates.repair.prompt}
    if cfg.verifier and cfg.verifier.enabled:
        names.add(cfg.verifier.prompt)
    for name in sorted(names):
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
        "rubric": rubric_label or cfg.rubric,
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
    if rubric_text:
        atomic_write_text(src / "rubric.md", rubric_text + "\n")
    if result.source_path:
        original = Path(result.source_path)
        dest = src / ("paper" + original.suffix.lower())
        dest.write_bytes(original.read_bytes())
        source["sha256"] = sha256_file(dest)
    else:
        source["sha256"] = sha256_text(paper_text or "")
    metadata["source"] = source
    store.save_metadata(metadata)
    store.init_state([s.state_entry() for s in specs], verification=bool(cfg.verifier and cfg.verifier.enabled))
    store.event("run_created", title=metadata["title"], agents=len(specs), source_format=result.source_format)
    return {
        "run_id": run_id,
        "run_dir": str(store.dir),
        "metadata": {k: metadata[k] for k in ("title", "title_source", "venue", "field", "submission_type", "rubric")}
        | {"source": source, "abstract": metadata["detected"]["abstract"]},
        "warnings": result.warnings + [str(i) for i in issues if i.level == "warning"],
        "agents": [s.public() for s in specs],
        "prompt_versions": prompts,
        "verifier": ({"model": registry.resolve(cfg.verifier.model).id, "effort": cfg.verifier.effort,
                      "prompt": cfg.verifier.prompt, "max_agents_per_batch": cfg.verifier.max_agents_per_batch}
                     if cfg.verifier and cfg.verifier.enabled else None),
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
    verifier = info.get("verifier")
    if verifier:
        lines.append(f"  {STAGE_LABEL['verifier']}: up to {verifier['max_agents_per_batch']} × {verifier['model']} "
                     f"(effort {verifier['effort']}), prompt {verifier['prompt']}; planned after the refuters, one "
                     "per prior paper behind a decisive novelty objection")
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
        elif isinstance(a.get("structured"), str) and a["structured"] != "ok":
            bits.append(f"structured data: {a['structured']}")
        desc += "  " + ", ".join(bits)
    elif status == "running":
        started = parse_iso(a.get("started_at"))
        elapsed = f", {fmt_duration((utcnow() - started).total_seconds())} elapsed" if started else ""
        desc += f"  attempt {a.get('attempts', 1)}{elapsed}"
    elif status in ("retrying", "waiting_plan_limit", "failed", "interrupted", "gating", "quarantined") \
            and a.get("detail"):
        desc += f"  {a['detail'][:220]}"
    if status == "complete":
        gate = a.get("gate") or {}
        if a.get("release"):
            desc += "  [released from quarantine]"
        elif not gate:
            desc += "  [not gated: created before gates existed]"
        elif gate.get("warnings"):
            desc += f"  [gate: {len(gate['warnings'])} warning(s)]"
    return desc


def run_state_label(state: dict, worker: dict) -> str:
    if worker.get("running"):
        return "running"
    agents = state.get("agents", {}).values()
    if agents and all(a["status"] == "complete" for a in agents):
        return "complete"
    return "idle"


def stale_agents(store: RunStore, state: dict) -> dict[str, list[str]]:
    """Complete agents whose inputs changed, became unusable (quarantined), or became available after they ran."""
    stale: dict[str, list[str]] = {}
    agents = state.get("agents", {})
    for aid, a in agents.items():
        if a["status"] != "complete":
            continue
        ctx = read_json(store.sidecar_path(aid, a["role"], ".context.json"), {}) or {}
        reasons = [str(m.get("agent_id") or m.get("path")) for m in ctx.get("manifest", []) if input_changed(store, m)]
        used = {m.get("agent_id") for m in ctx.get("manifest", []) if m.get("agent_id")}
        reasons += [f"{u} is now {agents[u]['status']}" for u in sorted(used)
                    if u in agents and agents[u]["status"] == "quarantined"]
        reasons += [f"{m} became available after it ran" for m in ctx.get("missing_agents") or []
                    if m in agents and agents[m]["status"] == "complete"]
        if reasons:
            stale[aid] = list(dict.fromkeys(reasons))
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


def next_step(state: dict, label: str, followup_configured: bool = False) -> str:
    if label == "running":
        return "Wait, or poll get_run_status(run_id, wait_seconds=50). cancel_run stops the worker."
    agents = state["agents"]
    quarantined = [aid for aid, a in agents.items() if a["status"] == "quarantined"]
    if quarantined:
        follow = [a for a in quarantined if int(agents[a].get("round") or 0)]
        how = ("rerun them (rerun_agents=[...] on the run_* tool of their stage"
               + ("; follow-up agents with run_followup(rerun_agents=[...])" if follow else "") + ")")
        return (f"Quarantined: {', '.join(quarantined)}. Read get_report(run_id, 'gates', agent_id) for the "
                f"reasons, then {how} or, if the reason does not hold, release them with a recorded reason "
                "(release_quarantine; isolation reasons need the CLI: paper-adversary release).")
    for stage, tool in (("novelty", "run_refuters"), ("rigor", "run_refuters"), ("fit", "run_refuters"),
                        ("judge", "run_judges"), ("synthesis", "run_synthesis"),
                        ("critic", "run_completeness_critic")):
        members = [a for a in agents.values() if a["role"] == stage]
        if members and any(a["status"] != "complete" for a in members):
            blockers = check_preconditions(state, stage)
            if blockers:
                return f"Blocked: {'; '.join(blockers)}. resume_run(run_id) retries unfinished agents."
            return f"Call {tool}(run_id) or resume_run(run_id) to continue."
    rounds = (state.get("followup") or {}).get("rounds") or {}
    pending_round = next((k for k, v in rounds.items() if v.get("status") != "complete"), None)
    if pending_round:
        return f"Follow-up round {pending_round} is unfinished: resume_run(run_id, through='followup') continues it."
    last = rounds.get(str(len(rounds))) if rounds else None
    if last and last.get("outcome") == "another_round":
        return ("Review complete; the re-check critic raised new serious items. run_followup(run_id, dry_run=true) "
                "shows the next round. The current memo is get_report(run_id, 'memo').")
    if last and last.get("outcome") == "max_rounds_reached" and last.get("new_items"):
        return (f"Review complete. The round cap was reached while the re-check critic still raised new serious "
                f"items ({', '.join(last['new_items'])}); they are not in the current memo. Read "
                "get_report(run_id, 'memo') and get_report(run_id, 'recheck').")
    return ("Review complete. Read get_report(run_id, 'memo') (the current memo) and get_report(run_id, 'critic'); "
            "the judgment matrix is get_report(run_id, 'matrix')."
            + (" run_followup(run_id, dry_run=true) shows what a follow-up round would do."
               if followup_configured and not rounds else ""))


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
    for batch_id, batch in ((state.get("verification") or {}).get("batches") or {}).items():
        counts = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted((batch.get("counts") or {}).items()))
        lines.append(f"Blind verification ({batch_id}): {batch.get('status')}"
                     + (f" — {len(batch.get('requests') or [])} request(s): {counts}" if counts else "")
                     + (f"; {batch['reason']}" if batch.get("reason") else ""))
    lines += followup.round_summary(store, state)
    gate = read_json(store.role_dir("judge") / "evidence_gate.json")
    if gate:
        flagged = gate.get("flagged_ids") or []
        lines.append("Evidence gate: " + (f"not shown: {', '.join(flagged)} (listed under 'Unverified threats')"
                                          if flagged else "every serious prior-work verdict is shown"))
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
    bad = [f"{aid} ({v})" for aid, v in audits.items() if v != "pass"]
    lines.append(f"Isolation audit: {'all ' + str(len(audits)) + ' completed agents pass' if not bad else 'NOT PASSED for ' + ', '.join(bad)}"
                 if audits else "Isolation audit: no completed agents yet")
    gated = {aid: a for aid, a in agents.items() if a["status"] in ("quarantined", "gating")
             or (a.get("gate") or {}).get("warnings") or a.get("release")}
    if gated:
        lines.append("Gates:")
        for aid, a in sorted(gated.items()):
            gate = a.get("gate") or {}
            if a["status"] == "quarantined":
                lines.append(f"  {aid}: QUARANTINED ({', '.join(gate.get('classes') or [])}) — "
                             + "; ".join(gate.get("reasons") or [])[:300])
            elif a["status"] == "gating":
                lines.append(f"  {aid}: output saved, checks not finished (resume_run finishes them)")
            elif a.get("release"):
                lines.append(f"  {aid}: released by {a['release'].get('via')} — {a['release'].get('reason', '')[:200]}")
            else:
                lines.append(f"  {aid}: passed with warnings — " + "; ".join(gate["warnings"])[:300])
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
    for k, rnd in sorted(((state.get("followup") or {}).get("rounds") or {}).items(), key=lambda kv: int(kv[0])):
        for aid in (rnd.get("adjudicators") or []) + [rnd.get("revision"), rnd.get("recheck")]:
            a = agents.get(aid) if aid else None
            if a and a["status"] == "complete":
                lines.append(f"  {store.rel(store.report_path(aid, a['role']))}")
        if (store.dir / "followup" / f"round-{k}" / "followup_matrix.md").exists():
            lines.append(f"  followup/round-{k}/items.md, followup/round-{k}/followup_matrix.md")
    configured = bool(store.load_raw_config().get("followup"))
    lines += ["  logs/events.jsonl, logs/usage.jsonl, logs/api_usage.json", "",
              "Next: " + next_step(state, label, configured)]
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
               offset: int = 0, rnd: int | None = None) -> str:
    store = RunStore.open(run)
    state = store.load_state()
    agents = state.get("agents", {})
    rtype = report_type.strip().lower()
    aid = agent_id.strip().upper() if agent_id else None
    rounds = (state.get("followup") or {}).get("rounds") or {}
    r = int(rnd) if rnd else (max((int(k) for k in rounds), default=1))
    if rtype == "memo":
        current = followup.current_memo(state)
        role = agents.get(current, {}).get("role", "synthesis")
        path = store.report_path(current, role)
        if not path.exists():
            return f"No memo yet ({current}: {agents.get(current, {}).get('status', 'not run')})."
        return _page(path.read_text(encoding="utf-8"), offset, max_chars, f"{current} memo (current)")
    if rtype == "followup":
        lines = followup.round_summary(store, state) or ["No follow-up round yet."]
        for k in sorted(rounds, key=int):
            result = read_json(followup.round_dir(store, int(k)) / "round.json")
            if result:
                lines.append(f"Round {k}: {result['outcome']}; new items {', '.join(i['id'] for i in result['new_items']) or 'none'}"
                             + (f"; re-raised {', '.join(i['id'] for i in result['disputed_repeats'])}"
                                if result.get("disputed_repeats") else ""))
        return "\n".join(lines)
    if rtype == "items":
        path = followup.round_dir(store, r) / "items.md"
        return path.read_text(encoding="utf-8") if path.exists() else f"No triaged items for round {r}."
    if rtype in ("followup_matrix", "follow_up_matrix"):
        path = followup.round_dir(store, r) / "followup_matrix.md"
        return path.read_text(encoding="utf-8") if path.exists() else f"No follow-up matrix for round {r}."

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
    if rtype == "evidence":
        from paper_adversary.evidence import evidence_markdown

        targets = [aid] if aid else sorted(a for a, v in agents.items() if v["role"] == "novelty")
        out = []
        for t_id in targets:
            rec = read_json(store.sidecar_path(t_id, "novelty", ".evidence.json"))
            out.append(evidence_markdown(rec) if rec else f"{t_id}: no evidence check yet")
        return "\n\n".join(out)
    if rtype in ("verification", "verifications"):
        from paper_adversary.verification import verification_markdown

        results = read_json(store.role_dir("verifier") / "results.json", {}) or {}
        return verification_markdown(results) or "No verification results yet."
    if rtype == "evidence_gate":
        path = store.role_dir("judge") / "evidence_gate.md"
        return path.read_text(encoding="utf-8") if path.exists() else "No evidence gate yet (judges not complete)."
    if rtype == "prior":
        from paper_adversary.search.fulltext import snapshot_index

        index = snapshot_index(store.prior_dir)
        rows = [f"{k}: {m.get('status')}" + (f" ({m.get('reason')})" if m.get("reason") else "")
                + f" — {m.get('title') or '?'}" + (f" [{m.get('source')} {m.get('version') or ''}]".rstrip()
                                                    if m.get("source") else "") for k, m in sorted(index.items())]
        return "Prior-work full texts read in this run:\n" + ("\n".join(rows) or "none")
    if rtype in ("gates", "gate"):
        targets = [aid] if aid else sorted(a for a, v in agents.items() if v.get("gate") or v["status"] == "gating")
        out = []
        for t in targets:
            if t not in agents:
                raise ValueError(f"unknown agent '{t}'")
            out.append(gate_text(store, t, agents[t]))
        return "\n\n".join(out) or "No gate records yet."
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
                         "critic, intake, verifier, memo, followup, items, followup_matrix, adjudicator, revision, "
                         "recheck, matrix, claims, metadata, config, paper, sections, references, "
                         "refcheck, evidence, verification, evidence_gate, prior, gates, usage, events, context, "
                         "transcript, prompt, search_log, index")
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
    banner = ""
    if agents[aid]["status"] in ("quarantined", "gating"):
        reasons = "; ".join((agents[aid].get("gate") or {}).get("reasons") or []) or "checks not finished"
        banner = (f"[{aid} is {agents[aid]['status'].upper()}: {reasons}. No later stage reads this report.]\n\n"
                  if offset == 0 else "")
    return banner + _page(path.read_text(encoding="utf-8"), offset, max_chars, f"{aid} report")


def gate_text(store: RunStore, aid: str, entry: dict) -> str:
    gate = read_json(store.gate_path(aid, entry["role"])) or {}
    lines = [f"{aid} ({entry['role']}): status {entry['status']}, gate {gate.get('verdict', 'none')}"]
    for c in gate.get("checks") or []:
        if c.get("result") != "pass":
            lines.append(f"  {c['result'].upper():<5} {c['check']} [{c['category']}]: {c.get('message') or ''}")
    st = gate.get("structured") or {}
    if st.get("source") not in (None, "ok", "n/a"):
        lines.append(f"  structured data: {st['source']} (as written: {st.get('as_written')})")
    rep = gate.get("repair") or {}
    if rep.get("attempts"):
        lines.append(f"  repair: {rep.get('outcome')} after {rep['attempts']} call(s) on {rep.get('model_requested')} "
                     f"({rep.get('effort')})" + (f"; rejected: {'; '.join(rep.get('rejections') or [])}"
                                                 if rep.get("rejections") else ""))
    audit = (gate.get("facts") or {}).get("audit") or {}
    for key in ("findings", "unverifiable", "denied"):
        for item in audit.get(key) or []:
            lines.append(f"  audit {key}: {item}")
    if entry.get("release"):
        r = entry["release"]
        lines.append(f"  released {r.get('at')} via {r.get('via')}: {r.get('reason')}")
    return "\n".join(lines)


def release_quarantine(run: str, agent_id: str, reason: str, via: str = "cli") -> str:
    """Make a quarantined output usable again, with a recorded reason. Isolation reasons need `via="cli"`."""
    store = RunStore.open(run)
    aid = agent_id.strip().upper()
    lock = FileLock(store.lock_path)
    if not lock.acquire(blocking=False):
        raise WorkerBusy("a worker is running for this run; wait for it or cancel it before releasing")
    try:
        state = store.load_state()
        entry = state["agents"].get(aid)
        if entry is None:
            raise ValueError(f"unknown agent '{aid}'")
        if entry["status"] != "quarantined":
            raise ValueError(f"{aid} is {entry['status']}, not quarantined")
        if len((reason or "").strip()) < 10:
            raise ValueError("give a reason of at least a few words; it is recorded with the release")
        gate = entry.get("gate") or {}
        if "isolation" in (gate.get("classes") or []) and via != "cli":
            raise PermissionError(
                f"{aid} was quarantined for an isolation reason ({'; '.join(gate.get('reasons') or [])}). Releasing "
                f"it is a deliberate override of the blindness guarantee, so it needs your terminal: "
                f'paper-adversary release {store.run_id} {aid} --reason "<why the finding does not hold>"')
        IsolationGuard(store).set_quarantined(aid, False)
        record = {"at": utcnow_iso(), "via": via, "reason": reason.strip(), "released_reasons": gate.get("reasons"),
                  "report_sha256": sha256_text(read_report(store.report_path(aid, entry["role"]))[1])}
        gate_path = store.gate_path(aid, entry["role"])
        sidecar = read_json(gate_path) or {}
        sidecar.setdefault("history", []).append({"event": "released", **record})
        atomic_write_json(gate_path, sidecar)
        side = read_json(store.sidecar_path(aid, entry["role"], ".json")) or {}

        def mutate(st: dict) -> None:
            a = st["agents"][aid]
            a.update(status="complete", release=record, detail=None,
                     objections=len(side.get("objections") or []) if entry["role"] in REFUTER_ROLES else None,
                     judgments=len(side.get("judgments") or []) if entry["role"] == "judge" else None)
        store.update_state(mutate)
        store.event("quarantine_released", agent_id=aid, via=via, reason=reason.strip())
        role, rnd = entry["role"], int(entry.get("round") or 0)
        if role == "intake":
            apply_intake(store, side.get("data"))
        if role == "revision":  # a released revised memo becomes the current memo, as a passing one would
            supersede_memo(store, aid, rnd)
        batch = batch_of(state, aid) if role == "verifier" else None
        if batch:  # its verdicts now count: fold them in and update the batch's outcome
            finish_batch(store, store.load_config(), batch)
        if role in ("adjudicator", "verifier") and rnd:
            followup.build_followup_matrix(store, store.load_config(), store.load_state(), rnd)
        if any(a["role"] == "judge" and a["status"] == "complete" for a in state["agents"].values()):
            build_matrix(store)
        after = stale_agents(store, store.load_state())
        note = (f" Agents that ran without it are now stale: {', '.join(sorted(after))}." if after else "")
        return f"Released {aid} (reason recorded).{note}"
    finally:
        lock.release()


def verification_batch(run: str, from_gate: bool, requests: list[dict] | None) -> tuple[str, int]:
    """Write a request file for a new verification batch; the worker plans and runs it. Returns (id, count)."""
    store = RunStore.open(run)
    state = store.load_state()
    if state.get("verification") is None:
        raise ValueError("this run was created without blind verification (verifier disabled in its config)")
    batches = (state["verification"].get("batches") or {})
    prefix = "gate" if from_gate else "user"
    batch_id = f"{prefix}{1 + sum(1 for b in batches if b.startswith(prefix))}"
    out: list[dict] = []
    if from_gate:
        from dataclasses import asdict

        from paper_adversary.verification import requests_from_objections

        gate = read_json(store.role_dir("judge") / "evidence_gate.json") or {}
        flagged = set(gate.get("flagged_ids") or [])
        results = (read_json(store.role_dir("verifier") / "results.json", {}) or {}).get("requests") or {}
        for req in requests_from_objections(store, state):  # the objections as they are now
            done = results.get(req.request_id) or {}
            answered = done.get("origin_hash") == req.origin_hash and done.get("status") in (
                "verified", "disputed", "cannot_tell")
            if set(req.origin_ids) & flagged and not answered:
                out.append({**asdict(req), "origin_ids": list(req.origin_ids)})
        if not out:
            raise ValueError("nothing to re-check: every flagged verdict already has a verifier's answer for its "
                             "current objection, or none is flagged (see get_report(run_id, 'evidence_gate'))")
    for i, req in enumerate(requests or [], start=1):
        ident = str(req.get("prior") or req.get("identifier") or "").strip()
        if not ident or not (req.get("claim_quote") or req.get("claim_location")):
            raise ValueError("each request needs 'prior' (arXiv ID, DOI or title) and 'claim_quote' (the "
                             "submission's exact words) or 'claim_location'")
        out.append({"request_id": f"{batch_id.upper()}-{i}", "origin": "user", "origin_ids": [],
                    "prior": {"identifier": ident, "title": req.get("title")}, "claim_quote": req.get("claim_quote"),
                    "claim_location": req.get("claim_location"), "note": req.get("note")})
    if not out:
        raise ValueError("give from_gate=true or at least one request")
    atomic_write_json(store.role_dir("verifier") / "batches" / f"{batch_id}.requests.json", out)
    return batch_id, len(out)


async def add_prior_fulltext(run: str, identifier: str, path: str, title: str | None = None) -> str:
    from paper_adversary.search.fulltext import FullTextStore

    store = RunStore.open(run)
    cfg = store.load_config()
    submission = (store.source_dir / "extracted_text.md").read_text(encoding="utf-8")
    async with FullTextStore(runs_root() / ".cache", cfg.search.fulltext, list(cfg.search.providers),
                             prior_dir=store.prior_dir) as fetch:
        result = await fetch.add_user_file(identifier, Path(path), title, submission)
    if not result.available:
        return f"Could not extract text from {path}: {result.reason} {result.detail or ''}".strip()
    store.event("prior_fulltext_added", identifier=identifier, key=result.key, sha256=result.sha256)
    return (f"Added {result.title or identifier} as {result.key} ({len(result.text_md or ''):,} characters, "
            f"{result.page_count or '?'} pages). Run run_verification(run_id, from_gate=true) to re-check the "
            "verdicts that waited on it.")


def resume_stages(state: dict, through: str = "critic") -> list[str]:
    """Stages that still have work (runnable agents, or outputs whose checks did not finish), up to `through`.
    With through="followup", an unfinished follow-up round (or the next one, if the last asked for it) too."""
    last = {"refuters": "fit", "judges": "judge", "synthesis": "synthesis", "critic": "critic",
            "followup": "critic"}[through]
    upto = STAGE_ORDER[: STAGE_ORDER.index(last) + 1]
    agents = [a for a in state.get("agents", {}).values() if not int(a.get("round") or 0)]
    stages = [s for s in upto if any(a["role"] == s and (a["status"] in RUNNABLE or a["status"] == "gating")
                                     for a in agents)]
    if through == "followup":
        rounds = (state.get("followup") or {}).get("rounds") or {}
        unfinished = any(v.get("status") != "complete" for v in rounds.values()) or \
            bool((state.get("followup") or {}).get("stale"))
        wants_more = bool(rounds) and rounds.get(str(len(rounds)), {}).get("outcome") == "another_round"
        if unfinished or wants_more or (not rounds and stages):
            stages.append("followup")
    return stages


def resolve_followup_rerun(state: dict, ids: list[str] | None) -> list[str]:
    out = [a.strip().upper() for a in ids or [] if a and a.strip()]
    for aid in out:
        entry = state.get("agents", {}).get(aid)
        if entry is None or not int(entry.get("round") or 0) or entry["status"] == "superseded":
            raise ValueError(f"{aid} is not an agent of a current follow-up round")
    return out


def full_review_stages(store: RunStore, info: dict) -> list[str]:
    stages = [s for s in STAGE_ORDER if any(a["role"] == s for a in info["agents"])]
    cfg = store.load_config()
    if cfg.followup is not None and cfg.followup.auto_start and cfg.followup.max_rounds > 0:
        stages.append("followup")
    return stages


def followup_plan(run: str) -> str:
    """What the next follow-up round would do, with a rough size, without starting anything."""
    store = RunStore.open(run)
    state = store.load_state()
    cfg = store.load_config()
    fc = cfg.followup
    if fc is None:
        return "Nothing to do: this run's config has no follow-up section."
    if (state.get("followup") or {}).get("stale"):
        return (f"The follow-up is out of date ({state['followup']['stale']}); running starts it over from round 1 "
                "(earlier rounds are archived).")
    rounds = (state.get("followup") or {}).get("rounds") or {}
    open_round = next((int(k) for k, v in rounds.items() if v.get("status") != "complete"), None)
    if open_round:
        return f"Round {open_round} is unfinished; running continues it."
    if rounds and rounds.get(str(len(rounds)), {}).get("outcome") != "another_round":
        return f"Nothing to do: the last round ended '{rounds[str(len(rounds))].get('outcome')}'."
    r = len(rounds) + 1
    if r > fc.max_rounds:
        return f"Nothing to do: the configured maximum of {fc.max_rounds} round(s) is reached."
    critic_id, critic_role = followup.round_critic(state, r)
    side = read_json(store.sidecar_path(critic_id, critic_role, ".json"), {}) or {}
    items = followup.critic_items(side.get("data"), critic_id)
    if not items:
        return f"Nothing to follow up: {critic_id} raised no structured items."
    floor = SEVERITY_RANK.get(fc.min_severity, 2)
    serious = [i for i in items if SEVERITY_RANK.get(i.get("severity") or "", -1) >= floor]
    refs = [i for i in items if i.get("candidate_references")]
    registry = ModelRegistry.load()
    lines = [f"Follow-up round {r} on {critic_id}'s {len(items)} item(s): {len(serious)} at or above "
             f"{fc.min_severity} go to {fc.adjudicators} adjudicators; {len(refs)} name prior work to verify.",
             f"Agents: {fc.adjudicators} × {registry.resolve(fc.adjudicator.model).id} ({fc.adjudicator.effort}), "
             f"revised memo S{followup.next_free(state, 'S')} on {registry.resolve(fc.revision.model).id} "
             f"({fc.revision.effort}), re-check C{followup.next_free(state, 'C')} on "
             f"{registry.resolve(fc.recheck.model).id} ({fc.recheck.effort})"
             + (f", plus up to {min(len(refs), cfg.verifier.max_agents_per_batch)} blind verifier(s)"
                if refs and cfg.verifier and cfg.verifier.enabled else "") + ".",
             "Each of these reads about as much as the completeness critic did (the whole review)."]
    if not serious and not refs:
        lines[0] = f"Nothing to follow up: {critic_id}'s items are all below {fc.min_severity}."
    return "\n".join(lines)


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
        held = sum(1 for a in agents if a["status"] == "quarantined")
        lines.append(f"  {s.run_id}: {label}, {done}/{len(agents)} agents complete"
                     + (f", {held} quarantined" if held else "") + " — "
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
        if cfg.verifier and cfg.verifier.enabled:
            setups.setdefault((registry.resolve(cfg.verifier.model).id, tuple(sorted(cfg.verifier.tools))),
                              []).append("verifier")
        if cfg.gates.repair.model != "agent":
            setups.setdefault((registry.resolve(cfg.gates.repair.model).id, ()), []).append("repair")
        for (model, tools), roles in sorted(setups.items()):
            label = f"{model} [{', '.join(tools) or 'no tools'}] for {', '.join(sorted(set(roles)))}"
            log_dir = probe_root / f"{model}-{'+'.join(tools) or 'no-tools'}"
            mode = ("open" if "literature" in tools else "scoped") if "prior_text" in tools else None
            server = (tool_server_spec(probe_root, cfg, "PROBE", log_dir / "tool_calls.jsonl", "literature" in tools,
                                       False, mode, [] if mode == "scoped" else None)
                      if "literature" in tools or "prior_text" in tools else None)
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

