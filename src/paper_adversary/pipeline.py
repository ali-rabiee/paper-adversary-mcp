"""Stage orchestration: refuters -> judges -> synthesis -> completeness critic.

Runs inside the worker process. Guarantees:
  * refuter groups (intake, novelty, rigor, fit) run concurrently, everything
    bounded by one semaphore; judges, synthesis and critic follow in order;
  * completed agents are never re-run unless named in `rerun` (their old outputs
    are archived, never deleted);
  * retries with exponential backoff and jitter for transient failures; plan
    usage limits are waited out (or fail, per config) without burning retries;
  * every agent's prompt passes the isolation guard before it starts, and its
    transcript is audited after it ends.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import random
import re
import statistics
import time
from pathlib import Path

from paper_adversary.budget import BudgetError
from paper_adversary.config import REFUTER_ROLES, STAGE_ORDER, AgentSpec
from paper_adversary.context import BuiltContext, ContextBuilder
from paper_adversary.isolation import IsolationGuard, IsolationViolation, _shingles, new_marker
from paper_adversary.providers import RUN_FATAL, AgentRequest, AgentResult, ErrorKind, ProviderError, make_provider
from paper_adversary.registry import ModelRegistry
from paper_adversary.reports import (
    build_judgment_matrix,
    extract_json_block,
    judge_judgments,
    matrix_markdown,
    read_report,
    references_from,
    refuter_objections,
    write_report,
)
from paper_adversary.store import RunStore
from paper_adversary.usage import summarize
from paper_adversary.util import (
    atomic_write_json,
    atomic_write_text,
    parse_iso,
    prompts_dir,
    read_json,
    read_jsonl,
    runs_root,
    utcnow,
    utcnow_iso,
)

GROUPS = (("intake", "novelty", "rigor", "fit"), ("judge",), ("synthesis",), ("critic",))
RUNNABLE = {"pending", "failed", "interrupted", "retrying", "waiting_plan_limit", "running"}


class PreconditionError(RuntimeError):
    pass


def spec_from_state(entry: dict) -> AgentSpec:
    return AgentSpec(
        agent_id=entry["agent_id"], role=entry["role"], index=entry.get("index", 1),
        model_alias=entry.get("model_alias", entry["model"]), model_id=entry["model"], effort=entry["effort"],
        prompt_name=entry["prompt"], tools=list(entry.get("tools") or []),
        paper_format=entry.get("paper_format", "text"), timeout_s=entry.get("timeout_s", 3600),
        max_turns=entry.get("max_turns"), lens=entry.get("lens_detail"),
    )


def check_preconditions(state: dict, stage: str, allow_incomplete: bool = False) -> list[str]:
    """Return blocking problems for starting `stage` (empty list = OK)."""
    agents = state.get("agents", {})

    def incomplete(roles: tuple[str, ...]) -> list[str]:
        return [f"{aid} ({a['status']})" for aid, a in agents.items()
                if a["role"] in roles and a["status"] != "complete"]

    def any_complete(roles: tuple[str, ...]) -> bool:
        return any(a["role"] in roles and a["status"] == "complete" for a in agents.values())

    needs = {"judge": [REFUTER_ROLES], "synthesis": [REFUTER_ROLES, ("judge",)],
             "critic": [REFUTER_ROLES, ("judge",), ("synthesis",)]}.get(stage, [])
    problems = []
    for roles in needs:
        missing = incomplete(roles)
        if not missing:
            continue
        if allow_incomplete and any_complete(roles):
            continue
        label = "refuters" if roles == REFUTER_ROLES else roles[0]
        problems.append(f"{label} not complete: {', '.join(missing)}")
    return problems


class Pipeline:
    def __init__(self, store: RunStore, provider=None, registry: ModelRegistry | None = None):
        self.store = store
        self.cfg = store.load_config()
        self.registry = registry or ModelRegistry.load()
        self.provider = provider or make_provider(self.cfg.provider)
        self.guard = IsolationGuard(store)
        self.cancel = asyncio.Event()
        self.fatal: ProviderError | None = None
        self.sem = asyncio.Semaphore(self.cfg.concurrency.max_parallel_agents)
        self._launch_lock = asyncio.Lock()
        self._last_launch = 0.0
        self._exclude_shingles: set[str] | None = None

    # ------------------------------------------------------------ state helpers

    def _set(self, aid: str, **fields) -> None:
        def mutate(state: dict) -> None:
            state["agents"][aid].update(fields)
        self.store.update_state(mutate)

    def _add_failure(self, aid: str, attempt: int, err: Exception, kind: str) -> None:
        def mutate(state: dict) -> None:
            state["agents"][aid].setdefault("failures", []).append(
                {"attempt": attempt, "kind": kind, "message": str(err)[:800], "at": utcnow_iso()})
        self.store.update_state(mutate)

    def _stage(self, stage: str, **fields) -> None:
        def mutate(state: dict) -> None:
            state["stages"].setdefault(stage, {}).update(fields)
        self.store.update_state(mutate)

    def _normalize_interrupted(self) -> None:
        """Agents left 'running' by a dead worker become 'interrupted' (their partial work is discarded)."""
        def mutate(state: dict) -> None:
            for a in state["agents"].values():
                if a["status"] in ("running", "retrying", "waiting_plan_limit"):
                    a["status"] = "interrupted"
                    a["detail"] = "worker stopped before this agent finished"
        self.store.update_state(mutate)

    def _mark_rerun(self, agent_ids: list[str]) -> None:
        state = self.store.load_state()
        for aid in agent_ids:
            if aid not in state["agents"]:
                raise PreconditionError(f"unknown agent '{aid}'")
        for aid in agent_ids:
            role = state["agents"][aid]["role"]
            dest = self.store.archive_agent_outputs(aid, role, "rerun requested")
            self.guard.archive(aid)
            ledger = self.store.source_dir / "claims_ledger.md"
            if role == "intake" and ledger.exists() and dest is not None:
                ledger.rename(dest / ledger.name)
            self._set(aid, status="pending", detail="rerun requested", report=None, job_attempts=0,
                      archived_to=self.store.rel(dest) if dest else None)
            self.store.event("agent_rerun_requested", agent_id=aid)

    def specs_for(self, stages: list[str], state: dict | None = None) -> list[AgentSpec]:
        state = state or self.store.load_state()
        return [spec_from_state(a) for a in state["agents"].values() if a["role"] in stages]

    # ------------------------------------------------------------ run

    async def run(self, stages: list[str], rerun: list[str] | None = None, allow_incomplete: bool = False) -> dict:
        stages = [s for s in STAGE_ORDER if s in set(stages)]
        self._normalize_interrupted()
        if rerun:
            self._mark_rerun(rerun)
        self.store.event("job_started", stages=stages, rerun=rerun or [], allow_incomplete=allow_incomplete)
        outcome = "complete"
        message = ""
        try:
            await self.preflight(stages)
            for group in GROUPS:
                wanted = [s for s in group if s in stages]
                if not wanted:
                    continue
                problems = check_preconditions(self.store.load_state(), wanted[0], allow_incomplete)
                if problems:
                    raise PreconditionError(f"cannot start {', '.join(wanted)}: " + "; ".join(problems))
                if wanted == ["judge"]:
                    await self.ensure_refchecks()
                if wanted == ["synthesis"] or wanted == ["critic"]:
                    self.build_matrix()
                await self._run_group(wanted)
                if self.fatal:
                    raise self.fatal
                if self.cancel.is_set():
                    outcome, message = "cancelled", "cancelled by request"
                    break
                if wanted == ["judge"]:
                    self.build_matrix()
                failed = [a["agent_id"] for a in self.store.load_state()["agents"].values()
                          if a["role"] in wanted and a["status"] != "complete"]
                if failed and group is not GROUPS[-1]:
                    later = [s for g in GROUPS[GROUPS.index(group) + 1 :] for s in g if s in stages]
                    if later and check_preconditions(self.store.load_state(), later[0], allow_incomplete):
                        outcome = "incomplete"
                        message = f"stopped before {later[0]}: {', '.join(failed)} did not complete"
                        break
        except ProviderError as exc:
            outcome, message = "failed", str(exc)
        except (PreconditionError, BudgetError, IsolationViolation) as exc:
            outcome, message = "blocked", str(exc)
        finally:
            self.write_usage_summary()
        if outcome == "complete":
            unfinished = [a["agent_id"] for a in self.store.load_state()["agents"].values()
                          if a["role"] in stages and a["status"] != "complete"]
            if unfinished:
                outcome, message = "incomplete", f"not complete: {', '.join(unfinished)}"
        self.store.update_state(lambda st: st.__setitem__("last_job", {
            "stages": stages, "outcome": outcome, "message": message, "finished_at": utcnow_iso()}))
        self.store.event("job_finished", outcome=outcome, message=message)
        return {"outcome": outcome, "message": message}

    async def _run_group(self, stages: list[str]) -> None:
        state = self.store.load_state()
        specs = [s for s in self.specs_for(stages, state) if state["agents"][s.agent_id]["status"] in RUNNABLE]
        for stage in stages:
            if any(s.role == stage for s in specs):
                self._stage(stage, status="running", started_at=utcnow_iso())
        await asyncio.gather(*(self._agent_task(s) for s in specs), return_exceptions=True)
        state = self.store.load_state()
        for stage in stages:
            members = [a for a in state["agents"].values() if a["role"] == stage]
            if not members:
                continue
            done = all(a["status"] == "complete" for a in members)
            self._stage(stage, status="complete" if done else "incomplete", finished_at=utcnow_iso())

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.cancel.wait(), timeout=max(0.0, seconds))
        except asyncio.TimeoutError:
            pass

    async def _stagger(self) -> None:
        async with self._launch_lock:
            gap = self.cfg.concurrency.stagger_seconds - (time.monotonic() - self._last_launch)
            if gap > 0:
                await self._sleep(gap)
            self._last_launch = time.monotonic()

    # ------------------------------------------------------------ preflight

    async def preflight(self, stages: list[str]) -> None:
        mode = self.cfg.provider.preflight
        state = self.store.load_state()
        pending = [s for s in self.specs_for(stages, state) if state["agents"][s.agent_id]["status"] != "complete"]
        if not pending or (mode == "off" and self.provider.name == "mock"):
            return
        auth = await self.provider.auth_status()
        self.store.update_state(lambda st: st.setdefault("preflight", {}).__setitem__("auth", auth))
        method = str(auth.get("authMethod") or "")
        if re.search(r"api|key|console", method, re.I) and not self.cfg.provider.allow_api_key:
            raise ProviderError(ErrorKind.BILLING_GUARD, f"Claude Code is logged in with '{method}', which bills "
                                "API usage. Log in with your Claude plan (`claude setup-token` or `claude auth "
                                "login`) or set provider.allow_api_key in config/default.yaml.")
        if mode != "probe":
            return
        cache_path = runs_root() / ".cache" / "model_probes.json"
        cache = read_json(cache_path, {}) or {}
        max_age = self.cfg.provider.probe_cache_hours * 3600
        groups: dict[tuple[str, tuple[str, ...]], list[AgentSpec]] = {}
        for spec in pending:
            groups.setdefault((spec.model_id, tuple(sorted(self._probe_tools(spec)))), []).append(spec)
        probes: dict[str, dict] = {}
        models: dict[str, dict] = {}
        for (model, tools), members in sorted(groups.items()):
            if self.cancel.is_set():
                return
            key = f"{self.provider.name}:{model}:{'+'.join(tools) or 'no-tools'}"
            hit = cache.get(key)
            checked = parse_iso((hit or {}).get("checked_at"))
            if hit and hit.get("ok") and checked and (utcnow() - checked).total_seconds() < max_age:
                probe = {**hit, "cached": True}
            else:
                probe = await self._probe(model, tools)
                if probe.get("ok"):
                    cache[key] = probe
                    atomic_write_json(cache_path, cache)
            probes[key] = probe
            if probe.get("ok") and (probe.get("context_window") or model not in models):
                models[model] = probe
            if not probe.get("ok"):
                self._probe_failed(model, tools, probe, members)
        self.store.update_state(lambda st: st.setdefault("preflight", {}).update({"models": models, "probes": probes}))
        self.store.event("preflight", auth={k: auth.get(k) for k in ("ok", "authMethod", "subscriptionType")},
                         probes={k: {"ok": r.get("ok"), "error": r.get("error")} for k, r in probes.items()})

    def _probe_tools(self, spec: AgentSpec) -> list[str]:
        """The tool setup a real agent of this spec gets, so the probe exercises the same init."""
        tools = [t for t in spec.tools if not (t == "web_search" and not self.cfg.search.web_search)
                 and not (t == "web_fetch" and not self.cfg.search.web_fetch)]
        if spec.paper_format == "pdf" and (self.store.source_dir / "paper.pdf").is_file():
            tools.append("read_pdf")
        return tools

    async def _probe(self, model: str, tools: tuple[str, ...]) -> dict:
        from paper_adversary.context import tool_server_spec

        log_dir = self.store.dir / "logs" / "preflight" / f"{model}-{'+'.join(tools) or 'no-tools'}"
        server = None
        if "literature" in tools:
            server = tool_server_spec(self.store.dir, self.cfg, "PROBE", log_dir / "tool_calls.jsonl", True, False)
        return await self.provider.probe(model, log_dir, list(tools), server)

    def _probe_failed(self, model: str, tools: tuple[str, ...], probe: dict, members: list[AgentSpec]) -> None:
        kind = probe.get("error_kind")
        setup = f"{model} with tools [{', '.join(tools) or 'none'}]"
        if kind in (ErrorKind.AUTH.value, ErrorKind.BILLING_GUARD.value):
            raise ProviderError(ErrorKind(kind), f"preflight for {setup} failed: {probe.get('error')}. Run "
                                "`claude setup-token` and put CLAUDE_CODE_OAUTH_TOKEN in the server's .env, or "
                                "`claude auth login`.")
        if kind in (ErrorKind.MODEL_UNAVAILABLE.value, ErrorKind.ISOLATION.value, ErrorKind.TOOL_SETUP.value):
            for spec in members:
                self._set(spec.agent_id, status="failed", detail=f"preflight ({setup}): {probe.get('error')}")
            raise ProviderError(ErrorKind(kind), f"preflight for {setup} failed: {probe.get('error')}")
        # Plan limits and transient errors do not block the run: agents on other models start now, and
        # agents on this model wait (plan_limit policy) or retry on their own.
        self.store.event("preflight_warning", setup=setup, error=probe.get("error"), kind=kind)

    # ------------------------------------------------------------ one agent

    async def _agent_task(self, spec: AgentSpec) -> None:
        result: tuple[AgentResult, BuiltContext] | None = None
        try:
            if self.cancel.is_set() or self.fatal:
                return
            result = await self._attempt_loop(spec)
            if result is not None:
                await self._postprocess(spec, *result)
        except asyncio.CancelledError:
            self._set(spec.agent_id, status="interrupted", detail="cancelled")
            raise
        except Exception as exc:  # contain the failure to this agent; the raw output (if any) is in its log dir
            detail = f"internal error: {type(exc).__name__}: {exc}"
            self._set(spec.agent_id, status="failed", detail=detail, finished_at=utcnow_iso())
            self.store.event("agent_crashed", agent_id=spec.agent_id, error=detail)

    async def _attempt_loop(self, spec: AgentSpec) -> tuple[AgentResult, BuiltContext] | None:
        aid = spec.agent_id
        rc = self.cfg.retry
        counted_failures = 0
        plan_wait_started: float | None = None
        budget_scale = 1.0
        while True:
            state = self.store.load_state()
            attempt = int(state["agents"][aid].get("attempts", 0)) + 1
            try:
                builder = ContextBuilder(self.store, self.cfg, self.registry)
                ctx = builder.build(spec, state, window_scale=budget_scale)
                self.guard.check_prompt(spec.role, aid, ctx.system_prompt + "\n" + ctx.user_text, ctx.allowed_text)
            except IsolationViolation as exc:
                self._set(aid, status="failed", detail=str(exc), finished_at=utcnow_iso())
                self._add_failure(aid, attempt, exc, ErrorKind.ISOLATION.value)
                self.store.event("isolation_violation", agent_id=aid, message=str(exc))
                return None
            except BudgetError as exc:
                self._set(aid, status="failed", detail=str(exc), finished_at=utcnow_iso())
                self._add_failure(aid, attempt, exc, ErrorKind.CONTEXT_OVERFLOW.value)
                return None
            log_dir = self.store.agent_log_dir(aid) / f"attempt-{attempt}"
            if ctx.tool_server:
                ctx.tool_server["args"] = [*ctx.tool_server["args"], "--attempt", str(attempt)]
            atomic_write_json(log_dir / "context.json", {"manifest": ctx.manifest, "doc_plan": ctx.doc_plan,
                                                         "tools": ctx.tools, "est_input_tokens": ctx.est_input_tokens,
                                                         "missing_inputs": ctx.missing_inputs})
            req = AgentRequest(
                run_id=self.store.run_id, agent_id=aid, role=spec.role, model=spec.model_id, effort=spec.effort,
                system_prompt=ctx.system_prompt, user_text=ctx.user_text, log_dir=log_dir, tools=ctx.tools,
                pdf_path=ctx.pdf_path, tool_server=ctx.tool_server, timeout_s=spec.timeout_s,
                idle_timeout_s=self.cfg.provider.idle_timeout_minutes * 60, max_turns=spec.max_turns,
                max_output_tokens=self.registry.resolve(spec.model_id).max_output_tokens,
            )
            started = time.monotonic()
            try:
                async with self.sem:  # a slot is held only while the provider call runs
                    await self._stagger()
                    if self.cancel.is_set() or self.fatal:
                        return None
                    self._set(aid, status="running", attempts=attempt, started_at=utcnow_iso(),
                              detail=f"attempt {attempt}", est_input_tokens=ctx.est_input_tokens,
                              doc_mode=ctx.doc_plan.get("mode"))
                    self.store.event("agent_started", agent_id=aid, attempt=attempt, model=spec.model_id,
                                     effort=spec.effort)
                    started = time.monotonic()
                    result = await self.provider.run(req, self.cancel)
                atomic_write_text(log_dir / "raw_output.md", result.text)  # kept even if post-processing fails
                return result, ctx
            except ProviderError as err:
                self._record_usage(spec, attempt, None, err, time.monotonic() - started)
                if self.cancel.is_set() and not self.fatal:
                    self._set(aid, status="interrupted", detail="cancelled")
                    self.store.event("agent_interrupted", agent_id=aid, attempt=attempt)
                    return None
                self._add_failure(aid, attempt, err, err.kind.value)
                self.store.event("agent_attempt_failed", agent_id=aid, attempt=attempt, kind=err.kind.value,
                                 message=err.message[:500])
                if err.kind is ErrorKind.CANCELLED:
                    self._set(aid, status="interrupted", detail="cancelled")
                    return None
                if err.kind in RUN_FATAL:
                    self.fatal = err
                    self.cancel.set()
                    self._set(aid, status="failed", detail=str(err), finished_at=utcnow_iso())
                    return None
                if err.kind is ErrorKind.PLAN_LIMIT:
                    plan_wait_started = plan_wait_started or time.monotonic()
                    waited = await self._wait_plan_limit(aid, err.message, err.reset_at, plan_wait_started)
                    if waited is None and self.cancel.is_set():
                        self._set(aid, status="interrupted", detail="cancelled while waiting for the plan limit")
                        return None
                    if waited is None:
                        self._set(aid, status="failed", detail=f"plan limit: {err.message}", finished_at=utcnow_iso())
                        return None
                    continue
                if err.kind is ErrorKind.CONTEXT_OVERFLOW and budget_scale == 1.0:
                    budget_scale = 0.5  # rebuild once with half the assumed window (sectioned mode)
                    continue
                counted_failures += 1
                if not err.retryable or counted_failures >= rc.max_attempts:
                    self._set(aid, status="failed", detail=str(err), finished_at=utcnow_iso())
                    return None
                delay = min(rc.max_delay_seconds, rc.base_delay_seconds * 2 ** (counted_failures - 1))
                delay *= 1 + random.uniform(-rc.jitter, rc.jitter)
                if err.retry_after_s:
                    delay = max(delay, err.retry_after_s)
                self._set(aid, status="retrying", detail=f"{err.kind.value}; retry {counted_failures + 1} of "
                          f"{rc.max_attempts} at {utcnow_iso()} + {delay:.0f}s")
                await self._sleep(delay)
                if self.cancel.is_set():
                    self._set(aid, status="interrupted", detail="cancelled while waiting to retry")
                    return None

    async def _wait_plan_limit(self, aid: str | None, message: str, reset_at: str | None,
                               started: float | None = None) -> float | None:
        """Sleep until the plan limit resets. Returns seconds waited, or None if the policy says stop."""
        pl = self.cfg.plan_limit
        if pl.policy == "fail":
            return None
        now = utcnow()
        reset = parse_iso(reset_at) if reset_at else None
        wait = (reset - now).total_seconds() + 30 if reset else pl.default_wait_minutes * 60
        wait = max(1.0, wait)
        spent = time.monotonic() - started if started else 0.0
        if spent + wait > pl.max_wait_hours * 3600:
            return None
        until = now + dt.timedelta(seconds=wait)
        if aid:
            self._set(aid, status="waiting_plan_limit", wait_until=until.isoformat(timespec="seconds"),
                      detail=f"plan usage limit reached; resuming at {until.astimezone().strftime('%H:%M')}")
        self.store.event("plan_limit_wait", agent_id=aid, seconds=int(wait), message=message[:300])
        await self._sleep(wait)
        return None if self.cancel.is_set() else wait

    # ------------------------------------------------------------ completion

    def _exclusions(self) -> set[str]:
        if self._exclude_shingles is None:
            text = (self.store.source_dir / "extracted_text.md").read_text(encoding="utf-8")
            exclude = set(_shingles(text))
            for path in prompts_dir().rglob("*"):
                if path.is_file() and path.suffix in (".md", ".yaml"):
                    exclude |= set(_shingles(path.read_text(encoding="utf-8")))
            self._exclude_shingles = exclude
        return self._exclude_shingles

    def _note_plan_usage(self, info: dict | None) -> None:
        if isinstance(info, dict) and info:
            self.store.update_state(lambda st: st.__setitem__("plan_usage", {**info, "observed_at": utcnow_iso()}))

    def _record_usage(self, spec: AgentSpec, attempt: int, result: AgentResult | None, err: ProviderError | None,
                      wall_s: float) -> None:
        self._note_plan_usage((result.runtime if result else {}).get("rate_limit") if result
                              else (err.detail or {}).get("rate_limit") if err else None)
        usage = result.usage if result else (err.usage if err else None)
        self.store.record_usage({
            "agent_id": spec.agent_id, "role": spec.role, "stage": spec.role, "attempt": attempt,
            "outcome": "complete" if result else f"failed:{err.kind.value if err else 'unknown'}",
            "provider": self.provider.name, "model_requested": spec.model_id, "effort": spec.effort,
            "served_models": result.served_models if result else [],
            "usage": usage, "model_usage": result.model_usage if result else None,
            "reported_cost_usd": result.reported_cost_usd if result else None,
            "duration_ms": result.duration_ms if result and result.duration_ms else int(wall_s * 1000),
            "duration_api_ms": result.duration_api_ms if result else None,
            "num_turns": result.num_turns if result else None,
        })

    async def _postprocess(self, spec: AgentSpec, result: AgentResult, ctx: BuiltContext) -> None:
        aid = spec.agent_id
        state = self.store.load_state()
        attempt = int(state["agents"][aid].get("attempts", 1))
        self._record_usage(spec, attempt, result, None, (result.duration_ms or 0) / 1000)
        text = result.text.strip()
        audit = self.guard.audit(spec.role, aid, result.transcript_path or Path("/nonexistent"),
                                 result.sandbox_dir, text)
        self.provider.cleanup(result)
        data, parse_error = (extract_json_block(text) if spec.role not in ("synthesis", "critic") else (None, None))
        objections = refuter_objections(data, aid) if spec.role in REFUTER_ROLES else []
        judgments = judge_judgments(data) if spec.role == "judge" else []
        usage = result.usage or {}
        meta = {
            "run_id": self.store.run_id,
            "agent_id": aid,
            "role": spec.role,
            "model": spec.model_id,
            "model_alias": spec.model_alias,
            "served_by": result.served_models,
            "effort": spec.effort,
            "prompt_version": ctx.prompt.name,
            "prompt_sha256": ctx.prompt.sha256,
            "lens": (spec.lens or {}).get("title"),
            "lens_set": (spec.lens or {}).get("lens_set"),
            "rubric": ctx.rubric_label,
            "timestamp": utcnow_iso(),
            "provider": result.provider or self.provider.name,
            "attempt": attempt,
            "paper_view": ctx.doc_plan.get("mode"),
            "inputs": [f"{m['kind']}:{m.get('agent_id') or m.get('path')}" for m in ctx.manifest],
            "missing_inputs": ctx.missing_inputs,
            "tools": ctx.tools,
            "usage": {k: usage.get(k) for k in ("input_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens", "output_tokens")} if usage else None,
            "reported_cost_usd": result.reported_cost_usd,
            "duration_s": round((result.duration_ms or 0) / 1000, 1),
            "num_turns": result.num_turns,
            "structured_block": "ok" if data else (parse_error or "n/a"),
            "isolation_audit": audit["status"],
            "warnings": result.warnings,
        }
        if spec.role in REFUTER_ROLES:
            meta["objections"] = len(objections)
        if spec.role == "judge":
            meta["judgments"] = len(judgments)
        report_path = self.store.report_path(aid, spec.role)
        if report_path.exists():
            self.store.archive_agent_outputs(aid, spec.role, "superseded by a new completion",
                                             keep_suffixes=(".search_log.jsonl",))
            self.guard.archive(aid)
        marker = new_marker(aid)
        write_report(report_path, meta, text, marker)
        atomic_write_json(self.store.sidecar_path(aid, spec.role, ".context.json"),
                          {"manifest": ctx.manifest, "doc_plan": ctx.doc_plan, "isolation_audit": audit})
        if data is not None:
            atomic_write_json(self.store.sidecar_path(aid, spec.role, ".json"),
                              {"agent_id": aid, "data": data, "objections": objections, "judgments": judgments})
        _, body = read_report(report_path)
        self.guard.register(aid, spec.role, marker, body, self._exclusions())
        self._calibrate(result, ctx)
        if audit["status"] != "pass":
            self.store.event("isolation_audit_failed", agent_id=aid, findings=audit["findings"])

        if spec.role == "intake":
            self._apply_intake(data)
        self._set(aid, status="complete", finished_at=utcnow_iso(), report=self.store.rel(report_path),
                  detail=None, served_by=result.served_models, warnings=result.warnings,
                  structured=bool(data), isolation_audit=audit["status"], duration_s=meta["duration_s"],
                  usage=meta["usage"], reported_cost_usd=result.reported_cost_usd,
                  objections=len(objections) if spec.role in REFUTER_ROLES else None,
                  judgments=len(judgments) if spec.role == "judge" else None)
        self.store.event("agent_complete", agent_id=aid, report=self.store.rel(report_path),
                         audit=audit["status"], warnings=result.warnings)
        if spec.role == "novelty" and self.cfg.search.refcheck:
            summary = await self._refcheck(aid, data)
            if summary is not None:
                self._set(aid, refcheck=summary)

    def _calibrate(self, result: AgentResult, ctx: BuiltContext) -> None:
        usage = result.usage or {}
        if not usage or result.num_turns != 1 or ctx.tools:
            return
        tokens = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                      "cache_read_input_tokens"))
        chars = len(ctx.system_prompt) + len(ctx.user_text)
        if tokens <= 0 or chars <= 0:
            return

        def mutate(state: dict) -> None:
            cal = state.setdefault("calibration", {"tokens_per_char": None, "samples": []})
            cal["samples"] = (cal.get("samples") or [])[-19:] + [round(tokens / chars, 5)]
            cal["tokens_per_char"] = statistics.median(cal["samples"])
        self.store.update_state(mutate)

    async def _refcheck(self, aid: str, data: dict | None) -> dict | None:
        """Verify a novelty refuter's references. None if cancelled (it is redone before the judges run)."""
        refs = references_from(data)
        sidecar = self.store.sidecar_path(aid, "novelty", ".refcheck.json")
        if not refs:
            atomic_write_json(sidecar, {"checked": 0, "skipped": 0, "counts": {}, "items": []})
            return {"checked": 0}
        from paper_adversary.search import LiteratureSearch
        from paper_adversary.search.refcheck import check_references

        providers = [p for p in self.cfg.search.providers if p != "arxiv"] or list(self.cfg.search.providers)
        try:
            async with LiteratureSearch(providers, runs_root() / ".cache", self.cfg.search.cache_ttl_days) as search:
                result = await check_references(refs, search, self.cfg.search.refcheck_max_refs, self.cancel)
        except Exception as exc:  # reference checking is advisory; never fail the agent over it
            self.store.event("refcheck_failed", agent_id=aid, error=f"{type(exc).__name__}: {exc}")
            return {"error": str(exc)}
        if result is None:
            return None
        atomic_write_json(sidecar, result)
        return {"checked": result["checked"], **result["counts"]}

    async def ensure_refchecks(self) -> None:
        """Run any reference check that an earlier job did not finish (judges rely on them)."""
        if not self.cfg.search.refcheck:
            return
        state = self.store.load_state()
        for aid, a in sorted(state["agents"].items()):
            if self.cancel.is_set():
                return
            if a["role"] != "novelty" or a["status"] != "complete":
                continue
            if self.store.sidecar_path(aid, "novelty", ".refcheck.json").is_file():
                continue
            data = (read_json(self.store.sidecar_path(aid, "novelty", ".json"), {}) or {}).get("data")
            summary = await self._refcheck(aid, data)
            if summary is not None:
                self._set(aid, refcheck=summary)

    def _apply_intake(self, data: dict | None) -> None:
        if not data:
            return
        claims = data.get("claims") or []
        lines = ["# Claims ledger (orchestrator intake pass)", "",
                 "A numbered list of the submission's own claims, made before any review. It is an aid for "
                 "checking coverage, not a verdict, and it may be incomplete.", ""]
        for c in claims:
            if isinstance(c, dict):
                loc = f" [{c.get('location')}]" if c.get("location") else ""
                kind = f" ({c.get('type')})" if c.get("type") else ""
                lines.append(f"- **{c.get('id', '?')}**{kind}{loc}: {c.get('claim', '')}")
        for key in ("stated_assumptions", "key_terms"):
            if data.get(key):
                lines += ["", f"## {key.replace('_', ' ').capitalize()}", ""] + [f"- {x}" for x in data[key]]
        atomic_write_text(self.store.source_dir / "claims_ledger.md", "\n".join(lines) + "\n")
        # Kept apart from title/field: agent headers use only user-given or extracted metadata, so the
        # orchestrator's reading of the paper never reaches refuters or judges.
        meta = self.store.load_metadata()
        meta["intake"] = {k: data.get(k) for k in ("title", "field", "submission_type", "venue_guess", "summary")}
        self.store.save_metadata(meta)

    # ------------------------------------------------------------ derived artifacts

    def build_matrix(self) -> dict | None:
        state = self.store.load_state()
        objections: dict[str, list[dict]] = {}
        judgments: dict[str, list[dict]] = {}
        unstructured = []
        for aid, a in state["agents"].items():
            if a["status"] != "complete":
                continue
            side = read_json(self.store.sidecar_path(aid, a["role"], ".json"))
            if a["role"] in REFUTER_ROLES:
                objections[aid] = (side or {}).get("objections") or []
            elif a["role"] == "judge":
                if side is None:
                    unstructured.append(aid)
                judgments[aid] = (side or {}).get("judgments") or []
        if not judgments:
            return None
        matrix = build_judgment_matrix(objections, judgments)
        matrix["judges_without_structured_output"] = unstructured
        atomic_write_json(self.store.role_dir("judge") / "judgment_matrix.json", matrix)
        atomic_write_text(self.store.role_dir("judge") / "judgment_matrix.md", matrix_markdown(matrix))
        return matrix

    def write_usage_summary(self) -> None:
        records = read_jsonl(self.store.usage_path)
        summary = summarize(records, self.registry)
        atomic_write_json(self.store.usage_summary_path, summary)

