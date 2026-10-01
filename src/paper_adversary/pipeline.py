"""Stage orchestration: refuters -> judges -> synthesis -> completeness critic.

Runs inside the worker process. Guarantees:
  * refuter groups (intake, novelty, rigor, fit) run concurrently, everything
    bounded by one semaphore; judges, synthesis and critic follow in order;
  * completed agents are never re-run unless named in `rerun` (their old outputs
    are archived, never deleted);
  * retries with exponential backoff and jitter for transient failures; plan
    usage limits are waited out (or fail, per config) without burning retries;
  * every agent's prompt passes the isolation guard before it starts, and its
    transcript is audited after it ends;
  * an output is saved first and then gated (gates.py): only a passing gate
    makes an agent `complete`, which is the only status later stages read. A
    blocking check quarantines the output instead. A crash or cancel while
    checking leaves the agent `gating`, and a resume finishes the check without
    re-running the agent.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import random
import re
import shutil
import statistics
import time

from paper_adversary import followup, gates
from paper_adversary.budget import BudgetError
from paper_adversary.evidence import check_agent_evidence, quoted_passages
from paper_adversary.evidence_gate import evaluate as evaluate_evidence
from paper_adversary.evidence_gate import gate_markdown, member_status
from paper_adversary.search.fulltext import FullTextStore
from paper_adversary.verification import VerificationRequest, collect, pending_requests, plan_batch
from paper_adversary.config import REFUTER_ROLES, STAGE_ORDER, AgentSpec
from paper_adversary.context import BuiltContext, ContextBuilder
from paper_adversary.isolation import (
    IsolationGuard,
    IsolationViolation,
    _shingles,
    input_changed,
    new_marker,
    position,
)
from paper_adversary.prompts import PromptTemplate, load_prompt
from paper_adversary.providers import RUN_FATAL, AgentRequest, AgentResult, ErrorKind, ProviderError, make_provider
from paper_adversary.registry import ModelRegistry
from paper_adversary.reports import (
    build_judgment_matrix,
    judge_judgments,
    matrix_markdown,
    read_report,
    references_from,
    refuter_objections,
    strip_marker,
    write_report,
)
from paper_adversary.store import RunStore
from paper_adversary.usage import summarize
from paper_adversary.util import (
    append_jsonl,
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

GROUPS = (("intake", "novelty", "rigor", "fit"), ("verifier",), ("judge",), ("synthesis",), ("critic",))
BATCH_DONE = ("complete", "incomplete", "skipped")
RUNNABLE = {"pending", "failed", "interrupted", "retrying", "waiting_plan_limit", "running"}


class PreconditionError(RuntimeError):
    pass


class GateInterrupted(Exception):
    """A gate could not finish now (cancel, plan limit, transient repair failure); the agent stays `gating`."""


def spec_from_state(entry: dict) -> AgentSpec:
    return AgentSpec(
        agent_id=entry["agent_id"], role=entry["role"], index=entry.get("index", 1),
        model_alias=entry.get("model_alias", entry["model"]), model_id=entry["model"], effort=entry["effort"],
        prompt_name=entry["prompt"], tools=list(entry.get("tools") or []),
        paper_format=entry.get("paper_format", "text"), timeout_s=entry.get("timeout_s", 3600),
        max_turns=entry.get("max_turns"), lens=entry.get("lens_detail"), task_id=entry.get("task_id"),
        round=int(entry.get("round") or 0),
    )


def check_preconditions(state: dict, stage: str, allow_incomplete: bool = False) -> list[str]:
    """Return blocking problems for starting `stage` (empty list = OK)."""
    agents = state.get("agents", {})

    base = {aid: a for aid, a in agents.items() if not int(a.get("round") or 0)}  # follow-up rounds aside

    def incomplete(roles: tuple[str, ...]) -> list[str]:
        return [gates.unavailable_label(aid, a) for aid, a in base.items()
                if a["role"] in roles and a["status"] != "complete"]

    def any_complete(roles: tuple[str, ...]) -> bool:
        return any(a["role"] in roles and a["status"] == "complete" for a in base.values())

    needs = {"verifier": [REFUTER_ROLES], "judge": [REFUTER_ROLES], "synthesis": [REFUTER_ROLES, ("judge",)],
             "critic": [REFUTER_ROLES, ("judge",), ("synthesis",)]}.get(stage, [])
    problems = []
    ver = state.get("verification")
    if ver is not None and stage in ("judge", "synthesis", "critic") and not allow_incomplete:
        batches = ver.get("batches") or {}
        unfinished = [k for k, b in batches.items() if k.startswith("refuters") and b.get("status") not in BATCH_DONE]
        if (batches.get("refuters") or {}).get("status") not in BATCH_DONE or unfinished:
            problems.append("blind verification of the novelty objections has not run yet (run_judges runs it first)")
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
        self._auto_reruns = 0  # automatic reruns after failed isolation checks, this job

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
            self._set(aid, status="pending", detail="rerun requested", report=None, gate=None, release=None,
                      structured=None, isolation_audit=None, objections=None, judgments=None, refcheck=None,
                      auto_reruns=0, archived_to=self.store.rel(dest) if dest else None)
            self.store.event("agent_rerun_requested", agent_id=aid)
            entry = state["agents"][aid]
            if role in ("synthesis", "critic") and not int(entry.get("round") or 0) and \
                    ((state.get("followup") or {}).get("rounds")):
                def mutate(st: dict, aid=aid) -> None:  # the follow-up was built on the output being replaced
                    st["followup"].update(stale=f"{aid} was rerun", current_memo="S1")
                self.store.update_state(mutate)

    def specs_for(self, stages: list[str], state: dict | None = None) -> list[AgentSpec]:
        """Base-review agents of these stages (follow-up rounds run their own agents, see run_followup)."""
        state = state or self.store.load_state()
        return [spec_from_state(a) for a in state["agents"].values()
                if a["role"] in stages and not int(a.get("round") or 0)]

    # ------------------------------------------------------------ run

    async def run(self, stages: list[str], rerun: list[str] | None = None, allow_incomplete: bool = False) -> dict:
        extra_batches = [s.split(":", 1)[1] for s in stages if s.startswith("verify:")]
        followup = "followup" in stages
        if "judge" in stages and self._verification_pending():
            stages = [*stages, "verifier"]  # judges always get the independent checks first
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
                if wanted == ["verifier"]:
                    await self.ensure_refchecks()
                    await self.ensure_evidence()
                    if await self.plan_refuter_verification():
                        await self.preflight(["verifier"])
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
                if wanted == ["verifier"]:
                    finished = self.finish_refuter_batches()
                    vc = self.cfg.verifier
                    if vc is not None and vc.blocking and "incomplete" in finished.values():
                        outcome = "incomplete"
                        message = ("blind verification did not finish and verifier.blocking is on: "
                                   + ", ".join(k for k, v in finished.items() if v == "incomplete"))
                        break
                    continue  # otherwise verification never blocks judging; gaps become evidence-gate labels
                if wanted == ["judge"]:
                    self.build_matrix()
                state = self.store.load_state()
                failed = [a["agent_id"] for a in state["agents"].values()
                          if a["role"] in wanted and a["status"] != "complete"]
                if failed and group is not GROUPS[-1]:
                    later = [s for g in GROUPS[GROUPS.index(group) + 1 :] for s in g if s in stages]
                    if later and check_preconditions(state, later[0], allow_incomplete):
                        quarantined = all(state["agents"][a]["status"] == "quarantined" for a in failed)
                        outcome = "quarantined" if quarantined else "incomplete"
                        message = (f"stopped before {later[0]}: "
                                   + ", ".join(gates.unavailable_label(a, state["agents"][a]) for a in failed))
                        break
            for batch_id in extra_batches:
                if outcome != "complete" or self.cancel.is_set():
                    break
                await self.ensure_evidence(refresh=batch_id.startswith("gate"))
                if await self.plan_verification(batch_id, self._batch_requests(batch_id)):
                    await self.preflight(["verifier"])
                    await self._run_group(["verifier"])
                    if self.fatal:
                        raise self.fatal
                self.finish_verification(batch_id)
                self.build_matrix()
            if followup and outcome == "complete" and not self.cancel.is_set():
                result = await self.run_followup()
                if result.get("outcome") in ("incomplete", "blocked"):
                    outcome, message = "incomplete", f"follow-up: {result.get('message') or result['outcome']}"
                elif result.get("outcome") == "cancelled" or self.cancel.is_set():
                    outcome, message = "cancelled", "cancelled by request (during the follow-up)"
                elif result.get("outcome"):
                    message = f"follow-up: {result['outcome']}"
        except ProviderError as exc:
            outcome, message = "failed", str(exc)
        except (PreconditionError, BudgetError, IsolationViolation) as exc:
            outcome, message = "blocked", str(exc)
        finally:
            self.write_usage_summary()
        if outcome == "complete":
            state = self.store.load_state()
            unfinished = [a["agent_id"] for a in state["agents"].values()
                          if a["role"] in stages and a["status"] != "complete" and not int(a.get("round") or 0)]
            if unfinished:
                quarantined = all(state["agents"][a]["status"] == "quarantined" for a in unfinished)
                outcome = "quarantined" if quarantined else "incomplete"
                message = "not complete: " + ", ".join(gates.unavailable_label(a, state["agents"][a])
                                                       for a in unfinished)
        self.store.update_state(lambda st: st.__setitem__("last_job", {
            "stages": stages, "outcome": outcome, "message": message, "finished_at": utcnow_iso()}))
        self.store.event("job_finished", outcome=outcome, message=message)
        return {"outcome": outcome, "message": message}

    async def _run_group(self, stages: list[str]) -> None:
        state = self.store.load_state()
        all_specs = self.specs_for(stages, state)
        specs = [s for s in all_specs if state["agents"][s.agent_id]["status"] in RUNNABLE]
        gating = [s for s in all_specs if state["agents"][s.agent_id]["status"] == "gating"]
        for stage in stages:
            if any(s.role == stage for s in specs + gating):
                self._stage(stage, status="running", started_at=utcnow_iso())
        await asyncio.gather(*(self._agent_task(s) for s in specs),
                             *(self._agent_task(s, gate_only=True) for s in gating), return_exceptions=True)
        state = self.store.load_state()
        for stage in stages:
            members = [a for a in state["agents"].values() if a["role"] == stage]
            if not members:
                continue
            done = all(a["status"] == "complete" for a in members)
            self._stage(stage, status="complete" if done else "incomplete", finished_at=utcnow_iso())

    async def _run_ids(self, ids: list[str]) -> None:
        """Run (or finish gating) specific agents, e.g. one step of a follow-up round."""
        state = self.store.load_state()
        specs = [spec_from_state(state["agents"][a]) for a in ids if a in state["agents"]]
        await asyncio.gather(*(self._agent_task(s) for s in specs if state["agents"][s.agent_id]["status"] in RUNNABLE),
                             *(self._agent_task(s, gate_only=True) for s in specs
                               if state["agents"][s.agent_id]["status"] == "gating"), return_exceptions=True)

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

    async def preflight(self, stages: list[str], specs: list[AgentSpec] | None = None) -> None:
        mode = self.cfg.provider.preflight
        state = self.store.load_state()
        pending = [s for s in (specs if specs is not None else self.specs_for(stages, state))
                   if state["agents"][s.agent_id]["status"] in RUNNABLE]
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
            if probe.get("ok") and probe.get("substituted") and self.cfg.gates.model_substitution == "block":
                probe = {**probe, "ok": False, "error_kind": ErrorKind.MODEL_UNAVAILABLE.value,
                         "error": f"requested {model} but {', '.join(probe['substituted'])} answered"}
                probes[key] = probe
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
        if "literature" in tools or "prior_text" in tools:
            mode = ("open" if "literature" in tools else "scoped") if "prior_text" in tools else None
            server = tool_server_spec(self.store.dir, self.cfg, "PROBE", log_dir / "tool_calls.jsonl",
                                      "literature" in tools, False, mode, [] if mode == "scoped" else None)
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

    async def _agent_task(self, spec: AgentSpec, gate_only: bool = False) -> None:
        try:
            if self.cancel.is_set() or self.fatal:
                return
            again = await self._finish_gate(spec) if gate_only else True
            while again:  # a failed isolation check may ask for one fresh run
                result = await self._attempt_loop(spec)
                if result is None:
                    return
                again = await self._postprocess(spec, *result)
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
                self.guard.check_prompt(spec.role, aid, ctx.system_prompt + "\n" + ctx.user_text, ctx.allowed_text,
                                        position(spec.role, spec.round))
                self.guard.check_secrets(aid, ctx.system_prompt + "\n" + ctx.user_text)
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
                      wall_s: float, stage: str | None = None, model: str | None = None,
                      effort: str | None = None) -> None:
        self._note_plan_usage((result.runtime if result else {}).get("rate_limit") if result
                              else (err.detail or {}).get("rate_limit") if err else None)
        usage = result.usage if result else (err.usage if err else None)
        self.store.record_usage({
            "agent_id": spec.agent_id, "role": spec.role, "round": spec.round, "attempt": attempt,
            "stage": (stage or spec.role) if not spec.round else f"round{spec.round}:{stage or spec.role}",
            "outcome": "complete" if result else f"failed:{err.kind.value if err else 'unknown'}",
            "provider": self.provider.name, "model_requested": model or spec.model_id,
            "effort": effort or spec.effort,
            "served_models": result.served_models if result else [],
            "usage": usage, "model_usage": result.model_usage if result else None,
            "reported_cost_usd": result.reported_cost_usd if result else None,
            "duration_ms": result.duration_ms if result and result.duration_ms else int(wall_s * 1000),
            "duration_api_ms": result.duration_api_ms if result else None,
            "num_turns": result.num_turns if result else None,
        })

    async def _postprocess(self, spec: AgentSpec, result: AgentResult, ctx: BuiltContext) -> bool:
        """Save the output (no model calls), then gate it. Returns True when the agent should run again."""
        aid = spec.agent_id
        state = self.store.load_state()
        attempt = int(state["agents"][aid].get("attempts", 1))
        self._record_usage(spec, attempt, result, None, (result.duration_ms or 0) / 1000)
        text = result.text.strip()
        audit = self.guard.audit(spec.role, aid, result.transcript_path, result.sandbox_dir, text,
                                 pos=position(spec.role, spec.round))
        self.provider.cleanup(result)
        structured = _is_structured(spec, ctx.prompt)
        data, as_written = None, "n/a"
        if structured:
            try:
                data, err = gates.parse_json_strict(text)
                hard = gates.validate_block(spec.role, data)[0] if data is not None else []
                as_written = err or ("invalid: " + "; ".join(hard[:3]) if hard else "ok")
            except Exception as exc:  # never lose a paid-for output over an odd block; the gate recovers it
                data, as_written = None, f"unreadable ({type(exc).__name__})"
            data = data if as_written == "ok" else None
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
            "structured_block": as_written,  # as the agent wrote it; the gate may still recover the data
            "isolation_audit": audit["status"],
            "warnings": result.warnings,
        }
        if spec.role in REFUTER_ROLES and data is not None:
            meta["objections"] = len(data.get("objections") or [])
        if spec.role == "judge" and data is not None:
            meta["judgments"] = len(data.get("judgments") or [])
        report_path = self.store.report_path(aid, spec.role)
        if report_path.exists():
            self.store.archive_agent_outputs(aid, spec.role, "superseded by a new completion",
                                             keep_suffixes=(".search_log.jsonl",))
            self.guard.archive(aid)
        marker = new_marker(aid)
        write_report(report_path, meta, text, marker)
        atomic_write_json(self.store.sidecar_path(aid, spec.role, ".context.json"),
                          {"manifest": ctx.manifest, "doc_plan": ctx.doc_plan, "isolation_audit": audit,
                           "missing_agents": ctx.missing_agents})
        _, body = read_report(report_path)
        exclude = self._exclusions()
        if spec.role in ("novelty", "verifier"):  # quoted third-party text is not the agent's own fingerprint
            try:
                quoted = quoted_passages(gates.parse_json_lenient(text)[0]) + _verifier_quotes(text)
            except Exception:
                quoted = []
            exclude = exclude | set(_shingles("\n".join(quoted)))
        self.guard.register(aid, spec.role, marker, body, exclude, position(spec.role, spec.round))
        self._calibrate(result, ctx)
        atomic_write_json(self.store.gate_path(aid, spec.role), {
            "agent_id": aid, "role": spec.role, "attempt": attempt, "report": self.store.rel(report_path),
            "prompt": {"name": ctx.prompt.name, "sha256": ctx.prompt.sha256}, "verdict": "pending",
            "facts": {"stop_reason": result.stop_reason, "requested_model": spec.model_id,
                      "served_models": result.served_models, "provider_warnings": result.warnings, "audit": audit},
            "checks": [], "history": [{"at": utcnow_iso(), "event": "saved"}]})
        if audit["status"] != "pass":
            self.store.event("isolation_audit_failed", agent_id=aid, status=audit["status"],
                             findings=audit["findings"] + audit["unverifiable"])
        self._set(aid, status="gating", detail="checking the output", report=self.store.rel(report_path),
                  served_by=result.served_models, warnings=result.warnings, isolation_audit=audit["status"],
                  duration_s=meta["duration_s"], usage=meta["usage"], reported_cost_usd=result.reported_cost_usd)
        return await self._finish_gate(spec)

    async def _finish_gate(self, spec: AgentSpec) -> bool:
        """Run the checks on a saved output and decide: complete, quarantine, or (once) run again.

        Everything is read back from disk, so a resume can finish a gate that a crash or cancel interrupted.
        """
        aid, role = spec.agent_id, spec.role
        gate_path = self.store.gate_path(aid, role)
        gate = read_json(gate_path)
        status = self.store.load_state()["agents"][aid]["status"]
        if not gate:
            if status == "gating":  # an interrupted automatic rerun already archived the output: run afresh
                self._set(aid, status="pending", detail="the saved output was archived; running again")
                return True
            return False
        if gate.get("verdict") != "pending":
            if status == "gating":  # decided, but a crash came before the decision was applied
                await self._apply_decision(spec, gate)
            return False
        try:
            _, body = read_report(self.store.report_path(aid, role))
            text = strip_marker(body).strip()
            prompt = load_prompt(gate["prompt"]["name"])
            facts = gate["facts"]
            gcfg = self.cfg.gates
            audit = gates.check_audit(facts["audit"], gcfg.audit.unverifiable)
            if audit.result == gates.BLOCK and self._may_auto_rerun(aid):
                self._auto_rerun(spec, audit.message)
                return True
            checks = [audit, gates.check_substitution(spec.model_id, facts.get("served_models") or [],
                                                      gcfg.model_substitution)]
            st = gates.Structured(None, "n/a", "n/a")
            if _is_structured(spec, prompt):
                repair_first = gcfg.structured.model_repair and role in gcfg.structured.repair_first
                st = gates.parse_structured(role, text, repair_first)
                if st.repair_mode:
                    st = await self._repair(spec, text, st, prompt)
                checks.append(gates.check_structured(st))
            checks.append(gates.check_truncation(facts.get("stop_reason"), st.source == "ok", gcfg.truncation))
            if role == "judge" and st.usable:
                checks.append(gates.check_coverage(judge_judgments(st.data), self._expected_objections(aid),
                                                   gcfg.judge_coverage.min_fraction))
            mode = {"synthesis": gcfg.sections.synthesis, "critic": gcfg.sections.critic,
                    "recheck": gcfg.sections.critic, "revision": "block_any"}.get(role, "warn")
            required = gates.required_sections(prompt.body)
            checks.append(gates.check_sections(text, required, mode))
            if role in ("synthesis", "revision"):
                flagged = (read_json(self.store.role_dir("judge") / "evidence_gate.json", {}) or {}).get("flagged_ids")
                checks.append(gates.check_unverified_placement(text, required, flagged or [], block=False))
            if role == "adjudicator" and st.usable:
                record = read_json(followup.round_dir(self.store, spec.round) / "items.json", {}) or {}
                expected = {it["id"] for it in record.get("items") or [] if it["route"] == "adjudicate"}
                rulings = [{"objection_ids": [str(i).upper() for i in r.get("item_ids") or []]}
                           for r in st.data.get("rulings") or [] if isinstance(r, dict)]
                checks.append(gates.check_coverage(rulings, expected, gcfg.judge_coverage.min_fraction))
            if role == "revision" and st.usable:
                checks += followup.check_dispositions(self.store, self.store.load_state(), spec.round, st.data)
        except GateInterrupted:
            return False
        except Exception as exc:  # the output is safe on disk; a resume retries the checks, but not forever
            gate = read_json(gate_path) or gate
            errors = gate.setdefault("errors", [])
            errors.append({"at": utcnow_iso(), "error": f"{type(exc).__name__}: {exc}"[:500]})
            self.store.event("gate_error", agent_id=aid, error=errors[-1]["error"])
            if len(errors) >= 2:
                gate.update(verdict="quarantine", decision={
                    "verdict": "quarantine", "at": utcnow_iso(), "classes": ["quality"], "warnings": [],
                    "reasons": [f"the checks themselves failed twice ({errors[-1]['error'][:200]})"],
                    "structured": None, "sidecar": self.store.rel(gate_path)})
                atomic_write_json(gate_path, gate)
                await self._apply_decision(spec, gate)
                return False
            atomic_write_json(gate_path, gate)
            self._set(aid, detail=f"checking the output failed ({type(exc).__name__}); resume to retry")
            return False
        result = gates.verdict(checks)
        objections = refuter_objections(st.data, aid) if role in REFUTER_ROLES and st.usable else []
        judgments = judge_judgments(st.data) if role == "judge" and st.usable else []
        if st.usable:
            atomic_write_json(self.store.sidecar_path(aid, role, ".json"), {
                "agent_id": aid, "data": st.data, "objections": objections, "judgments": judgments,
                "structured": {"source": st.source, "as_written": st.as_written, "warnings": st.warnings}})
        gate = read_json(gate_path) or gate
        summary = {"verdict": "quarantine" if result == gates.BLOCK else result, "at": utcnow_iso(),
                   "reasons": [c.message for c in checks if c.result == gates.BLOCK],
                   "classes": sorted({c.category for c in checks if c.result == gates.BLOCK}),
                   "warnings": [c.message for c in checks if c.result == gates.WARN and c.message],
                   "structured": st.source, "sidecar": self.store.rel(gate_path)}
        gate.update(verdict=summary["verdict"], checks=[c.to_dict() for c in checks], decision=summary,
                    structured={"source": st.source, "as_written": st.as_written, "warnings": st.warnings})
        gate.setdefault("history", []).append({"at": utcnow_iso(), "event": "gated", "verdict": gate["verdict"]})
        atomic_write_json(gate_path, gate)  # the decision is on disk before it is applied (see the top)
        await self._apply_decision(spec, gate)
        return False

    async def _apply_decision(self, spec: AgentSpec, gate: dict) -> None:
        """Quarantine or complete an agent according to its recorded gate decision (idempotent)."""
        summary = gate.get("decision") or {"verdict": gate.get("verdict"), "at": utcnow_iso(), "reasons": [],
                                           "classes": [], "warnings": [], "structured": None}
        if gate.get("verdict") == "quarantine":
            self._quarantine(spec, summary)
            return
        side = read_json(self.store.sidecar_path(spec.agent_id, spec.role, ".json")) or {}
        info = side.get("structured") or {}
        st = gates.Structured(side.get("data"), info.get("source") or "n/a", info.get("as_written") or "n/a",
                              warnings=info.get("warnings") or [])
        await self._complete(spec, st, summary, side.get("objections") or [], side.get("judgments") or [])

    async def _complete(self, spec: AgentSpec, st: gates.Structured, summary: dict, objections: list[dict],
                        judgments: list[dict]) -> None:
        aid = spec.agent_id
        if spec.role == "intake":
            apply_intake(self.store, st.data)
        if spec.role == "revision":
            self._supersede_memo(aid, spec.round)
        self._set(aid, status="complete", finished_at=utcnow_iso(), detail=None, gate=summary,
                  structured=st.source if st.source != "n/a" else None,
                  objections=len(objections) if spec.role in REFUTER_ROLES else None,
                  judgments=len(judgments) if spec.role == "judge" else None)
        state = self.store.load_state()
        self.store.event("agent_complete", agent_id=aid, report=state["agents"][aid].get("report"),
                         audit=state["agents"][aid].get("isolation_audit"), gate=summary["verdict"],
                         warnings=summary["warnings"])
        if spec.role == "novelty" and self.cfg.search.refcheck:
            note = ("references were not machine-checked: the structured block was unusable and was rebuilt from "
                    "headings" if st.source == "derived:headings" else None)
            result = await self._refcheck(aid, st.data, note)
            if result is not None:
                self._set(aid, refcheck=result)
        if spec.role == "novelty":
            await self._evidence(aid, st.data)

    def _supersede_memo(self, aid: str, rnd: int) -> None:
        """The revised memo becomes the current one; the earlier memo stays where it is, recorded as superseded."""
        supersede_memo(self.store, aid, rnd)

    def _quarantine(self, spec: AgentSpec, summary: dict) -> None:
        aid = spec.agent_id
        self.guard.set_quarantined(aid, True)
        self._set(aid, status="quarantined", finished_at=utcnow_iso(), gate=summary,
                  detail="quarantined: " + "; ".join(summary["reasons"])[:400])
        self.store.event("agent_quarantined", agent_id=aid, reasons=summary["reasons"], classes=summary["classes"])

    def _may_auto_rerun(self, aid: str) -> bool:
        audit_cfg = self.cfg.gates.audit
        done = int(self.store.load_state()["agents"][aid].get("auto_reruns") or 0)
        return done < audit_cfg.auto_rerun and self._auto_reruns < audit_cfg.max_auto_reruns_per_job

    def _auto_rerun(self, spec: AgentSpec, reason: str) -> None:
        """Archive the suspect output and start the agent afresh (once per agent, a few per job)."""
        aid = spec.agent_id
        self._auto_reruns += 1
        done = int(self.store.load_state()["agents"][aid].get("auto_reruns") or 0)
        # status first: a crash before the archive below only means the old output is archived on completion
        self._set(aid, status="pending", detail=f"running again after: {reason[:200]}", report=None, gate=None,
                  structured=None, isolation_audit=None, auto_reruns=done + 1)
        dest = self.store.archive_agent_outputs(aid, spec.role, f"automatic rerun after: {reason[:300]}")
        self.guard.archive(aid)
        self._set(aid, archived_to=self.store.rel(dest) if dest else None)
        self.store.event("agent_auto_rerun", agent_id=aid, reason=reason)

    def _expected_objections(self, judge_id: str) -> set[str]:
        """Objection IDs of the refuter reports this judge was actually given (from its context manifest)."""
        ctx = read_json(self.store.sidecar_path(judge_id, "judge", ".context.json"), {}) or {}
        ids: set[str] = set()
        for m in ctx.get("manifest") or []:
            if m.get("kind") in REFUTER_ROLES and m.get("agent_id"):
                side = read_json(self.store.sidecar_path(m["agent_id"], m["kind"], ".json"), {}) or {}
                ids |= {o["id"] for o in side.get("objections") or [] if o.get("id")}
        return ids

    async def _repair(self, spec: AgentSpec, text: str, st: gates.Structured, prompt: PromptTemplate) -> gates.Structured:
        """Ask a model to transcribe or fix the structured block; accept only what validate_repair allows."""
        aid, role = spec.agent_id, spec.role
        rcfg = self.cfg.gates.repair
        gate_path = self.store.gate_path(aid, role)
        gate = read_json(gate_path) or {}
        rep = gate.setdefault("repair", {"attempts": 0, "outcome": None})
        fallback_note = ("references were not machine-checked: the structured block could not be repaired"
                         if role == "novelty" else None)
        if rep.get("outcome") in ("rejected", "failed"):
            return gates.fallback(role, text, st.as_written, fallback_note)
        alias = spec.model_alias if rcfg.model == "agent" else rcfg.model
        info = self.registry.resolve(alias)
        repair_prompt = load_prompt(rcfg.prompt)
        system = repair_prompt.render({"agent_id": aid, "role": role})
        user = gates.repair_user_text(role, text, gates.json_shape(prompt.body), st.repair_mode or "transcribe",
                                      st.as_written, st.broken_block)
        self.guard.check_prompt(role, aid, system + "\n" + user, allowed_text=text, pos=position(role, spec.round))
        self.guard.check_secrets(aid, system + "\n" + user)
        rep.update(model_requested=info.id, effort=rcfg.effort, mode=st.repair_mode, prompt_version=repair_prompt.name,
                   prompt_sha256=repair_prompt.sha256)
        rc = self.cfg.retry
        transient = 0
        attempt = int(gate.get("attempt") or 1)
        while True:
            if transient >= rcfg.max_attempts:  # an outage, not a verdict: stay `gating`, retry on resume
                rep["transient_exhausted"] = int(rep.get("transient_exhausted") or 0) + 1
                atomic_write_json(gate_path, gate)
                self._set(aid, detail="the format-repair call keeps failing for transient reasons; resume to retry")
                raise GateInterrupted()
            started = time.monotonic()
            try:
                async with self.sem:
                    await self._stagger()
                    if self.cancel.is_set() or self.fatal:
                        raise GateInterrupted()
                    rep["attempts"] = int(rep.get("attempts") or 0) + 1  # counted only when a call is made
                    atomic_write_json(gate_path, gate)
                    log_dir = self.store.agent_log_dir(aid) / f"attempt-{attempt}" / f"repair-{rep['attempts']}"
                    req = AgentRequest(run_id=self.store.run_id, agent_id=f"{aid}-repair", role="repair",
                                       model=info.id, effort=rcfg.effort, system_prompt=system, user_text=user,
                                       log_dir=log_dir, timeout_s=rcfg.timeout_minutes * 60,
                                       idle_timeout_s=self.cfg.provider.idle_timeout_minutes * 60,
                                       max_output_tokens=info.max_output_tokens)
                    self._set(aid, detail=f"repairing the structured block (call {rep['attempts']})")
                    result = await self.provider.run(req, self.cancel)
            except ProviderError as err:
                self._record_usage(spec, attempt, None, err, time.monotonic() - started, stage="repair",
                                   model=info.id, effort=rcfg.effort)
                if err.kind is ErrorKind.CANCELLED or self.cancel.is_set():
                    raise GateInterrupted() from err
                if err.kind in RUN_FATAL:
                    self.fatal = err
                    self.cancel.set()
                    raise GateInterrupted() from err
                if err.kind is ErrorKind.PLAN_LIMIT:
                    if await self._wait_plan_limit(None, err.message, err.reset_at) is None:
                        raise GateInterrupted() from err
                    continue
                rep.setdefault("errors", []).append(f"{err.kind.value}: {err.message[:200]}")
                atomic_write_json(gate_path, gate)
                if not err.retryable:  # a permanent refusal: fall back now
                    rep.update(outcome="failed")
                    atomic_write_json(gate_path, gate)
                    self.store.event("repair_failed", agent_id=aid, errors=rep.get("errors", []))
                    return gates.fallback(role, text, st.as_written, fallback_note)
                transient += 1
                await self._sleep(min(rc.max_delay_seconds, rc.base_delay_seconds * 2 ** (transient - 1)))
                if self.cancel.is_set():
                    raise GateInterrupted() from err
                continue
            self._record_usage(spec, attempt, result, None, (result.duration_ms or 0) / 1000, stage="repair",
                               model=info.id, effort=rcfg.effort)
            audit = self.guard.audit(role, aid, result.transcript_path, result.sandbox_dir, result.text,
                                     pos=position(role, spec.round))
            self.provider.cleanup(result)
            data, rejections, warnings = gates.validate_repair(role, st.repair_mode or "transcribe", result.text,
                                                               text, st.broken_block)
            if audit["status"] != "pass":
                rejections.append(f"the repair call's isolation audit was {audit['status']}")
            other = gates.check_substitution(info.id, result.served_models, "block")
            if other.result != gates.PASS:
                rejections.append(other.message)
            rep.update(served_by=result.served_models, log_dir=self.store.rel(log_dir))
            if rejections or data is None:
                rep.update(outcome="rejected", rejections=rejections)
                atomic_write_json(gate_path, gate)
                self.store.event("repair_rejected", agent_id=aid, rejections=rejections)
                return gates.fallback(role, text, st.as_written, fallback_note)
            rep.update(outcome="accepted")
            atomic_write_json(gate_path, gate)
            self.store.event("repair_accepted", agent_id=aid, mode=st.repair_mode)
            return gates.accept_repair(role, text, st.as_written, data, st.repair_mode or "transcribe", warnings)

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

    async def _refcheck(self, aid: str, data: dict | None, note: str | None = None) -> dict | None:
        """Verify a novelty refuter's references. None if cancelled (it is redone before the judges run)."""
        refs = references_from(data)
        sidecar = self.store.sidecar_path(aid, "novelty", ".refcheck.json")
        if not refs:
            atomic_write_json(sidecar, {"checked": 0, "skipped": 0, "counts": {}, "items": [], "note": note})
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
            side = read_json(self.store.sidecar_path(aid, "novelty", ".json"), {}) or {}
            note = ("references were not machine-checked: the structured block was rebuilt from headings"
                    if (side.get("structured") or {}).get("source") == "derived:headings" or not side else None)
            summary = await self._refcheck(aid, side.get("data"), note)
            if summary is not None:
                self._set(aid, refcheck=summary)

    # ------------------------------------------------------------ evidence and verification

    def _fulltext(self) -> FullTextStore:
        return FullTextStore(runs_root() / ".cache", self.cfg.search.fulltext, list(self.cfg.search.providers),
                             prior_dir=self.store.prior_dir)

    async def _evidence(self, aid: str, data: dict | None) -> None:
        """Check a novelty refuter's quoted passages against the full texts (advisory: never fails the agent)."""
        sidecar = self.store.sidecar_path(aid, "novelty", ".evidence.json")
        try:
            async with self._fulltext() as fetch:
                record = await check_agent_evidence(self.store, self.cfg, aid, data, fetch)
        except Exception as exc:
            self.store.event("evidence_check_failed", agent_id=aid, error=f"{type(exc).__name__}: {exc}")
            return
        atomic_write_json(sidecar, record)
        counts: dict[str, int] = {}
        for entry in record["objections"].values():
            if entry.get("decisive"):
                counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        self._set(aid, evidence=counts)

    async def ensure_evidence(self, refresh: bool = False) -> None:
        """Check every complete novelty refuter's quotes; with refresh, recheck objections that were unverifiable
        (a full text may have been supplied since)."""
        state = self.store.load_state()
        for aid, a in sorted(state["agents"].items()):
            if self.cancel.is_set():
                return
            if a["role"] != "novelty" or a["status"] != "complete":
                continue
            path = self.store.sidecar_path(aid, "novelty", ".evidence.json")
            if path.is_file():
                record = read_json(path, {}) or {}
                stale = any(e.get("decisive") and e.get("status") in ("fulltext_unavailable", "prior_not_found",
                                                                      "no_evidence", "quotes_not_found")
                            for e in (record.get("objections") or {}).values())
                if not (refresh and stale):
                    continue
            data = (read_json(self.store.sidecar_path(aid, "novelty", ".json"), {}) or {}).get("data")
            await self._evidence(aid, data)

    def _verification_pending(self) -> bool:
        """Whether judges must wait for blind verification: never planned, a batch unfinished, or objections
        (new, or changed by a rerun) that no batch has checked yet."""
        state = self.store.load_state()
        ver = state.get("verification")
        if ver is None or self.cfg.verifier is None or not self.cfg.verifier.enabled:
            return False
        batches = ver.get("batches") or {}
        if "refuters" not in batches:
            return True
        if any(k.startswith("refuters") and b.get("status") not in BATCH_DONE for k, b in batches.items()):
            return True
        return bool(pending_requests(self.store, state))

    async def plan_refuter_verification(self) -> list[str]:
        """Plan (or resume) blind verification of the refuters' decisive objections. A later batch
        ("refuters2", ...) covers objections that are new or changed since the first one."""
        vc = self.cfg.verifier
        state = self.store.load_state()
        ver = state.get("verification")
        if ver is None or vc is None or not vc.enabled:
            return []
        batches = ver.get("batches") or {}
        unfinished = [a for k, b in batches.items() if k.startswith("refuters") and b.get("status") == "planned"
                      for a in b.get("agents") or []]
        if unfinished or any(k.startswith("refuters") and b.get("status") == "planned" for k, b in batches.items()):
            return [a for a in unfinished if state["agents"].get(a, {}).get("status") != "complete"]
        pending = pending_requests(self.store, state)
        n = sum(1 for k in batches if k.startswith("refuters"))
        if not pending:
            if n == 0:
                self._set_batch("refuters", {"id": "refuters", "status": "skipped", "requests": [], "tasks": [],
                                             "agents": [], "reason": "no prior-work objections to verify"})
            return []
        return await self.plan_verification("refuters" if n == 0 else f"refuters{n + 1}", pending)

    def finish_refuter_batches(self) -> dict[str, str]:
        state = self.store.load_state()
        batches = (state.get("verification") or {}).get("batches") or {}
        for k, b in batches.items():
            if k.startswith("refuters") and b.get("status") == "planned":
                self.finish_verification(k)
        batches = (self.store.load_state().get("verification") or {}).get("batches") or {}
        return {k: b.get("status") for k, b in batches.items() if k.startswith("refuters")}

    def _batch_requests(self, batch_id: str) -> list[VerificationRequest]:
        raw = read_json(self.store.role_dir("verifier") / "batches" / f"{batch_id}.requests.json", []) or []
        return [VerificationRequest(**{**r, "origin_ids": tuple(r.get("origin_ids") or [])}) for r in raw]

    async def plan_verification(self, batch_id: str, requests: list[VerificationRequest] | None = None,
                                rnd: int = 0) -> list[str]:
        """Create the verifier agents for a batch (once; a resumed batch keeps its agents). Returns their IDs."""
        vc = self.cfg.verifier
        state = self.store.load_state()
        ver = state.get("verification")
        if ver is None or vc is None or not vc.enabled:
            return []
        batch = (ver.get("batches") or {}).get(batch_id)
        if batch:
            return [a for a in batch.get("agents") or [] if state["agents"].get(a, {}).get("status") != "complete"]
        reqs = requests if requests is not None else pending_requests(self.store, state)
        if not reqs:
            self._set_batch(batch_id, {"id": batch_id, "status": "skipped", "requests": [], "tasks": [],
                                       "agents": [], "reason": "no prior-work objections to verify"})
            return []
        try:
            async with self._fulltext() as fetch:
                record, specs = await plan_batch(self.store, self.cfg, batch_id, reqs, fetch, state, rnd)
        except Exception as exc:  # a broken document or URL must not take the worker down
            self.store.event("verification_planning_failed", batch=batch_id, error=f"{type(exc).__name__}: {exc}")
            self._set_batch(batch_id, {"id": batch_id, "status": "incomplete", "requests": [], "tasks": [],
                                       "agents": [], "reason": f"planning failed: {type(exc).__name__}: {exc}"[:300]})
            return []
        for spec in specs:
            spec.model_id = self.registry.resolve(spec.model_alias).id
        record["agents"] = [s.agent_id for s in specs]
        next_index = record.pop("next_index")

        def mutate(st: dict) -> None:
            v = st.setdefault("verification", {"next_index": 1, "batches": {}})
            v["next_index"] = next_index
            v.setdefault("batches", {})[batch_id] = record
            for s in specs:
                st["agents"][s.agent_id] = {**s.state_entry(), "status": "pending", "attempts": 0, "failures": [],
                                            "batch": batch_id}
        self.store.update_state(mutate)
        self.store.event("verification_planned", batch=batch_id, requests=len(reqs), agents=record["agents"])
        return record["agents"]  # every caller finishes the batch (collects results) once its agents ran

    def _set_batch(self, batch_id: str, record: dict) -> None:
        def mutate(st: dict) -> None:
            st.setdefault("verification", {"next_index": 1, "batches": {}}).setdefault("batches", {})[batch_id] = record
        self.store.update_state(mutate)

    def finish_verification(self, batch_id: str) -> None:
        finish_batch(self.store, self.cfg, batch_id)

    # ------------------------------------------------------------ follow-up rounds

    def _round(self, r: int, **fields) -> None:
        def mutate(st: dict) -> None:
            fu = st.setdefault("followup", {"current_memo": "S1", "next_adjudicator": 1, "rounds": {}})
            fu.setdefault("rounds", {}).setdefault(str(r), {}).update(fields)
        self.store.update_state(mutate)

    async def run_followup(self) -> dict:
        """Run follow-up rounds: the open one if a job was interrupted, else the next one; continue while the
        re-check finds new items and auto_continue allows, up to max_rounds."""
        fc = self.cfg.followup
        if fc is None or fc.max_rounds < 1:
            return {"outcome": "disabled"}
        stale = (self.store.load_state().get("followup") or {}).get("stale")
        if stale:
            self._restart_followup(stale)
        while not self.cancel.is_set():
            state = self.store.load_state()
            rounds = (state.get("followup") or {}).get("rounds") or {}
            open_round = next((int(k) for k, v in rounds.items()
                               if v.get("status") != "complete" or _round_has_work(state, v)), None)
            if open_round is None:
                last = rounds.get(str(len(rounds))) if rounds else None
                if last and last.get("outcome") != "another_round":
                    return {"outcome": last.get("outcome"), "round": len(rounds)}
                if len(rounds) >= fc.max_rounds:
                    return {"outcome": "max_rounds_reached", "round": len(rounds)}
            r = open_round or len(rounds) + 1
            result = await self._followup_round(r, state, fresh=open_round is None)
            if result["outcome"] != "another_round" or not fc.auto_continue:
                return result
        return {"outcome": "cancelled"}

    def _restart_followup(self, reason: str) -> None:
        """Start the follow-up over (its base memo or critic changed): archive the rounds' records and outputs,
        mark their agents superseded, and keep the old state under followup_history. Nothing is deleted."""
        state = self.store.load_state()
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        rounds = sorted((self.store.dir / "followup").glob("round-*"))
        if rounds:
            dest = self.store.dir / "archive" / stamp / "followup"
            dest.mkdir(parents=True, exist_ok=True)
            for d in rounds:
                shutil.move(str(d), dest / d.name)
        agents = [aid for aid, a in state["agents"].items()
                  if int(a.get("round") or 0) >= 1 and a["status"] != "superseded"]
        for aid in agents:
            self.store.archive_agent_outputs(aid, state["agents"][aid]["role"], f"follow-up restarted: {reason}")
            self.guard.archive(aid)
        results_path = self.store.role_dir("verifier") / "results.json"
        results = read_json(results_path)
        if results:
            moved = {rid: r for rid, r in (results.get("requests") or {}).items()
                     if str(r.get("batch") or "").startswith("critic")}
            for rid in moved:
                results["requests"].pop(rid)
            results.setdefault("superseded", {}).update({f"{rid}@{stamp}": r for rid, r in moved.items()})
            atomic_write_json(results_path, results)

        def mutate(st: dict) -> None:
            old = st.get("followup") or {}
            st.setdefault("followup_history", []).append({**old, "superseded_at": utcnow_iso(), "reason": reason})
            st["followup"] = {"current_memo": "S1", "rounds": {}}
            for a in st["agents"].values():
                if a["role"] == "synthesis":
                    a.pop("superseded_by", None)
            for aid in agents:
                st["agents"][aid].update(status="superseded", detail=f"superseded: follow-up restarted ({reason})")
            batches = (st.get("verification") or {}).get("batches") or {}
            for k in [k for k in batches if k.startswith("critic") and "@" not in k]:
                batches[f"{k}@{stamp}"] = batches.pop(k)
        self.store.update_state(mutate)
        self.store.event("followup_restarted", reason=reason, superseded=agents)

    async def _followup_round(self, r: int, state: dict, fresh: bool) -> dict:
        critic_id, critic_role = followup.round_critic(state, r)
        critic = state["agents"].get(critic_id)
        if not critic or critic["status"] != "complete":
            label = gates.unavailable_label(critic_id, critic) if critic else f"{critic_id} (not run)"
            return {"outcome": "blocked", "message": f"follow-up round {r} needs {label}"}
        memo = followup.current_memo(state)
        if (state["agents"].get(memo) or {}).get("status") != "complete":
            return {"outcome": "blocked", "message": f"follow-up round {r} needs the current memo {memo}"}
        if not read_json(self.store.sidecar_path(critic_id, critic_role, ".json")):
            return {"outcome": "blocked", "message": f"{critic_id} wrote no structured items (use critic_v2)"}
        if fresh:  # items written about an earlier version of the memo would be revised into the wrong text
            ctx = read_json(self.store.sidecar_path(critic_id, critic_role, ".context.json"), {}) or {}
            if any(m.get("agent_id") == memo and input_changed(self.store, m) for m in ctx.get("manifest") or []):
                return {"outcome": "blocked", "message": f"{critic_id} reviewed an earlier version of memo {memo}; "
                        f"rerun {critic_id} first (rerun_agents=['{critic_id}'])"}
        record = followup.triage(self.store, self.cfg, state, r)
        if fresh:
            if not followup.routable(record):
                self._round(r, status="complete", outcome="nothing_to_follow_up", critic=critic_id)
                self.store.event("followup_round_skipped", round=r, reason="no item to adjudicate or verify")
                return {"outcome": "nothing_to_follow_up", "round": r}
            specs = followup.plan_agents(self.cfg, self.registry, state, r)
            revision = next(s.agent_id for s in specs if s.role == "revision")
            recheck = next(s.agent_id for s in specs if s.role == "recheck")

            def mutate(st: dict) -> None:
                fu = st.setdefault("followup", {"current_memo": "S1", "rounds": {}})
                fu.setdefault("current_memo", "S1")
                fu.setdefault("rounds", {})[str(r)] = {
                    "status": "planned", "critic": critic_id, "memo_before": memo, "planned_at": utcnow_iso(),
                    "adjudicators": [s.agent_id for s in specs if s.role == "adjudicator"],
                    "revision": revision, "recheck": recheck}
                for s in specs:
                    st["agents"][s.agent_id] = {**s.state_entry(), "status": "pending", "attempts": 0, "failures": []}
            self.store.update_state(mutate)
            self.store.event("followup_round_planned", round=r, critic=critic_id, items=len(record["items"]),
                             agents=[s.agent_id for s in specs])
        rnd = self.store.load_state()["followup"]["rounds"][str(r)]
        state = self.store.load_state()
        own = [spec_from_state(state["agents"][a]) for a in rnd["adjudicators"] + [rnd["revision"], rnd["recheck"]]]
        await self.preflight([], own)

        requests = followup.verification_requests(record)
        if requests and self.cfg.verifier is not None and self.cfg.verifier.enabled:
            self._round(r, status="verifying")
            batch = f"critic{r}"
            ids = await self.plan_verification(batch, requests, rnd=r)
            if ids:
                state = self.store.load_state()
                await self.preflight([], [spec_from_state(state["agents"][a]) for a in ids])
                await self._run_ids(ids)
            self.finish_verification(batch)
        steps = (("adjudicating", rnd["adjudicators"]), ("revising", [rnd["revision"]]),
                 ("rechecking", [rnd["recheck"]]))
        for status, ids in steps:
            if self.fatal:
                raise self.fatal
            if self.cancel.is_set():
                return {"outcome": "cancelled", "round": r}
            self._round(r, status=status)
            await self._run_ids(ids)
            if self.fatal:
                raise self.fatal
            state = self.store.load_state()
            done = [a for a in ids if state["agents"][a]["status"] == "complete"]
            if not done:
                why = ", ".join(gates.unavailable_label(a, state["agents"][a]) for a in ids)
                self._round(r, status="incomplete", message=why)
                return {"outcome": "incomplete", "round": r, "message": f"round {r}: {why}"}
            if status == "adjudicating":
                followup.build_followup_matrix(self.store, self.cfg, state, r)
        result = followup.stop_rule(self.store, self.cfg, self.store.load_state(), r)
        self._round(r, status="complete", outcome=result["outcome"], new_items=[i["id"] for i in result["new_items"]],
                    finished_at=utcnow_iso())
        self.store.event("followup_round_finished", round=r, outcome=result["outcome"],
                         new_items=[i["id"] for i in result["new_items"]])
        return result

    # ------------------------------------------------------------ derived artifacts

    def build_matrix(self) -> dict | None:
        return build_matrix(self.store)

    def write_usage_summary(self) -> None:
        records = read_jsonl(self.store.usage_path)
        summary = summarize(records, self.registry)
        atomic_write_json(self.store.usage_summary_path, summary)


def _is_structured(spec: AgentSpec, prompt: PromptTemplate) -> bool:
    """Whether the agent's prompt asks for a fenced JSON block (front matter `structured_output`, else by role)."""
    return bool(prompt.meta.get("structured_output", spec.role not in ("synthesis", "critic")))


def _round_has_work(state: dict, rnd: dict) -> bool:
    """A completed round reopens when one of its agents was marked for a rerun or was interrupted."""
    ids = (rnd.get("adjudicators") or []) + [rnd.get("revision"), rnd.get("recheck")]
    return any(state["agents"].get(a, {}).get("status") in ("pending", "interrupted", "gating", "retrying",
                                                             "waiting_plan_limit", "running") for a in ids if a)


def supersede_memo(store: RunStore, aid: str, rnd: int) -> None:
    """A revised memo becomes the current one; the earlier memo stays in place, recorded as superseded."""
    state = store.load_state()
    previous = followup.current_memo(state)
    if previous == aid:
        return  # already applied (e.g. a resumed gate)

    def mutate(st: dict) -> None:
        fu = st.setdefault("followup", {"rounds": {}})
        fu["current_memo"] = aid
        if previous in st["agents"]:
            st["agents"][previous]["superseded_by"] = aid
    store.update_state(mutate)
    prev_role = state["agents"].get(previous, {}).get("role", "synthesis")
    append_jsonl(store.dir / "archive" / "index.jsonl",
                 {"at": utcnow_iso(), "agent_id": previous, "role": prev_role, "kept_in_place": True,
                  "reason": f"superseded by {aid} (follow-up round {rnd})"})
    store.event("memo_superseded", previous=previous, current=aid, round=rnd)


def finish_batch(store: RunStore, cfg, batch_id: str) -> None:
    """Fold a verification batch's finished verifiers into the results and record the batch's outcome."""
    state = store.load_state()
    batch = ((state.get("verification") or {}).get("batches") or {}).get(batch_id)
    if not batch or batch.get("status") == "skipped":
        return
    results = collect(store, cfg, batch_id, state)
    statuses = [state["agents"].get(a, {}).get("status") for a in batch.get("agents") or []]
    counts: dict[str, int] = {}
    for rid in batch.get("requests") or []:
        st = (results["requests"].get(rid) or {}).get("status", "?")
        counts[st] = counts.get(st, 0) + 1
    record = {**batch, "status": "complete" if all(s == "complete" for s in statuses) else "incomplete",
              "finished_at": utcnow_iso(), "counts": counts}
    store.update_state(lambda st: st["verification"]["batches"].__setitem__(batch_id, record))
    store.event("verification_finished", batch=batch_id, counts=counts)


def batch_of(state: dict, agent_id: str) -> str | None:
    """The verification batch a verifier agent belongs to."""
    batches = (state.get("verification") or {}).get("batches") or {}
    return next((k for k, b in batches.items() if agent_id in (b.get("agents") or [])), None)


def apply_intake(store: RunStore, data: dict | None) -> None:
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
    atomic_write_text(store.source_dir / "claims_ledger.md", "\n".join(lines) + "\n")
    # Kept apart from title/field: agent headers use only user-given or extracted metadata, so the
    # orchestrator's reading of the paper never reaches refuters or judges.
    meta = store.load_metadata()
    meta["intake"] = {k: data.get(k) for k in ("title", "field", "submission_type", "venue_guess", "summary")}
    store.save_metadata(meta)


def build_matrix(store: RunStore) -> dict | None:
    """Cross-tabulate the usable (complete) refuters' objections against the complete judges' verdicts."""
    state = store.load_state()
    objections: dict[str, list[dict]] = {}
    judgments: dict[str, list[dict]] = {}
    unstructured, released_without_data = [], []
    for aid, a in state["agents"].items():
        if a["status"] != "complete":
            continue
        side = read_json(store.sidecar_path(aid, a["role"], ".json"))
        if a["role"] in REFUTER_ROLES:
            if side is None:
                released_without_data.append(aid)
            objections[aid] = (side or {}).get("objections") or []
        elif a["role"] == "judge":
            if side is None:
                unstructured.append(aid)
                continue
            judgments[aid] = side.get("judgments") or []
    if not judgments:
        return None
    excluded = {aid: gates.unavailable_label(aid, a) for aid, a in state["agents"].items()
                if a["role"] in REFUTER_ROLES + ("judge",) and a["status"] != "complete"}
    matrix = build_judgment_matrix(objections, judgments, excluded=excluded)
    matrix["judges_without_structured_output"] = unstructured
    matrix["refuters_without_structured_data"] = released_without_data
    gate = _evidence_gate(store, state, objections, judgments)
    if gate is not None:
        for row in matrix["rows"]:
            row["evidence"] = gate["by_objection"].get(row["id"])
        matrix["evidence_gate"] = {k: gate[k] for k in ("flagged_ids", "require_independent_check")}
    atomic_write_json(store.role_dir("judge") / "judgment_matrix.json", matrix)
    atomic_write_text(store.role_dir("judge") / "judgment_matrix.md", matrix_markdown(matrix))
    return matrix


def _evidence_gate(store: RunStore, state: dict, objections: dict[str, list[dict]],
                   judgments: dict[str, list[dict]]) -> dict | None:
    """Which serious prior-work verdicts are shown (evidence_gate.py); writes judges/evidence_gate.{json,md}."""
    cfg = store.load_config()
    evidence: dict[str, dict] = {}
    for aid, a in state["agents"].items():
        if a["role"] == "novelty" and a["status"] == "complete":
            evidence.update((read_json(store.sidecar_path(aid, "novelty", ".evidence.json"), {}) or {})
                            .get("objections") or {})
    verification = read_json(store.role_dir("verifier") / "results.json", {}) or {}
    known = {o["id"]: o for objs in objections.values() for o in objs}
    independent = cfg.evidence.require_independent_check and bool(cfg.verifier and cfg.verifier.enabled)
    gate = evaluate_evidence(judgments, known, evidence, verification, independent)
    by_objection: dict[str, str] = {}
    for oid, entry in evidence.items():
        if entry.get("decisive"):
            m = member_status(oid, evidence, verification)
            checks = ", ".join(f"{a}: {v.replace('anticipates_', '')}" for a, v in zip(m["verifiers"], m["verdicts"]))
            by_objection[oid] = m["quotes"].replace("_", " ") + (f"; {checks}" if checks else "")
    gate["by_objection"] = by_objection
    atomic_write_json(store.role_dir("judge") / "evidence_gate.json", gate)
    atomic_write_text(store.role_dir("judge") / "evidence_gate.md", gate_markdown(gate))
    return gate


def _verifier_quotes(text: str) -> list[str]:
    data, _ = gates.parse_json_lenient(text)
    out = []
    passages = (data or {}).get("passages")
    for p in passages if isinstance(passages, list) else []:
        if not isinstance(p, dict):
            continue
        for key in ("overlap", "differences"):
            pairs = p.get(key)
            for pair in pairs if isinstance(pairs, list) else []:
                if isinstance(pair, dict):
                    out += [str(pair.get(k) or "") for k in ("prior_passage", "submission_passage")]
    return [q for q in out if q.strip()]
