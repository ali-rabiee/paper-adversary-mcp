"""MCP server for Claude Desktop: isolated multi-agent adversarial review of papers and ideas."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from paper_adversary import __version__, service
from paper_adversary.budget import BudgetError
from paper_adversary.config import ConfigError
from paper_adversary.ingest import IngestError
from paper_adversary.pipeline import PreconditionError, check_preconditions
from paper_adversary.store import RunNotFound, RunStore
from paper_adversary.util import load_dotenv_into_environ
from paper_adversary.worker import WorkerBusy, cancel, launch, wait_until_idle

log = logging.getLogger("paper_adversary")

INSTRUCTIONS = """\
Adversarial pre-submission review of a research paper or research idea.

Pipeline: independent refuters (novelty, rigor, fit/feasibility) -> independent judges ->
synthesis memo -> completeness critic. Refuters never see each other's reports and judges
never see each other's; this is enforced in code. Agents run as headless Claude Code
processes on the user's Claude plan.

Long steps (run_* tools) start a background worker and return right away. Follow progress with
get_run_status(run_id, wait_seconds=50). A full review takes tens of minutes. Read outputs one at a
time with get_report instead of pulling the whole run into the conversation.
"""

mcp = MCPServer("paper-adversary", version=__version__, instructions=INSTRUCTIONS)

_USER_ERRORS = (ConfigError, IngestError, RunNotFound, ValueError, WorkerBusy, PreconditionError, BudgetError,
                FileNotFoundError)


def _tool_errors(fn):
    """Turn expected failures into clean tool errors instead of stack traces."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except _USER_ERRORS as exc:
            raise ToolError(str(exc)) from exc
    return wrapper


async def _follow(store: RunStore, wait_seconds: int, ctx: Context | None) -> str:
    """Optionally block (bounded) while the worker runs, sending progress, then return the status report."""
    wait_seconds = max(0, min(int(wait_seconds or 0), 600))
    if wait_seconds:
        async def tick() -> None:
            if ctx is None:
                return
            agents = store.load_state().get("agents", {}).values()
            done = sum(1 for a in agents if a["status"] == "complete")
            running = ", ".join(a["agent_id"] for a in agents if a["status"] == "running")
            try:
                await ctx.report_progress(done, len(agents), f"{done}/{len(agents)} complete; running: {running or '-'}")
            except Exception:  # progress is best-effort
                pass
        await wait_until_idle(store, wait_seconds, tick)
    return service.status_text(store.run_id)


async def _start(store: RunStore, stages: list[str], rerun: list[str] | None, allow_incomplete: bool,
                 wait_seconds: int, ctx: Context | None, note: str = "") -> str:
    info = await asyncio.to_thread(launch, store, stages, rerun, allow_incomplete)
    head = f"Started worker (pid {info['pid']}) for: {', '.join(stages)}." + (f" {note}" if note else "")
    return head + "\n\n" + await _follow(store, wait_seconds, ctx)


def _stale_warning(store: RunStore, rerun: list[str]) -> str:
    if not rerun:
        return ""
    state = store.load_state()
    roles = {state["agents"][a]["role"] for a in rerun}
    downstream = {"novelty": ("judge", "synthesis", "critic"), "rigor": ("judge", "synthesis", "critic"),
                  "fit": ("judge", "synthesis", "critic"), "judge": ("synthesis", "critic"),
                  "synthesis": ("critic",), "intake": ("synthesis", "critic")}
    affected = sorted({aid for r in roles for aid, a in state["agents"].items()
                       if a["role"] in downstream.get(r, ()) and a["status"] == "complete"})
    return (f"Note: {', '.join(affected)} used the old outputs and will show as stale until rerun."
            if affected else "")


# ---------------------------------------------------------------- tools


@mcp.tool()
@_tool_errors
async def create_review_run(paper_path: str | None = None, paper_text: str | None = None, title: str | None = None,
                            venue: str | None = None, field: str | None = None, rubric: str | None = None,
                            config_override: dict[str, Any] | str | None = None,
                            submission_type: Literal["auto", "paper", "idea"] = "auto") -> str:
    """Create a review run for a paper (PDF, Markdown, LaTeX .tex, or text file) or a pasted research idea.

    Ingests the submission (section-aware text extraction), records metadata, and plans the agents
    from the model/effort configuration. Does not start any agent. Returns the run_id, detected
    metadata and the planned agent configuration.

    Args:
        paper_path: Absolute path to the paper file. Give this or paper_text.
        paper_text: The paper or idea as text/Markdown.
        title: Title override (otherwise detected).
        venue: Target venue, e.g. "ICML 2027".
        field: Research field, e.g. "robot learning".
        rubric: Rubric name from prompts/rubrics/, a path to a rubric file, or the rubric text.
        config_override: Partial config (object, or YAML/JSON text, or a path) merged over config/default.yaml,
            e.g. {"novelty": {"agents": 2}, "rigor": {"effort": "high"}}.
        submission_type: "paper", "idea", or "auto" (detect).
    """
    info = await asyncio.to_thread(service.create_run, paper_path, paper_text, title, venue, field, rubric,
                                   config_override, submission_type)
    return service.format_created(info) + "\n\nNext: run_refuters(run_id) or run_full_review for the whole pipeline."


@mcp.tool()
@_tool_errors
async def run_refuters(run_id: str, phase: Literal["all", "novelty", "rigor", "fit"] = "all",
                       rerun_agents: list[str] | None = None, wait_seconds: int = 0, ctx: Context = None) -> str:
    """Run the independent refuters (novelty, rigor, fit/feasibility) in a background worker.

    Refuters run in parallel (bounded) and never see each other's outputs. Completed agents are skipped
    unless listed in rerun_agents (their old reports are archived). With phase="all", the orchestrator's
    intake pass runs too.

    Args:
        run_id: The run to work on.
        phase: Which refuter group to run.
        rerun_agents: Agent IDs to redo even if complete, e.g. ["N2", "R1"].
        wait_seconds: Block up to this many seconds (max 600) while it runs, then return the status.
    """
    store = RunStore.open(run_id)
    state = store.load_state()
    stages = service.stages_for_phase(phase, state)
    rerun = service.resolve_rerun(state, rerun_agents, stages)
    return await _start(store, stages, rerun, False, wait_seconds, ctx, _stale_warning(store, rerun))


@mcp.tool()
@_tool_errors
async def run_judges(run_id: str, rerun_agents: list[str] | None = None, allow_incomplete_refuters: bool = False,
                     wait_seconds: int = 0, ctx: Context = None) -> str:
    """Run the independent judges over all refuter reports (background worker).

    Requires every refuter to be complete, unless allow_incomplete_refuters is true (then judges are told
    which reports are missing). Each judge sees the paper, all refuter reports, reference checks and the
    rubric, but never another judge's output. Afterwards the orchestrator builds a judgment matrix.

    Args:
        run_id: The run to work on.
        rerun_agents: Judge IDs to redo even if complete, e.g. ["J2"].
        allow_incomplete_refuters: Proceed although some refuters failed.
        wait_seconds: Block up to this many seconds (max 600), then return the status.
    """
    store = RunStore.open(run_id)
    state = store.load_state()
    rerun = service.resolve_rerun(state, rerun_agents, ["judge"])
    problems = check_preconditions(state, "judge", allow_incomplete_refuters)
    if problems:
        raise ToolError("Cannot run judges yet: " + "; ".join(problems)
                        + ". Finish the refuters (run_refuters / resume_run) or pass allow_incomplete_refuters=true.")
    return await _start(store, ["judge"], rerun, allow_incomplete_refuters, wait_seconds, ctx,
                        _stale_warning(store, rerun))


@mcp.tool()
@_tool_errors
async def run_synthesis(run_id: str, rerun: bool = False, allow_incomplete: bool = False, wait_seconds: int = 0,
                        ctx: Context = None) -> str:
    """Write the synthesis memo from the paper, all refuter reports and all judge reports (background worker).

    Args:
        run_id: The run to work on.
        rerun: Redo the memo even if it exists (the old one is archived).
        allow_incomplete: Proceed although some refuters or judges failed.
        wait_seconds: Block up to this many seconds (max 600), then return the status.
    """
    store = RunStore.open(run_id)
    state = store.load_state()
    problems = check_preconditions(state, "synthesis", allow_incomplete)
    if problems:
        raise ToolError("Cannot run synthesis yet: " + "; ".join(problems))
    ids = [a for a, v in state["agents"].items() if v["role"] == "synthesis"] if rerun else None
    return await _start(store, ["synthesis"], ids, allow_incomplete, wait_seconds, ctx,
                        _stale_warning(store, ids or []))


@mcp.tool()
@_tool_errors
async def run_completeness_critic(run_id: str, rerun: bool = False, allow_incomplete: bool = False,
                                  wait_seconds: int = 0, ctx: Context = None) -> str:
    """Run the completeness critic over the entire run (background worker).

    The critic is not another reviewer: it hunts for failure modes, novelty threats, assumptions, missing
    controls and interpretation problems that every earlier agent missed, and checks whether the
    synthesis over-weighted consensus.

    Args:
        run_id: The run to work on.
        rerun: Redo the critique even if it exists (the old one is archived).
        allow_incomplete: Proceed although upstream agents failed.
        wait_seconds: Block up to this many seconds (max 600), then return the status.
    """
    store = RunStore.open(run_id)
    state = store.load_state()
    problems = check_preconditions(state, "critic", allow_incomplete)
    if problems:
        raise ToolError("Cannot run the completeness critic yet: " + "; ".join(problems))
    ids = [a for a, v in state["agents"].items() if v["role"] == "critic"] if rerun else None
    return await _start(store, ["critic"], ids, allow_incomplete, wait_seconds, ctx)


@mcp.tool()
@_tool_errors
async def run_full_review(paper_path: str | None = None, paper_text: str | None = None, title: str | None = None,
                          venue: str | None = None, field: str | None = None, rubric: str | None = None,
                          config: dict[str, Any] | str | None = None,
                          submission_type: Literal["auto", "paper", "idea"] = "auto", wait_seconds: int = 0,
                          ctx: Context = None) -> str:
    """Create a run and execute the whole pipeline in one background worker:
    intake + novelty/rigor/fit refuters -> judges -> synthesis -> completeness critic.

    Returns at once (or after wait_seconds) with a compact status report and the artifact locations.
    Poll get_run_status(run_id, wait_seconds=50) until it completes; if interrupted, resume_run continues
    from where it stopped without redoing finished agents.

    Args: same as create_review_run (config = config override), plus wait_seconds (max 600).
    """
    info = await asyncio.to_thread(service.create_run, paper_path, paper_text, title, venue, field, rubric, config,
                                   submission_type)
    store = RunStore.open(info["run_id"])
    stages = [s for s in ("intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic")
              if any(a["role"] == s for a in info["agents"])]
    started = await _start(store, stages, None, False, wait_seconds, ctx)
    return service.format_created(info) + "\n\n" + started


@mcp.tool()
@_tool_errors
async def resume_run(run_id: str, through: Literal["refuters", "judges", "synthesis", "critic"] = "critic",
                     allow_incomplete: bool = False, wait_seconds: int = 0, ctx: Context = None) -> str:
    """Continue an interrupted or partial run up to a stage, skipping every completed agent.

    Failed and interrupted agents are retried. For example, a run stopped after the refuters resumes
    at the judges.

    Args:
        run_id: The run to resume.
        through: Last stage to run.
        allow_incomplete: Let later stages proceed even if some upstream agents still fail.
        wait_seconds: Block up to this many seconds (max 600), then return the status.
    """
    store = RunStore.open(run_id)
    state = store.load_state()
    order = ["intake", "novelty", "rigor", "fit", "judge", "synthesis", "critic"]
    last = {"refuters": "fit", "judges": "judge", "synthesis": "synthesis", "critic": "critic"}[through]
    stages = [s for s in order[: order.index(last) + 1]
              if any(a["role"] == s and a["status"] != "complete" for a in state["agents"].values())]
    if not stages:
        return "Nothing to resume: every agent through that stage is complete.\n\n" + service.status_text(run_id)
    return await _start(store, stages, None, allow_incomplete, wait_seconds, ctx)


@mcp.tool()
@_tool_errors
async def get_run_status(run_id: str, wait_seconds: int = 0, ctx: Context = None) -> str:
    """Status of every agent (complete / running / retrying / waiting for a plan limit / failed), failed calls
    and retry counts, isolation-audit results, stale outputs, and artifact locations.

    Args:
        run_id: The run.
        wait_seconds: If a worker is running, wait up to this many seconds (max 600) for it to finish first.
    """
    store = RunStore.open(run_id)
    return await _follow(store, wait_seconds, ctx)


@mcp.tool()
@_tool_errors
async def get_report(run_id: str, report_type: str, agent_id: str | None = None, offset: int = 0,
                     max_chars: int = 40000) -> str:
    """Read one output of a run without loading everything.

    report_type: novelty | rigor | fit | judge | synthesis | critic | intake (profile) — with agent_id
    (e.g. "N2", "J1") for a single report, or without it for a one-line summary of each;
    matrix (judgment matrix) | claims (claims ledger) | refcheck | metadata | config | paper (extracted
    text) | sections | references | usage | events | index (list of files); per-agent diagnostics with
    agent_id: context (input manifest) | transcript | prompt | search_log.
    Long texts are paged: pass offset to continue.
    """
    max_chars = max(1000, min(int(max_chars), 200_000))
    return service.get_report(run_id, report_type, agent_id, max_chars, int(offset))


@mcp.tool()
@_tool_errors
async def get_run_cost(run_id: str) -> str:
    """Token usage and cost by phase and in total: input, cache writes/reads, output, web searches, agent time,
    the API-equivalent cost Claude Code reports, and an estimate from config/models.yaml prices.
    Calls that returned no usage are listed as missing, never estimated."""
    return await asyncio.to_thread(service.cost_text, run_id)


@mcp.tool()
@_tool_errors
async def list_runs(limit: int = 20) -> str:
    """List review runs, newest first, with their state and progress."""
    return service.list_runs_text(max(1, min(int(limit), 200)))


@mcp.tool()
@_tool_errors
async def cancel_run(run_id: str) -> str:
    """Stop the background worker of a run (and every agent process it started). Completed agents are kept;
    the run can be resumed later with resume_run."""
    store = RunStore.open(run_id)
    result = await asyncio.to_thread(cancel, store)
    head = ("Cancelled" + (" (forced)" if result.get("forced") else "") if result.get("cancelled")
            else f"Nothing cancelled: {result.get('reason')}")
    return head + "\n\n" + service.status_text(run_id)


@mcp.tool()
@_tool_errors
async def validate_config(config_override: dict[str, Any] | str | None = None, probe_models: bool = False) -> str:
    """Check the pipeline configuration: model aliases resolve, effort levels are supported, prompts exist,
    the Claude Code CLI is installed and logged in, and no API key would be billed. With probe_models=true,
    also sends one tiny low-effort request per distinct model to confirm your plan can use it."""
    return await service.validate_text(config_override, probe_models)


def main() -> None:
    load_dotenv_into_environ()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
