"""Command line: `paper-adversary serve` for Claude Desktop, plus the same operations for a terminal."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from paper_adversary.util import load_dotenv_into_environ


def _split(value: str | None) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _launch_or_run(store, stages: list[str], rerun: list[str], allow_incomplete: bool, foreground: bool,
                   wait: float) -> int:
    from paper_adversary import service
    from paper_adversary.worker import launch, worker_main

    if foreground:
        code = worker_main(str(store.dir), stages, rerun, allow_incomplete)
        print(service.status_text(store.run_id))
        return code
    info = launch(store, stages, rerun, allow_incomplete)
    print(f"Started worker pid {info['pid']} for {', '.join(stages)} (log: {store.dir / 'logs' / 'worker.log'})")
    if wait:
        from paper_adversary.util import lock_is_held

        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and lock_is_held(store.lock_path):
            time.sleep(5)
    print(service.status_text(store.run_id))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["tools-server"]:
        from paper_adversary.tools_server import main as tools_main

        return tools_main(argv[1:])
    if argv[:1] == ["extract-fulltext"]:
        from paper_adversary.search.extract import extract_main

        return extract_main(argv[1:])
    load_dotenv_into_environ()
    parser = argparse.ArgumentParser(prog="paper-adversary", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("serve", help="run the MCP server on stdio (what Claude Desktop launches)")

    w = sub.add_parser("worker", help=argparse.SUPPRESS)
    w.add_argument("--run-dir", required=True)
    w.add_argument("--stages", required=True)
    w.add_argument("--rerun", default="")
    w.add_argument("--allow-incomplete", action="store_true")

    def paper_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("paper", help="path to a .pdf/.md/.tex/.txt file, or '-' to read text from stdin")
        p.add_argument("--title")
        p.add_argument("--venue")
        p.add_argument("--field")
        p.add_argument("--rubric")
        p.add_argument("--config", help="config override: YAML/JSON file or inline YAML")
        p.add_argument("--type", default="auto", choices=["auto", "paper", "idea"])

    c = sub.add_parser("create", help="create a run (no agents started)")
    paper_args(c)

    r = sub.add_parser("review", help="create a run and execute the full pipeline")
    paper_args(r)
    r.add_argument("--foreground", action="store_true", help="run in this terminal instead of a detached worker")
    r.add_argument("--wait", type=float, default=0, help="seconds to wait before printing status")

    for name, helptext in (("refuters", "run refuters"), ("judges", "run judges"), ("synthesis", "run synthesis"),
                           ("critic", "run the completeness critic"), ("resume", "resume a run")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("run")
        p.add_argument("--foreground", action="store_true")
        p.add_argument("--wait", type=float, default=0)
        p.add_argument("--allow-incomplete", action="store_true")
        if name in ("refuters", "judges"):
            p.add_argument("--rerun", default="", help="comma-separated agent IDs to redo")
        if name == "refuters":
            p.add_argument("--phase", default="all", choices=["all", "novelty", "rigor", "fit"])
        if name in ("synthesis", "critic"):
            p.add_argument("--rerun", action="store_true")
        if name == "resume":
            p.add_argument("--through", default="critic",
                           choices=["refuters", "judges", "synthesis", "critic", "followup"])
    fu = sub.add_parser("followup", help="run the follow-up of the completeness critique (verify, adjudicate, "
                        "revise the memo, re-check)")
    fu.add_argument("run")
    fu.add_argument("--dry-run", action="store_true", help="only show what the next round would do")
    fu.add_argument("--rerun", default="", help="comma-separated follow-up agent IDs to redo")
    fu.add_argument("--foreground", action="store_true")
    fu.add_argument("--wait", type=float, default=0)
    vf = sub.add_parser("verify", help="re-check the evidence gate's unconfirmed verdicts with blind verifiers")
    vf.add_argument("run")
    vf.add_argument("--foreground", action="store_true")
    vf.add_argument("--wait", type=float, default=0)

    s = sub.add_parser("status", help="show run status")
    s.add_argument("run")
    rp = sub.add_parser("report", help="print a report")
    rp.add_argument("run")
    rp.add_argument("type")
    rp.add_argument("--agent")
    rp.add_argument("--offset", type=int, default=0)
    rp.add_argument("--max-chars", type=int, default=200000)
    rp.add_argument("--round", type=int, default=None, help="follow-up round for items / followup_matrix")
    co = sub.add_parser("cost", help="usage and cost by phase")
    co.add_argument("run")
    sub.add_parser("list", help="list runs")
    ca = sub.add_parser("cancel", help="stop a run's worker")
    ca.add_argument("run")
    d = sub.add_parser("desktop-config", help="print (or --install) the Claude Desktop MCP server entry")
    d.add_argument("--install", action="store_true", help="merge it into Claude Desktop's config (a backup is kept)")
    rel = sub.add_parser("release", help="release a quarantined agent output, with a recorded reason")
    rel.add_argument("run")
    rel.add_argument("agent")
    rel.add_argument("--reason", required=True, help="why the gate's finding does not hold")
    rel.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    v = sub.add_parser("validate", help="check config, CLI and login; --probe to test each model")
    v.add_argument("--config")
    v.add_argument("--probe", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from paper_adversary.server import main as serve

        serve()
        return 0
    if args.cmd == "worker":
        from paper_adversary.worker import worker_main

        return worker_main(args.run_dir, _split(args.stages), _split(args.rerun), args.allow_incomplete)

    from paper_adversary import service
    from paper_adversary.store import RunStore

    if args.cmd in ("create", "review"):
        text = sys.stdin.read() if args.paper == "-" else None
        info = service.create_run(None if text else args.paper, text, args.title, args.venue, args.field,
                                  args.rubric, args.config, args.type)
        print(service.format_created(info))
        if args.cmd == "create":
            return 0
        store = RunStore.open(info["run_id"])
        stages = service.full_review_stages(store, info)
        return _launch_or_run(store, stages, [], False, args.foreground, args.wait)
    if args.cmd in ("refuters", "judges", "synthesis", "critic", "resume"):
        store = RunStore.open(args.run)
        state = store.load_state()
        if args.cmd == "refuters":
            stages = service.stages_for_phase(args.phase, state)
            rerun = service.resolve_rerun(state, _split(args.rerun), stages)
        elif args.cmd == "judges":
            stages, rerun = ["judge"], service.resolve_rerun(state, _split(args.rerun), ["judge"])
        elif args.cmd in ("synthesis", "critic"):
            stages = [args.cmd]
            rerun = [a for a, v in state["agents"].items() if v["role"] == args.cmd] if args.rerun else []
        else:
            stages, rerun = service.resume_stages(state, args.through), []
            if not stages:
                print("Nothing to resume: every agent through that stage is complete or quarantined.")
                return 0
        return _launch_or_run(store, stages, rerun, args.allow_incomplete, args.foreground, args.wait)
    if args.cmd == "status":
        print(service.status_text(args.run))
    elif args.cmd == "report":
        print(service.get_report(args.run, args.type, args.agent, args.max_chars, args.offset, args.round))
    elif args.cmd == "followup":
        store = RunStore.open(args.run)
        rerun = service.resolve_followup_rerun(store.load_state(), _split(args.rerun))
        plan = service.followup_plan(args.run)
        print(plan)
        if args.dry_run or (plan.startswith("Nothing") and not rerun):
            return 0
        return _launch_or_run(store, ["followup"], rerun, False, args.foreground, args.wait)
    elif args.cmd == "verify":
        batch_id, n = service.verification_batch(args.run, True, None)
        print(f"Verification batch {batch_id}: {n} request(s)")
        return _launch_or_run(RunStore.open(args.run), [f"verify:{batch_id}"], [], False, args.foreground, args.wait)
    elif args.cmd == "cost":
        print(service.cost_text(args.run))
    elif args.cmd == "list":
        print(service.list_runs_text())
    elif args.cmd == "cancel":
        from paper_adversary.worker import cancel

        print(cancel(RunStore.open(args.run)))
    elif args.cmd == "release":
        store = RunStore.open(args.run)
        aid = args.agent.strip().upper()
        print(service.gate_text(store, aid, store.load_state()["agents"].get(aid) or {"role": "?", "status": "?"}))
        if not args.yes:
            answer = input(f"\nRelease {aid} so later stages read it? This overrides the gate. Type 'release' to "
                           "confirm: ")
            if answer.strip().lower() != "release":
                print("Not released.")
                return 1
        print(service.release_quarantine(args.run, aid, args.reason, "cli"))
    elif args.cmd == "validate":
        print(asyncio.run(service.validate_text(args.config, args.probe)))
    elif args.cmd == "desktop-config":
        return _desktop_config(args.install)
    return 0


def _desktop_config_path():
    from pathlib import Path

    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if sys.platform == "win32":
        import os

        return Path(os.environ.get("APPDATA", str(Path.home()))) / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config/Claude/claude_desktop_config.json"


def _desktop_config(install: bool) -> int:
    import json
    import shutil

    from paper_adversary.util import home_dir, utcnow

    entry = {"command": sys.executable, "args": ["-m", "paper_adversary", "serve"],
             "env": {"PAPER_ADVERSARY_HOME": str(home_dir())}}
    snippet = json.dumps({"mcpServers": {"paper-adversary": entry}}, indent=2)
    path = _desktop_config_path()
    if not install:
        print(f"Add this to {path} (merge into any existing mcpServers):\n{snippet}")
        return 0
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if path.exists():
        backup = path.with_name(path.name + f".bak-{utcnow().strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy2(path, backup)
        print(f"Backed up {path} to {backup}")
    data.setdefault("mcpServers", {})["paper-adversary"] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Registered 'paper-adversary' in {path}. Restart Claude Desktop to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
