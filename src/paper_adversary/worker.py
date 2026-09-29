"""Detached worker processes.

MCP tools return immediately; the actual pipeline runs in a separate process
that survives Claude Desktop restarts. One worker per run at a time, enforced
by an OS-level lock on run.lock (released automatically if the worker dies).
The worker leads its own process group, so cancelling it also stops every
`claude` child and tool server it started.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time

from paper_adversary.store import RunStore
from paper_adversary.util import (
    FileLock,
    atomic_write_json,
    load_dotenv_into_environ,
    lock_is_held,
    pid_alive,
    read_json,
    utcnow_iso,
)

BUSY_EXIT = 3


class WorkerBusy(RuntimeError):
    pass


def worker_info(store: RunStore) -> dict:
    info = read_json(store.worker_path, {}) or {}
    info["running"] = lock_is_held(store.lock_path)
    return info


def launch(store: RunStore, stages: list[str], rerun: list[str] | None = None, allow_incomplete: bool = False,
           settle_seconds: float = 8.0) -> dict:
    """Start a detached worker for `stages`. Raises WorkerBusy if one is already running."""
    if lock_is_held(store.lock_path):
        info = worker_info(store)
        raise WorkerBusy(f"a worker (pid {info.get('pid')}) is already running stages "
                         f"{', '.join(info.get('stages') or [])} for this run")
    cmd = [sys.executable, "-m", "paper_adversary", "worker", "--run-dir", str(store.dir),
           "--stages", ",".join(stages)]
    if rerun:
        cmd += ["--rerun", ",".join(rerun)]
    if allow_incomplete:
        cmd.append("--allow-incomplete")
    log_path = store.dir / "logs" / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {"stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT, "close_fds": True,
                    "cwd": str(store.dir), "env": os.environ.copy()}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(log_path, "a", encoding="utf-8") as log:
        kwargs["stdout"] = log
        proc = subprocess.Popen(cmd, **kwargs)
    store.event("worker_launched", pid=proc.pid, stages=stages, rerun=rerun or [])

    deadline = time.monotonic() + settle_seconds
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            if code == BUSY_EXIT:
                raise WorkerBusy("another worker took the run lock first")
            info = worker_info(store)
            if info.get("pid") == proc.pid:
                return {"started": True, "pid": proc.pid, "finished_already": True}
            raise RuntimeError(f"worker exited immediately (code {code}); see {store.rel(log_path)}")
        info = read_json(store.worker_path, {}) or {}
        if info.get("pid") == proc.pid and lock_is_held(store.lock_path):
            return {"started": True, "pid": proc.pid}
        time.sleep(0.2)
    return {"started": True, "pid": proc.pid, "note": "worker is still starting"}


def cancel(store: RunStore, grace_seconds: float = 25.0) -> dict:
    info = read_json(store.worker_path, {}) or {}
    if not lock_is_held(store.lock_path):
        return {"cancelled": False, "reason": "no worker is running for this run"}
    pid = info.get("pid")
    if not pid or not pid_alive(pid):
        return {"cancelled": False, "reason": "the run lock is held but the worker pid is unknown; "
                "check `ps` for paper_adversary worker processes"}
    store.event("cancel_requested", pid=pid)
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/PID", str(pid)], capture_output=True)
        else:
            pgid = os.getpgid(pid)
            if pgid != pid or pgid == os.getpgid(0):
                os.kill(pid, signal.SIGTERM)  # never signal our own group
            else:
                os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return {"cancelled": True, "pid": pid, "note": "the worker had already exited"}
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not lock_is_held(store.lock_path):
            return {"cancelled": True, "pid": pid}
        time.sleep(0.5)
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"cancelled": True, "pid": pid, "forced": True}


def worker_main(run_dir: str, stages: list[str], rerun: list[str], allow_incomplete: bool) -> int:
    load_dotenv_into_environ()
    store = RunStore(run_dir)
    lock = FileLock(store.lock_path)
    # A status poll briefly takes the lock to test it; retry for a moment before concluding "busy".
    for _ in range(20):
        if lock.acquire(blocking=False):
            break
        time.sleep(0.1)
    else:
        print(f"[{utcnow_iso()}] run is busy; another worker holds the lock", flush=True)
        return BUSY_EXIT
    started = utcnow_iso()
    base = {"pid": os.getpid(), "stages": stages, "rerun": rerun, "started_at": started,
            "host": socket.gethostname(), "python": sys.executable}
    atomic_write_json(store.worker_path, {**base, "status": "running"})
    print(f"[{started}] worker {os.getpid()} starting stages {stages} rerun={rerun}", flush=True)
    outcome: dict = {"outcome": "crashed", "message": ""}
    try:
        outcome = asyncio.run(_amain(store, stages, rerun, allow_incomplete))
        return 0
    except Exception as exc:  # recorded for get_run_status; the run stays resumable
        outcome = {"outcome": "crashed", "message": f"{type(exc).__name__}: {exc}"}
        store.event("worker_crashed", error=outcome["message"])
        import traceback

        traceback.print_exc()
        return 1
    finally:
        atomic_write_json(store.worker_path, {**base, "status": "exited", "finished_at": utcnow_iso(), **outcome})
        print(f"[{utcnow_iso()}] worker {os.getpid()} finished: {outcome}", flush=True)
        lock.release()


async def _amain(store: RunStore, stages: list[str], rerun: list[str], allow_incomplete: bool) -> dict:
    from paper_adversary.pipeline import Pipeline

    pipeline = Pipeline(store)
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            loop.add_signal_handler(sig, pipeline.cancel.set)
    return await pipeline.run(stages, rerun=rerun, allow_incomplete=allow_incomplete)


async def wait_until_idle(store: RunStore, seconds: float, on_tick=None) -> bool:
    """Poll until no worker holds the run lock or `seconds` elapse. Returns True if idle."""
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        if not lock_is_held(store.lock_path):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if on_tick:
            await on_tick()
        await asyncio.sleep(min(2.0, remaining))
