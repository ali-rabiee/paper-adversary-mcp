"""Run directory layout and persistent state.

Only the worker process mutates state.json after a run is created, and it does
so while holding run.lock. Every write is atomic (temp file + rename), so a
reader never sees a half-written file.
"""

from __future__ import annotations

import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

from paper_adversary.config import INTAKE_ID, STAGE_ORDER, PipelineConfig, validate_config
from paper_adversary.util import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    read_json,
    runs_root,
    slugify,
    utcnow,
    utcnow_iso,
)

ROLE_DIR = {
    "intake": "source",
    "novelty": "novelty",
    "rigor": "rigor",
    "fit": "fit",
    "verifier": "verify",
    "judge": "judges",
    "synthesis": "synthesis",
    "critic": "critic",
    "adjudicator": "followup",
    "revision": "synthesis",
    "recheck": "critic",
}

class RunNotFound(LookupError):
    pass


class RunStore:
    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir).resolve()
        self._lock = threading.RLock()

    # ------------------------------------------------------------ identity

    @property
    def run_id(self) -> str:
        return self.dir.name

    @staticmethod
    def new_run_id(root: Path, title: str | None, kind: str) -> str:
        slug = slugify(title or kind or "paper", max_len=28)
        year = utcnow().year
        pattern = re.compile(rf"^{re.escape(slug)}_{year}_(\d{{3,}})$")
        taken = [int(m.group(1)) for p in root.glob(f"{slug}_{year}_*") if (m := pattern.match(p.name))]
        return f"{slug}_{year}_{(max(taken) + 1) if taken else 1:03d}"

    @classmethod
    def open(cls, run: str | Path) -> "RunStore":
        candidate = Path(run).expanduser()
        if not candidate.is_absolute():
            candidate = runs_root() / str(run)
        if not (candidate / "metadata.json").is_file():
            raise RunNotFound(f"no run '{run}' (looked in {candidate})")
        return cls(candidate)

    @classmethod
    def create(cls, root: Path, run_id: str, metadata: dict, raw_config: dict) -> "RunStore":
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        store = cls(run_dir)
        for sub in ("source", "novelty", "rigor", "fit", "verify", "prior", "judges", "synthesis", "critic",
                    "followup", "logs/agents", "archive"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        atomic_write_json(store.metadata_path, metadata)
        atomic_write_text(store.config_path, yaml.safe_dump(raw_config, sort_keys=False, allow_unicode=True))
        return store

    # ------------------------------------------------------------ paths

    @property
    def metadata_path(self) -> Path:
        return self.dir / "metadata.json"

    @property
    def config_path(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def lock_path(self) -> Path:
        return self.dir / "run.lock"

    @property
    def worker_path(self) -> Path:
        return self.dir / "logs" / "worker.json"

    @property
    def events_path(self) -> Path:
        return self.dir / "logs" / "events.jsonl"

    @property
    def usage_path(self) -> Path:
        return self.dir / "logs" / "usage.jsonl"

    @property
    def usage_summary_path(self) -> Path:
        return self.dir / "logs" / "api_usage.json"

    @property
    def source_dir(self) -> Path:
        return self.dir / "source"

    @property
    def prior_dir(self) -> Path:
        """Text snapshots of the prior-work full texts read during this run (search/fulltext.py)."""
        return self.dir / "prior"

    def role_dir(self, role: str) -> Path:
        return self.dir / ROLE_DIR[role]

    def report_path(self, agent_id: str, role: str) -> Path:
        if role == "intake":
            return self.source_dir / "profile.md"
        if role in ("synthesis", "revision"):
            return self.role_dir(role) / ("memo.md" if agent_id == "S1" else f"memo_{agent_id}.md")
        if role in ("critic", "recheck"):
            return self.role_dir(role) / ("completeness.md" if agent_id == "C1" else f"completeness_{agent_id}.md")
        return self.role_dir(role) / f"{agent_id}.md"

    def sidecar_path(self, agent_id: str, role: str, suffix: str) -> Path:
        """suffix e.g. '.json', '.refcheck.json', '.search_log.jsonl'."""
        report = self.report_path(agent_id, role)
        return report.with_name(report.stem + suffix)

    def gate_path(self, agent_id: str, role: str) -> Path:
        return self.sidecar_path(agent_id, role, ".gate.json")

    def agent_log_dir(self, agent_id: str) -> Path:
        path = self.dir / "logs" / "agents" / agent_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.dir))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------ metadata / config

    def load_metadata(self) -> dict:
        return read_json(self.metadata_path, {}) or {}

    def save_metadata(self, meta: dict) -> None:
        with self._lock:
            atomic_write_json(self.metadata_path, meta)

    def load_raw_config(self) -> dict:
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}

    def load_config(self) -> PipelineConfig:
        return validate_config(self.load_raw_config())

    # ------------------------------------------------------------ state

    def load_state(self) -> dict:
        return read_json(self.state_path, {}) or {}

    def save_state(self, state: dict) -> None:
        with self._lock:
            state["updated_at"] = utcnow_iso()
            atomic_write_json(self.state_path, state)

    def update_state(self, mutate: Callable[[dict], Any]) -> Any:
        with self._lock:
            state = self.load_state()
            result = mutate(state)
            self.save_state(state)
            return result

    def init_state(self, agents: list[dict], verification: bool = False) -> None:
        state = {
            "run_id": self.run_id,
            "created_at": utcnow_iso(),
            "agents": {},
            "stages": {s: {"status": "pending"} for s in STAGE_ORDER},
            "calibration": {"tokens_per_char": None, "samples": []},
            "preflight": {},
        }
        if verification:  # verifier agents are planned later, from the refuters' objections
            state["verification"] = {"next_index": 1, "batches": {}}
        for spec in agents:
            state["agents"][spec["agent_id"]] = {
                **spec,
                "status": "pending",
                "attempts": 0,
                "failures": [],
            }
        if not any(a["agent_id"] == INTAKE_ID for a in agents):
            state["stages"]["intake"] = {"status": "skipped"}
        self.save_state(state)

    # ------------------------------------------------------------ logs

    def event(self, name: str, /, **data: Any) -> None:
        append_jsonl(self.events_path, {"at": utcnow_iso(), "run_id": self.run_id, "event": name, **data})

    def record_usage(self, record: dict) -> None:
        append_jsonl(self.usage_path, {"at": utcnow_iso(), "run_id": self.run_id, **record})

    # ------------------------------------------------------------ archive

    def archive_agent_outputs(self, agent_id: str, role: str, reason: str,
                              keep_suffixes: tuple[str, ...] = ()) -> Path | None:
        """Move an agent's previous outputs into archive/<timestamp>/ (never delete)."""
        report = self.report_path(agent_id, role)
        siblings = [p for p in report.parent.iterdir()
                    if p.is_file() and (p.name == report.name or p.name.startswith(report.stem + "."))
                    and not any(p.name.endswith(sfx) for sfx in keep_suffixes)]
        if not siblings:
            return None
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%fZ")  # microseconds: two archives in one second must not merge
        dest = self.dir / "archive" / stamp / ROLE_DIR[role]
        dest.mkdir(parents=True, exist_ok=True)
        for path in siblings:
            shutil.move(str(path), dest / path.name)
        append_jsonl(self.dir / "archive" / "index.jsonl",
                     {"at": utcnow_iso(), "agent_id": agent_id, "role": role, "reason": reason,
                      "moved_to": self.rel(dest), "files": [p.name for p in siblings]})
        return dest


def list_runs(root: Path | None = None) -> list[RunStore]:
    root = root or runs_root()
    if not root.is_dir():
        return []
    runs = [RunStore(p) for p in root.iterdir() if (p / "metadata.json").is_file()]
    return sorted(runs, key=lambda s: (s.load_state().get("created_at") or ""), reverse=True)
