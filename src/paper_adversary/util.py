"""Small shared helpers: paths, atomic writes, hashing, locks, time."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def home_dir() -> Path:
    """Directory holding config/, prompts/ and (by default) runs/."""
    env = os.environ.get("PAPER_ADVERSARY_HOME")
    return Path(env).expanduser().resolve() if env else REPO_ROOT


def runs_root() -> Path:
    env = os.environ.get("PAPER_ADVERSARY_RUNS_DIR")
    return Path(env).expanduser().resolve() if env else home_dir() / "runs"


def prompts_dir() -> Path:
    env = os.environ.get("PAPER_ADVERSARY_PROMPTS_DIR")
    return Path(env).expanduser().resolve() if env else home_dir() / "prompts"


def config_dir() -> Path:
    return home_dir() / "config"


# ---------------------------------------------------------------- time


def utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat(timespec="seconds")


def parse_iso(value: str | None) -> _dt.datetime | None:
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def local_hm(value: str | _dt.datetime | None) -> str:
    """Render a timestamp as local wall-clock time for status reports."""
    dt = parse_iso(value) if isinstance(value, str) else value
    if dt is None:
        return "?"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


# ---------------------------------------------------------------- files


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, default=str, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return out


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(text: str, max_len: int = 28) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    text = text[:max_len].rstrip("_")
    return text or "paper"


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE .env file (no interpolation)."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            values[key] = val
    return values


def load_dotenv_into_environ() -> None:
    """Load <home>/.env without overriding variables already set."""
    for key, val in read_dotenv(home_dir() / ".env").items():
        if val and key not in os.environ:
            os.environ[key] = val


# ---------------------------------------------------------------- locks


class FileLock:
    """Advisory exclusive lock held for the lifetime of a process.

    The OS releases it when the holder dies, so a crashed worker never leaves
    a stale lock behind.
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self, blocking: bool = False) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                msvcrt.locking(fh.fileno(), mode, 1)
            else:
                import fcntl

                flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(fh.fileno(), flags)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "FileLock":
        self.acquire(blocking=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def lock_is_held(path: Path) -> bool:
    if not path.exists():
        return False
    probe = FileLock(path)
    if probe.acquire(blocking=False):
        probe.release()
        return False
    return True


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import subprocess

        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
