"""Versioned prompt templates, lenses and rubrics, all stored outside the code in prompts/."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from paper_adversary.util import prompts_dir, sha256_text

_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
_NAME = re.compile(r"[a-z0-9][a-z0-9_]*")  # a file stem under prompts/; never a path
DEFAULT_LENS_SET = "lenses_v1"
RUBRIC_SUFFIXES = (".md", ".markdown", ".txt")
RUBRIC_MAX_BYTES = 100_000


class PromptError(ValueError):
    pass


@dataclass(frozen=True)
class PromptTemplate:
    name: str  # file stem, e.g. "novelty_v1"; recorded as the prompt version
    meta: dict
    body: str
    sha256: str  # of the whole file, so in-place edits of a version are detectable
    path: Path

    @property
    def variables(self) -> set[str]:
        return set(_VAR.findall(self.body))

    def render(self, values: dict[str, object]) -> str:
        missing = self.variables - set(values)
        if missing:
            raise PromptError(f"prompt {self.name}: missing variables {sorted(missing)}")
        return _VAR.sub(lambda m: str(values[m.group(1)]), self.body)


def _split_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            meta = yaml.safe_load(text[4:end]) or {}
            return meta, text[end + 5 :].lstrip("\n")
    return {}, text


def check_name(name: str, what: str = "prompt") -> str:
    """Prompt, lens-set and rubric names are file stems. Per-run overrides arrive through MCP tool arguments,
    so a name must never be able to point outside prompts/."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise PromptError(f"invalid {what} name {name!r}: use lowercase letters, digits and underscores")
    return name


def prompt_path(name: str) -> Path:
    return prompts_dir() / f"{check_name(name)}.md"


def prompt_exists(name: str) -> bool:
    return isinstance(name, str) and bool(_NAME.fullmatch(name)) and prompt_path(name).is_file()


def load_prompt(name: str) -> PromptTemplate:
    path = prompt_path(name)
    if not path.is_file():
        raise PromptError(f"prompt '{name}' not found at {path}")
    text = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(text)
    return PromptTemplate(name=name, meta=meta, body=body, sha256=sha256_text(text), path=path)


def tool_access_note(tools: list[str]) -> str:
    """The `{{tool_access}}` sentence: what the agent can actually open or call, so a prompt never mentions a PDF or
    tool the agent lacks (an agent told about ./paper.pdf without a Read tool writes imaginary tool calls)."""
    if not tools:
        return "You have no tools: everything you need is in this message; do not write tool calls."
    parts = []
    if "read_pdf" in tools:
        parts.append("The original PDF of the submission is in your working folder as ./paper.pdf. Text extracted "
                     "from a PDF garbles equations and tables, so open it with the Read tool and read the relevant "
                     "pages whenever an equation, a table or a figure matters, and before asserting that one is "
                     "wrong.")
    if "paper_sections" in tools:
        parts.append("Sections of the submission that are not shown inline can be read in full with the "
                     "read_paper_section tool.")
    if not set(tools) - {"read_pdf", "paper_sections"}:
        parts.append("You have no other tools; do not write any other tool calls.")
    return " ".join(parts)


# ---------------------------------------------------------------- lenses


@lru_cache(maxsize=8)
def _load_lens_set(lens_set: str) -> dict:
    path = prompts_dir() / f"{check_name(lens_set, 'lens set')}.yaml"
    if not path.is_file():
        raise PromptError(f"lens set '{lens_set}' not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def lens_by_id(role: str, lens_id: str, lens_set: str = DEFAULT_LENS_SET) -> dict:
    """A role's lens by its id (an explicit per-agent assignment)."""
    for lens in _load_lens_set(lens_set).get(role) or []:
        if lens.get("id") == lens_id:
            return {**lens, "lens_set": lens_set}
    raise PromptError(f"lens '{lens_id}' is not a {role} lens in {lens_set}")


def assign_lens(role: str, index: int, n_agents: int, lens_set: str = DEFAULT_LENS_SET) -> dict | None:
    """Deterministically give agent `index` (1-based) of `role` a primary focus."""
    lenses = _load_lens_set(lens_set).get(role) or []
    if not lenses:
        return None
    lens = dict(lenses[(index - 1) % len(lenses)])
    cycle = (index - 1) // len(lenses)
    if cycle:
        lens["title"] = f"{lens['title']} (independent pass {cycle + 1})"
        lens["focus"] = (lens.get("focus", "").rstrip() + "\n\nAnother reviewer may share this focus. Work it "
                         "independently and push past the obvious first findings.")
    lens["lens_set"] = lens_set
    return lens


# ---------------------------------------------------------------- rubrics


def rubric_path(name: str) -> Path:
    return prompts_dir() / "rubrics" / f"{check_name(name, 'rubric')}.md"


MAX_RUBRIC_TESTS = 30
_TEST_ID = re.compile(r"[A-Z][A-Z0-9_]{0,15}")
_RESERVED_ID = re.compile(r"[A-Z]+\d+-[OI]\d+|[NRFJSCVAL]\d+")  # objection, item, agent and claims-ledger IDs


def rubric_tests(text: str) -> list[dict]:
    """The explicit tests a rubric declares in its front matter (`tests: [{id: RT1, text: ...}, ...]`), each of
    which judges (judge_v3) must rule on; [] when it declares none. A malformed declaration raises PromptError."""
    if not text.startswith("---\n"):
        return []
    try:
        meta, _ = _split_front_matter(text)
    except yaml.YAMLError:
        return []  # front matter that is not YAML declares nothing (such rubrics worked before tests existed)
    if not isinstance(meta, dict) or meta.get("tests") is None:
        return []
    raw = meta["tests"]
    if not isinstance(raw, list) or not raw:
        raise PromptError("rubric front matter: 'tests' must be a non-empty list of {id, text}")
    if len(raw) > MAX_RUBRIC_TESTS:
        raise PromptError(f"rubric front matter declares {len(raw)} tests; at most {MAX_RUBRIC_TESTS}")
    out, seen = [], set()
    for i, t in enumerate(raw, start=1):
        if not isinstance(t, dict) or not str(t.get("text") or "").strip():
            raise PromptError(f"rubric test {i} needs an id and a text")
        tid = str(t.get("id") or "").strip().upper()
        if not _TEST_ID.fullmatch(tid) or _RESERVED_ID.fullmatch(tid):
            raise PromptError(f"rubric test {i}: id {t.get('id')!r} must be a short name such as RT1 (letters, digits, "
                              "underscores), not an agent, objection or item ID (nor a claims-ledger ID such as L1)")
        if tid in seen:
            raise PromptError(f"rubric test id {tid} is used twice")
        seen.add(tid)
        out.append({"id": tid, "text": " ".join(str(t["text"]).split())})
    return out


def _looks_like_path(spec: str) -> bool:
    if "\n" in spec or len(spec) >= 1024:
        return False
    if spec.startswith(("/", "~", "./", "../", ".\\", "..\\")) or re.match(r"[A-Za-z]:[\\/]", spec):
        return True
    if spec.lower().endswith(RUBRIC_SUFFIXES + (".env",)):
        return True
    try:
        return Path(spec).expanduser().exists()
    except (OSError, ValueError):
        return False


def _rubric_file(spec: str) -> Path:
    """A rubric file given by path: a visible .md/.txt file of modest size, never a dotfile such as .env."""
    path = Path(spec).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise PromptError(f"rubric file not found: {spec}") from exc
    if not path.is_file():
        raise PromptError(f"rubric path is not a file: {spec}")
    if any(part.startswith(".") for part in path.parts[1:]):
        raise PromptError(f"rubric file {spec} is hidden or inside a hidden folder; copy it to a visible .md file")
    if path.suffix.lower() not in RUBRIC_SUFFIXES:
        raise PromptError(f"rubric file must be one of {', '.join(RUBRIC_SUFFIXES)}: {spec}")
    if path.stat().st_size > RUBRIC_MAX_BYTES:
        raise PromptError(f"rubric file is larger than {RUBRIC_MAX_BYTES // 1000} KB: {spec}")
    return path


def rubric_exists(spec: str | None) -> bool:
    if not spec:
        return False
    try:
        load_rubric(spec)
    except PromptError:
        return False
    return True


def load_rubric(spec: str | None) -> tuple[str, str]:
    """Return (label, text) for a rubric given by name (prompts/rubrics/), file path, or literal text. A named
    rubric loses its front matter unless it declares tests (rubric_tests), which then travel with the text."""
    from paper_adversary.isolation import find_secrets

    if not spec:
        return ("none", "")
    spec = spec.strip()
    if _NAME.fullmatch(spec):
        path = rubric_path(spec)
        if not path.is_file():
            raise PromptError(f"rubric '{spec}' not found in prompts/rubrics/")
        raw = path.read_text(encoding="utf-8")
        _, body = _split_front_matter(raw)
        return (spec, raw.strip() if rubric_tests(raw) else body.strip())
    if _looks_like_path(spec):
        path = _rubric_file(spec)
        label, text = str(path), path.read_text(encoding="utf-8", errors="replace").strip()
    else:
        label, text = "inline", spec
    secrets = find_secrets(text)
    if secrets:
        raise PromptError(f"rubric {label} looks like it contains a secret ({', '.join(secrets)}); refusing to use it")
    rubric_tests(text)  # a malformed test declaration fails here, before a run is created
    return (label, text)
