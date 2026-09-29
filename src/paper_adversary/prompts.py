"""Versioned prompt templates, lenses and rubrics, all stored outside the code in prompts/."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from paper_adversary.util import prompts_dir, sha256_text

_VAR = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
DEFAULT_LENS_SET = "lenses_v1"


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


def prompt_path(name: str) -> Path:
    return prompts_dir() / f"{name}.md"


def prompt_exists(name: str) -> bool:
    return prompt_path(name).is_file()


def load_prompt(name: str) -> PromptTemplate:
    path = prompt_path(name)
    if not path.is_file():
        raise PromptError(f"prompt '{name}' not found at {path}")
    text = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(text)
    return PromptTemplate(name=name, meta=meta, body=body, sha256=sha256_text(text), path=path)


# ---------------------------------------------------------------- lenses


@lru_cache(maxsize=8)
def _load_lens_set(lens_set: str) -> dict:
    path = prompts_dir() / f"{lens_set}.yaml"
    if not path.is_file():
        raise PromptError(f"lens set '{lens_set}' not found at {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
    return prompts_dir() / "rubrics" / f"{name}.md"


def rubric_exists(spec: str | None) -> bool:
    if not spec:
        return False
    return rubric_path(spec).is_file() or Path(spec).expanduser().is_file() or "\n" in spec


def load_rubric(spec: str | None) -> tuple[str, str]:
    """Return (label, text) for a rubric given by name, file path, or literal text."""
    if not spec:
        return ("none", "")
    if rubric_path(spec).is_file():
        meta, body = _split_front_matter(rubric_path(spec).read_text(encoding="utf-8"))
        return (spec, body.strip())
    candidate = Path(spec).expanduser()
    if len(spec) < 1024 and "\n" not in spec and candidate.is_file():
        return (str(candidate), candidate.read_text(encoding="utf-8").strip())
    return ("inline", spec.strip())
