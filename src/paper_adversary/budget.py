"""Context-window budgeting and the long-document strategy.

A paper is never silently truncated. If the full text does not fit an agent's
input budget, the plan keeps the role's highest-priority sections inline in
document order, replaces every other section with an explicit placeholder,
and gives the agent a `read_paper_section` tool that returns any section in
full. The plan is recorded in the agent's metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from paper_adversary.ingest import Section


class BudgetError(RuntimeError):
    pass


@dataclass
class DocPlan:
    mode: str  # "full" | "sectioned"
    text: str  # what goes inline
    inline_ids: list[str] = field(default_factory=list)
    omitted: list[dict] = field(default_factory=list)
    paper_tokens: int = 0
    full_tokens: int = 0
    budget_tokens: int = 0

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "paper_tokens_inline_est": self.paper_tokens,
            "paper_tokens_full_est": self.full_tokens,
            "budget_tokens": self.budget_tokens,
            "omitted_sections": [o["id"] for o in self.omitted],
        }


def estimate_tokens(chars: int | str, tokens_per_char: float) -> int:
    n = chars if isinstance(chars, int) else len(chars)
    return int(n * tokens_per_char) + 1


def input_budget(context_window: int, output_reserve: int, safety_margin: int) -> int:
    return max(0, context_window - output_reserve - safety_margin)


def _units(sections: list[Section]) -> list[Section]:
    """Non-overlapping top-level units (title/level-0 and level-1 sections)."""
    return [s for s in sections if s.level <= 1]


def plan_document(text: str, sections: list[Section], budget_tokens: int, priorities: list[str],
                  tokens_per_char: float) -> DocPlan:
    full = estimate_tokens(text, tokens_per_char)
    if full <= budget_tokens:
        return DocPlan("full", text, [s.id for s in sections], [], full, full, budget_tokens)
    units = _units(sections)
    if not units:
        raise BudgetError(f"paper is ~{full} tokens but the budget is {budget_tokens}, and no section "
                          "structure was detected to split it; use a larger-context model or shorten the input")

    # Text before the first unit (if any) is kept as part of the first unit's slot.
    rank = {kind: i for i, kind in enumerate(priorities)}
    order = sorted(units, key=lambda s: (rank.get(s.kind, len(rank)), s.start))
    chosen: set[str] = set()
    used = estimate_tokens(units[0].start, tokens_per_char) if units[0].start else 0
    placeholder_cost = 60
    used += placeholder_cost * len(units)
    for s in order:
        cost = estimate_tokens(s.chars, tokens_per_char)
        if s.kind in {"title", "abstract"} or used + cost <= budget_tokens:
            chosen.add(s.id)
            used += cost
    if used > budget_tokens:
        raise BudgetError(f"even the title and abstract (~{used} tokens) exceed the input budget "
                          f"({budget_tokens}); the other inputs leave no room for the paper")

    parts: list[str] = [text[: units[0].start]] if units[0].start else []
    omitted: list[dict] = []
    for s in units:
        if s.id in chosen:
            parts.append(text[s.start : s.end])
            continue
        est = estimate_tokens(s.chars, tokens_per_char)
        pages = f", pp. {s.page_start}-{s.page_end}" if s.page_start else ""
        omitted.append({"id": s.id, "title": s.title, "kind": s.kind, "est_tokens": est,
                        "page_start": s.page_start, "page_end": s.page_end})
        parts.append(
            f"\n> [Section {s.id} \"{s.title}\" ({s.kind}{pages}, ~{est:,} tokens) is not shown inline. "
            f"Call read_paper_section(\"{s.id}\") to read it in full.]\n\n"
        )
    inline_text = "".join(parts)
    return DocPlan("sectioned", inline_text, sorted(chosen), omitted,
                   estimate_tokens(inline_text, tokens_per_char), full, budget_tokens)


def toc_markdown(sections: list[Section]) -> str:
    lines = []
    for s in sections:
        if s.level == 0:
            continue
        pages = f" (pp. {s.page_start}-{s.page_end})" if s.page_start else ""
        lines.append(f"{'  ' * (s.level - 1)}- {s.id} {s.title} [{s.kind}]{pages}")
    return "\n".join(lines)
