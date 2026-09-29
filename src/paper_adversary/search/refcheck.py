"""Verify the references a novelty refuter cites, against scholarly APIs.

Fabricated or garbled prior work is the classic failure of automated novelty
checks. Every reference in a refuter's structured block is looked up by DOI,
arXiv ID or title; judges then see which citations could be confirmed.
"""

from __future__ import annotations

import asyncio
import difflib

from paper_adversary.search import LiteratureSearch
from paper_adversary.search.base import PaperRecord, norm_title

VERIFIED = 0.88
PARTIAL = 0.60


def title_similarity(a: str, b: str) -> float:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    return max(seq, jac)


def _unique(refs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for ref in refs:
        key = (str(ref.get("doi") or "").lower() or str(ref.get("arxiv_id") or "").lower()
               or norm_title(str(ref.get("title") or "")))
        if key and key not in seen:
            seen.add(key)
            out.append(ref)
    return out


def _best(records: list[PaperRecord], title: str | None) -> tuple[PaperRecord | None, float]:
    if not records:
        return None, 0.0
    if not title:
        return records[0], 1.0
    scored = sorted(((title_similarity(title, r.title), r) for r in records), key=lambda x: -x[0])
    return scored[0][1], scored[0][0]


def _canonical(rec: PaperRecord) -> dict:
    return {"title": rec.title, "year": rec.year, "venue": rec.venue, "doi": rec.doi, "arxiv_id": rec.arxiv_id,
            "url": rec.url, "authors": rec.authors[:6], "found_via": rec.sources or [rec.source]}


async def check_references(refs: list[dict], search: LiteratureSearch, max_refs: int = 60,
                           cancel: asyncio.Event | None = None) -> dict | None:
    """Look up each reference. Returns None if cancelled part-way (nothing partial is reported)."""
    items = []
    unique = _unique(refs)
    for ref in unique[:max_refs]:
        if cancel is not None and cancel.is_set():
            return None
        title = str(ref.get("title") or "").strip() or None
        doi = str(ref.get("doi") or "").strip() or None
        arxiv = str(ref.get("arxiv_id") or ref.get("arxiv") or "").strip() or None
        try:
            year = int(ref.get("year")) if ref.get("year") else None
        except (TypeError, ValueError):
            year = None
        item = {"cited": {k: ref.get(k) for k in ("title", "authors", "year", "venue", "doi", "arxiv_id", "url")},
                "objection": ref.get("objection")}
        statuses: dict = {}
        best, sim, via = None, 0.0, None
        for ident in ([f"doi:{doi}"] if doi else []) + ([f"arxiv:{arxiv}"] if arxiv else []):
            found = await search.lookup(ident)
            statuses.update(found["status"])
            cand, s = _best(found["records"], title)
            if cand is not None:
                best, sim, via = cand, s, ident
                break
        if best is None and title:
            found = await search.lookup(title)
            statuses.update(found["status"])
            best, sim = _best(found["records"], title)
            via = "title"
        reachable = any(st in ("ok", "cached") for st in statuses.values())
        if best is None:
            item["status"] = "not_found" if reachable else ("unverifiable" if not (title or doi or arxiv)
                                                            else "unchecked")
        elif via and via != "title" and title and sim < PARTIAL:
            item["status"] = "mismatch"
            item["note"] = f"the identifier {via} resolves to a different paper"
        elif sim >= VERIFIED or (via and via != "title" and (not title or sim >= PARTIAL)):
            item["status"] = "verified"
            if year and best.year and abs(year - best.year) > 1:
                item["status"] = "partial"
                item["note"] = f"cited year {year}, record says {best.year}"
        elif sim >= PARTIAL:
            item["status"] = "partial"
            item["note"] = f"closest title match has similarity {sim:.2f}"
        else:
            item["status"] = "not_found"
        if best is not None and item["status"] != "not_found":
            item["match"] = _canonical(best)
            item["title_similarity"] = round(sim, 3)
        item["providers"] = statuses
        items.append(item)
    counts: dict[str, int] = {}
    for it in items:
        counts[it["status"]] = counts.get(it["status"], 0) + 1
    return {"checked": len(items), "skipped": max(0, len(unique) - max_refs), "counts": counts, "items": items}


def refcheck_markdown(result: dict, agent_id: str) -> str:
    counts = result.get("counts", {})
    parts = [f"Reference check for {agent_id} (orchestrator lookup in scholarly databases): "
             f"{result.get('checked', 0)} checked — " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))]
    for it in result.get("items", []):
        if it["status"] == "verified":
            continue
        cited = it["cited"]
        label = cited.get("title") or cited.get("doi") or cited.get("arxiv_id") or "?"
        line = f"- {it['status'].upper()}: \"{label}\" ({cited.get('year') or 'n.d.'})"
        if it.get("note"):
            line += f" — {it['note']}"
        if it.get("match") and it["status"] in ("partial", "mismatch"):
            m = it["match"]
            line += f"; closest record: \"{m['title']}\" ({m.get('year')})"
        parts.append(line)
    if result.get("skipped"):
        parts.append(f"- {result['skipped']} further references were not checked (limit reached)")
    return "\n".join(parts)

