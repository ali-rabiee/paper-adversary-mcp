"""Quote checks for novelty objections: are the refuter's passages really in the prior paper and the submission?

For every decisive prior-work objection (reports.is_decisive), each overlap_evidence pair is matched
deterministically (passages.py): the prior passage against the prior paper's full text, the submission passage
against the submission. Full texts come from the run's prior/ snapshot, fetched now if the refuter did not read
them through its tools. The result per objection is one of quotes_verified, quotes_approximate, quotes_not_found,
no_evidence, fulltext_unavailable, prior_not_found or not_required, stored in novelty/<agent>.evidence.json and
shown to judges next to the reference check.
"""

from __future__ import annotations

from paper_adversary.isolation import _shingles
from paper_adversary.passages import SourceText, match_passage
from paper_adversary.reports import is_decisive, refuter_objections
from paper_adversary.search.fulltext import FullTextResult, FullTextStore, title_similarity
from paper_adversary.util import read_json, utcnow_iso

SAME_DOCUMENT = 0.5  # share of word 8-grams a prior text may share with the submission before it *is* the submission


class SubmissionFingerprint:
    """Recognizes a 'prior paper' that is really the submission itself (e.g. its own arXiv preprint)."""

    def __init__(self, title: str | None, text: str):
        self.title = title or ""
        self.shingles = set(_shingles(text))

    def matches(self, result: FullTextResult) -> bool:
        if not result.available:
            return False
        if self.title and result.title and title_similarity(self.title, result.title) >= 0.9:
            return True
        other = set(_shingles(result.text_md or ""))
        small = min(len(other), len(self.shingles))
        return bool(small) and len(other & self.shingles) / small >= SAME_DOCUMENT


def classify_reference(ref: dict, result: FullTextResult, me: SubmissionFingerprint) -> str:
    """available | is_submission | mismatch (the identifier resolves to another paper) | not_found | unavailable"""
    if result.status == "prior_not_found":
        return "not_found"
    cited = str(ref.get("title") or "")
    if result.title and cited and title_similarity(cited, result.title) < 0.6:
        return "mismatch"
    if me.matches(result):
        return "is_submission"
    return "available" if result.available else "unavailable"


def reference_identifier(ref: dict) -> str | None:
    return (str(ref.get("arxiv_id") or "").strip() or str(ref.get("doi") or "").strip()
            or str(ref.get("title") or "").strip() or None)


def quoted_passages(data: dict | None) -> list[str]:
    """Every passage an agent quoted in overlap_evidence (excluded from its fingerprint: it is not its own text)."""
    out = []
    for obj in (data or {}).get("objections") or []:
        for pair in obj.get("overlap_evidence") or [] if isinstance(obj, dict) else []:
            if isinstance(pair, dict):
                out += [str(pair.get(k) or "") for k in ("prior_passage", "submission_passage")]
    return [p for p in out if p.strip()]


async def check_agent_evidence(store, cfg, aid: str, data: dict | None, fetch: FullTextStore) -> dict:
    """The evidence record for one novelty refuter (written by the caller to <agent>.evidence.json)."""
    index = read_json(store.source_dir / "sections.json", {}) or {}
    text = (store.source_dir / "extracted_text.md").read_text(encoding="utf-8")
    submission = SourceText(text, index.get("sections"))
    me = SubmissionFingerprint(store.load_metadata().get("title"), text)
    kinds = {s["id"]: s.get("kind") for s in index.get("sections") or []}
    ev = cfg.evidence
    sources: dict[str, SourceText] = {}
    record = {"agent_id": aid, "checked_at": utcnow_iso(), "objections": {}}
    for obj in refuter_objections(data, aid):
        refs = [r for r in obj.get("references") or [] if isinstance(r, dict)]
        entry: dict = {"title": obj.get("title"), "category": obj.get("category"),
                       "severity_estimate": obj.get("severity_estimate"), "decisive": is_decisive(obj),
                       "evidence_basis": obj.get("evidence_basis"), "references": [], "pairs": []}
        record["objections"][obj["id"]] = entry
        if not entry["decisive"]:
            entry["status"] = "not_required"
            continue
        texts: dict[int, FullTextResult] = {}
        usable: dict[int, bool] = {}
        for i, ref in enumerate(refs, start=1):
            ident = reference_identifier(ref)
            result = await fetch.get(ident, ref) if ident else FullTextResult("prior_not_found", reason="no identifier")
            texts[i] = result
            kind = classify_reference(ref, result, me)
            usable[i] = kind == "available"
            label = {"available": "available", "not_found": "not_found", "is_submission": "is_submission",
                     "mismatch": f"mismatch (resolves to {str(result.title or '?')[:120]!r})"}.get(
                kind, f"unavailable ({result.reason})")
            entry["references"].append({
                "index": i, "title": ref.get("title"), "resolved_title": result.title, "key": result.key,
                "source": result.source, "version": result.version, "sha256": result.sha256,
                "reason": result.reason, "full_text": label})
        for pair in obj.get("overlap_evidence") or []:
            if not isinstance(pair, dict):
                continue
            try:
                ref_index = int(pair.get("reference") or 1)
            except (TypeError, ValueError):
                ref_index = 1
            prior = texts.get(ref_index)
            item: dict = {"reference": ref_index, "relation": pair.get("relation"),
                          "prior_key": prior.key if prior else None}
            sub = match_passage(str(pair.get("submission_passage") or ""), submission,
                                min_words=ev.min_submission_words, max_words=ev.max_words,
                                claimed_location=pair.get("submission_location"))
            item["submission_passage"] = sub.to_dict()
            section = (sub.location or {}).get("section_id")
            if kinds.get(section) in ("related_work", "references"):  # describes prior work, not the claim
                item["submission_in"] = kinds[section]
            if prior is not None and usable.get(ref_index):
                if prior.sha256 not in sources:
                    sources[prior.sha256] = SourceText(prior.text_md or "", prior.sections)
                pm = match_passage(str(pair.get("prior_passage") or ""), sources[prior.sha256],
                                   min_words=ev.min_prior_words, max_words=ev.max_words,
                                   claimed_location=pair.get("prior_location"))
                item["prior_passage"] = pm.to_dict()
                in_refs = kinds_of(prior).get((pm.location or {}).get("section_id")) == "references"
                item["accepted"] = (accepted(pm, ev) and accepted(sub, ev) and not in_refs
                                    and "submission_in" not in item)
                item["exact"] = pm.status == "verified" and sub.status == "verified"
                if in_refs:
                    item["prior_in"] = "references"
            else:
                item["prior_passage"] = None
                item["accepted"] = False
                item["prior_unavailable"] = prior.reason if prior else "the pair names no listed reference"
            entry["pairs"].append(item)
        entry["status"] = _status(entry, texts)
    return record


def kinds_of(result: FullTextResult) -> dict[str, str]:
    return {s.get("id"): s.get("kind") for s in result.sections or []}


def accepted(m, ev) -> bool:
    """Accepted as evidence: verbatim (token-aligned, no ellipsis); near misses only if configured."""
    if hasattr(m, "accepted_at"):
        return m.accepted_at(ev.accept_approximate_score)
    if {"too_short", "ellipsis", "critical_difference"} & set(m.flags):
        return False
    return m.status == "verified" or (m.status == "approximate" and m.score >= ev.accept_approximate_score)


def _status(entry: dict, texts: dict[int, FullTextResult]) -> str:
    pairs = entry["pairs"]
    refs = entry["references"]
    if any(p.get("accepted") for p in pairs):
        return "quotes_verified" if any(p.get("exact") for p in pairs if p.get("accepted")) else "quotes_approximate"
    if refs and all(r["full_text"] == "is_submission" for r in refs):
        return "prior_is_submission"
    if refs and all(str(r["full_text"]).startswith("mismatch") for r in refs):
        return "reference_mismatch"
    if pairs and any(p.get("prior_passage") is not None for p in pairs):
        return "quotes_not_found"
    if texts and all(t.status == "prior_not_found" for t in texts.values()):
        return "prior_not_found"
    if texts and not any(t.available for t in texts.values()):
        return "fulltext_unavailable"
    return "no_evidence"


STATUS_TEXT = {
    "quotes_verified": "quotes VERIFIED in the full texts",
    "quotes_approximate": "quotes found near-verbatim",
    "quotes_not_found": "quoted passages NOT FOUND in the texts",
    "no_evidence": "NO FULL-TEXT EVIDENCE given",
    "fulltext_unavailable": "full text UNAVAILABLE",
    "prior_not_found": "cited paper NOT FOUND",
    "prior_is_submission": "the cited 'prior paper' IS THE SUBMISSION ITSELF",
    "reference_mismatch": "the identifier RESOLVES TO A DIFFERENT PAPER than the one cited",
}


def evidence_markdown(record: dict) -> str:
    """What judges see: one line per decisive objection, deterministic and terse."""
    items = [(oid, e) for oid, e in record.get("objections", {}).items() if e.get("decisive")]
    if not items:
        return f"Evidence check for {record.get('agent_id')}: no decisive prior-work objections to check."
    lines = [f"Evidence check for {record.get('agent_id')} (orchestrator; quotes matched against the full texts it "
             "retrieved, deterministic):"]
    for oid, e in items:
        refs = "; ".join(f"cited \"{r.get('title') or '?'}\" -> {r.get('key') or 'unresolved'}"
                         + (f" \"{str(r.get('resolved_title'))[:120]}\"" if r.get("resolved_title") else "")
                         + f" — {r['full_text']}"
                         + (f", {r['source']} {r.get('version') or ''}".rstrip() if r.get("source") else "")
                         for r in e.get("references") or [])
        basis = f", refuter's basis: {e['evidence_basis']}" if e.get("evidence_basis") else ""
        lines.append(f"- {oid} [{e.get('category')}, {e.get('severity_estimate')}]: "
                     f"{STATUS_TEXT.get(e['status'], e['status'])}{basis}. {refs}")
        for p in e.get("pairs") or []:
            pp, sp = p.get("prior_passage") or {}, p.get("submission_passage") or {}
            where = (pp.get("location") or {}).get("label") if pp else None
            notes = list(pp.get("flags") or []) if pp else []
            if p.get("submission_in"):
                notes.append(f"the submission passage is in its {p['submission_in'].replace('_', ' ')} section")
            if p.get("prior_in"):
                notes.append("the prior passage is in the prior paper's references")
            lines.append(f"    - pair {p['reference']}: prior passage {pp.get('status', 'unchecked')}"
                         + (f" at {where}" if where else "") + f"; submission passage {sp.get('status', '?')}"
                         + ("; ACCEPTED" if p.get("accepted") else "; not accepted")
                         + (f" ({'; '.join(notes)})" if notes else ""))
            if pp and pp.get("canonical"):  # what the prior paper actually says there
                lines.append(f"      prior text found: \"{' '.join(str(pp['canonical']).split())[:300]}\"")
    return "\n".join(lines)
