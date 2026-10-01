"""Blind verification: does a cited prior paper really anticipate a passage of the submission?

Requests come from decisive novelty objections, from the completeness critic's items, or from the user. Each
names a prior paper and, where possible, the submission passage at stake. Planning a batch resolves every prior
paper to its full text, locates the claim in the submission (only canonical spans of the paper's own text go
forward; a caller's prose never does), groups requests by prior paper into tasks, and gives each task to a fresh
verifier agent V<n>. A verifier sees the submission, its passages and the prior paper, never the argument that
led to the request, so its verdict is an independent check rather than a second opinion on the same case.
Identical tasks (same prior text, same passages, same prompt and model) reuse an earlier verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from paper_adversary.config import AgentSpec, PipelineConfig
from paper_adversary.passages import SourceText, match_passage
from paper_adversary.reports import is_decisive, refuter_objections
from paper_adversary.search.fulltext import FullTextResult, FullTextStore
from paper_adversary.util import atomic_write_json, read_json, utcnow_iso

VERDICTS = ("anticipates_fully", "anticipates_partially", "does_not_anticipate", "cannot_tell")
MAX_PASSAGES = 8
MAX_REFS_PER_OBJECTION = 3
CONTEXT_CHARS = 1500  # submission text used when a claim cannot be located more precisely


@dataclass(frozen=True)
class VerificationRequest:
    request_id: str  # "N1-O2:r1", "C1-I3:r1", "U2"
    origin: str  # refuter | critic | user | gate
    origin_ids: tuple[str, ...]  # routing only; never shown to a verifier
    prior: dict  # {"identifier", "title", "doi", "arxiv_id", "year"}
    claim_quote: str | None = None  # verbatim submission text, if known
    claim_location: str | None = None  # "Sec. 3.1", "p. 4", "Abstract"
    note: str | None = None  # the caller's prose; audit trail only, never in a verifier prompt
    origin_hash: str | None = None  # of the objection or item it came from: a rerun with the same ID is new work


def origin_hash(obj: dict) -> str:
    """Fingerprint of an objection or critic item. IDs are positional and reused when an agent is rerun, so a
    verdict counts for an objection only if it was requested for this exact content."""
    keep = {k: obj.get(k) for k in ("id", "title", "category", "type", "claim_targeted", "location", "references",
                                     "candidate_references", "overlap_evidence", "severity_estimate")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def current(record: dict, oid: str, hashes: dict[str, str] | None) -> bool:
    """Whether a stored request result still belongs to the objection now carrying this ID."""
    if not hashes or oid not in hashes or record.get("origin_hash") is None:
        return True
    return record.get("origin_hash") == hashes[oid]


def _ref_identifier(ref: dict) -> str | None:
    return (str(ref.get("arxiv_id") or "").strip() or str(ref.get("doi") or "").strip()
            or str(ref.get("title") or "").strip() or None)


def requests_from_objections(store, state: dict) -> list[VerificationRequest]:
    """One request per decisive novelty objection and load-bearing reference (at most three per objection)."""
    out: list[VerificationRequest] = []
    for aid, a in sorted(state["agents"].items()):
        if a["role"] != "novelty" or a["status"] != "complete":
            continue
        data = (read_json(store.sidecar_path(aid, "novelty", ".json"), {}) or {}).get("data")
        for obj in refuter_objections(data, aid):
            if not is_decisive(obj):
                continue
            refs = [r for r in obj.get("references") or [] if isinstance(r, dict) and _ref_identifier(r)]
            pairs = [p for p in obj.get("overlap_evidence") or [] if isinstance(p, dict)]
            cited = []
            for p in pairs:
                try:
                    cited.append(int(p.get("reference") or 1))
                except (TypeError, ValueError):
                    continue
            order = list(dict.fromkeys(cited + list(range(1, len(refs) + 1))))
            for k, index in enumerate([i for i in order if 1 <= i <= len(refs)][:MAX_REFS_PER_OBJECTION], start=1):
                ref = refs[index - 1]
                pair = next((p for p in pairs if str(p.get("reference") or 1) == str(index)), None) or {}
                quote = pair.get("submission_passage") or _quoted(obj.get("claim_targeted"))
                out.append(VerificationRequest(
                    request_id=f"{obj['id']}:r{k}", origin="refuter", origin_ids=(obj["id"],),
                    prior={"identifier": _ref_identifier(ref), "title": ref.get("title"), "doi": ref.get("doi"),
                           "arxiv_id": ref.get("arxiv_id"), "year": ref.get("year")},
                    claim_quote=quote, claim_location=pair.get("submission_location") or obj.get("claim_targeted"),
                    note=obj.get("argument"), origin_hash=origin_hash(obj)))
    return out


def pending_requests(store, state: dict) -> list[VerificationRequest]:
    """Requests the current decisive objections need that no batch has answered (new, or rerun objections)."""
    results = read_json(store.role_dir("verifier") / "results.json", {}) or {}
    done = results.get("requests") or {}
    return [req for req in requests_from_objections(store, state)
            if (done.get(req.request_id) or {}).get("origin_hash") != req.origin_hash]


def _quoted(text) -> str | None:
    m = re.search(r"[\"“]([^\"”]{30,600})[\"”]", str(text or "")[:2000])  # bounded: claim text is untrusted
    return m.group(1) if m else None


NOT_CLAIMS = ("related_work", "references")  # these sections describe prior work, not the submission's claim


def _locate(claim_quote: str | None, claim_location: str | None, submission: SourceText, text: str,
            sections: list[dict]) -> tuple[str | None, str | None]:
    """A canonical span of the submission for the claim: its verbatim quote, else the named section's opening.
    Spans in the related-work or reference sections are never used: there the submission describes the prior
    work itself, so "does the prior paper establish this?" would be trivially yes."""
    kinds = {s["id"]: s.get("kind") for s in sections}
    loc = str(claim_location or "")[:300]
    # the quote, else the location's own words when it restates the claim ("We are the first ... (Idea, claim 1)")
    candidates = [claim_quote] if claim_quote else []
    stated = re.sub(r"\s*\([^()]{0,80}\)\s*$", "", loc).strip()
    if len(stated.split()) >= 6:
        candidates.append(stated)
    for quote in candidates:
        m = match_passage(quote, submission, min_words=4, max_words=200)
        if m.accepted and m.canonical and kinds.get((m.location or {}).get("section_id")) not in NOT_CLAIMS:
            return m.canonical, (m.location or {}).get("label")
    sec_num = re.search(r"(?:§|sec(?:tion)?\.?)\s*([\dA-Z](?:\.\d+)*)", loc, re.I)
    claims = [s for s in sections if s.get("kind") not in NOT_CLAIMS]
    for s in claims:
        title = s.get("title", "")
        if (sec_num and re.match(rf"{re.escape(sec_num.group(1))}\b", title)) or \
                ("abstract" in loc.lower() and s.get("kind") == "abstract"):
            return _section_body(text, s), f"{s['id']} {title}"
    named = [s for s in claims if _named_in(s.get("title", ""), loc)]  # "(Idea, claim 1)" -> section "Idea"
    if named:
        s = max(named, key=lambda s: len(_bare_title(s.get("title", ""))))
        return _section_body(text, s), f"{s['id']} {s.get('title', '')}"
    return None, None


def _bare_title(title: str) -> str:
    return re.sub(r"^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)+\.?|[A-Z]\.)\s+", "", str(title)).strip().lower()


def _named_in(title: str, location: str) -> bool:
    """Whether a location names this section by its title (whole words; numbering ignored)."""
    bare = _bare_title(title)
    return len(bare) >= 4 and re.search(rf"(?<!\w){re.escape(bare)}(?!\w)", location.lower()) is not None


def _section_body(text: str, s: dict) -> str:
    body = re.sub(r"<!--[^>]{0,200}-->", "", text[s["start"]: s["end"]])
    body = re.sub(r"^\s*#+[ \t][^\n]*\n", "", body).strip()  # the section's heading is not the claim
    return body[:CONTEXT_CHARS]


def _task_hash(prior_sha: str, passages: list[dict], prompt: str, model: str, effort: str) -> str:
    blob = json.dumps({"prior": prior_sha, "passages": [p["text"] for p in passages], "prompt": prompt,
                       "model": model, "effort": effort}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


async def plan_batch(store, cfg: PipelineConfig, batch_id: str, requests: list[VerificationRequest],
                     fetch: FullTextStore, state: dict, rnd: int = 0) -> tuple[dict, list[AgentSpec]]:
    """Resolve, locate and group the requests; return the batch record and the verifier agents to create."""
    from paper_adversary.evidence import SubmissionFingerprint, classify_reference

    vc = cfg.verifier
    index = read_json(store.source_dir / "sections.json", {}) or {}
    text = (store.source_dir / "extracted_text.md").read_text(encoding="utf-8")
    submission = SourceText(text, index.get("sections"))
    me = SubmissionFingerprint(store.load_metadata().get("title"), text)
    vdir = store.role_dir("verifier")
    results = read_json(vdir / "results.json", {"requests": {}, "tasks": {}}) or {"requests": {}, "tasks": {}}
    ver = state.get("verification") or {}
    batch = {"id": batch_id, "status": "planned", "planned_at": utcnow_iso(), "requests": [], "tasks": []}
    groups: dict[str, dict] = {}
    for req in requests:
        rec = {**asdict(req), "origin_ids": list(req.origin_ids), "batch": batch_id}
        try:
            prior = await fetch.get(req.prior.get("identifier") or "", req.prior)
        except Exception as exc:  # one broken reference must not sink the batch
            prior = FullTextResult("fulltext_unavailable", reason="fetch_failed", detail=f"{type(exc).__name__}")
        rec.update(prior_key=prior.key, prior_title=prior.title or req.prior.get("title"), prior_source=prior.source,
                   prior_version=prior.version, prior_sha=prior.sha256)
        kind = classify_reference(req.prior, prior, me)
        if prior.status == "prior_not_found":
            rec.update(status="prior_not_found", reason=prior.reason)
        elif kind == "is_submission":
            rec.update(status="prior_is_submission", reason="the cited prior paper is the submission itself")
        elif kind == "mismatch":
            rec.update(status="reference_mismatch",
                       reason=f"the identifier resolves to {str(prior.title)[:120]!r}, not the paper cited")
        elif not prior.available:
            rec.update(status="fulltext_unavailable", reason=prior.reason, detail=prior.detail)
        else:
            span, where = _locate(req.claim_quote, req.claim_location, submission, text, index.get("sections") or [])
            if span is None:
                rec.update(status="claim_not_located",
                           reason="the claim could not be found in the submission; give its exact words")
            else:
                group = groups.setdefault(prior.key, {"prior": prior, "passages": [], "requests": []})
                pid = next((p["pid"] for p in group["passages"] if p["text"] == span), None)
                if pid is None and len(group["passages"]) < MAX_PASSAGES:
                    pid = f"P{len(group['passages']) + 1}"
                    group["passages"].append({"pid": pid, "text": span, "location": where})
                if pid is None:
                    rec.update(status="not_run", reason="too many passages for one paper")
                else:
                    rec.update(status="pending", passage_id=pid)
                    group["requests"].append(req.request_id)
        results["requests"][req.request_id] = rec
        batch["requests"].append(req.request_id)

    used = sum(1 for a in state["agents"].values() if a["role"] == "verifier")
    next_index = int(ver.get("next_index") or 1)
    specs: list[AgentSpec] = []
    ranked = sorted(groups.values(), key=lambda g: (-_weight(g, results), g["prior"].key))
    for group in ranked:
        prior = group["prior"]
        thash = _task_hash(prior.sha256, group["passages"], vc.prompt, vc.model, vc.effort)
        reuse = next((tid for tid, tk in results["tasks"].items() if tk.get("hash") == thash
                      and tk.get("status") == "complete"), None)
        if reuse:  # an identical check already has a verdict; collect() maps it onto these requests
            task = results["tasks"][reuse]
            task["requests"] = list(dict.fromkeys((task.get("requests") or []) + group["requests"]))
            for rid in group["requests"]:
                results["requests"][rid].update(task_id=reuse, agent_id=task.get("agent_id"), status="pending",
                                                reused=True)
            batch["tasks"].append(reuse)
            continue
        if len(specs) >= vc.max_agents_per_batch or used + len(specs) >= vc.max_agents_per_run:
            for rid in group["requests"]:
                results["requests"][rid].update(status="not_run", reason="verifier cap reached")
            continue
        task_id = f"T{len(results['tasks']) + 1}"
        aid = f"V{next_index}"
        next_index += 1
        atomic_write_json(vdir / "tasks" / f"{task_id}.json", {  # what the verifier may see: no IDs, no arguments
            "task_id": task_id, "prior_key": prior.key, "prior_sha": prior.sha256, "prior_title": prior.title,
            "prior_source": prior.source, "prior_version": prior.version, "passages": group["passages"]})
        results["tasks"][task_id] = {"hash": thash, "agent_id": aid, "prior_key": prior.key, "status": "pending",
                                     "requests": group["requests"], "batch": batch_id}
        for rid in group["requests"]:
            results["requests"][rid].update(task_id=task_id, agent_id=aid)
        batch["tasks"].append(task_id)
        specs.append(AgentSpec(agent_id=aid, role="verifier", index=next_index - 1, model_alias=vc.model,
                               model_id=vc.model, effort=vc.effort, prompt_name=vc.prompt, tools=list(vc.tools),
                               paper_format="text", timeout_s=vc.timeout_minutes * 60, max_turns=vc.max_turns,
                               task_id=task_id, round=rnd))
    atomic_write_json(vdir / "results.json", results)
    batch["next_index"] = next_index
    return batch, specs


def _weight(group: dict, results: dict) -> int:
    """Order tasks by how much rides on them: more requests (more refuters citing the paper) first."""
    return len(group["requests"])


def collect(store, cfg: PipelineConfig, batch_id: str, state: dict) -> dict:
    """Fold finished verifier reports back into per-request results; returns the updated results."""
    vdir = store.role_dir("verifier")
    results = read_json(vdir / "results.json", {"requests": {}, "tasks": {}}) or {"requests": {}, "tasks": {}}
    for task_id, task in results["tasks"].items():
        aid = task.get("agent_id")
        entry = state["agents"].get(aid) or {}
        if task.get("status") == "complete" or not aid:
            verdicts = task.get("verdicts") or {}
        elif entry.get("status") == "complete":
            verdicts = _verdicts(store, cfg, aid, task_id)
            task.update(status="complete", verdicts=verdicts)
        else:
            task["status"] = entry.get("status", "pending")
            verdicts = {}
        for rid in task.get("requests") or []:
            req = results["requests"].get(rid)
            if req is None or req.get("status") not in ("pending", "verifier_failed", "verified", "disputed",
                                                        "cannot_tell"):
                continue
            v = verdicts.get(req.get("passage_id"))
            if v is None:
                if task.get("status") == "complete":  # the verifier finished but gave no verdict for this passage
                    req.update(status="cannot_tell", verdict="cannot_tell", agent_id=aid,
                               note="the verifier gave no verdict for this passage")
                    continue
                failed = entry.get("status") in ("failed", "quarantined", "superseded")
                req.update(status="verifier_failed" if failed else "pending", agent_id=aid)
                continue
            status = {"anticipates_fully": "verified", "anticipates_partially": "verified",
                      "does_not_anticipate": "disputed"}.get(v["verdict"], "cannot_tell")
            req.update(status=status, verdict=v["verdict"], agent_id=aid, confidence=v.get("confidence"),
                       overlap=v.get("overlap"), differences=v.get("differences"), rationale=v.get("rationale"),
                       note=v.get("note"))
    atomic_write_json(vdir / "results.json", results)
    write_base_view(store, results)
    return results


def write_base_view(store, results: dict) -> None:
    """The base review's own checks (not the follow-up rounds'), so base readers' inputs only change when theirs do."""
    base = {"requests": {rid: r for rid, r in (results.get("requests") or {}).items()
                         if not str(r.get("batch") or "").startswith("critic")}}
    atomic_write_json(store.role_dir("verifier") / "base_results.json", base)


def _verdicts(store, cfg: PipelineConfig, aid: str, task_id: str) -> dict[str, dict]:
    """A verifier's verdicts, each anticipation claim checked: its overlap quotes must be found in both texts."""
    side = (read_json(store.sidecar_path(aid, "verifier", ".json"), {}) or {}).get("data") or {}
    task = read_json(store.role_dir("verifier") / "tasks" / f"{task_id}.json", {}) or {}
    prior_text = store.prior_dir / f"{task.get('prior_sha')}.md"
    if not prior_text.is_file():
        return {}
    prior = SourceText(prior_text.read_text(encoding="utf-8"),
                       read_json(store.prior_dir / f"{task.get('prior_sha')}.sections.json", []) or [])
    from paper_adversary.evidence import accepted

    ev = cfg.evidence
    examined = {x["pid"]: SourceText(x["text"]) for x in task.get("passages") or []}
    prior_kinds = {s.get("id"): s.get("kind") for s in read_json(
        store.prior_dir / f"{task.get('prior_sha')}.sections.json", []) or []}
    out: dict[str, dict] = {}
    passages = side.get("passages")
    for p in passages if isinstance(passages, list) else []:
        if not isinstance(p, dict) or str(p.get("verdict") or "").strip().lower() not in VERDICTS:
            continue
        item = {k: p.get(k) for k in ("confidence", "rationale")}
        item["verdict"] = str(p["verdict"]).strip().lower()
        own = examined.get(str(p.get("passage_id") or ""))
        checked = []
        overlap = p.get("overlap") if isinstance(p.get("overlap"), list) else []
        for pair in overlap[:4]:
            if not isinstance(pair, dict):
                continue
            pm = match_passage(str(pair.get("prior_passage") or ""), prior, min_words=8, max_words=ev.max_words)
            # the submission side must quote the passage under examination, not any sentence of the paper
            sm = match_passage(str(pair.get("submission_passage") or ""), own, min_words=4,
                               max_words=ev.max_words) if own is not None else None
            in_refs = prior_kinds.get((pm.location or {}).get("section_id")) == "references"
            ok = accepted(pm, ev) and sm is not None and accepted(sm, ev) and not in_refs
            checked.append({**pair, "prior_check": pm.status, "submission_check": sm.status if sm else "no_passage",
                            "accepted": ok, "prior_in_references": in_refs,
                            "prior_location_found": (pm.location or {}).get("label"),
                            "prior_text_found": " ".join(str(pm.canonical or "").split())[:300] or None})
        item["overlap"] = checked
        item["differences"] = p.get("differences") if isinstance(p.get("differences"), list) else []
        if item["verdict"] in ("anticipates_fully", "anticipates_partially") and not any(c["accepted"] for c in checked):
            item["note"] = (f"the verifier said {item['verdict']}, but none of its overlap quotes were found verbatim "
                            "in the passage under examination and the prior paper's body")
            item["verdict"] = "cannot_tell"
        out[p.get("passage_id")] = item
    return out


def visible_batches(pos: tuple[int, int]):
    """Which verification batches a reader at (round, step) may see: base batches always; a follow-up round's
    critic batch only from later steps of that round or later rounds."""
    rnd, step = pos

    def ok(batch_id: str | None) -> bool:
        if not batch_id or not batch_id.startswith("critic"):
            return True
        m = re.match(r"critic(\d+)", batch_id)
        k = int(m.group(1)) if m else 0
        return rnd >= 1 and (k < rnd or (k == rnd and step > 0))
    return ok


def verification_markdown(results: dict, origin_ids: set[str] | None = None, batches=None,
                          hashes: dict[str, str] | None = None) -> str:
    """The independent checks, per prior paper, for judges, synthesis and the critic. With `hashes` (objection or
    item ID -> origin_hash), results requested for an earlier version of an objection are left out."""
    reqs = [(rid, r) for rid, r in sorted(results.get("requests", {}).items())
            if (origin_ids is None or set(r.get("origin_ids") or []) & origin_ids)
            and (batches is None or batches(r.get("batch")))
            and all(current(r, oid, hashes) for oid in r.get("origin_ids") or [])]
    if not reqs:
        return ""
    lines = ["Independent checks by blind verifiers (each saw only the submission's passages and the prior paper's "
             "full text, never the objection; anticipation claims without verbatim-verified overlap quotes count "
             "as cannot_tell):"]
    for rid, r in reqs:
        head = f"- {', '.join(r.get('origin_ids') or [])} vs {r.get('prior_title') or r.get('prior_key') or '?'}"
        if r.get("status") in ("verified", "disputed", "cannot_tell"):
            lines.append(f"{head} ({r.get('prior_source')} {r.get('prior_version') or ''}): {r.get('agent_id')} "
                         f"says {r.get('verdict')} ({r.get('confidence') or '?'} confidence).".replace("  ", " "))
            for pair in (r.get("overlap") or [])[:2]:
                if pair.get("accepted"):
                    lines.append(f"    overlap: prior \"{str(pair.get('prior_passage'))[:300]}\" "
                                 f"({pair.get('prior_location_found') or pair.get('prior_location') or '?'})")
            for diff in (r.get("differences") or [])[:2]:
                if isinstance(diff, dict):
                    lines.append(f"    difference: {str(diff.get('explanation') or '')[:300]}")
            if r.get("note"):
                lines.append(f"    note: {r['note']}")
        else:
            why = r.get("reason") or r.get("status")
            lines.append(f"{head}: not independently checked — {r.get('status')}" + (f" ({why})" if why else ""))
    return "\n".join(lines)
