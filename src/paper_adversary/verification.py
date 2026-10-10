"""Blind verification: does a cited prior paper really anticipate a passage of the submission?

Requests come from decisive novelty objections (and, on a gate re-check, from prior-work objections the judges
escalated), from the completeness critic's items, or from the user. Each names a prior paper and, where possible,
the submission passage at stake: the refuter's quoted submission passages (up to two per reference), a critic
item's submission_quote, else the claims-ledger claims it cites, else its location. Planning a batch resolves
every prior paper to its full text, locates the claim in the submission (only canonical spans of the paper's own
text go forward; a caller's prose never does), groups requests by prior paper into tasks, and gives each task to
a fresh verifier agent V<n>. A verifier sees the whole submission, its passages and the prior paper, never the
argument that led to the request, so its verdict is an independent check rather than a second opinion.

With `routing: cross_model` the verifier runs on the other model than the objection's author (an Opus
refuter's objection goes to a Fable verifier and vice versa; a critic item, to the other model than the
critic's). A prior paper cited by authors on both models gets one verifier, routed by the request that ranks
it. Papers are chosen greedily: first a paper that gives an objection without an independent check its
first one, then a threat to a central claim, severity, doubt (the author's lower confidence and thinner
evidence, used only as a ranking, never as a probability), and the number of unchecked objections it serves;
remaining ties alternate between authors, then follow the request ID. Each kind of batch has its own budget: the
refuters' batches share `max_papers_per_run` / `max_agents_per_run`, each follow-up round and each gate or user
batch has its own (see BUDGETS). Requests beyond a budget are recorded as not run with the reason that decided
it, and stay labelled unverified; so are references beyond the third of an objection. A passage already checked
against the same prior text is never checked again, whichever model checked it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

from paper_adversary.config import AgentSpec, PipelineConfig
from paper_adversary.passages import SourceText, match_passage
from paper_adversary.reports import is_decisive, is_prior_work_objection, refuter_objections
from paper_adversary.search.fulltext import FullTextResult, FullTextStore
from paper_adversary.util import atomic_write_json, read_json, utcnow_iso

VERDICTS = ("anticipates_fully", "anticipates_partially", "does_not_anticipate", "cannot_tell")
MAX_PASSAGES = 8
MAX_REFS_PER_OBJECTION = 3
MAX_PASSAGES_PER_REFERENCE = 2  # distinct submission passages of one objection checked against one paper
CONTEXT_CHARS = 1500  # submission text used when a claim cannot be located more precisely

# Verification budgets by kind of batch: (papers key, agents key, what the budget covers). The refuters' batches
# (refuters, refuters2, ...) share the per-run caps; each follow-up round (critic<r>) and each gate or user batch
# gets its own, so an earlier stage can never use up a later one's.
BUDGETS = {"base": ("max_papers_per_run", "max_agents_per_run", "per run for the refuters' objections"),
           "followup": ("max_papers_per_followup_round", "max_agents_per_followup_round", "per follow-up round"),
           "user": ("max_papers_per_user_batch", "max_agents_per_user_batch", "per gate or user batch")}


def budget_pool(batch_id: str | None) -> str:
    """Which budget a batch draws on: base (the refuters' batches), followup (critic<r>) or user (gate*, user*)."""
    b = str(batch_id or "")
    return "base" if b.startswith("refuters") else "followup" if b.startswith("critic") else "user"


@dataclass(frozen=True)
class VerificationRequest:
    request_id: str  # "N1-O2:r1", "N1-O2:r1.p2" (a second passage for that reference), "C1-I3:r1", "USER1-2"
    origin: str  # refuter | critic | user | gate
    origin_ids: tuple[str, ...]  # routing only; never shown to a verifier
    prior: dict  # {"identifier", "title", "doi", "arxiv_id", "year"}
    claim_quote: str | None = None  # verbatim submission text, if known
    claim_location: str | None = None  # "Sec. 3.1", "p. 4", "Abstract"
    note: str | None = None  # the caller's prose; audit trail only, never in a verifier prompt
    origin_hash: str | None = None  # of the objection or item it came from: a rerun with the same ID is new work
    author: str | None = None  # the agent that raised it (N1, C1): routes the verifier to the other model
    severity: str | None = None  # the author's estimate; for ranking only
    confidence: str | None = None  # the author's stated confidence; an ordinal tie-break, never a probability
    basis: str | None = None  # the author's evidence basis (full_text, abstract_only, none)
    quote_label: str = "claim_quote"  # where claim_quote comes from (overlap_evidence, submission_quote, ...)
    alt_quotes: tuple = ()  # ((label, quote), ...) tried in order when claim_quote is missing or not found
    secondary: bool = False  # a further passage for the same (objection, reference): located verbatim only
    limited: bool = False  # a reference beyond MAX_REFS_PER_OBJECTION: recorded as not run, never planned


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


def requests_from_objections(store, state: dict, include_ids: set[str] | None = None) -> list[VerificationRequest]:
    """Requests for the decisive novelty objections and, given include_ids (a gate re-check), for these
    prior-work objections whatever severity their refuter gave them (e.g. ones the judges escalated). One request
    per load-bearing reference, with a second one (`:r1.p2`) when the refuter quoted another, distinct submission
    passage against the same paper. Only the first three references are checked; later ones get `limited`
    requests, recorded as not run, so they are never silently dropped."""
    out: list[VerificationRequest] = []
    for aid, a in sorted(state["agents"].items()):
        if a["role"] != "novelty" or a["status"] != "complete":
            continue
        data = (read_json(store.sidecar_path(aid, "novelty", ".json"), {}) or {}).get("data")
        for obj in refuter_objections(data, aid):
            if not is_decisive(obj) and not (include_ids and obj["id"] in include_ids
                                             and is_prior_work_objection(obj)):
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
            for k, index in enumerate([i for i in order if 1 <= i <= len(refs)], start=1):
                ref = refs[index - 1]
                mine = [p for p in pairs if str(p.get("reference") or 1) == str(index)]
                pair = mine[0] if mine else {}
                quote = pair.get("submission_passage") or _quoted(obj.get("claim_targeted"))
                common = dict(origin="refuter", origin_ids=(obj["id"],), note=obj.get("argument"),
                              prior={"identifier": _ref_identifier(ref), "title": ref.get("title"),
                                     "doi": ref.get("doi"), "arxiv_id": ref.get("arxiv_id"), "year": ref.get("year")},
                              origin_hash=origin_hash(obj), author=aid, severity=obj.get("severity_estimate"),
                              confidence=obj.get("confidence"), basis=obj.get("evidence_basis"))
                limited = k > MAX_REFS_PER_OBJECTION
                out.append(VerificationRequest(
                    request_id=f"{obj['id']}:r{k}", claim_quote=quote, limited=limited,
                    quote_label="overlap_evidence" if pair.get("submission_passage") else "claim_targeted",
                    claim_location=pair.get("submission_location") or obj.get("claim_targeted"), **common))
                if limited:
                    continue
                seen, n = {_squash(quote)}, 1
                for p in mine[1:]:
                    text = str(p.get("submission_passage") or "").strip()
                    if n >= MAX_PASSAGES_PER_REFERENCE:
                        break
                    if not text or _squash(text) in seen:
                        continue
                    seen.add(_squash(text))
                    n += 1
                    out.append(VerificationRequest(
                        request_id=f"{obj['id']}:r{k}.p{n}", claim_quote=text, quote_label="overlap_evidence",
                        claim_location=p.get("submission_location"), secondary=True, **common))
    return out


def _squash(text) -> str:
    return " ".join(str(text or "").lower().split())


def pending_requests(store, state: dict) -> list[VerificationRequest]:
    """Requests the current decisive objections need that no batch has answered (new, or rerun objections). A
    further passage (`:r1.p2`) or a reference past the per-objection limit is pending only with its objection's
    first request: a batch records all of an objection's requests together, so an objection whose first request
    is recorded for its current content while these are not was planned before they existed (runs created before
    2026-10-02). Such a run does not grow a new batch, and stale judges, for them; a gate re-check covers them."""
    results = read_json(store.role_dir("verifier") / "results.json", {}) or {}
    done = results.get("requests") or {}

    def recorded(rid: str, req: VerificationRequest) -> bool:
        return (done.get(rid) or {}).get("origin_hash") == req.origin_hash

    return [req for req in requests_from_objections(store, state) if not recorded(req.request_id, req)
            and not ((req.secondary or req.limited) and recorded(f"{req.origin_ids[0]}:r1", req))]


def gate_requests(store, state: dict, ids: set[str]) -> list[VerificationRequest]:
    """The requests a gate re-check makes for these objections (flagged or escalated ones, whatever severity their
    refuter gave them): every reference without a verifier's answer for the objection's current content, and each
    reference past the per-objection limit that is not recorded yet, which plan_batch records as not run without
    fetching it. Empty when there is nothing to check but such references."""
    done = (read_json(store.role_dir("verifier") / "results.json", {}) or {}).get("requests") or {}
    out = []
    for req in requests_from_objections(store, state, include_ids=ids):
        if not set(req.origin_ids) & ids:
            continue
        rec = done.get(req.request_id) or {}
        current = rec.get("origin_hash") == req.origin_hash
        if (not current) if req.limited else not (current and rec.get("status") in ("verified", "disputed",
                                                                                     "cannot_tell")):
            out.append(req)
    return out if any(not r.limited for r in out) else []


def _quoted(text) -> str | None:
    m = re.search(r"[\"“]([^\"”]{30,600})[\"”]", str(text or "")[:2000])  # bounded: claim text is untrusted
    return m.group(1) if m else None


NOT_CLAIMS = ("related_work", "references")  # these sections describe prior work, not the submission's claim


def _locate(claim_quote: str | None, claim_location: str | None, submission: SourceText, text: str,
            sections: list[dict]) -> tuple[str | None, str | None]:
    """A canonical span of the submission for the claim (see locate_claim)."""
    span, where, _, _ = locate_claim(claim_quote, claim_location, submission, text, sections)
    return span, where


def locate_claim(claim_quote: str | None, claim_location: str | None, submission: SourceText, text: str,
                 sections: list[dict], alternates=(), quote_label: str = "claim_quote",
                 strict: bool = False) -> tuple[str | None, str | None, str | None, list[str]]:
    """A canonical span of the submission for the claim: its verbatim quote, else the first alternate quote that
    is found (a critic item's cited ledger claims), else the location's own words, else the named section's
    opening; with strict, the quote only. Returns (span, where, which source located it, the quotes tried that
    were not found). Spans in the related-work or reference sections are never used: there the submission
    describes the prior work itself, so "does the prior paper establish this?" would be trivially yes."""
    kinds = {s["id"]: s.get("kind") for s in sections}
    loc = "" if strict else str(claim_location or "")[:300]
    candidates = [(quote_label, claim_quote)] if claim_quote else []
    candidates += [] if strict else [(str(label), str(q)) for label, q in alternates or () if q]
    # the location's own words, when it restates the claim ("We are the first ... (Idea, claim 1)")
    stated = re.sub(r"\s*\([^()]{0,80}\)\s*$", "", loc).strip()
    if len(stated.split()) >= 6:
        candidates.append(("location", stated))
    missed: list[str] = []
    for label, quote in candidates:
        m = match_passage(quote, submission, min_words=4, max_words=200)
        if m.accepted and m.canonical and kinds.get((m.location or {}).get("section_id")) not in NOT_CLAIMS:
            return m.canonical, (m.location or {}).get("label"), label, missed
        if label != "location":
            missed.append(label)
    sec_num = re.search(r"(?:§|sec(?:tion)?\.?)\s*([\dA-Z](?:\.\d+)*)", loc, re.I)
    claims = [s for s in sections if s.get("kind") not in NOT_CLAIMS]
    for s in claims:
        title = s.get("title", "")
        if (sec_num and re.match(rf"{re.escape(sec_num.group(1))}\b", title)) or \
                ("abstract" in loc.lower() and s.get("kind") == "abstract"):
            return _section_body(text, s), f"{s['id']} {title}", "location (section opening)", missed
    named = [s for s in claims if _named_in(s.get("title", ""), loc)]  # "(Idea, claim 1)" -> section "Idea"
    if named:
        s = max(named, key=lambda s: len(_bare_title(s.get("title", ""))))
        return _section_body(text, s), f"{s['id']} {s.get('title', '')}", "location (section opening)", missed
    return None, None, None, missed


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
    claims = ledger_claims(store)
    abstract_ids = {s["id"] for s in index.get("sections") or [] if s.get("kind") == "abstract"}
    me = SubmissionFingerprint(store.load_metadata().get("title"), text)
    vdir = store.role_dir("verifier")
    results = read_json(vdir / "results.json", {"requests": {}, "tasks": {}}) or {"requests": {}, "tasks": {}}
    ver = state.get("verification") or {}
    batch = {"id": batch_id, "status": "planned", "planned_at": utcnow_iso(), "requests": [], "tasks": []}
    groups: dict[str, dict] = {}
    # every request's own passage first, so a further passage can never crowd one out of MAX_PASSAGES
    for req in [r for r in requests if not r.secondary] + [r for r in requests if r.secondary]:
        rec = {**asdict(req), "origin_ids": list(req.origin_ids), "alt_quotes": [list(x) for x in req.alt_quotes],
               "batch": batch_id}
        results["requests"][req.request_id] = rec
        batch["requests"].append(req.request_id)
        if req.limited:  # never fetched or planned, but recorded (with the paper it names), so no view drops it
            rec.update(status="not_run", prior_title=req.prior.get("title") or req.prior.get("identifier"),
                       reason=f"per-objection reference limit: only the first {MAX_REFS_PER_OBJECTION} references "
                              "of an objection are verified")
            continue
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
            span, where, how, missed = locate_claim(req.claim_quote, req.claim_location, submission, text,
                                                    index.get("sections") or [], req.alt_quotes, req.quote_label,
                                                    strict=req.secondary)
            rec.update(located_by=how, not_located=missed)
            if span is None:
                rec.update(status="claim_not_located",
                           reason="the passage was not found verbatim in the submission's own text" if req.secondary
                           else "the claim could not be found in the submission; give its exact words")
            else:
                group = groups.setdefault(prior.key, {"prior": prior, "passages": [], "requests": []})
                pid = next((p["pid"] for p in group["passages"] if p["text"] == span), None)
                if pid is None and len(group["passages"]) < MAX_PASSAGES:
                    pid = f"P{len(group['passages']) + 1}"
                    group["passages"].append({"pid": pid, "text": span, "location": where})
                if pid is None:
                    rec.update(status="not_run", reason="too many passages for one paper")
                else:
                    rec.update(status="pending", passage_id=pid, passage_location=where,
                               central=is_central(span, where, claims, abstract_ids))
                    group["requests"].append(req.request_id)

    pool = budget_pool(batch_id)
    papers_key, agents_key, scope = BUDGETS[pool]
    max_papers, max_agents = getattr(vc, papers_key), getattr(vc, agents_key)
    batches = ver.get("batches") or {}
    # the refuters' batches share one budget; a follow-up round or a gate/user batch is planned once, from zero
    papers = {tk.get("prior_key") for tk in results["tasks"].values()
              if pool == "base" and budget_pool(tk.get("batch")) == "base"}
    used = sum(len(b.get("agents") or []) for k, b in batches.items()
               if pool == "base" and budget_pool(k) == "base" and k != batch_id)
    earlier = bool(papers) or used > 0
    now = {oid: r.origin_hash for r in requests for oid in r.origin_ids if r.origin_hash}
    covered = _covered(results, now)  # objection or item ID -> a task that already checks its current content
    next_index = int(ver.get("next_index") or 1)
    specs: list[AgentSpec] = []
    fresh = []
    for group in groups.values():
        reqs = [results["requests"][rid] for rid in group["requests"]]
        alias = _alias(vc, state, min(reqs, key=priority))
        chash = _content_hash(group["prior"].sha256, group["passages"], vc.prompt)
        thash = _task_hash(group["prior"].sha256, group["passages"], vc.prompt, alias, vc.effort)
        reuse = next((tid for tid, tk in results["tasks"].items() if tk.get("status") == "complete"
                      and (tk.get("content_hash") == chash or tk.get("hash") == thash)), None)
        if not reuse:
            fresh.append(group)
            continue
        task = results["tasks"][reuse]  # an identical check already has a verdict; collect() maps it onto these
        task["requests"] = list(dict.fromkeys((task.get("requests") or []) + group["requests"]))
        for rid in group["requests"]:
            results["requests"][rid].update(task_id=reuse, agent_id=task.get("agent_id"), status="pending",
                                            reused=True)
            for oid in results["requests"][rid].get("origin_ids") or []:
                covered.setdefault(oid, reuse)
        batch["tasks"].append(reuse)

    turns: dict[str, int] = {}  # tasks per author so far in this batch: ties alternate between authors
    last = None  # the last group admitted, with every remaining group's key at that pick
    while fresh:  # greedy: re-rank after every pick, since coverage changes
        keys = {id(g): _group_key(g, results, covered, turns) for g in fresh}
        group = min(fresh, key=lambda g: keys[id(g)][0])
        fresh.remove(group)
        top = keys[id(group)][1]
        prior = group["prior"]
        by_earlier = last is None and earlier and pool == "base"  # nothing admitted yet: earlier batches used it
        cap = None
        if prior.key not in papers and len(papers) >= max_papers:
            cap = (f"verification cap: the run's budget of {max_papers} prior paper(s) for the refuters' objections "
                   f"(verifier.{papers_key}) was used by an earlier batch" if by_earlier
                   else f"verification cap: at most {max_papers} prior paper(s) are verified {scope} "
                        f"(verifier.{papers_key})")
        elif len(specs) >= vc.max_agents_per_batch:
            cap = (f"verifier cap: at most {vc.max_agents_per_batch} verifier(s) per batch "
                   "(verifier.max_agents_per_batch)")
        elif used + len(specs) >= max_agents:
            cap = (f"verifier cap: the run's budget of {max_agents} verifier(s) for the refuters' objections "
                   f"(verifier.{agents_key}) was used by an earlier batch" if by_earlier
                   else f"verifier cap: at most {max_agents} verifier(s) {scope} (verifier.{agents_key})")
        if cap:
            why = _why_not(group, keys, last, covered, results)
            for rid in group["requests"]:
                results["requests"][rid].update(status="not_run", reason=f"{cap}{'; ' + why if why else ''}; "
                                                                         "not independently checked")
            continue
        authors = {results["requests"][rid].get("author") for rid in group["requests"]}
        author_model = {a: (state["agents"].get(a) or {}).get("model_alias") for a in authors if a}
        alias = vc.model_for(author_model.get(top.get("author")))
        chash = _content_hash(prior.sha256, group["passages"], vc.prompt)
        thash = _task_hash(prior.sha256, group["passages"], vc.prompt, alias, vc.effort)
        papers.add(prior.key)
        task_id = f"T{len(results['tasks']) + 1}"
        aid = f"V{next_index}"
        next_index += 1
        atomic_write_json(vdir / "tasks" / f"{task_id}.json", {  # what the verifier may see: no IDs, no arguments
            "task_id": task_id, "prior_key": prior.key, "prior_sha": prior.sha256, "prior_title": prior.title,
            "prior_source": prior.source, "prior_version": prior.version, "passages": group["passages"]})
        results["tasks"][task_id] = {"hash": thash, "content_hash": chash, "agent_id": aid, "prior_key": prior.key,
                                     "status": "pending", "requests": group["requests"], "batch": batch_id,
                                     "verifier_model": alias, "author_models": sorted(set(author_model.values()) - {None}),
                                     "routed_by": top["request_id"],
                                     "same_model_authors": sorted(a for a, m in author_model.items() if m == alias)}
        for rid in group["requests"]:
            results["requests"][rid].update(task_id=task_id, agent_id=aid)
            for oid in results["requests"][rid].get("origin_ids") or []:
                covered.setdefault(oid, task_id)
        turns[str(top.get("author"))] = turns.get(str(top.get("author")), 0) + 1
        last = (group, top, keys)
        batch["tasks"].append(task_id)
        specs.append(AgentSpec(agent_id=aid, role="verifier", index=next_index - 1, model_alias=alias,
                               model_id=alias, effort=vc.effort, prompt_name=vc.prompt, tools=list(vc.tools),
                               paper_format="text", timeout_s=vc.timeout_minutes * 60, max_turns=vc.max_turns,
                               task_id=task_id, round=rnd))
    atomic_write_json(vdir / "results.json", results)
    batch["next_index"] = next_index
    return batch, specs


def _alias(vc, state: dict, top: dict) -> str:
    """The verifier's model for a group routed by `top` (cross-model: the other model than its author's)."""
    return vc.model_for((state["agents"].get(top.get("author")) or {}).get("model_alias"))


def _covered(results: dict, now: dict[str, str]) -> dict[str, str]:
    """Objection or item ID -> the task already checking it (planned or done) from an earlier batch, counted only
    for the content the ID carries now (`now`: ID -> origin_hash of this batch's requests)."""
    out: dict[str, str] = {}
    for tid, tk in results.get("tasks", {}).items():
        if tk.get("status") in ("failed", "quarantined", "superseded"):
            continue
        for rid in tk.get("requests") or []:
            rec = results["requests"].get(rid) or {}
            if rec.get("task_id") != tid:  # re-planned since then: no longer this task's
                continue
            for oid in rec.get("origin_ids") or []:
                if now.get(oid, rec.get("origin_hash")) == rec.get("origin_hash"):
                    out.setdefault(oid, tid)
    return out


def _group_key(group: dict, results: dict, covered: dict[str, str], turns: dict[str, int]) -> tuple[tuple, dict]:
    """The rank of a prior paper's group now (smaller first) and the request that ranks and routes it: a paper that
    gives some objection its first independent check comes first; within that, a central claim, severity, doubt,
    the number of unchecked objections it serves; ties alternate between authors, then follow the request ID."""
    reqs = [results["requests"][rid] for rid in group["requests"]]
    uncovered = {oid for r in reqs for oid in r.get("origin_ids") or [] if oid not in covered}
    serving = [r for r in reqs if set(r.get("origin_ids") or []) & uncovered] or reqs
    top = min(serving, key=priority)
    central, sev, doubt, rid = priority(top)
    return (0 if uncovered else 1, central, sev, doubt, -len(uncovered), turns.get(str(top.get("author")), 0),
            rid), top


_WHY = {0: "ranked below {rid}, which gave an objection without an independent check its first one",
        1: "ranked below {rid}, a threat to a central claim (this one is not)",
        2: "ranked below {rid}, rated more severe ({theirs} against {mine})",
        3: "ranked below {rid}, whose author was less certain or had a thinner evidence basis",
        4: "ranked below {rid}, which serves more objections without an independent check"}


def _why_not(group: dict, keys: dict, last, covered: dict[str, str], results: dict) -> str | None:
    """What decided that this group lost its slot, stated from the keys that actually differed."""
    oids = sorted({oid for rid in group["requests"] for oid in results["requests"][rid].get("origin_ids") or []})
    if oids and all(o in covered for o in oids):
        return (f"every objection it serves ({', '.join(oids)}) already has an independent check "
                f"({', '.join(sorted({covered[o] for o in oids}))})")
    if last is None:
        return None
    winner, w_top, at_pick = last
    mine, theirs = at_pick[id(group)][0], at_pick[id(winner)][0]
    i = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), len(mine))
    if i in _WHY:
        return _WHY[i].format(rid=w_top["request_id"], theirs=w_top.get("severity") or "?",
                              mine=at_pick[id(group)][1].get("severity") or "?")
    return (f"tied with {w_top['request_id']} on coverage, centrality, severity and doubt; lost the tie-break "
            "(authors take turns, then request order)")


_SEVERITY_ORDER = {"fatal": 3, "major": 2, "major_fixable": 2, "major_but_fixable": 2, "minor": 1}
_DOUBT = {"low": 2, "medium": 1, "high": 0}
_THIN = {"none": 2, "abstract_only": 1}


def priority(rec: dict) -> tuple:
    """Sort key of a request record (smaller first): a central claim, then severity, then doubt (the author's
    lower confidence, thinner evidence basis), then the request ID. Confidence is an ordinal, not a probability.
    plan_batch ranks prior papers by coverage first and alternates authors before the ID (_group_key)."""
    sev = _SEVERITY_ORDER.get(str(rec.get("severity") or "").strip().lower().replace(" ", "_"), 0)
    doubt = _DOUBT.get(str(rec.get("confidence") or "").strip().lower(), 1) + \
        _THIN.get(str(rec.get("basis") or "").strip().lower(), 0)
    return (0 if rec.get("central") else 1, -sev, -doubt, rec["request_id"])


LEDGER_PREFIX = "L"  # claims-ledger IDs L1, L2, ...: no agent ID starts with L, so a claim is never read as an agent


def ledger_entries(data: dict | None) -> list[tuple[str, dict]]:
    """The intake's claims in ledger order as (ledger ID, claim). The IDs are L1, L2, ... by position, whatever the
    intake model wrote ("C1" in intake_v1's example, the critic's own ID), so they never collide with an agent ID."""
    claims = [c for c in (data or {}).get("claims") or [] if isinstance(c, dict)]
    return [(f"{LEDGER_PREFIX}{i}", c) for i, c in enumerate(claims, start=1)]


def _intake_data(store) -> dict:
    return (read_json(store.sidecar_path("INTAKE", "intake", ".json"), {}) or {}).get("data") or {}


def ledger_claims(store) -> list[str]:
    """The submission's claims as the intake pass listed them (used only to rank verification requests)."""
    return [str(c["claim"]) for _, c in ledger_entries(_intake_data(store)) if c.get("claim")]


def ledger_claim_map(store) -> dict[str, str]:
    """Claims-ledger ID ("L24") -> the claim as the intake pass quoted or closely paraphrased it. The ID the intake
    model wrote ("C24") maps to the same claim, so runs whose ledger was written before L IDs (it showed the
    model's IDs) and critics that copy a C-form example still find their claims; an L ID always wins."""
    data = _intake_data(store)
    out = {lid: str(c["claim"]) for lid, c in ledger_entries(data) if c.get("claim")}
    for i, c in enumerate(data.get("claims") or [], start=1):
        if isinstance(c, dict) and c.get("claim"):
            out.setdefault(str(c.get("id") or f"C{i}").strip().upper(), str(c["claim"]))
    return out


def _trigrams(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i: i + 3]) for i in range(len(words) - 2)}


def is_central(span: str, where: str | None, claims: list[str], abstract_ids: set[str]) -> bool:
    """Whether a located submission span is about a central claim: it lies in the abstract, or it restates a
    claim of the intake ledger (at least half of the claim's word trigrams)."""
    if (where or "").split(" ")[0] in abstract_ids:
        return True
    grams = _trigrams(span)
    for claim in claims:
        cg = _trigrams(claim)
        if cg and len(cg & grams) / len(cg) >= 0.5:
            return True
    return False


def _content_hash(prior_sha: str, passages: list[dict], prompt: str) -> str:
    """What a verification checks, whichever model and effort check it."""
    blob = json.dumps({"prior": prior_sha, "passages": [p["text"] for p in passages], "prompt": prompt},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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
    write_base_view(store, results, judges=budget_pool(batch_id) == "base")
    return results


def write_base_view(store, results: dict, judges: bool = False) -> None:
    """The base review's own checks (not the follow-up rounds'), so base readers' inputs only change when theirs
    do. The judges' view is refreshed only with a refuters' batch (or when there is none yet): gate and user
    batches run after the judges, so they recompute the evidence gate instead of making the judges stale."""
    vdir = store.role_dir("verifier")
    base = {"requests": {rid: r for rid, r in (results.get("requests") or {}).items()
                         if not str(r.get("batch") or "").startswith("critic")}}
    atomic_write_json(vdir / "base_results.json", base)
    if judges or not (vdir / "judge_results.json").is_file():
        atomic_write_json(vdir / "judge_results.json", base)


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
    lines = ["Independent checks by blind verifiers (each saw the whole submission, the submission passages at "
             "stake, as quoted or located for the objection or item that asked for the check, and the prior paper's "
             "full text; never the objection's or item's argument, category, severity or author. Anticipation "
             "claims without verbatim-verified overlap quotes count as cannot_tell):"]
    for rid, r in reqs:
        prior = r.get("prior") or {}
        named = r.get("prior_title") or r.get("prior_key") or prior.get("title") or prior.get("identifier") or "?"
        head = f"- {', '.join(r.get('origin_ids') or [])} vs {named}"
        if r.get("passage_location"):
            head += f" [submission passage at {r['passage_location']}]"
        if r.get("status") in ("verified", "disputed", "cannot_tell"):
            lines.append(f"{head} ({r.get('prior_source')} {r.get('prior_version') or ''}): {r.get('agent_id')} "
                         f"says {r.get('verdict')} ({r.get('confidence') or '?'} confidence).".replace("  ", " "))
            shown = [p for p in r.get("overlap") or [] if isinstance(p, dict) and p.get("accepted")]
            for pair in shown[:3]:  # verified evidence first, so a rejected pair never hides an accepted one
                anchor = " ".join(str(pair.get("submission_passage") or "").split())
                lines.append(f"    overlap: prior \"{str(pair.get('prior_passage'))[:300]}\" "
                             f"({pair.get('prior_location_found') or pair.get('prior_location') or '?'}), on the "
                             f"submission's \"{anchor[:160]}{'...' if len(anchor) > 160 else ''}\"")
            if len(shown) > 3:
                lines.append(f"    (+{len(shown) - 3} more verbatim-verified overlap pairs)")
            for diff in (r.get("differences") or [])[:2]:
                if isinstance(diff, dict):
                    lines.append(f"    difference: {str(diff.get('explanation') or '')[:300]}")
            if r.get("note"):
                lines.append(f"    note: {r['note']}")
        else:
            why = r.get("reason") or r.get("status")
            lines.append(f"{head}: not independently checked — {r.get('status')}" + (f" ({why})" if why else ""))
    return "\n".join(lines)
