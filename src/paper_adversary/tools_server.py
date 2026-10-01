"""MCP server handed to individual agents (via `claude --mcp-config`).

It exposes only (a) literature search over the scholarly APIs, (b) the
submission's own sections, and (c) verbatim full texts of prior work. It reads
only the submission (source/), the run's prior-work snapshots (prior/), its own
scope file and the shared cache, never anything under the run's report folders,
so it cannot become a path around the isolation policy. Every call is logged to
the agent's own log file for the audit trail.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from paper_adversary.util import append_jsonl, read_json, utcnow_iso

SECTION_PART_CHARS = 60_000
PRIOR_PART_CHARS = 30_000
HIT_CHARS = 600


def build_server(run_dir: Path, agent_id: str, log_file: Path, literature: bool, sections: bool,
                 search_opts: dict, attempt: int | None = None, prior_text: str | None = None,
                 prior_scope: list[str] | None = None, fulltext_opts: dict | None = None) -> MCPServer:
    """prior_text: None, "open" (fetch any paper; refuters) or "scoped" (only `prior_scope` keys, offline)."""
    instructions = ("Tools for one reviewer in an adversarial pre-submission review. "
                    + ("Use search_literature, lookup_paper and find_citing_papers to find and verify prior work. "
                       if literature else "")
                    + ("Use read_paper_section to read parts of the submission that are not shown inline. "
                       if sections else "")
                    + ("Use read_prior_paper and find_in_prior_paper to read prior papers verbatim, and check_quote "
                       "before quoting anything." if prior_text else ""))
    mcp = MCPServer("paper-adversary-tools", instructions=instructions)
    search_holder: dict = {}

    def log(tool: str, args: dict, summary: dict) -> None:
        append_jsonl(log_file, {"at": utcnow_iso(), "agent_id": agent_id, "attempt": attempt, "tool": tool,
                                "args": args, **summary})

    async def get_search():
        if "s" not in search_holder:
            from paper_adversary.search import LiteratureSearch

            search_holder["s"] = LiteratureSearch(search_opts["providers"], Path(search_opts["cache_root"]),
                                                  search_opts.get("ttl_days", 14))
        return search_holder["s"]

    def render(result: dict, header: str, limit: int) -> str:
        records = result["records"][:limit]
        status = ", ".join(f"{k}: {v}" for k, v in result["status"].items())
        if not records:
            return f"{header}\nNo results. Provider status: {status}"
        body = "\n".join(r.format(i) for i, r in enumerate(records, 1))
        return f"{header}\n{body}\n\nProvider status: {status}"

    if literature:
        @mcp.tool()
        async def search_literature(query: str, year_from: int | None = None, year_to: int | None = None,
                                    limit: int = 10, sources: list[str] | None = None) -> str:
            """Search scholarly databases (OpenAlex, Semantic Scholar, arXiv, Crossref) for papers.

            Call this to find prior work that overlaps with the submission's claimed contribution. Try
            several phrasings: the paper's own terms, synonyms from adjacent fields, and the underlying
            mathematical problem. Results are merged and de-duplicated; each shows title, year,
            authors, venue, DOI/arXiv ID, citation count, URL and an abstract excerpt.

            Args:
                query: Free-text query, e.g. "classifier-free guidance constraint satisfaction diffusion".
                year_from: Earliest publication year to include.
                year_to: Latest publication year to include.
                limit: Results per provider (1-25).
                sources: Optional subset of providers: openalex, semantic_scholar, arxiv, crossref.
            """
            limit = max(1, min(int(limit), 25))
            s = await get_search()
            result = await s.search(query, year_from, year_to, limit, sources)
            log("search_literature", {"query": query, "year_from": year_from, "year_to": year_to,
                                      "limit": limit, "sources": sources},
                {"results": [r.title for r in result["records"][:limit]], "status": result["status"]})
            return render(result, f"Results for: {query}", limit + 5)

        @mcp.tool()
        async def lookup_paper(identifier: str) -> str:
            """Resolve one paper by DOI, arXiv ID, URL, or exact title and return its canonical record.

            Call this before citing a paper in your report, to confirm that it exists and that the
            title, authors, year and venue you give are correct.

            Args:
                identifier: e.g. "10.1145/3292500.3330701", "arXiv:2106.09685", "https://arxiv.org/abs/2106.09685",
                    or a full title.
            """
            s = await get_search()
            result = await s.lookup(identifier)
            log("lookup_paper", {"identifier": identifier},
                {"results": [r.title for r in result["records"][:5]], "status": result["status"]})
            return render(result, f"Lookup: {identifier}", 5)

        @mcp.tool()
        async def find_citing_papers(identifier: str, limit: int = 15) -> str:
            """List papers that cite a given paper (via OpenAlex and Semantic Scholar), most cited first.

            Call this on the closest prior work you have found: follow-up papers citing it are the
            most likely place for concurrent or overlapping work.

            Args:
                identifier: DOI, arXiv ID, URL or exact title of the cited paper.
                limit: Maximum results per provider (1-40).
            """
            limit = max(1, min(int(limit), 40))
            s = await get_search()
            result = await s.citing(identifier, limit)
            log("find_citing_papers", {"identifier": identifier, "limit": limit},
                {"results": [r.title for r in result["records"][:limit]], "status": result["status"]})
            return render(result, f"Papers citing: {identifier}", limit)

    if sections:
        index = read_json(run_dir / "source" / "sections.json", {}) or {}
        text_path = run_dir / "source" / "extracted_text.md"

        @mcp.tool()
        def read_paper_section(section_id: str, part: int = 1) -> str:
            """Return the full text of one section of the submission, by ID (e.g. "S07").

            Call this for any section shown only as a placeholder in your prompt. Long sections come
            in parts of about 60,000 characters; ask for part 2, 3, ... until told there are no more.

            Args:
                section_id: Section ID from the table of contents.
                part: 1-based part number for long sections.
            """
            text = text_path.read_text(encoding="utf-8")
            sec = next((s for s in index.get("sections", []) if s["id"] == section_id.strip().upper()), None)
            if sec is None:
                ids = ", ".join(s["id"] for s in index.get("sections", []))
                return f"No section '{section_id}'. Valid IDs: {ids}"
            body = text[sec["start"] : sec["end"]]
            parts = max(1, -(-len(body) // SECTION_PART_CHARS))
            part = max(1, int(part))
            if part > parts:
                return f"Section {sec['id']} has only {parts} part(s)."
            chunk = body[(part - 1) * SECTION_PART_CHARS : part * SECTION_PART_CHARS]
            log("read_paper_section", {"section_id": sec["id"], "part": part}, {"chars": len(chunk)})
            more = f"\n\n[part {part} of {parts}; call again with part={part + 1}]" if part < parts else ""
            return f"Section {sec['id']} — {sec['title']} (part {part} of {parts})\n\n{chunk}{more}"

    if prior_text:
        _add_prior_tools(mcp, run_dir, log, prior_text, set(prior_scope or []), search_opts, fulltext_opts or {})
    return mcp


def _add_prior_tools(mcp: MCPServer, run_dir: Path, log, mode: str, scope: set[str], search_opts: dict,
                     fulltext_opts: dict) -> None:
    from paper_adversary.config import FullTextConfig
    from paper_adversary.passages import SourceText, describe_location, match_passage
    from paper_adversary.search.fulltext import FullTextStore, describe, read_snapshot, snapshot_key

    prior_dir = run_dir / "prior"
    cfg = FullTextConfig.model_validate({**fulltext_opts, "offline": fulltext_opts.get("offline") or mode == "scoped"})
    holder: dict = {}

    async def fetch(identifier: str):
        if mode == "scoped":
            key = snapshot_key(prior_dir, identifier)
            if key is None or key not in scope:
                return None, (f"'{identifier}' is not one of the papers you were given. Your papers: "
                              + (", ".join(sorted(scope)) or "none"))
            return read_snapshot(prior_dir, key), None
        if "store" not in holder:
            holder["store"] = FullTextStore(Path(search_opts["cache_root"]), cfg, search_opts.get("providers"),
                                            prior_dir=prior_dir)
        return await holder["store"].get(identifier), None

    def source_for(result) -> SourceText:
        cache = holder.setdefault("sources", {})
        if result.sha256 not in cache:
            cache[result.sha256] = SourceText(result.text_md or "", result.sections)
        return cache[result.sha256]

    def summary(result) -> dict:
        return {"key": result.key, "sha256": result.sha256, "source": result.source, "status": result.status,
                "reason": result.reason}

    @mcp.tool()
    async def read_prior_paper(identifier: str, section: str | None = None, page: int | None = None,
                               part: int = 1) -> str:
        """Read a prior paper's full text verbatim (open-access copies only), by arXiv ID, DOI or exact title.

        Without section or page you get the paper's header, its table of contents and the first part of the
        text. Then ask for a section (e.g. section="S04") or a page (page=5). Quote only from text this tool
        returns, character for character; use check_quote before quoting.

        Args:
            identifier: arXiv ID ("2404.13040"), DOI, or the paper's exact title.
            section: Section ID from the table of contents, e.g. "S04".
            page: A PDF page number.
            part: 1-based part number for long texts (about 30,000 characters each).
        """
        result, refusal = await fetch(identifier)
        if refusal:
            return refusal
        if result is None or not result.available:
            log("read_prior_paper", {"identifier": identifier}, summary(result) if result else {"status": "none"})
            return describe(result) if result else f"FULL TEXT UNAVAILABLE: {identifier} is not in this run's papers."
        text = result.text_md or ""
        label = "whole text"
        if section:
            sec = next((s for s in result.sections if s["id"] == section.strip().upper()), None)
            if sec is None:
                return f"No section '{section}'. Valid IDs: " + ", ".join(s["id"] for s in result.sections)
            text, label = text[sec["start"] : sec["end"]], f"section {sec['id']} {sec['title']}"
        elif page is not None:
            m = re.search(rf"<!-- page {int(page)} -->", text)
            if not m:
                return f"No page {page} in this text (page markers exist only for PDF sources)."
            nxt = re.search(r"<!-- page \d+ -->", text[m.end():])
            text, label = text[m.start() : m.end() + (nxt.start() if nxt else len(text))], f"page {page}"
        parts = max(1, -(-len(text) // PRIOR_PART_CHARS))
        part = max(1, int(part))
        if part > parts:
            return f"The {label} has only {parts} part(s)."
        chunk = text[(part - 1) * PRIOR_PART_CHARS : part * PRIOR_PART_CHARS]
        head = describe(result) + "\n"
        if not section and page is None and part == 1:
            toc = "\n".join(f"{s['id']} {'  ' * max(0, s.get('level', 1) - 1)}{s['title']}"
                             + (f" (p. {s['page_start']})" if s.get("page_start") else "") for s in result.sections)
            head += f"Table of contents:\n{toc or '(no sections detected)'}\n"
        more = f"\n\n[part {part} of {parts}; call again with part={part + 1}]" if part < parts else ""
        log("read_prior_paper", {"identifier": identifier, "section": section, "page": page, "part": part},
            {**summary(result), "chars": len(chunk)})
        return f"{head}\n[{label}, part {part} of {parts}]\n\n{chunk}{more}"

    @mcp.tool()
    async def find_in_prior_paper(identifier: str, query: str, max_hits: int = 8) -> str:
        """Find the passages of a prior paper that bear on a query, returned verbatim with their locations.

        Args:
            identifier: arXiv ID, DOI or exact title of a paper (as for read_prior_paper).
            query: Words or a phrase to look for, e.g. "monotonically increasing guidance schedule".
            max_hits: Maximum passages returned (1-20).
        """
        result, refusal = await fetch(identifier)
        if refusal:
            return refusal
        if result is None or not result.available:
            return describe(result) if result else f"FULL TEXT UNAVAILABLE: {identifier}"
        text = result.text_md or ""
        words = [w for w in re.findall(r"[^\W_]+", query.lower()) if len(w) > 2]
        phrase = " ".join(words)
        paras = [(m.start(), m.end()) for m in re.finditer(r"(?:(?!\n\s*\n).)+", text, re.S)]
        scored = []
        for start, end in paras:
            body = re.sub(r"\s+", " ", text[start:end].lower())
            hits = sum(1 for w in set(words) if w in body)
            if hits:
                scored.append((hits + (3 if phrase and phrase in body else 0), start, end))
        scored.sort(key=lambda x: (-x[0], x[1]))
        src = source_for(result)
        out = []
        for _, start, end in scored[: max(1, min(int(max_hits), 20))]:
            loc = describe_location(src, start, end)
            out.append(f"[{loc['label']}]\n{text[start:min(end, start + HIT_CHARS)].strip()}")
        log("find_in_prior_paper", {"identifier": identifier, "query": query}, {**summary(result), "hits": len(out)})
        return (describe(result) + f"\n\nPassages for: {query}\n\n" + "\n\n".join(out)) if out else \
            f"No passage of {result.title or identifier} mentions those words."

    @mcp.tool()
    async def check_quote(passage: str, identifier: str | None = None) -> str:
        """Check that a passage you want to quote appears verbatim in a prior paper (identifier given) or in the
        submission (no identifier). Returns the match status and the exact text found, so you can fix the quote.

        Args:
            passage: The text you intend to quote.
            identifier: The prior paper's arXiv ID, DOI or title; omit to check against the submission.
        """
        if identifier:
            result, refusal = await fetch(identifier)
            if refusal:
                return refusal
            if result is None or not result.available:
                return describe(result) if result else f"FULL TEXT UNAVAILABLE: {identifier}"
            src = source_for(result)
            target = result.title or identifier
        else:
            if "submission" not in holder:
                index = read_json(run_dir / "source" / "sections.json", {}) or {}
                holder["submission"] = SourceText((run_dir / "source" / "extracted_text.md").read_text(encoding="utf-8"),
                                                  index.get("sections"))
            src, target = holder["submission"], "the submission"
        m = match_passage(passage, src)
        log("check_quote", {"identifier": identifier, "chars": len(passage)},
            {"status": m.status, "score": round(m.score, 3)})
        lines = [f"{m.status.upper()} in {target}" + (f" at {m.location['label']}" if m.location else "")
                 + f" (score {m.score:.2f}, coverage {m.coverage:.2f})"]
        if m.flags:
            lines.append("Flags: " + ", ".join(m.flags) + (f" — {m.note}" if m.note else ""))
        if m.canonical and m.status != "verified":
            lines.append(f"Closest text in the source: {m.canonical[:800]}")
        if not m.accepted:
            lines.append("This would not count as a verbatim quote. Copy the text exactly from the source.")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paper-adversary tools-server")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--literature", action="store_true")
    parser.add_argument("--sections", action="store_true")
    parser.add_argument("--search-opts", default="{}", help="JSON: providers, cache_root, ttl_days")
    parser.add_argument("--attempt", type=int, default=None)
    parser.add_argument("--prior-text", choices=("open", "scoped"), default=None)
    parser.add_argument("--prior-keys", default=None, help="JSON list of the prior-paper keys (scoped mode)")
    parser.add_argument("--fulltext-opts", default="{}", help="JSON: search.fulltext settings")
    args = parser.parse_args(argv)
    scope = json.loads(args.prior_keys) if args.prior_keys else None
    server = build_server(Path(args.run_dir), args.agent_id, Path(args.log_file), args.literature, args.sections,
                          json.loads(args.search_opts), args.attempt, args.prior_text, scope,
                          json.loads(args.fulltext_opts))
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # per-request INFO lines are noise here
    try:
        server.run("stdio")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
