"""MCP server handed to individual agents (via `claude --mcp-config`).

It exposes only (a) literature search over the scholarly APIs and (b) the
submission's own sections. It never reads anything under the run's report
folders, so it cannot become a path around the isolation policy. Every call is
logged to the agent's own log file for the audit trail.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from paper_adversary.util import append_jsonl, read_json, utcnow_iso

SECTION_PART_CHARS = 60_000


def build_server(run_dir: Path, agent_id: str, log_file: Path, literature: bool, sections: bool,
                 search_opts: dict, attempt: int | None = None) -> MCPServer:
    instructions = ("Tools for one reviewer in an adversarial pre-submission review. "
                    + ("Use search_literature, lookup_paper and find_citing_papers to find and verify prior work. "
                       if literature else "")
                    + ("Use read_paper_section to read parts of the submission that are not shown inline."
                       if sections else ""))
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

    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="paper-adversary tools-server")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--literature", action="store_true")
    parser.add_argument("--sections", action="store_true")
    parser.add_argument("--search-opts", default="{}", help="JSON: providers, cache_root, ttl_days")
    parser.add_argument("--attempt", type=int, default=None)
    args = parser.parse_args(argv)
    server = build_server(Path(args.run_dir), args.agent_id, Path(args.log_file), args.literature, args.sections,
                          json.loads(args.search_opts), args.attempt)
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # per-request INFO lines are noise here
    try:
        server.run("stdio")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
