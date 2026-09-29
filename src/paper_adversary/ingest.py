"""Paper ingestion: PDF, Markdown, plain text and LaTeX -> sectioned Markdown.

The output keeps section boundaries (as Markdown headings) and PDF page markers
so agents can cite locations, and records a section index used by the
long-document strategy in budget.py.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

SECTION_KINDS = (
    "title", "abstract", "introduction", "related_work", "method", "theory", "experiments", "results",
    "limitations", "conclusion", "acknowledgements", "references", "appendix", "other",
)

_KIND_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("abstract", ("abstract",)),
    ("references", ("references", "bibliography", "works cited")),
    ("acknowledgements", ("acknowledg",)),
    ("appendix", ("appendix", "supplementary material", "supplemental material")),
    ("limitations", ("limitation", "broader impact", "impact statement", "ethic", "societal", "future work",
                     "reproducibility statement", "checklist")),
    ("conclusion", ("conclusion", "concluding", "summary and")),
    ("introduction", ("introduction", "motivation")),
    ("related_work", ("related work", "prior work", "background", "literature", "related literature")),
    ("theory", ("theor", "proof", "lemma", "guarantee", "convergence", "bound", "preliminar", "analysis of the")),
    ("experiments", ("experiment", "evaluation", "empirical", "benchmark", "setup", "implementation detail",
                     "dataset", "training detail")),
    ("results", ("result", "ablation", "discussion", "findings", "analysis")),
    ("method", ("method", "approach", "framework", "algorithm", "proposed", "architecture", "model",
                "formulation", "problem", "our ")),
]

_KNOWN_UNNUMBERED = {
    "abstract", "references", "bibliography", "acknowledgments", "acknowledgements", "acknowledgment",
    "impact statement", "broader impact", "broader impacts", "limitations", "appendix", "appendices",
    "conclusion", "conclusions", "introduction", "related work", "checklist", "ethics statement",
    "reproducibility statement", "supplementary material",
}


class IngestError(ValueError):
    pass


@dataclass
class Section:
    id: str
    title: str
    level: int
    kind: str
    start: int
    end: int
    page_start: int | None = None
    page_end: int | None = None

    @property
    def chars(self) -> int:
        return self.end - self.start


@dataclass
class IngestResult:
    source_format: str
    title: str | None
    abstract: str | None
    text_md: str
    sections: list[Section]
    references: list[dict]
    page_count: int | None
    submission_type: str
    warnings: list[str] = field(default_factory=list)
    source_path: str | None = None

    def section_text(self, section: Section) -> str:
        return self.text_md[section.start : section.end]

    def to_index(self) -> dict:
        return {
            "source_format": self.source_format,
            "title": self.title,
            "page_count": self.page_count,
            "submission_type": self.submission_type,
            "chars": len(self.text_md),
            "words": len(self.text_md.split()),
            "sections": [asdict(s) | {"chars": s.chars} for s in self.sections],
            "reference_count": len(self.references),
            "warnings": self.warnings,
        }


def classify_section(title: str, parent_kind: str | None = None) -> str:
    t = re.sub(r"^[\dA-Z](\.\d+)*\.?\s+", "", title.strip()).lower()
    for kind, needles in _KIND_RULES:
        if any(n in t for n in needles):
            return kind
    return parent_kind or "other"


# ---------------------------------------------------------------- entry point


def ingest(paper_path: str | Path | None = None, paper_text: str | None = None,
           submission_type: str = "auto") -> IngestResult:
    if paper_path and paper_text:
        raise IngestError("give either paper_path or paper_text, not both")
    if not paper_path and not (paper_text and paper_text.strip()):
        raise IngestError("no paper given: pass paper_path or paper_text")
    if paper_path:
        path = Path(paper_path).expanduser().resolve()
        if not path.is_file():
            raise IngestError(f"file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            result = ingest_pdf(path)
        elif suffix in {".md", ".markdown"}:
            result = ingest_markdown(path.read_text(encoding="utf-8", errors="replace"), "markdown")
        elif suffix == ".tex":
            result = ingest_latex(path)
        elif suffix in {".txt", ".text", ""}:
            result = ingest_plain_text(path.read_text(encoding="utf-8", errors="replace"), "text")
        else:
            raise IngestError(f"unsupported file type '{suffix}' (supported: .pdf .md .markdown .tex .txt)")
        result.source_path = str(path)
    else:
        text = paper_text or ""
        looks_md = bool(re.search(r"^#{1,6}\s+\S", text, re.M))
        result = ingest_markdown(text, "inline") if looks_md else ingest_plain_text(text, "inline")
    result.submission_type = _decide_type(result, submission_type)
    if len(result.text_md.strip()) < 200:
        result.warnings.append("very little text was extracted; check the source (scanned PDF?)")
    return result


def _decide_type(result: IngestResult, requested: str) -> str:
    if requested in {"paper", "idea"}:
        return requested
    words = len(result.text_md.split())
    has_refs = any(s.kind == "references" for s in result.sections)
    top_sections = [s for s in result.sections if s.level == 1 and s.kind not in {"title", "abstract"}]
    if words < 2500 and not has_refs and len(top_sections) < 4:
        return "idea"
    return "paper"


# ---------------------------------------------------------------- Markdown


_ATX = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.M)


def ingest_markdown(text: str, fmt: str) -> IngestResult:
    text = text.replace("\r\n", "\n")
    headings = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in _ATX.finditer(text)]
    title = None
    h1 = [h for h in headings if h[1] == 1]
    if len(h1) == 1 and headings and headings[0][1] == 1 and len(headings) > 1:
        title = h1[0][2]
    body = [(pos, lvl, t) for pos, lvl, t in headings if not (title and lvl == 1 and t == title)]
    base = min((lvl for _, lvl, _ in body), default=1)  # the shallowest body heading becomes level 1
    sections = _sections_from_headings(text, [(pos, lvl - base + 1, t) for pos, lvl, t in body])
    if title:
        sections.insert(0, Section("S00", title, 0, "title", 0, sections[0].start if sections else len(text)))
    references = _split_references(text, sections)
    abstract = _abstract_from(text, sections)
    return IngestResult(fmt, title, abstract, text, _renumber(sections), references, None, "paper")


def _sections_from_headings(text: str, heads: list[tuple[int, int, str]]) -> list[Section]:
    sections: list[Section] = []
    parents: dict[int, str] = {}
    for i, (pos, level, title) in enumerate(heads):
        end = len(text)
        for pos2, level2, _ in heads[i + 1 :]:
            if level2 <= level:
                end = pos2
                break
        parent_kind = None
        for lvl in sorted(parents, reverse=True):
            if lvl < level:
                parent_kind = parents[lvl]
                break
        own = classify_section(title)
        # Subsections follow their parent unless their own title marks a distinct part of the paper.
        kind = own if parent_kind is None or own in _STANDALONE_KINDS else parent_kind
        parents = {k: v for k, v in parents.items() if k < level}
        parents[level] = kind
        sections.append(Section("", title, level, kind, pos, end))
    _infer_method_sections(sections)
    _mark_appendix(sections)
    return sections


_STANDALONE_KINDS = {"abstract", "references", "acknowledgements", "limitations", "appendix"}


def _infer_method_sections(sections: list[Section]) -> None:
    """Custom-titled sections between the introduction and the experiments are the method."""
    tops = [s for s in sections if s.level == 1]
    for i, s in enumerate(tops):
        if s.kind != "other":
            continue
        before = {t.kind for t in tops[:i]}
        after = {t.kind for t in tops[i + 1 :]}
        if before & {"introduction", "related_work", "abstract"} and after & {"experiments", "results", "conclusion"}:
            s.kind = "method"
    parent: Section | None = None
    for s in sections:
        if s.level <= 1:
            parent = s
        elif parent is not None and s.kind == "other":
            s.kind = parent.kind


def _mark_appendix(sections: list[Section]) -> None:
    """Sections after the bibliography or an explicit Appendix heading are supplementary material."""
    in_appendix = False
    for s in sections:
        if s.kind in {"references", "appendix"}:
            in_appendix = True
            continue
        if in_appendix and s.kind not in {"acknowledgements", "limitations"}:
            s.kind = "appendix"


def _renumber(sections: list[Section]) -> list[Section]:
    for i, s in enumerate(sections):
        s.id = f"S{i:02d}"
    return sections


def _abstract_from(text: str, sections: list[Section]) -> str | None:
    for s in sections:
        if s.kind == "abstract":
            body = text[s.start : s.end].split("\n", 1)[-1].strip()
            return _squash(body)[:4000] or None
    return None


def _squash(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- plain text


_NUMBERED_HEAD = re.compile(r"^(?P<num>(\d+|[A-Z])(\.\d+)*)\.?\s+(?P<title>[A-Z][^\n]{1,100})$")


def ingest_plain_text(text: str, fmt: str) -> IngestResult:
    text = text.replace("\r\n", "\n")
    out_lines: list[str] = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.strip()
        prev_blank = i == 0 or not lines[i - 1].strip()
        level = _plain_heading_level(stripped) if prev_blank else None
        if level:
            out_lines.append("#" * (level + 1) + " " + stripped)
        else:
            out_lines.append(line)
    md = "\n".join(out_lines)
    result = ingest_markdown(md, fmt)
    if result.title is None:
        first = next((ln.strip() for ln in lines if ln.strip()), "")
        if 3 <= len(first.split()) <= 25 and not first.endswith("."):
            result.title = first
    return result


def _plain_heading_level(line: str) -> int | None:
    if not line or len(line) > 110 or line.endswith((".", ",", ";", ":")):
        return None
    if line.lower().rstrip(":") in _KNOWN_UNNUMBERED:
        return 1
    m = _NUMBERED_HEAD.match(line)
    if m and len(line.split()) <= 12:
        return 1 + m.group("num").count(".")
    if line.isupper() and 1 <= len(line.split()) <= 8 and re.search(r"[A-Z]{3}", line):
        return 1
    return None


# ---------------------------------------------------------------- references


def _split_references(text: str, sections: list[Section]) -> list[dict]:
    refs_sec = next((s for s in sections if s.kind == "references"), None)
    if not refs_sec:
        return []
    body = text[refs_sec.start : refs_sec.end].split("\n", 1)[-1]
    body = re.sub(r"<!--.*?-->", "\n", body)
    if re.search(r"^\s*[-*]\s+", body, re.M) and not re.search(r"^\s*\[\d+\]", body, re.M):
        parts = re.split(r"\n\s*[-*]\s+", "\n" + body)
    else:
        parts = re.split(r"\n\s*\n", body)
    # Numbered styles: split further wherever the next number in sequence appears ("[12] ..." or "12. ...").
    for pattern in (r"(?:^|(?<=\s))\[(\d{1,3})\]\s", r"(?:^|(?<=\s))(\d{1,3})\.\s+(?=[A-Z])"):
        joined = "\n\n".join(parts)
        chain, expect = [], 1
        for m in re.finditer(pattern, joined):
            if int(m.group(1)) == expect:
                chain.append(m.start())
                expect += 1
        if len(chain) >= 3 and len(chain) > len([p for p in parts if p.strip()]):
            parts = [joined[a:b] for a, b in zip(chain, chain[1:] + [len(joined)])]
            break
    entries: list[str] = []
    for part in parts:
        entry = _squash(part)
        if len(entry) < 12:
            continue
        if entries and (entry[:1].islower() or entry.startswith(("(", "pp.", "In:", "vol"))):
            entries[-1] = f"{entries[-1]} {entry}"  # continuation split off by a page or column break
        else:
            entries.append(entry)
    return [{"n": i, "text": e} for i, e in enumerate(entries, 1) if len(e) >= 20]


# ---------------------------------------------------------------- LaTeX


def _strip_tex_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def _expand_inputs(path: Path, seen: set[Path], depth: int = 0) -> str:
    if depth > 20 or path in seen:
        return ""
    seen.add(path)
    text = _strip_tex_comments(path.read_text(encoding="utf-8", errors="replace"))

    def repl(m: re.Match) -> str:
        name = m.group(2).strip()
        candidate = (path.parent / name)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".tex")
        if candidate.is_file():
            return _expand_inputs(candidate.resolve(), seen, depth + 1)
        return m.group(0)

    return re.sub(r"\\(input|include)\s*\{([^}]+)\}", repl, text)


def _brace_arg(text: str, start: int) -> tuple[str, int]:
    """Return the content of the {...} group starting at text[start] == '{'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return text[start + 1 :], len(text)


def _tex_command_arg(text: str, command: str) -> str | None:
    m = re.search(rf"\\{command}\*?\s*(\[[^\]]*\])?\s*\{{", text)
    if not m:
        return None
    arg, _ = _brace_arg(text, m.end() - 1)
    return arg


def _tex_inline_clean(s: str) -> str:
    s = re.sub(r"\\\\", " ", s)
    s = re.sub(r"\\(textbf|textit|emph|texttt|textsc|mathrm)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\\thanks\{[^{}]*\}", "", s)
    return _squash(s)


def ingest_latex(path: Path) -> IngestResult:
    full = _expand_inputs(path.resolve(), set())
    warnings: list[str] = []
    title = _tex_command_arg(full, "title")
    title = _tex_inline_clean(title) if title else None
    body = full
    m = re.search(r"\\begin\{document\}", full)
    if m:
        body = full[m.end() :]
    body = re.sub(r"\\end\{document\}.*", "", body, flags=re.S)
    abstract = None
    am = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if am:
        abstract = am.group(1).strip()
        body = body[: am.start()] + "\n## Abstract\n\n" + abstract + "\n\n" + body[am.end() :]
    body = re.sub(r"\\maketitle", "", body)

    out: list[str] = []
    pos = 0
    for hm in re.finditer(r"\\(section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\s*\{", body):
        if hm.start() < pos:
            continue
        out.append(body[pos : hm.start()])
        arg, end = _brace_arg(body, hm.end() - 1)
        level = {"section": 2, "subsection": 3, "subsubsection": 4}[hm.group(1)]
        out.append("\n" + "#" * level + " " + _tex_inline_clean(arg) + "\n")
        pos = end
    out.append(body[pos:])
    body = "".join(out)
    body = re.sub(r"\\appendix\b", "\n## Appendix\n", body)
    body = re.sub(r"\\paragraph\*?\{([^{}]*)\}", r"\n**\1**", body)

    refs_md = _latex_references(path, full)
    if refs_md:
        body = re.sub(r"\\bibliography\{[^}]*\}|\\printbibliography", "", body)
        body += "\n\n## References\n\n" + refs_md
    else:
        warnings.append("no .bbl or .bib bibliography found next to the .tex source; references are missing")
    md = (f"# {title}\n\n" if title else "") + body.strip() + "\n"
    md = re.sub(r"\n{3,}", "\n\n", md)
    result = ingest_markdown(md, "latex")
    result.title = title or result.title
    result.abstract = _squash(abstract) if abstract else result.abstract
    result.warnings += warnings
    return result


def _latex_references(path: Path, full: str) -> str:
    bbl = path.with_suffix(".bbl")
    if bbl.is_file():
        text = bbl.read_text(encoding="utf-8", errors="replace")
        items = re.split(r"\\bibitem(?:\[[^\]]*\])?\{[^}]*\}", text)[1:]
        cleaned = []
        for item in items:
            item = re.sub(r"\\end\{thebibliography\}.*", "", item, flags=re.S)
            item = re.sub(r"\\newblock", " ", item)
            item = re.sub(r"[{}]", "", item)
            item = _squash(re.sub(r"\\[a-zA-Z]+\s?", " ", item))
            if item:
                cleaned.append(item)
        return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(cleaned, 1))
    bib_names = []
    for m in re.finditer(r"\\(?:bibliography|addbibresource)\{([^}]*)\}", full):
        bib_names += [n.strip() for n in m.group(1).split(",") if n.strip()]
    entries: dict[str, dict] = {}
    for name in bib_names:
        bib = path.parent / (name if name.endswith(".bib") else name + ".bib")
        if bib.is_file():
            entries.update(parse_bibtex(bib.read_text(encoding="utf-8", errors="replace")))
    if not entries:
        return ""
    cited: list[str] = []
    for m in re.finditer(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}", full):
        for key in m.group(1).split(","):
            key = key.strip()
            if key and key not in cited:
                cited.append(key)
    keys = [k for k in cited if k in entries] or list(entries)
    lines = []
    for i, key in enumerate(keys, 1):
        e = entries[key]
        venue = e.get("booktitle") or e.get("journal") or e.get("publisher") or ""
        ids = " ".join(x for x in (f"doi:{e['doi']}" if e.get("doi") else "",
                                   f"arXiv:{e['eprint']}" if e.get("eprint") else "") if x)
        lines.append(f"[{i}] {e.get('author', '?')} ({e.get('year', 'n.d.')}). {e.get('title', '?')}. {venue} {ids}".strip())
    return "\n\n".join(lines)


def parse_bibtex(text: str) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        if m.group(1).lower() in {"comment", "string", "preamble"}:
            continue
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start : i - 1]
        fields: dict[str, str] = {"type": m.group(1).lower()}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            key = fm.group(1).lower()
            rest = body[fm.end() :]
            if rest.startswith("{"):
                val, _ = _brace_arg(rest, 0)
            elif rest.startswith('"'):
                end = rest.find('"', 1)
                val = rest[1:end]
            else:
                val = re.split(r"[,\n]", rest, 1)[0]
            fields.setdefault(key, _squash(re.sub(r"[{}]", "", val)))
        entries[m.group(2)] = fields
    return entries


# ---------------------------------------------------------------- PDF

_ARXIV_STAMP = re.compile(r"^arXiv:\s*\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\]", re.I)
_NUMBER_ONLY = re.compile(r"^((?:\d+|[A-H])(?:\.\d+)*)\.?$")
_NUMBERED = re.compile(r"^((?:\d+|[A-H])(?:\.\d+)*)\.?\s+([A-Z(][^\n]{0,110})$")
_RUN_IN_ABSTRACT = re.compile(r"^(Abstract|ABSTRACT)\s*[.:—–-]\s*(\S.{20,})$")


@dataclass
class _Line:
    text: str
    size: float
    bold: float  # fraction of characters in bold spans
    x0: float
    y0: float
    y1: float
    page: int
    block: int
    width: float = 612.0


class _HeadingState:
    """Accept numbered headings only when their numbering advances (figure labels do not)."""

    def __init__(self) -> None:
        self.top = 0
        self.subs: dict[str, int] = {}
        self.letter = ""

    def accept(self, number: str) -> bool:
        parts = number.split(".")
        head = parts[0]
        if head.isdigit():
            k = int(head)
            if len(parts) == 1:
                if self.top < k <= self.top + 3:
                    self.top, self.subs = k, {}
                    return True
                return False
            if k != self.top:
                return False
        else:
            if len(parts) == 1:
                expected = chr(ord(self.letter) + 1) if self.letter else "A"
                if head == expected:
                    self.letter, self.subs = head, {}
                    return True
                return False
            if head != self.letter:
                return False
        key = ".".join(parts[:-1])
        last = int(parts[-1]) if parts[-1].isdigit() else 0
        if last > self.subs.get(key, 0):
            self.subs[key] = last
            return True
        return False

    def note(self, number: str | None) -> None:
        """Advance the numeric state for a heading accepted by the PDF outline.

        Single letters are ignored: in "A Study of X" the "A" is an article, not appendix A.
        """
        if not number or not number.split(".")[0].isdigit():
            return
        parts = number.split(".")
        if len(parts) == 1:
            self.top, self.subs = int(parts[0]), {}
        elif parts[-1].isdigit():
            self.subs[".".join(parts[:-1])] = int(parts[-1])


def _strip_number(text: str) -> tuple[str | None, str]:
    m = re.match(r"^((?:\d+|[A-H])(?:\.\d+)*)\.?\s+(.*)$", text.strip())
    return (m.group(1), m.group(2)) if m else (None, text.strip())


def ingest_pdf(path: Path) -> IngestResult:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover
        raise IngestError("PyMuPDF is required for PDF input (pip install pymupdf)") from exc

    warnings: list[str] = []
    doc = pymupdf.open(path)
    page_count = doc.page_count
    toc = doc.get_toc(simple=True) or []
    lines: list[_Line] = []
    heights: dict[int, float] = {}
    for pno, page in enumerate(doc, start=1):
        heights[pno] = page.rect.height or 792.0
        width = page.rect.width or 612.0
        data = page.get_text("dict")
        for bno, block in enumerate(data.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                direction = line.get("dir") or (1.0, 0.0)
                if abs(direction[1]) > 0.5:  # vertical text: margin stamps, rotated labels
                    continue
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                if _ARXIV_STAMP.match(text):
                    continue
                chars = sum(len(s["text"]) for s in spans) or 1
                bold = sum(len(s["text"]) for s in spans
                           if (s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower()) / chars
                size = max(s.get("size", 0) for s in spans)
                x0, y0, _, y1 = line["bbox"]
                lines.append(_Line(text, round(size, 1), bold, x0, y0, y1, pno, bno, width))
    doc_meta = doc.metadata or {}
    doc.close()
    if not lines:
        raise IngestError("no extractable text in the PDF (scanned image?); OCR it first or provide the source")

    body_size = _body_font_size(lines)
    lines = _drop_running_headers(lines, heights, page_count)
    lines = _merge_number_lines(lines, body_size)
    title = _pdf_title(doc_meta, lines, body_size)

    # Pass 1: decide which lines are headings.
    toc_left = {}
    for lvl, t, pg in toc:
        num, name = _strip_number(t)
        if name:
            toc_left.setdefault((_norm(name), pg), (lvl, num))
    state = _HeadingState()
    levels: list[int | None] = []
    skip_title = bool(title)
    for ln in lines:
        if skip_title and ln.page == 1 and ln.size > body_size + 0.5 and _norm(ln.text) \
                and _norm(ln.text) in _norm(title or ""):
            levels.append(-1)  # part of the title block
            continue
        level = _pdf_heading_level(ln, body_size, toc_left, state)
        if level:
            skip_title = False
        levels.append(level)

    # Pass 2: assemble Markdown; reference lists are split into entries using hanging indents.
    md_parts: list[str] = [f"# {title}\n" if title else ""]
    current_page = 0
    para: list[str] = []
    para_key: tuple[int, int] | None = None
    in_refs = False
    abstract_seen = False
    ref_left = _reference_left_edges(lines, levels)

    def flush() -> None:
        nonlocal para
        if para:
            md_parts.append(_join_lines(para) + "\n")
            para = []

    for ln, level in zip(lines, levels):
        if level == -1:
            continue
        if ln.page != current_page:
            flush()
            current_page = ln.page
            md_parts.append(f"<!-- page {ln.page} -->\n")
        if level:
            flush()
            heading = _clean_heading(ln.text)
            md_parts.append("#" * (level + 1) + " " + heading + "\n")
            para_key = None
            kind = classify_section(heading)
            abstract_seen = abstract_seen or kind == "abstract"
            in_refs = kind == "references"
            continue
        if not abstract_seen and ln.page <= 2:
            m = _RUN_IN_ABSTRACT.match(ln.text)
            if m:
                flush()
                md_parts.append("## Abstract\n")
                abstract_seen = True
                para, para_key = [m.group(2)], (ln.page, ln.block)
                continue
        if in_refs:
            column = 0 if ln.x0 < ln.width / 2 else 1
            left = ref_left.get((ln.page, column))
            if left is not None and ln.x0 <= left + 1.5:
                flush()  # a line at the column's left edge starts a new entry
            para.append(ln.text)
            para_key = (ln.page, ln.block)
            continue
        key = (ln.page, ln.block)
        if para_key is not None and key != para_key:
            flush()
        para_key = key
        para.append(ln.text)
    flush()

    md = "\n".join(p for p in md_parts if p)
    md = re.sub(r"\n{3,}", "\n\n", md)
    result = ingest_markdown(md, "pdf")
    result.title = title or result.title
    result.page_count = page_count
    _attach_pages(result)
    if not any(s.kind == "references" for s in result.sections):
        warnings.append("no References section was detected in the PDF text")
    if len([s for s in result.sections if s.level == 1]) < 3:
        warnings.append("few section headings were detected; section boundaries may be approximate")
    warnings.append("PDF text extraction can garble equations and tables; roles with paper_format=pdf also get the PDF")
    result.warnings += warnings
    return result


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _body_font_size(lines: list[_Line]) -> float:
    weights: Counter = Counter()
    for ln in lines:
        weights[ln.size] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _drop_running_headers(lines: list[_Line], heights: dict[int, float], pages: int) -> list[_Line]:
    if pages < 3:
        return lines
    edge: Counter = Counter()
    for ln in lines:
        h = heights.get(ln.page, 792.0)
        if ln.y1 < 0.07 * h or ln.y0 > 0.93 * h:
            edge[re.sub(r"\d+", "#", _norm(ln.text))] += 1
    repeated = {k for k, c in edge.items() if c >= max(3, 0.4 * pages)}
    out = []
    for ln in lines:
        h = heights.get(ln.page, 792.0)
        at_edge = ln.y1 < 0.07 * h or ln.y0 > 0.93 * h
        key = re.sub(r"\d+", "#", _norm(ln.text))
        if at_edge and (key in repeated or re.fullmatch(r"#?", key)):
            continue
        out.append(ln)
    return out


def _emphasized(ln: _Line, body_size: float) -> bool:
    return ln.bold >= 0.6 or ln.size >= body_size + 0.9


def _merge_number_lines(lines: list[_Line], body_size: float) -> list[_Line]:
    """Join a heading number printed on its own line with the title line that follows it."""
    out: list[_Line] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (nxt is not None and _NUMBER_ONLY.match(ln.text) and nxt.page == ln.page
                and abs(nxt.y0 - ln.y0) < 3 * max(ln.size, 1) and _emphasized(ln, body_size)
                and _emphasized(nxt, body_size) and nxt.text[:1].isupper() and len(nxt.text) <= 110):
            out.append(_Line(f"{ln.text.rstrip('.')} {nxt.text}", max(ln.size, nxt.size), min(ln.bold, nxt.bold),
                             ln.x0, ln.y0, nxt.y1, ln.page, ln.block, ln.width))
            i += 2
            continue
        out.append(ln)
        i += 1
    return out


def _pdf_title(meta: dict, lines: list[_Line], body_size: float) -> str | None:
    mt = (meta.get("title") or "").strip()
    if 10 <= len(mt) <= 250 and not re.search(r"microsoft word|untitled|\.dvi|\.pdf|\.tex|^arxiv:", mt, re.I):
        return _squash(mt)
    first = [ln for ln in lines if ln.page == 1 and len(ln.text) > 3]
    if not first:
        return None
    biggest = max(ln.size for ln in first)
    if biggest <= body_size + 1:
        return None
    parts = [ln.text for ln in first if abs(ln.size - biggest) <= 0.6][:4]
    title = _squash(" ".join(parts))
    return title if 3 <= len(title) <= 300 else None


def _pdf_heading_level(ln: _Line, body_size: float, toc: dict, state: _HeadingState) -> int | None:
    text = ln.text.strip()
    if not text or len(text) > 120:
        return None
    number, name = _strip_number(text)
    hit = toc.pop((_norm(name), ln.page), None) if name else None
    if hit is not None:  # the PDF outline is authoritative
        state.note(number or hit[1])
        return min(hit[0], 3)
    if ln.size < body_size - 0.6 or not _emphasized(ln, body_size):
        return None  # captions and figure labels are smaller than body text; headings are emphasized
    if text.lower().rstrip(".:") in _KNOWN_UNNUMBERED:
        return 1
    m = _NUMBERED.match(text)
    if m and len(text.split()) <= 14 and not text.endswith(".") and state.accept(m.group(1)):
        return min(1 + m.group(1).count("."), 3)
    return None


def _reference_left_edges(lines: list[_Line], levels: list[int | None]) -> dict[tuple[int, int], float]:
    """Left edge of each (page, column) inside the reference list; entries start there, continuations indent."""
    edges: dict[tuple[int, int], float] = {}
    in_refs = False
    for ln, level in zip(lines, levels):
        if level and level != -1:
            in_refs = classify_section(_clean_heading(ln.text)) == "references"
            continue
        if in_refs:
            key = (ln.page, 0 if ln.x0 < ln.width / 2 else 1)
            edges[key] = min(edges.get(key, ln.x0), ln.x0)
    return edges


def _clean_heading(text: str) -> str:
    return _squash(text).rstrip(".:")


def _join_lines(parts: list[str]) -> str:
    out = ""
    for part in parts:
        part = part.strip()
        if not out:
            out = part
        elif out.endswith("-") and part[:1].islower() and "-" not in out.rsplit(" ", 1)[-1][:-1]:
            out = out[:-1] + part
        else:
            out = out + " " + part
    return out


def _attach_pages(result: IngestResult) -> None:
    markers = [(m.start(), int(m.group(1))) for m in re.finditer(r"<!-- page (\d+) -->", result.text_md)]
    if not markers:
        return

    def page_at(pos: int) -> int:
        page = markers[0][1]
        for mpos, mpage in markers:
            if mpos <= pos:
                page = mpage
            else:
                break
        return page

    for s in result.sections:
        s.page_start = page_at(s.start)
        s.page_end = page_at(max(s.start, s.end - 1))
