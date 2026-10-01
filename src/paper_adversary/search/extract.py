"""Prior-work full text: arXiv LaTeXML HTML and PDFs -> the sectioned Markdown of ingest.py.

Reviewer agents read and quote prior work verbatim and a matcher checks the
quotes, so the Markdown keeps `<!-- anchor ID -->` (HTML paragraphs) and
`<!-- page N -->` (PDF pages) markers. Documents are untrusted input: they are
parsed in a child process with resource limits and a timeout, so a malformed
file can never crash or hang the caller.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict
from html.parser import HTMLParser
from pathlib import Path

from paper_adversary.ingest import IngestError, ingest_markdown, ingest_pdf

try:
    import resource
except ImportError:  # pragma: no cover - Windows: only the timeout applies
    resource = None

HTML_MIN_CHARS = 2000  # below this an arXiv HTML conversion is treated as failed
KINDS = ("pdf", "html")
FSIZE_MB = 256  # largest file the child may write
LOG_TAIL_CHARS = 2000
MAX_RESULT_BYTES = 64 * 1024 * 1024

_RESULT_KEYS = ("status", "reason", "detail", "text_md", "sections", "title", "page_count", "source_format",
                "warnings")


def _result(status: str, reason: str | None = None, detail: str | None = None, **fields) -> dict:
    out = {"status": status, "reason": reason, "detail": detail, "text_md": None, "sections": [], "title": None,
           "page_count": None, "source_format": None, "warnings": []}
    out.update(fields)
    return out


# ---------------------------------------------------------------- arXiv HTML (LaTeXML)

_HEADING_LEVELS = {
    "ltx_title_document": 1, "ltx_title_part": 2, "ltx_title_chapter": 2, "ltx_title_section": 2,
    "ltx_title_appendix": 2, "ltx_title_acknowledgements": 2, "ltx_title_subsection": 3,
    "ltx_title_subsubsection": 4, "ltx_title_paragraph": 5, "ltx_title_subparagraph": 6,
}
_CHAPTER_LEVELS = _HEADING_LEVELS | {  # theses and books: sections sit below chapters
    "ltx_title_section": 3, "ltx_title_subsection": 4, "ltx_title_subsubsection": 5, "ltx_title_paragraph": 6,
}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track",
              "wbr"}
_HIDDEN_TAGS = {"script", "style", "noscript", "template"}
_DROP_TAGS = _HIDDEN_TAGS | {"nav", "header", "footer", "button", "svg", "form", "iframe", "object", "canvas"}
_DROP_CLASSES = {"ltx_page_header", "ltx_page_footer", "ltx_page_logo", "ltx_page_navbar", "ltx_TOC", "ltx_note_mark",
                 "ltx_note_type", "ltx_ERROR", "ltx_listing_data", "ltx_title_abstract", "ltx_title_bibliography",
                 "package-alerts"}
_AUTHOR_DROP_CLASSES = {"ltx_author_notes", "ltx_contact", "ltx_role_affiliation", "ltx_role_address",
                        "ltx_role_email", "ltx_author_before", "ltx_author_after", "ltx_note"}
_BLOCK_TAGS = {"address", "article", "aside", "blockquote", "caption", "dd", "div", "dl", "dt", "figcaption", "figure",
               "h1", "h2", "h3", "h4", "h5", "h6", "li", "main", "ol", "p", "pre", "section", "table", "tbody", "td",
               "tfoot", "th", "thead", "tr", "ul"}
_EQUATION_CLASSES = {"ltx_equation", "ltx_equationgroup", "ltx_eqn_table"}
_FAILED = re.compile(r"Conversion to HTML had a Fatal error|HTML is not available for the source|No HTML for\b")
_ANCHOR_ID = re.compile(r"[\w.:]+(?:-[\w.:]+)*")
_SOFT = "\ue000"  # separator around tags: a space, unless punctuation follows ("Theorem 1.")


def _collapse(text: str) -> str:
    text = re.sub(_SOFT + r"+(?=\s*[.,;:!?)\]])", "", text)
    return re.sub(r"\s+", " ", text.replace(_SOFT, " ")).strip()


def _escape(line: str) -> str:
    return "\\" + line if re.match(r"#{1,6}[ \t]", line) else line


class _LatexmlParser(HTMLParser):
    """Streams LaTeXML HTML into Markdown blocks; an element's role comes from its tag and ltx_* classes."""

    def __init__(self, scoped: bool, chapters: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.scoped = scoped  # convert only <article class="ltx_document">, not the arXiv page around it
        self.levels = _CHAPTER_LEVELS if chapters else _HEADING_LEVELS
        self.in_doc = 0
        self.skip = 0  # open dropped elements: their text is ignored
        self.hidden = 0  # open script/style elements: their text is not even visible
        self.inline = 0  # inside headings, notes, tables, ...: no paragraph breaks or anchors
        self.in_heading = 0
        self.in_bib = 0
        self.notes = 0
        self.stack: list[tuple[str, str | None, object]] = []
        self.open_tags: dict[str, int] = {}  # stray end tags are skipped without searching the stack
        self.sinks: list[list[str]] = []  # text of the heading, cell, note, ... being read
        self.buf: list[str] = []  # the paragraph being read
        self.prefix: list[str] = []  # run-in titles and item labels waiting for their paragraph
        self.anchor: str | None = None
        self.anchor_used = True
        self.blocks: list[str] = []
        self.body_chars = 0
        self.visible: list[str] = []
        self.capture: list[str] | None = None  # TeX annotation of a math element without alttext
        self.title: str | None = None
        self.html_title: str | None = None
        self.title_parts: list[str] | None = None
        self.authors: dict | None = None
        self.math: dict | None = None
        self.eq: dict | None = None
        self.tables: list[list[list[str]]] = []
        self.headings = 0
        self.paras = 0
        self.has_bib = False
        self.alerts = False
        self.errors = 0
        self.lost_math = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.title_parts is not None:  # <title> holds only text: a tag means it was left open
            self._end_title()
        attr = {k: v or "" for k, v in attrs}
        cls = set(attr.get("class", "").split())
        self.alerts = self.alerts or "package-alerts" in cls
        if tag not in _VOID_TAGS:
            self.stack.append((tag, *self._open(tag, cls, attr)))
            self.open_tags[tag] = self.open_tags.get(tag, 0) + 1
        elif tag == "br" and self._live():
            if self.authors is not None:
                self.authors["cut"] = True  # what follows a line break in an author block is an affiliation
            else:
                self._text(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self.open_tags.get(tag):
            return
        while self._pop() != tag:  # elements left open inside are closed with it
            pass

    def handle_data(self, data: str) -> None:
        data = data.replace(_SOFT, "")
        if not self.hidden:
            self.visible.append(data)
        if self.title_parts is not None:
            self.title_parts.append(data)
        elif self.capture is not None:
            self.capture.append(data)
        elif self._live():
            if self.eq is not None:
                if self.eq["tag"] is not None:
                    self.eq["tag"].append(data)
            elif self.authors is not None:
                if not self.authors["cut"]:
                    self.authors["cur"].append(data)
            else:
                self._text(data)

    def close(self) -> None:
        super().close()
        while self.stack:
            self._pop()
        self._flush(force=True)

    def _pop(self) -> str:
        tag, role, data = self.stack.pop()
        self.open_tags[tag] -= 1
        self._close(tag, role, data)
        return tag

    # -- element roles

    def _open(self, tag: str, cls: set[str], attr: dict[str, str]) -> tuple[str | None, object]:
        if tag in _HIDDEN_TAGS:
            self.hidden += 1
            self.skip += 1
            return "drop", True
        if self.skip:
            if self.math is not None and not self.math["alt"] and tag == "annotation" \
                    and "tex" in attr.get("encoding", ""):
                self.capture = self.math["tex"]
                return "capture", None
            return None, None
        if tag == "title" and not self.in_doc and self.html_title is None:
            self.title_parts = []
            return "html_title", None
        if self.scoped and not self.in_doc:
            if tag == "article" and "ltx_document" in cls:
                self.in_doc += 1
                return "doc", None
            return None, None
        if tag in _DROP_TAGS or cls & _DROP_CLASSES or (self.notes and "ltx_tag" in cls) \
                or (self.in_heading and "ltx_note" in cls):
            self.errors += "ltx_ERROR" in cls
            self.skip += 1
            return "drop", False
        if self.authors is not None:
            return self._open_in_authors(tag, cls)
        if tag == "math":
            self.skip += 1
            self.math = {"alt": _collapse(attr.get("alttext", "")), "tex": [], "block": attr.get("display") == "block"}
            return "math", None
        if self.eq is not None:
            if tag == "tr":
                self.eq["rows"].append({"math": [], "tag": ""})
            elif "ltx_tag" in cls and self.eq["tag"] is None:
                self.eq["tag"] = []
                return "eqtag", None
            return None, None
        if cls & _EQUATION_CLASSES:
            self._break()
            self.eq = {"rows": [], "tag": None}
            return "eq", None
        level = min((self.levels[c] for c in cls if c in self.levels), default=None)
        if level is not None:
            self._break(force=True)
            self.in_heading += 1
            self.inline += 1
            self.sinks.append([])
            return "heading", level
        if self.in_heading and "ltx_tag" in cls:
            self.sinks.append([])
            return "htag", None
        if "ltx_authors" in cls:
            self._break(force=True)
            self.authors = {"names": [], "cur": [], "cut": False, "simple": True}
            return "authors", None
        if "ltx_abstract" in cls:
            self._break(force=True)
            self.blocks.append("## Abstract")
            return "block", False
        if "ltx_bibliography" in cls:
            self._break(force=True)
            self.blocks.append("## References")
            self.has_bib = True
            self.in_bib += 1
            return "bib", None
        if "ltx_bibitem" in cls:
            self._break()
            self.inline += 1
            return "bibitem", None
        if "ltx_note" in cls:
            self.notes += 1
            self.inline += 1
            return "note", None
        if "ltx_note_content" in cls:
            self.sinks.append([])
            return "notecontent", None
        if tag == "table" or "ltx_tabular" in cls:
            block = tag == "table" and not self.inline
            self._break()
            self.tables.append([])
            self.inline += 1
            return "table", block
        if self.tables and (tag == "tr" or "ltx_tr" in cls):
            self.tables[-1].append([])
            return None, None
        if self.tables and (tag in ("td", "th") or "ltx_td" in cls):
            self.sinks.append([])
            return "cell", None
        if not self.inline and cls & {"ltx_tag_item", "ltx_runin"}:
            self._break()
            self.inline += 1
            self.sinks.append([])
            return "prefix", None
        if "ltx_para" in cls and not self.inline and _ANCHOR_ID.fullmatch(attr.get("id", "")):
            self._break()
            self.paras += 1
            saved = (self.anchor, self.anchor_used)
            self.anchor, self.anchor_used = attr["id"], False
            return "para", saved
        if tag in _BLOCK_TAGS or "ltx_caption" in cls:
            self._break()
            return "block", tag in ("li", "dd") or bool(cls & {"ltx_theorem", "ltx_proof"})
        if cls & {"ltx_tag", "ltx_bibblock"}:
            self._text(_SOFT)
            return "soft", None
        return None, None

    def _open_in_authors(self, tag: str, cls: set[str]) -> tuple[str | None, object]:
        if tag in ("sup", "math") or cls & _AUTHOR_DROP_CLASSES:
            if tag in ("sup", "math") and not self.authors["cut"]:
                self.authors["cur"].append(",")  # affiliation marks are often the only separator between names
            self.skip += 1
            return "drop", False
        if tag == "table" or cls & {"ltx_tabular", "ltx_td"}:
            self.authors["simple"] = False  # names laid out in a table: too irregular for one line
        if "ltx_creator" in cls:
            self._next_author()
            return "creator", None
        return None, None

    def _close(self, tag: str, role: str | None, data: object) -> None:
        if role is None:
            return
        if role == "drop":
            self.skip -= 1
            self.hidden -= bool(data)
        elif role == "capture":
            self.capture = None
        elif role == "html_title":
            if self.title_parts is not None:
                self._end_title()
        elif role == "doc":
            self._flush(force=True)
            self.in_doc -= 1
        elif role == "math":
            self._close_math()
        elif role == "eqtag":
            self._eq_row()["tag"] = _collapse("".join(self.eq["tag"]))
            self.eq["tag"] = None
        elif role == "eq":
            self._close_equation()
        elif role == "heading":
            self.in_heading -= 1
            self.inline -= 1
            text = _collapse("".join(self.sinks.pop()))
            if text:
                self._heading(data, text)
        elif role == "htag":
            self._text(_collapse("".join(self.sinks.pop())).rstrip(".:").strip() + " ")
        elif role == "creator":
            self._next_author()
        elif role == "authors":
            self._close_authors()
        elif role == "bib":
            self._flush(force=True)
            self.in_bib -= 1
        elif role == "bibitem":
            self.inline -= 1
            self._break()
        elif role == "note":
            self.notes -= 1
            self.inline -= 1
        elif role == "notecontent":
            text = _collapse("".join(self.sinks.pop()))
            if text:
                self._text(f"{_SOFT}({text}){_SOFT}")
        elif role == "table":
            self._close_table(data)
        elif role == "cell":
            text = _collapse("".join(self.sinks.pop()))
            if not self.tables[-1]:
                self.tables[-1].append([])
            self.tables[-1][-1].append(text)
        elif role == "prefix":
            self.inline -= 1
            text = _collapse("".join(self.sinks.pop()))
            if text:
                self.prefix.append(text)
        elif role == "para":
            self._flush()
            nested_used = self.anchor_used
            self.anchor, used = data
            self.anchor_used = used and not nested_used  # text after a nested anchored block re-states the anchor
        elif role == "block":
            self._break(force=bool(data))
        elif role == "soft":
            self._text(_SOFT)

    def _close_math(self) -> None:
        self.skip -= 1
        math, self.math = self.math, None
        tex = math["alt"] or _collapse("".join(math["tex"]))
        if not tex:
            self.lost_math += 1
        elif self.eq is not None:
            self._eq_row()["math"].append(tex)
        else:
            self._text(f"{_SOFT}$${tex}$${_SOFT}" if math["block"] else f"${tex}$")

    def _eq_row(self) -> dict:
        if not self.eq["rows"]:
            self.eq["rows"].append({"math": [], "tag": ""})
        return self.eq["rows"][-1]

    def _close_equation(self) -> None:
        eq, self.eq = self.eq, None
        for row in eq["rows"]:
            if row["math"]:
                line = "$$" + " ".join(row["math"]) + "$$" + (f" {row['tag']}" if row["tag"] else "")
                if self.inline:
                    self._text(f"{_SOFT}{line}{_SOFT}")
                else:
                    self._emit(line)

    def _close_table(self, block: bool) -> None:
        rows = self.tables.pop()
        self.inline -= 1
        lines = [" | ".join(row).strip() for row in rows if any(row)]
        if block:
            self._flush()  # stray text outside the cells
            if lines:
                self._emit("\n".join(lines))
        elif lines:
            self._text(_SOFT + " ".join(lines) + _SOFT)

    def _end_title(self) -> None:
        self.html_title = _collapse("".join(self.title_parts))
        self.title_parts = None

    def _heading(self, level: int, text: str) -> None:
        if level == 1 and self.title is None:
            self.title = text
            self.blocks.append(f"# {text}")
            return
        self.headings += 1
        self.blocks.append("#" * max(level, 2) + " " + text)

    def _next_author(self) -> None:
        name = _collapse("".join(self.authors["cur"]))
        if name.strip(" ,;"):
            self.authors["names"].append(name)
        self.authors["cur"], self.authors["cut"] = [], False

    def _close_authors(self) -> None:
        self._next_author()
        authors, self.authors = self.authors, None
        line = re.sub(r"\s*,[\s,]*", ", ", ", ".join(authors["names"])).strip(" ,;")
        if authors["simple"] and line and len(line) <= 1000 and "@" not in line:
            self.blocks.append(_escape(line))

    # -- output

    def _live(self) -> bool:
        return not self.skip and (self.in_doc > 0 or not self.scoped)

    def _text(self, text: str) -> None:
        (self.sinks[-1] if self.sinks else self.buf).append(text)

    def _break(self, force: bool = False) -> None:
        if not self.inline:
            self._flush(force)

    def _flush(self, force: bool = False) -> None:
        text = _collapse("".join(self.buf))
        self.buf = []
        if self.prefix and (text or force):
            text = _collapse(" ".join(self.prefix + [text]))
            self.prefix = []
        if text:
            self._emit(text)

    def _emit(self, text: str) -> None:
        text = "\n".join(_escape(line) for line in text.split("\n"))
        if not self.in_bib:
            self.body_chars += len(text)
        if self.anchor and not self.anchor_used:
            text = f"<!-- anchor {self.anchor} -->\n{text}"
            self.anchor_used = True
        self.blocks.append(text)


def latexml_to_markdown(html: str) -> tuple[str, dict]:
    """arXiv LaTeXML HTML -> (markdown, info).

    info = {"ok": bool, "title": str|None, "warnings": [...], "reason": str|None}; ok is False when the page
    reports a failed conversion, is not a LaTeXML paper, or yields fewer than HTML_MIN_CHARS of body text.
    """
    parser = _LatexmlParser(scoped=bool(re.search(r"<article\b[^>]*\bltx_document\b", html, re.I)),
                            chapters="ltx_title_chapter" in html)
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # html.parser is lenient, but the input is untrusted
        return "", {"ok": False, "title": None, "warnings": [], "reason": f"the HTML could not be parsed: {exc}"}
    title = parser.title
    if title is None and parser.html_title:
        title = re.sub(r"^\[[^\]]*\]\s*", "", parser.html_title) or None  # ar5iv titles start with "[arXiv id]"
        if title:
            parser.blocks.insert(0, f"# {title}")
    md = "\n\n".join(parser.blocks) + "\n"
    warnings = []
    if parser.alerts:
        warnings.append("arXiv reports LaTeX packages its HTML converter does not support; some content may be missing")
    if parser.errors:
        warnings.append(f"{parser.errors} LaTeX command(s) could not be converted to HTML; nearby text may be garbled")
    if parser.lost_math:
        warnings.append(f"{parser.lost_math} math expression(s) had no LaTeX source and were left out")
    if not parser.headings:
        warnings.append("no section headings were found in the HTML")
    if not parser.has_bib:
        warnings.append("no bibliography was found in the HTML")
    failed = _FAILED.search(_collapse(" ".join(parser.visible)))
    if failed:
        reason = f'the page says the HTML conversion failed ("{failed.group(0)}")'
    elif not parser.paras:
        reason = "no LaTeXML paragraphs were found; this does not look like an arXiv HTML paper"
    elif parser.body_chars < HTML_MIN_CHARS:
        reason = f"only {parser.body_chars} characters of body text were converted (minimum {HTML_MIN_CHARS})"
    else:
        reason = None
    return md, {"ok": reason is None, "title": title, "warnings": warnings, "reason": reason}


# ---------------------------------------------------------------- child process


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def extract_main(argv: list[str]) -> int:
    """Child-process entry: --in PATH --kind pdf|html --out JSON [--max-pages N]. Writes the JSON result; exit 0 even on
    extraction errors (the error is in the JSON); nonzero only for usage errors."""
    parser = argparse.ArgumentParser(prog="paper-adversary extract-fulltext",
                                     description="Convert one prior-work document (normally run by run_extraction).")
    parser.add_argument("--in", dest="src", required=True, help="the PDF or arXiv HTML file")
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--out", required=True, help="where to write the JSON result")
    parser.add_argument("--max-pages", type=_positive_int, help="read at most this many PDF pages")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    src = Path(args.src)
    try:
        result = _extract_pdf(src, args.max_pages) if args.kind == "pdf" else _extract_html(src)
    except Exception as exc:  # anything the parsers did not anticipate in hostile input
        reason = "extraction_failed" if args.kind == "pdf" else "conversion_failed"
        result = _result("error", reason, f"{type(exc).__name__}: {exc}")
    Path(args.out).write_text(json.dumps(result), encoding="utf-8")
    return 0


def _extract_pdf(path: Path, max_pages: int | None) -> dict:
    import pymupdf

    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception as exc:
        return _result("error", "extraction_failed", f"cannot open the PDF: {exc}")
    try:
        locked, pages = bool(doc.needs_pass or doc.is_encrypted), doc.page_count
    finally:
        doc.close()
    if locked:
        return _result("error", "encrypted", "the PDF is password-protected")
    if not pages:
        return _result("error", "extraction_failed", "the PDF has no pages")
    try:
        result = ingest_pdf(path, max_pages=max_pages)
    except IngestError as exc:
        return _result("error", "no_text_layer" if "no extractable text" in str(exc) else "extraction_failed", str(exc))
    except Exception as exc:
        return _result("error", "extraction_failed", f"{type(exc).__name__}: {exc}")
    warnings = list(result.warnings)
    read = min(pages, max_pages or pages)
    if len(result.text_md) < 100 * read:
        warnings.append(f"very little text was extracted ({len(result.text_md)} characters from {read} pages); "
                        "the PDF may be scanned")
    return _result("ok", text_md=result.text_md, sections=[asdict(s) for s in result.sections], title=result.title,
                   page_count=result.page_count, source_format="pdf", warnings=warnings)


def _extract_html(path: Path) -> dict:
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _result("error", "extraction_failed", f"cannot read the HTML file: {exc}")
    md, info = latexml_to_markdown(html)
    if not info["ok"]:
        return _result("error", "conversion_failed", info["reason"], title=info["title"], warnings=info["warnings"])
    result = ingest_markdown(md, "arxiv_html")
    return _result("ok", text_md=result.text_md, sections=[asdict(s) for s in result.sections],
                   title=info["title"] or result.title, source_format="arxiv_html",
                   warnings=info["warnings"] + result.warnings)


# ---------------------------------------------------------------- parent side


def run_extraction(src: Path, kind: str, *, max_pages: int | None = 80, timeout_s: float = 180, mem_mb: int = 2048,
                   cpu_s: int = 120) -> dict:
    """Run extract_main in a child process (sys.executable -m paper_adversary extract-fulltext ...) with
    resource limits. Never raises for a bad document: problems come back as status "error" with a reason."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        return _result("error", "extraction_failed", f"file not found: {src}")
    with tempfile.TemporaryDirectory(prefix="pa-extract-", ignore_cleanup_errors=True) as tmp:
        out, log = Path(tmp) / "result.json", Path(tmp) / "child.log"
        cmd = [sys.executable, "-m", "paper_adversary", "extract-fulltext", f"--in={src}", f"--kind={kind}",
               f"--out={out}"]
        if kind == "pdf" and max_pages:
            cmd.append(f"--max-pages={int(max_pages)}")
        kwargs: dict = {"cwd": tmp, "env": _child_env(), "stdin": subprocess.DEVNULL, "stderr": subprocess.STDOUT}
        if os.name == "posix":
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = functools.partial(_limit_child, mem_mb, cpu_s)
        # Output goes to a file, not a pipe: a flood of parser warnings cannot fill this process's memory.
        with open(log, "wb") as log_fh:
            try:
                proc = subprocess.Popen(cmd, stdout=log_fh, **kwargs)
            except Exception as exc:  # OSError, SubprocessError, or preexec_fn refused (subinterpreters)
                return _result("error", "crashed", f"could not start the extraction process: {exc}")
            try:
                code = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                _kill(proc)
                return _result("error", "timeout", f"the extraction did not finish within {timeout_s:g} s")
            except BaseException:
                _kill(proc)
                raise
        return _read_result(code, out, log, cpu_s)


_CHILD_ENV = ("PATH", "HOME", "LANG", "LANGUAGE", "TMPDIR", "TEMP", "TMP", "PYTHONPATH", "PYTHONHOME",
              "PYTHONIOENCODING", "PYTHONUTF8", "SYSTEMROOT", "WINDIR")


def _child_env() -> dict[str, str]:
    """The child parses untrusted documents: it gets what Python needs to start and no credentials (it is
    dispatched before the CLI loads .env, so the token never enters its environment)."""
    return {k: v for k, v in os.environ.items() if k in _CHILD_ENV or k.startswith("LC_")}


def _limit_child(mem_mb: int, cpu_s: int) -> None:
    """preexec_fn: cap the child's address space, CPU time and file sizes; each limit is best effort."""
    mb = 1024 * 1024
    limits = [(resource.RLIMIT_FSIZE, FSIZE_MB * mb, FSIZE_MB * mb), (resource.RLIMIT_CORE, 0, 0)]
    if mem_mb:
        limits.append((resource.RLIMIT_AS, mem_mb * mb, mem_mb * mb))
    if cpu_s:
        limits.append((resource.RLIMIT_CPU, cpu_s, cpu_s + 5))  # SIGXCPU at the soft limit, SIGKILL at the hard one
    for res, soft, hard in limits:
        try:
            _, current = resource.getrlimit(res)
            if current != resource.RLIM_INFINITY:
                soft, hard = min(soft, current), min(hard, current)
            resource.setrlimit(res, (soft, hard))
        except (ValueError, OSError):
            pass


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)  # the child leads its own session
        else:
            proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _read_result(code: int, out: Path, log: Path, cpu_s: int) -> dict:
    xcpu = getattr(signal, "SIGXCPU", None)
    if xcpu is not None and code == -xcpu:
        return _result("error", "timeout", f"the extraction used more than {cpu_s} s of CPU time")
    tail = _tail(log)
    tail = f": {tail}" if tail else ""
    if code != 0:
        return _result("error", "crashed", f"the extraction process {_exit_text(code)}{tail}")
    try:
        if out.stat().st_size > MAX_RESULT_BYTES:
            return _result("error", "extraction_failed",
                           f"the extracted text is larger than {MAX_RESULT_BYTES // (1024 * 1024)} MB")
        data = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _result("error", "crashed", f"the extraction process wrote no valid result ({exc}){tail}")
    if not isinstance(data, dict) or data.get("status") not in ("ok", "error"):
        return _result("error", "crashed", "the extraction process wrote an unexpected result")
    return _result(**{k: data[k] for k in _RESULT_KEYS if k in data})


def _exit_text(code: int) -> str:
    if code < 0:
        try:
            return f"was killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"was killed by signal {-code}"
    return f"exited with code {code}"


def _tail(path: Path, chars: int = LOG_TAIL_CHARS) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 4 * chars))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return text[-chars:].strip()
