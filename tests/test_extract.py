"""Prior-work extraction: arXiv LaTeXML HTML -> Markdown, and the sandboxed child process for PDFs and HTML."""

import json
import os
import random
import signal
import sys
import time

import pytest

from conftest import FAKE_SECRETS
from paper_adversary.ingest import ingest_markdown, ingest_pdf
from paper_adversary.search import extract
from paper_adversary.search.extract import extract_main, latexml_to_markdown, run_extraction

FILLER = ("Noise schedules decide how quickly the signal is destroyed during the forward process, and they shape "
          "both sample quality and the number of steps that sampling needs. ")

# A LaTeXML paper as arxiv.org/html serves it: <article class="ltx_document"> inside the site's chrome.
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Diffusion Schedules Revisited</title>
<script>window.leak = "SCRIPT_LEAK";</script>
<style>.ltx_para { color: #333; } /* STYLE_LEAK */</style>
</head>
<body>
<header class="desktop_header"><a href="https://arxiv.org">arXiv</a> CHROME_LEAK</header>
<div class="package-alerts ltx_document" role="status"><p>HTML conversions sometimes display errors.</p>
<ul><li>failed: xr-hyper</li></ul></div>
<div class="ltx_page_main">
<div class="ltx_page_header">PAGE_HEADER_LEAK</div>
<div class="ltx_page_content">
<article class="ltx_document ltx_authors_1line">
<h1 class="ltx_title ltx_title_document">Diffusion Schedules
  Revisited<span class="ltx_note ltx_role_thanks"><sup class="ltx_note_mark">1</sup><span class="ltx_note_outer"><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup>THANKS_LEAK</span></span></span></h1>
<div class="ltx_authors">
<span class="ltx_creator ltx_role_author"><span class="ltx_personname">Ada Lovelace<sup class="ltx_sup">1</sup></span><span class="ltx_author_notes"><span class="ltx_contact ltx_role_affiliation">Analytical Engine Lab</span>
<span class="ltx_contact ltx_role_email">ada@example.org</span></span></span>
<span class="ltx_author_before">&amp;</span><span class="ltx_creator ltx_role_author"><span class="ltx_personname">Alan Turing<br class="ltx_break">Bletchley Park</span></span>
</div>
<div class="ltx_abstract">
<h6 class="ltx_title ltx_title_abstract">Abstract</h6>
<p class="ltx_p" id="id1.id1">We revisit noise schedules for diffusion models and find that a cosine schedule is a robust default.</p>
</div>
<section class="ltx_section" id="S1">
<h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">1 </span>Introduction</h2>
<div class="ltx_para" id="S1.p1">
<p class="ltx_p" id="S1.p1.1">Diffusion models <cite class="ltx_cite ltx_citemacro_cite">[<a class="ltx_ref" href="#bib.bib1" title="">1</a>]</cite> invert a fixed noising process<span class="ltx_note ltx_role_footnote" id="footnote1"><sup class="ltx_note_mark">1</sup><span class="ltx_note_outer"><span class="ltx_note_content"><sup class="ltx_note_mark">1</sup><span class="ltx_tag ltx_tag_note">1</span>Code is available on request.</span></span></span>. FILLER</p>
</div>
<div class="ltx_para" id="S1.p2">
<p class="ltx_p" id="S1.p2.1">Our contributions are:</p>
<ul class="ltx_itemize" id="S1.I1">
<li class="ltx_item" id="S1.I1.i1"><span class="ltx_tag ltx_tag_item">•</span>
<div class="ltx_para" id="S1.I1.i1.p1"><p class="ltx_p" id="S1.I1.i1.p1.1">a finite-step analysis of schedules;</p></div></li>
<li class="ltx_item" id="S1.I1.i2"><span class="ltx_tag ltx_tag_item">•</span>
<div class="ltx_para" id="S1.I1.i2.p1"><p class="ltx_p" id="S1.I1.i2.p1.1">a benchmark.</p></div></li>
</ul>
<p class="ltx_p" id="S1.p2.2">Both are released.</p>
</div>
</section>
<section class="ltx_section" id="S2">
<h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">2 </span>Method</h2>
<div class="ltx_para" id="S2.p1">
<p class="ltx_p" id="S2.p1.1">The schedule <math alttext="\alpha_t" class="ltx_Math" display="inline" id="S2.p1.m1"><semantics><msub><mi>α</mi><mi>t</mi></msub><annotation-xml encoding="MathML-Content"><ci>MATHML_LEAK</ci></annotation-xml><annotation encoding="application/x-tex">\alpha_t</annotation></semantics></math> decays smoothly. FILLER</p>
<table class="ltx_equation ltx_eqn_table" id="S2.E1">
<tbody><tr class="ltx_equation ltx_eqn_row ltx_align_baseline">
<td class="ltx_eqn_cell ltx_eqn_center_padleft"></td>
<td class="ltx_eqn_cell ltx_align_center"><math alttext="x_t=\sqrt{\bar{\alpha}_t}\,x_0" class="ltx_Math" display="block" id="S2.E1.m1"><mi>EQ_MATHML_LEAK</mi></math></td>
<td class="ltx_eqn_cell ltx_eqn_center_padright"></td>
<td class="ltx_eqn_cell ltx_eqn_eqno ltx_align_middle ltx_align_right" rowspan="1"><span class="ltx_tag ltx_tag_equation ltx_align_right">(1)</span></td>
</tr></tbody>
</table>
<p class="ltx_p" id="S2.p1.2">where <math alttext="x_0" class="ltx_Math" display="inline" id="S2.p1.m2"><mi>x</mi></math> is a data point.</p>
</div>
<div class="ltx_theorem ltx_theorem_theorem" id="Thmtheorem1">
<h6 class="ltx_title ltx_runin ltx_title_theorem"><span class="ltx_tag ltx_tag_theorem">Theorem 1</span>.</h6>
<div class="ltx_para" id="Thmtheorem1.p1"><p class="ltx_p" id="Thmtheorem1.p1.1">Every monotone schedule converges.</p></div>
</div>
<section class="ltx_subsection" id="S2.SS1">
<h3 class="ltx_title ltx_title_subsection"><span class="ltx_tag ltx_tag_subsection">2.1 </span>Cosine
  schedules</h3>
<div class="ltx_para" id="S2.SS1.p1"><p class="ltx_p" id="S2.SS1.p1.1">FILLER</p></div>
<figure class="ltx_figure" id="S2.F1"><img alt="Refer to caption" class="ltx_graphics" src="x1.png">
<figcaption class="ltx_caption ltx_centering"><span class="ltx_tag ltx_tag_figure">Figure 1: </span>Cosine and linear schedules.</figcaption>
</figure>
<figure class="ltx_table" id="S2.T1">
<figcaption class="ltx_caption ltx_centering"><span class="ltx_tag ltx_tag_table">Table 1: </span>Sample quality.</figcaption>
<table class="ltx_tabular ltx_align_middle" id="S2.T1.1">
<thead class="ltx_thead"><tr class="ltx_tr"><th class="ltx_td ltx_th">Schedule</th><th class="ltx_td ltx_th">FID</th></tr></thead>
<tbody class="ltx_tbody"><tr class="ltx_tr"><td class="ltx_td">cosine</td><td class="ltx_td">3.1</td></tr>
<tr class="ltx_tr"><td class="ltx_td">linear</td><td class="ltx_td">4.2</td></tr></tbody>
</table>
</figure>
</section>
</section>
<section class="ltx_bibliography" id="bib">
<h2 class="ltx_title ltx_title_bibliography">References</h2>
<ul class="ltx_biblist">
<li class="ltx_bibitem" id="bib.bib1"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">[1]</span>
<span class="ltx_bibblock">J. Ho, A. Jain, and P. Abbeel.</span>
<span class="ltx_bibblock">Denoising diffusion probabilistic models.</span>
<span class="ltx_bibblock">In <em class="ltx_emph ltx_font_italic">NeurIPS</em>, 2020.</span></li>
<li class="ltx_bibitem" id="bib.bib2"><span class="ltx_tag ltx_role_refnum ltx_tag_bibitem">[2]</span><span class="ltx_bibblock">A. Nichol and P. Dhariwal.</span><span class="ltx_bibblock">Improved denoising diffusion probabilistic models.</span><span class="ltx_bibblock">In ICML, 2021.</span></li>
</ul>
</section>
<section class="ltx_appendix" id="A1">
<h2 class="ltx_title ltx_title_appendix"><span class="ltx_tag ltx_tag_appendix">Appendix A </span>Proofs</h2>
<div class="ltx_para" id="A1.p1"><p class="ltx_p" id="A1.p1.1">Theorem 1 follows from monotonicity.</p></div>
</section>
</article>
</div>
<footer class="ltx_page_footer"><div class="ltx_page_logo">Generated by LaTeXML FOOTER_LEAK</div></footer>
</div>
<button class="report-issue">BUTTON_LEAK</button>
</body>
</html>
""".replace("FILLER", FILLER * 5)

NO_HTML_PAGE = """<!DOCTYPE html><html><head><title>arXiv</title></head><body>
<header>arXiv</header><main><h1>No HTML for '2401.00001'</h1>
<p>HTML is not available for the source. This could be due to the source files not being HTML, LaTeX, or a
conversion failure.</p></main></body></html>"""

RESULT_KEYS = {"status", "reason", "detail", "text_md", "sections", "title", "page_count", "source_format", "warnings"}


def test_latexml_to_markdown_structure():
    md, info = latexml_to_markdown(PAGE)
    assert info["ok"] and info["reason"] is None
    assert info["title"] == "Diffusion Schedules Revisited"
    assert md.startswith("# Diffusion Schedules Revisited\n\nAda Lovelace, Alan Turing\n\n## Abstract\n\nWe revisit")
    for heading in ("## 1 Introduction", "## 2 Method", "### 2.1 Cosine schedules", "## References"):
        assert f"\n{heading}\n" in md, heading
    assert "<!-- anchor S1.p1 -->\nDiffusion models [1] invert a fixed noising process (Code is available on " \
           "request.). Noise schedules" in md
    assert "<!-- anchor S2.p1 -->\nThe schedule $\\alpha_t$ decays smoothly." in md
    assert "\n\n$$x_t=\\sqrt{\\bar{\\alpha}_t}\\,x_0$$ (1)\n\nwhere $x_0$ is a data point.\n" in md
    assert "<!-- anchor S1.I1.i1.p1 -->\n• a finite-step analysis of schedules;" in md
    assert "<!-- anchor S1.p2 -->\nBoth are released." in md  # text after a nested list re-states its anchor
    assert "<!-- anchor Thmtheorem1.p1 -->\nTheorem 1. Every monotone schedule converges." in md
    assert "\nFigure 1: Cosine and linear schedules.\n" in md
    assert "\nSchedule | FID\ncosine | 3.1\nlinear | 4.2\n" in md
    assert "\n[1] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.\n" in md
    assert "\n[2] A. Nichol and P. Dhariwal. Improved denoising diffusion probabilistic models. In ICML, 2021.\n" in md
    assert "\n## Appendix A Proofs\n\n<!-- anchor A1.p1 -->\nTheorem 1 follows from monotonicity.\n" in md
    for dropped in ("SCRIPT_LEAK", "STYLE_LEAK", "CHROME_LEAK", "PAGE_HEADER_LEAK", "FOOTER_LEAK", "BUTTON_LEAK",
                    "xr-hyper", "MATHML_LEAK", "α", "THANKS_LEAK", "Analytical Engine", "ada@example.org", "Bletchley",
                    "Refer to caption"):
        assert dropped not in md, dropped
    assert md.count("Abstract") == 1  # LaTeXML's own abstract heading is replaced, not repeated
    assert any("does not support" in w for w in info["warnings"])  # arXiv's unsupported-package alert

    r = ingest_markdown(md, "arxiv_html")
    kinds = {s.title: s.kind for s in r.sections}
    assert kinds["1 Introduction"] == "introduction" and kinds["2.1 Cosine schedules"] == "method"
    assert kinds["References"] == "references" and kinds["Appendix A Proofs"] == "appendix"
    assert r.abstract.startswith("We revisit noise schedules")
    assert [ref["text"][:6] for ref in r.references] == ["[1] J.", "[2] A."]


def test_conversion_failures_are_detected(monkeypatch):
    _, info = latexml_to_markdown(NO_HTML_PAGE)
    assert not info["ok"] and "conversion failed" in info["reason"]

    banner = ('<div class="ltx_ERROR">Conversion to HTML had a Fatal error and exited abruptly. This document may be '
              'truncated or damaged.</div>')
    md, info = latexml_to_markdown(PAGE.replace("<article class=\"ltx_document ltx_authors_1line\">",
                                                "<article class=\"ltx_document ltx_authors_1line\">" + banner))
    assert not info["ok"] and "Fatal error" in info["reason"]
    assert "Fatal error" not in md

    short = PAGE.replace(FILLER * 5, "Short.")
    _, info = latexml_to_markdown(short)
    assert not info["ok"] and "characters" in info["reason"]
    monkeypatch.setattr(extract, "HTML_MIN_CHARS", 100)
    assert latexml_to_markdown(short)[1]["ok"]

    _, info = latexml_to_markdown("<html><body><p>" + FILLER * 30 + "</p></body></html>")  # not LaTeXML at all
    assert not info["ok"] and "LaTeXML" in info["reason"]


def _paper_pdf(path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    # Base-14 fonts cannot encode the "ﬁ" ligature (it comes out as "·"), so the body uses plain "fi".
    body = "The finite horizon analysis covers every schedule that this paper considers in detail. "
    for number, title in enumerate(("Introduction", "Method", "Experiments"), start=1):
        page = doc.new_page()
        if number == 1:
            page.insert_text((72, 60), "Schedules for Diffusion Models", fontsize=18, fontname="hebo")
        page.insert_text((72, 100), f"{number} {title}", fontsize=12, fontname="hebo")
        y = 130
        for _ in range(10):
            page.insert_text((72, y), body, fontsize=10)
            y += 14
        page.insert_text((72, y + 14), f"Closing words of page {number}.", fontsize=10)
    doc.save(path)
    return path


def test_ingest_pdf_max_pages(tmp_path):
    pdf = _paper_pdf(tmp_path / "paper.pdf")
    full = ingest_pdf(pdf)
    assert full.page_count == 3 and "<!-- page 3 -->" in full.text_md and "3 Experiments" in full.text_md
    assert not any("pages were extracted" in w for w in full.warnings)
    cut = ingest_pdf(pdf, max_pages=2)
    assert cut.page_count == 3  # the true page count, not the number read
    assert "<!-- page 2 -->" in cut.text_md and "<!-- page 3 -->" not in cut.text_md
    assert "3 Experiments" not in cut.text_md
    assert cut.warnings[0] == "only the first 2 of 3 pages were extracted"
    assert ingest_pdf(pdf, max_pages=3).text_md == full.text_md


def test_run_extraction_pdf_and_page_limit(tmp_path):
    pdf = _paper_pdf(tmp_path / "prior work.pdf")  # a space in the path, like this repository's
    r = run_extraction(pdf, "pdf")
    assert set(r) == RESULT_KEYS
    assert (r["status"], r["reason"], r["source_format"], r["page_count"]) == ("ok", None, "pdf", 3), r
    assert r["title"] == "Schedules for Diffusion Models"
    assert "<!-- page 2 -->" in r["text_md"] and "The finite horizon analysis" in r["text_md"]
    method = next(s for s in r["sections"] if s["title"] == "2 Method")
    assert method["page_start"] == 2 and r["text_md"][method["start"]:].startswith("## 2 Method")

    cut = run_extraction(pdf, "pdf", max_pages=2)
    assert cut["status"] == "ok" and cut["page_count"] == 3
    assert "<!-- page 2 -->" in cut["text_md"] and "<!-- page 3 -->" not in cut["text_md"]
    assert "only the first 2 of 3 pages were extracted" in cut["warnings"]


def test_run_extraction_html(tmp_path):
    page = tmp_path / "2401.00001v1.html"
    page.write_text(PAGE, encoding="utf-8")
    r = run_extraction(page, "html")
    assert (r["status"], r["source_format"], r["page_count"]) == ("ok", "arxiv_html", None), r
    assert r["title"] == "Diffusion Schedules Revisited"
    sec = next(s for s in r["sections"] if s["title"] == "2.1 Cosine schedules")
    assert (sec["kind"], sec["level"]) == ("method", 2)
    body = r["text_md"][sec["start"]:sec["end"]]
    assert body.startswith("### 2.1 Cosine schedules") and "<!-- anchor S2.SS1.p1 -->" in body

    missing = tmp_path / "no-html.html"
    missing.write_text(NO_HTML_PAGE, encoding="utf-8")
    r = run_extraction(missing, "html")
    assert (r["status"], r["reason"], r["text_md"]) == ("error", "conversion_failed", None) and r["detail"]


def test_run_extraction_encrypted_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Secret prior work.")
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    r = run_extraction(path, "pdf")
    assert (r["status"], r["reason"]) == ("error", "encrypted")
    assert r["text_md"] is None and r["sections"] == []


def test_run_extraction_survives_garbage(tmp_path):
    path = tmp_path / "garbage.pdf"
    path.write_bytes(random.Random(7).randbytes(8192))
    r = run_extraction(path, "pdf")
    assert r["status"] == "error" and r["reason"] in ("extraction_failed", "crashed") and r["detail"]


def test_run_extraction_timeout_returns_promptly(tmp_path):
    pdf = _paper_pdf(tmp_path / "paper.pdf")
    start = time.monotonic()
    r = run_extraction(pdf, "pdf", timeout_s=0.01)
    assert (r["status"], r["reason"]) == ("error", "timeout")
    assert time.monotonic() - start < 5


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is enforced reliably only on Linux")
def test_run_extraction_applies_memory_limit(tmp_path):
    pdf = _paper_pdf(tmp_path / "paper.pdf")
    r = run_extraction(pdf, "pdf", mem_mb=48)  # too little to load the interpreter and MuPDF
    assert r["status"] == "error" and r["reason"] in ("extraction_failed", "crashed"), r


@pytest.mark.skipif(os.name != "posix", reason="uses a shell script as a stand-in child")
@pytest.mark.parametrize("script, reason, detail", [
    ("echo boom >&2; exit 3", "crashed", "exited with code 3: boom"),
    ("kill -KILL $$", "crashed", "killed by SIGKILL"),
    ("exit 0", "crashed", "no valid result"),
])
def test_run_extraction_reports_child_failures(tmp_path, monkeypatch, script, reason, detail):
    child = tmp_path / "child.sh"
    child.write_text(f"#!/bin/sh\n{script}\n")
    child.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(child))
    src = tmp_path / "paper.html"
    src.write_text(PAGE, encoding="utf-8")
    r = run_extraction(src, "html")
    assert (r["status"], r["reason"]) == ("error", reason) and detail in r["detail"], r


@pytest.mark.skipif(os.name != "posix", reason="uses a shell script as a stand-in child")
def test_the_extraction_child_gets_no_credentials(tmp_path, monkeypatch):
    """The child parses hostile documents; nothing it could leak should be in its environment."""
    child = tmp_path / "child.sh"
    child.write_text('#!/bin/sh\nfor a in "$@"; do case "$a" in --out=*) out="${a#--out=}";; esac; done\n'
                     'printf \'{"status":"error","reason":"extraction_failed","detail":"%s|%s|%s"}\' '
                     '"${CLAUDE_CODE_OAUTH_TOKEN:-none}" "${S2_API_KEY:-none}" "${LC_ALL:-unset}" > "$out"\n')
    child.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(child))
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", FAKE_SECRETS["anthropic"])
    monkeypatch.setenv("S2_API_KEY", "s2secretvalue1234567890")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    src = tmp_path / "paper.html"
    src.write_text(PAGE, encoding="utf-8")
    r = run_extraction(src, "html")
    assert r["detail"] == "none|none|C.UTF-8", r
    assert "PATH" in extract._child_env() and "CLAUDE_CODE_OAUTH_TOKEN" not in extract._child_env()


@pytest.mark.skipif(not hasattr(signal, "SIGXCPU"), reason="POSIX only")
def test_cpu_limit_counts_as_timeout(tmp_path):
    r = extract._read_result(-signal.SIGXCPU, tmp_path / "result.json", tmp_path / "child.log", cpu_s=120)
    assert (r["status"], r["reason"]) == ("error", "timeout")


def test_extract_main_usage_errors_and_error_json(tmp_path):
    out = tmp_path / "result.json"
    assert extract_main(["--kind", "pdf", "--out", str(out)]) == 2  # no --in
    assert extract_main(["--in", "x.pdf", "--kind", "docx", "--out", str(out)]) == 2
    assert extract_main(["--in", "x.pdf", "--kind", "pdf", "--out", str(out), "--max-pages", "0"]) == 2
    assert not out.exists()
    # Problems with the document itself are reported in the JSON, not through the exit code.
    assert extract_main(["--in", str(tmp_path / "missing.pdf"), "--kind", "pdf", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == RESULT_KEYS and (data["status"], data["reason"]) == ("error", "extraction_failed")


def test_extract_main_reports_missing_text_layer(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page()  # no text at all, like a scan that was never OCR'd
    path = tmp_path / "scan.pdf"
    doc.save(path)
    out = tmp_path / "result.json"
    assert extract_main(["--in", str(path), "--kind", "pdf", "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["status"], data["reason"]) == ("error", "no_text_layer")


def test_run_extraction_argument_checks(tmp_path):
    with pytest.raises(ValueError):
        run_extraction(tmp_path / "x.pdf", "docx")
    r = run_extraction(tmp_path / "missing.pdf", "pdf")
    assert (r["status"], r["reason"]) == ("error", "extraction_failed") and "not found" in r["detail"]
