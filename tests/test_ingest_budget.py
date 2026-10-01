"""Ingestion of Markdown, text, LaTeX and PDF, plus the long-document strategy."""

import pytest

from conftest import FAKE_SECRETS
from paper_adversary.budget import BudgetError, plan_document
from paper_adversary.ingest import IngestError, ingest


def _kinds(result):
    return {s.title: s.kind for s in result.sections}


def test_markdown_sections_and_references(sample_paper):
    r = ingest(sample_paper)
    assert r.title.startswith("Guided Noise Editing")
    assert r.abstract and "Guided Noise Editing (GNE)" in r.abstract
    kinds = _kinds(r)
    assert kinds["1 Introduction"] == "introduction"
    assert kinds["2 Related Work"] == "related_work"
    assert kinds["3 Method"] == "method"
    assert kinds["3.1 Training objective"] == "method"  # inherits its parent's kind
    assert kinds["4 Theory"] == "theory"
    assert kinds["References"] == "references"
    assert kinds["A Proof of Theorem 1"] == "appendix"
    assert len(r.references) == 3
    assert r.submission_type == "paper"  # has sections and a bibliography


def test_plain_text_idea():
    text = ("Idea: contrastive noise editing\n\nWe want to learn a small network that edits diffusion noise so "
            "that protected attributes stay fixed. We would test it on a 2D maze benchmark.\n")
    r = ingest(paper_text=text)
    assert r.submission_type == "idea"
    assert r.source_format == "inline"


def test_latex_inputs_and_bibliography(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "intro.tex").write_text(r"\section{Introduction}We build on \cite{ho2020}. % comment" "\n")
    (tmp_path / "refs.bib").write_text(
        "@inproceedings{ho2020,\n title={Denoising Diffusion Probabilistic Models},\n author={Ho, Jonathan and "
        "Jain, Ajay},\n booktitle={NeurIPS},\n year={2020}\n}\n")
    (tmp_path / "main.tex").write_text(
        r"\documentclass{article}\title{A \textbf{Great} Paper}\begin{document}\maketitle"
        r"\begin{abstract}We do things.\end{abstract}\input{sections/intro}"
        r"\section{Method}$x = \sum_i y_i$ \subsection{Details} text \bibliography{refs}\end{document}")
    r = ingest(tmp_path / "main.tex")
    assert r.title == "A Great Paper"
    assert r.abstract == "We do things."
    kinds = _kinds(r)
    assert kinds["Introduction"] == "introduction" and kinds["Method"] == "method"
    assert "comment" not in r.text_md
    assert r"\sum_i y_i" in r.text_md  # math kept verbatim
    assert r.references and "Denoising Diffusion Probabilistic Models" in r.references[0]["text"]


def _make_pdf(path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    body = "This is body text of the paper that goes on for a while to set the body font size. " * 3
    pages = [
        [("A Study of Things", 18, False), ("Abstract", 11, True), ("We study things carefully.", 10, False),
         ("1 Introduction", 11, True), (body, 10, False)],
        [("2 Method", 11, True), (body, 10, False), ("3 Experiments", 11, True), (body, 10, False)],
        [("4 Conclusion", 11, True), (body, 10, False), ("References", 11, True),
         ("[1] A. Author. A cited paper. In NeurIPS, 2020.", 10, False),
         ("[2] B. Author. Another cited paper. In ICML, 2021.", 10, False)],
    ]
    for items in pages:
        page = doc.new_page()
        y = 72
        for text, size, bold in items:
            rect = pymupdf.Rect(72, y, 540, y + 120)
            page.insert_textbox(rect, text, fontsize=size, fontname="hebo" if bold else "helv")
            y += 30 if len(text) < 80 else 110
    doc.save(path)


def test_pdf_sections_pages_and_references(tmp_path):
    pdf = tmp_path / "paper.pdf"
    _make_pdf(pdf)
    r = ingest(pdf)
    assert r.source_format == "pdf" and r.page_count == 3
    titles = [s.title for s in r.sections]
    for expected in ("Abstract", "1 Introduction", "2 Method", "3 Experiments", "4 Conclusion", "References"):
        assert expected in titles, (expected, titles)
    assert "<!-- page 2 -->" in r.text_md
    method = next(s for s in r.sections if s.title == "2 Method")
    assert method.page_start == 2
    assert len(r.references) == 2
    assert r.title == "A Study of Things"


def _long_paper() -> str:
    filler = "Sentence about the method and its evaluation in detail. " * 120
    parts = ["# A Long Paper", "## Abstract", "We study a long problem with care.",
             "## 1 Introduction", filler, "## 2 Related Work", filler, "## 3 Method", "The key method idea. " + filler,
             "## 4 Experiments", filler, "## 5 Conclusion", filler, "## References", "[1] A. Ref. 2020.",
             "## A Appendix Proofs", filler * 3]
    return "\n\n".join(parts) + "\n"


def test_plan_full_and_sectioned():
    r = ingest(paper_text=_long_paper())
    full = plan_document(r.text_md, r.sections, budget_tokens=10**6, priorities=["abstract"], tokens_per_char=0.3)
    assert full.mode == "full" and full.text == r.text_md
    budget = int(len(r.text_md) * 0.3 * 0.4)
    plan = plan_document(r.text_md, r.sections, budget, ["title", "abstract", "method", "experiments"], 0.3)
    assert plan.mode == "sectioned"
    assert "We study a long problem with care." in plan.text  # abstract always kept
    assert "The key method idea." in plan.text  # highest-priority section kept inline
    assert "read_paper_section" in plan.text and plan.omitted
    assert plan.paper_tokens <= budget
    omitted = {o["title"] for o in plan.omitted}
    assert "A Appendix Proofs" in omitted and "3 Method" not in omitted
    # omitted sections are replaced in place, so document order is preserved
    assert plan.text.index("## 3 Method") < plan.text.index("A Appendix Proofs")


def test_plan_refuses_when_nothing_fits(sample_paper):
    r = ingest(sample_paper)
    with pytest.raises(BudgetError):
        plan_document(r.text_md, r.sections, budget_tokens=50, priorities=[], tokens_per_char=0.3)


def test_pdf_real_world_layout_quirks(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    body = "Body text that sets the dominant font size for this synthetic paper and fills the line. "
    page = doc.new_page()
    page.insert_text((20, 700), "arXiv:2401.00001v1 [cs.LG] 1 Jan 2024", fontsize=20, rotate=90)  # margin stamp
    page.insert_text((72, 80), "Real Title of the Paper", fontsize=18, fontname="hebo")
    page.insert_text((72, 120), "Abstract. We study something important and report it here in detail.", fontsize=10)
    page.insert_text((72, 160), "1", fontsize=12, fontname="hebo")  # number printed apart from its title
    page.insert_text((90, 160), "Introduction", fontsize=12, fontname="hebo")
    y = 180
    for _ in range(6):
        page.insert_text((72, y), body, fontsize=10)
        y += 14
    page.insert_text((72, y + 10), "2 Components", fontsize=7, fontname="hebo")  # small bold figure label
    page.insert_text((72, y + 40), "2 Method", fontsize=12, fontname="hebo")
    for _ in range(6):
        page.insert_text((72, y + 60), body, fontsize=10)
        y += 14
    page2 = doc.new_page()
    page2.insert_text((72, 80), "References", fontsize=12, fontname="hebo")
    refs = [("Ho, J. and Salimans, T. Classifier-free diffusion", "guidance. arXiv preprint, 2022."),
            ("Ho, J., Jain, A., and Abbeel, P. Denoising diffusion", "probabilistic models. In NeurIPS, 2020."),
            ("Song, Y. and Ermon, S. Generative modeling by", "estimating gradients. In NeurIPS, 2019.")]
    y = 100
    for first, cont in refs:  # author-year style with a hanging indent, as in ICML's template
        page2.insert_text((72, y), first, fontsize=10)
        page2.insert_text((82, y + 12), cont, fontsize=10)
        y += 30
    path = tmp_path / "quirks.pdf"
    doc.save(path)
    r = ingest(path)
    assert r.title == "Real Title of the Paper"
    assert "arXiv:2401" not in r.text_md
    titles = [s.title for s in r.sections]
    assert "1 Introduction" in titles and "2 Method" in titles and "2 Components" not in titles
    assert r.abstract and r.abstract.startswith("We study something important")
    assert [x["text"][:10] for x in r.references] == ["Ho, J. and", "Ho, J., Ja", "Song, Y. a"]


def test_ingest_refuses_hidden_files_bare_files_and_credentials(tmp_path):
    hidden = tmp_path / ".config" / "paper.md"
    hidden.parent.mkdir()
    hidden.write_text("# A paper\n\nText.\n")
    with pytest.raises(IngestError, match="hidden"):
        ingest(hidden)
    bare = tmp_path / "credentials"  # no extension: never read as text
    bare.write_text("# Notes\n\nsome text\n")
    with pytest.raises(IngestError, match="unsupported file type"):
        ingest(bare)
    leaky = tmp_path / "draft.md"
    leaky.write_text("# Draft\n\nSetup: S2_API_KEY=" + FAKE_SECRETS["s2"] + "\n" + "Body text. " * 50)
    with pytest.raises(IngestError, match="credentials"):
        ingest(leaky)
    placeholder = tmp_path / "tooling.md"  # a paper about LLM tooling may print placeholders
    placeholder.write_text("# Agents\n\nRun `export ANTHROPIC_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXXX`.\n"
                           + "Body text. " * 50)
    assert ingest(placeholder).text_md
