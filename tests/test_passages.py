"""Quote verification against extracted source text: normalization, exact and fuzzy matching, locations."""

import json
import random
import time

from paper_adversary.passages import ACCEPT_APPROXIMATE, SourceText, describe_location, match_passage, normalize

PAPER = """<!-- page 1 -->
# Adaptive Noise Schedules

## 1 Introduction

Diffusion models generate samples by reversing a gradual noising process. Training them is expensive, and the
choice of the noise schedule has a large effect on sample quality.

<!-- page 2 -->
## 2 Results

We find that the proposed schedule does not converge faster than the cosine schedule on small datasets. However,
it improves sample quality by 10% on ImageNet while keeping the training cost unchanged.
"""

HTML = """<!-- anchor S3.p1 -->
We define the schedule $\\alpha_t$ as a monotone function of the step index and keep it fixed.

<!-- anchor S3.p2 -->
The editor is trained with a preservation loss that penalizes changes to protected attributes.
"""


def test_exact_quote_is_verified_with_offsets_canonical_and_context():
    m = match_passage("Training them is expensive, and the choice of the noise schedule has a large effect on sample "
                      "quality.", SourceText(PAPER))
    assert (m.status, m.score, m.coverage, m.flags, m.note, m.accepted) == ("verified", 1.0, 1.0, [], None, True)
    assert PAPER[m.start : m.end].startswith("Training them") and PAPER[m.start : m.end].endswith("sample quality")
    assert m.canonical == ("Training them is expensive, and the choice of the noise schedule has a large effect on "
                           "sample quality")
    # One sentence before; the next "sentence" is a heading, which is skipped.
    assert m.context == f"Diffusion models generate samples by reversing a gradual noising process. {m.canonical}."
    assert (m.location["page_start"], m.location["page_end"], m.location["label"]) == (1, 1, "p. 1")


def test_ligatures_match_plain_letters_both_ways():
    source = SourceText("The \ufb01nite-sample analysis is \ufb02exible enough to cover the ef\ufb01cient of\ufb02ine "
                        "setting.")
    m = match_passage("The finite-sample analysis is flexible enough to cover the efficient offline setting", source)
    assert m.status == "verified" and "\ufb01nite" in m.canonical  # the original text is reported
    plain = SourceText("The finite-sample analysis is flexible enough to cover the efficient offline setting.")
    m = match_passage("The \ufb01nite-sample analysis is \ufb02exible enough to cover the ef\ufb01cient", plain)
    assert m.status == "verified"


def test_hyphenation_across_line_and_page_breaks():
    text = ("<!-- page 2 -->\nWe introduce a contin-\nuous relaxation of the discrete objective that keeps the "
            "gradient well de-\n<!-- page 3 -->\nfined everywhere, and a contin- uous schedule on top of it.")
    source = SourceText(text)
    m = match_passage("We introduce a continuous relaxation of the discrete objective that keeps the gradient well "
                      "defined everywhere", source)
    assert m.status == "verified" and m.accepted
    assert (m.location["page_start"], m.location["page_end"]) == (2, 3)
    assert "<!--" not in m.canonical and m.canonical.endswith("well de- fined everywhere")
    assert match_passage("and a continuous schedule on top of it", source, min_words=1).status == "verified"


def test_smart_quotes_dashes_spaces_and_soft_hyphens():
    text = ("The authors call it the \u201cwarm\u2013start\u201d regime\u2014a phase in which the\u00a0loss plateaus "
            "and the opti\u00admizer\u2019s step size is reduced.")
    m = match_passage("the \"warm-start\" regime - a phase in which the loss plateaus and the optimizer's step size "
                      "is reduced", SourceText(text))
    assert m.status == "verified" and m.canonical.startswith("the \u201cwarm\u2013start\u201d regime\u2014a phase")
    assert normalize("\u201cWarm\u2013start\u201d\u00a0op\u00adtimizer\u2019s \ufb01nite <!-- page 3 --> \u2212x") \
        == "\"warm-start\" optimizer's finite -x"


def test_one_changed_word_in_twenty_is_approximate_and_accepted_only_on_request():
    source = SourceText("<!-- page 4 -->\nOur analysis shows that the adaptive schedule reaches the target loss "
                        "within a modest compute budget compared with the cosine baseline on both datasets.")
    quote = ("analysis shows that the adaptive schedule reaches the target loss within a reasonable compute budget "
             "compared with the cosine baseline")
    assert len(quote.split()) == 20
    m = match_passage(quote, source)
    assert m.status == "approximate" and m.score >= max(0.9, ACCEPT_APPROXIMATE)
    assert not m.accepted and m.accepted_at(ACCEPT_APPROXIMATE) and not m.accepted_at(0.96)
    assert m.coverage == 0.95 and "critical_difference" not in m.flags
    assert "a modest compute budget" in m.canonical and m.location["page_start"] == 4


def test_negation_flip_is_not_found_with_critical_difference():
    m = match_passage("We find that the proposed schedule does converge faster than the cosine schedule on small "
                      "datasets.", SourceText(PAPER))
    assert m.status == "not_found" and not m.accepted
    assert "critical_difference" in m.flags and "not" in m.note
    assert "does not converge" in m.canonical  # the near miss is still reported


def test_changed_number_is_not_found_with_critical_difference():
    m = match_passage("However, it improves sample quality by 12% on ImageNet while keeping the training cost "
                      "unchanged.", SourceText(PAPER))
    assert m.status == "not_found" and "critical_difference" in m.flags
    assert "12" in m.note and "10" in m.note


def test_cut_words_negating_prefixes_and_decimals_are_critical():
    source = SourceText("It is impossible to train the editor without the protected attributes, so we use a learning "
                        "rate of 1.5 in all runs, and the loss never diverges on any of these benchmarks.")
    # Each quote occurs letter for letter in the source, or differs only by a decimal point, yet must not pass.
    for quote in ("possible to train the editor without the protected attributes",
                  "so we use a learning rate of 15 in all runs",
                  "so we use a learning rate of 1",
                  "ever diverges on any of these benchmarks"):
        m = match_passage(quote, source)
        assert m.status == "not_found" and "critical_difference" in m.flags, quote
    m = match_passage("possible to train the editor without the protected attributes", source)
    assert m.canonical.startswith("impossible to train")  # the whole source word is reported
    m = match_passage("impossible to train the editor without the protected attributes",
                      SourceText("It is possible to train the editor without the protected attributes."))
    assert m.status == "not_found" and "critical_difference" in m.flags


def test_interleaved_footnote_line_is_not_a_number_difference():
    text = ("<!-- page 5 -->\nThe editor converges within a few hundred updates on each of the control tasks "
            "that we\n\n\u00b9See Appendix B.\n\n<!-- page 6 -->\ntried, including the long-horizon manipulation "
            "tasks with sparse rewards.")
    m = match_passage("The editor converges within a few hundred updates on each of the control tasks that we tried, "
                      "including the long-horizon manipulation tasks with sparse rewards", SourceText(text))
    assert m.status == "approximate" and m.accepted_at(ACCEPT_APPROXIMATE) and m.flags == []
    assert (m.location["page_start"], m.location["page_end"]) == (5, 6)


def test_dropped_citations_are_not_a_number_difference():
    source = SourceText("Diffusion models generate samples by reversing a gradual noising process (Ho et al., 2020) "
                        "that slowly destroys the structure of the training data [12, 15] over many small steps.")
    m = match_passage("Diffusion models generate samples by reversing a gradual noising process that slowly destroys "
                      "the structure of the training data over many small steps", source)
    assert m.status == "approximate" and m.flags == [] and "(Ho et al., 2020)" in m.canonical
    # A parenthetical that is not a citation still counts.
    m = match_passage("The bound holds for every input of the network we consider in this work", SourceText(
        "The bound holds (for n > 10) for every input of the network we consider in this work."))
    assert m.status == "not_found" and "critical_difference" in m.flags


def test_ellipsis_segments_in_order_are_found_but_never_exact_evidence():
    source = SourceText(PAPER)
    for gap in ("[...]", "\u2026", "...", "[\u2026]"):
        m = match_passage(f"Diffusion models generate samples by reversing a gradual noising process {gap} the choice "
                          "of the noise schedule has a large effect on sample quality", source)
        assert m.status == "approximate" and m.flags == ["ellipsis"] and m.coverage == 1.0
        assert not m.accepted and not m.accepted_at(0.0)
        assert m.canonical.startswith("Diffusion models") and m.canonical.endswith("sample quality")


def test_elided_text_is_inspected():
    # The ellipsis hides a negation, or a short segment ("always") that is not in the source.
    m = match_passage("an approach ... scale to high-dimensional inputs in practice",
                      SourceText("We use an approach that does not scale to high-dimensional inputs in practice."))
    assert m.status == "not_found" and m.flags == ["ellipsis", "critical_difference"] and "negation" in m.note
    m = match_passage("We find that our guidance schedule is ... always ... optimal",
                      SourceText("We find that our guidance schedule is never optimal on these tasks."))
    assert m.status == "not_found" and not m.accepted_at(0.0)
    # Short segments are matched exactly and in order, including one before the first long segment.
    m = match_passage("We ... find that our guidance schedule is ... optimal",
                      SourceText("We find that our guidance schedule is optimal on these tasks."))
    assert m.status == "approximate" and m.canonical == "We find that our guidance schedule is optimal"


def test_ellipsis_segments_out_of_order_are_not_found():
    m = match_passage("the choice of the noise schedule has a large effect on sample quality ... Diffusion models "
                      "generate samples by reversing a gradual noising process", SourceText(PAPER))
    assert m.status == "not_found" and m.flags == ["ellipsis"] and m.start is None


def test_short_quotes_are_flagged_and_never_accepted():
    source = SourceText(PAPER)
    m = match_passage("the noise schedule has", source)
    assert m.status == "verified" and m.flags == ["too_short"] and not m.accepted
    m = match_passage("Training them is expensive, and the choice of the noise schedule", source, max_words=5)
    assert m.flags == ["too_long"] and m.accepted


def test_latex_math_matches_its_own_rendering_exactly_and_extracted_symbols_approximately():
    latex = SourceText("In our method $\\alpha_t$ controls the schedule of the noise level at each step.")
    extracted = SourceText("In our method αt controls the schedule of the noise level at each step.")
    for quote in ("$\\alpha_t$ controls the schedule of the noise level",
                  "α_t controls the schedule of the noise level"):
        m = match_passage(quote, latex)
        assert m.status == "verified" and m.canonical == "$\\alpha_t$ controls the schedule of the noise level"
    # "αt" is one word and "$\alpha_t$" two, so across renderings the match is approximate, not exact.
    m = match_passage("αt controls the schedule of the noise level", latex)
    assert m.status == "approximate" and m.canonical == "$\\alpha_t$ controls the schedule of the noise level"
    m = match_passage("$\\alpha_t$ controls the schedule of the noise level", extracted)
    assert m.status == "approximate" and m.canonical == "αt controls the schedule of the noise level"
    assert normalize("$\\mathbf{x}_t + \\varepsilon \\Gamma$") == "x_t + ε γ"


def test_exact_pass_requires_the_same_words():
    cases = [("We set the learning rate to $10^{-3}$ for all of the experiments in this work.",
              "We set the learning rate to 103 for all of the experiments"),
             ("We set the learning rate to $10^5$ for all of the experiments in this work.",
              "We set the learning rate to 105 for all of the experiments"),
             ("We use a mixing weight of 1/2 for both of the loss terms in the objective.",
              "We use a mixing weight of 12 for both of the loss terms"),
             ("This is atypical behaviour for a diffusion model trained on the benchmark.",
              "This is a typical behaviour for a diffusion model trained on the benchmark"),
             ("The model is trained for 15 epochs on each of the datasets we consider.",
              "The model is trained for 1 5 epochs on each of the datasets")]
    for text, quote in cases:
        m = match_passage(quote, SourceText(text))
        assert m.status == "not_found" and "critical_difference" in m.flags and not m.accepted, quote
    # Only a hyphen at a line or page break joins two source words into one.
    m = match_passage("We use selfattention in every layer of the network",
                      SourceText("We use self-attention in every layer of the network we train."))
    assert m.status == "approximate"
    m = match_passage("We use selfattention in every layer of the network",
                      SourceText("We use self-\n<!-- page 2 -->\nattention in every layer of the network we train."))
    assert m.status == "verified"


def test_acceptance_is_exact_only_by_default():
    source = SourceText("To train the editor we minimize the expected reconstruction error of the decoder over all "
                        "tasks.")
    m = match_passage("To train the editor we maximize the expected reconstruction error of the decoder over all "
                      "tasks", source)
    assert m.status == "approximate" and m.score >= ACCEPT_APPROXIMATE and not m.accepted
    assert m.accepted_at(ACCEPT_APPROXIMATE)  # an antonym swap: why approximate acceptance is opt-in
    m = match_passage("To train the editor we minimize the expected reconstruction error", source)
    assert m.accepted and m.accepted_at(1.1)


def test_quote_spanning_two_pages():
    text = ("<!-- page 3 -->\nThe ablation removes the guidance term and retrains the editor from scratch with the "
            "same\n<!-- page 4 -->\nbudget, which isolates the effect of the preservation loss.")
    source = SourceText(text)
    m = match_passage("retrains the editor from scratch with the same budget, which isolates the effect", source)
    assert m.status == "verified"
    assert (m.location["page_start"], m.location["page_end"]) == (3, 4) and "pp. 3\u20134" in m.location["label"]
    assert describe_location(source, m.start, m.end) == m.location


def test_claimed_page_mismatch_is_flagged_without_changing_status():
    source = SourceText(PAPER)
    quote = "We find that the proposed schedule does not converge faster than the cosine schedule"
    m = match_passage(quote, source, claimed_location="p. 9")
    assert m.status == "verified" and m.location["page_start"] == 2 and m.flags == ["location_mismatch"]
    for claim in ("p. 2", "pp. 1-2", "page 2", "Sec. 2, p.2"):
        assert match_passage(quote, source, claimed_location=claim).flags == []


def test_anchor_is_reported_and_checked():
    source = SourceText(HTML)
    quote = "The editor is trained with a preservation loss that penalizes changes to protected attributes"
    m = match_passage(quote, source, claimed_location="#S3.p2")
    assert m.location["anchor"] == "S3.p2" and m.location["label"] == "¶ S3.p2" and m.flags == []
    assert match_passage(quote, source, claimed_location="#S3.p1").flags == ["location_mismatch"]
    assert match_passage(quote, source, claimed_location="¶ S3").flags == []
    m = match_passage("the schedule $\\alpha_t$ as a monotone function of the step index", source)
    assert m.status == "verified" and m.location["anchor"] == "S3.p1"


def test_section_label_prefers_the_deepest_containing_section():
    text = ("<!-- page 5 -->\n## 3 Method\n\nWe describe the editor and its training procedure in this section.\n\n"
            "### 3.2 Schedules\n\nThe cosine schedule spends many steps at low noise levels for our editor.\n\n"
            "### 3.3 Training\n\nThe editor is trained for two days on a single GPU with mixed precision.\n")
    s3, s32, s33 = text.index("## 3 Method"), text.index("### 3.2"), text.index("### 3.3")
    sections = [{"id": "S03", "title": "3 Method", "start": s3, "end": len(text)},
                {"id": "S04", "title": "3.2 Schedules", "start": s32, "end": s33},
                {"id": "S05", "title": "3.3 Training", "start": s33, "end": len(text)}]
    source = SourceText(text, sections)
    quote = "The cosine schedule spends many steps at low noise levels"
    m = match_passage(quote, source, claimed_location="Sec. 3.2")
    assert m.location["label"] == "S04 3.2 Schedules, p. 5" and m.flags == []
    assert match_passage(quote, source, claimed_location="Section 4").flags == ["location_mismatch"]
    assert match_passage(quote, source, claimed_location="§3").flags == []
    assert match_passage("We describe the editor and its training procedure", source).location["section_id"] == "S03"
    m = match_passage("The editor is trained for two days on a single GPU", source)
    assert m.location["section_title"] == "3.3 Training"


def test_repeated_passage_reports_first_occurrence_and_count():
    sentence = "The editor keeps protected attributes fixed while changing the requested ones."
    m = match_passage(sentence, SourceText(f"<!-- page 1 -->\n{sentence}\n\n<!-- page 2 -->\n{sentence}\n"))
    assert m.status == "verified" and m.note == "occurs 2 times" and m.location["page_start"] == 1


def test_unrelated_quote_is_not_found_without_span():
    m = match_passage("A completely unrelated sentence about the weather in a remote mountain village",
                      SourceText(PAPER))
    assert (m.status, m.start, m.end, m.location, m.context) == ("not_found", None, None, None, None)
    assert m.note == "no similar passage found"


def test_large_source_with_altered_quote_is_fast():
    rng = random.Random(1234)
    vocab = [a + b + c for a in ("ka", "lo", "mi", "nu", "pe", "ro", "su", "ta", "vo", "ze")
             for b in ("bar", "dor", "fin", "gal", "hem") for c in ("", "s", "ed", "ing")]
    words = [rng.choice(vocab) for _ in range(190_000)]
    text = "\n\n".join(f"<!-- page {k // 500 + 1} -->\n" + " ".join(words[k : k + 500])
                       for k in range(0, len(words), 500))
    assert len(text) > 1_400_000
    quote = words[95_123 : 95_123 + 60]
    quote[12], quote[41] = "zzzalpha", "zzzbeta"
    t0 = time.perf_counter()
    m = match_passage(" ".join(quote), SourceText(text))  # includes preparing the source
    assert time.perf_counter() - t0 < 2.0
    assert m.status == "approximate" and m.accepted_at(ACCEPT_APPROXIMATE)
    assert m.location["page_start"] == 95_123 // 500 + 1


def test_adversarial_input_stays_fast_and_long_quotes_are_refused():
    # Unclosed comment openers, long runs of spaces, quote marks and math delimiters.
    text = (" ".join(f"<!-- w{i}" for i in range(20_000)) + " " * 100_000 + "\u201c\u201d\"'" * 20_000
            + "$\\{}" * 20_000 + " The final sentence of this adversarial source is quoted below in the test.")
    quote = " ".join(f"word{i}" for i in range(1000))
    t0 = time.perf_counter()
    source = SourceText(text)
    m = match_passage(quote, source)
    assert m.status == "not_found" and m.flags == ["too_long"] and m.start is None
    m2 = match_passage("<!-- w19998 <!-- w19999 The final sentence of this adversarial source is quoted", source)
    assert time.perf_counter() - t0 < 1.0
    assert m2.status == "verified"
    assert match_passage(quote, source, max_words=5000).flags == ["too_long"]  # the hard cap is not configurable


def test_to_dict_is_json_serializable():
    source = SourceText(PAPER, [{"id": "S01", "title": "1 Introduction", "start": 0, "end": len(PAPER)}])
    for quote in ("Training them is expensive, and the choice of the noise schedule",
                  "We find that the proposed schedule does converge faster than the cosine schedule",
                  "nothing like this sentence appears anywhere in the source text at all"):
        d = match_passage(quote, source, claimed_location="p. 3").to_dict()
        assert json.loads(json.dumps(d)) == d and {"status", "accepted", "location", "flags"} <= set(d)
