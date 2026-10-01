"""Completion gates as pure functions: parsing, recovery, schema rules, coverage, sections, repairs."""

import json

from paper_adversary import gates
from paper_adversary.prompts import load_prompt

REFUTER_REPORT = """# Report

## Objections

### O1: Statistics insufficient to support a 10% claim
A single seed.

### O2 — Missing ablation (of the schedule)
No ablation.

{block}
"""

OBJECTIONS = {"summary_verdict": "x", "objections": [
    {"id": "O1", "title": "Single seed, 10k-sample FID", "category": "statistics", "severity_estimate": "major",
     "confidence": "high"},
    {"id": "O2", "title": "Missing ablation", "category": "missing_ablation", "severity_estimate": "minor",
     "confidence": "medium"}]}


def fenced(data) -> str:
    return "```json\n" + json.dumps(data, indent=2) + "\n```"


def test_strict_uses_only_the_last_block():
    text = 'The paper shows ```json\n{"objections": []}\n``` in its appendix.\n\n```json\n{"objections": [\n```'
    data, err = gates.parse_json_strict(text)
    assert data is None and "invalid JSON" in err  # never falls back to the quoted snippet


def test_lenient_handles_trailing_commas_and_unfenced_objects():
    data, _ = gates.parse_json_lenient('```json\n{"a": [1, 2,], "b": {"c": "x, ]",},}\n```')
    assert data == {"a": [1, 2], "b": {"c": "x, ]"}}
    data, _ = gates.parse_json_lenient('Report text.\n\n{"objections": [{"id": "O1"}]}\n')
    assert data == {"objections": [{"id": "O1"}]}
    data, _ = gates.parse_json_lenient('{"q": "a \\" b", "r": 1,}')
    assert data == {"q": 'a " b', "r": 1}


def test_schema_hard_and_soft_rules():
    hard, soft = gates.validate_block("rigor", OBJECTIONS)
    assert hard == [] and soft == []
    hard, _ = gates.validate_block("rigor", {"objections": [{"id": "first"}, {"id": "O1"}, {"id": "O1"}]})
    assert any("not O<n>" in h for h in hard) and any("used twice" in h for h in hard)
    _, soft = gates.validate_block("rigor", {"objections": [{"id": "O1", "severity_estimate": "huge"}]})
    assert any("severity_estimate" in s for s in soft)
    hard, _ = gates.validate_block("judge", {"judgments": [{"objection_ids": ["N1-O1", "bogus"], "severity": "X"}]})
    assert any("BOGUS" in h for h in hard) and any("severity" in h for h in hard)
    hard, _ = gates.validate_block("intake", {"claims": [{"id": "C1"}]})
    assert hard


def test_ids_are_reconciled_with_headings_not_titles():
    # The real R1: the heading title and the JSON title differ; only IDs are compared.
    st = gates.parse_structured("rigor", REFUTER_REPORT.format(block=fenced(OBJECTIONS)), False)
    assert st.source == "ok" and st.warnings == []
    only_head = {"objections": [OBJECTIONS["objections"][0]]}
    st = gates.parse_structured("rigor", REFUTER_REPORT.format(block=fenced(only_head)), False)
    assert [o["id"] for o in st.data["objections"]] == ["O1", "O2"]
    assert st.data["objections"][1]["title"].startswith("Missing ablation") and st.warnings


def test_unusable_block_is_rebuilt_from_headings_or_sent_to_repair():
    text = REFUTER_REPORT.format(block="```json\n{\"objections\": [\n```")
    st = gates.parse_structured("rigor", text, False)
    assert st.source == "derived:headings" and [o["id"] for o in st.data["objections"]] == ["O1", "O2"]
    st = gates.parse_structured("novelty", text, True)
    assert st.data is None and st.repair_mode == "syntax" and st.broken_block
    st = gates.parse_structured("novelty", REFUTER_REPORT.format(block=""), True)
    assert st.repair_mode == "transcribe"
    st = gates.parse_structured("intake", "no block, no headings", False)
    assert not st.usable and gates.check_structured(st).result == gates.BLOCK


def test_judgment_headings_take_the_last_parenthesised_ids():
    text = ("### FATAL: Core method exists (as I recall) (N1-O1, N1-O2, R1-O1)\n\n"
            "### MAJOR BUT FIXABLE: Equal CLIP score unmeasured (N1-O4; F1-O1)\n"
            "### NOT CONVINCING: Speculation (F1-O3)\r\n")
    heads = gates.judgment_headings(text)
    assert heads[0]["objection_ids"] == ["N1-O1", "N1-O2", "R1-O1"] and heads[0]["severity"] == "FATAL"
    assert heads[1]["objection_ids"] == ["N1-O4", "F1-O1"] and heads[1]["severity"] == "MAJOR_FIXABLE"
    assert heads[2]["severity"] == "NOT_CONVINCING"
    data = {"judgments": [{"objection_ids": ["N1-O1"], "severity": "MINOR"}]}
    warnings = gates.compare_judgments(data, text)
    assert any("severities differ" in w for w in warnings) and any("not the JSON block" in w for w in warnings)


def test_coverage():
    expected = {"N1-O1", "N1-O2", "R1-O1", "R1-O2"}
    js = [{"objection_ids": ["N1-O1", "R1-O1"], "severity": "MAJOR"}, {"objection_ids": ["N1-O2"], "severity": "MINOR"},
          {"objection_ids": ["N1-O1", "X1-O9"], "severity": "MAJOR BUT FIXABLE"}]
    check = gates.check_coverage(js, expected, 0.5)
    assert check.result == gates.WARN
    assert check.details["missing"] == ["R1-O2"] and check.details["unknown"] == ["X1-O9"]
    assert check.details["repeated"] == ["N1-O1"] and not check.details["conflicts"]
    assert gates.check_coverage(js, expected, 1.0).result == gates.BLOCK  # the default: every objection
    assert gates.check_coverage([{"objection_ids": ["N1-O1"]}], expected, 0.5).result == gates.BLOCK
    two = js[:2] + [{"objection_ids": ["N1-O1"], "severity": "FATAL"}, {"objection_ids": ["R1-O2"], "severity": "MINOR"}]
    check = gates.check_coverage(two, expected, 1.0)  # complete, but one objection has two different verdicts
    assert check.result == gates.BLOCK and check.details["conflicts"] == {"N1-O1": ["MAJOR_FIXABLE", "FATAL"]}
    merged = gates.drop_ids(js, "objection_ids", {"N1-O1"}, keep_first=True)
    assert [e["objection_ids"] for e in merged] == [["N1-O1", "R1-O1"], ["N1-O2"], ["X1-O9"]]
    assert [e["objection_ids"] for e in gates.drop_ids(js, "objection_ids", {"N1-O1", "N1-O2"})] == [["R1-O1"],
                                                                                                      ["X1-O9"]]
    single = [{"objection_id": "n1-o1", "severity": "MINOR"}, {"objection_ids": ["N1-O1"], "severity": "MINOR"}]
    status = gates.coverage_status(single, "objection_ids", {"N1-O1"})
    assert status["repeated"] == ["N1-O1"] and not status["missing"]  # a single objection_id counts too
    assert gates.drop_ids(single, "objection_ids", {"N1-O1"}, keep_first=True) == [
        {"severity": "MINOR", "objection_ids": ["N1-O1"]}]


def test_placement_reads_entry_leads_and_splices_two_sections():
    required = [(5, "Criticisms that survived judging"), (6, "Unverified threats — check before acting"),
                (7, "Criticisms that were rejected")]
    memo = ("## 5. Criticisms that survived judging\n| Objection(s) | J1 | View |\n|---|---|---|\n"
            "| N1-O1, R1-O1: anticipated | FATAL | Shown; unlike N1-O2. |\n- **R1-O3** — single seed\n"
            "Prose that mentions N1-O2 in passing.\n\n## 6. Unverified threats — check before acting\n"
            "**N1-O2: interval guidance.**\n- Stakes: FATAL.\n\n## 7. Criticisms that were rejected\nNone.\n")
    assert gates.check_unverified_placement(memo, required, ["N1-O2"], True).result == gates.PASS
    check = gates.check_unverified_placement(memo, required, ["R1-O3", "N9-O1"], True)
    assert check.result == gates.BLOCK and check.details == {"misplaced": ["R1-O3"], "absent": ["N9-O1", "R1-O3"]}
    fix = ("## 5. Criticisms that survived judging\n- **N1-O1** — kept\n\n"
           "## 6. Unverified threats — check before acting\n- **R1-O3** — moved\n")
    fixed = gates.splice_sections(memo, fix, 5, 6)
    assert fixed.endswith("## 7. Criticisms that were rejected\nNone.\n") and "- **R1-O3** — moved" in fixed
    without_six = memo.replace(memo[memo.index("## 6."):memo.index("## 7.")], "")
    assert "## 6." in gates.splice_sections(without_six, fix, 5, 6)  # a missing section 6 is inserted
    assert gates.splice_sections(memo, "## 5. Criticisms that survived judging\nonly five\n", 5, 6) is None


def test_required_sections_come_from_the_prompt_version_used():
    assert len(gates.required_sections(load_prompt("synthesis_v1").body)) == 12
    assert len(gates.required_sections(load_prompt("critic_v1").body)) == 5
    required = gates.required_sections(load_prompt("synthesis_v1").body)
    memo = "\n".join(f"## {n}. {t}\n\nText.\n" for n, t in required)
    assert gates.check_sections(memo, required, "block_incomplete").result == gates.PASS
    cut = "\n".join(f"## {n}. {t}\n\nText.\n" for n, t in required[:9])
    assert gates.check_sections(cut, required, "block_incomplete").result == gates.BLOCK
    gap = "\n".join(f"## {n}. {t}\n\nText.\n" for n, t in required if n != 6)
    assert gates.check_sections(gap, required, "block_incomplete").result == gates.WARN
    assert gates.check_sections(gap, required, "block_any").result == gates.BLOCK
    assert gates.check_sections(cut, required, "warn").result == gates.WARN


def test_truncation_and_substitution():
    assert gates.check_truncation("end_turn", False, "block").result == gates.PASS
    assert gates.check_truncation("max_tokens", False, "block").result == gates.BLOCK
    assert gates.check_truncation("max_tokens", True, "block").result == gates.WARN
    assert gates.check_substitution("claude-fable-5-1", ["claude-fable-5-1[1m]"], "block").result == gates.PASS
    assert gates.check_substitution("claude-haiku-4-5", ["claude-haiku-4-5-20251001"], "block").result == gates.PASS
    assert gates.check_substitution("claude-fable-5-1", ["claude-opus-5-5"], "block").result == gates.BLOCK
    assert gates.check_substitution("claude-fable-5-1", ["claude-opus-5-5"], "warn").result == gates.WARN


def test_audit_check():
    assert gates.check_audit({"status": "fail", "findings": ["x"]}, "quarantine").result == gates.BLOCK
    assert gates.check_audit({"status": "unverifiable", "unverifiable": ["y"]}, "quarantine").result == gates.BLOCK
    assert gates.check_audit({"status": "unverifiable", "unverifiable": ["y"]}, "warn").result == gates.WARN
    assert gates.check_audit({"status": "pass", "denied": ["z"]}, "quarantine").result == gates.WARN


def test_repair_validation_rejects_invented_content():
    report = REFUTER_REPORT.format(block="") + 'Evidence: Vaswani et al., "Attention Is All You Need" (2017).'
    good = {"objections": [{"id": "O1", "title": "Statistics insufficient to support a 10% claim",
                            "references": [{"title": "Attention Is All You Need"}]},
                           {"id": "O2", "title": "Missing ablation (of the schedule)"}]}
    data, rejections, _ = gates.validate_repair("novelty", "transcribe", fenced(good), report, None)
    assert data is not None and rejections == []
    invented = json.loads(json.dumps(good))
    invented["objections"].append({"id": "O9", "title": "New"})
    assert gates.validate_repair("novelty", "transcribe", fenced(invented), report, None)[0] is None
    fake_ref = json.loads(json.dumps(good))
    fake_ref["objections"][0]["references"].append({"title": "A Paper That Does Not Exist"})
    data, rejections, _ = gates.validate_repair("novelty", "transcribe", fenced(fake_ref), report, None)
    assert data is None and any("does not appear" in r for r in rejections)
    reworded = json.loads(json.dumps(good))
    reworded["objections"][1]["title"] = "Ablations are missing"
    assert gates.validate_repair("novelty", "transcribe", fenced(reworded), report, None)[0] is None
    # syntax mode: every string must already be in the broken block
    broken = '{"claims": [{"id": "C1", "claim": "Mock claim", "location": "Abstract"},]'
    intake_report = "## Claims\n\n- C1: Mock claim (Abstract)\n\n```json\n" + broken + "\n```"
    ok = {"claims": [{"id": "C1", "claim": "Mock claim", "location": "Abstract"}]}
    assert gates.validate_repair("intake", "syntax", fenced(ok), intake_report, broken)[0] == ok
    added = {"claims": [{"id": "C1", "claim": "Mock claim, which is much stronger", "location": "Abstract"}]}
    assert gates.validate_repair("intake", "syntax", fenced(added), intake_report, broken)[0] is None


def test_unavailable_label_never_carries_content():
    entry = {"status": "quarantined", "gate": {"reasons": ["isolation audit failed: Read read /etc/passwd"]}}
    assert gates.unavailable_label("R2", entry).startswith("R2 (quarantined: isolation audit failed")
    assert gates.unavailable_label("F1", {"status": "failed"}) == "F1 (failed)"


def test_parsers_are_linear_on_hostile_output():
    """Agent output is untrusted: unclosed fences and endless headings must not stall the gate."""
    import time

    hostile = ["```json\n" * 20_000, "```\n{" * 20_000, "### O1: " + "x " * 100_000, "### FATAL: (" * 20_000,
               "## 1." + " " * 200_000, "### MAJOR: t (" + "O1, " * 50_000]
    for text in hostile:
        t0 = time.perf_counter()
        gates.parse_json_strict(text)
        gates.parse_json_lenient(text)
        gates.objection_headings(text)
        gates.judgment_headings(text)
        gates.numbered_sections(text)
        assert time.perf_counter() - t0 < 2.0, text[:20]


def test_fences_and_headings_keep_their_meaning():
    text = ("### O1: Real objection (with parentheses)\n```json\n### O9: inside a fence\n```\n"
            "### MAJOR: Weak baseline (see Sec. 4) (N1-O1, R2-O3)\n## 1. Summary\n## 2. Details\n"
            "```json\n{\"a\": 1}\n```\ntrailing\n```json\n{\"b\": 2}\n```\n")
    assert gates.objection_headings(text) == [{"id": "O1", "title": "Real objection (with parentheses)"}]
    j = gates.judgment_headings(text)
    assert j == [{"objection_ids": ["N1-O1", "R2-O3"], "title": "Weak baseline (see Sec. 4)",
                 "severity": "MAJOR_FIXABLE"}]
    assert gates.numbered_sections(text) == [(1, "Summary"), (2, "Details")]
    assert gates.parse_json_strict(text)[0] == {"b": 2}  # the last fenced block only
    unclosed = "```json\n{\"a\": 1}\n"
    assert gates.parse_json_strict(unclosed)[0] is None


def test_a_structured_warning_says_what_it_is():
    report = REFUTER_REPORT.format(block=fenced({"summary_verdict": "x", "objections": OBJECTIONS["objections"][:1]}))
    st = gates.parse_structured("rigor", report, False)
    check = gates.check_structured(st)
    assert check.result == gates.WARN and "O2" in check.message, check
