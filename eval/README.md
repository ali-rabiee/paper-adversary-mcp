# Offline evaluation of presets

This folder holds the format for evaluating review presets (`routine`, `milestone`, `legacy_thorough`) on
cases a person has adjudicated. Nothing here has been measured yet. The cheaper presets were chosen for cost,
and no claim that they find as much, or cost as little as expected, holds until it has been measured on real
cases.

## Cases

A case is a folder, `eval/cases/<id>/`, made from `eval/case_template/`:

- `case.yaml` gives the category, status, submission, evidence pack and ground truth.
- The submission is the paper or idea under review.
- `evidence/` holds the full texts of the prior papers both presets may read (the fixed evidence pack).

The **ground truth** is written by a person who read the submission and the evidence. It has two lists:
- `serious_issues`: what a good review must raise. Each is a prior-work overlap, naming the anticipating
  paper by arXiv ID or DOI, or a technical problem, with its location.
- `not_fatal`: accusations that would be wrong as FATAL, such as a similar-looking but materially different
  paper.

Never generate ground truth with a model, and never copy a review's output into it.

`status` says what a case is worth:

| status | runnable | evidence about quality |
|---|---|---|
| `template` | no | no |
| `synthetic` | yes, for testing the harness (see `tests/fixtures/eval_synthetic/`) | never |
| `adjudicated` | yes | yes, as one data point; requires `adjudicated_by` and `adjudicated_on` |

`eval/cases.yaml` lists twelve slots, three per category:
- `genuine_overlap`: real prior-work overlap;
- `different_prior_work`: similar-looking but materially different prior work;
- `valid_objection`: a valid technical objection;
- `incorrect_objection`: a plausible but incorrect objection.

All slots are empty until real cases are supplied.

```bash
.venv/bin/python -m paper_adversary eval validate eval/cases/<id>
```

## Paired runs

```bash
.venv/bin/python -m paper_adversary eval prepare eval/cases/<id> --presets routine,legacy_thorough --mode reasoning
```

This creates one run per preset with identical inputs and starts nothing. Each run uses your plan when you
start it with the printed `paper-adversary resume <run> --through followup` command.

**Reasoning mode** compares reasoning on fixed evidence:
- **Same evidence:** every run gets the same frozen snapshot of the evidence pack, and the report checks that
  the snapshots are identical.
- **No search:** novelty refuters have no web search, literature search, downloads or reference lookups. Their
  prior-paper tools serve only the pack, which is listed in their prompt.
- **Why:** differences come from the presets, not from what a search happened to return.

**Discovery mode** keeps search on and seeds nothing. It measures whether a preset finds the right prior
work. Its results vary between runs and are never mixed with reasoning results.

## Report

```bash
.venv/bin/python -m paper_adversary eval report <pair dir> [--scores <filled manual_scores.yaml>]
```

The report writes `comparison.json` and `comparison.md` next to `pair.json` (under `runs/eval/`). For each run it
records:

- **Prior-work ground truth, scored deterministically:**
  - which issues were found (a FATAL or MAJOR BUT FIXABLE verdict on an objection citing the anticipating
    paper), and whether the evidence gate found them shown;
  - which issues were missed;
  - false FATAL accusations against a `not_fatal` paper.
- **Evidence support and citation correctness:**
  - quote-check results for decisive objections;
  - blind-verifier outcomes;
  - shown vs unverified prior-work verdicts;
  - citations outside the evidence pack.
- **Cost and effort:**
  - tokens (input, cache write, cache read, output);
  - the API-equivalent cost Claude Code reports;
  - wall-clock and agent time;
  - calls, failed calls, format repairs, coverage supplements and placement fixes.

Technical issues and incorrect objections need judgment. The report writes `manual_scores.yaml`, where the
adjudicator lists for each run the missed issues and the false FATAL accusations. `--scores` folds those in.

## Reading the results

- **One case is not evidence:** one adjudicated case is one data point. Compare across many cases and all
  four categories before concluding that a preset is better, worse, or cheaper.
- **Dollars are not plan usage:** the API-equivalent cost is what the same tokens would cost on the API. On a
  subscription nothing is billed per token, and plan limits are not dollars.
- **Saturation is not readiness:** a review that ends `ready_for_next_gate` has nothing serious left open in
  the review. That does not mean the research is ready (see the main README).
