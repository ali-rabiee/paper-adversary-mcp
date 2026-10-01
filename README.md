# paper-adversary-mcp

A local MCP server for Claude Desktop that stress-tests a research paper or research idea before submission. It runs an adversarial review with several independent agents, keeps them isolated from each other in code, and lets a report reach later stages only once it passes deterministic checks.

```
                ┌─ N1..N4  novelty refuters      (web + scholarly search + verbatim full texts)
paper ──────────┼─ R1..R3  rigor refuters        (+ the PDF, for equations/tables)
  │             └─ F1..F3  fit / feasibility refuters
  └─ intake (claims ledger)          │  refuters never see each other; every output is gated
                                     ▼
       quote checks (deterministic)  +  V1..Vk blind verifiers (one per cited prior paper;
                                     │                          never see the refuter's argument)
                                     ▼
                    J1 J2 J3  judges (all refuter reports, evidence, verifications; never each other)
                                     ▼
                    judgment matrix + evidence gate (computed, not averaged)
                                     ▼
                    S1  synthesis memo (everything; unverified threats kept apart)
                                     ▼
                    C1  completeness critic (everything + memo; typed items)
                                     ▼
        follow-up round r: verify items → A1 A2 adjudicators (blind to each other)
                           → S<r+1> revised memo → C<r+1> re-check critic → stop or next round
```

Agents run as headless Claude Code processes (`claude -p`) on **your Claude plan**. No Anthropic API key is used. A guard aborts any agent that would bill an API key.

## Setup

1. Python environment (already created if you are reading this in the repo):

   ```bash
   cd "paper-adversary-mcp"
   python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
   ```

2. Give headless Claude Code a login that lasts. Background `claude -p` processes cannot use Claude Desktop's own login, so create a long-lived token for your plan:

   ```bash
   claude setup-token
   ```

   Put it in `paper-adversary-mcp/.env` (gitignored; see `.env.example`):

   ```
   CLAUDE_CODE_OAUTH_TOKEN=...
   ```

   Running `claude auth login` in a terminal also works, but that login can expire.

3. Register the server in Claude Desktop. `.venv/bin/python -m paper_adversary desktop-config` prints the exact entry for this checkout, and `--install` merges it into Claude Desktop's config, keeping a backup. The config lives at `~/.config/Claude/claude_desktop_config.json` on Linux, `~/Library/Application Support/Claude/` on macOS, and `%APPDATA%\Claude\` on Windows. The entry looks like:

   ```json
   {
     "mcpServers": {
       "paper-adversary": {
         "command": "/ABSOLUTE/PATH/paper-adversary-mcp/.venv/bin/python",
         "args": ["-m", "paper_adversary", "serve"],
         "env": {"PAPER_ADVERSARY_HOME": "/ABSOLUTE/PATH/paper-adversary-mcp"}
       }
     }
   }
   ```

   It uses the venv's `python -m paper_adversary`, not a shebang script: a path containing a space (like `ICML 2027`) breaks shebang lines. Restart Claude Desktop afterwards.

4. Check everything, including one tiny low-effort call per model and tool setup on your plan:

   ```bash
   .venv/bin/python -m paper_adversary validate --probe
   ```

## Using it from Claude Desktop

Ask in plain language, for example "Run a full adversarial review of /path/paper.pdf for ICML 2027". Or call the tools directly:

| Tool | What it does |
|---|---|
| `create_review_run` | Ingest a paper (PDF, Markdown, LaTeX `.tex`, text) or pasted idea; record metadata; plan agents. Starts nothing. |
| `run_refuters` | Run novelty, rigor and fit refuters (`phase`: all, novelty, rigor or fit) in a background worker. |
| `run_judges` | Blind verification of the decisive novelty objections (once), then the judges. Requires the refuters to be complete (or `allow_incomplete_refuters`). |
| `run_synthesis` | The synthesis memo. |
| `run_completeness_critic` | The critic that hunts for what everyone missed. |
| `run_followup` | One or more follow-up rounds on the critic's items (verify, adjudicate, revise the memo, re-check). `dry_run=true` shows what the next round would do. |
| `run_full_review` | Create the run and execute everything, follow-up rounds included, in one background worker. |
| `resume_run` | Continue an interrupted or partial run (`through="followup"` includes an unfinished follow-up round); completed agents are never redone. |
| `run_verification` | Re-check the evidence gate's unconfirmed verdicts with blind verifiers (e.g. after `add_prior_fulltext`), or check your own (prior paper, claim) pairs. |
| `add_prior_fulltext` | Supply a PDF/HTML of a prior paper the server could not download (e.g. paywalled). |
| `release_quarantine` | Make a quarantined output usable again, with a recorded reason (isolation failures need the CLI). |
| `get_run_status` | Per-agent status, gates and quarantines, failed calls, retry counts, isolation audit, verification, follow-up rounds, stale outputs, artifact paths. `wait_seconds` blocks up to 10 min with progress updates. |
| `get_report` | One output at a time: `memo` (the current memo), `novelty`, `rigor`, `fit`, `verifier`, `judge`, `synthesis`, `critic`, `adjudicator`, `revision`, `recheck` (with `agent_id` for one report), `matrix`, `evidence_gate`, `evidence`, `verification`, `prior`, `gates`, `followup`, `items`, `followup_matrix` (with `round`), `claims`, `refcheck`, `paper`, `sections`, `references`, `usage`, `events`, `index`, or per-agent `context`, `transcript`, `prompt`, `search_log`. Long texts are paged. |
| `get_run_cost` | Tokens and cost by phase (follow-up rounds separately) and in total. |
| `cancel_run` | Stop a run's worker and every agent process it started. |
| `list_runs`, `validate_config` | Housekeeping. |

A full review takes tens of minutes to hours at high effort, so the `run_*` tools return immediately. The worker is a separate process. It keeps running if Claude Desktop quits, and `cancel_run` stops it.

The same operations work from a terminal:

```bash
.venv/bin/python -m paper_adversary review paper.pdf --venue "ICML 2027"   # detached, follow-up included
.venv/bin/python -m paper_adversary status <run_id>
.venv/bin/python -m paper_adversary report <run_id> memo
.venv/bin/python -m paper_adversary refuters <run_id> --rerun N2 --foreground
.venv/bin/python -m paper_adversary followup <run_id> --dry-run
.venv/bin/python -m paper_adversary verify <run_id>                         # re-check the evidence gate
.venv/bin/python -m paper_adversary release <run_id> R2 --reason "..."     # asks for confirmation
```

## Configuration

`config/default.yaml` holds the pipeline and `config/models.yaml` holds the models. The orchestration code never names a model. Roles refer to aliases (`fable-5.1`, `opus-5.5`), and `models.yaml` maps each alias to its model ID, documented limits, supported effort levels and API-equivalent prices. Defaults follow your split:

| Role | Model | Effort | Agents | Prompt |
|---|---|---|---|---|
| orchestrator (intake) | fable-5.1 | high | 1 | intake_v1 |
| novelty | opus-5.5 | high | 4 | novelty_v2 |
| rigor | fable-5.1 | max | 3 | rigor_v1 |
| fit | opus-5.5 | high | 3 | fit_v1 |
| verifier (blind) | opus-5.5 | high | ≤4 per batch, ≤8 per run | verifier_v1 |
| judge | fable-5.1 | max | 3 | judge_v2 |
| synthesis | fable-5.1 | max | 1 | synthesis_v3 |
| critic | fable-5.1 | xhigh | 1 | critic_v2 |
| follow-up: adjudicator | fable-5.1 | max | 2 per round | adjudicator_v1 |
| follow-up: revised memo | fable-5.1 | max | 1 per round | synthesis_revision_v2 |
| follow-up: re-check critic | fable-5.1 | xhigh | 1 per round | critic_recheck_v1 |
| format repair (rare) | opus-5.5 | low | as needed | repair_v1 |
| coverage supplement (rare) | the judge's or adjudicator's own | its own | as needed | coverage_supplement_v1 |
| memo placement fix (rare) | the memo's own | its own | as needed | placement_fix_v1 |

Override per run with `config_override` (an object, YAML text or a file path), e.g. `{"novelty": {"agents": 2}, "concurrency": {"max_parallel_agents": 5}}`. Set `$PAPER_ADVERSARY_CONFIG` for your own standing defaults. Unknown keys, unsupported efforts and unknown aliases are rejected before anything runs.

Per-run overrides come through MCP tool arguments, which a model chooses after reading untrusted paper text. So some keys can only be set in `config/default.yaml` or your `$PAPER_ADVERSARY_CONFIG` file:
- the provider's security keys: `allow_api_key`, `extra_args`, `env_passthrough`, `claude_binary`, `safe_mode`;
- every gate setting except `gates.repair.{model, effort, timeout_minutes}`;
- which hosts documents are downloaded from, and how politely: `search.fulltext.{host_policy, allow_hosts, max_pdf_mb, max_html_mb, arxiv_document_interval_seconds, host_interval_seconds}`;
- what counts as shown evidence: every `evidence.*` key, and turning the blind verifier off (`verifier.enabled: false`);
- follow-up rounds can be reduced per run (fewer rounds, fewer adjudicators, `auto_start`/`auto_continue` off), never increased.

Flags that would widen an agent's access (`--add-dir`, `--permission-mode`, `--settings`, `--mcp-config`, …) are rejected in `extra_args` in any config. Prompt, lens-set and rubric names are file stems only; a rubric given as a path must be a visible `.md`/`.txt` file, and its text is copied into the run.

Validation against what your plan actually serves happens before each run: the Claude Code login is checked for free, and a run on an API-key login is refused. Then a preflight probe sends one tiny low-effort request per distinct model and tool setup, cached for 24 h. It uses the same tools and MCP server as the real agents, so a broken tool setup stops the run before any real agent is spent. The probe also fails if another model answers or if Claude Code's output lacks the events the isolation audit needs.

Main knobs: `concurrency.max_parallel_agents` (default 3, plan-friendly), `retry.*`, `plan_limit.policy` (`wait` sleeps until a usage limit resets, then continues; `fail` stops), `search.providers`, `search.fulltext.*`, `evidence.require_independent_check`, `followup.*`, `gates.*` (including `gates.coverage_supplement` and `gates.placement_fix`), `lens_set`, `rubric`.

## Completion gates and quarantine

Every agent's output is saved first, then checked (`gates.py`). Only a passing gate (or a release you record) makes an agent `complete`, which is the only status later stages read. A crash or cancel while checking leaves the agent `gating`; a resume finishes the checks without rerunning the agent.

| Check | Outcome |
|---|---|
| Isolation audit failed (a file outside the sandbox was read, forbidden content in a tool result, another report's marker) or unverifiable (no transcript, no `init`/`result` event) | one automatic fresh rerun (old output archived; at most 2 per job), then quarantine |
| A call the sandbox refused | warning (often a sign of prompt injection in the paper) |
| Another model answered than configured | quarantine |
| Report cut at the output-token limit | quarantine, unless its final JSON block is intact |
| Structured block missing or invalid | free fixes first: strict parse, lenient parse, rebuild from the report's own headings (rigor, fit, judges). Intake and novelty reports get a format-repair call instead, validated so it can transcribe but never add content. Otherwise quarantine. The data's source is recorded. |
| A judge or adjudicator leaves objections (items) unclassified, or gives one two different severities | one targeted supplement call on its own model and effort, ruling on exactly those and seeing only the reports that raised them; its rulings are merged and its reasoning appended to the report. Still incomplete → quarantine. Repeats with one severity are merged for free; judges also get an index of every objection ID |
| Memo's last section missing | quarantine (a gap in the middle is a warning) |
| A memo lists an unverified prior-work verdict under "Criticisms that survived judging", or leaves it out of "Unverified threats" | one targeted call on the memo's own model rewrites only those two sections; the result is spliced in and checked again; still wrong → quarantine. Only the IDs an entry starts with count, so a passing mention does not trigger it |
| Revised memo: an item without a disposition, or an unverified prior-work item not filed as an unverified threat | quarantine |

Targeted calls (a supplement or a placement fix) are made only for an output that no other check blocks, at most once per output, and a resume never pays for one twice. Each is recorded in the gate sidecar, and the edited report keeps its previous version in the archive.

A quarantined output stays on disk but no later stage reads it: downstream prompts only see a label such as `R2 (quarantined: isolation audit failed)`, the matrix lists it as excluded, and the isolation guard forbids its content for every role. Later stages wait for it as for a failure (or proceed with `allow_incomplete`). Rerun it, or release it with a recorded reason: `release_quarantine` for quality reasons, the terminal (`paper-adversary release`) for isolation reasons, so a prompt-injected paper cannot release a contamination. `get_report(run_id, "gates", agent_id)` shows every check.

## Evidence checks and the evidence gate

Bibliographic reference checks (does a cited paper exist?) are not enough: a refuter can cite a real paper for something it does not say. So decisive novelty objections must be shown with verbatim passages:

- **Verbatim tools.** Novelty refuters read prior papers through `read_prior_paper`, `find_in_prior_paper` and `check_quote`, which return the paper's own text. Claude Code's web fetch returns a model-written summary, so `novelty_v2` forbids quoting from it (or from memory).
- **Decisive objections** (already done, partially anticipated, framing exists, gap not real, concurrent work; or incremental/integrity with references; severity fatal or major) must give `overlap_evidence`: a verbatim prior passage and the verbatim submission passage it bears on, with locations.
- **Quote check** (`passages.py`, `evidence.py`). After a novelty refuter completes, every passage is matched against the prior paper's full text and the submission. Ligatures, hyphenation, page breaks and typography are normalized, and a changed negation, number or quantifier never counts as a match. By default only word-for-word matches are accepted. Near-verbatim matches and quotes with an ellipsis are reported, with the text actually found, but not accepted; set `evidence.accept_approximate_score` (at most 1) in your standing config to accept near misses above that score. A prior passage taken from the cited paper's reference list does not count. Nor does a submission passage from the submission's own related-work or reference section, which describes the prior work rather than the claim. Results: `novelty/N*.evidence.json`, shown to judges.
- **Identity checks.** If a reference's identifier resolves to a paper with another title, the result is a REFERENCE MISMATCH. If the "prior paper" is the submission itself (e.g. its own preprint), it is INVALID.
- **Blind verifiers** (`verification.py`). For each prior paper behind a decisive objection, a verifier gets the submission, the passages at stake and the prior paper's full text. It never sees the refuter's argument, category, severity or chosen passages; the isolation guard enforces this. It says whether the prior work anticipates each passage fully, partially or not at all, with its own quotes, which are checked too. An "anticipates" verdict becomes "cannot tell" in any of these cases:
  - its quotes do not check out;
  - its submission quote lies outside the passage it was shown;
  - its prior quote is shorter than 8 words or comes from the reference list.
- **Reruns.** Verdicts are tied to an objection's content, not its ID. When a refuter is rerun, its changed decisive objections are verified again (batch `refuters2`, …) before judges run.
- **Evidence gate** (`evidence_gate.py`). Judges keep the four severities (`judge_v2` adds an evidence-status field). A FATAL or MAJOR BUT FIXABLE verdict on prior work counts as shown only when a blind verifier found anticipation (fully for FATAL, at least partially for MAJOR) with verified quotes. Otherwise it is labelled (UNVERIFIED with the reason, DISPUTED, UNCLEAR, SCOPE, REFERENCE NOT FOUND, REFERENCE MISMATCH, INVALID), never downgraded. The independent check is required while the verifier is enabled; with it turned off in your standing config, accepted refuter quotes suffice. The memo (`synthesis_v3`) lists labelled verdicts under "6. Unverified threats — check before acting", not "Criticisms that survived judging", and starts every entry of those two sections with its IDs. A memo that misfiles one is sent back for those two sections (see the gates table). `get_report(run_id, "evidence_gate")`.
- **Closing a gap.** If a paper was unavailable, give the server its PDF with `add_prior_fulltext`, then `run_verification(run_id, from_gate=true)`. The gate is recomputed without rerunning the judges. The file must be identified by arXiv ID or DOI, its title must match the cited paper, and it must not be the submission. It never replaces a downloaded text, and it is stored in this run's `prior/` only, never in the machine-wide cache.

## Full-text sources and politeness

`search/fulltext.py` resolves a reference by arXiv ID, DOI or title through the scholarly APIs, then tries, in order:
1. the arXiv HTML rendering (LaTeXML: clean text, math as LaTeX, paragraph anchors);
2. the arXiv PDF;
3. open-access PDFs that OpenAlex or Semantic Scholar list, but only on allowed hosts: arXiv, OpenReview, PMLR, NeurIPS, ACL Anthology, JMLR, CVF, AAAI, IJCAI, PMC/Europe PMC and Semantic Scholar's PDF host, plus `allow_hosts`.

Download URLs come only from API metadata, never from an agent. Redirects are followed by hand and each hop is host-checked. Bodies are size-capped and type-checked. Documents are extracted in a child process with memory, CPU and time limits and no credentials in its environment (`search/extract.py`), so a malicious PDF cannot take the worker down. arXiv documents are fetched at most every 15 s machine-wide (its robots.txt crawl delay), and an arXiv block pauses arXiv for every agent and the search backend for 2 h. Results are cached machine-wide and snapshotted into the run's `prior/` folder, so every reader sees the same bytes. A paper with no open text is reported `fulltext_unavailable` with the reason; the abstract is never passed off as the full text.

## Follow-up rounds

`critic_v2` writes every finding as a typed item (new issue, minority critique, judging flaw, synthesis flaw, novelty to verify) with a location, severity estimate, and what would settle it. A follow-up round (`followup.py`):

1. **Triage** (deterministic, free): validates items (cited objection and judge IDs must exist), routes them (adjudicate; revision only for memo flaws; noted when below `min_severity`; invalid), and attaches the judges' verdicts on cited objections.
2. **Verification**: items naming suspected prior work go to blind verifiers. They see only the submission's passage and the prior paper, never the critic's prose. Identical checks reuse earlier verdicts.
3. **Adjudication**: two adjudicators, blind to each other, rule on every routed item with the judges' four severities and may re-rate base objections. A follow-up matrix flags contested items and rubber-stamping, and the evidence gate applies to prior-work items.
4. **Revised memo** `S<r+1>`: a full memo with a closing "What changed after the completeness critique" table. Every item gets a disposition (incorporated, rejected, needs evidence, unverified threat, noted, invalid). It becomes the current memo; earlier memos stay in place, recorded as superseded.
5. **Re-check critic** `C<r+1>`: a fresh critic on the revised memo. New items at `min_severity` or above start the next round automatically, up to `max_rounds` (default 2). Items it re-raises never start a round, since the adjudicators already ruled on them, but they stay open.

`run_full_review` starts round 1 by itself when the critic raises an item at `min_severity` or above (`followup.auto_start`).

**How a review ends.** A review that stops without new items is labelled `ready_for_next_gate` only when nothing serious is open. Otherwise it is `review_saturated_with_open_issues`, and the status lists the open issues. These block `ready`:
- a FATAL verdict from a judge or adjudicator that the adjudicators did not overturn (overturning takes every adjudicator of the latest round that re-rated it);
- an unverified FATAL prior-work threat;
- an item the re-check critic re-raised.

Open MAJOR BUT FIXABLE issues are listed but don't block. A review cut off at the round cap with new items is `max_rounds_reached`. The same label applies when the critic raises nothing to follow up. Each round's `round.json` records the open issues with the earlier rulings and dispositions of re-raised items.

**Reruns and restarts.**
- `run_followup(rerun_agents=[...])` reruns agents of the current round. The round reopens, and its finished agents are not rerun.
- Rerunning the base memo or critic makes the follow-up stale. The next follow-up job then:
  - archives the old rounds' records and outputs;
  - marks their agents superseded (the old state is kept under `followup_history`);
  - starts over with new IDs.
- A fresh round refuses to start if its critic reviewed an earlier version of the memo; rerun the critic first.
- IDs are never reused. A base run with two memos already has S2, so its first revised memo is S3.

## Isolation

This is enforced in software, in layers:

1. **Access policy** (`isolation.py`, `ROLE_VISIBILITY`). The context builder can read prior outputs only through a policy-checked reader:
   - refuters may read only the paper;
   - blind verifiers may read only the paper, their passages and their prior paper;
   - judges may read the paper, rubric, refuter reports and the orchestrator's checks, never another judge;
   - synthesis and the critic see everything upstream.

   The table lives in code, so no config can loosen it.
2. **Position rule.** An agent may read only artifacts from strictly earlier steps. So the two adjudicators never see each other, a rerun never sees what came after it, and a rerun of the base critic never sees the follow-up.
3. **Pre-flight guard.** Every report carries a marker, and distinctive word 8-grams of each report are registered. Before an agent starts, its fully assembled prompt is scanned for fingerprints of every artifact its role must not see, and for credentials (token patterns, and the values of the secrets the server holds). A hit aborts the agent. Documentation placeholders such as `sk-ant-api03-XXXX…` or AWS's `…EXAMPLE` key don't count, so papers about LLM tooling stay reviewable. The same scan runs when a paper is ingested; hidden files (`.env`, `~/.ssh/…`) and files without an extension are refused outright. Phrases that also occur in inputs the agent may legitimately see don't count as evidence, and neither do passages quoted from third-party papers.
4. **Process sandbox.** Each agent is its own `claude -p` process. It runs in a fresh, empty folder outside the project, with `--restricted` (no shell, file tools confined to that folder) and an explicit tool allowlist. It uses `--strict-mcp-config`, `--safe-mode` when no tools are needed, and `--no-session-persistence`, with a scrubbed environment. When the source is a PDF, rigor agents, judges and adjudicators get a copy of it (and nothing else) in their folder.
5. **Transcript audit and gates.** After each agent, its transcript is checked: file access outside its sandbox (paired with the tool's result, so refused attempts are told apart), forbidden fingerprints in tool results, and the events that prove the checks ran. A failed or unverifiable audit triggers one rerun, then quarantine (see above).
6. **Manifests.** Each agent's `*.context.json` lists every input it received, with its hash, and which planned inputs were missing. An agent is marked stale when an input changes, becomes quarantined, or becomes available after it ran.
7. **No side channels.** The intake pass's reading of the paper (title, field, claims) is stored separately and reaches only synthesis, the critic and the follow-up roles. Refuter and judge headers use only what you supplied or what was extracted from the file.

## Storage

```
runs/<run_id>/
  metadata.json  config.yaml  state.json
  source/     paper.pdf|md|tex, extracted_text.md, sections.json, references.json, rubric.md (per-run rubric),
              profile.md, claims_ledger.md
  novelty/    N1.md (+ N1.json structured data, N1.gate.json, N1.refcheck.json, N1.evidence.json,
              N1.search_log.jsonl, N1.context.json)
  rigor/  fit/
  prior/      text snapshots of the prior-work full texts read in this run, index.json
  verify/     V1.md (+ sidecars), tasks/T*.json (what each verifier saw), results.json, batches/
  judges/     J*.md, judgment_matrix.md/.json, evidence_gate.md/.json
  synthesis/  memo.md (S1), memo_S2.md, ...
  critic/     completeness.md (C1), completeness_C2.md, ...
  followup/   A1.md, A2.md, ...; round-<r>/ items.json, items.md, followup_matrix.md/.json, round.json
  logs/       events.jsonl, usage.jsonl, api_usage.json, worker.log, fingerprints.json,
              agents/<id>/attempt-<n>/{system_prompt.md, user_prompt.md, transcript.jsonl, stderr.log,
              context.json, repair-<k>/}
  archive/    superseded outputs (reruns never delete), index.jsonl
```

Every report starts with YAML front matter. It records:
- the model requested and the models that actually served it;
- effort, prompt version and its SHA-256, lens;
- timestamp, run ID, role and round;
- usage, attempts;
- the structured block as written, the isolation audit, and warnings.

Gate verdicts are in `*.gate.json`.

## Prompts and experiments

Prompts live in `prompts/` as `<role>_v<N>.md` (front matter + template). Lenses are in `lenses_v1.yaml` and rubrics in `rubrics/`. Versions are never edited in place: a change is a new file (as `novelty_v2`, `judge_v2`, `synthesis_v3`, `synthesis_revision_v2` and `critic_v2` are), selected in the config or per run, e.g. `config_override={"novelty": {"prompt": "novelty_v3"}}`. Each report records the prompt version and the file's hash, so in-place edits are detectable too. A prompt whose front matter says `structured_output: true` must end its output with a fenced JSON block.

## Long papers

If a paper does not fit an agent's budget, it is never silently truncated. The budget is the context window minus output reserve, safety margin and other inputs. The role's priority sections stay inline in document order, every other section becomes an explicit placeholder, and the agent gets a `read_paper_section` tool that returns any section in full. Verifiers split their budget between the submission and the prior paper and read omitted prior sections with `read_prior_paper`. The plan is recorded per agent. Token estimates start from a conservative heuristic and are then calibrated from the usage agents report.

## Novelty search

Novelty refuters get Claude Code's web search and fetch, plus a small MCP tool server (`tools_server.py`). It offers `search_literature`, `lookup_paper` and `find_citing_papers` over OpenAlex, Semantic Scholar, arXiv and Crossref, and the verbatim full-text tools above. Queries are fanned out and de-duplicated, cached on disk, rate-limited per provider across all agents, and protected by circuit breakers (arXiv blocks heavy users for hours, so it gets a slow gate). Add a provider by implementing `search.base.SearchBackend` and calling `search.register_backend(name, factory)`.

After each novelty refuter is complete, the orchestrator looks up every reference in its structured block by DOI, arXiv ID or title, and checks its quoted passages. A check interrupted by a cancel or crash is redone before the verifiers and judges start.

Optional: `S2_API_KEY` for a higher Semantic Scholar rate limit; `PAPER_ADVERSARY_CONTACT_EMAIL` for the OpenAlex/Crossref polite pool.

## Cost and usage

`get_run_status` also shows how much of your plan's 5-hour and 7-day windows is used, and whether paid overage is on, as Claude Code last reported. When a plan limit hits, the worker waits until the reset time Claude Code reports.

`get_run_cost` reports input, cache-write, cache-read and output tokens, web searches and agent time per phase: verifiers, format repairs, coverage supplements, memo placement fixes and each follow-up round are listed separately. It shows two cost figures: the API-equivalent cost Claude Code reports, and an estimate from `config/models.yaml` prices. On a plan neither is billed per token; they show how heavy a run was. Calls that returned no usage (for example, killed on timeout) are listed as missing, never estimated.

## Reliability

- Transient failures get exponential backoff with jitter: overload, 5xx, network, timeouts, silent streams, empty output.
- Plan usage limits are waited out until the reset time the CLI reports, then the run continues, within `plan_limit.max_wait_hours`.
- Auth failures and API-key billing stop the run immediately.
- A missing model, a substituted model or a broken tool setup is caught by the preflight probe.
- An oversized prompt is retried once in sectioned mode.
- Outputs are saved before they are checked; a crash during checks is finished on resume without rerunning the agent. Gate decisions are recorded before they are applied, so a crash in between applies the recorded decision.
- Parsing of agent output and of downloaded documents runs in linear time, so hostile text (thousands of unclosed fences or comments) cannot stall the worker.
- All state writes are atomic. One worker per run is enforced by an OS lock that dies with the process. `resume_run` retries failed or interrupted agents, finishes interrupted checks and follow-up rounds, and skips completed work.

## Tests

```bash
.venv/bin/python -m pytest
```

The suite uses no plan usage and no network. It runs the full pipeline with an offline mock provider whose reports carry canaries, proving from the outputs which agent saw what. It injects every gate failure: audit breaches, missing transcripts, bad JSON, truncation, substitution and coverage gaps. It runs blind verification, the evidence gate and follow-up rounds end to end, including the smoke run's regression case: a FATAL overlap claimed from an abstract comes out labelled unverified and filed under unverified threats.

It also covers:
- the quote matcher;
- full-text retrieval against canned HTTP;
- sandboxed extraction;
- the Claude Code backend against a fake `claude` binary;
- the scholarly APIs;
- detached workers with cancel and resume.

## Known limitations

- **Plan usage.** Your account is on the Pro plan. A full default run is 16 base agents (9 on Fable 5.1, 8 of them at max or xhigh effort, plus 7 on Opus 5.5). On top come up to 8 Opus verifiers and, per follow-up round, 4 Fable calls (2 adjudicators and the revised memo at max, the re-check at xhigh), each reading about as much as the completeness critic. Two automatic rounds can nearly double the Fable usage of a run. A judge, adjudicator or memo that needs a coverage supplement or placement fix costs one more call on its own model (smaller inputs than the agent itself). The worker waits through plan limits. Lower `agents`, `followup.max_rounds`, or set `followup.auto_start: false` per run if you need results sooner.
- **Paywalled prior work** cannot be verified until you supply the PDF (`add_prior_fulltext`). Until then, verdicts that rest on it stay labelled unverified.
- **Text that is hard to check.** Scanned PDFs without a text layer are not OCR'd. Quotes of mathematics match poorly; the prompts ask for the prose around an equation.
- **Copyright.** Run folders contain text snapshots of third-party papers (for personal research use, which arXiv's terms allow); do not redistribute them.
- **CLI version.** Opus 5.5 needs Claude Code 2.1.280 or newer. With an older CLI, the preflight stops the run with Claude Code's own "run `claude update`" message.
- **Web search** inside `claude -p` depends on Claude Code's WebSearch tool being available to your account.
