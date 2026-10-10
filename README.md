# paper-adversary-mcp

A local MCP server for Claude Desktop that stress-tests a research paper or research idea before submission. It runs an adversarial review with several independent agents, keeps them isolated from each other in code, and lets a report reach later stages only once it passes deterministic checks.

```
                ┌─ N1..   novelty refuters       (web + scholarly search + verbatim full texts)
paper ──────────┼─ R1..   rigor refuters         (+ the PDF, for equations/tables)
  │             └─ F1..   fit / feasibility refuters
  └─ intake (claims ledger)          │  refuters never see each other; every output is gated
                                     ▼
       quote checks (deterministic)  +  V1..Vk blind verifiers (one per cited prior paper, on the other
                                     │      model than the objection's author; never see its argument)
                                     ▼
                    J1 J2 ..  judges (all refuter reports, evidence, verifications; never each other)
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

Agents run as headless Claude Code processes (`claude -p`) on **your Claude plan**. No Anthropic API key is used. A guard aborts any agent that would bill an API key. How many agents run, on which model and at what effort comes from a **preset**: `routine` by default, a mixed Opus 5.5 / Fable 5.1 panel (see [Presets](#presets)).

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
| `create_review_run` | Ingest a paper (PDF, Markdown, LaTeX `.tex`, text) or pasted idea; record metadata; plan agents from a `preset` (`routine` by default, `milestone`, `legacy_thorough`). Starts nothing. |
| `run_refuters` | Run novelty, rigor and fit refuters (`phase`: all, novelty, rigor or fit) in a background worker. |
| `run_judges` | Blind verification of the decisive novelty objections (once), then the judges (then, where `verifier.auto_gate_batch` is on, one gate batch for objections the judges escalated). Requires the refuters to be complete (or `allow_incomplete_refuters`). |
| `run_synthesis` | The synthesis memo. |
| `run_completeness_critic` | The critic that hunts for what everyone missed. |
| `run_followup` | One or more follow-up rounds on the critic's items (verify, adjudicate, revise the memo, re-check). `dry_run=true` shows what the next round would do. |
| `run_full_review` | Create the run and execute everything, follow-up rounds included, in one background worker. |
| `resume_run` | Continue an interrupted or partial run (`through="followup"` includes an unfinished follow-up round). Completed agents are not redone unless their inputs changed; those, and the stages after them, are rerun. |
| `run_verification` | Re-check the evidence gate's unconfirmed verdicts with blind verifiers (e.g. after `add_prior_fulltext`), or check your own (prior paper, claim) pairs. |
| `add_prior_fulltext` | Supply a PDF/HTML of a prior paper the server could not download (e.g. paywalled). |
| `release_quarantine` | Make a quarantined output usable again, with a recorded reason (isolation failures need the CLI). |
| `get_run_status` | Per-agent status (follow-up agents and their verifiers under their round; superseded memos marked), plan windows (a model-specific one under its model's name), literature-search health and preflight warnings, gates and quarantines, failed calls, retry counts, isolation audit, verification, follow-up rounds with every re-check item and the open issues, stale outputs, artifact paths. `wait_seconds` blocks up to 10 min with progress updates. |
| `get_report` | One output at a time: `memo` (the current memo), `novelty`, `rigor`, `fit`, `verifier`, `judge`, `synthesis`, `critic`, `adjudicator`, `revision`, `recheck` (with `agent_id` for one report), `matrix`, `evidence_gate`, `evidence`, `verification`, `prior`, `gates`, `followup`, `items`, `followup_matrix` (with `round`), `claims`, `refcheck`, `paper`, `sections`, `references`, `usage`, `events`, `index`, or per-agent `context`, `transcript`, `prompt`, `search_log`. Long texts are paged. |
| `get_run_cost` | Tokens and cost by phase (follow-up rounds separately) and in total. |
| `cancel_run` | Stop a run's worker and every agent process it started. |
| `list_runs`, `validate_config` | Housekeeping. |

A full review takes tens of minutes to hours at high effort, so the `run_*` tools return immediately. The worker is a separate process. It keeps running if Claude Desktop quits, and `cancel_run` stops it.

The same operations work from a terminal:

```bash
.venv/bin/python -m paper_adversary review paper.pdf --venue "ICML 2027"   # detached, follow-up included
.venv/bin/python -m paper_adversary review paper.pdf --preset milestone     # a wider panel for a milestone
.venv/bin/python -m paper_adversary status <run_id>
.venv/bin/python -m paper_adversary report <run_id> memo
.venv/bin/python -m paper_adversary refuters <run_id> --rerun N2 --foreground
.venv/bin/python -m paper_adversary followup <run_id> --dry-run
.venv/bin/python -m paper_adversary verify <run_id>                         # re-check the evidence gate
.venv/bin/python -m paper_adversary release <run_id> R2 --reason "..."     # asks for confirmation
```

## Presets

A preset decides who reviews: how many agents of each role, on which model and effort, with which lens. It
also sets how many prior papers get blind verification and how many follow-up rounds run. Presets live in
`config/presets/`. New runs use `routine`. Pick another per run, or make it the default in your standing settings
(`preset: milestone` in `$PAPER_ADVERSARY_CONFIG`):
- **Claude Desktop:** ask for it ("run a milestone review of …"); the tools take a `preset` argument.
- **CLI:** `--preset milestone` on `create` or `review`.

| | routine (default) | milestone | legacy_thorough |
|---|---|---|---|
| Intake | Opus medium | Opus medium | Fable high |
| Novelty | Opus high (direct and recent competitors), Fable high (equivalent ideas, adjacent fields, gap claims) | 4 lenses: Opus high ×2, Fable high ×2, one focus each | 4 × Opus high |
| Rigor | Fable xhigh (mathematics), Opus high (empirical methodology, claim–evidence consistency) | 3 lenses: Fable xhigh (mathematics), Opus high (empirical), Fable high (claims vs evidence) | 3 × Fable max |
| Fit / feasibility | Opus medium | Opus medium | 3 × Opus high |
| Judges | Opus high, Fable high | Opus xhigh, Fable xhigh | 3 × Fable max |
| Synthesis | Opus high | Opus high | Fable max |
| Completeness critic | Fable high | Fable xhigh | Fable xhigh |
| Blind verifiers | the other model than the objection's author, high; ≤ 3 prior papers per run | same; ≤ 6 papers | Opus high; ≤ 8 |
| Follow-up | 1 round; adjudicators Opus high + Fable high; revised memo Opus high; re-check Fable high | up to 2 rounds | up to 2 rounds, all Fable (max; re-check xhigh) |
| Format repair | Opus low | Opus low | Opus low |
| Max effort | nowhere | nowhere | rigor, judges, memos, adjudicators |

**`routine`** puts one agent on each model wherever a role has two, so the two models check each other. It never
uses max effort.

**`milestone`** widens `routine` for a milestone review. Max effort stays reserved for a difficult technical
question you choose explicitly, in trusted settings. For example, give the mathematics refuter max effort with
`rigor: {per_agent: {1: {effort: max}}}`.

**`legacy_thorough`** is the configuration every run used before presets, kept unchanged. Runs created before
presets keep their own saved copy of it.

**`stage2_split`** is this project's preset for the Stage-2 candidate reviews. It uses max effort, so it is
selected in trusted settings (`--trusted-config`), never per run. Its header comment records each panel decision
and its date. Since 2026-10-02 it has two judges (Fable 5.1 max), adjudicators on Opus 5.5 high and Fable 5.1 max,
a re-check critic on Fable 5.1 high, and the newest prompt versions (`novelty_v3`, `rigor_v2`, `fit_v2`,
`verifier_v2`, `judge_v3`, `synthesis_v4`, `critic_v3`, `adjudicator_v2`, `critic_recheck_v2`).

**Verification beyond the refuters' cap.** In every preset, each follow-up round and each gate or user re-check
(`run_verification`) has its own budget of 2 prior papers and 2 verifiers (`config/default.yaml`), so the
refuters' objections can never use it up.

**Proposed allocations, not proven optima.** `routine` and `milestone` were chosen for cost. They have not been
measured against `legacy_thorough`, so whether they miss more or cost less is untested.
[`eval/README.md`](eval/README.md) describes how to compare presets on cases you adjudicate; nothing has been
measured yet.

**Per-agent overrides.** Any role (and the follow-up adjudicators) can give single agents their own model,
effort or lens over the role's defaults: `judge: {per_agent: {2: {model: opus-5.5}}}`. Agent numbers must exist,
models and efforts must be supported, lenses must exist in the lens set (`prompts/lenses_v2.yaml`), and no two
agents of a role may share an explicit lens.

**Saved assignments.** The resolved assignment of every agent is saved when a run is created: `state.json`,
`config.yaml`, and a readable `assignment.json`. Resuming uses that saved assignment, so editing a preset later
never changes an existing run.

## Configuration

`config/default.yaml` holds the settings every preset shares, `config/presets/` holds the presets, and `config/models.yaml` holds the models. The orchestration code never names a model. Roles refer to aliases (`fable-5.1`, `opus-5.5`), and `models.yaml` maps each alias to its model ID, documented limits, supported effort levels and API-equivalent prices.

Precedence, later wins: `default.yaml` → the preset → `$PAPER_ADVERSARY_CONFIG` (your standing settings, trusted) → `--trusted-config FILE` on the CLI (trusted) → a run's `config_override` (restricted, below). Unknown keys, unsupported efforts and unknown aliases are rejected before anything runs.

Per-run overrides (`config_override`, and the `preset` argument) come through MCP tool arguments, which a model chooses after reading untrusted paper text. A per-run override may pick a preset listed in `presets.selectable` and lower or redirect settings. The rest needs trusted settings (`config/`, `$PAPER_ADVERSARY_CONFIG`, or `--trusted-config` in your terminal):
- raising agent counts or limits above the preset's: agents per role, the verifier budgets (`verifier.max_papers_per_run`, `max_agents_per_run`, `max_agents_per_batch`, `max_papers_per_followup_round`, `max_agents_per_followup_round`, `max_papers_per_user_batch`, `max_agents_per_user_batch`), `plan_limit.max_waits_per_run` and `max_wait_hours`;
- `plan_limit.model_window_reserve` in either direction (lowering it would let a run start that the plan window cannot hold);
- turning on `verifier.auto_gate_batch`;
- `effort: max` anywhere (a role, one agent, a follow-up role, the format repair);
- `presets` itself;
- the provider's security keys: `allow_api_key`, `extra_args`, `env_passthrough`, `claude_binary`, `safe_mode`;
- every gate setting except `gates.repair.{model, effort, timeout_minutes}`;
- which hosts documents are downloaded from, and how politely: `search.fulltext.{host_policy, allow_hosts, max_pdf_mb, max_html_mb, arxiv_document_interval_seconds, host_interval_seconds}`;
- what counts as shown evidence: every `evidence.*` key, and turning the blind verifier off (`verifier.enabled: false`);
- follow-up rounds can be reduced per run (fewer rounds, fewer adjudicators, `auto_start`/`auto_continue` off), never increased.

Flags that would widen an agent's access (`--add-dir`, `--permission-mode`, `--settings`, `--mcp-config`, …) are rejected in `extra_args` in any config. Prompt, lens-set and rubric names are file stems only; a rubric given as a path must be a visible `.md`/`.txt` file, and its text is copied into the run.

**Rubric tests.** A rubric may declare explicit tests in its YAML front matter:

```yaml
tests:
  - {id: RT1, text: "A concept that reduces to an existing method plus a tweak is incremental: FATAL"}
  - {id: RT2, text: "A theorem target implied by first-order null-space arguments does not count"}
```

IDs are short names (letters, digits, underscores) that cannot look like an agent, objection, item or claims-ledger ID; at most 30 tests. A malformed list is rejected when the run is created (or by `validate`). A named rubric that declares tests keeps its front matter, so the tests travel with the run's copy. Judges whose prompt rules on tests (`judge_v3`, front matter `rubric_tests: true`) get the tests under `<rubric_tests>` and must rule on each one, whether or not a refuter raised it: whether it applies (yes, no, partly) and with what severity. The gate checks this like objection coverage. The judgment matrix gets one row per test and one column per judge, which the memo and the critic read. A test ruled to apply with FATAL severity counts like a FATAL judgment in the open issues. With `judge_v2` the tests are listed in the matrix but nothing rules on them, and creating such a run warns. Rubrics without tests behave as before. Refuters whose prompt reads the rubric (`novelty_v3`, `rigor_v2`, `fit_v2`) see the whole rubric, its tests included, and calibrate their severities against it; the judges still rule on every test.

Validation against what your plan actually serves happens before each run: the Claude Code login is checked for free, and a run on an API-key login is refused. Then a preflight probe sends one tiny low-effort request per distinct model and tool setup, cached for 24 h. It uses the same tools and MCP server as the real agents, so a broken tool setup stops the run before any real agent is spent. The probe also fails if another model answers or if Claude Code's output lacks the events the isolation audit needs. Later preflights (before verifiers and follow-up steps) cover only the models still to run; their results are merged into the run's record by model and tool setup, never replacing the earlier ones.

Main knobs: `concurrency.max_parallel_agents` (default 3, plan-friendly), `retry.*`, `plan_limit.*` (see [Plan limits](#plan-limits)), `search.providers`, `search.fulltext.*`, `evidence.require_independent_check`, `followup.*`, `gates.*` (including `gates.coverage_supplement` and `gates.placement_fix`), `lens_set`, `rubric`.

## Completion gates and quarantine

Every agent's output is saved first, then checked (`gates.py`). Only a passing gate (or a release you record) makes an agent `complete`, which is the only status later stages read. A crash or cancel while checking leaves the agent `gating`; a resume finishes the checks without rerunning the agent.

| Check | Outcome |
|---|---|
| Isolation audit failed (a file outside the sandbox was read, forbidden content in a tool result, another report's marker) or unverifiable (no transcript, no `init`/`result` event) | one automatic fresh rerun (old output archived; at most 2 per job), then quarantine |
| A call the sandbox refused | warning (often a sign of prompt injection in the paper) |
| Another model answered than configured | quarantine |
| Report cut at the output-token limit | quarantine, unless its final JSON block is intact |
| Hidden turns: model turns before the final message whose text the report lacks (an agent without tools took more than one turn by Claude Code's count, or an earlier turn stopped at the output-token limit; a streamed message that never ended, a request Claude Code retried, is no turn) | warning when the report is intact; quarantine when it lacks its leading section (`## Overall assessment` for judges, `## 1. Executive summary` for memos) or has no structured block at all (one that is there but invalid shows the final turn holds the report's end), since its start may be in a discarded turn |
| Structured block missing or invalid | free fixes first: strict parse, lenient parse, rebuild from the report's own headings (rigor, fit, judges). Intake and novelty reports get a format-repair call instead, validated so it can transcribe but never add content. Otherwise quarantine. The data's source is recorded. |
| A judge or adjudicator leaves objections (items) unclassified, or gives one two different severities; a `judge_v3` judge leaves a test the rubric declares without exactly one valid ruling | one targeted supplement call on its own model and effort, ruling on exactly those and seeing only the reports that raised them (and those rubric tests); its rulings are merged and its reasoning appended to the report. Still incomplete → quarantine. Repeats with one severity (or one ruling) are merged for free; judges also get an index of every objection ID. An objection's severity is its own `member_severity` where a grouped judgment gives one (one rated above the judgment's severity is kept, with a warning) |
| Memo's last section missing | quarantine (a gap in the middle is a warning) |
| A memo lists an unverified prior-work verdict under "Criticisms that survived judging", or leaves it out of "Unverified threats" | one targeted call on the memo's own model rewrites only those two sections; the result is spliced in and checked again; still wrong → quarantine. Only the IDs an entry starts with count, so a passing mention does not trigger it |
| Revised memo: an item without a disposition, or an unverified prior-work item not filed as an unverified threat | quarantine |

Targeted calls (a supplement or a placement fix) are made only for an output that no other check blocks, at most once per output, and a resume never pays for one twice. Each is recorded in the gate sidecar, and the edited report keeps its previous version in the archive.

A quarantined output stays on disk but no later stage reads it: downstream prompts only see a label such as `R2 (quarantined: isolation audit failed)`, the matrix lists it as excluded, and the isolation guard forbids its content for every role. Later stages wait for it as for a failure (or proceed with `allow_incomplete`). Rerun it, or release it with a recorded reason: `release_quarantine` for quality reasons, the terminal (`paper-adversary release`) for isolation reasons, so a prompt-injected paper cannot release a contamination. `get_report(run_id, "gates", agent_id)` shows every check.

## Evidence checks and the evidence gate

Bibliographic reference checks (does a cited paper exist?) are not enough: a refuter can cite a real paper for something it does not say. So decisive novelty objections must be shown with verbatim passages:

- **Verbatim tools.** Novelty refuters read prior papers through `read_prior_paper`, `find_in_prior_paper` and `check_quote`, which return the paper's own text. Claude Code's web fetch returns a model-written summary, so `novelty_v2` forbids quoting from it (or from memory).
- **Decisive objections** (already done, partially anticipated, framing exists, gap not real, concurrent work; or incremental/integrity with references; severity fatal or major) must give `overlap_evidence`: a verbatim prior passage and the verbatim submission passage it bears on, with locations.
- **Quote check** (`passages.py`, `evidence.py`). After a novelty refuter completes, every passage is matched against the prior paper's full text and the submission. This covers decisive objections and every other prior-work objection that gives `overlap_evidence`, because a judge may rate an objection more severely than its refuter did; judges see both, the others marked "below major". Ligatures, hyphenation, page breaks and typography are normalized, and a changed negation, number or quantifier never counts as a match. By default only word-for-word matches are accepted. Near-verbatim matches and quotes with an ellipsis are reported, with the text actually found, but not accepted; set `evidence.accept_approximate_score` (at most 1) in your standing config to accept near misses above that score. A prior passage taken from the cited paper's reference list does not count. Nor does a submission passage from the submission's own related-work or reference section, which describes the prior work rather than the claim. Results: `novelty/N*.evidence.json`, shown to judges.
- **Identity checks.** If a reference's identifier resolves to a paper with another title, the result is a REFERENCE MISMATCH. If the "prior paper" is the submission itself (e.g. its own preprint), it is INVALID.
- **Blind verifiers** (`verification.py`). For each prior paper behind a decisive objection, a verifier gets the whole submission, the passages at stake and the prior paper's full text. The passages are the submission passages the objection quoted against that paper (up to two distinct ones per reference), or, for a critic item, its `submission_quote`. It never sees the objection's argument, category, severity or author; the isolation guard enforces this. It says whether the prior work anticipates each passage fully, partially or not at all, with its own quotes, which are checked too. `verifier_v2` tells it that its submission quotes must come from inside the passage, and its `check_quote` tool says when one does not. An "anticipates" verdict becomes "cannot tell" in any of these cases:
  - its quotes do not check out;
  - its submission quote lies outside the passage it was shown;
  - its prior quote is shorter than 8 words or comes from the reference list.
- **Cross-model routing** (`verifier.routing: cross_model`, in `routine` and `milestone`). A decisive objection is verified on the other model than its author's: an Opus refuter's objection goes to a Fable verifier, a Fable refuter's to an Opus verifier, and a critic item to the other model than the critic's. A prior paper cited by authors on both models gets one verifier, routed by the request that ranks it (below); the task records which authors shared the verifier's model.
- **Priorities and budgets.** Each kind of batch has its own budget:
  - the refuters' batches share `verifier.max_papers_per_run` and `max_agents_per_run` (3 in `routine`, 6 in `milestone`, 8 in `legacy_thorough`);
  - each follow-up round has `max_papers_per_followup_round` and `max_agents_per_followup_round`;
  - each gate or user batch has `max_papers_per_user_batch` and `max_agents_per_user_batch` (2 each by default).

  Within a budget, prior papers are chosen one at a time, re-ranked after each choice:
  1. a paper that gives an objection (or item) without an independent check its first one, ahead of a second check for an objection already covered (by this batch or an earlier one);
  2. then a threat to a central claim (in the abstract, or restating a claim of the intake ledger);
  3. then severity;
  4. then doubt (the author's lower stated confidence, a thinner evidence basis);
  5. then the number of unchecked objections the paper serves;
  6. remaining ties alternate between authors, then follow the request ID.

  Confidence is used only as an ordinal tie-break, never as a probability. Threats beyond a budget are recorded as not run, with the reason that decided it (for example "every objection it serves already has an independent check (T1)", "ranked below N2-O1:r1, rated more severe", or "tied ...; lost the tie-break"), and the gate labels verdicts resting on them UNVERIFIED with "cap reached" as the reason; they are never silently dropped. Only the first three references of an objection are verified; later ones are recorded as not run ("per-objection reference limit"), with the paper they name, also in a gate batch for an objection the judges escalated. A passage already checked against the same prior text is never checked again, whichever model checked it.
- **Reruns.** Verdicts are tied to an objection's content, not its ID. When a refuter is rerun, its changed decisive objections are verified again (batch `refuters2`, …) before judges run. A batch whose verifiers failed is recorded incomplete; when a later job reruns them (the verifier stage of `resume_run`, or the automatic gate step before the memo), their verdicts are collected and the evidence gate is recomputed. Runs planned before 2026-10-02 do not grow the second-passage and past-the-limit requests their batches were planned without: an objection whose first request is recorded for its current content gets no new batch for them (and the judges do not turn stale); a gate re-check covers them.
- **Evidence gate** (`evidence_gate.py`). Judges keep the four severities (`judge_v2` adds an evidence-status field, `judge_v3` per-member severities). Each objection of a grouped judgment is held to its own severity: a judgment's severity is its strongest member's, and `judge_v3` lists members it rates lower in `member_severity`. A member whose own severity is below MAJOR is listed apart ("cited in this judgment, but J3's own severity for it is MINOR"); it is neither checked as part of the serious verdict nor flagged. A judgment that says it relies on prior work (`relies_on_prior_work`) while none of its prior-work members carries its severity (it has none, or rates each lower) has its serious members checked instead, so adding a lower-rated prior-work member never takes a serious verdict out of the gate. A flagged objection whose serious verdicts all come from judgments that gave it no severity of its own (`judge_v2` groups; under `judge_v3` a member left out of `member_severity` is rated at the judgment's severity by the judge's own choice), while the judges that ruled on it alone rated it lower, is named as flagged only through grouped verdicts. A FATAL or MAJOR BUT FIXABLE verdict on prior work counts as shown only when a blind verifier found anticipation (fully for FATAL, at least partially for MAJOR) with verified quotes. Otherwise it is labelled (UNVERIFIED with the reason, DISPUTED, UNCLEAR, SCOPE, REFERENCE NOT FOUND, REFERENCE MISMATCH, INVALID), never downgraded. Each objection of a grouped verdict carries its own reason for having no independent check (for example "cap reached", "not requested: the refuter rated it minor; judges escalated it", or, for a member that is not a prior-work objection, "not requested: it cites no prior paper to check"); the cap label is used only when every one of them was capped. A judge's declared evidence status is compared with what that status claims: `verified_quotes_only` needs some objection's quotes verified, `verified_independent` a blind verifier's anticipation (fully for FATAL), whether or not the gate requires the independent check. The independent check is required while the verifier is enabled; with it turned off in your standing config, accepted refuter quotes suffice. The memo (`synthesis_v3`) lists labelled verdicts under "6. Unverified threats — check before acting", not "Criticisms that survived judging", and starts every entry of those two sections with its IDs. A memo that misfiles one is sent back for those two sections (see the gates table). `get_report(run_id, "evidence_gate")`.
- **Judgment matrix** (`judges/judgment_matrix.md`). Each cell is one judge's verdict on one objection at that objection's own severity; a cell from a `judge_v2` grouped judgment that gave the objection no severity of its own is marked "grouped" (under `judge_v3`, whose front matter says `member_severity: true`, such a member's severity is the judge's own rating, so it is not marked). The Evidence column covers every decisive objection, every member of a serious prior-work verdict and every objection a verification was requested for: its quote check, its blind verdicts, and what was not checked and why ("1 more not run (cap)", "not blind-checked (not requested: the refuter rated it minor; judges escalated it)"). Rows the gate flags carry "not shown (evidence gate)".
- **Escalated objections.** With `verifier.auto_gate_batch` (on in the project's `stage2_split` preset, off in the others), the worker runs one gate batch right after the judges for the objections the gate flags that no batch was ever asked to check (typically ones the refuter rated minor and a judge rated FATAL or MAJOR), within the gate batch budget, then recomputes the gate. The judges are not rerun. This happens once per run; a batch that was cut short is finished before the memo is written, never planned twice, and verifiers of it that failed are rerun by a later job (the verifier stage of `resume_run`, or the step before the memo; not again in the job where they failed). The batch also records an escalated objection's references past the third as not run. Later escalations need `run_verification(run_id, from_gate=true)`.
- **Closing a gap.** If a paper was unavailable, give the server its PDF with `add_prior_fulltext`, then `run_verification(run_id, from_gate=true)`. It re-checks every flagged objection, including ones the judges escalated, within its own budget; a budget of 0 is refused before anything is fetched. The gate is recomputed without rerunning the judges. They read only the refuters' verification batches, so a gate batch's verdicts never make them stale. Before the batch, each refuter with a quote check that could not succeed (text unavailable, quotes not found) is checked again; its record is rewritten only when a result changed, for example with the supplied text, and only then do the judges' and the memo's inputs change (`resume_run` refreshes what changed). The memo also reads the recomputed gate. The file must be identified by arXiv ID or DOI, its title must match the cited paper, and it must not be the submission. It never replaces a downloaded text, and it is stored in this run's `prior/` only, never in the shared cache.

## Full-text sources and politeness

`search/fulltext.py` resolves a reference by arXiv ID, DOI or title through the scholarly APIs, then tries, in order:
1. the arXiv HTML rendering (LaTeXML: clean text, math as LaTeX, paragraph anchors);
2. the arXiv PDF;
3. open-access PDFs that OpenAlex or Semantic Scholar list, but only on allowed hosts: arXiv, OpenReview, PMLR, NeurIPS, ACL Anthology, JMLR, CVF, AAAI, IJCAI, PMC/Europe PMC and Semantic Scholar's PDF host, plus `allow_hosts`.

Download URLs come only from API metadata, never from an agent. Redirects are followed by hand and each hop is host-checked. Bodies are size-capped and type-checked. Documents are extracted in a child process with memory, CPU and time limits and no credentials in its environment (`search/extract.py`), so a malicious PDF cannot take the worker down. arXiv documents are fetched at most every 15 s per runs directory (its robots.txt crawl delay), and an arXiv block pauses arXiv for every agent and the search backend for 2 h. Results are cached per runs directory and snapshotted into the run's `prior/` folder, so every reader sees the same bytes. A paper with no open text is reported `fulltext_unavailable` with the reason; the abstract is never passed off as the full text.

**Shared per runs directory, not machine-wide.** The caches (search results, full texts, the model probes, the plan windows), the rate gates and the circuit breakers, the arXiv gate included, live in `.cache/` under the runs directory (`PAPER_ADVERSARY_RUNS_DIR`, default `runs/`). Runs share them only when they share a runs directory, so runs that go in parallel must use the same one (the Stage-2 batch does, `runs/stage2_2026-10`). A Claude Desktop review under `runs/` and a CLI batch under another directory each have their own arXiv gate and breakers: together they can exceed arXiv's crawl delay, neither sees the other's block, and texts one has downloaded are downloaded again by the other.

## Follow-up rounds

`critic_v2` writes every finding as a typed item (new issue, minority critique, judging flaw, synthesis flaw, novelty to verify) with a location, severity estimate, and what would settle it. `critic_v3` and `critic_recheck_v2` (in `stage2_split`) add `submission_quote` to a novelty-to-verify item: the submission's exact words that the suspected prior work would anticipate. They also tell the critic what its severity estimate does: which items are adjudicated, which are only noted, and, for the re-check critic, what happens to its items when no further round runs. A follow-up round (`followup.py`):

1. **Triage** (deterministic, free): validates items (cited objection and judge IDs must exist), routes them (adjudicate; revision only for memo flaws; noted when below `min_severity`; invalid), and attaches the judges' verdicts on cited objections. A judging-flaw item that cites an objection with a FATAL or MAJOR BUT FIXABLE verdict (or a rubric test a judge ruled to apply with such a severity), or a minority-critique item that rates such an objection or test differently, disputes a computed result (those verdicts feed the open issues and the evidence gate). It is adjudicated whatever its own rating; `items.md` says why and lists the cited rulings.
2. **Verification**: items naming suspected prior work go to blind verifiers, within the round's own budget. They see the whole submission, the passage at stake and the prior paper, never the critic's prose. The passage is the item's `submission_quote`; if that is missing or not found verbatim, the first claims-ledger claim the item cites that is found; then its location. A cited claim may be written in words ("ledger claim C3", as `critic_v2` asks); the ledger ID is read from it. The request records which one located it (`located_by`). Identical checks reuse earlier verdicts. An item's verdicts are tied to its content; its `submission_quote` and cited claims count only for items that give a `submission_quote`, so items written before it existed keep their verdicts. The round's agents see these results once, under `<followup_verifications>`; earlier rounds' results stay under `<independent_verifications>`.
3. **Adjudication**: two adjudicators (one on each model in `routine` and `milestone`), blind to each other, rule on every routed item with the judges' four severities and may re-rate base objections (`objection_updates`). `adjudicator_v2` (in `stage2_split`) settles a disputed verdict that way (a disputed rubric test under its test ID) and names the one objection an item repeats, if any. A follow-up matrix flags contested items and rubber-stamping, lists which objections every adjudicator re-rated, and the evidence gate applies to prior-work items.
4. **Revised memo** `S<r+1>`: a full memo with a closing "What changed after the completeness critique" table. Every item gets a disposition (incorporated, rejected, needs evidence, unverified threat, noted, invalid). It becomes the current memo; earlier memos stay in place, recorded as superseded.
5. **Re-check critic** `C<r+1>`: a fresh critic on the revised memo. New items at `min_severity` or above start the next round automatically, up to `max_rounds` (default 2). Items it re-raises never start a round, since the adjudicators already ruled on them, but they stay open. `round.json` records every re-check item: new, re-raised, below `min_severity` (recorded only, as the first critic's noted items are), or of an unknown type. The status lists them, including suspected prior work that no round will verify.

`run_full_review` starts round 1 by itself when the critic raises an item at `min_severity` or above (`followup.auto_start`).

**How a review ends.** A review that stops without new items is labelled `ready_for_next_gate` only when nothing serious is open. Otherwise it is `review_saturated_with_open_issues`, and the status lists the open issues. These block `ready`:
- a FATAL verdict from a judge or adjudicator that the adjudicators did not overturn;
- a rubric test that a judge ruled applies with FATAL severity, likewise;
- an unverified FATAL prior-work threat;
- an item the re-check critic re-raised;
- a new item of the re-check critic (at or above `min_severity`) rated MAJOR BUT FIXABLE or FATAL that no round adjudicated, at the round cap. Items below `min_severity` never block, whichever critic raised them.

Overturning takes every adjudicator of the latest round that re-rated the objection: the verdicts more severe than the highest of their ratings then no longer count, MAJOR BUT FIXABLE ones included. One adjudicator alone overturns nothing. Each objection counts at its own severity (`member_severity`), not its group's.

Open MAJOR BUT FIXABLE issues are listed but don't block. They are counted as issues, not IDs:
- IDs that every judge ruled on in one and the same judgment are one issue;
- a follow-up item joins the one base objection that every adjudicator said it repeats (ruled not new, or only partly new; named as its only covered objection, or re-rated for it);
- an issue never mixes IDs the evidence gates flagged with IDs they did not;
- IDs that are MAJOR only through grouped verdicts, while the judges that ruled on them alone rated them lower, are listed apart and not counted among the issues.

The grouping is conservative: recomputed on the K1 pilot, 23 IDs give 12 issues plus 4 grouped-only IDs, where the revised memo lists 14 entries. `round.json` keeps the per-ID entries, each with its issue. A review cut off at the round cap with new items is `max_rounds_reached`. A job stopped at a plan limit ends `incomplete`, never with a review label.

**Review saturation is not research readiness.** These labels say how the review ended, not whether the work is good. `ready_for_next_gate` means only that this panel, on this evidence, raised nothing serious that is still open. It cannot tell you that the experiments will work, that the search found all prior work, or that the paper will be accepted. `review_saturated_with_open_issues` means the review has stopped finding new issues while serious ones remain; fix or check them, then review again. The same label applies when the critic raises nothing to follow up. Each round's `round.json` records the open issues with the earlier rulings and dispositions of re-raised items.

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
   - refuters may read only the paper and the rubric (a user input; only refuter prompts that calibrate against it, `novelty_v3`, `rigor_v2` and `fit_v2`, are given it);
   - blind verifiers may read only the paper, their passages and their prior paper;
   - judges may read the paper, rubric, refuter reports and the orchestrator's checks, never another judge;
   - synthesis and the critic see everything upstream.

   The table lives in code, so no config can loosen it.
2. **Position rule.** An agent may read only artifacts from strictly earlier steps. So the two adjudicators never see each other, a rerun never sees what came after it, and a rerun of the base critic never sees the follow-up.
3. **Pre-flight guard.** Every report carries a marker, and distinctive word 8-grams of each report are registered. Before an agent starts, its fully assembled prompt is scanned for fingerprints of every artifact its role must not see, and for credentials (token patterns, and the values of the secrets the server holds). A hit aborts the agent. Documentation placeholders such as `sk-ant-api03-XXXX…` or AWS's `…EXAMPLE` key don't count, so papers about LLM tooling stay reviewable. The same scan runs when a paper is ingested; hidden files (`.env`, `~/.ssh/…`) and files without an extension are refused outright. Phrases that also occur in inputs the agent may legitimately see don't count as evidence, and neither do passages quoted from third-party papers.
4. **Process sandbox.** Each agent is its own `claude -p` process. It runs in a fresh, empty folder outside the project, with `--restricted` (no shell, file tools confined to that folder) and an explicit tool allowlist. It uses `--strict-mcp-config`, `--safe-mode` when no tools are needed, and `--no-session-persistence`, with a scrubbed environment. When the source is a PDF, rigor agents, judges and adjudicators get a copy of it (and nothing else) in their folder.
5. **Transcript audit and gates.** After each agent, its transcript is checked: file access outside its sandbox (paired with the tool's result, so refused attempts are told apart), forbidden fingerprints in tool results, and the events that prove the checks ran. A failed or unverifiable audit triggers one rerun, then quarantine (see above).
6. **Manifests.** Each agent's `*.context.json` lists every input it received, once, with its hash, and which planned inputs were missing. Inputs it was not shown are not listed: a follow-up agent's manifest names only the memo, critic and adjudications in its prompt. An agent is marked stale when an input changes, becomes quarantined, or becomes available after it ran.
7. **No side channels.** The intake pass's reading of the paper (title, field, claims) is stored separately and reaches only synthesis, the critic and the follow-up roles. Refuter and judge headers use only what you supplied or what was extracted from the file.

## Storage

```
runs/<run_id>/
  metadata.json  config.yaml  state.json
  source/     paper.pdf|md|tex, extracted_text.md, sections.json, references.json, rubric.md (per-run rubric),
              profile.md, claims_ledger.md (claims L1, L2, ...)
  novelty/    N1.md (+ N1.json structured data, N1.gate.json, N1.refcheck.json, N1.evidence.json,
              N1.search_log.jsonl, N1.context.json)
  rigor/  fit/
  prior/      text snapshots of the prior-work full texts read in this run, index.json
  verify/     V1.md (+ sidecars), tasks/T*.json (what each verifier saw), results.json, batches/,
              base_results.json (without follow-up checks), judge_results.json (the refuters' batches: what judges read)
  judges/     J*.md, judgment_matrix.md/.json, evidence_gate.md/.json
  synthesis/  memo.md (S1), memo_S2.md, ...
  critic/     completeness.md (C1), completeness_C2.md, ...
  followup/   A1.md, A2.md, ...; round-<r>/ items.json, items.md, followup_matrix.md/.json, round.json
  assignment.json   who does what: every agent's model, effort and lens, verifier routing, follow-up panel
  logs/       events.jsonl, usage.jsonl, api_usage.json, worker.log, fingerprints.json,
              agents/<id>/attempt-<n>/{system_prompt.md, user_prompt.md, transcript.jsonl, stderr.log,
              context.json, repair-<k>/}
  archive/    superseded outputs (reruns never delete), index.jsonl
runs/eval/<case>/<pair>/   pair.json, comparison.md/.json, manual_scores.yaml (eval/README.md)
runs/.cache/      shared by every run of this runs directory: search/, fulltext/, ratelimit/, breaker/,
                  model_probes.json, plan_windows.json (the newest use of each model-specific plan window)
```

Every report starts with YAML front matter. It records:
- the model requested and the models that actually served it;
- effort, prompt version and its SHA-256, lens;
- timestamp, run ID, role and round;
- usage, attempts;
- the structured block as written, the isolation audit, and warnings.

Gate verdicts are in `*.gate.json`.

## Prompts and experiments

Prompts live in `prompts/` as `<role>_v<N>.md` (front matter + template). Lenses are in `lenses_v1.yaml` and rubrics in `rubrics/`. Versions are never edited in place: a change is a new file (as `novelty_v2`, `novelty_v3`, `rigor_v2`, `fit_v2`, `judge_v2`, `judge_v3`, `synthesis_v3`, `synthesis_v4`, `synthesis_revision_v2`, `critic_v2`, `critic_v3`, `critic_recheck_v2`, `adjudicator_v2` and `verifier_v2` are), selected in the config or per run, e.g. `config_override={"novelty": {"prompt": "novelty_v3"}}`. Each report records the prompt version and the file's hash, so in-place edits are detectable too. A prompt whose front matter says `structured_output: true` must end its output with a fenced JSON block. Besides the agent's ID, panel size, venue and field, a template may use `{{tool_access}}`, a sentence naming exactly the tools the agent has. It mentions `./paper.pdf` only when the PDF is in the agent's folder and it has a Read tool, telling it that text extracted from a PDF garbles equations and tables, so it reads the relevant pages before calling one wrong (as `rigor_v1` said); an agent with no tools is told so: in the K1 pilot, a tool-less judge told about the PDF spent 128,000 output tokens on imaginary tool calls. A template may also use `{{min_severity}}`, the run's follow-up threshold in the rubric's words. `novelty_v3`, `rigor_v2`, `fit_v2`, `judge_v3` and `adjudicator_v2` use `{{tool_access}}`; `critic_v3` and `critic_recheck_v2` use `{{min_severity}}`.

Refuter prompts whose front matter says `rubric: true` (`novelty_v3`, `rigor_v2`, `fit_v2`) get the run's rubric. They calibrate their severities against it and raise a rubric criterion within their remit when it applies, without narrowing that remit; the rubric's instructions for the judges (a recommendation line, who rules on its tests) are not theirs. For a research idea they judge the planned contribution, theorem targets and tests, not missing results: `novelty_v3` checks the concept against any list of existing or incremental methods the submission or rubric gives, `rigor_v2` checks each theorem target against bounds the submission's own material already proves, and `fit_v2`'s fatal criterion for an idea is that no planned test could establish the central claim. Earlier refuter prompts see only the paper, as before.

**IDs.** An agent ID is a role letter and a number (N1, R1, F1, V1, J1, S1, C1, A1); objections and items add `-O<n>` and `-I<n>`. In runs whose memo, critic and re-check critic prompts cite the ledger by L IDs (`synthesis_v4`, `critic_v3` and `critic_recheck_v2`, front matter `ledger_ids: L`; the `stage2_split` preset), the claims ledger numbers the submission's claims L1, L2, ... by position, whatever IDs the intake model wrote, so a claim never shares an ID with an agent (in the K1 pilot, C1 was both a ledger claim and the critic). Runs on the older memo and critic prompts, which the other presets still use and which cite C IDs ("ledger claim C3"), keep the IDs the intake model wrote (`intake_v1` writes C1, C2, ...), as before, so their memos cite what their ledger shows; a run that mixes the two shows both ("**L2** = **C2**"). Either ID resolves, so a critic item citing C3 finds its claim in any run, as do the items of runs created before L IDs. Every prompt's `<review_context>` also says that agent IDs name the agents of this review, and that an identifier written the same way inside the submission (in the K1 pilot, the submission's own A1, R1 and F1) names the submission's own entity.

## Long papers

If a paper does not fit an agent's budget, it is never silently truncated. The budget is the context window minus output reserve, safety margin and other inputs. The role's priority sections stay inline in document order, every other section becomes an explicit placeholder, and the agent gets a `read_paper_section` tool that returns any section in full. Verifiers split their budget between the submission and the prior paper and read omitted prior sections with `read_prior_paper`. The plan is recorded per agent. Token estimates start from a conservative heuristic and are then calibrated from the usage agents report.

## Novelty search

Novelty refuters get Claude Code's web search and fetch, plus a small MCP tool server (`tools_server.py`). It offers `search_literature`, `lookup_paper` and `find_citing_papers` over OpenAlex, Semantic Scholar, arXiv and Crossref, and the verbatim full-text tools above. Queries are fanned out and de-duplicated, cached on disk, rate-limited per provider across all agents of the runs directory, and protected by circuit breakers (arXiv blocks heavy users for hours, so it gets a slow gate). Add a provider by implementing `search.base.SearchBackend` and calling `search.register_backend(name, factory)`.

After each novelty refuter is complete, the orchestrator looks up every reference in its structured block by DOI, arXiv ID or title, and checks its quoted passages. A check interrupted by a cancel or crash is redone before the verifiers and judges start.

Optional but strongly advised: `S2_API_KEY` for a usable Semantic Scholar rate limit, and `OPENALEX_API_KEY` (free, created at openalex.org in a minute) for OpenAlex. Since February 2026 OpenAlex bills a daily budget per key: without one it covers about 100 searches a day (about two reviews; it ran out in the K2 run on 2026-10-02), with a free key about 1,000. A 429 whose credit headers say the budget is spent pauses OpenAlex until the reset they give, without retrying. A run missing either key gets a preflight warning. `PAPER_ADVERSARY_CONTACT_EMAIL` joins the OpenAlex/Crossref polite pool.

**Semantic Scholar without a key.** Keyless requests share a public pool that answers HTTP 429 in bursts. In the K1 pilot, 37 of the refuters' 38 Semantic Scholar calls found it unavailable, behind a 600 s pause that outlasted their searches. Without a key, a 429 burst now gets four jittered retries (about a minute; a `Retry-After` header is honoured in seconds or as an HTTP date, up to 60 s) before Semantic Scholar is paused for 90 s. After the pause one probe request goes through: if it is answered, Semantic Scholar is open again for every agent; if not, it is paused again. With a key, requests are spaced 1.2 s apart (the key's limit is 1 request per second; a live test on 2026-10-02 still drew an occasional 429 just above 1 s), a 429 gets three retries, and a 429 that persists through them pauses Semantic Scholar for 120 s, after which one probe goes through: in the K2 run the earlier 600 s pause outlasted the whole novelty phase. OpenAlex follows the same short half-open pause for a 429 burst. A run whose literature agents would use Semantic Scholar without `S2_API_KEY` gets a preflight warning, in its events and its status (with its time); the next preflight after the key is set removes it.

**Search health.** After the novelty refuters, the orchestrator counts per provider, from each refuter's search log (the attempt that wrote its report), how many literature calls it answered and how many found it unavailable. The counts are kept in `state.json` (`search_health`) and shown in the status. A provider unavailable for at least a quarter of the calls that asked it counts as degraded: a `search_degraded` event, and one line under `<search_coverage>` for the judges, the memo, the critic and the follow-up agents, for example "semantic_scholar was unavailable for 37 of 38 calls (97%; N1 20 of 21, N2 17 of 17)". The line is a fact the orchestrator recorded, not an agent's output, so the isolation rules allow it. Each agent's `*.search_log.jsonl` has a row for every literature and prior-text call, including prior-text calls that were refused or found no full text.

## Cost and usage

`get_run_status` also shows how much of your plan's 5-hour and 7-day windows is used, and whether paid overage is on, as Claude Code last reported. Fable 5.1 also has a weekly window of its own (`seven_day_overage_included`, reported only by Fable calls); the status shows it as "Fable 5.1 (model-specific) weekly window", with its use and reset, from the last Fable call. What happens at a limit is described under [Plan limits](#plan-limits).

**Subscription usage is not dollars.** On a plan, a run consumes your 5-hour and 7-day allowances, which Claude Code reports only as percentages after calls are made. Nothing here can promise a dollar cap on them. What controls usage:
- the preset;
- agent counts and efforts;
- the verification budgets;
- the number of follow-up rounds;
- the plan-limit wait cap.

All of these take effect before calls are dispatched. There is no run-wide token or call budget.

`get_run_cost` reports input, cache-write, cache-read and output tokens, web searches and agent time per phase: verifiers, format repairs, coverage supplements, memo placement fixes and each follow-up round are listed separately. It shows two cost figures: the API-equivalent cost Claude Code reports, and an estimate from `config/models.yaml` prices. On a plan neither is billed per token; they show how heavy a run was. Calls that returned no usage (for example, killed on timeout) are listed as missing, never estimated. Output tokens spent on hidden turns, whose text no report contains, are listed per agent with an estimate at the output price (K1 pilot: J1, 128,000 tokens).

### Plan limits

Claude Code reports which plan window refused a call. A limit on one model (the rejected window is `seven_day_overage_included`, Fable's weekly one, or the message says "You've reached your Fable limit") refuses only that model's calls. Other limits (5-hour, 7-day) refuse every call.

- **Waiting.** With `plan_limit.policy: wait`, the worker waits until the reported reset (`default_wait_minutes` when none is reported), then continues. At most `plan_limit.max_waits_per_run` reset windows (default 1) are waited out per run. Agents hitting the same window count once; a window without a reported reset time is keyed by limit type, and by model for a model-specific limit (account-wide ones share one key), so agents hitting it while its wait lasts count once and a hit after it is a new window.
- **A reset beyond `max_wait_hours`** (default 6; Fable's weekly window resets days away) is not waited for, and no wait is counted for it. The agents are interrupted, not failed, with the model, the limit type and the reset time in their status (an agent whose coverage supplement, format repair or placement fix hit the limit stays `gating`, with the same detail), a `plan_limit_skip` event is written, and the job ends `incomplete`. `resume_run` continues after the reset. With the waits used up the same happens, with a `plan_limit_stop` event.
- **A model-specific limit stops only that model.** Its agents are interrupted when they would start; agents on other models keep running. A later stage starts only when its inputs are complete: judges need every refuter, a verification batch or follow-up step with an interrupted agent stays open, and the memo waits for the automatic gate batch. An account-wide limit stops the whole job.
- **Window guard.** `plan_limit.model_window_reserve` is the share of a model-specific window one run is expected to use: 0.15 in `stage2_split` (the K1 pilot used 15% of Fable's weekly window), 0 (off) elsewhere; it can be set only in trusted settings. Before a run's first agent on such a model starts, the job stops `incomplete` when the window's last known use plus the reserve exceeds 100%; at the start of a job, that is before any of its agents runs. The last known use is the newest one Claude Code reported to this run (the guard is checked again after the job's preflight probes, which report the windows too), to any run in the same runs directory, or to `validate --probe` run with that directory (`.cache/plan_windows.json`); a window whose reset time has passed is not counted. The reserve is an estimate, and the window is shared with everything else you do on Fable, so the guard can stop a run that would have fit, or miss one that does not. A run that has already started agents on the model is not stopped by the guard.

## Reliability

- Transient failures get exponential backoff with jitter: overload, 5xx, network, timeouts, silent streams, empty output.
- Plan usage limits are waited out until the reset time the CLI reports, then the run continues, within `plan_limit.max_wait_hours` and at most `plan_limit.max_waits_per_run` resets per run. A later reset stops the job resumably, and a model-specific limit stops only that model's agents ([Plan limits](#plan-limits)).
- Each model turn's stop reason and output tokens are kept in the transcript, so a turn whose text the report lacks is reported (gates table). `--max-turns 1` is not used for this: Claude Code treats hitting it as a permanent error, which would discard a good second turn after the first one had been paid for.
- Auth failures and API-key billing stop the run immediately.
- A missing model, a substituted model or a broken tool setup is caught by the preflight probe.
- An oversized prompt is retried once in sectioned mode.
- Outputs are saved before they are checked; a crash during checks is finished on resume without rerunning the agent. Gate decisions are recorded before they are applied, so a crash in between applies the recorded decision.
- Parsing of agent output and of downloaded documents runs in linear time, so hostile text (thousands of unclosed fences or comments) cannot stall the worker.
- All state writes are atomic. One worker per run is enforced by an OS lock that dies with the process. `resume_run` retries failed or interrupted agents, finishes interrupted checks and follow-up rounds, and skips completed work.
- Resuming also reruns complete agents whose inputs changed since they ran (for example a refuter you reran), then each later stage whose inputs that changed, so a change carries through. Unchanged outputs and literature snapshots are reused.

## Tests

```bash
.venv/bin/python -m pytest
```

The suite uses no plan usage and no network. It runs the full pipeline with an offline mock provider whose reports carry canaries, proving from the outputs which agent saw what. It injects every gate failure: audit breaches, missing transcripts, bad JSON, truncation, substitution and coverage gaps. It runs blind verification, the evidence gate and follow-up rounds end to end, including the smoke run's regression case: a FATAL overlap claimed from an abstract comes out labelled unverified and filed under unverified threats.

It also covers:
- the presets, per-agent overrides and per-run limits, saved assignments, and runs created before presets;
- cross-model verifier routing, the verification budgets, coverage-first selection (authors taking turns, papers serving more objections first) and the reasons it records, second passages, references past the limit, escalated objections and the automatic gate batch, failed verifiers collected when rerun, gate re-checks that leave the judges current, and runs planned before second passages;
- per-member severities in grouped judgments, through the matrix, the evidence gate and the open issues, on the K1 pilot's grouping pattern, `judge_v3` groups, members rated above their judgment and judgments relying on prior work through a lower-rated member; open issues counted as issues; disputes of computed verdicts and of rubric tests, and their correction; re-check items at the round cap and below the floor;
- rubric tests: their declaration, the judges' rulings, the supplement and quarantine, the matrix table, and FATAL rulings among the open issues;
- the refuters' rubric (new prompt versions only), claims-ledger IDs per prompt set and old ledgers, ledger claims a critic cites in words, the agent-ID note in every prompt, and follow-up prompts and manifests (each input once, and only what the agent is shown);
- the `stage2_split` preset end to end, and a Stage-2 regression run (`tests/test_stage2_regressions.py`): rubric tests, grouped judgments, a critic's quoted novelty item, the automatic gate batch, a follow-up round, and a Fable limit while Opus agents go on;
- peer blindness with mixed models, including format repairs, coverage supplements and follow-up calls;
- stale reruns on resume and the plan-limit wait cap;
- plan limits: a reset beyond `max_wait_hours` (no wait counted, agents interrupted, resumable), three parallel Fable judges at Fable's weekly limit, agents on other models running on, resets without a time counted once, account-wide stops, a limit in a gate's side call, and the window guard across runs and after the job's own probes;
- hidden turns: per-turn stop reasons from the stream and the usage fallback, retried messages and num_turns, the gate's warning and quarantine (an invalid block counts as present), and their tokens in the cost report;
- search health and the coverage line, keyless Semantic Scholar's backoff and half-open breaker (a failed probe pauses it again), `Retry-After` dates, the missing-key warning and its removal, a search-log row for every prior-text call, and preflight records merged across calls (failed probes, context windows kept);
- the tool sentence of an agent with the PDF, and of its coverage supplement;
- the evaluation harness on a synthetic case;
- the quote matcher;
- full-text retrieval against canned HTTP;
- sandboxed extraction;
- the Claude Code backend against a fake `claude` binary;
- the scholarly APIs;
- detached workers with cancel and resume.

## Known limitations

- **Plan usage.** Your account is on the Pro plan.
  - **`routine`:** 10 base agents (4 on Fable 5.1, one of them at xhigh), up to 3 verifiers, and at most one follow-up round of 4 calls.
  - **`milestone`:** 13 base agents (6 on Fable 5.1), up to 6 verifiers, and up to two rounds.
  - **`legacy_thorough`:** 16 base agents (9 on Fable, 8 of them at max or xhigh), up to 8 verifiers, and up to two all-Fable rounds.
  - **Verifiers after the judges:** up to 2 more per follow-up round, and 2 per gate re-check (`run_verification`, or the automatic gate batch where it is on).
  - **Extra calls:** a judge, adjudicator or memo that needs a coverage supplement or placement fix costs one more call on its own model.
  - **Untested savings:** none of this says how the presets compare in quality; that is what the evaluation harness is for.
  - **Fable's weekly window:** one `stage2_split` run used 15% of it in the K1 pilot, so about six such runs fit in one week. The window guard stops a run before it starts when it would not fit; schedule a batch accordingly.
- **Presets are untested allocations.** `routine` and `milestone` follow a proposed design. Whether they find as much as `legacy_thorough` is unknown until adjudicated cases are run (`eval/README.md`).
- **No independent provider.** Every agent is a Claude model. A milestone review could add a reviewer from another provider, as an extra adjudicator or a second critic. It would have the same isolation, gates and evidence rules, so model-family blind spots get a different check. No such adapter exists yet. It would need its own provider class, tests against a fake binary, and a usage policy before use.
- **Paywalled prior work** cannot be verified until you supply the PDF (`add_prior_fulltext`). Until then, verdicts that rest on it stay labelled unverified.
- **Text that is hard to check.** Scanned PDFs without a text layer are not OCR'd. Quotes of mathematics match poorly; the prompts ask for the prose around an equation.
- **Copyright.** Run folders contain text snapshots of third-party papers (for personal research use, which arXiv's terms allow); do not redistribute them.
- **CLI version.** Opus 5.5 needs Claude Code 2.1.280 or newer. With an older CLI, the preflight stops the run with Claude Code's own "run `claude update`" message.
- **Web search** inside `claude -p` depends on Claude Code's WebSearch tool being available to your account.
