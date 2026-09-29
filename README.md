# paper-adversary-mcp

A local MCP server for Claude Desktop that stress-tests a research paper or research idea before submission. It runs an adversarial review with several independent agents and keeps them isolated from each other in code.

```
                ┌─ N1..N4  novelty refuters      (web + scholarly search)
paper ──────────┼─ R1..R3  rigor refuters        (+ the PDF, for equations/tables)
  │             └─ F1..F3  fit / feasibility refuters
  └─ intake (claims ledger)          │  refuters never see each other
                                     ▼
                    J1 J2 J3  judges (all refuter reports + rubric; never each other)
                                     ▼
                    judgment matrix (computed, not averaged)
                                     ▼
                    S1  synthesis memo (everything)
                                     ▼
                    C1  completeness critic (everything + memo)
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

4. Check everything, including one tiny low-effort call per model on your plan:

   ```bash
   .venv/bin/python -m paper_adversary validate --probe
   ```

## Using it from Claude Desktop

Ask in plain language, for example "Run a full adversarial review of /path/paper.pdf for ICML 2027". Or call the tools directly:

| Tool | What it does |
|---|---|
| `create_review_run` | Ingest a paper (PDF, Markdown, LaTeX `.tex`, text) or pasted idea; record metadata; plan agents. Starts nothing. |
| `run_refuters` | Run novelty, rigor and fit refuters (`phase`: all, novelty, rigor or fit) in a background worker. |
| `run_judges` | Judges over all refuter reports. Requires the refuters to be complete (or `allow_incomplete_refuters`). |
| `run_synthesis` | The synthesis memo. |
| `run_completeness_critic` | The critic that hunts for what everyone missed. |
| `run_full_review` | Create the run and execute everything in one background worker. |
| `resume_run` | Continue an interrupted or partial run; completed agents are never redone. |
| `get_run_status` | Per-agent status, failed calls, retry counts, isolation audit, stale outputs, artifact paths. `wait_seconds` blocks up to 10 min with progress updates. |
| `get_report` | One output at a time: `novelty`, `rigor`, `fit`, `judge`, `synthesis`, `critic` (with `agent_id` for a single report), `matrix`, `claims`, `refcheck`, `paper`, `sections`, `references`, `usage`, `events`, `index`, or per-agent `context`, `transcript`, `prompt`, `search_log`. Long texts are paged. |
| `get_run_cost` | Tokens and cost by phase and in total. |
| `cancel_run` | Stop a run's worker and every agent process it started. |
| `list_runs`, `validate_config` | Housekeeping. |

A full review takes tens of minutes, longer at high effort, so the `run_*` tools return immediately. The worker is a separate process. It keeps running if Claude Desktop quits, and `cancel_run` stops it.

The same operations work from a terminal:

```bash
.venv/bin/python -m paper_adversary review paper.pdf --venue "ICML 2027"   # detached
.venv/bin/python -m paper_adversary status <run_id>
.venv/bin/python -m paper_adversary report <run_id> synthesis
.venv/bin/python -m paper_adversary refuters <run_id> --rerun N2 --foreground
```

## Configuration

`config/default.yaml` holds the pipeline and `config/models.yaml` holds the models. The orchestration code never names a model. Roles refer to aliases (`fable-5.1`, `opus-5.5`), and `models.yaml` maps each alias to its model ID, documented limits, supported effort levels and API-equivalent prices. Defaults follow your split:

| Role | Model | Effort | Agents |
|---|---|---|---|
| orchestrator (intake) | fable-5.1 | high | 1 |
| novelty | opus-5.5 | high | 4 |
| rigor | fable-5.1 | max | 3 |
| fit | opus-5.5 | high | 3 |
| judge | fable-5.1 | max | 3 |
| synthesis | fable-5.1 | max | 1 |
| critic | fable-5.1 | xhigh | 1 |

Override per run with `config_override` (an object, YAML text or a file path), e.g. `{"novelty": {"agents": 2}, "concurrency": {"max_parallel_agents": 5}}`. Set `$PAPER_ADVERSARY_CONFIG` for your own standing defaults. Unknown keys, unsupported efforts and unknown aliases are rejected before anything runs.

Per-run overrides come through MCP tool arguments, which a model chooses after reading untrusted paper text. So they cannot touch the security-relevant provider keys: `allow_api_key`, `extra_args`, `env_passthrough`, `claude_binary` and `safe_mode`. Those can only be set in `config/default.yaml` or your `$PAPER_ADVERSARY_CONFIG` file. Flags that would widen an agent's access (`--add-dir`, `--permission-mode`, `--settings`, `--mcp-config`, …) are rejected in `extra_args` in any config.

Validation against what your plan actually serves happens in two places:
- **Before each run**, the Claude Code login is checked for free, and a run on an API-key login is refused. Then a preflight probe sends one tiny low-effort request per distinct model and tool setup, cached for 24 h. It uses the same tools and MCP server as the real agents, so a broken tool setup stops the run before any real agent is spent. It also records which model answered and the context window Claude Code reports.
- **During the run**, if any turn is served by a different model than requested (for example a fallback), the report and status are flagged `MODEL SUBSTITUTION`.

Main knobs: `concurrency.max_parallel_agents` (default 3, plan-friendly), `retry.*`, `plan_limit.policy` (`wait` sleeps until a usage limit resets, then continues; `fail` stops), `search.providers`, `lens_set`, `rubric`.

## Isolation

This is enforced in software, in layers:

1. **Access policy** (`isolation.py`, `ROLE_VISIBILITY`). The context builder can read prior outputs only through a policy-checked reader. Refuters may read only the paper. Judges may read the paper, rubric, refuter reports and reference checks, never another judge. Synthesis and the critic see everything upstream. The table lives in code, so no config can loosen it.
2. **Pre-flight guard.** Every report carries a marker, and distinctive word 8-grams of each report are registered. Before an agent starts, its fully assembled prompt is scanned for fingerprints of every existing artifact its role must not see. A hit aborts the agent. Phrases that also occur in inputs the agent may legitimately see don't count as evidence. For example, when a judge quotes a refuter, that quote is also in every other judge's legitimate input.
3. **Process sandbox.** Each agent is its own `claude -p` process. It runs in a fresh, empty folder outside the project, with `--restricted` (no shell, file tools confined to that folder) and an explicit tool allowlist. It uses `--strict-mcp-config`, `--safe-mode` when no tools are needed, and `--no-session-persistence`, with a scrubbed environment. When the source is a PDF, rigor agents and judges get a copy of it (and nothing else) in their folder.
4. **Transcript audit.** After each agent, its transcript is checked for file access outside its sandbox and for forbidden fingerprints in tool results. The verdict is in each report's metadata and in `get_run_status`.
5. **Manifests.** Each agent's `*.context.json` lists every input it received, with its hash. Rerunning an upstream agent marks the downstream outputs that used the old version as stale.
6. **No side channels.** The intake pass's reading of the paper (title, field, claims) is stored separately and reaches only synthesis and the critic. Refuter and judge headers use only what you supplied or what was extracted from the file.

## Storage

```
runs/<run_id>/
  metadata.json  config.yaml  state.json
  source/     paper.pdf|md|tex, extracted_text.md, sections.json, references.json, profile.md, claims_ledger.md
  novelty/    N1.md (+ N1.json structured block, N1.refcheck.json, N1.search_log.jsonl, N1.context.json)
  rigor/  fit/  judges/ (+ judgment_matrix.md/.json)  synthesis/memo.md  critic/completeness.md
  logs/       events.jsonl, usage.jsonl, api_usage.json, worker.log, agents/<id>/attempt-<n>/{system_prompt.md,
              user_prompt.md, transcript.jsonl, stderr.log, context.json}
  archive/    superseded outputs (reruns never delete)
```

Every report starts with YAML front matter. It records the model requested and the models that actually served it, effort, prompt version and SHA-256, lens, timestamp, run ID, role, usage, attempts, isolation audit, and warnings.

## Prompts and experiments

Prompts live in `prompts/` as `<role>_v<N>.md` (front matter + template). Lenses are in `lenses_v1.yaml` and rubrics in `rubrics/`. To try a variant, copy `novelty_v1.md` to `novelty_v2.md`, edit it, and run with `config_override={"novelty": {"prompt": "novelty_v2"}}`. Each report records the prompt version and the file's hash, so in-place edits are detectable too.

## Long papers

If a paper does not fit an agent's budget, it is never silently truncated. The budget is the context window minus output reserve, safety margin and other inputs. The role's priority sections stay inline in document order, every other section becomes an explicit placeholder, and the agent gets a `read_paper_section` tool that returns any section in full. The plan is recorded per agent. Token estimates start from a conservative heuristic and are then calibrated from the usage agents report.

## Novelty search

Novelty refuters get Claude Code's web search and fetch, plus a small MCP tool server (`tools_server.py`). It offers `search_literature`, `lookup_paper` and `find_citing_papers` over OpenAlex, Semantic Scholar, arXiv and Crossref. Queries are fanned out and de-duplicated, cached on disk, rate-limited per provider across all agents, and protected by circuit breakers (arXiv blocks heavy users for hours, so it gets a slow gate). Add a provider by implementing `search.base.SearchBackend` and calling `search.register_backend(name, factory)`.

After each novelty refuter finishes and is marked complete, the orchestrator looks up every reference in its structured block by DOI, arXiv ID or title. A check interrupted by a cancel or crash is redone before the judges start. Judges see which citations were verified, partially matched, mismatched or not found.

Optional: `S2_API_KEY` for a higher Semantic Scholar rate limit; `PAPER_ADVERSARY_CONTACT_EMAIL` for the OpenAlex/Crossref polite pool.

## Cost and usage

`get_run_status` also shows how much of your plan's 5-hour and 7-day windows is used, and whether paid overage is on, as Claude Code last reported. When a plan limit hits, the worker waits until the reset time Claude Code reports.

`get_run_cost` reports input, cache-write, cache-read and output tokens, web searches and agent time per phase. It shows two cost figures: the API-equivalent cost Claude Code reports, and an estimate from `config/models.yaml` prices. On a plan neither is billed per token; they show how heavy a run was. Calls that returned no usage (for example, killed on timeout) are listed as missing, never estimated.

## Reliability

- Transient failures get exponential backoff with jitter: overload, 5xx, network, timeouts, silent streams, empty output.
- Plan usage limits are waited out until the reset time the CLI reports, then the run continues, within `plan_limit.max_wait_hours`.
- Auth failures and API-key billing stop the run immediately.
- A missing model is caught by the preflight probe.
- An oversized prompt is retried once in sectioned mode.
- All state writes are atomic. One worker per run is enforced by an OS lock that dies with the process. `resume_run` retries failed or interrupted agents and skips completed ones.

## Tests

```bash
.venv/bin/python -m pytest
```

The suite uses no plan usage. It runs the full pipeline with an offline mock provider whose reports carry canaries, proving from the outputs which agent saw what. It also exercises the Claude Code backend against a fake `claude` binary (flags, sandbox, environment scrubbing, error classification), the scholarly APIs against canned responses, and detached workers with cancel and resume.

## Known limitations

- Your account is on the Pro plan. A full default run is 16 agents: 9 on Fable 5.1, 8 of them at max or xhigh effort, plus 7 on Opus 5.5. It may hit plan limits; the worker then waits for the reset. Lower `agents` or effort per run if you need results faster.
- Opus 5.5 needs Claude Code 2.1.280 or newer. With an older CLI, the preflight stops the run with Claude Code's own "run `claude update`" message.
- Web search inside `claude -p` depends on Claude Code's WebSearch tool being available to your account.
