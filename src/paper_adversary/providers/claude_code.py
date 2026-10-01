"""Run each agent as a headless Claude Code process (`claude -p`) on the user's Claude plan.

Isolation at the process level:
  * a fresh, empty sandbox folder outside the project as the working directory;
  * --restricted: no shell or code-running tools, file tools confined to the
    sandbox, user/project/local settings ignored;
  * --tools allowlist per role, --strict-mcp-config (only our tool server),
    --disable-slash-commands, --no-session-persistence, --permission-prompts none;
  * --safe-mode (no CLAUDE.md, skills, plugins or hooks) when the agent needs no MCP tools;
  * a scrubbed environment: ANTHROPIC_API_KEY is never passed unless
    provider.allow_api_key is set, and the init event is checked so a key can
    never bill silently.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

from paper_adversary.config import ProviderConfig
from paper_adversary.providers.base import AgentRequest, AgentResult, ErrorKind, ProviderError
from paper_adversary.registry import normalize_model_id, substituted_models
from paper_adversary.util import append_jsonl, atomic_write_json, atomic_write_text, read_jsonl, utcnow

BUILTIN_TOOL = {"web_search": "WebSearch", "web_fetch": "WebFetch", "read_pdf": "Read"}
MCP_SERVER = "lit"
PROBE_WORD = "PINEAPPLE"  # the probe's system prompt asks for this word, proving --system-prompt-file took effect
# Built-ins Claude Code attaches whenever an MCP server is configured (List/Read/ReadDir of MCP resources).
# They can only reach our own tool server, which serves no resources.
MCP_RESOURCE_TOOL = re.compile(r"^(List|Read)McpResource\w*Tool$")
TOOL_RESULT_KEEP = 200_000  # chars of each tool result kept in the transcript (used by the isolation audit)

_ENV_KEEP = (
    "HOME", "USER", "LOGNAME", "PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TMPDIR", "TMP", "TEMP",
    "SHELL", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SYSTEMROOT", "SYSTEMDRIVE", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "COMSPEC", "PATHEXT", "WINDIR",
    "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR",
)


def _find_binary(configured: str | None) -> str:
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        found = shutil.which(configured)
        if found:
            return found
        raise ProviderError(ErrorKind.TOOL_SETUP, f"claude binary not found at {configured}")
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (Path.home() / ".local/bin/claude", Path.home() / ".claude/local/claude",
                      Path("/usr/local/bin/claude"), Path("/opt/homebrew/bin/claude")):
        if candidate.is_file():
            return str(candidate)
    raise ProviderError(ErrorKind.TOOL_SETUP, "Claude Code CLI ('claude') not found; install it or set "
                        "provider.claude_binary in the config")


class ClaudeCodeProvider:
    name = "claude_code"

    def __init__(self, cfg: ProviderConfig, sandbox_root: Path | None = None):
        self.cfg = cfg
        uid = os.getuid() if hasattr(os, "getuid") else os.environ.get("USERNAME", "user")
        self.sandbox_root = sandbox_root or Path(tempfile.gettempdir()) / f"paper-adversary-{uid}"

    # ------------------------------------------------------------ setup

    def binary(self) -> str:
        return _find_binary(self.cfg.claude_binary)

    def child_env(self, max_output_tokens: int | None = None) -> dict[str, str]:
        env = {k: os.environ[k] for k in _ENV_KEEP if os.environ.get(k)}
        for key in self.cfg.env_passthrough:
            if key in os.environ:
                env[key] = os.environ[key]
        if self.cfg.allow_api_key:
            for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
                if key in os.environ:
                    env[key] = os.environ[key]
        else:
            for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
                env.pop(key, None)
        bin_dir = str(Path(self.binary()).parent)
        parts = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
        if bin_dir not in parts:
            env["PATH"] = os.pathsep.join([bin_dir, *parts]) if parts else bin_dir
        env["DISABLE_AUTOUPDATER"] = "1"
        if max_output_tokens:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
        return env

    def _command(self, req: AgentRequest, system_prompt_file: Path, mcp_config: Path | None) -> list[str]:
        builtin = [BUILTIN_TOOL[t] for t in req.tools if t in BUILTIN_TOOL]
        cmd = [
            self.binary(), "-p",
            "--model", req.model,
            "--effort", req.effort,
            "--output-format", "stream-json", "--verbose", "--include-partial-messages",
            "--no-session-persistence",
            "--restricted",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--permission-prompts", "none",
            "--system-prompt-file", str(system_prompt_file),
            "--tools", ",".join(builtin),
        ]
        allowed = list(builtin)
        if mcp_config is not None:
            cmd += ["--mcp-config", str(mcp_config)]
            allowed.append(f"mcp__{MCP_SERVER}")
        elif self.cfg.safe_mode:
            cmd.append("--safe-mode")
        if allowed:
            cmd += ["--allowedTools", ",".join(allowed)]
        if req.max_turns:
            cmd += ["--max-turns", str(req.max_turns)]
        cmd += list(self.cfg.extra_args)
        return cmd

    def _prepare(self, req: AgentRequest) -> tuple[Path, Path, Path | None]:
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix=f"{req.run_id[:40]}-{req.agent_id}-", dir=self.sandbox_root))
        if req.pdf_path and "read_pdf" in req.tools:
            shutil.copy2(req.pdf_path, sandbox / "paper.pdf")
        req.log_dir.mkdir(parents=True, exist_ok=True)
        system_file = req.log_dir / "system_prompt.md"
        atomic_write_text(system_file, req.system_prompt)
        atomic_write_text(req.log_dir / "user_prompt.md", req.user_text)
        mcp_config = None
        if req.tool_server:
            mcp_config = req.log_dir / "mcp_config.json"
            atomic_write_json(mcp_config, {"mcpServers": {MCP_SERVER: {
                "type": "stdio",
                "command": req.tool_server.get("command", sys.executable),
                "args": req.tool_server["args"],
                "env": req.tool_server.get("env", {}),
            }}})
        return sandbox, system_file, mcp_config

    def cleanup(self, result: AgentResult) -> None:
        if result.sandbox_dir and result.sandbox_dir.exists():
            shutil.rmtree(result.sandbox_dir, ignore_errors=True)

    # ------------------------------------------------------------ run

    async def run(self, req: AgentRequest, cancel: asyncio.Event | None = None) -> AgentResult:
        sandbox, system_file, mcp_config = self._prepare(req)
        try:
            return await self._run(req, cancel, sandbox, system_file, mcp_config)
        except BaseException:
            shutil.rmtree(sandbox, ignore_errors=True)
            raise

    async def _run(self, req: AgentRequest, cancel: asyncio.Event | None, sandbox: Path, system_file: Path,
                   mcp_config: Path | None) -> AgentResult:
        cmd = self._command(req, system_file, mcp_config)
        env = self.child_env(req.max_output_tokens)
        transcript = req.log_dir / "transcript.jsonl"
        atomic_write_json(req.log_dir / "command.json", {"argv": cmd, "cwd": str(sandbox),
                                                         "env_keys": sorted(env)})
        state = _StreamState(req, transcript, self.cfg.allow_api_key)
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(sandbox), env=env, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=64 * 1024 * 1024,
            )
        except OSError as exc:
            raise ProviderError(ErrorKind.TOOL_SETUP, f"could not start claude: {exc}") from exc

        stderr_tail: deque[str] = deque(maxlen=200)

        async def pump_stderr() -> None:
            assert proc.stderr is not None
            with open(req.log_dir / "stderr.log", "a", encoding="utf-8") as fh:
                while True:
                    raw = await proc.stderr.readline()
                    if not raw:
                        return
                    line = raw.decode("utf-8", "replace")
                    fh.write(line)
                    stderr_tail.append(line.rstrip())

        async def pump_stdout() -> None:
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    return
                state.last_activity = time.monotonic()
                try:
                    state.handle(raw)
                except Exception as exc:  # a malformed event must not stall the stream
                    state.log({"kind": "parse_error", "error": f"{type(exc).__name__}: {exc}", "line": raw[:500].decode("utf-8", "replace")})
                if state.abort is not None:
                    await _terminate(proc)  # right away: do not let a guarded session send a request
                    return

        async def feed_stdin() -> None:
            assert proc.stdin is not None
            try:
                proc.stdin.write(req.user_text.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                proc.stdin.close()

        stdin_task = asyncio.create_task(feed_stdin())
        err_task = asyncio.create_task(pump_stderr())
        out_task = asyncio.create_task(pump_stdout())
        failure: ProviderError | None = None
        try:
            while not out_task.done():
                await asyncio.wait({out_task}, timeout=2.0)
                now = time.monotonic()
                if cancel is not None and cancel.is_set():
                    failure = ProviderError(ErrorKind.CANCELLED, "cancelled")
                elif state.abort is not None:
                    failure = state.abort
                elif now - started > req.timeout_s:
                    failure = ProviderError(ErrorKind.TIMEOUT, f"no result after {req.timeout_s / 60:.0f} min")
                elif now - state.last_activity > req.idle_timeout_s:
                    failure = ProviderError(ErrorKind.IDLE_TIMEOUT,
                                            f"no output for {req.idle_timeout_s / 60:.0f} min")
                if failure is not None:
                    await _terminate(proc)
                    break
            await asyncio.wait({out_task, err_task, stdin_task}, timeout=15)
            await asyncio.wait_for(proc.wait(), timeout=15)
        except asyncio.CancelledError:
            await _terminate(proc)
            raise
        except asyncio.TimeoutError:
            await _terminate(proc)
        finally:
            for task in (out_task, err_task, stdin_task):
                if not task.done():
                    task.cancel()

        stderr_text = "\n".join(stderr_tail)
        if failure is None and state.abort is not None:
            failure = state.abort
        if failure is not None:
            failure.usage = state.usage_or_none()
            raise failure
        return state.finish(proc.returncode, stderr_text, sandbox)

    # ------------------------------------------------------------ helpers

    async def auth_status(self) -> dict:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                self.binary(), "auth", "status", env=self.child_env(), stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
        except (OSError, asyncio.TimeoutError, ProviderError) as exc:
            if proc is not None:
                await _terminate(proc)
            return {"ok": False, "error": str(exc) or type(exc).__name__}
        try:
            data = json.loads(out.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return {"ok": False, "error": (out or err).decode("utf-8", "replace")[:500]}
        keep = {k: data.get(k) for k in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")}
        return {"ok": bool(data.get("loggedIn")), **keep,
                "oauth_token_in_env": bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")),
                "api_key_in_server_env": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "api_key_passed_to_agents": bool(self.cfg.allow_api_key and os.environ.get("ANTHROPIC_API_KEY"))}

    async def probe(self, model: str, log_dir: Path, tools: list[str] | tuple = (),
                    tool_server: dict | None = None) -> dict:
        """One tiny low-effort request with the same tool setup as a real agent (so its init is checked too)."""
        req = AgentRequest(run_id="probe", agent_id="PROBE", role="probe", model=model, effort="low",
                           system_prompt=f"You are a connectivity probe. Whatever the user writes, reply with exactly "
                                         f"one word: {PROBE_WORD}. Do not use tools.",
                           user_text="Reply with the single word your instructions specify.", log_dir=log_dir,
                           tools=list(tools), tool_server=tool_server, timeout_s=300, idle_timeout_s=240)
        out: dict = {"model_requested": model, "tools": sorted(tools),
                     "checked_at": utcnow().isoformat(timespec="seconds")}
        try:
            result = await self.run(req)
        except ProviderError as exc:
            out.update(ok=False, error_kind=exc.kind.value, error=exc.message, reset_at=exc.reset_at,
                       rate_limit=exc.detail.get("rate_limit"))
            return out
        self.cleanup(result)
        served = [m for m in result.served_models if m]
        ctx = None
        for key, val in (result.model_usage or {}).items():
            if normalize_model_id(key).startswith(normalize_model_id(model)) and isinstance(val, dict):
                ctx = val.get("contextWindow") or ctx
        applied = PROBE_WORD in result.text.upper()
        kinds = {ev.get("kind") for ev in read_jsonl(result.transcript_path)} if result.transcript_path else set()
        out.update(ok=applied, served_models=served, context_window=ctx, warnings=result.warnings,
                   runtime=result.runtime, reply=result.text[:80], system_prompt_applied=applied,
                   substituted=substituted_models(model, served))
        if not applied:  # agents would silently lose their role instructions and output format
            out.update(error_kind=ErrorKind.TOOL_SETUP.value,
                       error=f"the agent system prompt was not applied (reply: {result.text[:60]!r}); try "
                             "provider.safe_mode: false in config/default.yaml")
        elif not {"init", "result"} <= kinds:  # without them every agent's isolation audit is unverifiable
            out.update(ok=False, error_kind=ErrorKind.ISOLATION.value,
                       error="Claude Code's output stream had no init or result event, so agents' isolation "
                             "could not be audited; check `claude --version` and `claude update`")
        return out


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=10)
    except (ProcessLookupError, asyncio.TimeoutError):
        try:
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (ProcessLookupError, asyncio.TimeoutError):
            pass


class _StreamState:
    """Incremental parser for `claude -p --output-format stream-json` events."""

    def __init__(self, req: AgentRequest, transcript: Path, allow_api_key: bool):
        self.req = req
        self.transcript = transcript
        self.allow_api_key = allow_api_key
        self.last_activity = time.monotonic()
        self.init: dict | None = None
        self.result: dict | None = None
        self.last_text: list[str] = []
        self.served: list[str] = []
        self.tool_calls: list[dict] = []
        self.error_codes: list[str] = []
        self.error_texts: list[str] = []
        self.warnings: list[str] = []
        self.abort: ProviderError | None = None
        self.rate_limit: dict | None = None  # latest plan-usage snapshot (rate_limit_event.rate_limit_info)

    def log(self, entry: dict) -> None:
        append_jsonl(self.transcript, {"at": utcnow().isoformat(timespec="seconds"), **entry})

    def handle(self, raw: bytes) -> None:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            return
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            self.log({"kind": "unparsed", "line": line[:2000]})
            return
        if not isinstance(ev, dict):
            self.log({"kind": "unparsed", "line": line[:2000]})
            return
        etype = ev.get("type")
        if etype == "stream_event":
            return
        if etype == "system" and ev.get("subtype") == "init":
            self._on_init(ev)
        elif etype == "assistant":
            self._on_assistant(ev)
        elif etype == "user":
            self._on_user(ev)
        elif etype == "result":
            self.result = ev
            self.log({"kind": "result", **{k: ev.get(k) for k in (
                "subtype", "is_error", "stop_reason", "terminal_reason", "api_error_status", "num_turns",
                "duration_ms", "duration_api_ms", "total_cost_usd", "usage", "modelUsage", "permission_denials")}})
        elif etype == "system":
            self.log({"kind": "system", "subtype": ev.get("subtype"),
                      "data": {k: v for k, v in ev.items() if k not in {"type", "subtype", "session_id", "uuid"}}})
        else:
            if etype == "rate_limit_event" and isinstance(ev.get("rate_limit_info"), dict):
                self.rate_limit = ev["rate_limit_info"]
            self.log({"kind": "event", "type": etype, "data": ev})

    def _on_init(self, ev: dict) -> None:
        self.init = ev
        self.log({"kind": "init", **{k: ev.get(k) for k in (
            "model", "tools", "mcp_servers", "apiKeySource", "permissionMode", "claude_code_version", "cwd")}})
        source = ev.get("apiKeySource")
        if source not in (None, "none") and not self.allow_api_key:
            self.abort = ProviderError(ErrorKind.BILLING_GUARD,
                                       f"Claude Code would bill an API key ({source}) instead of your plan; "
                                       "unset it or set provider.allow_api_key")
            return
        expected = {BUILTIN_TOOL[t] for t in self.req.tools if t in BUILTIN_TOOL}
        tools = set(ev.get("tools") or [])
        builtin_seen = {t for t in tools if not t.startswith("mcp__")}
        if self.req.tool_server:
            builtin_seen = {t for t in builtin_seen if not MCP_RESOURCE_TOOL.match(t)}
        extra = builtin_seen - expected
        if extra:
            self.abort = ProviderError(ErrorKind.ISOLATION, f"agent was given tools outside its allowlist: "
                                       f"{sorted(extra)}")
            return
        servers = {s.get("name"): s.get("status") for s in ev.get("mcp_servers") or [] if isinstance(s, dict)}
        foreign = set(servers) - ({MCP_SERVER} if self.req.tool_server else set())
        if foreign:
            self.abort = ProviderError(ErrorKind.ISOLATION, f"unexpected MCP servers attached: {sorted(foreign)}")
            return
        if self.req.tool_server:
            status = servers.get(MCP_SERVER)
            if MCP_SERVER not in servers or status in ("failed", "error", "disconnected"):
                self.abort = ProviderError(ErrorKind.TOOL_SETUP,
                                           f"literature tool server did not start (status: {status})")
            elif status != "connected":
                self.warnings.append(f"literature tool server status at start: {status}")

    def _on_assistant(self, ev: dict) -> None:
        msg = ev.get("message") or {}
        model = msg.get("model")
        if model and model != "<synthetic>" and model not in self.served:
            self.served.append(model)
        if ev.get("error"):
            self.error_codes.append(str(ev["error"]))
        texts = []
        for block in msg.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                texts.append(block.get("text", ""))
            elif btype == "tool_use":
                call = {"name": block.get("name"), "input": block.get("input")}
                self.tool_calls.append(call)
                self.log({"kind": "tool_use", "id": block.get("id"), **call})
            elif btype == "thinking":
                self.log({"kind": "thinking", "chars": len(block.get("thinking") or "")})
        if texts:
            joined = "\n".join(texts)
            if ev.get("is_api_error_message") or ev.get("error"):
                self.error_texts.append(joined)
            else:
                self.last_text = texts
            self.log({"kind": "assistant_text", "text": joined[:20000], "model": model,
                      "stop_reason": msg.get("stop_reason"), "api_error": bool(ev.get("is_api_error_message"))})

    def _on_user(self, ev: dict) -> None:
        msg = ev.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            return
        for block in content:
            if block.get("type") != "tool_result":
                continue
            body = block.get("content")
            if isinstance(body, list):
                body = "\n".join(b.get("text", "") for b in body if isinstance(b, dict))
            self.log({"kind": "tool_result", "tool_use_id": block.get("tool_use_id"),
                      "is_error": bool(block.get("is_error")), "content": str(body or "")[:TOOL_RESULT_KEEP]})

    def usage_or_none(self) -> dict | None:
        return (self.result or {}).get("usage")

    def limit_reset(self) -> str | None:
        """Reset time of the binding plan window, from the latest rate_limit_event."""
        info = self.rate_limit or {}
        stamp = info.get("resetsAt")
        if isinstance(stamp, (int, float)) and stamp > 0:
            return dt.datetime.fromtimestamp(stamp, dt.timezone.utc).isoformat(timespec="seconds")
        return None

    def limit_rejected(self) -> bool:
        return (self.rate_limit or {}).get("status") == "rejected"

    def _with_plan_window(self, kind: ErrorKind, reset_at: str | None) -> tuple[ErrorKind, str | None]:
        """A throttle while the plan window says "rejected" is a plan limit; take its reset time if none was given."""
        if kind is ErrorKind.RATE_LIMIT and self.limit_rejected():
            kind = ErrorKind.PLAN_LIMIT
        if kind is ErrorKind.PLAN_LIMIT and reset_at is None:
            reset_at = self.limit_reset()
        return kind, reset_at

    def finish(self, exit_code: int | None, stderr_text: str, sandbox: Path) -> AgentResult:
        res = self.result
        if "unrecognized_model" in stderr_text:
            self.warnings.append(f"this Claude Code version does not recognize '{self.req.model}'; its context "
                                 "window and cost figures for it may be wrong (try `claude update`)")
        if res is None:
            kind, reset_at, retry_after = classify_error(self.error_texts + [stderr_text], self.error_codes, None, None)
            if kind is ErrorKind.UNKNOWN:
                kind = ErrorKind.EMPTY_OUTPUT if exit_code == 0 else ErrorKind.UNKNOWN
            kind, reset_at = self._with_plan_window(kind, reset_at)
            raise ProviderError(kind, f"claude exited (code {exit_code}) without a result"
                                + (f": {stderr_text[-600:]}" if stderr_text else ""),
                                reset_at=reset_at, retry_after_s=retry_after, detail={"rate_limit": self.rate_limit})
        text = (res.get("result") or "").strip() if isinstance(res.get("result"), str) else ""
        errors = [e if isinstance(e, str) else json.dumps(e) for e in (res.get("errors") or [])]
        if res.get("is_error") or res.get("subtype", "success") != "success":
            if res.get("subtype") == "error_max_turns":
                raise ProviderError(ErrorKind.MAX_TURNS, f"agent hit --max-turns ({self.req.max_turns}) "
                                    "before writing its report", usage=res.get("usage"))
            kind, reset_at, retry_after = classify_error(
                [text, *errors] + self.error_texts + [stderr_text], self.error_codes, res.get("api_error_status"),
                res.get("stop_reason"))
            kind, reset_at = self._with_plan_window(kind, reset_at)
            raise ProviderError(kind, text or "; ".join(errors + self.error_texts) or f"claude reported an error "
                                f"({res.get('terminal_reason')})", reset_at=reset_at, retry_after_s=retry_after,
                                usage=res.get("usage"), detail={"terminal_reason": res.get("terminal_reason"),
                                                                "rate_limit": self.rate_limit})
        if res.get("stop_reason") == "refusal":
            raise ProviderError(ErrorKind.REFUSAL, text or "the model declined the request", usage=res.get("usage"))
        if not text:
            text = "\n".join(self.last_text).strip()
        if not text:
            raise ProviderError(ErrorKind.EMPTY_OUTPUT, "the agent produced no final text", usage=res.get("usage"))
        substituted = substituted_models(self.req.model, self.served)
        if substituted:
            self.warnings.append(f"MODEL SUBSTITUTION: requested {self.req.model} but turns were served by "
                                 f"{', '.join(substituted)}")
        if res.get("stop_reason") == "max_tokens":
            self.warnings.append("the final message stopped at the output-token limit; the report may be cut off")
        if res.get("permission_denials"):
            self.warnings.append(f"{len(res['permission_denials'])} tool call(s) were denied by the sandbox")
        init = self.init or {}
        return AgentResult(
            text=text,
            stop_reason=res.get("stop_reason"),
            usage=res.get("usage"),
            model_usage=res.get("modelUsage"),
            reported_cost_usd=res.get("total_cost_usd"),
            duration_ms=res.get("duration_ms"),
            duration_api_ms=res.get("duration_api_ms"),
            num_turns=res.get("num_turns"),
            served_models=self.served,
            tool_calls=self.tool_calls,
            warnings=self.warnings,
            transcript_path=self.transcript,
            sandbox_dir=sandbox,
            session_id=res.get("session_id"),
            provider="claude_code",
            runtime={"claude_code_version": init.get("claude_code_version"),
                     "apiKeySource": init.get("apiKeySource"), "tools": init.get("tools"),
                     "mcp_servers": init.get("mcp_servers"), "rate_limit": self.rate_limit},
        )


# ---------------------------------------------------------------- error classification

_PATTERNS: list[tuple[ErrorKind, re.Pattern]] = [
    (ErrorKind.AUTH, re.compile(r"authenticat|oauth|not logged in|please run /login|invalid api key|unauthorized|"
                                r"\b401\b|token (has )?expired", re.I)),
    (ErrorKind.PLAN_LIMIT, re.compile(r"usage limit|limit reached|(?:hit|reached) your [\w\s.'-]{0,40}?limit|"
                                      r"out of (?:\w+ )?usage|weekly limit|session limit|5-hour limit|limit resets|"
                                      r"usage credits|credit balance|billing", re.I)),
    (ErrorKind.CONTEXT_OVERFLOW, re.compile(r"prompt is too long|context (window|length)|too many tokens|"
                                            r"exceeds? the (maximum|context)|input is too long", re.I)),
    (ErrorKind.MODEL_UNAVAILABLE, re.compile(r"model[^.\n]{0,60}(not found|not available|does not exist|invalid|"
                                             r"not supported|unknown|not allowed)|not_found_error|\b404\b|"
                                             r"does not support this model|or newer is required", re.I)),
    (ErrorKind.REFUSAL, re.compile(r"usage policy|unable to respond to this request|declined to", re.I)),
    (ErrorKind.OVERLOADED, re.compile(r"overloaded|\b529\b", re.I)),
    (ErrorKind.RATE_LIMIT, re.compile(r"rate.?limit|too many requests|\b429\b", re.I)),
    (ErrorKind.SERVER, re.compile(r"internal server error|api error: 5\d\d|\b50[0234]\b|service unavailable|"
                                  r"bad gateway|api_error", re.I)),
    (ErrorKind.NETWORK, re.compile(r"connection|network|econnreset|etimedout|socket|fetch failed|getaddrinfo|"
                                   r"eai_again|timed out", re.I)),
    (ErrorKind.INVALID_REQUEST, re.compile(r"invalid_request|\b400\b|bad request", re.I)),
]

_CODE_MAP = {
    "authentication_failed": ErrorKind.AUTH,
    "billing_error": ErrorKind.PLAN_LIMIT,
    "rate_limit": ErrorKind.RATE_LIMIT,
    "server_error": ErrorKind.SERVER,
    "invalid_request": ErrorKind.INVALID_REQUEST,
    "overloaded": ErrorKind.OVERLOADED,
}


def classify_error(texts: list[str], codes: list[str], api_status: int | None,
                   stop_reason: str | None) -> tuple[ErrorKind, str | None, float | None]:
    """Map Claude Code error signals to an ErrorKind plus an optional reset time / retry delay."""
    blob = "\n".join(t for t in texts if t)
    reset_at = parse_reset_time(blob)
    if stop_reason == "refusal":
        return ErrorKind.REFUSAL, None, None
    by_kind = dict(_PATTERNS)
    # Specific wording beats generic status codes: a plan limit, a login problem or an oversized prompt.
    if by_kind[ErrorKind.PLAN_LIMIT].search(blob):
        return ErrorKind.PLAN_LIMIT, reset_at, None
    if by_kind[ErrorKind.AUTH].search(blob):
        return ErrorKind.AUTH, None, None
    if by_kind[ErrorKind.CONTEXT_OVERFLOW].search(blob):
        return ErrorKind.CONTEXT_OVERFLOW, None, None
    if re.search(r"does not support this model|or newer is required", blob, re.I):
        return ErrorKind.MODEL_UNAVAILABLE, None, None  # e.g. the CLI is too old for this model: `claude update`
    for code in codes:
        if code in _CODE_MAP:
            kind = _CODE_MAP[code]
            if kind is ErrorKind.RATE_LIMIT and reset_at:
                return ErrorKind.PLAN_LIMIT, reset_at, None
            return kind, reset_at, _retry_after(blob)
    if api_status:
        if api_status == 429:
            return (ErrorKind.PLAN_LIMIT if reset_at else ErrorKind.RATE_LIMIT), reset_at, _retry_after(blob)
        if api_status == 529:
            return ErrorKind.OVERLOADED, None, _retry_after(blob)
        if api_status >= 500:
            return ErrorKind.SERVER, None, _retry_after(blob)
        if api_status in (401, 403):
            return ErrorKind.AUTH, None, None
        if api_status == 404:
            return ErrorKind.MODEL_UNAVAILABLE, None, None
        if api_status == 413:
            return ErrorKind.CONTEXT_OVERFLOW, None, None
        if api_status == 400:
            return ErrorKind.INVALID_REQUEST, None, None
    for kind, pattern in _PATTERNS:
        if kind in (ErrorKind.PLAN_LIMIT, ErrorKind.AUTH, ErrorKind.CONTEXT_OVERFLOW):
            continue
        if pattern.search(blob):
            return kind, reset_at, _retry_after(blob)
    return ErrorKind.UNKNOWN, reset_at, None


def _retry_after(text: str) -> float | None:
    m = re.search(r"(?:retry|try again)\s+(?:after|in)\s+(\d+(?:\.\d+)?)\s*(s|sec|seconds|m|min|minutes|h|hours)?",
                  text, re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "s").lower()
    return value * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)


def parse_reset_time(text: str, now: dt.datetime | None = None) -> str | None:
    """Best-effort parse of a plan-limit reset time into a UTC ISO timestamp."""
    now = now or dt.datetime.now(dt.timezone.utc)
    m = re.search(r"\|(\d{10})\b", text) or re.search(r"reset[s]?[^0-9\n]{0,20}(\d{10})\b", text, re.I)
    if m:
        return dt.datetime.fromtimestamp(int(m.group(1)), dt.timezone.utc).isoformat(timespec="seconds")
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)", text)
    if m and re.search(r"reset|until|try again", text, re.I):
        try:
            stamp = dt.datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=dt.timezone.utc)
            return stamp.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
        except ValueError:
            pass
    m = re.search(r"try again in\s+(\d+)\s*(minutes?|mins?|hours?|hrs?|seconds?|secs?)", text, re.I)
    if m:
        value = int(m.group(1))
        unit = m.group(2).lower()
        delta = dt.timedelta(hours=value) if unit.startswith("h") else (
            dt.timedelta(minutes=value) if unit.startswith("m") else dt.timedelta(seconds=value))
        return (now + delta).isoformat(timespec="seconds")
    m = re.search(r"resets?\s+(?:on\s+)?([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(?:at\s+)?"
                  r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([A-Za-z_/+\-0-9]+)\))?", text, re.I)
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    if m and m.group(1).lower() in months:
        hour = int(m.group(3)) % 12 if m.group(5) else int(m.group(3))
        if m.group(5) and m.group(5).lower() == "pm":
            hour += 12
        tz = _zone(m.group(6))
        local_now = now.astimezone(tz) if tz else now.astimezone()
        try:
            target = local_now.replace(month=months.index(m.group(1).lower()) + 1, day=int(m.group(2)), hour=hour,
                                       minute=int(m.group(4) or 0), second=0, microsecond=0)
        except ValueError:
            return None
        if target < local_now - dt.timedelta(days=1):
            target = target.replace(year=target.year + 1)
        return target.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([A-Za-z_/+\-0-9]+)\))?", text, re.I)
    if m:
        hour = int(m.group(1)) % 12 if m.group(3) else int(m.group(1))
        if m.group(3) and m.group(3).lower() == "pm":
            hour += 12
        minute = int(m.group(2) or 0)
        tz = _zone(m.group(4))
        local_now = now.astimezone(tz) if tz else now.astimezone()
        try:
            target = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            return None
        if target <= local_now:
            target += dt.timedelta(days=1)
        return target.astimezone(dt.timezone.utc).isoformat(timespec="seconds")
    return None


def _zone(name: str | None):
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:
        return None
