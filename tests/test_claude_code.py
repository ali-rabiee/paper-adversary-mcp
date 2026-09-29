"""The Claude Code provider against a fake `claude` binary (no plan usage)."""

import asyncio
import datetime as dt
import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from paper_adversary.config import ProviderConfig
from paper_adversary.providers.base import AgentRequest, ErrorKind, ProviderError
from paper_adversary.providers.claude_code import ClaudeCodeProvider, classify_error, parse_reset_time

FAKE = textwrap.dedent('''\
    import json, os, sys, time
    home = os.environ["HOME"]
    scenario = json.load(open(os.path.join(home, "scenario.json")))
    argv = sys.argv[1:]
    def arg(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default
    stdin = sys.stdin.read()
    json.dump({{"argv": argv, "env": sorted(os.environ), "cwd": os.getcwd(), "stdin": stdin,
               "cwd_files": sorted(os.listdir("."))}}, open(os.path.join(home, "record.json"), "w"))
    tools = [t for t in (arg("--tools") or "").split(",") if t]
    mode = scenario["mode"]
    def emit(obj):
        print(json.dumps(obj), flush=True)
    servers = [{{"name": "lit", "status": scenario.get("mcp_status", "connected")}}] if "--mcp-config" in argv else []
    emit({{"type": "system", "subtype": "init", "model": arg("--model"), "tools": tools + scenario.get("extra_tools", []),
          "mcp_servers": servers, "apiKeySource": scenario.get("api_key_source", "none"),
          "claude_code_version": "9.9.9"}})
    if mode == "hang":
        time.sleep(60)
    if mode == "request_after_init":
        time.sleep(1.0)
        open(os.path.join(home, "request_sent"), "w").write("1")
        time.sleep(30)
    rl = {{"status": "allowed", "resetsAt": 1790704200, "rateLimitType": "five_hour", "overageStatus": "rejected",
          "overageDisabledReason": "org_level_disabled", "isUsingOverage": False,
          "unifiedWindows": {{"five_hour": {{"utilization": 0.22, "resetsAt": 1790704200}},
                             "seven_day": {{"utilization": 0.62, "resetsAt": 1790910000}}}}}}
    if mode == "limit_event":
        emit({{"type": "rate_limit_event", "rate_limit_info": {{**rl, "status": "rejected", "resetsAt": 4102444800}}}})
        emit({{"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
              "result": "You've reached your Fable limit. Switch to another model, or manage usage credits."}})
        sys.exit(1)
    emit({{"type": "rate_limit_event", "rate_limit_info": rl}})
    if mode == "errors_field":
        emit({{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "",
              "errors": ["Invalid API key · Please run /login"]}})
        sys.exit(1)
    if mode == "auth":
        emit({{"type": "assistant", "error": "authentication_failed", "is_api_error_message": True,
              "message": {{"model": "<synthetic>", "content": [{{"type": "text",
              "text": "Failed to authenticate: OAuth session expired and could not be refreshed"}}]}}}})
        emit({{"type": "result", "subtype": "success", "is_error": True, "terminal_reason": "api_error",
              "result": "Failed to authenticate: OAuth session expired and could not be refreshed",
              "usage": {{"input_tokens": 0, "output_tokens": 0}}}})
        sys.exit(1)
    if mode == "plan_limit":
        emit({{"type": "result", "subtype": "success", "is_error": True, "terminal_reason": "api_error",
              "api_error_status": 429, "result": "Claude AI usage limit reached|4102444800"}})
        sys.exit(1)
    if mode == "probe":
        system = open(arg("--system-prompt-file")).read()
        word = "PINEAPPLE" if scenario.get("honour_system", True) and "PINEAPPLE" in system else "OK"
        emit({{"type": "assistant", "message": {{"model": arg("--model"), "content": [{{"type": "text", "text": word}}]}}}})
        emit({{"type": "result", "subtype": "success", "is_error": False, "result": word, "num_turns": 1,
              "usage": {{"input_tokens": 5, "output_tokens": 1}},
              "modelUsage": {{arg("--model"): {{"inputTokens": 5, "outputTokens": 1, "contextWindow": 200000}}}}}})
        sys.exit(0)
    served = scenario.get("served_model", arg("--model"))
    emit({{"type": "stream_event", "event": {{"type": "content_block_delta"}}}})
    emit({{"type": "assistant", "message": {{"model": served, "content": [
        {{"type": "thinking", "thinking": ""}},
        {{"type": "tool_use", "id": "t1", "name": "mcp__lit__search_literature", "input": {{"query": "x"}}}}]}}}})
    emit({{"type": "user", "message": {{"content": [{{"type": "tool_result", "tool_use_id": "t1",
          "content": [{{"type": "text", "text": "[1] Some paper (2020)"}}]}}]}}}})
    emit({{"type": "assistant", "message": {{"model": served, "content": [{{"type": "text", "text": "## Verdict\\nok"}}]}}}})
    emit({{"type": "result", "subtype": "success", "is_error": False, "result": "## Verdict\\nok",
          "stop_reason": "end_turn", "num_turns": 2, "duration_ms": 1200, "duration_api_ms": 900,
          "total_cost_usd": 0.0123, "session_id": "s1",
          "usage": {{"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 10,
                    "cache_creation_input_tokens": 20, "server_tool_use": {{"web_search_requests": 1}}}},
          "modelUsage": {{served: {{"inputTokens": 100, "outputTokens": 50, "contextWindow": 1000000}}}}}})
''')


@pytest.fixture
def fake(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE.format(python=sys.executable))
    binary = tmp_path / "claude"  # sh wrapper: shebang lines cannot contain the space in "ICML 2027"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-reach-agents")
    monkeypatch.setenv("CLAUDECODE", "1")

    def run(mode="success", tools=(), tool_server=None, pdf=None, idle=30, **scenario):
        (home / "scenario.json").write_text(json.dumps({"mode": mode, **scenario}))
        provider = ClaudeCodeProvider(ProviderConfig(claude_binary=str(binary)), sandbox_root=tmp_path / "sandboxes")
        req = AgentRequest(run_id="run_x", agent_id="N1", role="novelty", model="claude-opus-5-5", effort="high",
                           system_prompt="SYSTEM", user_text="USER PROMPT", log_dir=tmp_path / "logs" / mode,
                           tools=list(tools), tool_server=tool_server, pdf_path=pdf, timeout_s=60,
                           idle_timeout_s=idle, max_output_tokens=128000)
        result = asyncio.run(provider.run(req))
        record = json.loads((home / "record.json").read_text())
        return provider, result, record

    return run, home


def test_success_and_sandboxing(fake, tmp_path):
    run, home = fake
    provider, result, record = run()
    argv = record["argv"]
    for flag in ("-p", "--restricted", "--strict-mcp-config", "--no-session-persistence", "--safe-mode",
                 "--disable-slash-commands", "--include-partial-messages"):
        assert flag in argv, flag
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--effort") + 1] == "high"
    assert Path(argv[argv.index("--system-prompt-file") + 1]).read_text() == "SYSTEM"
    assert record["stdin"] == "USER PROMPT"
    assert "ANTHROPIC_API_KEY" not in record["env"] and "CLAUDECODE" not in record["env"]
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" in record["env"]
    assert record["cwd_files"] == []  # empty sandbox
    assert Path(record["cwd"]).parent == tmp_path / "sandboxes"
    assert result.text == "## Verdict\nok"
    assert result.usage["output_tokens"] == 50 and result.reported_cost_usd == 0.0123
    assert result.served_models == ["claude-opus-5-5"] and not result.warnings
    lines = [json.loads(x) for x in result.transcript_path.read_text().splitlines()]
    assert any(x["kind"] == "tool_result" and "Some paper" in x["content"] for x in lines)
    provider.cleanup(result)
    assert not Path(record["cwd"]).exists()


def test_pdf_and_tool_server_flags(fake, tmp_path):
    run, _ = fake
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    server = {"command": sys.executable, "args": ["-m", "paper_adversary", "tools-server"], "env": {}}
    _, _, record = run(tools=["web_search", "read_pdf", "literature"], tool_server=server, pdf=pdf)
    argv = record["argv"]
    assert argv[argv.index("--tools") + 1] == "WebSearch,Read"
    assert "mcp__lit" in argv[argv.index("--allowedTools") + 1]
    assert "--safe-mode" not in argv and "--mcp-config" in argv
    assert record["cwd_files"] == ["paper.pdf"]


@pytest.mark.parametrize("mode, scenario, kind", [
    ("auth", {}, ErrorKind.AUTH),
    ("plan_limit", {}, ErrorKind.PLAN_LIMIT),
    ("success", {"api_key_source": "ANTHROPIC_API_KEY"}, ErrorKind.BILLING_GUARD),
    ("success", {"extra_tools": ["Bash"]}, ErrorKind.ISOLATION),
    ("success", {"mcp_status": "failed", "tool_server": True}, ErrorKind.TOOL_SETUP),
])
def test_failures_are_classified(fake, mode, scenario, kind):
    run, _ = fake
    server = {"command": sys.executable, "args": ["x"], "env": {}} if scenario.pop("tool_server", False) else None
    with pytest.raises(ProviderError) as info:
        run(mode, tool_server=server, tools=["literature"] if server else (), **scenario)
    assert info.value.kind is kind
    if kind is ErrorKind.PLAN_LIMIT:
        assert info.value.reset_at.startswith("2100-01-01")


def test_idle_timeout(fake):
    run, _ = fake
    with pytest.raises(ProviderError) as info:
        run("hang", idle=1)
    assert info.value.kind is ErrorKind.IDLE_TIMEOUT


def test_model_substitution_is_flagged(fake):
    run, _ = fake
    _, result, _ = run(served_model="claude-opus-4-8")
    assert any("MODEL SUBSTITUTION" in w for w in result.warnings)


@pytest.mark.parametrize("text, codes, status, expected", [
    ("Failed to authenticate: OAuth session expired", ["authentication_failed"], None, ErrorKind.AUTH),
    ("Claude AI usage limit reached|1759185600", [], 429, ErrorKind.PLAN_LIMIT),
    ("You've hit your limit · resets 3pm (America/New_York)", [], None, ErrorKind.PLAN_LIMIT),
    ("API Error: 529 {\"type\":\"overloaded_error\"}", [], None, ErrorKind.OVERLOADED),
    ("API Error: 500 Internal server error", [], None, ErrorKind.SERVER),
    ("Prompt is too long: 1200000 tokens > 1000000 maximum", [], None, ErrorKind.CONTEXT_OVERFLOW),
    ("rate limited", [], 429, ErrorKind.RATE_LIMIT),
    ("There's an issue with the selected model (claude-foo). It may not exist or you may not have access to it.",
     [], 404, ErrorKind.MODEL_UNAVAILABLE),
    ("Request timed out", [], None, ErrorKind.NETWORK),
    ("something odd", [], None, ErrorKind.UNKNOWN),
])
def test_classify_error(text, codes, status, expected):
    kind, _, _ = classify_error([text], codes, status, None)
    assert kind is expected


def test_parse_reset_time():
    now = dt.datetime(2026, 9, 29, 12, 0, tzinfo=dt.timezone.utc)
    assert parse_reset_time("limit|1759185600", now) == "2025-09-29T22:40:00+00:00"
    assert parse_reset_time("try again in 30 minutes", now) == "2026-09-29T12:30:00+00:00"
    got = parse_reset_time("resets 3pm (UTC)", now)
    assert got == "2026-09-29T15:00:00+00:00"
    assert parse_reset_time("resets 9am (UTC)", now) == "2026-09-30T09:00:00+00:00"
    assert parse_reset_time("no time here", now) is None


def test_env_never_contains_api_key_unless_allowed(monkeypatch, tmp_path):
    binary = tmp_path / "claude"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    env = ClaudeCodeProvider(ProviderConfig(claude_binary=str(binary))).child_env()
    assert "ANTHROPIC_API_KEY" not in env and env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok"
    env2 = ClaudeCodeProvider(ProviderConfig(claude_binary=str(binary), allow_api_key=True)).child_env()
    assert env2["ANTHROPIC_API_KEY"] == "sk-x"
    assert os.path.dirname(str(binary)) in env["PATH"].split(os.pathsep)[0]


def test_mcp_resource_tools_are_accepted_with_a_tool_server(fake):
    """Claude Code attaches List/Read/ReadDir MCP resource tools whenever an MCP server is configured."""
    run, _ = fake
    server = {"command": sys.executable, "args": ["x"], "env": {}}
    family = ["ListMcpResourcesTool", "ReadMcpResourceTool", "ReadMcpResourceDirTool"]
    _, result, _ = run(tools=["literature", "web_search"], tool_server=server, extra_tools=family)
    assert result.text.startswith("## Verdict")
    with pytest.raises(ProviderError) as info:  # without our server they are unexpected
        run(extra_tools=["ReadMcpResourceDirTool"])
    assert info.value.kind is ErrorKind.ISOLATION


def test_billing_guard_kills_before_any_request(fake):
    run, home = fake
    with pytest.raises(ProviderError) as info:
        run("request_after_init", api_key_source="/login managed key")
    assert info.value.kind is ErrorKind.BILLING_GUARD
    assert not (home / "request_sent").exists(), "the guarded session lived long enough to send a request"


def test_errors_field_is_classified(fake):
    run, _ = fake
    with pytest.raises(ProviderError) as info:
        run("errors_field")
    assert info.value.kind is ErrorKind.AUTH


def test_plan_limit_wordings():
    kind, reset, _ = classify_error(["You've hit your Opus limit"], ["rate_limit"], None, None)
    assert kind is ErrorKind.PLAN_LIMIT
    now = dt.datetime(2026, 9, 29, 12, 0, tzinfo=dt.timezone.utc)
    assert parse_reset_time("You've hit your weekly limit · resets Oct 3, 2pm (UTC)", now) == "2026-10-03T14:00:00+00:00"
    assert parse_reset_time("resets Jan 2 at 9am (UTC)", now) == "2027-01-02T09:00:00+00:00"


def test_probe_checks_that_the_system_prompt_is_applied(fake, tmp_path):
    _, home = fake
    binary = tmp_path / "claude"

    def probe(**scenario):
        (home / "scenario.json").write_text(json.dumps({"mode": "probe", **scenario}))
        provider = ClaudeCodeProvider(ProviderConfig(claude_binary=str(binary)), sandbox_root=tmp_path / "sb")
        return asyncio.run(provider.probe("claude-fable-5-1", tmp_path / "probe"))

    good = probe()
    assert good["ok"] and good["system_prompt_applied"] and good["context_window"] == 200000
    bad = probe(honour_system=False)
    assert not bad["ok"] and bad["error_kind"] == "tool_setup" and "system prompt" in bad["error"]


def test_plan_usage_is_captured_and_used_for_resets(fake):
    run, _ = fake
    _, result, _ = run()
    info = result.runtime["rate_limit"]
    assert info["unifiedWindows"]["seven_day"]["utilization"] == 0.62 and info["overageStatus"] == "rejected"
    with pytest.raises(ProviderError) as err:
        run("limit_event")
    assert err.value.kind is ErrorKind.PLAN_LIMIT
    assert err.value.reset_at == "2100-01-01T00:00:00+00:00"  # the message had no time; the event did


def test_plan_usage_line():
    from paper_adversary.service import plan_usage_line

    line = plan_usage_line({"status": "allowed", "overageStatus": "rejected", "isUsingOverage": False,
                            "unifiedWindows": {"five_hour": {"utilization": 0.22, "resetsAt": 1790704200},
                                               "seven_day": {"utilization": 0.62, "resetsAt": 1790910000}},
                            "observed_at": "2026-09-29T15:15:00+00:00"})
    assert "five-hour 22% used" in line and "seven-day 62% used" in line and "paid overage off" in line
