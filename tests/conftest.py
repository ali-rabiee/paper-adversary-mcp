from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def mock_override(**mock_options) -> dict:
    """Config override for fast offline runs with the mock provider."""
    return {
        "provider": {"type": "mock", "preflight": "off", "mock": {"delay_seconds": 0.01, **mock_options}},
        "concurrency": {"max_parallel_agents": 4, "stagger_seconds": 0},
        "retry": {"max_attempts": 3, "base_delay_seconds": 0, "max_delay_seconds": 0, "jitter": 0},
        "plan_limit": {"policy": "wait", "max_wait_hours": 0.05, "default_wait_minutes": 0.001},
        "search": {"refcheck": False},
    }


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("PAPER_ADVERSARY_RUNS_DIR", str(root))
    for key in ("ANTHROPIC_API_KEY", "PAPER_ADVERSARY_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    return root


@pytest.fixture
def sample_paper() -> Path:
    return FIXTURES / "sample_paper.md"
