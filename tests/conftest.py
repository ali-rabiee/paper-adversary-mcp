from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


# Fake credentials for the secret-scan tests. They are assembled at runtime so that no scanner (GitHub push
# protection, trufflehog, ...) mistakes this source for a leak; none of them is or was ever a real credential.
_FAKE_BODY = "Zq3vT9kLmN2pR7sW4xY8bC1dF6gH0jK5aE"
FAKE_SECRETS = {
    "anthropic": "sk-" + "ant-oat01-" + _FAKE_BODY,
    "anthropic_api": "sk-" + "ant-api03-" + _FAKE_BODY[:26],
    "aws": "AK" + "IAZ7Q4XHJ3MNBVKWTE",
    "github": "gh" + "p_aB3dE6gH9jK2mN5pQ8sT1vW4yZ7cF0iL3oR6",
    "s2": "aB3dE6gH9jK2mN5pQ8sT1vW4yZ7",
    "pem_header": "-----BEGIN " + "PRIVATE KEY-----",
    "rsa_header": "-----BEGIN RSA " + "PRIVATE KEY-----",
    "pem_body": "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj",
}


def mock_override(**mock_options) -> dict:
    """Config override for fast offline runs with the mock provider."""
    return {
        "provider": {"type": "mock", "preflight": "off", "mock": {"delay_seconds": 0.01, **mock_options}},
        "concurrency": {"max_parallel_agents": 4, "stagger_seconds": 0},
        "retry": {"max_attempts": 3, "base_delay_seconds": 0, "max_delay_seconds": 0, "jitter": 0},
        "plan_limit": {"policy": "wait", "max_wait_hours": 0.05, "default_wait_minutes": 0.001},
        "search": {"refcheck": False, "fulltext": {"offline": True}},
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
