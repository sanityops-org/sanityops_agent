# tests/unit/test_config_gemini.py

from __future__ import annotations

from sanityops_agent.config import ProviderConfig


def test_gemini_fields_default_to_disabled():
    """Without explicit values, Gemini extras stay unset and thinking is off."""
    config = ProviderConfig(PROVIDER="gemini", API_KEY="test-key")
    assert config.GEMINI_PROJECT_ID is None
    assert config.GEMINI_LOCATION is None
    assert config.GEMINI_THINKING_BUDGET is None
    assert config.GEMINI_STORE_THINKING is False


def test_gemini_fields_accept_explicit_values():
    config = ProviderConfig(
        PROVIDER="gemini",
        API_KEY="test-key",
        GEMINI_PROJECT_ID="my-project",
        GEMINI_LOCATION="us-central1",
        GEMINI_THINKING_BUDGET=2048,
        GEMINI_STORE_THINKING=True,
    )
    assert config.GEMINI_PROJECT_ID == "my-project"
    assert config.GEMINI_LOCATION == "us-central1"
    assert config.GEMINI_THINKING_BUDGET == 2048
    assert config.GEMINI_STORE_THINKING is True


def test_gemini_fields_read_from_env(monkeypatch):
    """Env vars use the LLM_ prefix from ProviderConfig.model_config."""
    monkeypatch.setenv("LLM_GEMINI_PROJECT_ID", "env-project")
    monkeypatch.setenv("LLM_GEMINI_THINKING_BUDGET", "1024")
    monkeypatch.setenv("LLM_GEMINI_STORE_THINKING", "true")
    config = ProviderConfig(PROVIDER="gemini", API_KEY="test-key")
    assert config.GEMINI_PROJECT_ID == "env-project"
    assert config.GEMINI_THINKING_BUDGET == 1024
    assert config.GEMINI_STORE_THINKING is True
