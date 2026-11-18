"""Unit tests for AgentConfig and ProviderConfig settings."""

from __future__ import annotations

from sanityops_agent.config import AgentConfig, ProviderConfig


def test_agent_config_max_tokens_defaults_to_none(monkeypatch, tmp_path):
    """AgentConfig.MAX_TOKENS defaults to None and parses AGENT_MAX_TOKENS."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_MAX_TOKENS", raising=False)
    assert AgentConfig().MAX_TOKENS is None

    monkeypatch.setenv("AGENT_MAX_TOKENS", "16384")
    assert AgentConfig().MAX_TOKENS == 16384


def test_provider_config_extra_body_defaults_to_none(monkeypatch, tmp_path):
    """ProviderConfig.EXTRA_BODY defaults to None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    assert ProviderConfig().EXTRA_BODY is None


def test_provider_config_extra_body_from_env(monkeypatch, tmp_path):
    """ProviderConfig.EXTRA_BODY parses a JSON object from LLM_EXTRA_BODY."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    config = ProviderConfig()
    assert config.EXTRA_BODY == {"thinking": {"type": "disabled"}}
