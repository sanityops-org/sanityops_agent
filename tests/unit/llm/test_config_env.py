"""Tests that ProviderConfig reads the documented LLM_PROVIDER env var."""

import pytest

from sanityops_agent.config import AgentConfig, AppConfig, ProviderConfig
from sanityops_agent.llm.factory import ProviderFactory


def test_agent_config_max_tokens_defaults_to_none(monkeypatch, tmp_path):
    """AgentConfig.MAX_TOKENS defaults to None and parses AGENT_MAX_TOKENS."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_MAX_TOKENS", raising=False)
    assert AgentConfig().MAX_TOKENS is None

    monkeypatch.setenv("AGENT_MAX_TOKENS", "16384")
    assert AgentConfig().MAX_TOKENS == 16384


def test_provider_reads_llm_provider_env(monkeypatch, tmp_path):
    """LLM_PROVIDER (not LLM_LLM_PROVIDER) must populate the PROVIDER field."""
    monkeypatch.chdir(tmp_path)  # isolate from any repo-root .env
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_LLM_PROVIDER", raising=False)
    config = ProviderConfig()
    assert config.PROVIDER == "openai"


def test_provider_default_is_anthropic(monkeypatch, tmp_path):
    """Without LLM_PROVIDER set, the default is anthropic."""
    monkeypatch.chdir(tmp_path)  # isolate from any repo-root .env
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    config = ProviderConfig()
    assert config.PROVIDER == "anthropic"


def test_managed_config_ignores_dotenv_in_current_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    (tmp_path / ".env").write_text("LLM_API_KEY=synthetic-local-value\n")
    assert AppConfig.load(managed=True).provider.API_KEY == ""


def test_managed_provider_requires_approved_destination():
    config = ProviderConfig(PROVIDER="openai", BASE_URL="https://elsewhere.example/v1")
    with pytest.raises(ValueError, match="approved"):
        ProviderFactory.create(
            config, allowed_base_urls={"https://approved.example/v1"}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openai", "anthropic"])
async def test_managed_provider_uses_validating_transport_without_redirects(
    name, monkeypatch
):
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    base_url = "https://approved.example/v1"
    config = ProviderConfig(PROVIDER=name, API_KEY="fixture", BASE_URL=base_url)
    provider = ProviderFactory.create(config, allowed_base_urls={base_url})
    try:
        client = provider._get_client()
        assert client._client.follow_redirects is False
        assert client._client.trust_env is False
        assert not client._client._mounts
    finally:
        await provider.close()


def test_provider_config_does_not_emit_credential():
    config = ProviderConfig(API_KEY="fixture-credential")
    assert "fixture-credential" not in repr(config)
    assert "API_KEY" not in config.model_dump()
