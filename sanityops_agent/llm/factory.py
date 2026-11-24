from __future__ import annotations

import inspect
from collections.abc import Collection
from typing import Any
from urllib.parse import urlsplit

from sanityops_agent.config import ProviderConfig
from sanityops_agent.llm.base import LLMProvider
from sanityops_agent.llm.providers.anthropic import AnthropicProvider
from sanityops_agent.llm.providers.gemini import GeminiProvider
from sanityops_agent.llm.providers.openai import OpenAIProvider


class ProviderFactory:
    """Factory for creating LLMProvider instances"""

    _registry: dict[str, type[LLMProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMProvider]):
        """Register a provider implementation"""
        cls._registry[name.lower()] = provider_cls

    @classmethod
    def create(
        cls, config: ProviderConfig, *, allowed_base_urls: Collection[str] | None = None
    ) -> LLMProvider:
        """Create provider from config"""
        provider_name = config.PROVIDER.lower()

        if provider_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown provider '{config.PROVIDER}'. Available: {available}"
            )

        provider_cls = cls._registry[provider_name]
        sig = inspect.signature(provider_cls.__init__)
        params = sig.parameters
        kwargs: dict[str, Any] = {
            "api_key": config.API_KEY,
            "base_url": config.BASE_URL,
            "model": config.MODEL_ID,
        }
        if allowed_base_urls is not None:
            endpoint = str(
                config.BASE_URL or getattr(provider_cls, "DEFAULT_BASE_URL", "") or ""
            )
            parts = urlsplit(endpoint)
            if (
                parts.scheme != "https"
                or not parts.hostname
                or parts.username
                or parts.password
                or parts.query
                or parts.fragment
                or endpoint.rstrip("/")
                not in {url.rstrip("/") for url in allowed_base_urls}
            ):
                raise ValueError(
                    "Provider endpoint is not an approved HTTPS destination"
                )
            if "follow_redirects" not in params:
                raise ValueError("Managed providers must support disabling redirects")
            kwargs["base_url"] = endpoint
            kwargs["follow_redirects"] = False
        if provider_name == "gemini":
            kwargs.update(
                {
                    "project_id": config.GEMINI_PROJECT_ID,
                    "location": config.GEMINI_LOCATION,
                    "thinking_budget": config.GEMINI_THINKING_BUDGET,
                    "store_thinking": config.GEMINI_STORE_THINKING,
                }
            )
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if "extra_body" in params or has_var_keyword:
            kwargs["extra_body"] = config.EXTRA_BODY

        return provider_cls(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._registry.keys())


# Auto-register built-in providers
ProviderFactory.register("openai", OpenAIProvider)
ProviderFactory.register("anthropic", AnthropicProvider)
ProviderFactory.register("gemini", GeminiProvider)
