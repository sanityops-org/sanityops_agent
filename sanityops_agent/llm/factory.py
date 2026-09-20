from __future__ import annotations

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
    def create(cls, config: ProviderConfig) -> LLMProvider:
        """Create provider from config"""
        provider_name = config.PROVIDER.lower()

        if provider_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown provider '{config.PROVIDER}'. Available: {available}")

        provider_cls = cls._registry[provider_name]

        if provider_name == "gemini":
            return provider_cls(
                api_key=config.API_KEY,
                base_url=config.BASE_URL,
                model=config.MODEL_ID,
                project_id=config.GEMINI_PROJECT_ID,
                location=config.GEMINI_LOCATION,
                thinking_budget=config.GEMINI_THINKING_BUDGET,
                store_thinking=config.GEMINI_STORE_THINKING,
            )
        else:
            return provider_cls(
                api_key=config.API_KEY,
                base_url=config.BASE_URL,
                model=config.MODEL_ID,
            )

    @classmethod
    def list_providers(cls) -> list[str]:
        return list(cls._registry.keys())


# Auto-register built-in providers
ProviderFactory.register("openai", OpenAIProvider)
ProviderFactory.register("anthropic", AnthropicProvider)
ProviderFactory.register("gemini", GeminiProvider)
