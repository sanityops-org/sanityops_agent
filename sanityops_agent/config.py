"""Configuration for the Agent OOP refactor project."""

from __future__ import annotations

import warnings
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    PROVIDER: str = "anthropic"
    API_KEY: str = ""
    BASE_URL: str | None = None
    MODEL_ID: str = "claude-sonnet-4-20250514"
    EXTRA_BODY: dict | None = None

    # Gemini-specific fields
    GEMINI_PROJECT_ID: str | None = None      # Vertex AI project ID
    GEMINI_LOCATION: str | None = None        # Vertex AI region
    GEMINI_THINKING_BUDGET: int | None = None # thinking token budget (None=disabled)
    GEMINI_STORE_THINKING: bool = False       # whether to store thinking in history

    # Backward compatibility: LLM_PROVIDER is deprecated, use PROVIDER
    LLM_PROVIDER: str | None = None

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def _warn_deprecated_llm_provider(cls, v: str | None) -> str | None:
        if v is not None:
            warnings.warn(
                "LLM_PROVIDER is deprecated, use PROVIDER instead",
                DeprecationWarning,
                stacklevel=2,
            )
        return v

    def model_post_init(self, __context) -> None:
        # If LLM_PROVIDER was set, use it as PROVIDER (backward compatibility)
        # Note: LLM_PROVIDER takes precedence over PROVIDER if both are set,
        # matching the behavior of the original config before the rename.
        if self.LLM_PROVIDER is not None:
            object.__setattr__(self, "PROVIDER", self.LLM_PROVIDER)


class AgentConfig(BaseSettings):
    """Agent runtime configuration."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    MAX_LOOPS: int = 30
    MAX_TOKENS: int | None = None
    TOTAL_TIMEOUT: int = 300
    TOKEN_BUDGET: int = 100000
    MAX_RETRIES: int = 3


class ToolConfig(BaseSettings):
    """Tool subsystem configuration."""

    model_config = SettingsConfigDict(env_prefix="TOOL_", env_file=".env", extra="ignore")

    WORKDIR: str = "."
    DEBUG_MODE: bool = False
    USE_LEGACY: bool = False


class AppConfig(BaseModel):
    """Combined application configuration."""

    provider: ProviderConfig
    agent: AgentConfig
    tool: ToolConfig

    @classmethod
    def load(cls, env_file: Path | str | None = None) -> AppConfig:
        """Load configuration from environment / .env file.

        Args:
            env_file: Optional path to a .env file. If None, the default
                ``.env`` in the current working directory is used.

        Returns:
            A fully populated ``AppConfig`` instance.
        """
        provider = ProviderConfig(_env_file=env_file) if env_file else ProviderConfig()
        agent = AgentConfig(_env_file=env_file) if env_file else AgentConfig()
        tool = ToolConfig(_env_file=env_file) if env_file else ToolConfig()
        return cls(provider=provider, agent=agent, tool=tool)
