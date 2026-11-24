"""Configuration for the Agent OOP refactor project."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_", env_file=".env", extra="ignore"
    )

    PROVIDER: str = "anthropic"
    API_KEY: str = Field(default="", repr=False, exclude=True)
    BASE_URL: str | None = None
    MODEL_ID: str = "claude-sonnet-4-20250514"
    EXTRA_BODY: dict | None = None

    # Gemini-specific fields
    GEMINI_PROJECT_ID: str | None = None  # Vertex AI project ID
    GEMINI_LOCATION: str | None = None  # Vertex AI region
    GEMINI_THINKING_BUDGET: int | None = None  # thinking budget (None = disabled)
    GEMINI_STORE_THINKING: bool = False  # keep thinking blocks in history


class AgentConfig(BaseSettings):
    """Agent runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_", env_file=".env", extra="ignore"
    )

    MAX_LOOPS: int = 30
    MAX_TOKENS: int | None = None
    TOTAL_TIMEOUT: int = 300
    TOKEN_BUDGET: int = 100000
    MAX_RETRIES: int = 3

    def to_runtime_config(self):
        from sanityops_agent.core.agent import AgentConfig as RuntimeAgentConfig

        return RuntimeAgentConfig(
            max_loops=self.MAX_LOOPS,
            max_tokens=self.MAX_TOKENS,
            total_timeout=self.TOTAL_TIMEOUT,
            token_budget=self.TOKEN_BUDGET,
            max_retries=self.MAX_RETRIES,
        )


class ToolConfig(BaseSettings):
    """Tool subsystem configuration."""

    model_config = SettingsConfigDict(
        env_prefix="TOOL_", env_file=".env", extra="ignore"
    )

    WORKDIR: str = "."
    DEBUG_MODE: bool = False
    USE_LEGACY: bool = False


class AppConfig(BaseModel):
    """Combined application configuration."""

    provider: ProviderConfig
    agent: AgentConfig
    tool: ToolConfig

    @classmethod
    def load(
        cls, env_file: Path | str | None = None, *, managed: bool = False
    ) -> AppConfig:
        """Load configuration from environment / .env file.

        Args:
            env_file: Optional path to a .env file. If None, the default
                ``.env`` in the current working directory is used.
            managed: Read the process environment and an explicitly configured
                absolute env_file, if supplied.

        Returns:
            A fully populated ``AppConfig`` instance.
        """
        if managed and env_file is not None and not Path(env_file).is_absolute():
            raise ValueError("Managed configuration requires an absolute env_file path")
        if managed or env_file is not None:
            return cls(
                provider=ProviderConfig(_env_file=env_file),  # type: ignore[call-arg]
                agent=AgentConfig(_env_file=env_file),  # type: ignore[call-arg]
                tool=ToolConfig(_env_file=env_file),  # type: ignore[call-arg]
            )
        return cls(provider=ProviderConfig(), agent=AgentConfig(), tool=ToolConfig())
