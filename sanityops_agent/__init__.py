"""
Sanityops Agent — A modular, extensible AI Agent framework with multi-provider support
and plugin architecture.

Install via: pip install sanityops_agent
"""

__version__ = "0.0.5"

# Config
# Agents
from sanityops_agent.agents import (
    AgentFactory,
    HookBlockedException,
    ParentAgent,
    SubAgent,
    SubAgentRequest,
    SubAgentResponse,
)
from sanityops_agent.config import AgentConfig, AppConfig, ProviderConfig, ToolConfig

# Core
from sanityops_agent.core import (
    AgentContext,
    AgentError,
    AgentResult,
    BaseAgent,
    CompactConfig,
    ContentBlock,
    ContentType,
    ErrorBlock,
    ErrorType,
    FinishReason,
    Message,
    MessageToolResult,
    Role,
    StreamChunk,
    TerminationReason,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolResultSummary,
    ToolUseBlock,
)

# Hooks
from sanityops_agent.hooks import (
    BLOCKING_EVENTS,
    Hook,
    HookContext,
    HookEvent,
    HookExecutor,
    HookFailMode,
)
from sanityops_agent.hooks.builtin import (
    DebugHook,
    LoggingHook,
    MetricsData,
    MetricsHook,
    RateLimitHandlerHook,
    SafetyCheckHook,
)

# LLM
from sanityops_agent.llm import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    ContextWindowExceededError,
    LLMError,
    LLMProvider,
    LLMResponse,
    ProviderFactory,
    ProviderUnavailableError,
    RateLimitError,
)

# LLM Providers
from sanityops_agent.llm.adapters import AnthropicAdapter, MessageAdapter, OpenAIAdapter
from sanityops_agent.llm.providers import AnthropicProvider, OpenAIProvider

# Managers
from sanityops_agent.managers import (
    SkillManager,
    StateManager,
    TodoItem,
    TodoManager,
    ToolHistoryEntry,
    ToolHistoryStore,
)

# Tools
from sanityops_agent.tools import Tool, ToolContext, ToolRegistry, ToolResult

# Tools - Builtins
from sanityops_agent.tools.builtins import BashTool, FileOpsTool

__all__ = [
    # Version
    "__version__",
    # Config
    "AppConfig",
    "AgentConfig",
    "ProviderConfig",
    "ToolConfig",
    # Core Agent
    "BaseAgent",
    "AgentResult",
    "AgentError",
    "TerminationReason",
    # Core Context
    "AgentContext",
    "CompactConfig",
    "ToolResultSummary",
    # Messages
    "Message",
    "StreamChunk",
    "Role",
    "ContentType",
    "FinishReason",
    "ContentBlock",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ErrorBlock",
    "ToolDefinition",
    "ErrorType",
    "MessageToolResult",
    # LLM
    "LLMProvider",
    "LLMResponse",
    "ProviderFactory",
    "LLMError",
    "RateLimitError",
    "ContextWindowExceededError",
    "AuthenticationError",
    "ProviderUnavailableError",
    "APIError",
    "APIConnectionError",
    # LLM Providers
    "OpenAIProvider",
    "AnthropicProvider",
    # LLM Adapters
    "MessageAdapter",
    "OpenAIAdapter",
    "AnthropicAdapter",
    # Tools
    "Tool",
    "ToolResult",
    "ToolContext",
    "ToolRegistry",
    # Tools - Builtins
    "BashTool",
    "FileOpsTool",
    # Agents
    "AgentFactory",
    "ParentAgent",
    "HookBlockedException",
    "SubAgent",
    "SubAgentRequest",
    "SubAgentResponse",
    # Hooks
    "Hook",
    "HookEvent",
    "HookFailMode",
    "HookContext",
    "HookExecutor",
    "BLOCKING_EVENTS",
    "LoggingHook",
    "MetricsData",
    "MetricsHook",
    "SafetyCheckHook",
    "DebugHook",
    "RateLimitHandlerHook",
    # Managers
    "StateManager",
    "TodoItem",
    "TodoManager",
    "SkillManager",
    "ToolHistoryEntry",
    "ToolHistoryStore",
]
