# Sanityops Agent

[![PyPI version](https://img.shields.io/pypi/v/sanityops_agent.svg)](https://pypi.org/project/sanityops_agent/)
[![Python versions](https://img.shields.io/pypi/pyversions/sanityops_agent.svg)](https://pypi.org/project/sanityops_agent/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A modular, extensible AI Agent framework with OOP design, multi-provider support,
and a plugin architecture for tools, hooks, and multi-agent workflows.

## Features

- **Unified Message Model**: Subclassed content blocks (`TextBlock`, `ToolUseBlock`,
  `ThinkingBlock`, etc.) with type-safe access and `FinishReason` enums.
- **LLM Provider Abstraction**: Swap between OpenAI, Anthropic, or any custom provider
  via the `LLMProvider` interface. Adapters handle protocol differences.
- **Plugin Tools**: Register custom tools by subclassing `Tool` with JSON Schema
  parameters. Tools support tags (e.g., `parent_only`) for access control.
- **Multi-Agent**: Parent agent can spawn sub-agents with filtered tool access and
  tighter limits (max_loops=10, timeout=120s).
- **Hook System**: Intercept tool calls, LLM responses, and agent lifecycle events
  for approval workflows, logging, or blocking.
- **State Persistence**: Save/restore conversation context as JSON.
- **Streaming**: Token-by-token streaming with `StreamChunk` dataclass.
- **Retry Logic**: Exponential backoff for transient LLM errors.

## Installation

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd new_agent
uv sync

# Install with dev dependencies (pytest, etc.)
uv sync --all-extras
```

### Manual pip install

```bash
pip install -e ".[dev]"
```

## Quick Start

### Minimal Example

```python
import asyncio
from sanityops_agent.llm.providers.openai import OpenAIProvider
from sanityops_agent.agents.factory import AgentFactory
from sanityops_agent.tools.registry import ToolRegistry


async def main():
    provider = OpenAIProvider(
        api_key="your-key",  # or set OPENAI_API_KEY env var
        model="gpt-4o-mini",
    )

    factory = AgentFactory(
        provider=provider,
        tool_registry=ToolRegistry(),
    )
    agent = factory.create_parent_agent(
        system_prompt="You are a helpful assistant.",
    )

    result = await agent.run("What is the capital of France?")
    print(result.text)
    print(f"Tokens: {result.tokens_used}, Loops: {result.loops_used}")

    await provider.close()


asyncio.run(main())
```

### Custom Tool

```python
from sanityops_agent.tools.base import Tool, ToolResult


class WeatherTool(Tool):
    name = "get_weather"
    description = "Get weather for a city."
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
        },
        "required": ["city"],
    }

    async def execute(self, city: str, **kwargs) -> ToolResult:
        # Call weather API here
        return ToolResult(content=f"Sunny, 25°C in {city}")


# Register it
registry = ToolRegistry()
registry.register(WeatherTool())

factory = AgentFactory(provider=provider, tool_registry=registry)
agent = factory.create_parent_agent()
result = await agent.run("What's the weather in Tokyo?")
```

### State Persistence

```python
from sanityops_agent.managers.state import StateManager

state = StateManager(state_path="session.json")

# Save
await state.save(agent.context)

# Resume
context = await state.load()
agent.context = context
```

### Sub-Agent Creation

```python
# Sub-agents have filtered tools (no parent_only) and tighter limits
sub = factory.create_sub_agent(
    system_prompt="You are a code reviewer.",
)
result = await sub.run("Review this function: def foo(x): return x + 1")
```

## Project Structure

```
sanityops_agent/
├── core/          # Message model, Agent base class, Context
├── llm/           # Provider abstraction, adapters, exceptions
│   └── providers/ # OpenAI, Anthropic implementations
├── tools/         # Tool base class, registry, context
│   └── builtins/  # Built-in tools (bash, file_ops, todo, etc.)
├── agents/        # Parent agent, Sub-agent, Factory
├── hooks/         # Hook system (before/after tool, LLM, etc.)
├── managers/      # State, skill, todo, history persistence
└── legacy/        # Backward compatibility adapter

tests/
├── unit/          # Isolated component tests
├── integration/   # Multi-component workflow tests
├── mocks/         # MockLLMProvider, mock tools
└── conftest.py    # Pytest fixtures

examples/
├── basic_usage.py           # Interactive examples (6 scenarios)
└── streaming_react_agent.py # Streaming ReAct agent with .env config
```

## Configuration

### Using .env File

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Supported environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | Provider name: `openai`, `anthropic`, or `gemini` |
| `API_KEY` | *(required)* | Your API key |
| `BASE_URL` | *(empty)* | Custom endpoint (vLLM, Azure, etc.) |
| `MODEL` | `gpt-4o-mini` | Model name |
| `EXTRA_BODY` | *(empty)* | JSON object forwarded to the provider SDK request (Anthropic/OpenAI) |
| `MAX_LOOPS` | `30` | Max agent loops per run |
| `MAX_TOKENS` | unset | Max output tokens per request; the example file leaves it unset. Anthropic requires an explicit positive limit. |
| `TOTAL_TIMEOUT` | `300` | Total timeout in seconds |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `SYSTEM_PROMPT` | *(see .env)* | System prompt for the agent |

### Runtime configuration and limits

`AgentFactory` accepts the environment-backed `sanityops_agent.AgentConfig`
or the explicit `RuntimeAgentConfig`. The latter is also available from
`sanityops_agent.core.agent` as `AgentConfig`.

Anthropic accepts `max_tokens` on the request, in the effective `extra_body`,
or as an explicit provider constructor argument, in that order. The streaming
example passes its configured limit to the parent and delegated agents.

Each `run()` has its own deadline, result and token budget. Delegated work
uses the parent's remaining time and shares its budget. If a provider reports
incomplete usage, `AgentResult.usage_complete` is false. For strict budgeting,
set `strict_token_budget=True` and supply a provider with an async
`count_input_tokens(messages, tools)` method that counts the intended
request. The agent stops before requests it cannot budget.

### Tool permissions and parameters

The agent uses the same tool selection when it builds a model request and
executes its response. Hosts can set `RuntimeAgentConfig.allowed_tool_names`
or `ToolContext.set("allowed_tools", names)` to a set of registered names.
An explicit empty set disables tools. The existing tag filter remains
available. A delegated task can only narrow the parent's tool access.

OpenAI tools use regular Schema validation by default. To explicitly request
OpenAI strict mode, set `definition.provider_config["strict"] = True` on the
`ToolDefinition` returned by the tool. Strict schemas must satisfy OpenAI's
object requirements. The agent validates arguments against the registered
Schema before executing the tool in either mode.

The streaming example registers calculation, time, history, local to-dos
and delegation. To add built-in file tools, supply a host-selected POSIX
workspace with `ToolContext.set("workdir", absolute_path)`. Write and patch
also require `allow_file_writes=True` in that context. A model-requested Bash
call requires a host-provided isolated `shell_runner` in the same context.
The runner is an async callable accepting the keyword arguments `command`,
`timeout` and `max_output_bytes` and returning a `ToolResult`. The host
controls mounts, inherited environment, network access and process cleanup.

### Managed services

Use one factory, tool registry, tool context and set of stateful managers per
authenticated session. Reuse those objects for sequential turns in that
session. For persistent state, pass the session identifier and trusted
system prompt to `StateManager` and keep the file in an application-owned
private directory.

`AppConfig.load(managed=True)` reads from the process environment or an
explicit absolute `env_file`. Pass `allowed_base_urls` to
`ProviderFactory.create` to restrict the provider to approved HTTPS
destinations and disable redirects and environment proxies.

## Running Examples

```bash
# 1. Set up your .env first
cp .env.example .env
# Edit .env with your API key

# 2. Streaming ReAct Agent (reads .env, streams output)
uv run python examples/streaming_react_agent.py "What's 25 * 13?"

# 3. Interactive basic examples menu
uv run python examples/basic_usage.py
```



## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Subclassed content blocks | Type safety, static analysis catches wrong field access |
| `FinishReason` enum over strings | No magic strings, IDE autocomplete |
| `content: list[...]` always | Simplifies adapter logic, no str/list branching |
| Provider is pure call, no retry | Single responsibility, retry belongs in Agent layer |
| Tools use JSON Schema params | Compatible with OpenAI/Anthropic tool calling APIs |
| Sequential tool execution | Simpler error handling, predictable ordering |
| AgentFactory owns config | Factory config propagates to created agents |

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.
