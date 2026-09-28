"""
Streaming ReAct Agent — reads config from .env, streams LLM output with
real-time logging of thought process, tool calls, and observations.

Usage:
    cp .env.example .env   # fill in your API key and base_url
    uv run python examples/streaming_react_agent.py "What's 25 * 13?"
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# .env configuration via pydantic-settings
# ---------------------------------------------------------------------------
from pydantic_settings import BaseSettings

from sanityops_agent.agents.factory import AgentFactory
from sanityops_agent.core.agent import (
    MAX_RETRY_DELAY,
    AgentConfig,
    AgentResult,
    BaseAgent,
)
from sanityops_agent.core.message import (
    ContentType,
    FinishReason,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.llm.base import LLMProvider, LLMResponse
from sanityops_agent.llm.providers.anthropic import AnthropicProvider
from sanityops_agent.managers.history import ToolHistoryStore
from sanityops_agent.managers.skill import SkillManager
from sanityops_agent.managers.todo import TodoManager
from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.builtins import (
    BashTool,
    FileOpsTool,
    GlobTool,
    HistoryTool,
    SkillTool,
    TaskTool,
    TodoTool,
)
from sanityops_agent.tools.context import ToolContext
from sanityops_agent.tools.registry import ToolRegistry


class AgentEnv(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # LLM provider
    LLM_PROVIDER: str = "openai"  # "openai" | "anthropic"
    API_KEY: str = ""
    BASE_URL: str = ""
    MODEL: str = "gpt-4o-mini"

    # Agent behaviour
    MAX_LOOPS: int = 30
    MAX_TOKENS: int = 8000
    TOTAL_TIMEOUT: int = 300
    TEMPERATURE: float = 0.7
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Use tools when needed. "
        "Think step by step."
    )


# ---------------------------------------------------------------------------
# Terminal colour helpers
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    THINK = "\033[36m"       # cyan
    ACT = "\033[33m"         # yellow
    OBSERVE = "\033[32m"     # green
    ANSWER = "\033[1;35m"    # bold magenta
    ERROR = "\033[1;31m"     # bold red
    DIVIDER = "\033[2;90m"   # dim grey


# Maximum characters to preview in [OBSERVE] output
MAX_RESULT_PREVIEW = 200


def _print_tag(tag: str, colour: str) -> None:
    print(f"\n{C.DIVIDER}{'─' * 60}{C.RESET}")
    print(f"{colour}{C.BOLD}{tag}{C.RESET}")


# ---------------------------------------------------------------------------
# Simple built-in tools for the demo
# ---------------------------------------------------------------------------
class CalculatorTool(Tool):
    """Evaluate a simple arithmetic expression safely.

    Note: Only basic arithmetic is supported (+, -, *, /, **, %, ()).
    Scientific notation (e.g., 1e10) is not supported.
    """

    name = "calculator"
    description = (
        "Evaluate a math expression. "
        "Supports: +, -, *, /, ** (power), % (modulo), (). "
        "No scientific notation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression, e.g. '25 * 13'"},
        },
        "required": ["expression"],
    }

    async def execute(self, expression: str, **kwargs: Any) -> ToolResult:
        allowed = set("0123456789+-*/.()% ")
        if any(c not in allowed for c in expression):
            return ToolResult(content=f"Error: invalid characters in '{expression}'", success=False)
        try:
            # Safe evaluation using AST parsing instead of eval()
            # Supports: +, -, *, /, **, %, () and numeric literals
            import ast
            tree = ast.parse(expression, mode='eval')
            # Only allow BinOp with allowed operators and numeric literals
            def _eval_node(node):
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, int | float):
                        return node.value
                    raise ValueError(f"Invalid constant type: {type(node.value)}")
                if isinstance(node, ast.BinOp):
                    left = _eval_node(node.left)
                    right = _eval_node(node.right)
                    if isinstance(node.op, ast.Add):
                        return left + right
                    if isinstance(node.op, ast.Sub):
                        return left - right
                    if isinstance(node.op, ast.Mult):
                        return left * right
                    if isinstance(node.op, ast.Div):
                        return left / right
                    if isinstance(node.op, ast.Mod):
                        return left % right
                    if isinstance(node.op, ast.Pow):
                        return left ** right
                    if isinstance(node.op, ast.FloorDiv):
                        return left // right
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                if isinstance(node, ast.UnaryOp):
                    if isinstance(node.op, ast.USub):
                        return -_eval_node(node.operand)
                    if isinstance(node.op, ast.UAdd):
                        return _eval_node(node.operand)
                if isinstance(node, ast.Expression):
                    return _eval_node(node.body)
                raise ValueError(f"Unsupported expression element: {type(node).__name__}")
            result = _eval_node(tree)
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=f"Error: {e}", success=False)


class CurrentTimeTool(Tool):
    """Return the current local time."""

    name = "current_time"
    description = "Return the current local time as a string."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> ToolResult:
        from datetime import datetime
        return ToolResult(content=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------------------
# Provider factory from env
# ---------------------------------------------------------------------------
def create_provider(env: AgentEnv) -> LLMProvider:
    """Instantiate an LLM provider from environment config."""
    if env.LLM_PROVIDER.lower() == "anthropic":
        kwargs: dict[str, Any] = {"api_key": env.API_KEY, "model": env.MODEL}
        if env.BASE_URL:
            kwargs["base_url"] = env.BASE_URL
        return AnthropicProvider(**kwargs)
    else:
        from sanityops_agent.llm.providers.openai import OpenAIProvider
        kwargs: dict[str, Any] = {"api_key": env.API_KEY, "model": env.MODEL}
        if env.BASE_URL:
            kwargs["base_url"] = env.BASE_URL
        return OpenAIProvider(**kwargs)


# ---------------------------------------------------------------------------
# Streaming ReAct Agent
# ---------------------------------------------------------------------------
class StreamingReActAgent(BaseAgent):
    """ReAct agent that streams the LLM output and logs each phase.

    Overrides the BaseAgent loop to use provider.stream() and print
    chunks in real-time:
      [THINK]   — streaming text output
      [ACT]     — tool name + arguments when tool_use detected
      [OBSERVE] — tool result after execution
      [ANSWER]  — final text answer
    """

    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        tool_registry: ToolRegistry,
    ):
        super().__init__(config)
        self.provider = provider
        self.tool_registry = tool_registry

    # -- overrides required by Basesanityops_agent.run() --

    async def _call_llm(self) -> LLMResponse:
        """Stream the LLM response, print chunks in real-time, and buffer into LLMResponse."""
        tool_defs = self.tool_registry.to_tool_definitions(
            self.config.allowed_tool_tags,
        )

        # Buffers for streamed content
        text_buffer: str = ""
        thinking_buffer: str = ""
        # key: tool index -> {name, input_str, tool_call_id}
        tool_call_buffers: dict[int, dict[str, Any]] = {}
        stop_reason = FinishReason.END_TURN
        thinking_tag_printed = False

        # Stream and print in real-time
        async for chunk in self.provider.stream(
            self.context.messages, tools=tool_defs,
        ):
            if chunk.type == ContentType.THINKING and chunk.delta:
                # Print the tag lazily, on the first thinking chunk, so a
                # response with no reasoning does not emit an empty [THINK].
                if not thinking_tag_printed:
                    _print_tag("[THINK]", C.THINK)
                    thinking_tag_printed = True
                thinking_buffer += chunk.delta
                print(f"{C.DIM}{chunk.delta}{C.RESET}", end="", flush=True)

            elif chunk.type == ContentType.TEXT and chunk.delta:
                text_buffer += chunk.delta
                print(chunk.delta, end="", flush=True)

            elif chunk.type == ContentType.TOOL_USE and chunk.delta:
                # Identify which tool call this chunk belongs to
                idx = chunk.index
                if idx not in tool_call_buffers:
                    tool_call_buffers[idx] = {
                        "name": "",
                        "input_str": "",
                        "tool_call_id": chunk.tool_call_id or f"stream_call_{idx}",
                    }

                buf = tool_call_buffers[idx]

                if isinstance(chunk.delta, dict):
                    # OpenAI format: {"name": "...", "arguments": "..."}
                    # Anthropic format: {"name": "...", "input": {...}}
                    if "name" in chunk.delta:
                        chunk_name = chunk.delta["name"]
                        if chunk_name:
                            buf["name"] = chunk_name

                    # OpenAI uses "arguments" (string, incremental JSON fragment)
                    if "arguments" in chunk.delta:
                        buf["input_str"] += chunk.delta["arguments"]

                    # Anthropic uses "input" (dict, full replacement each chunk)
                    if "input" in chunk.delta:
                        inp = chunk.delta["input"]
                        buf["input_str"] = json.dumps(inp) if isinstance(inp, dict) else str(inp)

                    # Track tool_call_id
                    if chunk.tool_call_id:
                        buf["tool_call_id"] = chunk.tool_call_id

            elif chunk.finish_reason:
                # Last finish_reason wins; the stream ends with the final value
                stop_reason = chunk.finish_reason

        # Close the streamed line (thinking or text) before the next phase prints.
        print()

        # Build content blocks from buffered data
        content: list = []
        if thinking_buffer.strip():
            content.append(ThinkingBlock(thinking=thinking_buffer.strip()))
        if text_buffer.strip():
            content.append(TextBlock(text=text_buffer.strip()))

        for idx in sorted(tool_call_buffers.keys()):
            buf = tool_call_buffers[idx]
            name = buf.get("name", "unknown_tool")
            input_str = buf.get("input_str", "{}")
            try:
                tool_input = json.loads(input_str) if input_str else {}
            except json.JSONDecodeError:
                # Malformed JSON in tool arguments. Keep the raw text under a
                # reserved key so the tool still receives something actionable
                # and the full payload is visible in logs.
                logging.getLogger(__name__).warning(
                    "Malformed JSON in tool input for %s (raw: %r); "
                    "passing as {\"_raw_input\": ...}",
                    name,
                    input_str[:200],
                )
                tool_input = {"_raw_input": input_str}
            content.append(ToolUseBlock(
                tool_call_id=buf.get("tool_call_id", f"stream_call_{idx}"),
                tool_name=name,
                tool_input=tool_input,
            ))

        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            model=self.provider.model if hasattr(self.provider, "model") else "unknown",
            usage={"total_tokens": 0},
        )

    async def _call_llm_with_retry(self) -> LLMResponse:
        """Agent-level retry with recoverable check and error logging."""
        for attempt in range(self.config.max_retries):
            try:
                return await self._call_llm()
            except Exception as e:
                recoverable = getattr(e, "recoverable", False)
                print(
                    f"\n{C.ERROR}[ERROR] LLM call failed "
                    f"(attempt {attempt+1}/{self.config.max_retries}): {e}{C.RESET}"
                )
                self._record_error("llm_error", str(e), 0, recoverable=recoverable)
                if not recoverable:
                    raise
                if attempt == self.config.max_retries - 1:
                    raise
                delay = self._compute_retry_delay(e, attempt)
                await asyncio.sleep(delay)

    def _compute_retry_delay(self, e: Exception, attempt: int) -> float:
        """Prefer the provider's Retry-After, capped; fall back to backoff."""
        retry_after = getattr(e, "retry_after", None)
        if retry_after and retry_after > 0:
            return float(min(retry_after, MAX_RETRY_DELAY))
        return float(min(2**attempt, MAX_RETRY_DELAY))

    async def _execute_tools(self, tool_calls: list) -> None:
        """Execute tools and log the observations."""
        for tc in tool_calls:
            tool_name = tc.tool_name
            tool_input = tc.tool_input
            tool_call_id = tc.tool_call_id

            _print_tag(f"[ACT] Calling {tool_name}({json.dumps(tool_input)})", C.ACT)

            tool = self.tool_registry.get(tool_name)
            if tool is None:
                result_content = f"Tool not found: {tool_name}"
                is_error = True
            else:
                try:
                    result = await tool.run(**tool_input)
                    result_content = result.content
                    is_error = not result.success
                except Exception as e:
                    result_content = f"Tool error: {e}"
                    is_error = True

            _print_tag(f"[OBSERVE] {result_content[:MAX_RESULT_PREVIEW]}{'...' if len(result_content) > MAX_RESULT_PREVIEW else ''}", C.OBSERVE)

            # Anthropic requires tool results to be in user messages with tool_result
            # content blocks. OpenAI requires role='tool'.
            tool_result_role = Role.USER if isinstance(self.provider, AnthropicProvider) else Role.TOOL

            await self.context.add_message(
                Message(
                    role=tool_result_role,
                    content=[
                        ToolResultBlock(
                            tool_call_id=tool_call_id,
                            tool_result_content=result_content,
                            is_error=is_error,
                        )
                    ],
                )
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main():
    # Load .env
    if not os.path.exists(".env"):
        print(f"{C.ERROR}Error: .env file not found.{C.RESET}")
        print("Create a .env file with your API credentials. See .env.example for format.")
        sys.exit(1)

    env = AgentEnv()

    if not env.API_KEY:
        print(f"{C.ERROR}Error: API_KEY is not set in .env{C.RESET}")
        sys.exit(1)

    # Build provider and tools
    provider = create_provider(env)

    # Create shared context for tools
    tool_context = ToolContext()
    tool_context.set("todo_manager", TodoManager())
    tool_context.set("skill_manager", SkillManager())
    tool_context.set("history_store", ToolHistoryStore())
    tool_context.set("agent_factory", AgentFactory(provider))

    registry = ToolRegistry(context=tool_context)
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())
    registry.register(BashTool())
    registry.register(FileOpsTool())
    registry.register(GlobTool())
    registry.register(HistoryTool())
    registry.register(SkillTool())
    registry.register(TodoTool())
    registry.register(TaskTool())

    # Build config from env
    config = AgentConfig(
        max_loops=env.MAX_LOOPS,
        max_tokens=env.MAX_TOKENS,
        total_timeout=env.TOTAL_TIMEOUT,
        system_prompt=env.SYSTEM_PROMPT,
    )

    # Build streaming agent
    agent = StreamingReActAgent(config, provider, registry)

    # Get user input from CLI args or stdin
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input(f"\n{C.BOLD}You: {C.RESET}")

    print(f"\n{C.DIM}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}Model: {env.MODEL} | Provider: {env.LLM_PROVIDER}{C.RESET}")
    print(f"{C.DIM}{'═' * 60}{C.RESET}")

    # Run
    t0 = time.time()
    result: AgentResult = await agent.run(prompt)
    elapsed = time.time() - t0

    # Final summary
    print(f"\n{C.DIVIDER}{'─' * 60}{C.RESET}")
    print(f"{C.ANSWER}{C.BOLD}[ANSWER] Summary{C.RESET}")
    print(f"  Reason:       {result.termination_reason.value}")
    print(f"  Loops:        {result.loops_used}")
    print(f"  Time:         {elapsed:.2f}s")
    if result.text:
        print(f"\n{C.ANSWER}{result.text}{C.RESET}")
    if result.error:
        print(f"\n{C.ERROR}Error: {result.error}{C.RESET}")
    print(f"{C.DIVIDER}{'─' * 60}{C.RESET}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
