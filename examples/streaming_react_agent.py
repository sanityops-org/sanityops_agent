"""
Streaming ReAct Agent — reads config from .env, runs agent with tools.

Features:
- Uses AgentFactory.create_parent_agent() from SDK (no custom _call_llm needed)
- Colored logging via Hook mechanism
- Supports OpenAI, Anthropic, and Gemini providers

Usage:
    cp .env.example .env   # fill in your API key
    uv run python examples/streaming_react_agent.py "What's 25 * 13?"
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from pydantic_settings import BaseSettings

from sanityops_agent.agents.factory import AgentFactory
from sanityops_agent.core.agent import AgentConfig
from sanityops_agent.core.message import TextBlock, ThinkingBlock, ToolUseBlock
from sanityops_agent.hooks.base import Hook, HookContext, HookEvent, HookExecutor
from sanityops_agent.llm.base import LLMProvider
from sanityops_agent.managers.history import ToolHistoryStore
from sanityops_agent.managers.skill import SkillManager
from sanityops_agent.managers.todo import TodoManager
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
    LLM_PROVIDER: str = "openai"  # "openai" | "anthropic" | "gemini"
    API_KEY: str = ""
    BASE_URL: str = ""
    MODEL: str = "gpt-4o-mini"

    # Gemini-specific (Vertex AI)
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = ""

    # Thinking configuration (Gemini 2.5+)
    THINKING_BUDGET: int = 0  # 0 = disabled, -1 = dynamic, >0 = token budget
    STORE_THINKING: bool = True

    # Agent behaviour
    MAX_LOOPS: int = 30
    MAX_TOKENS: int = 8000
    TOTAL_TIMEOUT: int = 300
    TEMPERATURE: float = 0.7
    SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Use tools when needed. Think step by step."
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


# Display limits for readable output
MAX_THINKING_DISPLAY = 500
MAX_TOOL_INPUT_DISPLAY = 200
MAX_TOOL_RESULT_DISPLAY = 300
MAX_SIGNATURE_DISPLAY = 20


def _print_tag(tag: str, colour: str) -> None:
    print(f"\n{C.DIVIDER}{'─' * 60}{C.RESET}")
    print(f"{colour}{C.BOLD}{tag}{C.RESET}")


# ---------------------------------------------------------------------------
# Logging Hook for colored output
# ---------------------------------------------------------------------------
class LoggingHook(Hook):
    """Hook that prints colored logs for LLM calls, tool execution, and errors."""

    name = "logging"
    description = "Print colored logs for agent events"
    priority = 0  # Run first
    events = {
        HookEvent.AFTER_LLM_CALL,
        HookEvent.BEFORE_TOOL_EXEC,
        HookEvent.AFTER_TOOL_EXEC,
        HookEvent.ON_ERROR,
        HookEvent.ON_TERMINATION,
    }

    async def handle(self, ctx: HookContext) -> HookContext:
        if ctx.event == HookEvent.AFTER_LLM_CALL:
            response = ctx.response
            if not response:
                return ctx

            # Print thinking blocks
            for block in response.content:
                if isinstance(block, ThinkingBlock):
                    _print_tag("[THINK]", C.THINK)
                    text = block.thinking
                    print(f"{C.DIM}{text[:MAX_THINKING_DISPLAY]}{'...' if len(text) > MAX_THINKING_DISPLAY else ''}{C.RESET}")
                    # Show signature if present
                    if block.thinking_signature:
                        sig = block.thinking_signature[:MAX_SIGNATURE_DISPLAY]
                        ellipsis = "..." if len(block.thinking_signature) > MAX_SIGNATURE_DISPLAY else ""
                        print(f"{C.DIM}  [signature: {sig}{ellipsis}]{C.RESET}")

            # Print text blocks
            for block in response.content:
                if isinstance(block, TextBlock):
                    _print_tag("[TEXT]", C.ANSWER)
                    print(block.text)
                    # Show signature if present
                    if block.text_signature:
                        sig = block.text_signature[:MAX_SIGNATURE_DISPLAY]
                        ellipsis = "..." if len(block.text_signature) > MAX_SIGNATURE_DISPLAY else ""
                        print(f"{C.DIM}  [signature: {sig}{ellipsis}]{C.RESET}")

            # Print tool calls
            tool_calls = [b for b in response.content if isinstance(b, ToolUseBlock)]
            if tool_calls:
                for tc in tool_calls:
                    sig_note = f" [sig: {tc.thought_signature[:MAX_SIGNATURE_DISPLAY]}...]" if tc.thought_signature else ""
                    print(f"\n{C.ACT}→ Tool: {tc.tool_name}({json.dumps(tc.tool_input)}){sig_note}{C.RESET}")

        elif ctx.event == HookEvent.BEFORE_TOOL_EXEC:
            _print_tag(f"[ACT] Calling {ctx.tool_name}", C.ACT)
            if ctx.tool_input:
                print(f"  Input: {json.dumps(ctx.tool_input)[:MAX_TOOL_INPUT_DISPLAY]}")

        elif ctx.event == HookEvent.AFTER_TOOL_EXEC:
            result = str(ctx.result) if ctx.result else ""
            status = "❌" if ctx.is_error else "✓"
            colour = C.ERROR if ctx.is_error else C.OBSERVE
            print(f"\n{colour}{status} [OBSERVE] {result[:MAX_TOOL_RESULT_DISPLAY]}{'...' if len(result) > MAX_TOOL_RESULT_DISPLAY else ''}{C.RESET}")

        elif ctx.event == HookEvent.ON_ERROR:
            _print_tag(f"[ERROR] {ctx.error}", C.ERROR)

        elif ctx.event == HookEvent.ON_TERMINATION:
            print(f"\n{C.DIVIDER}{'─' * 60}{C.RESET}")
            reason = ctx.data.get("reason", "unknown") if ctx.data else "unknown"
            print(f"{C.ANSWER}{C.BOLD}[DONE] {reason}{C.RESET}")

        return ctx


# ---------------------------------------------------------------------------
# Provider factory from env
# ---------------------------------------------------------------------------
def create_provider(env: AgentEnv) -> LLMProvider:
    """Instantiate an LLM provider from environment config."""
    provider_type = env.LLM_PROVIDER.lower()

    if provider_type == "anthropic":
        from sanityops_agent.llm.providers.anthropic import AnthropicProvider
        kwargs: dict[str, Any] = {"api_key": env.API_KEY, "model": env.MODEL}
        if env.BASE_URL:
            kwargs["base_url"] = env.BASE_URL
        return AnthropicProvider(**kwargs)

    elif provider_type == "gemini":
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        kwargs: dict[str, Any] = {
            "api_key": env.API_KEY,
            "model": env.MODEL,
            "store_thinking": env.STORE_THINKING,
        }
        if env.THINKING_BUDGET != 0:
            # Only -1 should map to None (dynamic budget), warn about other negative values
            import logging
            if env.THINKING_BUDGET < -1:
                logging.getLogger(__name__).warning(
                    "THINKING_BUDGET=%d is invalid. Only -1 (dynamic) or >0 (token budget) are valid. "
                    "Defaulting to dynamic budget.",
                    env.THINKING_BUDGET,
                )
            # -1 maps to None (dynamic budget), >0 maps to specific token budget
            kwargs["thinking_budget"] = env.THINKING_BUDGET if env.THINKING_BUDGET > 0 else None
        if env.GOOGLE_CLOUD_PROJECT:
            kwargs["project_id"] = env.GOOGLE_CLOUD_PROJECT
            if env.GOOGLE_CLOUD_LOCATION:
                kwargs["location"] = env.GOOGLE_CLOUD_LOCATION
        return GeminiProvider(**kwargs)

    else:  # openai (default)
        import logging
        if provider_type != "openai":
            logging.getLogger(__name__).warning(
                "Unknown LLM_PROVIDER '%s', defaulting to OpenAI. "
                "Valid providers: openai, anthropic, gemini",
                env.LLM_PROVIDER,
            )
        from sanityops_agent.llm.providers.openai import OpenAIProvider
        kwargs: dict[str, Any] = {"api_key": env.API_KEY, "model": env.MODEL}
        if env.BASE_URL:
            kwargs["base_url"] = env.BASE_URL
        return OpenAIProvider(**kwargs)


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

    # Build provider
    provider = create_provider(env)

    # Create hook executor with logging hook
    hook_executor = HookExecutor()
    hook_executor.register(LoggingHook())

    # Create shared context for tools
    tool_context = ToolContext()
    tool_context.set("todo_manager", TodoManager())
    tool_context.set("skill_manager", SkillManager())
    tool_context.set("history_store", ToolHistoryStore())
    tool_context.set("agent_factory", AgentFactory(provider))

    # Register tools
    registry = ToolRegistry(context=tool_context)
    registry.register(BashTool())
    registry.register(FileOpsTool())
    registry.register(GlobTool())
    registry.register(HistoryTool())
    registry.register(SkillTool())
    registry.register(TodoTool())
    registry.register(TaskTool())

    # Build factory
    factory = AgentFactory(
        provider=provider,
        config=AgentConfig(
            max_loops=env.MAX_LOOPS,
            max_tokens=env.MAX_TOKENS,
            total_timeout=env.TOTAL_TIMEOUT,
            system_prompt=env.SYSTEM_PROMPT,
        ),
        tool_registry=registry,
        hook_executor=hook_executor,
    )

    # Get user input from CLI args or stdin
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input(f"\n{C.BOLD}You: {C.RESET}")

    print(f"\n{C.DIM}{'═' * 60}{C.RESET}")
    print(f"{C.BOLD}Model: {env.MODEL} | Provider: {env.LLM_PROVIDER}{C.RESET}")
    if env.LLM_PROVIDER == "gemini" and env.THINKING_BUDGET != 0:
        print(f"{C.DIM}Thinking Budget: {env.THINKING_BUDGET} tokens{C.RESET}")
    print(f"{C.DIM}{'═' * 60}{C.RESET}")

    # Create agent and run
    agent = factory.create_parent_agent()

    t0 = time.time()
    result = await agent.run(prompt)
    elapsed = time.time() - t0

    # Final summary
    print(f"\n{C.DIVIDER}{'─' * 60}{C.RESET}")
    print(f"{C.ANSWER}{C.BOLD}[SUMMARY]{C.RESET}")
    print(f"  Reason:       {result.termination_reason.value}")
    print(f"  Loops:        {result.loops_used}")
    print(f"  Time:         {elapsed:.2f}s")
    print(f"  Tokens:       {result.tokens_used}")
    if result.error:
        print(f"\n{C.ERROR}Error: {result.error}{C.RESET}")
    print(f"{C.DIVIDER}{'─' * 60}{C.RESET}")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
