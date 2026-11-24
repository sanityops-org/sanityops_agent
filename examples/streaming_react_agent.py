"""Stream model answers and report tool progress.

Supports OpenAI, Anthropic, and Gemini providers.

Copy .env.example to .env in a trusted project directory, then run
uv run python examples/streaming_react_agent.py "What's 25 * 13?"
"""

from __future__ import annotations

import ast
import asyncio
import operator
import sys
from datetime import datetime

from pydantic import Field
from pydantic_settings import BaseSettings

from sanityops_agent.agents.factory import AgentFactory
from sanityops_agent.agents.parent import ParentAgent
from sanityops_agent.config import ProviderConfig
from sanityops_agent.core.agent import AgentConfig
from sanityops_agent.core.message import ContentType
from sanityops_agent.hooks.base import Hook, HookContext, HookEvent, HookExecutor
from sanityops_agent.llm.base import LLMProvider, LLMResponse
from sanityops_agent.llm.factory import ProviderFactory
from sanityops_agent.llm.stream import collect_stream
from sanityops_agent.managers.history import ToolHistoryStore
from sanityops_agent.managers.todo import TodoManager
from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.builtins import HistoryTool, TaskTool, TodoTool
from sanityops_agent.tools.context import ToolContext
from sanityops_agent.tools.registry import ToolRegistry


class AgentEnv(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    LLM_PROVIDER: str = "openai"  # "openai" | "anthropic" | "gemini"
    API_KEY: str = Field(default="", repr=False, exclude=True)
    BASE_URL: str = ""
    MODEL: str = "gpt-4o-mini"

    # Gemini-specific (Vertex AI - optional, uses ADC if PROJECT is set)
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = ""
    # 0 = disabled, -1 = dynamic budget, >0 = specific token budget
    THINKING_BUDGET: int = 0
    STORE_THINKING: bool = True

    MAX_LOOPS: int = 30
    MAX_TOKENS: int | None = None
    TOTAL_TIMEOUT: float = 300
    TEMPERATURE: float = 0.7
    SYSTEM_PROMPT: str = "You are a helpful assistant. Use tools when needed."


class CalculatorTool(Tool):
    """Evaluate bounded arithmetic expressions."""

    name = "calculator"
    description = "Evaluate arithmetic with +, -, *, /, //, %, ** and parentheses."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "maxLength": 256}},
        "required": ["expression"],
        "additionalProperties": False,
    }
    _binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    @classmethod
    def _evaluate(cls, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in cls._unary:
            result = cls._unary[type(node.op)](cls._evaluate(node.operand))
        elif isinstance(node, ast.BinOp) and type(node.op) in cls._binary:
            left, right = cls._evaluate(node.left), cls._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent out of range")
            result = cls._binary[type(node.op)](left, right)
        else:
            raise ValueError("Unsupported arithmetic expression")
        if type(result) not in (int, float) or not -1e100 <= result <= 1e100:
            raise ValueError("Result out of range")
        return result

    async def execute(self, expression: str, **kwargs) -> ToolResult:
        if len(expression) > 256:
            return ToolResult("Expression is too long", success=False)
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            return ToolResult(str(self._evaluate(tree.body)))
        except (SyntaxError, ValueError, ArithmeticError):
            return ToolResult("Invalid or out-of-range arithmetic", success=False)


class CurrentTimeTool(Tool):
    name = "current_time"
    description = "Return the current local time"
    parameters = {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class ToolProgress(Hook):
    events = [HookEvent.AFTER_TOOL_EXEC]

    async def handle(self, ctx: HookContext) -> HookContext:
        status = "failed" if ctx.is_error else "completed"
        print(f"[Tool] {ctx.tool_name} {status}", file=sys.stderr)
        return ctx


def create_provider(env: AgentEnv) -> LLMProvider:
    if env.THINKING_BUDGET < -1:
        sys.exit(
            f"THINKING_BUDGET={env.THINKING_BUDGET} is invalid. "
            "Only -1 (dynamic) or >0 (token budget) are supported."
        )
    config = ProviderConfig(
        _env_file=None,
        PROVIDER=env.LLM_PROVIDER,
        API_KEY=env.API_KEY,
        BASE_URL=env.BASE_URL or None,
        MODEL_ID=env.MODEL,
        # Thinking budget 0 and -1 both disable the explicit budget field.
        GEMINI_THINKING_BUDGET=(
            env.THINKING_BUDGET if env.THINKING_BUDGET > 0 else None
        ),
        GEMINI_STORE_THINKING=env.STORE_THINKING,
        GEMINI_PROJECT_ID=env.GOOGLE_CLOUD_PROJECT or None,
        GEMINI_LOCATION=env.GOOGLE_CLOUD_LOCATION or None,
    )
    return ProviderFactory.create(config)


class StreamingReActAgent(ParentAgent):
    """Use the parent agent's policy and execute complete streamed responses."""

    async def _call_llm(self) -> LLMResponse:
        assert self.context is not None
        tools = self._tool_definitions()
        kwargs = await self._provider_kwargs(self.provider, tools)

        def display(chunk):
            if chunk.type is ContentType.TEXT and isinstance(chunk.delta, str):
                print(chunk.delta, end="", flush=True)

        result = await collect_stream(
            self.provider.stream(self.context.messages, tools=tools, **kwargs),
            model=getattr(self.provider, "model", "unknown"),
            on_chunk=display,
        )
        print()
        if self._usage is None:
            self._record_usage(result.usage)
        if self.hook_executor:
            await self.hook_executor.execute(
                HookEvent.AFTER_LLM_CALL, {"response": result}
            )
        return result


async def main():
    env = AgentEnv()
    if not env.API_KEY and not env.GOOGLE_CLOUD_PROJECT:
        sys.exit("API_KEY is required in the environment or a trusted .env")
    if env.LLM_PROVIDER.lower() == "anthropic" and not env.MAX_TOKENS:
        sys.exit("MAX_TOKENS must be a positive integer when using Anthropic")

    config = AgentConfig(
        max_loops=env.MAX_LOOPS,
        max_tokens=env.MAX_TOKENS,
        total_timeout=env.TOTAL_TIMEOUT,
        system_prompt=env.SYSTEM_PROMPT,
    )
    provider = create_provider(env)
    context = ToolContext()
    context.set("todo_manager", TodoManager())
    context.set("history_store", ToolHistoryStore())
    registry = ToolRegistry(context)
    for tool in (
        CalculatorTool(),
        CurrentTimeTool(),
        HistoryTool(),
        TodoTool(),
        TaskTool(),
    ):
        registry.register(tool)

    hooks = HookExecutor()
    hooks.register(ToolProgress())
    factory = AgentFactory(provider, config, registry, hooks)
    agent = StreamingReActAgent(config, provider, registry, hooks)

    try:
        prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("You: ")
        result = await agent.run(prompt)
        usage = (
            str(result.tokens_used)
            if result.usage_complete
            else f"{result.tokens_used} reported"
        )
        print(
            f"[{result.termination_reason.value}] {result.loops_used} turns, {usage} tokens"
        )
        if result.error:
            print(result.error, file=sys.stderr)
    finally:
        factory.close()
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
