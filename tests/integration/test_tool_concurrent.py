"""Integration tests for tool concurrent execution."""

import asyncio

import pytest

from sanityops_agent.agents.factory import AgentFactory
from sanityops_agent.core.agent import TerminationReason
from sanityops_agent.core.message import FinishReason, TextBlock, ToolUseBlock
from sanityops_agent.llm.adapters.anthropic_adapter import AnthropicAdapter
from sanityops_agent.llm.adapters.openai_adapter import OpenAIAdapter
from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider, MockResponseConfig
from tests.mocks.mock_tools import MockToolWithError


class SlowTool(Tool):
    """A tool that simulates slow execution."""

    name = "slow_tool"
    description = "A slow tool"
    parameters = {
        "type": "object",
        "properties": {"delay": {"type": "number"}},
        "required": [],
    }

    async def execute(self, delay: float = 0.5, **kwargs) -> ToolResult:
        await asyncio.sleep(delay)
        return ToolResult(content=f"Slept {delay}s")


class TestToolConcurrent:
    """Test tool execution timing."""

    @pytest.mark.asyncio
    async def test_multiple_slow_tools_execute(self):
        """Multiple slow tools should all complete (currently sequential)."""
        tool = SlowTool()
        provider = MockLLMProvider()
        provider.set_responses(
            [
                MockResponseConfig(
                    content=[
                        ToolUseBlock(
                            tool_call_id="c1",
                            tool_name="slow_tool",
                            tool_input={"delay": 0.1},
                        ),
                        ToolUseBlock(
                            tool_call_id="c2",
                            tool_name="slow_tool",
                            tool_input={"delay": 0.1},
                        ),
                        ToolUseBlock(
                            tool_call_id="c3",
                            tool_name="slow_tool",
                            tool_input={"delay": 0.1},
                        ),
                    ],
                    stop_reason=FinishReason.TOOL_USE,
                ),
                MockResponseConfig(content=[TextBlock(text="Done")]),
            ]
        )
        registry = ToolRegistry()
        registry.register(tool)

        factory = AgentFactory(provider=provider, tool_registry=registry)
        agent = factory.create_parent_agent()

        result = await agent.run("Do 3 slow things")
        assert result.termination_reason == TerminationReason.END_TURN

    @pytest.mark.asyncio
    async def test_concurrent_tool_errors_all_reported(self):
        """When multiple tools fail, all errors should be handled."""
        failing_tool = MockToolWithError(error_message="boom")
        provider = MockLLMProvider()
        provider.set_responses(
            [
                MockResponseConfig(
                    content=[
                        ToolUseBlock(
                            tool_call_id="c1",
                            tool_name="mock_tool_error",
                            tool_input={},
                        ),
                        ToolUseBlock(
                            tool_call_id="c2",
                            tool_name="mock_tool_error",
                            tool_input={},
                        ),
                    ],
                    stop_reason=FinishReason.TOOL_USE,
                ),
                MockResponseConfig(content=[TextBlock(text="Done")]),
            ]
        )
        registry = ToolRegistry()
        registry.register(failing_tool)

        factory = AgentFactory(provider=provider, tool_registry=registry)
        agent = factory.create_parent_agent()

        result = await agent.run("Do two failing things")
        assert result.termination_reason == TerminationReason.END_TURN
        # Agent should continue even with tool errors


class TestParallelToolCallPayload:
    """The history a parallel tool turn produces must serialize to a valid
    payload for each provider (regression for the consecutive-message 400)."""

    @staticmethod
    async def _run_parallel_turn() -> tuple[list, list]:
        """Run a 2-parallel-call turn, return (messages of 1st, 2nd LLM call)."""
        tool = SlowTool()
        provider = MockLLMProvider()
        provider.set_responses(
            [
                MockResponseConfig(
                    content=[
                        ToolUseBlock(
                            tool_call_id="c1",
                            tool_name="slow_tool",
                            tool_input={"delay": 0},
                        ),
                        ToolUseBlock(
                            tool_call_id="c2",
                            tool_name="slow_tool",
                            tool_input={"delay": 0},
                        ),
                    ],
                    stop_reason=FinishReason.TOOL_USE,
                ),
                MockResponseConfig(content=[TextBlock(text="Done")]),
            ]
        )
        registry = ToolRegistry()
        registry.register(tool)

        factory = AgentFactory(provider=provider, tool_registry=registry)
        agent = factory.create_parent_agent()

        result = await agent.run("Do 2 things in parallel")
        assert result.termination_reason == TerminationReason.END_TURN

        # recorded_calls[0] is the first (user-only) call; [1] is the call whose
        # history already holds both tool results.
        assert len(provider.recorded_calls) == 2
        return provider.recorded_calls[0][0], provider.recorded_calls[1][0]

    @pytest.mark.asyncio
    async def test_history_has_one_user_message_per_result(self):
        """Documents the loop's raw shape: one USER message per tool result."""
        _, messages = await self._run_parallel_turn()
        tool_result_users = [
            m
            for m in messages
            if m.role.value == "user"
            and m.content
            and hasattr(m.content[0], "tool_call_id")
        ]
        assert len(tool_result_users) == 2

    @pytest.mark.asyncio
    async def test_anthropic_payload_merges_parallel_results(self):
        _, messages = await self._run_parallel_turn()
        payload = AnthropicAdapter().to_provider(messages)
        out = payload["messages"]

        # Roles alternate: all results land in a single user message.
        roles = [m["role"] for m in out]
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False))

        result_msgs = [
            m
            for m in out
            if m["role"] == "user"
            and any(b["type"] == "tool_result" for b in m["content"])
        ]
        assert len(result_msgs) == 1
        ids = [b["tool_use_id"] for b in result_msgs[0]["content"]]
        assert ids == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_openai_payload_has_tool_calls_then_tool_messages(self):
        _, messages = await self._run_parallel_turn()
        payload = OpenAIAdapter().to_provider(messages)
        out = payload["messages"]

        tool_msgs = [m for m in out if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]

        # The assistant message carrying tool_calls precedes them.
        assistant_idx = next(i for i, m in enumerate(out) if m.get("tool_calls"))
        first_tool_idx = next(i for i, m in enumerate(out) if m["role"] == "tool")
        assert assistant_idx < first_tool_idx
