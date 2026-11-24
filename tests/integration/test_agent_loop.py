"""Integration tests - agent loop with mock LLM."""

import asyncio

import pytest

from sanityops_agent import AgentConfig as EnvironmentAgentConfig
from sanityops_agent.agents.factory import AgentFactory
from sanityops_agent.core.agent import AgentConfig, TerminationReason
from sanityops_agent.core.message import FinishReason, TextBlock, ToolUseBlock
from sanityops_agent.hooks.base import Hook, HookContext, HookEvent, HookExecutor
from sanityops_agent.llm.exceptions import LLMError
from sanityops_agent.managers.history import ToolHistoryStore
from sanityops_agent.tools.builtins.task import TaskTool
from sanityops_agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider, MockResponseConfig
from tests.mocks.mock_tools import MockTool


class TestAgentLoop:
    """Test the full agent loop with mock LLM and tools."""

    @pytest.mark.asyncio
    async def test_text_only_response_ends_loop(self):
        """Text-only response should end the loop immediately."""
        provider = MockLLMProvider()
        provider.set_responses(
            [
                MockResponseConfig(content=[TextBlock(text="Hello there!")]),
            ]
        )
        registry = ToolRegistry()
        factory = AgentFactory(provider=provider, tool_registry=registry)
        agent = factory.create_parent_agent(system_prompt="You are helpful.")

        result = await agent.run("Say hello")
        assert result.termination_reason == TerminationReason.END_TURN
        assert "Hello there!" in result.text
        assert result.loops_used == 1

    @pytest.mark.asyncio
    async def test_tool_use_then_end_turn(self):
        """Tool use followed by end_turn should complete successfully."""
        tool = MockTool(response="tool output")
        provider = MockLLMProvider()
        provider.set_responses(
            [
                MockResponseConfig(
                    content=[
                        ToolUseBlock(
                            tool_call_id="call_1",
                            tool_name="mock_tool",
                            tool_input={"input": "test"},
                        )
                    ],
                    stop_reason=FinishReason.TOOL_USE,
                ),
                MockResponseConfig(
                    content=[TextBlock(text="Done!")],
                    stop_reason=FinishReason.END_TURN,
                ),
            ]
        )
        registry = ToolRegistry()
        registry.register(tool)

        factory = AgentFactory(provider=provider, tool_registry=registry)
        agent = factory.create_parent_agent(system_prompt="You are helpful.")

        result = await agent.run("Do something")
        assert result.termination_reason == TerminationReason.END_TURN
        assert "Done!" in result.text
        assert result.loops_used == 2

    @pytest.mark.asyncio
    async def test_max_loops_termination(self):
        """Agent should stop after max_loops."""
        tool = MockTool(response="ok")
        provider = MockLLMProvider()
        # Keep returning tool_use to force more loops
        for i in range(10):
            provider.response_sequence.append(
                MockResponseConfig(
                    content=[
                        ToolUseBlock(
                            tool_call_id=f"call_{i}",
                            tool_name="mock_tool",
                            tool_input={"input": "loop"},
                        )
                    ],
                    stop_reason=FinishReason.TOOL_USE,
                ),
            )
        registry = ToolRegistry()
        registry.register(tool)

        config = AgentConfig(max_loops=5)
        factory = AgentFactory(provider=provider, config=config, tool_registry=registry)
        agent = factory.create_parent_agent()

        result = await agent.run("keep going")
        assert result.termination_reason == TerminationReason.MAX_LOOPS
        assert result.loops_used == 5

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_all_execute(self):
        """Multiple tool calls in one response should all execute."""
        tool = MockTool(response="result")
        provider = MockLLMProvider()
        provider.set_responses(
            [
                MockResponseConfig(
                    content=[
                        ToolUseBlock(
                            tool_call_id="c1",
                            tool_name="mock_tool",
                            tool_input={"input": "a"},
                        ),
                        ToolUseBlock(
                            tool_call_id="c2",
                            tool_name="mock_tool",
                            tool_input={"input": "b"},
                        ),
                    ],
                    stop_reason=FinishReason.TOOL_USE,
                ),
                MockResponseConfig(
                    content=[TextBlock(text="All done")],
                ),
            ]
        )
        registry = ToolRegistry()
        registry.register(tool)

        factory = AgentFactory(provider=provider, tool_registry=registry)
        agent = factory.create_parent_agent()

        result = await agent.run("Do two things")
        assert result.termination_reason == TerminationReason.END_TURN
        assert tool.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_max_tokens_does_not_spin_parent_agent(self):
        """Gateway empty response with MAX_TOKENS must fail fast, not exhaust max_loops."""
        provider = MockLLMProvider()
        provider.set_responses(
            [
                MockResponseConfig(
                    content=[],
                    stop_reason=FinishReason.MAX_TOKENS,
                    usage={"input_tokens": 0, "output_tokens": 0},
                )
            ]
            * 5
        )
        factory = AgentFactory(
            provider=provider,
            config=AgentConfig(max_loops=10),
            tool_registry=ToolRegistry(),
        )
        agent = factory.create_parent_agent(system_prompt="You are helpful.")

        result = await agent.run("Large input")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 1
        assert provider.call_count == 1
        assert result.error_detail
        assert "max_tokens" in result.error_detail[0].message


def tool_response(**arguments):
    return MockResponseConfig(
        content=[ToolUseBlock("sample-call", "mock_tool", arguments)],
        stop_reason=FinishReason.TOOL_USE,
    )


async def test_failed_followup_has_no_previous_answer():
    provider = MockLLMProvider()
    provider.set_responses(
        [
            MockResponseConfig(content=[TextBlock("Previous answer")]),
            MockResponseConfig(content=[], error=LLMError("Temporary failure")),
        ]
    )
    agent = AgentFactory(provider).create_parent_agent()
    assert (await agent.run("First request")).text == "Previous answer"

    result = await agent.run("Follow-up request")

    assert result.termination_reason is TerminationReason.ERROR
    assert result.text == ""


async def test_result_joins_text_blocks_in_order():
    provider = MockLLMProvider()
    provider.set_response([TextBlock("one"), TextBlock(" two")])
    result = await AgentFactory(provider).create_parent_agent().run("List two items")
    assert result.text == "one two"


async def test_deadline_interrupts_in_flight_provider():
    class SlowProvider(MockLLMProvider):
        async def create(self, *args, **kwargs):
            await asyncio.sleep(0.05)
            return await super().create(*args, **kwargs)

    agent = AgentFactory(
        SlowProvider(), AgentConfig(total_timeout=0.01)
    ).create_parent_agent()

    assert (
        await agent.run("Answer slowly")
    ).termination_reason is TerminationReason.TIMEOUT


async def test_anthropic_usage_is_counted_per_turn():
    provider = MockLLMProvider()
    provider.set_responses(
        [
            MockResponseConfig(
                content=[TextBlock("first")],
                usage={"input_tokens": 12, "output_tokens": 5},
            ),
            MockResponseConfig(
                content=[TextBlock("second")],
                usage={"input_tokens": 2, "output_tokens": 3},
            ),
        ]
    )
    agent = AgentFactory(provider).create_parent_agent()
    assert (await agent.run("First")).tokens_used == 17
    assert (await agent.run("Second")).tokens_used == 5


async def test_invalid_tool_argument_type_does_not_call_callback():
    provider = MockLLMProvider()
    provider.set_responses([tool_response(input=123)])
    registry = ToolRegistry()
    tool = MockTool()
    registry.register(tool)
    agent = AgentFactory(provider, tool_registry=registry).create_parent_agent()

    await agent.run("Use the sample tool")

    assert tool.call_count == 0
    assert agent.context.tool_result_content("sample-call")


async def test_tool_approval_honors_returned_context():
    class DenyHook(Hook):
        events = [HookEvent.BEFORE_TOOL_EXEC]

        async def handle(self, ctx):
            return HookContext(ctx.event, data=ctx.data, blocked=True)

    provider = MockLLMProvider()
    provider.set_responses([tool_response(input="sample")])
    registry = ToolRegistry()
    tool = MockTool()
    registry.register(tool)
    hooks = HookExecutor()
    hooks.register(DenyHook())
    agent = AgentFactory(
        provider, tool_registry=registry, hook_executor=hooks
    ).create_parent_agent()

    await agent.run("Use the sample tool")

    assert tool.call_count == 0
    assert "blocked" in agent.context.tool_result_content("sample-call")


async def test_tool_run_is_written_to_session_history():
    provider = MockLLMProvider()
    provider.set_responses([tool_response(input="sample")])
    registry = ToolRegistry()
    history = ToolHistoryStore("test-session")
    registry.context.set("history_store", history)
    registry.register(MockTool(response="recorded result"))
    agent = AgentFactory(provider, tool_registry=registry).create_parent_agent()

    await agent.run("Use the sample tool")

    result = next(
        msg.content[0]
        for msg in agent.context.messages
        if getattr(msg.content[0], "tool_call_id", None) == "sample-call"
        and hasattr(msg.content[0], "history_key")
    )
    assert result.history_key
    assert (await history.get(result.history_key)).content == "recorded result"


def test_factory_accepts_public_environment_config():
    config = EnvironmentAgentConfig(MAX_LOOPS=7, MAX_TOKENS=128)
    child = AgentFactory(MockLLMProvider(), config).create_sub_agent()

    assert child.config.max_loops == 7
    assert child.config.max_tokens == 128


async def test_delegated_error_is_a_failed_tool_result():
    provider = MockLLMProvider()
    provider.set_responses(
        [
            MockResponseConfig(
                content=[
                    ToolUseBlock("delegate", "task", {"goal": "Write a greeting"})
                ],
                stop_reason=FinishReason.TOOL_USE,
            ),
            MockResponseConfig(content=[], error=LLMError("Provider unavailable")),
        ]
    )
    registry = ToolRegistry()
    registry.register(TaskTool())
    agent = AgentFactory(provider, tool_registry=registry).create_parent_agent()

    await agent.run("Delegate the greeting")

    result = agent.context.messages[-2].content[0]
    assert result.tool_call_id == "delegate"
    assert result.is_error is True


async def test_parent_counts_child_usage_once():
    provider = MockLLMProvider()
    provider.set_responses(
        [
            MockResponseConfig(
                content=[
                    ToolUseBlock("delegate", "task", {"goal": "Write a greeting"})
                ],
                stop_reason=FinishReason.TOOL_USE,
                usage={"total_tokens": 3},
            ),
            MockResponseConfig(content=[TextBlock("Hello")], usage={"total_tokens": 5}),
            MockResponseConfig(content=[TextBlock("Done")], usage={"total_tokens": 7}),
        ]
    )
    registry = ToolRegistry()
    registry.register(TaskTool())
    agent = AgentFactory(provider, tool_registry=registry).create_parent_agent()

    assert (await agent.run("Delegate the greeting")).tokens_used == 15


async def test_cancelled_tool_batch_keeps_each_result_paired():
    entered = asyncio.Event()

    class WaitingTool(MockTool):
        name = "waiting_tool"

        async def execute(self, **params):
            entered.set()
            await asyncio.Event().wait()

    provider = MockLLMProvider()
    provider.set_response(
        [
            ToolUseBlock("completed-call", "mock_tool", {}),
            ToolUseBlock("waiting-call", "waiting_tool", {}),
        ],
        stop_reason=FinishReason.TOOL_USE,
    )
    registry = ToolRegistry()
    registry.register(MockTool(response="completed"))
    registry.register(WaitingTool())
    agent = AgentFactory(provider, tool_registry=registry).create_parent_agent()
    running = asyncio.create_task(agent.run("Process a sample"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert agent.context.tool_result_content("completed-call") == "completed"
    assert agent.context.tool_result_content("waiting-call") == "Tool call interrupted"
    assert (
        await agent.run("A new request")
    ).termination_reason is TerminationReason.END_TURN


async def test_strict_budget_caps_provider_output_after_counting_input():
    class MeteredProvider(MockLLMProvider):
        async def count_input_tokens(self, messages, tools):
            return 6

        async def create(self, messages, tools=None, **kwargs):
            self.output_limit = kwargs["max_tokens"]
            self.set_response([TextBlock("done")], usage={"total_tokens": 7})
            return await super().create(messages, tools, **kwargs)

    provider = MeteredProvider()
    config = AgentConfig(max_tokens=20, token_budget=10, strict_token_budget=True)
    result = (
        await AgentFactory(provider, config).create_parent_agent().run("Short input")
    )
    assert result.tokens_used == 7
    assert provider.output_limit == 4


async def test_permission_revoked_in_hook_blocks_execution():
    registry = ToolRegistry()
    tool = MockTool()
    registry.register(tool)
    registry.context.set("allowed_tools", {"mock_tool"})

    class RevokingHook(Hook):
        events = [HookEvent.BEFORE_TOOL_EXEC]

        async def handle(self, ctx):
            registry.context.set("allowed_tools", set())
            return ctx

    hooks = HookExecutor()
    hooks.register(RevokingHook())
    provider = MockLLMProvider()
    provider.set_responses([tool_response(input="sample")])
    agent = AgentFactory(
        provider, tool_registry=registry, hook_executor=hooks
    ).create_parent_agent()

    await agent.run("Use the sample tool")

    assert tool.call_count == 0
    assert "authorization changed" in (
        agent.context.tool_result_content("sample-call") or ""
    )


async def test_delegated_budget_exhaustion_stops_parent():
    provider = MockLLMProvider()
    provider.set_responses(
        [
            MockResponseConfig(
                content=[ToolUseBlock("delegate", "task", {"goal": "Check"})],
                stop_reason=FinishReason.TOOL_USE,
                usage={"total_tokens": 6},
            ),
            MockResponseConfig(content=[TextBlock("Child")], usage={"total_tokens": 6}),
        ]
    )
    registry = ToolRegistry()
    registry.register(TaskTool())
    agent = AgentFactory(
        provider, AgentConfig(max_loops=1, token_budget=10), tool_registry=registry
    ).create_parent_agent()

    result = await agent.run("Delegate")

    assert result.termination_reason is TerminationReason.TOKEN_BUDGET
    assert result.tokens_used == 12
