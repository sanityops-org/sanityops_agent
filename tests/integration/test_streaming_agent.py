"""Streaming through the same runtime as the non-streaming agents."""

from examples.streaming_react_agent import StreamingReActAgent
from sanityops_agent.core.agent import AgentConfig, TerminationReason
from sanityops_agent.core.message import ContentType, FinishReason, StreamChunk
from sanityops_agent.hooks.base import Hook, HookContext, HookEvent, HookExecutor
from sanityops_agent.tools.registry import ToolRegistry
from tests.mocks.mock_llm import MockLLMProvider
from tests.mocks.mock_tools import MockTool


async def test_streamed_result_reports_provider_usage():
    class Provider(MockLLMProvider):
        async def stream(self, messages, tools=None, **kwargs):
            yield StreamChunk(ContentType.TEXT, delta="Complete")
            yield StreamChunk(ContentType.TEXT, finish_reason=FinishReason.STOP)
            yield StreamChunk(ContentType.TEXT, usage={"total_tokens": 21})

    agent = StreamingReActAgent(AgentConfig(), Provider(), ToolRegistry())
    result = await agent.run("Report completion")

    assert result.text == "Complete"
    assert result.tokens_used == 21


async def test_streaming_uses_explicit_empty_tool_scope():
    class Provider(MockLLMProvider):
        async def stream(self, messages, tools=None, **kwargs):
            self.recorded_calls.append((messages, tools))
            yield StreamChunk(
                ContentType.TOOL_USE,
                delta={"name": "mock_tool", "arguments": "{}"},
                tool_call_id="fixture-tool",
            )
            yield StreamChunk(ContentType.TEXT, finish_reason=FinishReason.TOOL_USE)

    provider = Provider()
    registry = ToolRegistry()
    tool = MockTool()
    registry.register(tool)
    config = AgentConfig(max_loops=1, allowed_tool_names=frozenset())
    agent = StreamingReActAgent(config, provider, registry)

    result = await agent.run("Process a sample")

    assert result.termination_reason is TerminationReason.MAX_LOOPS
    assert provider.recorded_calls[0][1] == []
    assert tool.call_count == 0


async def test_streaming_agent_fires_after_llm_hook_and_enforces_before_tool_block():
    seen_events: list[HookEvent] = []

    class AuditHook(Hook):
        events = [HookEvent.AFTER_LLM_CALL, HookEvent.BEFORE_TOOL_EXEC]

        async def handle(self, ctx):
            seen_events.append(ctx.event)
            if ctx.event is HookEvent.BEFORE_TOOL_EXEC:
                return HookContext(ctx.event, data=ctx.data, blocked=True)
            return ctx

    class Provider(MockLLMProvider):
        async def stream(self, messages, tools=None, **kwargs):
            yield StreamChunk(
                ContentType.TOOL_USE,
                delta={"name": "mock_tool", "arguments": '{"input": "x"}'},
                tool_call_id="call-1",
            )
            yield StreamChunk(ContentType.TEXT, finish_reason=FinishReason.TOOL_USE)

    provider = Provider()
    registry = ToolRegistry()
    tool = MockTool()
    registry.register(tool)
    hooks = HookExecutor()
    hooks.register(AuditHook())
    agent = StreamingReActAgent(
        AgentConfig(max_loops=1), provider, registry, hook_executor=hooks
    )

    await agent.run("Run tool")

    assert seen_events == [HookEvent.AFTER_LLM_CALL, HookEvent.BEFORE_TOOL_EXEC]
    assert tool.call_count == 0
    assert "blocked" in (agent.context.tool_result_content("call-1") or "")
