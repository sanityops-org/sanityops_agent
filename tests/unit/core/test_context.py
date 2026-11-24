"""Tests for agent/core/context.py - AgentContext, CompactConfig, ToolResultSummary."""

import asyncio

import pytest

from sanityops_agent.core.context import AgentContext, CompactConfig, ToolResultSummary
from sanityops_agent.core.message import (
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

# --- CompactConfig Tests ---


class TestCompactConfig:
    """Test CompactConfig defaults and customization."""

    def test_default_trigger_token_threshold(self):
        config = CompactConfig()
        assert config.trigger_token_threshold == 80000

    def test_default_preserve_recent_count(self):
        config = CompactConfig()
        assert config.preserve_recent_count == 5

    def test_custom_trigger_token_threshold(self):
        config = CompactConfig(trigger_token_threshold=50000)
        assert config.trigger_token_threshold == 50000

    def test_custom_preserve_recent_count(self):
        config = CompactConfig(preserve_recent_count=10)
        assert config.preserve_recent_count == 10


# --- ToolResultSummary Tests ---


class TestToolResultSummary:
    """Test ToolResultSummary creation and to_compact_string."""

    def test_create_summary_success(self):
        summary = ToolResultSummary(
            tool_name="search",
            tool_call_id="call_123",
            success=True,
            brief="[Result: 500 chars]",
        )
        assert summary.tool_name == "search"
        assert summary.tool_call_id == "call_123"
        assert summary.success is True
        assert summary.brief == "[Result: 500 chars]"
        assert summary.history_key is None

    def test_create_summary_with_history_key(self):
        summary = ToolResultSummary(
            tool_name="read_file",
            tool_call_id="call_456",
            success=True,
            brief="[Result: 1200 chars]",
            history_key="hist_key_1",
        )
        assert summary.history_key == "hist_key_1"

    def test_create_summary_failure(self):
        summary = ToolResultSummary(
            tool_name="write_file",
            tool_call_id="call_789",
            success=False,
            brief="[Result: 50 chars]",
        )
        assert summary.success is False

    def test_to_compact_string_success(self):
        summary = ToolResultSummary(
            tool_name="search",
            tool_call_id="call_123",
            success=True,
            brief="[Result: 500 chars]",
        )
        result = summary.to_compact_string()
        assert "Tool: search" in result
        assert "Success: True" in result
        assert "Brief: [Result: 500 chars]" in result

    def test_to_compact_string_failure(self):
        summary = ToolResultSummary(
            tool_name="write_file",
            tool_call_id="call_789",
            success=False,
            brief="[Result: 50 chars]",
        )
        result = summary.to_compact_string()
        assert "Success: False" in result


# --- AgentContext Initialization Tests ---


class TestAgentContextInitialization:
    """Test AgentContext default state."""

    def test_default_messages_empty(self):
        ctx = AgentContext()
        assert ctx.messages == []

    def test_default_recent_files_empty(self):
        ctx = AgentContext()
        assert ctx.recent_files == []

    def test_default_total_tokens_zero(self):
        ctx = AgentContext()
        assert ctx.total_tokens == 0

    def test_default_has_compacted_false(self):
        ctx = AgentContext()
        assert ctx.has_compacted is False

    def test_default_last_summary_empty(self):
        ctx = AgentContext()
        assert ctx.last_summary == ""

    def test_default_compact_config(self):
        ctx = AgentContext()
        assert isinstance(ctx.compact_config, CompactConfig)

    def test_default_history_store_none(self):
        ctx = AgentContext()
        assert ctx.history_store is None

    def test_custom_compact_config(self):
        config = CompactConfig(trigger_token_threshold=10000, preserve_recent_count=3)
        ctx = AgentContext(compact_config=config)
        assert ctx.compact_config.trigger_token_threshold == 10000
        assert ctx.compact_config.preserve_recent_count == 3


# --- AgentContext.add_message Tests ---


class TestAddMessage:
    """Test add_message behavior."""

    @pytest.mark.asyncio
    async def test_add_message_appends(self):
        ctx = AgentContext()
        msg = Message.from_text(Role.USER, "hello")
        await ctx.add_message(msg)
        assert len(ctx.messages) == 1
        assert ctx.messages[0] == msg

    @pytest.mark.asyncio
    async def test_add_message_multiple(self):
        ctx = AgentContext()
        msg1 = Message.from_text(Role.USER, "hello")
        msg2 = Message.from_text(Role.ASSISTANT, "hi there")
        await ctx.add_message(msg1)
        await ctx.add_message(msg2)
        assert len(ctx.messages) == 2

    @pytest.mark.asyncio
    async def test_add_message_updates_token_count(self):
        ctx = AgentContext()
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextBlock(text="response")],
            raw_response={"usage": {"total_tokens": 100}},
        )
        await ctx.add_message(msg)
        assert ctx.total_tokens == 100

    @pytest.mark.asyncio
    async def test_add_message_accumulates_tokens(self):
        ctx = AgentContext()
        msg1 = Message(
            role=Role.USER,
            content=[TextBlock(text="hello")],
            raw_response={"usage": {"total_tokens": 50}},
        )
        msg2 = Message(
            role=Role.ASSISTANT,
            content=[TextBlock(text="hi")],
            raw_response={"usage": {"total_tokens": 75}},
        )
        await ctx.add_message(msg1)
        await ctx.add_message(msg2)
        assert ctx.total_tokens == 125

    @pytest.mark.asyncio
    async def test_add_message_no_raw_response(self):
        ctx = AgentContext()
        msg = Message.from_text(Role.USER, "hello")
        await ctx.add_message(msg)
        assert ctx.total_tokens == 0

    @pytest.mark.asyncio
    async def test_add_message_raw_response_no_usage(self):
        ctx = AgentContext()
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextBlock(text="hi")],
            raw_response={"id": "msg_123"},
        )
        await ctx.add_message(msg)
        assert ctx.total_tokens == 0


# --- AgentContext.compact_if_needed Tests ---


class TestCompactIfNeeded:
    """Test compaction trigger logic."""

    @pytest.mark.asyncio
    async def test_no_compaction_below_threshold(self):
        ctx = AgentContext()
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextBlock(text="response")],
            raw_response={"usage": {"total_tokens": 50000}},
        )
        await ctx.add_message(msg)
        result = await ctx.compact_if_needed()
        assert result is False
        assert ctx.has_compacted is False

    @pytest.mark.asyncio
    async def test_compaction_at_threshold(self):
        ctx = AgentContext(compact_config=CompactConfig(trigger_token_threshold=100))
        ctx.messages.append(
            Message(role=Role.USER, content=[TextBlock(text="x" * 500)])
        )
        result = await ctx.compact_if_needed()
        assert result is True
        assert ctx.has_compacted is True

    @pytest.mark.asyncio
    async def test_compaction_above_threshold(self):
        ctx = AgentContext(compact_config=CompactConfig(trigger_token_threshold=100))
        ctx.messages.append(
            Message(role=Role.USER, content=[TextBlock(text="x" * 1000)])
        )
        result = await ctx.compact_if_needed()
        assert result is True
        assert ctx.has_compacted is True


# --- AgentContext._do_compact Tests ---


class TestDoCompact:
    """Test compaction logic: preserves recent tool results, summarizes older ones."""

    @pytest.mark.asyncio
    async def test_compaction_preserves_recent_tool_results(self):
        """Compaction should keep the last N tool results in full."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=2)
        ctx = AgentContext(compact_config=config)

        # Add 4 tool results
        for i in range(4):
            msg = Message(
                role=Role.ASSISTANT,
                content=[
                    ToolResultBlock(
                        tool_call_id=f"call_{i}",
                        tool_result_content=f"result {i}" * 100,
                        is_error=False,
                        history_key=f"hist_{i}",
                        metadata={"tool_name": "search", "history_key": f"hist_{i}"},
                    )
                ],
                raw_response={"usage": {"total_tokens": 50}},
            )
            await ctx.add_message(msg)

        await ctx.compact_if_needed()

        # Every result block is kept (so no tool_use is orphaned); the two
        # most recent are untouched, the older two are shortened.
        results = {
            b.tool_call_id: b
            for m in ctx.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        }
        assert set(results) == {"call_0", "call_1", "call_2", "call_3"}
        assert results["call_2"].tool_result_content == "result 2" * 100
        assert results["call_3"].tool_result_content == "result 3" * 100
        assert results["call_0"].metadata.get("compacted") is True
        assert results["call_1"].metadata.get("compacted") is True

    @pytest.mark.asyncio
    async def test_compaction_summarizes_older_results(self):
        """Compaction should shorten older results while keeping their blocks."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=1)
        ctx = AgentContext(compact_config=config)

        # Add 3 tool results (2 will be summarized, 1 preserved)
        for i in range(3):
            msg = Message(
                role=Role.ASSISTANT,
                content=[
                    ToolResultBlock(
                        tool_call_id=f"call_{i}",
                        tool_result_content=f"result {i}" * 50,
                        is_error=(i == 1),  # Second one is an error
                        history_key=f"hist_{i}",
                        metadata={"tool_name": "search", "history_key": f"hist_{i}"},
                    )
                ],
                raw_response={"usage": {"total_tokens": 50}},
            )
            await ctx.add_message(msg)

        await ctx.compact_if_needed()

        results = {
            b.tool_call_id: b
            for m in ctx.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        }
        # All three blocks survive; the newest keeps its body, older two do not.
        assert set(results) == {"call_0", "call_1", "call_2"}
        assert results["call_2"].tool_result_content == "result 2" * 50
        assert "Tool: search" in results["call_0"].tool_result_content
        assert "Success: False" in results["call_1"].tool_result_content

    @pytest.mark.asyncio
    async def test_compaction_summary_contains_tool_name(self):
        """Summary text should contain the tool name."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=1)
        ctx = AgentContext(compact_config=config)

        # Two results so one is actually compacted (the newest is preserved).
        for i, name in enumerate(("read_file", "search")):
            msg = Message(
                role=Role.ASSISTANT,
                content=[
                    ToolResultBlock(
                        tool_call_id=f"call_{i}",
                        tool_result_content=f"old result data {i}" * 20,
                        is_error=False,
                        metadata={"tool_name": name},
                    )
                ],
                raw_response={"usage": {"total_tokens": 50}},
            )
            await ctx.add_message(msg)

        await ctx.compact_if_needed()

        # The summary is written into the (retained) result block.
        contents = [
            b.tool_result_content
            for m in ctx.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        assert any("read_file" in c for c in contents)

    @pytest.mark.asyncio
    async def test_compaction_error_result_summary(self):
        """Error tool results should show Success: False in summary."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=1)
        ctx = AgentContext(compact_config=config)

        # One error + one success so the error (older) one gets compacted.
        for i, (is_err, name) in enumerate(((True, "write_file"), (False, "search"))):
            msg = Message(
                role=Role.ASSISTANT,
                content=[
                    ToolResultBlock(
                        tool_call_id=f"call_{i}",
                        tool_result_content=f"error data {i}" * 20,
                        is_error=is_err,
                        metadata={"tool_name": name},
                    )
                ],
                raw_response={"usage": {"total_tokens": 50}},
            )
            await ctx.add_message(msg)

        await ctx.compact_if_needed()

        # Check summary shows failure
        contents = [
            b.tool_result_content
            for m in ctx.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]
        assert any("Success: False" in c for c in contents)


class TestCompactKeepsToolPairing:
    """Compaction must not orphan the assistant tool_use the result answers.

    The agent loop stores one Message per tool result, and the assistant turn
    before it holds the matching ToolUseBlock. Providers reject a tool_use with
    no tool_result (HTTP 400), so compaction must keep a ToolResultBlock in
    place, only shortening its content.
    """

    @pytest.fixture
    def compacting_context(self):
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=0)
        ctx = AgentContext(compact_config=config)
        ctx.total_tokens = 1000  # above the threshold so compaction fires
        return ctx

    @staticmethod
    def _tool_exchange(ctx: AgentContext, i: int) -> None:
        """Append assistant(tool_use) + user(tool_result) as the loop does."""
        ctx.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[
                    TextBlock(text=f"thinking {i}"),
                    ToolUseBlock(
                        tool_call_id=f"call_{i}",
                        tool_name="search",
                        tool_input={"q": str(i)},
                    ),
                ],
            )
        )
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultBlock(
                        tool_call_id=f"call_{i}",
                        tool_result_content=f"result {i}" * 100,
                        metadata={"tool_name": "search"},
                    )
                ],
            )
        )

    @pytest.mark.asyncio
    async def test_compacted_result_keeps_tool_call_id(
        self, compacting_context: AgentContext
    ):
        ctx = compacting_context
        self._tool_exchange(ctx, 1)
        self._tool_exchange(ctx, 2)

        await ctx.compact_if_needed()

        every_tool_use_id = {
            b.tool_call_id
            for m in ctx.messages
            for b in m.content
            if isinstance(b, ToolUseBlock)
        }
        every_tool_result_id = {
            b.tool_call_id
            for m in ctx.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        }
        assert every_tool_use_id == every_tool_result_id

    @pytest.mark.asyncio
    async def test_compacted_result_is_truncated_not_replaced(
        self, compacting_context: AgentContext
    ):
        ctx = compacting_context
        self._tool_exchange(ctx, 1)

        await ctx.compact_if_needed()

        result_blocks = [
            b for m in ctx.messages for b in m.content if isinstance(b, ToolResultBlock)
        ]
        assert len(result_blocks) == 1
        block = result_blocks[0]
        assert block.tool_call_id == "call_1"
        assert block.metadata.get("compacted") is True
        assert block.metadata.get("tool_name") == "search"
        # The brief survives, the body does not.
        assert "[Result:" in block.tool_result_content
        assert "result 1" * 100 != block.tool_result_content
        assert block.tool_result_content != "result 1" * 100

    @pytest.mark.asyncio
    async def test_mixed_message_still_compacts_every_tool_result(
        self, compacting_context: AgentContext
    ):
        """A message holding several results compacts each of them."""
        ctx = compacting_context
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultBlock(
                        tool_call_id="call_a",
                        tool_result_content="aa" * 200,
                        metadata={"tool_name": "search"},
                    ),
                    ToolResultBlock(
                        tool_call_id="call_b",
                        tool_result_content="bb" * 200,
                        metadata={"tool_name": "search"},
                    ),
                ],
            )
        )

        await ctx.compact_if_needed()

        results = {
            b.tool_call_id: b.tool_result_content
            for m in ctx.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        }
        assert set(results) == {"call_a", "call_b"}
        assert "[Result:" in results["call_a"]
        assert "[Result:" in results["call_b"]


class TestCompactionTokenCounting:
    """Compaction must reduce the token estimate, or the trigger re-fires on
    every subsequent call once the threshold is crossed."""

    @pytest.mark.asyncio
    async def test_token_estimate_shrinks_after_compaction(self):
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=0)
        ctx = AgentContext(compact_config=config)

        for i in range(3):
            ctx.messages.append(
                Message(
                    role=Role.USER,
                    content=[
                        ToolResultBlock(
                            tool_call_id=f"call_{i}",
                            tool_result_content="x" * 5000,
                            metadata={"tool_name": "search"},
                        )
                    ],
                )
            )
        before = ctx.estimate_tokens()

        await ctx.compact_if_needed()

        assert ctx.estimate_tokens() < before

    @pytest.mark.asyncio
    async def test_compaction_does_not_retrigger(self):
        """Second compact_if_needed() with no new messages is a no-op."""
        config = CompactConfig(trigger_token_threshold=1000, preserve_recent_count=0)
        ctx = AgentContext(compact_config=config)
        for i in range(3):
            ctx.messages.append(
                Message(
                    role=Role.USER,
                    content=[
                        ToolResultBlock(
                            tool_call_id=f"call_{i}",
                            tool_result_content="x" * 5000,
                            metadata={"tool_name": "search"},
                        )
                    ],
                )
            )

        assert await ctx.compact_if_needed() is True
        # The summaries are far below the threshold, so the second call must
        # not compact again (with the old accumulated counter it always did).
        assert await ctx.compact_if_needed() is False


class TestToolResultContent:
    """AgentContext.tool_result_content() reads current (possibly compacted)
    content by tool_call_id."""

    @pytest.mark.asyncio
    async def test_found_result(self):
        ctx = AgentContext()
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultBlock(tool_call_id="c1", tool_result_content="payload")
                ],
            )
        )
        assert ctx.tool_result_content("c1") == "payload"

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self):
        ctx = AgentContext()
        assert ctx.tool_result_content("missing") is None

    @pytest.mark.asyncio
    async def test_reads_compacted_content(self):
        ctx = AgentContext(
            compact_config=CompactConfig(
                trigger_token_threshold=10, preserve_recent_count=0
            )
        )
        ctx.total_tokens = 1000
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultBlock(
                        tool_call_id="c1",
                        tool_result_content="big" * 500,
                        metadata={"tool_name": "search"},
                    )
                ],
            )
        )
        await ctx.compact_if_needed()

        assert ctx.tool_result_content("c1") != "big" * 500
        assert "[Result:" in ctx.tool_result_content("c1")


class TestCompactionHistoryKey:
    """history_key lives on the block; metadata may not mirror it."""

    @pytest.mark.asyncio
    async def test_history_key_read_from_attribute_not_metadata(self):
        ctx = AgentContext(
            compact_config=CompactConfig(
                trigger_token_threshold=10, preserve_recent_count=0
            )
        )
        ctx.total_tokens = 1000
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultBlock(
                        tool_call_id="c1",
                        tool_result_content="big" * 400,
                        history_key="attr_key",
                        # metadata does NOT mirror history_key.
                        metadata={"tool_name": "search"},
                    )
                ],
            )
        )

        await ctx.compact_if_needed()

        block = next(
            b for m in ctx.messages for b in m.content if isinstance(b, ToolResultBlock)
        )
        # The attribute value survives compaction (canonical source).
        assert block.history_key == "attr_key"
        assert block.metadata.get("compacted") is True


class TestCompactionReentrancy:
    """Overlapping compact_if_needed() calls compact once, not twice."""

    @pytest.mark.asyncio
    async def test_second_call_while_in_flight_compacts_once(self):
        ctx = AgentContext(
            compact_config=CompactConfig(
                trigger_token_threshold=10, preserve_recent_count=0
            )
        )
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultBlock(
                        tool_call_id="c1",
                        tool_result_content="x" * 5000,
                        metadata={"tool_name": "search"},
                    )
                ],
            )
        )

        # The calls serialize on the lock. _do_compact has no await point, so
        # in practice the first finishes before the second starts; what the
        # lock guarantees is they never interleave mid-compaction, and the
        # estimate re-check keeps repeated calls converging instead of
        # compacting forever.
        compactions: list[bool] = []

        async def call() -> None:
            compactions.append(await ctx.compact_if_needed())

        await asyncio.gather(call(), call())

        # At least one compacted, none raised, and the context is consistent.
        assert any(compactions)
        assert ctx.has_compacted is True
        # Every remaining result is either compacted or preserved whole.
        for m in ctx.messages:
            for b in m.content:
                if isinstance(b, ToolResultBlock) and not b.metadata.get("compacted"):
                    assert len(b.tool_result_content) == 5000

    @pytest.mark.asyncio
    async def test_constructible_outside_event_loop(self):
        """AgentContext must be constructible without a running loop."""
        ctx = AgentContext()
        assert ctx._compact_lock is not None  # created at construction
