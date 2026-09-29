"""Tests for agent/core/context.py - AgentContext, CompactConfig, ToolResultSummary."""

import pytest

from sanityops_agent.core.context import AgentContext, CompactConfig, ToolResultSummary
from sanityops_agent.core.message import Message, Role, TextBlock, ToolResultBlock

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
        # New behavior: compaction returns True when threshold exceeded,
        # even if no tool results to compact
        ctx = AgentContext(compact_config=CompactConfig(trigger_token_threshold=10))
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextBlock(text="x" * 100)],  # 100 chars -> ~25 tokens estimate
        )
        await ctx.add_message(msg)
        result = await ctx.compact_if_needed()
        assert result is True  # Threshold exceeded
        assert ctx.has_compacted is True

    @pytest.mark.asyncio
    async def test_compaction_above_threshold(self):
        # Threshold exceeded, but no tool results to compact
        ctx = AgentContext(compact_config=CompactConfig(trigger_token_threshold=10))
        msg = Message(
            role=Role.ASSISTANT,
            content=[TextBlock(text="x" * 500)],  # Enough chars to exceed threshold
        )
        await ctx.add_message(msg)
        result = await ctx.compact_if_needed()
        assert result is True  # Threshold exceeded
        assert ctx.has_compacted is True


    @pytest.mark.asyncio
    async def test_reentrant_compaction_does_not_deadlock(self):
        """A reentrant call must return False, not deadlock on the lock.

        asyncio.Lock is not reentrant, so the in-flight flag has to be checked
        before acquiring it. Without that ordering this test hangs (caught by
        the timeout) rather than failing an assertion.
        """
        import asyncio

        config = CompactConfig(trigger_token_threshold=1, preserve_recent_count=0)
        ctx = AgentContext(compact_config=config)
        ctx.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[
                    ToolResultBlock(tool_call_id="c1", tool_result_content="x" * 100)
                ],
            )
        )

        inner_results: list[bool] = []

        original_do_compact = ctx._do_compact

        async def patched_do_compact():
            # We are now inside the lock with the flag set; this must not hang.
            inner_results.append(await ctx.compact_if_needed())
            await original_do_compact()

        ctx._do_compact = patched_do_compact

        # Generous timeout: this test's job is to catch an INDEFINITE deadlock,
        # not to bound normal operation, so slow CI must not false-positive.
        outer = await asyncio.wait_for(ctx.compact_if_needed(), timeout=5.0)

        assert outer is True
        assert inner_results == [False]
        assert ctx._compaction_in_flight is False


# --- AgentContext._do_compact Tests ---


class TestDoCompact:
    """Test compaction logic: preserves recent tool results, summarizes older ones."""

    @pytest.mark.asyncio
    async def test_compaction_preserves_recent_tool_results(self):
        """Compaction should preserve the last N tool results (as ToolResultBlock with replaced content)."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=2)
        ctx = AgentContext(compact_config=config)

        # Add 4 tool results with enough content to trigger compaction
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
            )
            await ctx.add_message(msg)

        await ctx.compact_if_needed()

        # All 4 ToolResultBlocks remain (just content replaced for older ones)
        tool_results = []
        for msg in ctx.messages:
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    tool_results.append(block)

        assert len(tool_results) == 4
        # The older ones (call_0, call_1) should have compacted content
        # The recent ones (call_2, call_3) should have original content
        assert tool_results[0].tool_call_id == "call_0"
        assert tool_results[1].tool_call_id == "call_1"
        assert tool_results[2].tool_call_id == "call_2"
        assert tool_results[3].tool_call_id == "call_3"
        # Check metadata shows compacted status
        assert tool_results[0].metadata.get("compacted") is True
        assert tool_results[1].metadata.get("compacted") is True
        assert tool_results[2].metadata.get("compacted") is None  # preserved
        assert tool_results[3].metadata.get("compacted") is None  # preserved

    @pytest.mark.asyncio
    async def test_compaction_summarizes_older_results(self):
        """Compaction should replace content in older tool results, preserving tool_call_id."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=1)
        ctx = AgentContext(compact_config=config)

        # Add 3 tool results (2 will be compacted, 1 preserved)
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
            )
            await ctx.add_message(msg)

        await ctx.compact_if_needed()

        # All 3 remain as ToolResultBlocks (not TextBlock summaries)
        tool_blocks = []
        for msg in ctx.messages:
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    tool_blocks.append(block)

        assert len(tool_blocks) == 3
        # call_0 and call_1 should have compacted content (summary string)
        assert "Tool: search" in tool_blocks[0].tool_result_content
        assert "Tool: search" in tool_blocks[1].tool_result_content
        # call_2 should have original content
        assert tool_blocks[2].tool_result_content == "result 2" * 50

    @pytest.mark.asyncio
    async def test_compaction_summary_contains_tool_name(self):
        """Summary text should contain the tool name."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=0)
        ctx = AgentContext(compact_config=config)

        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolResultBlock(
                    tool_call_id="call_old",
                    tool_result_content="x" * 100,  # Enough chars to exceed threshold
                    is_error=False,
                    metadata={"tool_name": "read_file"},
                )
            ],
        )
        await ctx.add_message(msg)

        await ctx.compact_if_needed()

        # The ToolResultBlock content is replaced with summary
        for m in ctx.messages:
            for block in m.content:
                if isinstance(block, ToolResultBlock):
                    assert "read_file" in block.tool_result_content

    @pytest.mark.asyncio
    async def test_compaction_error_result_summary(self):
        """Error tool results should show Success: False in summary."""
        config = CompactConfig(trigger_token_threshold=10, preserve_recent_count=0)
        ctx = AgentContext(compact_config=config)

        msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolResultBlock(
                    tool_call_id="call_err",
                    tool_result_content="x" * 100,  # Enough chars to exceed threshold
                    is_error=True,
                    metadata={"tool_name": "write_file"},
                )
            ],
        )
        await ctx.add_message(msg)

        await ctx.compact_if_needed()

        # The ToolResultBlock content shows failure
        for m in ctx.messages:
            for block in m.content:
                if isinstance(block, ToolResultBlock):
                    assert "Success: False" in block.tool_result_content
