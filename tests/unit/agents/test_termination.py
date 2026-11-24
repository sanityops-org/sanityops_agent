"""Unit tests for BaseAgent, AgentConfig, AgentResult, AgentError, TerminationReason.

Following TDD: tests written first, verified failing, then implementation added.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from sanityops_agent.core.agent import (
    AgentConfig,
    AgentError,
    AgentResult,
    BaseAgent,
    TerminationReason,
)
from sanityops_agent.core.message import (
    FinishReason,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
)

# ---------------------------------------------------------------------------
# Concrete test subclass of BaseAgent
# ---------------------------------------------------------------------------


class DummyResponse:
    """Mock LLM response for testing."""

    def __init__(
        self,
        stop_reason: FinishReason = FinishReason.END_TURN,
        has_tool_calls: bool = False,
        tool_calls: list | None = None,
        text: str = "hello",
        content: list | None = None,
    ):
        self.stop_reason = stop_reason
        self.has_tool_calls = has_tool_calls
        self._tool_calls = tool_calls or []
        self._text = text
        self.content = content if content is not None else [TextBlock(text=text)]

    def get_tool_calls(self):
        return self._tool_calls


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""

    def __init__(self, config: AgentConfig, **kwargs):
        super().__init__(config)
        self._mock_response: DummyResponse | None = kwargs.pop("mock_response", None)
        self._mock_exception: Exception | None = kwargs.pop("mock_exception", None)
        self._call_count: int = 0
        self._tool_call_log: list = []

    async def _call_llm(self):
        self._call_count += 1
        if self._mock_exception:
            raise self._mock_exception
        return self._mock_response or DummyResponse()

    async def _execute_tools(self, tool_calls: list) -> None:
        self._tool_call_log.extend(tool_calls)


# ===========================================================================
# TerminationReason enum tests
# ===========================================================================


class TestTerminationReason:
    def test_all_values_exist(self):
        assert TerminationReason.END_TURN.value == "end_turn"
        assert TerminationReason.MAX_LOOPS.value == "max_loops"
        assert TerminationReason.TIMEOUT.value == "timeout"
        assert TerminationReason.TOKEN_BUDGET.value == "token_budget"
        assert TerminationReason.ERROR.value == "error"
        assert TerminationReason.HOOK_BLOCKED.value == "hook_blocked"

    def test_member_count(self):
        assert len(TerminationReason) == 6


# ===========================================================================
# AgentConfig tests
# ===========================================================================


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_loops == 30
        assert cfg.max_tokens is None
        assert cfg.system_prompt == ""
        assert cfg.total_timeout == 300
        assert cfg.token_budget == 100000
        assert cfg.max_empty_responses == 3
        assert cfg.allowed_tool_tags == []
        assert cfg.max_retries == 3

    def test_custom_values(self):
        cfg = AgentConfig(
            max_loops=10,
            max_tokens=4000,
            system_prompt="You are helpful",
            total_timeout=60,
            token_budget=50000,
            max_empty_responses=2,
            allowed_tool_tags=["read", "write"],
            max_retries=5,
        )
        assert cfg.max_loops == 10
        assert cfg.max_tokens == 4000
        assert cfg.system_prompt == "You are helpful"
        assert cfg.total_timeout == 60
        assert cfg.token_budget == 50000
        assert cfg.max_empty_responses == 2
        assert cfg.allowed_tool_tags == ["read", "write"]
        assert cfg.max_retries == 5


# ===========================================================================
# AgentError tests
# ===========================================================================


class TestAgentError:
    def test_basic_construction(self):
        err = AgentError(
            error_type="llm_error",
            message="Connection refused",
            loop_iteration=2,
        )
        assert err.error_type == "llm_error"
        assert err.message == "Connection refused"
        assert err.loop_iteration == 2
        assert err.tool_name is None
        assert err.recoverable is False

    def test_with_tool_name(self):
        err = AgentError(
            error_type="tool_error",
            message="File not found",
            loop_iteration=0,
            tool_name="read_file",
            recoverable=True,
        )
        assert err.tool_name == "read_file"
        assert err.recoverable is True


# ===========================================================================
# AgentResult tests
# ===========================================================================


class TestAgentResult:
    def test_construction(self):
        result = AgentResult(
            text="Hello world",
            termination_reason=TerminationReason.END_TURN,
            loops_used=3,
            tokens_used=1500,
            time_elapsed=2.5,
        )
        assert result.text == "Hello world"
        assert result.termination_reason == TerminationReason.END_TURN
        assert result.loops_used == 3
        assert result.tokens_used == 1500
        assert result.time_elapsed == 2.5
        assert result.error is None
        assert result.error_detail == []

    def test_with_error(self):
        detail = [AgentError("llm_error", "boom", 0)]
        result = AgentResult(
            text="",
            termination_reason=TerminationReason.ERROR,
            loops_used=1,
            tokens_used=0,
            time_elapsed=0.1,
            error="boom",
            error_detail=detail,
        )
        assert result.error == "boom"
        assert len(result.error_detail) == 1


# ===========================================================================
# BaseAgent _check_termination_conditions tests
# ===========================================================================


class TestCheckTerminationConditions:
    @pytest.mark.asyncio
    async def test_no_termination_when_within_limits(self):
        agent = ConcreteAgent(AgentConfig(total_timeout=300, token_budget=100000))
        agent.start_time = time.time()
        agent.tokens_used = 500
        reason = await agent._check_termination_conditions(0)
        assert reason is None

    @pytest.mark.asyncio
    async def test_timeout_termination(self):
        agent = ConcreteAgent(AgentConfig(total_timeout=1, token_budget=100000))
        # Simulate that 2 seconds have passed
        agent.start_time = time.time() - 2.0
        agent.tokens_used = 500
        reason = await agent._check_termination_conditions(0)
        assert reason == TerminationReason.TIMEOUT

    @pytest.mark.asyncio
    async def test_token_budget_termination(self):
        agent = ConcreteAgent(AgentConfig(total_timeout=300, token_budget=1000))
        agent.start_time = time.time()
        agent.tokens_used = 1500  # Over budget
        reason = await agent._check_termination_conditions(0)
        assert reason == TerminationReason.TOKEN_BUDGET


# ===========================================================================
# BaseAgent _build_result tests
# ===========================================================================


class TestBuildResult:
    @pytest.mark.asyncio
    async def test_build_result_extracts_text(self):
        agent = ConcreteAgent(AgentConfig())
        # Manually set up context with messages
        agent.context = MagicMock()
        agent.context.messages = [
            Message.from_text(Role.USER, "What is 2+2?"),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="2+2 is 4.")]),
            Message.from_text(Role.USER, "Thanks"),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="You're welcome!")]),
        ]
        agent.tokens_used = 100
        agent.start_time = time.time() - 1.0

        result = agent._build_result(TerminationReason.END_TURN, 1)

        # _build_result now only extracts text from the LAST assistant message
        assert "You're welcome!" in result.text
        assert result.termination_reason == TerminationReason.END_TURN
        assert result.loops_used == 2  # iteration + 1
        assert result.tokens_used == 100
        assert result.time_elapsed >= 1.0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_build_result_with_error(self):
        agent = ConcreteAgent(AgentConfig())
        agent.context = MagicMock()
        agent.context.messages = []
        agent.tokens_used = 0
        agent.start_time = time.time()

        result = agent._build_result(
            TerminationReason.ERROR, 0, error="Connection failed"
        )

        assert result.error == "Connection failed"
        assert result.termination_reason == TerminationReason.ERROR

    @pytest.mark.asyncio
    async def test_build_result_no_context(self):
        agent = ConcreteAgent(AgentConfig())
        agent.context = None
        agent.tokens_used = 0
        agent.start_time = time.time()

        result = agent._build_result(TerminationReason.END_TURN, 0)

        assert result.text == ""
        assert result.loops_used == 1


# ===========================================================================
# BaseAgent _record_error tests
# ===========================================================================


class TestRecordError:
    def test_record_single_error(self):
        agent = ConcreteAgent(AgentConfig())
        agent._record_error("llm_error", "timeout", 3)
        assert len(agent._errors) == 1
        assert agent._errors[0].error_type == "llm_error"
        assert agent._errors[0].message == "timeout"
        assert agent._errors[0].loop_iteration == 3
        assert agent._errors[0].recoverable is False

    def test_record_multiple_errors(self):
        agent = ConcreteAgent(AgentConfig())
        agent._record_error("llm_error", "err1", 0, recoverable=True)
        agent._record_error("tool_error", "err2", 1, tool_name="read_file")
        agent._record_error("validation_error", "err3", 2)
        assert len(agent._errors) == 3
        assert agent._errors[0].error_type == "llm_error"
        assert agent._errors[1].tool_name == "read_file"
        assert agent._errors[2].loop_iteration == 2

    def test_record_error_with_tool_name(self):
        agent = ConcreteAgent(AgentConfig())
        agent._record_error(
            "tool_error", "File missing", 0, tool_name="read_file", recoverable=True
        )
        err = agent._errors[0]
        assert err.tool_name == "read_file"
        assert err.recoverable is True

    def test_errors_included_in_result(self):
        agent = ConcreteAgent(AgentConfig())
        agent._record_error("llm_error", "boom", 0)
        agent.context = MagicMock()
        agent.context.messages = []
        agent.tokens_used = 0
        agent.start_time = time.time()

        result = agent._build_result(TerminationReason.ERROR, 0, error="boom")
        assert len(result.error_detail) == 1
        assert result.error_detail[0].message == "boom"


# ===========================================================================
# BaseAgent.run() loop termination tests
# ===========================================================================


class TestRunLoopTermination:
    """Test that the main loop terminates correctly under various response patterns."""

    @pytest.mark.asyncio
    async def test_text_only_with_end_turn_ends_loop(self):
        """Pure text response with END_TURN should end the loop immediately."""
        agent = ConcreteAgent(AgentConfig(max_loops=30))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.END_TURN,
            has_tool_calls=False,
            text="Hello!",
        )
        result = await agent.run("Say hello")
        assert result.termination_reason == TerminationReason.END_TURN
        assert result.loops_used == 1
        assert "Hello!" in result.text

    @pytest.mark.asyncio
    async def test_text_plus_tool_calls_does_not_end_loop(self):
        """Response with both text AND tool calls should NOT end the loop —
        tools must be executed first. This is the core bug fix: previously
        stop_reason=END_TURN would cause early return before tool execution."""
        tool_calls = [MagicMock()]
        agent = ConcreteAgent(AgentConfig(max_loops=30))
        # First response: text + tool calls + END_TURN
        # Second response: text only + END_TURN (to end the loop)
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.END_TURN,
            has_tool_calls=True,
            tool_calls=tool_calls,
            text="Let me check that for you.",
        )
        call_count = 0

        async def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DummyResponse(
                    stop_reason=FinishReason.END_TURN,
                    has_tool_calls=True,
                    tool_calls=tool_calls,
                    text="Let me check that for you.",
                )
            else:
                return DummyResponse(
                    stop_reason=FinishReason.END_TURN,
                    has_tool_calls=False,
                    text="Done!",
                )

        agent._call_llm = side_effect
        result = await agent.run("Do something")
        assert result.termination_reason == TerminationReason.END_TURN
        assert result.loops_used == 2
        assert call_count == 2
        # Verify tool was actually executed
        assert len(agent._tool_call_log) == 1

    @pytest.mark.asyncio
    async def test_empty_max_tokens_terminates_immediately(self):
        """Empty response with MAX_TOKENS must fail immediately without spinning."""
        agent = ConcreteAgent(AgentConfig(max_loops=20))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.MAX_TOKENS,
            has_tool_calls=False,
            content=[],
        )
        result = await agent.run("Analyze large context")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 1
        assert agent._call_count == 1
        assert result.error is not None
        assert "max_tokens" in result.error
        assert len(result.error_detail) == 1
        assert "max_tokens" in result.error_detail[0].message
        assert result.error_detail[0].recoverable is False
        assert agent.context is not None
        assert [m.role for m in agent.context.messages] == [Role.USER]

    @pytest.mark.asyncio
    async def test_thinking_only_max_tokens_terminates_without_polluting_context(self):
        """Thinking-only MAX_TOKENS response must fail immediately without saving Assistant."""
        agent = ConcreteAgent(AgentConfig(max_loops=10))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.MAX_TOKENS,
            has_tool_calls=False,
            content=[ThinkingBlock(thinking="internal reasoning")],
        )
        result = await agent.run("Think hard")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 1
        assert agent._call_count == 1
        assert agent.context is not None
        assert [m.role for m in agent.context.messages] == [Role.USER]

    @pytest.mark.asyncio
    async def test_whitespace_only_text_max_tokens_terminates_immediately(self):
        """Whitespace-only text with MAX_TOKENS is treated as empty and fails immediately."""
        agent = ConcreteAgent(AgentConfig(max_loops=10))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.MAX_TOKENS,
            has_tool_calls=False,
            content=[TextBlock(text="   \n")],
        )
        result = await agent.run("Test blank")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 1
        assert agent._call_count == 1
        assert agent.context is not None
        assert [m.role for m in agent.context.messages] == [Role.USER]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stop_reason",
        [FinishReason.CONTENT_FILTER, FinishReason.ERROR],
    )
    async def test_empty_abnormal_stop_reason_terminates_immediately(self, stop_reason):
        """Empty response with CONTENT_FILTER or ERROR fails immediately."""
        agent = ConcreteAgent(AgentConfig(max_loops=10))
        agent._mock_response = DummyResponse(
            stop_reason=stop_reason,
            has_tool_calls=False,
            content=[],
        )
        result = await agent.run("Trigger stop reason")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 1
        assert agent._call_count == 1
        assert stop_reason.value in (result.error or "")

    @pytest.mark.asyncio
    async def test_partial_text_with_max_tokens_terminates_with_error(self):
        """Non-empty MAX_TOKENS response records partial text and fails closed."""
        agent = ConcreteAgent(AgentConfig(max_loops=10))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.MAX_TOKENS,
            has_tool_calls=False,
            content=[TextBlock(text="partial output")],
        )
        result = await agent.run("Long answer")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 1
        assert agent._call_count == 1
        assert result.text == "partial output"
        assert "max_tokens" in (result.error or "")

    @pytest.mark.asyncio
    async def test_truncated_tool_call_with_max_tokens_not_executed(self):
        """Tool calls on a MAX_TOKENS response must not execute or corrupt context."""
        agent = ConcreteAgent(AgentConfig(max_loops=10))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.MAX_TOKENS,
            has_tool_calls=True,
            tool_calls=[MagicMock()],
            content=[MagicMock()],
        )
        result = await agent.run("Truncated tool call")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 1
        assert len(agent._tool_call_log) == 0
        assert agent.context is not None
        assert [m.role for m in agent.context.messages] == [Role.USER]

    @pytest.mark.asyncio
    async def test_consecutive_empty_end_turn_terminates_at_limit(self):
        """Repeated empty END_TURN/STOP responses fail after max_empty_responses."""
        agent = ConcreteAgent(AgentConfig(max_loops=20, max_empty_responses=3))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.END_TURN,
            has_tool_calls=False,
            content=[],
        )
        result = await agent.run("Say something")
        assert result.termination_reason == TerminationReason.ERROR
        assert result.loops_used == 3
        assert agent._call_count == 3
        assert len(result.error_detail) == 3
        assert result.error_detail[0].recoverable is True
        assert result.error_detail[1].recoverable is True
        assert result.error_detail[2].recoverable is False
        assert agent.context is not None
        assert [m.role for m in agent.context.messages] == [Role.USER]

    @pytest.mark.asyncio
    async def test_consecutive_empty_counter_resets_after_valid_response(self):
        """Non-empty response resets the consecutive empty counter."""
        agent = ConcreteAgent(AgentConfig(max_loops=10, max_empty_responses=2))
        responses = [
            DummyResponse(stop_reason=FinishReason.END_TURN, content=[]),
            DummyResponse(
                stop_reason=FinishReason.TOOL_USE,
                has_tool_calls=True,
                tool_calls=[MagicMock()],
                content=[MagicMock()],
            ),
            DummyResponse(stop_reason=FinishReason.STOP, content=[]),
            DummyResponse(stop_reason=FinishReason.END_TURN, text="Recovered!"),
        ]
        idx = 0

        async def side_effect():
            nonlocal idx
            resp = responses[idx]
            idx += 1
            return resp

        agent._call_llm = side_effect
        result = await agent.run("Recover across empty responses")
        assert result.termination_reason == TerminationReason.END_TURN
        assert result.loops_used == 4
        assert result.text == "Recovered!"
        assert len(agent._tool_call_log) == 1

    @pytest.mark.asyncio
    async def test_stop_sequence_without_tools_ends_turn(self):
        """STOP_SEQUENCE without tool calls should terminate cleanly as END_TURN."""
        agent = ConcreteAgent(AgentConfig(max_loops=5))
        agent._mock_response = DummyResponse(
            stop_reason=FinishReason.STOP_SEQUENCE,
            has_tool_calls=False,
            text="Stopped cleanly",
        )
        result = await agent.run("Stop on sequence")
        assert result.termination_reason == TerminationReason.END_TURN
        assert result.loops_used == 1
        assert result.text == "Stopped cleanly"
