"""Tests for retry logic in the agent framework."""

import pytest

from sanityops_agent.core.agent import AgentConfig, BaseAgent, TerminationReason
from sanityops_agent.core.context import AgentContext
from sanityops_agent.core.message import FinishReason, TextBlock
from sanityops_agent.llm.base import LLMResponse
from sanityops_agent.llm.exceptions import (
    AuthenticationError,
    LLMError,
    RateLimitError,
)


class _TestAgent(BaseAgent):
    """Concrete agent implementation for testing retry logic."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.context = AgentContext()
        self._responses: list = []
        self._response_index: int = 0

    def set_responses(self, responses: list):
        self._responses = responses
        self._response_index = 0

    async def _call_llm(self):
        if self._response_index < len(self._responses):
            resp = self._responses[self._response_index]
            self._response_index += 1
            if isinstance(resp, Exception):
                raise resp
            return resp
        # After exhausting configured responses, keep returning end_turn with content
        return LLMResponse(
            content=[TextBlock(text="done")],
            stop_reason=FinishReason.END_TURN,
            model="mock",
            usage={"total_tokens": 10},
        )

    async def _execute_tools(self, tool_calls: list) -> None:
        pass


class TestAgentRetry:
    """Test agent-level retry control."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_transient_failure(self):
        """Succeeds after one rate limit error."""
        config = AgentConfig(max_retries=3, max_loops=5)
        agent = _TestAgent(config)

        # Response must have content to avoid continue in the agent loop
        success_resp = LLMResponse(
            content=[TextBlock(text="Success!")],
            stop_reason=FinishReason.END_TURN,
            model="mock",
            usage={"total_tokens": 10},
        )
        agent.set_responses(
            [
                RateLimitError("rate limited", retry_after=0.01),
                success_resp,
            ]
        )
        result = await agent.run("test prompt")
        # After rate limit error, retry succeeds with end_turn -> END_TURN
        assert result.termination_reason == TerminationReason.END_TURN

    @pytest.mark.asyncio
    async def test_retry_exhausted_on_persistent_failure(self):
        """Raises after max_retries attempts."""
        config = AgentConfig(max_retries=2, max_loops=5)
        agent = _TestAgent(config)

        agent.set_responses(
            [
                RateLimitError("rate limited", retry_after=0.01),
                RateLimitError("rate limited", retry_after=0.01),
            ]
        )
        result = await agent.run("test prompt")
        assert result.termination_reason == TerminationReason.ERROR

    @pytest.mark.asyncio
    async def test_auth_error_not_retried(self):
        """Non-recoverable errors fail immediately, without retrying."""
        config = AgentConfig(max_retries=2, max_loops=5)
        agent = _TestAgent(config)

        agent.set_responses(
            [
                AuthenticationError("bad auth"),
                AuthenticationError("bad auth"),
            ]
        )
        result = await agent.run("test prompt")
        assert result.termination_reason == TerminationReason.ERROR
        # Only the first call was attempted — no retry.
        assert agent._response_index == 1
        # The recorded error is marked non-recoverable.
        assert result.error_detail[0].recoverable is False

    @pytest.mark.asyncio
    async def test_generic_exception_not_retried(self):
        """A non-LLMError exception (no recoverable attr) is not retried."""
        config = AgentConfig(max_retries=2, max_loops=5)
        agent = _TestAgent(config)

        agent.set_responses(
            [
                ValueError("boom"),
                ValueError("boom"),
            ]
        )
        result = await agent.run("test prompt")
        assert result.termination_reason == TerminationReason.ERROR
        assert agent._response_index == 1

    def test_retry_after_is_honored(self):
        """retry_after is used directly, capped at MAX_RETRY_DELAY."""

        agent = _TestAgent(AgentConfig(max_retries=1))

        # Happy path: retry_after=5 -> 5s.
        err = RateLimitError("rate limited", retry_after=5)
        assert agent._compute_retry_delay(err, 0) == 5

        # Cap path: retry_after=120 -> capped at 60.
        err = RateLimitError("rate limited", retry_after=120)
        assert agent._compute_retry_delay(err, 0) == 60

    def test_retry_after_absents_falls_back_to_backoff(self):
        """Without retry_after, exponential backoff is used."""
        agent = _TestAgent(AgentConfig(max_retries=1))
        err = RateLimitError("rate limited")
        assert agent._compute_retry_delay(err, 0) == 1
        assert agent._compute_retry_delay(err, 2) == 4

    @pytest.mark.asyncio
    async def test_retry_loop_sleeps_retry_after(self, monkeypatch):
        """The retry loop sleeps the retry_after value, not 2**attempt."""
        from sanityops_agent.core import agent as agent_module

        config = AgentConfig(max_retries=3, max_loops=5)
        agent = _TestAgent(config)

        success_resp = LLMResponse(
            content=[TextBlock(text="Success!")],
            stop_reason=FinishReason.END_TURN,
            model="mock",
            usage={"total_tokens": 10},
        )
        agent.set_responses(
            [
                RateLimitError("rate limited", retry_after=7),
                success_resp,
            ]
        )

        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(agent_module.asyncio, "sleep", fake_sleep)
        await agent.run("test prompt")

        assert sleeps == [7], f"expected sleep(7), got {sleeps}"

    def test_llm_error_not_recoverable(self):
        """Base LLMError without status_code is not recoverable by default."""
        err = LLMError("generic")
        assert err.recoverable is False

    def test_rate_limit_error_is_recoverable(self):
        """RateLimitError is marked as recoverable."""
        err = RateLimitError("rate limited", retry_after=10)
        assert err.recoverable is True

    def test_auth_error_not_recoverable(self):
        """AuthenticationError is not recoverable."""
        err = AuthenticationError("bad key")
        assert err.recoverable is False
