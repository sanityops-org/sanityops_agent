from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from sanityops_agent.core.context import AgentContext, CompactConfig
from sanityops_agent.core.message import (
    ContentBlockUnion,
    FinishReason,
    Message,
    RedactedThinkingBlock,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
)
from sanityops_agent.llm.base import normalize_usage

MAX_RETRY_DELAY = 60.0
DEFAULT_MAX_EMPTY_RESPONSES = 3

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    END_TURN = "end_turn"
    MAX_LOOPS = "max_loops"
    TIMEOUT = "timeout"
    TOKEN_BUDGET = "token_budget"
    ERROR = "error"
    HOOK_BLOCKED = "hook_blocked"  # Hook blocks execution


@dataclass
class AgentError:
    """Single error record"""

    error_type: str  # "llm_error", "tool_error", "validation_error", "timeout"
    message: str
    loop_iteration: int
    tool_name: str | None = None
    recoverable: bool = False


@dataclass
class AgentConfig:
    max_loops: int = 30
    max_tokens: int | None = None
    system_prompt: str = ""  # Converted to Message at startup

    # Termination control
    total_timeout: float = 300
    token_budget: int = 100000
    max_empty_responses: int = DEFAULT_MAX_EMPTY_RESPONSES

    # Tool permissions
    allowed_tool_tags: list[str] = field(default_factory=list)

    # Retry config
    max_retries: int = 3
    allowed_tool_names: frozenset[str] | None = None
    strict_token_budget: bool = False


@dataclass
class AgentResult:
    text: str
    termination_reason: TerminationReason
    loops_used: int
    tokens_used: int
    time_elapsed: float
    error: str | None = None
    error_detail: list[AgentError] = field(default_factory=list)
    usage_complete: bool = True


@dataclass
class _Usage:
    tokens: int = 0
    unreported: int = 0


class _BudgetExceeded(Exception):
    pass


class BaseAgent(ABC):
    """Abstract base agent with async loop + termination + retry"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.context: AgentContext | None = None
        self.start_time: float = 0
        self.tokens_used: int = 0
        self._errors: list[AgentError] = []
        self._run_lock = asyncio.Lock()
        self._turn_start = 0
        self._turn_id: str | None = None
        self._iteration = 0
        self._run_started = 0.0
        self._deadline: float | None = None
        self._parent_agent: BaseAgent | None = None
        self._usage: _Usage | None = None
        self._starting_usage = _Usage()
        self._budget_end = config.token_budget

    async def run(self, initial_input: str) -> AgentResult:
        """Process one turn while retaining this agent's conversation."""
        async with self._run_lock:
            self.start_time = time.time()
            self.tokens_used = 0
            self._errors = []
            self._iteration = 0
            self._turn_start = len(self.context.messages) if self.context else 0
            self._turn_id = uuid4().hex
            loop = asyncio.get_running_loop()
            self._run_started = loop.time()
            deadline = self._run_started + self.config.total_timeout
            parent = self._parent_agent
            self._usage = parent._usage if parent and parent._usage else _Usage()
            self._starting_usage = _Usage(self._usage.tokens, self._usage.unreported)
            self._budget_end = self._usage.tokens + self.config.token_budget
            if parent and parent._deadline is not None:
                deadline = min(deadline, parent._deadline)
                self._budget_end = min(self._budget_end, parent._budget_end)
            self._deadline = deadline
            timer = asyncio.timeout_at(deadline)
            try:
                async with timer:
                    return await self._run_turn(initial_input)
            except TimeoutError:
                if not timer.expired():
                    raise
                return self._timeout_result()
            finally:
                self._deadline = None
                self._usage = None

    async def _run_turn(self, initial_input: str) -> AgentResult:
        # Fix: Only initialize context when None (preserve multi-turn conversation history)
        if self.context is None:
            self.context = AgentContext(
                compact_config=CompactConfig(
                    trigger_token_threshold=int(self.config.token_budget * 0.8)
                )
            )
            if self.config.system_prompt:
                await self.context.add_message(
                    Message.from_text(Role.SYSTEM, self.config.system_prompt)
                )

        # Add new user message to existing context
        await self.context.add_message(Message.from_text(Role.USER, initial_input))

        max_empty = max(1, self.config.max_empty_responses)
        consecutive_empty = 0

        # 2. Main loop
        for iteration in range(self.config.max_loops):
            self._iteration = iteration
            # Termination check at start of each iteration (feedback point 13)
            reason = await self._check_termination_conditions(iteration)
            if reason:
                return self._build_result(reason, iteration)

            # LLM call (with retry)
            try:
                response = await self._call_llm_with_retry()
            except _BudgetExceeded as e:
                return self._build_result(
                    TerminationReason.TOKEN_BUDGET, iteration, str(e)
                )
            except Exception as e:
                return self._fail(str(e), iteration)

            self._record_usage(getattr(response, "usage", None))
            if self._deadline_expired():
                return self._timeout_result()
            stop_reason = response.stop_reason
            stop_label = (
                stop_reason.value
                if isinstance(stop_reason, FinishReason)
                else str(stop_reason)
            )

            # Reject empty / thinking-only responses before mutating context
            if self._is_empty_response(response):
                if stop_reason == FinishReason.MAX_TOKENS:
                    return self._fail(
                        "LLM response was truncated by max_tokens before producing "
                        "any text or tool calls.",
                        iteration,
                    )
                if stop_reason in (FinishReason.CONTENT_FILTER, FinishReason.ERROR):
                    return self._fail(
                        f"LLM returned an empty response with stop_reason={stop_label}.",
                        iteration,
                    )

                consecutive_empty += 1
                msg = (
                    f"LLM returned an empty response ({consecutive_empty}/{max_empty}, "
                    f"stop_reason={stop_label})."
                )
                logger.warning(msg)
                if consecutive_empty >= max_empty:
                    return self._fail(msg, iteration)
                self._record_error("llm_error", msg, iteration, recoverable=True)
                continue

            consecutive_empty = 0

            # Fail closed on abnormal / truncated stop reasons; skip saving unpaired
            # tool_use blocks when truncated so multi-turn context stays valid.
            if stop_reason in (
                FinishReason.MAX_TOKENS,
                FinishReason.CONTENT_FILTER,
                FinishReason.ERROR,
            ):
                if not response.has_tool_calls:
                    await self.context.add_message(
                        Message(
                            role=Role.ASSISTANT,
                            content=response.content,
                            provider=getattr(response, "provider", None),
                            model=getattr(response, "model", None),
                        )
                    )
                err_msg = (
                    "LLM response was truncated by max_tokens."
                    if stop_reason == FinishReason.MAX_TOKENS
                    else f"LLM terminated abnormally with stop_reason={stop_label}."
                )
                return self._fail(err_msg, iteration)

            # Add LLM response to context as assistant message
            await self.context.add_message(
                Message(
                    role=Role.ASSISTANT,
                    content=response.content,
                    provider=getattr(response, "provider", None),
                    model=getattr(response, "model", None),
                )
            )
            results_start = len(self.context.messages)

            # Termination check after LLM returns (feedback point 13)
            # Only terminate if there are no pending tool calls to execute
            if not response.has_tool_calls and stop_reason in (
                FinishReason.END_TURN,
                FinishReason.STOP,
                FinishReason.STOP_SEQUENCE,
            ):
                return self._build_result(TerminationReason.END_TURN, iteration)

            # Execute tools (if any)
            if response.has_tool_calls:
                tool_calls = response.get_tool_calls()
                if self._remaining_tokens() <= 0:
                    await self._complete_tool_batch(
                        tool_calls,
                        results_start,
                        "Tool call skipped: token budget exhausted",
                    )
                    return self._build_result(TerminationReason.TOKEN_BUDGET, iteration)
                try:
                    await self._execute_tools(tool_calls)
                except asyncio.CancelledError:
                    await self._complete_tool_batch(
                        tool_calls, results_start, "Tool call interrupted"
                    )
                    raise
                if self._deadline_expired():
                    return self._timeout_result()
                if self._remaining_tokens() <= 0:
                    return self._build_result(TerminationReason.TOKEN_BUDGET, iteration)
                if not self.context.has_signed_turn(self._turn_start):
                    await self.context.compact_if_needed()

        return self._build_result(
            TerminationReason.MAX_LOOPS, self.config.max_loops - 1
        )

    @staticmethod
    def _is_empty_response(response: Any) -> bool:
        """Return True when response has neither tool calls nor non-blank text."""
        if response.has_tool_calls:
            return False
        content = response.content or []
        if not content:
            return True
        return all(
            isinstance(block, (ThinkingBlock, RedactedThinkingBlock))
            or (isinstance(block, TextBlock) and not block.text.strip())
            for block in content
        )

    def _fail(self, message: str, iteration: int) -> AgentResult:
        """Record a terminal LLM error and build an ERROR AgentResult."""
        self._record_error("llm_error", message, iteration, recoverable=False)
        return self._build_result(TerminationReason.ERROR, iteration, message)

    def _deadline_expired(self) -> bool:
        return (
            self._deadline is not None
            and asyncio.get_running_loop().time() >= self._deadline
        )

    def _timeout_result(self) -> AgentResult:
        message = "Agent exceeded its time limit"
        self._record_error("timeout", message, self._iteration)
        return self._build_result(TerminationReason.TIMEOUT, self._iteration, message)

    def _llm_call_kwargs(self) -> dict[str, Any]:
        """Build optional provider kwargs derived from AgentConfig."""
        if self.config.max_tokens is not None:
            return {"max_tokens": min(self.config.max_tokens, self._remaining_tokens())}
        return {}

    async def _provider_kwargs(self, provider, tools) -> dict[str, Any]:
        kwargs = self._llm_call_kwargs()
        if not self._requires_exact_budget():
            return kwargs
        if self._usage and self._usage.unreported:
            raise _BudgetExceeded("Provider token usage is incomplete")
        counter = getattr(provider, "count_input_tokens", None)
        if not callable(counter):
            raise _BudgetExceeded("A provider token counter is required")
        assert self.context is not None
        input_tokens = await counter(self.context.messages, tools)
        if type(input_tokens) is not int or input_tokens < 0:
            raise _BudgetExceeded("Provider did not return a valid input token count")
        remaining = self._remaining_tokens() - input_tokens
        if remaining <= 0:
            raise _BudgetExceeded("Insufficient token budget for this request")
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", remaining), remaining)
        return kwargs

    def _requires_exact_budget(self) -> bool:
        return self.config.strict_token_budget or (
            self._parent_agent is not None
            and self._parent_agent._requires_exact_budget()
        )

    def _remaining_tokens(self) -> int:
        if self._usage is None:
            return self.config.token_budget - self.tokens_used
        return self._budget_end - self._usage.tokens

    def _record_usage(self, usage: dict | None) -> None:
        recorded = normalize_usage(usage)
        tokens = recorded.get("total_tokens")
        complete = (
            isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
        )
        token_count = (
            tokens
            if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
            else 0
        )
        self.tokens_used += token_count
        if self._usage is not None:
            self._usage.tokens += token_count
            if not complete:
                self._usage.unreported += 1

    async def _complete_tool_batch(
        self, tool_calls, results_start: int, message: str
    ) -> None:
        assert self.context is not None
        completed = {
            block.tool_call_id
            for item in self.context.messages[results_start:]
            for block in item.content
            if isinstance(block, ToolResultBlock)
        }
        remaining: list[ContentBlockUnion] = [
            ToolResultBlock(call.tool_call_id, message, is_error=True)
            for call in tool_calls
            if call.tool_call_id not in completed
        ]
        if remaining:
            await self.context.add_message(Message(role=Role.USER, content=remaining))

    async def _check_termination_conditions(
        self, iteration: int
    ) -> TerminationReason | None:
        """Check at start of each iteration (feedback point 13)"""
        if self._deadline is not None:
            expired = self._deadline_expired()
        else:
            expired = time.time() - self.start_time > self.config.total_timeout
        if expired:
            return TerminationReason.TIMEOUT

        if self._remaining_tokens() <= 0:
            return TerminationReason.TOKEN_BUDGET

        return None

    async def _call_llm_with_retry(self):
        """Agent-level retry control with recoverable check.

        Retries only recoverable errors (rate limits, network issues, 5xx).
        Non-recoverable errors (auth, context window) fail immediately.
        Honors Retry-After header when provided by the provider.
        """
        attempts = max(1, self.config.max_retries)
        for attempt in range(attempts):
            try:
                return await self._call_llm()
            except _BudgetExceeded:
                raise
            except Exception as e:
                recoverable = getattr(e, "recoverable", False)
                self._record_error("llm_error", str(e), 0, recoverable=recoverable)
                # Cannot succeed on retry — fail immediately.
                if not recoverable:
                    raise
                if attempt == attempts - 1:
                    raise
                delay = self._compute_retry_delay(e, attempt)
                logger.debug("LLM retry attempt %d, sleeping %.1fs", attempt + 1, delay)
                await asyncio.sleep(delay)

    def _compute_retry_delay(self, e: Exception, attempt: int) -> float:
        """Prefer the provider's Retry-After, capped; fall back to backoff."""
        retry_after: int | None = getattr(e, "retry_after", None)
        if retry_after and retry_after > 0:
            return float(min(retry_after, MAX_RETRY_DELAY))
        return float(min(2**attempt, MAX_RETRY_DELAY))

    def _record_error(
        self,
        error_type: str,
        message: str,
        iteration: int,
        tool_name: str | None = None,
        recoverable: bool = False,
    ):
        """Record an error"""
        self._errors.append(
            AgentError(
                error_type=error_type,
                message=message,
                loop_iteration=iteration,
                tool_name=tool_name,
                recoverable=recoverable,
            )
        )

    def _build_result(
        self, reason: TerminationReason, iteration: int, error: str | None = None
    ) -> AgentResult:
        """Build final result"""
        text = ""
        if self.context:
            for msg in reversed(self.context.messages[self._turn_start :]):
                if msg.role == Role.ASSISTANT:
                    text = "".join(
                        block.text
                        for block in msg.content
                        if isinstance(block, TextBlock)
                    )
                    if text:
                        break

        usage = self._usage
        elapsed = time.time() - self.start_time
        if usage is not None:
            self.tokens_used = usage.tokens - self._starting_usage.tokens
            elapsed = asyncio.get_running_loop().time() - self._run_started
        return AgentResult(
            text=text,
            termination_reason=reason,
            loops_used=iteration + 1,
            tokens_used=self.tokens_used,
            time_elapsed=elapsed,
            error=error,
            error_detail=self._errors[:],
            usage_complete=usage is None
            or usage.unreported == self._starting_usage.unreported,
        )

    @abstractmethod
    async def _call_llm(self):
        """Implement in subclass"""
        pass

    @abstractmethod
    async def _execute_tools(self, tool_calls: list) -> None:
        """Implement in subclass"""
        pass
