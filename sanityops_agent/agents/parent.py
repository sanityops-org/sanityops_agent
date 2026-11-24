"""ParentAgent - concrete agent extending BaseAgent with LLM and tool implementations."""

from __future__ import annotations

from sanityops_agent.agents.tool_agent import ToolAgent
from sanityops_agent.core.agent import AgentConfig, TerminationReason
from sanityops_agent.hooks.base import HookEvent, HookExecutor
from sanityops_agent.llm.base import LLMProvider, LLMResponse
from sanityops_agent.tools.registry import ToolRegistry


class HookBlockedException(Exception):
    """Raised when a hook blocks execution."""

    def __init__(self, event: HookEvent, message: str = "Execution blocked by hook"):
        self.event = event
        super().__init__(message)


class ParentAgent(ToolAgent):
    """Concrete parent agent with real LLM calls and tool execution.

    Extends BaseAgent with:
    - _call_llm: calls provider.create() with context messages and tool definitions
    - _execute_tools: iterates tool calls, executes via registry, adds results to context
    - Hook integration: fires BEFORE/AFTER hooks for LLM and tool operations
    """

    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        tool_registry: ToolRegistry,
        hook_executor: HookExecutor | None = None,
    ):
        super().__init__(config)
        self.provider = provider
        self.tool_registry = tool_registry
        self.hook_executor = hook_executor

    # ------------------------------------------------------------------
    # Override _check_termination_conditions to fire BEFORE_LLM_CALL hook
    # ------------------------------------------------------------------

    async def _check_termination_conditions(
        self, iteration: int
    ) -> TerminationReason | None:
        """Check termination + fire BEFORE_LLM_CALL hook.

        If the BEFORE_LLM_CALL hook blocks, return HOOK_BLOCKED immediately.
        """
        # First check standard conditions (timeout, token budget)
        result = await super()._check_termination_conditions(iteration)
        if result:
            return result

        # Fire BEFORE_LLM_CALL hook as part of iteration check
        if self.hook_executor:
            hook_ctx = await self.hook_executor.execute(
                HookEvent.BEFORE_LLM_CALL,
                {
                    "messages": self.context.messages if self.context else [],
                    "iteration": iteration,
                },
            )
            if hook_ctx.blocked:
                return TerminationReason.HOOK_BLOCKED

        return None

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_llm(self) -> LLMResponse:
        """Call the LLM provider with context messages and tool definitions.

        Fires AFTER_LLM_CALL hook after receiving the response.
        BEFORE_LLM_CALL is fired in _check_termination_conditions so that
        blocking can produce TerminationReason.HOOK_BLOCKED.
        """
        assert self.context is not None  # invariant: context initialized in run()
        # Build tool definitions for available tools
        tool_defs = self._tool_definitions()

        response = await self.provider.create(
            self.context.messages,
            tools=tool_defs,
            **await self._provider_kwargs(self.provider, tool_defs),
        )

        if self._usage is None:
            self._record_usage(response.usage)

        # Fire AFTER_LLM_CALL hook
        if self.hook_executor:
            await self.hook_executor.execute(
                HookEvent.AFTER_LLM_CALL, {"response": response}
            )

        return response
