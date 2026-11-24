from __future__ import annotations

from typing import Any

from sanityops_agent.agents.sub import SubAgent
from sanityops_agent.core.agent import TerminationReason
from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.context import ToolContext, current_tool_invocation


class TaskTool(Tool):
    """Delegate tasks to subagents"""

    name = "task"
    description = "Delegate tasks to subagents"
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "goal": {
                "type": "string",
                "description": "The goal/task to delegate to a sub-agent",
            },
            "context": {
                "type": "string",
                "description": "Additional context for the sub-agent",
            },
            "toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tool sets to make available to the sub-agent",
            },
        },
        "required": ["goal"],
    }
    tags = ["delegation"]

    def __init__(self, context: ToolContext | None = None):
        super().__init__(context)

    async def execute(self, **kwargs) -> ToolResult:
        goal = kwargs.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            return ToolResult(
                content="Error: goal is required",
                success=False,
                error="goal is required",
            )

        subagent_context = kwargs.get("context", "")
        invocation = current_tool_invocation.get()
        system_prompt = kwargs.get("system_prompt") if invocation is None else None
        system_prompt = (
            system_prompt
            or "You are a specialized sub-agent. Complete the assigned task."
        )
        task = f"Task: {goal}"
        if subagent_context:
            task += f"\n\nAdditional context: {subagent_context}"

        # Get the AgentFactory from ToolContext to create sub-agents
        factory = self.context.agent_factory if self.context else None
        if factory is None:
            return ToolResult(
                content=f"Error: agent_factory not available in ToolContext. Cannot delegate task: {goal}",
                success=False,
                error="agent_factory not available",
            )

        try:
            available = invocation.allowed_tools if invocation else None
            overrides: dict[str, Any] = {}
            toolsets = kwargs.get("toolsets")
            if toolsets is not None:
                if not isinstance(toolsets, list) or any(
                    not isinstance(tag, str) for tag in toolsets
                ):
                    raise ValueError("toolsets must be a list of tool tags")
                available = frozenset(
                    tool.name
                    for tool in factory.tool_registry.list_all()
                    if (available is None or tool.name in available)
                    and set(toolsets).intersection(getattr(tool, "tags", None) or ())
                )
            if available is not None:
                overrides["allowed_tool_names"] = available
            if invocation and invocation.agent.config.max_tokens is not None:
                limit = invocation.agent.config.max_tokens
                configured = factory.config.max_tokens
                overrides["max_tokens"] = (
                    min(limit, configured) if configured is not None else limit
                )
            sub_agent = factory.create_sub_agent(
                system_prompt=system_prompt, **overrides
            )
            if invocation:
                sub_agent._parent_agent = invocation.agent
            result = await sub_agent.run(task)
        except Exception as e:
            return ToolResult(
                content=f"Sub-agent execution failed: {e}",
                success=False,
                error=str(e),
                metadata={"termination_reason": TerminationReason.ERROR.value},
            )

        response = SubAgent.build_response(result)
        # Build result content from sub-agent output
        content_lines = []
        if result.text:
            content_lines.append(result.text)
        content_lines.append(
            f"\n--- Sub-agent Summary ---\n"
            f"Termination: {result.termination_reason.value}\n"
            f"Loops: {result.loops_used}\n"
            f"Tokens: {result.tokens_used}\n"
            f"Elapsed: {result.time_elapsed:.1f}s"
        )
        if result.error:
            content_lines.append(f"Error: {result.error}")

        return ToolResult(
            content="\n".join(content_lines),
            success=response.success,
            error=None
            if response.success
            else (
                response.error
                or f"Sub-agent stopped: {response.termination_reason.value}"
            ),
            metadata={
                "sub_agent_result": result,
                "termination_reason": result.termination_reason.value,
                "loops_used": result.loops_used,
                "tokens_used": result.tokens_used,
            },
        )
