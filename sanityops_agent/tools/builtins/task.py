from __future__ import annotations

from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.context import ToolContext


class TaskTool(Tool):
    """Delegate tasks to subagents"""

    name = "task"
    description = "Delegate tasks to subagents"
    parameters = {
        "type": "object",
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
        if not goal:
            return ToolResult(
                content="Error: goal is required",
                success=False,
                error="goal is required",
            )

        subagent_context = kwargs.get("context", "")
        system_prompt = kwargs.get("system_prompt", "")

        # Get the AgentFactory from ToolContext to create sub-agents
        factory = self.context.agent_factory if self.context else None
        if factory is None:
            return ToolResult(
                content=f"Error: agent_factory not available in ToolContext. Cannot delegate task: {goal}",
                success=False,
                error="agent_factory not available",
            )

        # Build the sub-agent system prompt
        if not system_prompt:
            if subagent_context:
                system_prompt = (
                    f"You are a specialized sub- Complete the following task.\n\n"
                    # f"Task: {goal}\n\n"
                    f"Additional context: {subagent_context}"
                )
            else:
                system_prompt = "You are a specialized sub- Complete the following task."

        # Create sub-agent via factory
        sub_agent = factory.create_sub_agent(
            system_prompt=system_prompt,
        )

        # Run the sub-agent
        try:
            result = await sub_agent.run(f"Task: {goal}")
        except Exception as e:
            return ToolResult(
                content=f"Sub-agent execution failed: {e}",
                success=False,
                error=str(e),
            )

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
            metadata={
                "sub_agent_result": result,
                "termination_reason": result.termination_reason.value,
                "loops_used": result.loops_used,
                "tokens_used": result.tokens_used,
            },
        )
