"""Unit tests for sanityops_agent.tools.builtins.task - TaskTool"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from sanityops_agent.core.agent import TerminationReason
from sanityops_agent.tools.builtins.task import TaskTool
from sanityops_agent.tools.context import ToolContext


class TestTaskToolExecute:
    """Test TaskTool execute method."""

    @pytest.mark.asyncio
    async def test_execute_without_agent_factory_returns_error(self):
        """When agent_factory is not set, TaskTool returns an error."""
        tool = TaskTool()
        result = await tool.execute(goal="Write a summary")
        assert result.success is False
        assert "agent_factory not available" in result.content

    @pytest.mark.asyncio
    async def test_execute_without_goal_returns_error(self):
        tool = TaskTool()
        result = await tool.execute()
        assert result.success is False
        assert "goal is required" in result.content
        assert result.error == "goal is required"

    @pytest.mark.asyncio
    async def test_execute_with_empty_goal_returns_error(self):
        tool = TaskTool()
        result = await tool.execute(goal="")
        assert result.success is False
        assert "goal is required" in result.content

    @pytest.mark.asyncio
    async def test_execute_delegates_to_sub_agent(self):
        """TaskTool creates and runs a sub-agent via AgentFactory."""
        # Mock the sub-agent result
        mock_result = MagicMock()
        mock_result.text = "Sub-agent completed the task successfully."
        mock_result.termination_reason = TerminationReason.END_TURN
        mock_result.loops_used = 3
        mock_result.tokens_used = 500
        mock_result.time_elapsed = 1.5
        mock_result.error = None

        mock_sub_agent = MagicMock()
        mock_sub_agent.run = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock()
        mock_factory.create_sub_agent.return_value = mock_sub_agent

        ctx = ToolContext()
        ctx.set("agent_factory", mock_factory)
        tool = TaskTool(context=ctx)

        result = await tool.execute(
            goal="Review this file",
            context="Focus on error handling",
        )

        assert result.success is True
        assert "Sub-agent completed the task successfully." in result.content
        assert "end_turn" in result.content
        assert result.metadata["sub_agent_result"] is mock_result
        mock_factory.create_sub_agent.assert_called_once()
        # TaskTool prefixes goal with "Task: "
        mock_sub_agent.run.assert_called_once_with(
            "Task: Review this file\n\nAdditional context: Focus on error handling"
        )

    @pytest.mark.asyncio
    async def test_execute_forwards_system_prompt(self):
        """TaskTool forwards custom system_prompt to sub-agent."""
        mock_result = MagicMock()
        mock_result.text = "done"
        mock_result.termination_reason = TerminationReason.END_TURN
        mock_result.loops_used = 1
        mock_result.tokens_used = 100
        mock_result.time_elapsed = 0.5
        mock_result.error = None

        mock_sub_agent = MagicMock()
        mock_sub_agent.run = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock()
        mock_factory.create_sub_agent.return_value = mock_sub_agent

        ctx = ToolContext()
        ctx.set("agent_factory", mock_factory)
        tool = TaskTool(context=ctx)

        await tool.execute(
            goal="Analyze code",
            system_prompt="You are a code analyzer.",
        )

        call_kwargs = mock_factory.create_sub_agent.call_args[1]
        assert call_kwargs["system_prompt"] == "You are a code analyzer."

    @pytest.mark.asyncio
    async def test_execute_sub_agent_failure(self):
        """When sub-agent raises an exception, TaskTool returns error."""
        mock_factory = MagicMock()
        mock_sub_agent = MagicMock()
        mock_sub_agent.run = AsyncMock(side_effect=RuntimeError("LLM connection lost"))
        mock_factory.create_sub_agent.return_value = mock_sub_agent

        ctx = ToolContext()
        ctx.set("agent_factory", mock_factory)
        tool = TaskTool(context=ctx)

        result = await tool.execute(goal="Something")
        assert result.success is False
        assert "Sub-agent execution failed" in result.content
        assert "LLM connection lost" in result.content


class TestTaskToolDefinition:
    """Test TaskTool metadata."""

    def test_name(self):
        assert TaskTool.name == "task"

    def test_description(self):
        assert "Delegate tasks to subagents" in TaskTool.description

    def test_tags(self):
        assert "delegation" in TaskTool.tags

    def test_to_tool_definition(self):
        tool = TaskTool()
        definition = tool.to_tool_definition()
        assert definition.name == "task"
        props = definition.input_schema["properties"]
        assert "goal" in props
        assert "context" in props
        assert "toolsets" in props

    def test_goal_is_required(self):
        tool = TaskTool()
        definition = tool.to_tool_definition()
        assert "goal" in definition.input_schema["required"]


class TestTaskToolRun:
    """Test TaskTool via the run template method."""

    @pytest.mark.asyncio
    async def test_run_without_agent_factory_returns_error(self):
        """Without agent_factory, run() returns an error."""
        tool = TaskTool()
        result = await tool.run(goal="Do something")
        assert result.success is False
        assert "agent_factory not available" in result.content
