"""Shared tool policy and dispatch for the built-in agent loops."""

from __future__ import annotations

import asyncio
from copy import deepcopy

from sanityops_agent.core.agent import BaseAgent
from sanityops_agent.core.message import (
    Message,
    Role,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.hooks.base import HookEvent, HookExecutor
from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.context import ToolInvocation, current_tool_invocation
from sanityops_agent.tools.registry import ToolRegistry


class ToolAgent(BaseAgent):
    tool_registry: ToolRegistry
    hook_executor: HookExecutor | None = None
    _request_tools: dict[str, Tool] | None = None

    def _get_available_tools(self) -> list[Tool]:
        host_names = self.tool_registry.context.get("allowed_tools")
        return [
            tool
            for tool in self.tool_registry.list_available(
                self.config.allowed_tool_tags,
                allowed_names=self.config.allowed_tool_names,
            )
            if host_names is None or tool.name in host_names
            if tool.context is self.tool_registry.context
        ]

    def _tool_definitions(self) -> list[ToolDefinition]:
        parent = self._parent_agent
        parent_tools = (
            parent._authorized_tools() if isinstance(parent, ToolAgent) else None
        )
        self._request_tools = {
            tool.name: tool
            for tool in self._get_available_tools()
            if parent_tools is None or parent_tools.get(tool.name) is tool
        }
        return [tool.to_tool_definition() for tool in self._request_tools.values()]

    def _authorized_tools(self) -> dict[str, Tool]:
        live_tools = {tool.name: tool for tool in self._get_available_tools()}
        offered = self._request_tools if self._request_tools is not None else live_tools
        parent = self._parent_agent
        parent_tools = (
            parent._authorized_tools() if isinstance(parent, ToolAgent) else None
        )
        return {
            name: tool
            for name, tool in offered.items()
            if live_tools.get(name) is tool
            and (parent_tools is None or parent_tools.get(name) is tool)
        }

    async def _invoke_tool(self, call: ToolUseBlock) -> ToolResult:
        name = call.tool_name
        tool = self._authorized_tools().get(name)
        if tool is None:
            if self.tool_registry.get(name) is None:
                message = f"Tool not found: {name}"
            else:
                message = f"Tool '{name}' is not available to this agent"
            self._record_error("tool_error", message, self._iteration, tool_name=name)
            return ToolResult(message, success=False)

        invalid = self.tool_registry.validate_input(name, call.tool_input)
        if invalid:
            return ToolResult(invalid, success=False)

        arguments = deepcopy(call.tool_input)
        if self.hook_executor:
            try:
                approved = await self.hook_executor.execute(
                    HookEvent.BEFORE_TOOL_EXEC,
                    {
                        "tool_name": name,
                        "tool_input": deepcopy(arguments),
                        "tool_call_id": call.tool_call_id,
                    },
                )
                allowed = (
                    not approved.blocked
                    and approved.tool_name == name
                    and approved.tool_input == arguments
                    and approved.tool_call_id == call.tool_call_id
                )
            except Exception:
                allowed = False
            if not allowed:
                return ToolResult(
                    f"Tool execution blocked by hook: {name}", success=False
                )

        authorized = self._authorized_tools()
        if authorized.get(name) is not tool:
            return ToolResult(f"Tool authorization changed: {name}", success=False)
        if (
            self._deadline is not None
            and self._deadline <= asyncio.get_running_loop().time()
        ):
            return ToolResult("Tool call skipped: time limit reached", success=False)
        invocation = current_tool_invocation.set(
            ToolInvocation(self, frozenset(authorized))
        )
        try:
            return await tool.run(**arguments)
        except Exception as exc:
            self._record_error("tool_error", str(exc), self._iteration, tool_name=name)
            return ToolResult(f"Tool error: {exc}", success=False)
        finally:
            current_tool_invocation.reset(invocation)

    async def _execute_tools(self, tool_calls: list) -> None:
        if self.context is None:
            raise RuntimeError("Tool execution context is not initialised")
        for call in tool_calls:
            if isinstance(call, dict):
                call = ToolUseBlock(
                    call.get("id", ""), call.get("name", ""), call.get("input", {})
                )
            if self._remaining_tokens() <= 0:
                result = ToolResult(
                    "Tool call skipped: token budget exhausted", success=False
                )
            else:
                result = await self._invoke_tool(call)
            block = ToolResultBlock(
                call.tool_call_id,
                result.content or result.error or "",
                is_error=not result.success,
                history_key=getattr(result, "history_key", None),
                metadata={"tool_name": call.tool_name},
            )
            await self.context.add_message(Message(role=Role.USER, content=[block]))
            history = self.tool_registry.context.history_store
            if history is not None:
                block.history_key = await history.store(
                    call.tool_name,
                    block.tool_result_content,
                    call.tool_call_id,
                    turn_id=self._turn_id,
                )
            if self.hook_executor:
                await self.hook_executor.execute(
                    HookEvent.AFTER_TOOL_EXEC,
                    {
                        "tool_name": call.tool_name,
                        "tool_call_id": call.tool_call_id,
                        "result": block.tool_result_content,
                        "success": result.success,
                        "is_error": block.is_error,
                    },
                )
