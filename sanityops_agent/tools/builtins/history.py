from __future__ import annotations

from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.context import ToolContext


class HistoryTool(Tool):
    """Query tool execution history"""

    name = "history"
    description = "Query tool execution history"
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {
                "type": "string",
                "description": "Query string to search in tool execution history",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10,
            },
        },
    }
    tags = ["history"]

    def __init__(self, context: ToolContext | None = None):
        super().__init__(context)

    async def execute(self, **kwargs) -> ToolResult:
        store = self.context.history_store

        if store is None:
            return ToolResult(
                content="Error: history_store is not configured",
                success=False,
                error="history_store is not configured",
            )

        limit = kwargs.get("limit", 10)
        query = kwargs.get("query", "")

        if not callable(getattr(store, "search", None)):
            return ToolResult(
                content="Error: history_store does not support querying",
                success=False,
                error="history_store does not support querying",
            )

        entries = await store.search(query, limit)

        if not entries:
            return ToolResult(content="No matching history entries found")

        lines = []
        for e in entries:
            lines.append(
                f"[{e.timestamp:.0f}] {e.tool_name} ({e.tool_call_id}): {e.content[:100]}"
            )
        return ToolResult(content="\n".join(lines))
