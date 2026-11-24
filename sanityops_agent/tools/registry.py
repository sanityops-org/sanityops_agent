from __future__ import annotations

from collections.abc import Collection
from copy import deepcopy

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable

from sanityops_agent.core.message import ToolDefinition
from sanityops_agent.tools.base import Tool
from sanityops_agent.tools.context import ToolContext


class ToolRegistry:
    """Tool registry with permission filtering"""

    def __init__(self, context: ToolContext | None = None):
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Validator] = {}
        self.context = context or ToolContext()

    def register(self, tool: Tool) -> None:
        """Register a tool instance"""
        if not tool.name:
            raise ValueError("Tool must have a name")
        previous_context = getattr(tool, "_registered_context", None)
        if previous_context is not None and previous_context is not self.context:
            raise ValueError("Create a new tool instance for a different tool context")
        definition = getattr(tool, "to_tool_definition", None)
        schema = deepcopy(
            definition().input_schema
            if callable(definition)
            else getattr(tool, "parameters", {})
        )
        validator = validator_for(schema)
        try:
            validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"Invalid JSON Schema for tool '{tool.name}'") from exc
        tool.context = self.context  # Inject shared context
        tool._registered_context = self.context
        self._tools[tool.name] = tool
        self._validators[tool.name] = validator(schema, registry=Registry())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        return list(self._tools.values())

    def list_available(
        self,
        allowed_tags: list[str] | None = None,
        *,
        allowed_names: Collection[str] | None = None,
    ) -> list[Tool]:
        """Return tools matching the optional tag policy and name restriction."""
        return [
            tool
            for name, tool in self._tools.items()
            if (allowed_names is None or name in allowed_names)
            and (
                not allowed_tags
                or any(
                    tag in allowed_tags for tag in (getattr(tool, "tags", None) or ())
                )
            )
        ]

    def validate_input(self, name: str, arguments: object) -> str | None:
        if not isinstance(arguments, dict):
            return "Tool arguments must be a JSON object"
        try:
            self._validators[name].validate(arguments)
        except ValidationError as exc:
            # ValidationError.message can contain the actual user input.
            field = ".".join(map(str, exc.absolute_path))
            target = f" at {field}" if field else ""
            return f"Invalid arguments for tool '{name}'{target} ({exc.validator})"
        except Unresolvable:
            return f"Tool '{name}' uses an unavailable JSON Schema reference"
        return None

    def to_tool_definitions(
        self, allowed_tags: list[str] | None = None
    ) -> list[ToolDefinition]:
        """Convert available tools to LLM ToolDefinition format"""
        return [t.to_tool_definition() for t in self.list_available(allowed_tags)]

    def remove(self, name: str) -> bool:
        """Remove tool by name, return True if found"""
        if name in self._tools:
            del self._tools[name]
            del self._validators[name]
            return True
        return False

    def clear(self) -> None:
        self._tools.clear()
        self._validators.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
