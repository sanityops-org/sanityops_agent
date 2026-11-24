from __future__ import annotations

import asyncio
from contextlib import closing
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from sanityops_agent.tools import workspace
from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.workspace import WorkspacePath


def _matches(path: Path, parts: tuple[str, ...]) -> bool:
    """Match complete path components; ** consumes zero or more components."""
    positions = {0}
    for pattern in parts:
        if not positions:
            return False
        if pattern == "**":
            positions = set(range(min(positions), len(path.parts) + 1))
        else:
            positions = {
                index + 1
                for index in positions
                if index < len(path.parts) and fnmatchcase(path.parts[index], pattern)
            }
    return len(path.parts) in positions


class GlobTool(Tool):
    """Find regular files without following directory or file symlinks."""

    name = "glob"
    description = "Find files by name pattern"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Glob pattern, e.g. '*.py' or '**/*.json'",
            },
            "path": {
                "type": "string",
                "minLength": 1,
                "description": "Directory to search (defaults to the workspace root)",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }
    tags = ["file"]

    async def execute(  # type: ignore[override]
        self, pattern: str, path: str | None = None, **kwargs: Any
    ) -> ToolResult:
        if not isinstance(pattern, str) or not pattern:
            return ToolResult("", False, "Pattern is required")
        glob = Path(pattern)
        if len(pattern) > 256 or glob.is_absolute() or ".." in glob.parts:
            return ToolResult("", False, "Invalid workspace glob pattern")
        try:
            base = WorkspacePath(self.context, path if path is not None else ".")
            depth = None if "**" in glob.parts else len(glob.parts)
            matches = []
            remaining = workspace.MAX_OUTPUT_CHARS
            truncated = False
            async with asyncio.timeout(workspace.TIMEOUT):
                with (
                    base.directory(base.relative) as directory,
                    closing(workspace.walk(directory, depth=depth)) as entries,
                ):
                    for _, relative, regular in entries:
                        await asyncio.sleep(0)
                        if regular and _matches(relative, glob.parts):
                            match = str(base.root / base.relative / relative)
                            if len(match) > remaining:
                                truncated = True
                                break
                            matches.append(match)
                            remaining -= len(match) + 1
                            if len(matches) >= workspace.MAX_MATCHES:
                                truncated = True
                                break
            return ToolResult(
                "\n".join(sorted(matches)) or "No files found",
                metadata={"match_count": len(matches), "truncated": truncated},
            )
        except FileNotFoundError:
            return ToolResult("", False, "Path not found")
        except NotADirectoryError:
            return ToolResult("", False, "Not a directory")
        except TimeoutError:
            return ToolResult("", False, "Glob timed out")
        except (OSError, ValueError) as exc:
            return ToolResult("", False, str(exc))
