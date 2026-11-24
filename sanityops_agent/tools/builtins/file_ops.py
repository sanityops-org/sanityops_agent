from __future__ import annotations

import asyncio
import os
import secrets
import stat
from contextlib import closing
from typing import Any

import regex
from jsonschema import Draft202012Validator

from sanityops_agent.tools import workspace
from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.workspace import WorkspacePath, read_regular


class FileOpsTool(Tool):
    """Read, write, patch, and search bounded UTF-8 files."""

    name = "file_ops"
    description = "Read, write, patch, search files"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "patch", "search"],
            },
            "path": {"type": "string", "minLength": 1, "maxLength": 4096},
            "content": {"type": "string", "maxLength": workspace.MAX_FILE_BYTES},
            "old_string": {
                "type": "string",
                "minLength": 1,
                "maxLength": workspace.MAX_FILE_BYTES,
            },
            "new_string": {"type": "string", "maxLength": workspace.MAX_FILE_BYTES},
            "replace_all": {"type": "boolean", "default": False},
            "offset": {"type": "integer", "default": 1},
            "limit": {"type": "integer", "default": 500},
            "pattern": {"type": "string", "minLength": 1, "maxLength": 256},
        },
        "required": ["operation", "path"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"properties": {"operation": {"const": op}}},
                "then": {"required": fields},
            }
            for op, fields in [
                ("write", ["content"]),
                ("patch", ["old_string", "new_string"]),
                ("search", ["pattern"]),
            ]
        ],
    }
    tags = ["file"]
    _validator = Draft202012Validator(parameters)

    # The typed parameters mirror the model-facing schema; the base class
    # accepts generic keyword arguments.
    async def execute(self, operation: str, path: str, **kwargs: Any) -> ToolResult:  # type: ignore[override]
        if operation not in self.parameters["properties"]["operation"]["enum"]:
            return ToolResult("", False, "Unknown operation")
        invalid = next(
            self._validator.iter_errors(
                {"operation": operation, "path": path, **kwargs}
            ),
            None,
        )
        if invalid is not None:
            field = ".".join(map(str, invalid.path)) or operation
            reason = (
                invalid.message
                if invalid.validator == "required"
                else invalid.validator
            )
            return ToolResult("", False, f"Invalid {field} arguments ({reason})")
        try:
            writing = operation in ("write", "patch")
            target = WorkspacePath(self.context, path, writing=writing)
            async with asyncio.timeout(workspace.TIMEOUT):
                if operation == "search":
                    return await self._search(target, kwargs["pattern"])
                data = kwargs["content"].encode() if operation == "write" else None
                if data is not None and len(data) > workspace.MAX_FILE_BYTES:
                    raise ValueError("File exceeds write limit")
                with target.directory(
                    target.relative.parent, create=operation == "write"
                ) as directory:
                    if writing:
                        return await self._modify(target, directory, data, kwargs)
                    raw = await read_regular(directory, target.name)
                    return self._read(
                        raw.decode(),
                        int(kwargs.get("offset", 1)),
                        int(kwargs.get("limit", 500)),
                    )
        except FileNotFoundError:
            return ToolResult("", False, "File not found")
        except IsADirectoryError:
            return ToolResult("", False, "Not a file")
        except UnicodeError:
            return ToolResult("", False, "Cannot process binary file")
        except regex.error:
            return ToolResult("", False, "Invalid regex pattern")
        except TimeoutError:
            return ToolResult("", False, "File operation timed out")
        except (OSError, ValueError, OverflowError, RecursionError) as exc:
            return ToolResult("", False, str(exc))

    @staticmethod
    def _read(text: str, offset: int, limit: int) -> ToolResult:
        lines = text.splitlines()
        start = max(0, offset - 1)
        end = min(len(lines), start + limit) if limit > 0 else len(lines)
        output: list[str] = []
        remaining = workspace.MAX_OUTPUT_CHARS
        truncated = False
        for i in range(start, end):
            record = f"{i + 1}|{lines[i]}"
            if len(record) > remaining:
                output.append(record[:remaining])
                truncated = True
                break
            output.append(record)
            remaining -= len(record) + 1
            if remaining <= 0:
                truncated = i + 1 < end
                break
        metadata = {"total_lines": len(lines), "offset": offset, "limit": limit}
        if truncated or end < len(lines):
            metadata["has_more"] = True
        if truncated:
            metadata["truncated"] = True
        return ToolResult("\n".join(output), metadata=metadata)

    @staticmethod
    async def _modify(
        target: WorkspacePath, directory: int, data: bytes | None, args: dict
    ) -> ToolResult:
        import fcntl

        # A directory lock survives replacement of the target inode and also
        # serializes concurrent writers from other application worker processes.
        while True:
            try:
                fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        try:
            info = os.stat(target.name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Target must be a regular file")
            mode = stat.S_IMODE(info.st_mode)
        except FileNotFoundError:
            mode = 0o600

        if data is None:
            text = (await read_regular(directory, target.name)).decode()
            old, new = args["old_string"], args["new_string"]
            count = (
                text.count(old) if args.get("replace_all", False) else int(old in text)
            )
            if not count:
                raise ValueError("old_string not found in file")
            size = len(text.encode()) + count * (len(new.encode()) - len(old.encode()))
            if size > workspace.MAX_FILE_BYTES:
                raise ValueError("File exceeds write limit")
            data = text.replace(old, new, count).encode()
            result = ToolResult(
                f"Patched: {target.relative}",
                metadata={"old_length": len(old), "new_length": len(new)},
            )
        else:
            result = ToolResult(f"File written: {target.relative} ({len(data)} bytes)")

        temp_name = f".sanityops-{secrets.token_hex(16)}.tmp"
        fd = os.open(
            temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                for start in range(0, len(data), 65536):
                    await asyncio.sleep(0)
                    stream.write(data[start : start + 65536])
                stream.flush()
                os.fchmod(stream.fileno(), mode)
            await asyncio.sleep(0)
            os.replace(
                temp_name, target.name, src_dir_fd=directory, dst_dir_fd=directory
            )
        finally:
            try:
                os.unlink(temp_name, dir_fd=directory)
            except FileNotFoundError:
                pass
        return result

    @staticmethod
    async def _search(target: WorkspacePath, pattern: str) -> ToolResult:
        compiled = regex.compile(pattern)
        matches: list[str] = []
        remaining = workspace.MAX_OUTPUT_CHARS
        scan_bytes = 4 * workspace.MAX_FILE_BYTES

        async def collect(
            fd: int, name: str, label: str, *, single: bool = False
        ) -> bool:
            nonlocal remaining, scan_bytes
            try:
                file_size = os.stat(name, dir_fd=fd, follow_symlinks=False).st_size
                if file_size > workspace.MAX_FILE_BYTES and not single:
                    return False
                if file_size > scan_bytes:
                    raise ValueError("Search byte limit exceeded")
                raw = await read_regular(
                    fd, name, min(scan_bytes, workspace.MAX_FILE_BYTES)
                )
                scan_bytes -= len(raw)
                lines = raw.decode().splitlines()
            except UnicodeDecodeError:
                if single:
                    raise
                return False
            for i, line in enumerate(lines, 1):
                if i % 32 == 0:
                    await asyncio.sleep(0)
                if compiled.search(line, timeout=0.03):
                    record = f"{label}:{i}:{line.strip()}"
                    matches.append(record[:remaining])
                    remaining -= len(record) + 1
                    if remaining <= 0 or len(matches) >= workspace.MAX_MATCHES:
                        return True
            return False

        truncated = False
        with target.directory(target.relative.parent) as parent:
            info = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                truncated = await collect(
                    parent, target.name, str(target.root / target.relative), single=True
                )
            elif stat.S_ISDIR(info.st_mode):
                with (
                    target.directory(target.relative) as base,
                    closing(workspace.walk(base)) as entries,
                ):
                    for fd, path, regular in entries:
                        await asyncio.sleep(0)
                        if regular and await collect(
                            fd, path.name, str(target.root / target.relative / path)
                        ):
                            truncated = True
                            break
            else:
                raise ValueError("Target must be a regular file or directory")
        return ToolResult(
            "\n".join(matches) or "No matches found",
            metadata={"match_count": len(matches), "truncated": truncated},
        )
