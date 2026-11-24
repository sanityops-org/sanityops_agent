"""Descriptor-relative access to a host-selected workspace on POSIX."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path

from sanityops_agent.tools.context import ToolContext, current_tool_invocation

MAX_FILE_BYTES = 1_048_576
MAX_OUTPUT_CHARS = 50_000
MAX_ENTRIES = 4_096
MAX_DEPTH = 32
MAX_MATCHES = 200
TIMEOUT = 5.0


class WorkspacePath:
    """Keep the configured root separate from the model-provided path."""

    def __init__(self, context: ToolContext, path: str, *, writing: bool = False):
        if os.name != "posix":
            raise OSError(
                "Workspace file tools require POSIX descriptor-relative access"
            )
        if not isinstance(path, str) or not path or len(path) > 4096:
            raise ValueError("Invalid workspace path")
        target = Path(path)
        root = context.get("workdir")
        if current_tool_invocation.get() is not None:
            if not root:
                raise PermissionError("Host must configure a workspace")
            if writing and context.get("allow_file_writes") is not True:
                raise PermissionError("File writes are disabled")
        if root:
            self.root = Path(root).absolute()
            if target.is_absolute():
                target = target.relative_to(self.root)
            if ".." in target.parts:
                raise ValueError("Path must stay inside the workspace")
            self.relative = target
        else:
            # Direct Python calls retain the caller's working directory and paths.
            absolute = target.resolve()
            self.root = Path(absolute.anchor)
            self.relative = absolute.relative_to(self.root)
        if len(self.relative.parts) > 128:
            raise ValueError("Path is too deep")

    @contextmanager
    def directory(self, relative: Path, *, create: bool = False) -> Iterator[int]:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(self.root, flags)
        try:
            for part in relative.parts:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=fd)
                # Swap before closing so the finally block can never re-close
                # an already-closed descriptor if the close itself fails.
                fd, child = child, fd
                os.close(child)
            yield fd
        finally:
            os.close(fd)

    @property
    def name(self) -> str:
        return self.relative.name or "."


async def read_regular(directory: int, name: str, limit: int | None = None) -> bytes:
    if limit is None:
        limit = MAX_FILE_BYTES
    if not stat.S_ISREG(os.stat(name, dir_fd=directory, follow_symlinks=False).st_mode):
        raise ValueError("Not a file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(name, flags, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Not a file")
        if info.st_size > limit:
            raise ValueError("File exceeds read limit")
        data = bytearray()
        while True:
            await asyncio.sleep(0)
            chunk = os.read(fd, min(65536, limit + 1 - len(data)))
            if not chunk:
                return bytes(data)
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError("File exceeds read limit")
    finally:
        os.close(fd)


def walk(
    directory: int, *, depth: int | None = None
) -> Generator[tuple[int, Path, bool], None, None]:
    """Yield each entry with its pinned parent fd. Links are never traversed."""
    if depth is not None and depth > MAX_DEPTH:
        raise ValueError("Glob exceeds workspace depth limit")
    count = 0

    def visit(
        fd: int, prefix: Path, remaining: int
    ) -> Iterator[tuple[int, Path, bool]]:
        nonlocal count
        with os.scandir(fd) as entries:
            for entry in entries:
                count += 1
                if count > MAX_ENTRIES:
                    raise ValueError("Workspace entry limit exceeded")
                path = prefix / entry.name
                yield fd, path, entry.is_file(follow_symlinks=False)
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if remaining <= 1:
                    if depth is None:
                        raise ValueError("Workspace depth limit exceeded")
                    continue
                child = os.open(
                    entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                try:
                    yield from visit(child, path, remaining - 1)
                finally:
                    os.close(child)

    yield from visit(directory, Path(), MAX_DEPTH if depth is None else depth)
