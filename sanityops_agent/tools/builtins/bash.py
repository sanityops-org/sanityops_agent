from __future__ import annotations

import asyncio
import os
import signal
from typing import Any

from sanityops_agent.tools.base import Tool, ToolResult
from sanityops_agent.tools.context import ToolContext, current_tool_invocation

# Dangerous command patterns that should be rejected
_DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "mkfs",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "sudo reboot",
    "sudo shutdown",
    "init 0",
    ":(){:|:&};:",  # fork bomb
]


class BashTool(Tool):
    """Execute shell commands safely with timeout and output limits."""

    name = "bash"
    description = "Execute a shell command and return its output"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    tags = ["shell", "terminal"]

    MAX_OUTPUT_CHARS = 50_000  # Limit output to prevent context overflow

    def __init__(self, context: ToolContext | None = None, timeout: float = 30.0):
        super().__init__(context=context)
        self.timeout = timeout

    def _is_dangerous(self, command: str) -> str | None:
        """Check if command matches dangerous patterns. Returns reason or None."""
        cmd = command.strip()
        cmd_lower = cmd.lower()

        # Special handling for rm -rf: only reject when target is / or /* or ~
        import re

        rm_match = re.match(
            r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\s+(.+)",
            cmd_lower,
        )
        if rm_match:
            target = rm_match.group(2).strip()
            # Split on whitespace to get the first target argument
            first_target = target.split()[0] if target.split() else ""
            if first_target in ("/", "/*", "~", "$HOME", "${home}"):
                return f"dangerous command pattern detected: rm -rf {first_target}"

        for pattern in _DANGEROUS_PATTERNS:
            if pattern.lower().startswith("rm "):
                continue  # Already handled above
            if pattern.lower() in cmd_lower:
                return f"dangerous command pattern detected: {pattern}"
        return None

    async def execute(  # type: ignore[override]
        self,
        command: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute the shell command via asyncio subprocess."""
        if not command or not command.strip():
            return ToolResult(
                content="",
                success=False,
                error="Empty command",
            )

        # Check for dangerous commands
        dangerous_reason = self._is_dangerous(command)
        if dangerous_reason:
            return ToolResult(
                content="",
                success=False,
                error=dangerous_reason,
            )

        try:
            async with asyncio.timeout(self.timeout):
                if current_tool_invocation.get() is not None:
                    runner = self.context.get("shell_runner")
                    if runner is None:
                        return ToolResult(
                            "", False, "Host must configure an isolated shell runner"
                        )
                    result = await runner(
                        command=command,
                        timeout=self.timeout,
                        max_output_bytes=self.MAX_OUTPUT_CHARS,
                    )
                    content = self._bounded_text(result.content)
                    error = (
                        self._bounded_text(result.error)
                        if result.error
                        else result.error
                    )
                    metadata = result.metadata.copy()
                    if content != result.content or error != result.error:
                        metadata["truncated"] = True
                    return ToolResult(content, result.success, error, metadata)
                return await self._run_local(command)
        except TimeoutError:
            return ToolResult(
                "",
                False,
                f"Command timed out after {self.timeout}s",
                metadata={"returncode": -1, "timed_out": True},
            )
        except Exception as exc:
            return ToolResult("", False, str(exc))

    def _bounded_text(self, text: str) -> str:
        # Slice before encoding to avoid copying an oversized runner response.
        prefix = text[: self.MAX_OUTPUT_CHARS].encode("utf-8", errors="replace")
        return prefix[: self.MAX_OUTPUT_CHARS].decode("utf-8", errors="ignore")

    async def _run_local(self, command: str) -> ToolResult:
        """Direct, trusted Python entry point. Drain pipes even after the cap."""
        spawn = asyncio.create_task(
            asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        try:
            proc = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            proc = await spawn
            self._kill_group(proc)
            await self._collect(proc)
            raise

        collected = asyncio.create_task(self._collect(proc))
        try:
            return await asyncio.shield(collected)
        except asyncio.CancelledError:
            self._kill_group(proc)
            await collected
            raise
        except Exception:
            self._kill_group(proc)
            await proc.wait()
            raise

    @staticmethod
    def _kill_group(proc: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _collect(self, proc: asyncio.subprocess.Process) -> ToolResult:
        async def drain(stream: asyncio.StreamReader) -> tuple[str, int]:
            retained = bytearray()
            size = 0
            while chunk := await stream.read(8192):
                size += len(chunk)
                retained.extend(chunk[: self.MAX_OUTPUT_CHARS - len(retained)])
            return retained.decode("utf-8", errors="replace"), size

        assert proc.stdout is not None and proc.stderr is not None
        (stdout, out_bytes), (stderr, err_bytes) = await asyncio.gather(
            drain(proc.stdout), drain(proc.stderr)
        )
        await proc.wait()
        code = proc.returncode
        content = stdout.rstrip("\n")
        if stderr:
            content += ("\nSTDERR:\n" if content else "") + stderr.rstrip("\n")
        truncated = (
            out_bytes + err_bytes > self.MAX_OUTPUT_CHARS
            or len(content) > self.MAX_OUTPUT_CHARS
        )
        if truncated:
            suffix = "\n... [output truncated]"
            prefix = self._bounded_text(content).encode()
            available = max(0, self.MAX_OUTPUT_CHARS - len(suffix))
            content = (
                prefix[:available].decode("utf-8", errors="ignore")
                + suffix[: self.MAX_OUTPUT_CHARS]
            )
        return ToolResult(
            self._bounded_text(content),
            code == 0,
            self._bounded_text(stderr.strip() or f"Command exited with status {code}")
            if code
            else None,
            metadata={
                "returncode": code,
                "truncated": truncated,
                "stdout_length": out_bytes,
                "stderr_length": err_bytes,
            },
        )
