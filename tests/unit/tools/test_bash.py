import asyncio
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import AsyncMock, MagicMock, patch

from sanityops_agent.tools.base import ToolResult
from sanityops_agent.tools.builtins.bash import BashTool
from sanityops_agent.tools.context import (
    ToolContext,
    ToolInvocation,
    current_tool_invocation,
)


class TestBashTool(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = BashTool(timeout=5.0)

    async def test_simple_command(self):
        result = await self.tool.execute(command="echo hello")
        self.assertTrue(result.success)
        self.assertEqual(result.content, "hello")

    async def test_command_with_output(self):
        result = await self.tool.execute(command="pwd")
        self.assertTrue(result.success)
        self.assertTrue(len(result.content) > 0)

    async def test_failed_command(self):
        result = await self.tool.execute(command="ls /nonexistent_path_xyz_12345")
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    async def test_empty_command(self):
        result = await self.tool.execute(command="")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Empty command")

    async def test_whitespace_only_command(self):
        result = await self.tool.execute(command="   ")
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Empty command")

    @patch("asyncio.create_subprocess_shell", new_callable=AsyncMock)
    async def test_dangerous_rm_rf_root(self, spawn):
        result = await self.tool.execute(command="rm -rf /")
        self.assertFalse(result.success)
        spawn.assert_not_called()
        self.assertIn("dangerous", result.error.lower())

    @patch("asyncio.create_subprocess_shell", new_callable=AsyncMock)
    async def test_dangerous_mkfs(self, spawn):
        result = await self.tool.execute(command="mkfs.ext4 /dev/sda1")
        self.assertFalse(result.success)
        spawn.assert_not_called()
        self.assertIn("dangerous", result.error.lower())

    async def test_pipe_command(self):
        result = await self.tool.execute(command="echo test | grep test")
        self.assertTrue(result.success)
        self.assertIn("test", result.content)

    async def test_multiline_command(self):
        result = await self.tool.execute(command="echo line1 && echo line2")
        self.assertTrue(result.success)
        self.assertIn("line1", result.content)
        self.assertIn("line2", result.content)

    async def test_returncode_metadata(self):
        result = await self.tool.execute(command="echo ok")
        self.assertTrue(result.success)
        self.assertIn("returncode", result.metadata)
        self.assertEqual(result.metadata["returncode"], 0)

    @patch("asyncio.create_subprocess_shell", new_callable=AsyncMock)
    async def test_dangerous_sudo_reboot(self, spawn):
        result = await self.tool.execute(command="sudo reboot")
        self.assertFalse(result.success)
        spawn.assert_not_called()
        self.assertIn("dangerous", result.error.lower())

    @patch("asyncio.create_subprocess_shell", new_callable=AsyncMock)
    async def test_dangerous_dd(self, spawn):
        result = await self.tool.execute(command="dd if=/dev/zero of=/dev/sda")
        self.assertFalse(result.success)
        spawn.assert_not_called()
        self.assertIn("dangerous", result.error.lower())

    def test_filter_accepts_non_root_target(self):
        self.assertIsNone(self.tool._is_dangerous("rm -rf /tmp/test-fixture"))


class TestBashBoundary(IsolatedAsyncioTestCase):
    async def test_model_uses_only_host_runner(self):
        context = ToolContext()
        tool = BashTool(context, timeout=2.0)
        token = current_tool_invocation.set(
            ToolInvocation(agent=None, allowed_tools=frozenset())
        )
        self.addCleanup(current_tool_invocation.reset, token)
        with patch("asyncio.create_subprocess_shell", new_callable=AsyncMock) as spawn:
            result = await tool.execute(command="fixture")
            self.assertFalse(result.success)
            spawn.assert_not_called()
            runner = AsyncMock(return_value=ToolResult(content="sandbox fixture"))
            context.set("shell_runner", runner)
            result = await tool.execute(command="fixture")
            self.assertEqual(result.content, "sandbox fixture")
            runner.assert_awaited_once_with(
                command="fixture", timeout=2.0, max_output_bytes=tool.MAX_OUTPUT_CHARS
            )
            spawn.assert_not_called()

    async def test_runner_output_and_errors_respect_byte_limit(self):
        context = ToolContext()
        tool = BashTool(context)
        tool.MAX_OUTPUT_CHARS = 12
        context.set(
            "shell_runner",
            AsyncMock(return_value=ToolResult("data" * 10, False, "data" * 10)),
        )
        token = current_tool_invocation.set(
            ToolInvocation(agent=None, allowed_tools=frozenset())
        )
        self.addCleanup(current_tool_invocation.reset, token)
        result = await tool.execute(command="fixture")
        self.assertLessEqual(len(result.content.encode()), 12)
        self.assertLessEqual(len(result.error.encode()), 12)
        self.assertTrue(result.metadata["truncated"])

    async def test_cancel_propagates_to_runner(self):
        context = ToolContext()
        tool = BashTool(context)
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def runner(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        context.set("shell_runner", runner)
        token = current_tool_invocation.set(
            ToolInvocation(agent=None, allowed_tools=frozenset())
        )
        self.addCleanup(current_tool_invocation.reset, token)
        task = asyncio.create_task(tool.execute(command="fixture"))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cleaned.is_set())

    def fake_process(self):
        proc = MagicMock(pid=12345)
        proc.stdout = asyncio.StreamReader()
        proc.stderr = asyncio.StreamReader()
        proc.wait = AsyncMock(return_value=0)
        proc.returncode = 0
        return proc

    async def test_bounded_output_and_error(self):
        tool = BashTool()
        tool.MAX_OUTPUT_CHARS = 20
        proc = self.fake_process()
        proc.returncode = 1
        proc.stderr.feed_data(b"x" * 200)
        proc.stderr.feed_eof()
        proc.stdout.feed_eof()
        with patch(
            "asyncio.create_subprocess_shell", new_callable=AsyncMock, return_value=proc
        ):
            result = await tool.execute(command="fixture")
        self.assertFalse(result.success)
        self.assertTrue(result.metadata["truncated"])
        self.assertLessEqual(len(result.error), 20)
        self.assertEqual(result.metadata["stderr_length"], 200)

    async def test_timeout_reaps_process_group(self):
        await self.check_reaped(cancel=False)

    async def test_cancellation_reaps_process_group(self):
        await self.check_reaped(cancel=True)

    async def check_reaped(self, *, cancel):
        tool = BashTool(timeout=0.02 if not cancel else 60)
        proc = self.fake_process()
        started = asyncio.Event()

        async def spawn(*args, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            started.set()
            return proc

        def kill(*args):
            proc.stderr.feed_eof()
            proc.stdout.feed_eof()

        with (
            patch("asyncio.create_subprocess_shell", side_effect=spawn),
            patch("os.killpg", side_effect=kill) as group,
        ):
            task = asyncio.create_task(tool.execute(command="fixture"))
            await asyncio.wait_for(started.wait(), 1)
            if cancel:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                self.assertTrue(result.metadata["timed_out"])
            group.assert_called_once()
            proc.wait.assert_awaited()


if __name__ == "__main__":
    main()
