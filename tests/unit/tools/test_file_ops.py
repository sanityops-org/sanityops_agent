import os
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, main

from sanityops_agent.tools.builtins.file_ops import FileOpsTool


class TestFileOpsRead(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = FileOpsTool()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_read_file(self):
        path = os.path.join(self.tmpdir.name, "test.txt")
        Path(path).write_text("line1\nline2\nline3\n")
        result = await self.tool.execute(operation="read", path=path)
        self.assertTrue(result.success)
        self.assertIn("1|line1", result.content)
        self.assertIn("2|line2", result.content)
        self.assertIn("3|line3", result.content)

    async def test_read_with_offset(self):
        path = os.path.join(self.tmpdir.name, "test.txt")
        Path(path).write_text("line1\nline2\nline3\n")
        result = await self.tool.execute(operation="read", path=path, offset=2)
        self.assertTrue(result.success)
        self.assertIn("2|line2", result.content)
        self.assertIn("3|line3", result.content)
        self.assertNotIn("1|line1", result.content)

    async def test_read_with_limit(self):
        path = os.path.join(self.tmpdir.name, "test.txt")
        Path(path).write_text("line1\nline2\nline3\nline4\nline5\n")
        result = await self.tool.execute(operation="read", path=path, limit=2)
        self.assertTrue(result.success)
        self.assertIn("1|line1", result.content)
        self.assertIn("2|line2", result.content)
        self.assertNotIn("3|line3", result.content)
        self.assertIn("has_more", result.metadata)

    async def test_read_nonexistent(self):
        result = await self.tool.execute(operation="read", path="/nonexistent_xyz.txt")
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())

    async def test_read_directory(self):
        result = await self.tool.execute(operation="read", path=self.tmpdir.name)
        self.assertFalse(result.success)
        self.assertIn("not a file", result.error.lower())


class TestFileOpsWrite(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = FileOpsTool()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_write_file(self):
        path = os.path.join(self.tmpdir.name, "output.txt")
        result = await self.tool.execute(
            operation="write", path=path, content="hello world"
        )
        self.assertTrue(result.success)
        self.assertIn("written", result.content.lower())
        self.assertEqual(Path(path).read_text(), "hello world")

    async def test_write_creates_parent_dirs(self):
        path = os.path.join(self.tmpdir.name, "a", "b", "c", "file.txt")
        result = await self.tool.execute(operation="write", path=path, content="nested")
        self.assertTrue(result.success)
        self.assertEqual(Path(path).read_text(), "nested")

    async def test_write_overwrites(self):
        path = os.path.join(self.tmpdir.name, "output.txt")
        Path(path).write_text("old content")
        result = await self.tool.execute(
            operation="write", path=path, content="new content"
        )
        self.assertTrue(result.success)
        self.assertEqual(Path(path).read_text(), "new content")

    async def test_write_empty_content(self):
        path = os.path.join(self.tmpdir.name, "empty.txt")
        result = await self.tool.execute(operation="write", path=path, content="")
        self.assertTrue(result.success)
        self.assertEqual(Path(path).read_text(), "")


class TestFileOpsPatch(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = FileOpsTool()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_patch_file(self):
        path = os.path.join(self.tmpdir.name, "patch_me.txt")
        Path(path).write_text("hello world")
        result = await self.tool.execute(
            operation="patch", path=path, old_string="world", new_string="there"
        )
        self.assertTrue(result.success)
        self.assertEqual(Path(path).read_text(), "hello there")

    async def test_patch_not_found(self):
        path = os.path.join(self.tmpdir.name, "patch_me.txt")
        Path(path).write_text("hello world")
        result = await self.tool.execute(
            operation="patch", path=path, old_string="nonexistent", new_string="foo"
        )
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())

    async def test_patch_nonexistent_file(self):
        result = await self.tool.execute(
            operation="patch", path="/nonexistent.txt", old_string="a", new_string="b"
        )
        self.assertFalse(result.success)

    async def test_patch_multiple_occurrences_single(self):
        path = os.path.join(self.tmpdir.name, "multi.txt")
        Path(path).write_text("foo bar foo bar")
        result = await self.tool.execute(
            operation="patch", path=path, old_string="foo", new_string="baz"
        )
        self.assertTrue(result.success)
        self.assertEqual(Path(path).read_text(), "baz bar foo bar")

    async def test_patch_replace_all(self):
        path = os.path.join(self.tmpdir.name, "multi.txt")
        Path(path).write_text("foo bar foo bar")
        result = await self.tool.execute(
            operation="patch",
            path=path,
            old_string="foo",
            new_string="baz",
            replace_all=True,
        )
        self.assertTrue(result.success)
        self.assertEqual(Path(path).read_text(), "baz bar baz bar")


class TestFileOpsSearch(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = FileOpsTool()
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_search_in_file(self):
        path = os.path.join(self.tmpdir.name, "search.txt")
        Path(path).write_text("hello world\nfoo bar\nhello again\n")
        result = await self.tool.execute(operation="search", path=path, pattern="hello")
        self.assertTrue(result.success)
        self.assertIn("hello world", result.content)
        self.assertIn("hello again", result.content)

    async def test_search_in_directory(self):
        path = os.path.join(self.tmpdir.name, "dir1")
        os.makedirs(path)
        Path(os.path.join(path, "a.txt")).write_text("apple banana")
        Path(os.path.join(path, "b.txt")).write_text("cherry date")

        result = await self.tool.execute(operation="search", path=path, pattern="apple")
        self.assertTrue(result.success)
        self.assertIn("apple", result.content)

    async def test_search_no_matches(self):
        path = os.path.join(self.tmpdir.name, "search.txt")
        Path(path).write_text("no match here")
        result = await self.tool.execute(
            operation="search", path=path, pattern="xyz_not_found"
        )
        self.assertTrue(result.success)
        self.assertIn("no match", result.content.lower())

    async def test_search_nonexistent_path(self):
        result = await self.tool.execute(
            operation="search", path="/nonexistent_dir", pattern="test"
        )
        self.assertFalse(result.success)

    async def test_search_invalid_regex(self):
        path = os.path.join(self.tmpdir.name, "search.txt")
        Path(path).write_text("test content")
        result = await self.tool.execute(
            operation="search", path=path, pattern="[invalid"
        )
        self.assertFalse(result.success)
        self.assertIn("invalid", result.error.lower())

    async def test_search_match_count(self):
        path = os.path.join(self.tmpdir.name, "count.txt")
        Path(path).write_text("aaa\nbbb\naaa\naaa\n")
        result = await self.tool.execute(operation="search", path=path, pattern="aaa")
        self.assertTrue(result.success)
        self.assertEqual(result.metadata.get("match_count"), 3)


class TestFileOpsUnknownOperation(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = FileOpsTool()

    async def test_unknown_operation(self):
        result = await self.tool.execute(operation="delete", path="/tmp/test.txt")
        self.assertFalse(result.success)
        self.assertIn("unknown", result.error.lower())


class TestWorkspaceFileOperations(IsolatedAsyncioTestCase):
    def setUp(self):
        from sanityops_agent.tools.context import (
            ToolContext,
            ToolInvocation,
            current_tool_invocation,
        )

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name) / "work"
        self.root.mkdir()
        self.context = ToolContext()
        self.context.set("workdir", self.root)
        self.tool = FileOpsTool(self.context)
        token = current_tool_invocation.set(
            ToolInvocation(agent=None, allowed_tools=frozenset())
        )
        self.addCleanup(current_tool_invocation.reset, token)

    async def test_permissions_and_scope(self):
        outside = self.root.parent / "outside"
        outside.write_text("dummy")
        (self.root / "link").symlink_to(outside)
        (self.root / "directory_link").symlink_to(
            self.root.parent, target_is_directory=True
        )
        for path in [str(outside), "../outside", "link", "directory_link/outside"]:
            result = await self.tool.execute(operation="read", path=path)
            self.assertFalse(result.success, path)
        result = await self.tool.execute(
            operation="write", path=str(self.root / "blocked"), content="no"
        )
        self.assertFalse(result.success)
        self.assertFalse((self.root / "blocked").exists())
        self.context.set("allow_file_writes", True)
        for path in ["link", "directory_link/outside"]:
            result = await self.tool.execute(
                operation="write", path=path, content="other"
            )
            self.assertFalse(result.success)
        self.assertEqual(outside.read_text(), "dummy")
        self.context.remove("workdir")
        result = await self.tool.execute(operation="read", path=str(outside))
        self.assertFalse(result.success)

    async def test_validate_before_writing_and_preserve_permissions(self):
        self.context.set("allow_file_writes", True)
        target = self.root / "existing"
        target.write_text("aaa")
        target.chmod(0o640)
        for arguments in [
            {"operation": "write"},
            {"operation": "write", "content": False},
            {"operation": "patch", "old_string": "a"},
            {"operation": "patch", "old_string": "", "new_string": "x"},
            {
                "operation": "patch",
                "old_string": "a",
                "new_string": "",
                "replace_all": 1,
            },
        ]:
            result = await self.tool.execute(path=str(target), **arguments)
            self.assertFalse(result.success, arguments)
            self.assertEqual(target.read_text(), "aaa")
        result = await self.tool.execute(
            operation="write", path=str(self.root / "newdir" / "f")
        )
        self.assertFalse(result.success)
        self.assertFalse((self.root / "newdir").exists())
        before = target.stat()
        result = await self.tool.execute(
            operation="patch", path="existing", old_string="a", new_string=""
        )
        self.assertTrue(result.success, result.error)
        self.assertEqual(target.read_text(), "aa")
        self.assertNotEqual(target.stat().st_ino, before.st_ino)
        self.assertEqual(target.stat().st_mode, before.st_mode)

    async def test_schema_validates_per_operation(self):
        from sanityops_agent.tools.registry import ToolRegistry

        registry = ToolRegistry(self.context)
        registry.register(self.tool)
        for arguments in [
            dict(operation="write"),
            dict(operation="patch", old_string="", new_string=""),
        ]:
            self.assertIsNotNone(
                registry.validate_input("file_ops", {"path": "f", **arguments})
            )
        self.assertIsNone(
            registry.validate_input(
                "file_ops", {"path": "f", "operation": "write", "content": ""}
            )
        )

    async def test_failed_atomic_replace_preserves_original(self):
        from unittest.mock import patch

        self.context.set("allow_file_writes", True)
        fixture = self.root / "fixture"
        fixture.write_text("first")
        fixture.chmod(0o640)
        original = fixture.stat()
        with patch("os.replace", side_effect=OSError("fixture failure")):
            result = await self.tool.execute(
                operation="write", path="fixture", content="second"
            )
        self.assertFalse(result.success)
        self.assertEqual(fixture.read_text(), "first")
        self.assertEqual(fixture.stat().st_mode, original.st_mode)
        self.assertEqual(list(self.root.iterdir()), [fixture])

    async def test_independent_tools_patch_same_file(self):
        import asyncio

        self.context.set("allow_file_writes", True)
        fixture = self.root / "fixture"
        fixture.write_text("one two")
        other = FileOpsTool(self.context)
        outcomes = await asyncio.gather(
            self.tool.execute(
                operation="patch", path="fixture", old_string="one", new_string="1"
            ),
            other.execute(
                operation="patch", path="fixture", old_string="two", new_string="2"
            ),
        )
        self.assertTrue(all(result.success for result in outcomes))
        self.assertEqual(fixture.read_text(), "1 2")

    async def test_file_and_search_limits(self):
        from unittest.mock import patch

        from sanityops_agent.tools import workspace

        (self.root / "fixture").write_text("alpha\nalpha\nalpha\n")
        with patch.object(workspace, "MAX_MATCHES", 2):
            result = await self.tool.execute(
                operation="search", path=".", pattern="alpha"
            )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.metadata["match_count"], 2)
        self.assertTrue(result.metadata["truncated"])
        with patch.object(workspace, "MAX_OUTPUT_CHARS", 7):
            result = await self.tool.execute(operation="read", path="fixture")
        self.assertLessEqual(len(result.content), 7)
        self.assertTrue(result.metadata["truncated"])

    async def test_open_uses_pinned_directory_and_nofollow(self):
        from unittest.mock import patch

        (self.root / "fixture").write_text("test")
        with patch("os.open", wraps=os.open) as opened:
            result = await self.tool.execute(operation="read", path="fixture")
        self.assertTrue(result.success)
        final = opened.call_args
        self.assertEqual(final.args[0], "fixture")
        self.assertTrue(final.args[1] & os.O_NOFOLLOW)
        self.assertIsInstance(final.kwargs["dir_fd"], int)

    async def test_search_limits_regex_time(self):
        from unittest.mock import patch

        (self.root / "fixture").write_text("dummy")
        with patch("regex.compile") as compile_pattern:
            compile_pattern.return_value.search.side_effect = TimeoutError
            result = await self.tool.execute(
                operation="search", path="fixture", pattern="dummy"
            )
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)
        self.assertGreater(
            compile_pattern.return_value.search.call_args.kwargs["timeout"], 0
        )

    async def test_cancel_during_atomic_write_keeps_original(self):
        import asyncio
        from unittest.mock import patch

        self.context.set("allow_file_writes", True)
        path = self.root / "fixture"
        path.write_text("original")
        with patch("asyncio.sleep", side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await self.tool.execute(
                    operation="write", path="fixture", content="new"
                )
        self.assertEqual(path.read_text(), "original")
        self.assertEqual(list(self.root.iterdir()), [path])

    async def test_refuses_oversize_read_and_patch_expansion(self):
        from unittest.mock import patch

        from sanityops_agent.tools import workspace

        self.context.set("allow_file_writes", True)
        path = self.root / "fixture"
        path.write_text("abcabc")
        with patch.object(workspace, "MAX_FILE_BYTES", 4):
            self.assertFalse(
                (await self.tool.execute(operation="read", path="fixture")).success
            )
        with patch.object(workspace, "MAX_FILE_BYTES", 10):
            result = await self.tool.execute(
                operation="patch",
                path="fixture",
                old_string="abc",
                new_string="longer",
                replace_all=True,
            )
            self.assertFalse(result.success)
        self.assertEqual(path.read_text(), "abcabc")


if __name__ == "__main__":
    main()
