import os
import tempfile
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, main

from sanityops_agent.tools.builtins.glob import GlobTool


class TestGlobTool(IsolatedAsyncioTestCase):
    def setUp(self):
        self.tool = GlobTool()
        self.tmpdir = tempfile.TemporaryDirectory()
        self._setup_test_files()

    def _setup_test_files(self):
        """Create a directory structure for testing."""
        # Root level files
        Path(os.path.join(self.tmpdir.name, "file1.py")).write_text("# py1")
        Path(os.path.join(self.tmpdir.name, "file2.py")).write_text("# py2")
        Path(os.path.join(self.tmpdir.name, "data.json")).write_text("{}")
        Path(os.path.join(self.tmpdir.name, "README.md")).write_text("# Readme")

        # Subdirectory
        subdir = os.path.join(self.tmpdir.name, "sub")
        os.makedirs(subdir)
        Path(os.path.join(subdir, "sub_file.py")).write_text("# sub py")
        Path(os.path.join(subdir, "config.json")).write_text("{}")

        # Nested subdirectory
        nested = os.path.join(subdir, "nested")
        os.makedirs(nested)
        Path(os.path.join(nested, "deep.txt")).write_text("deep")

    def tearDown(self):
        self.tmpdir.cleanup()

    async def test_glob_simple_pattern(self):
        result = await self.tool.execute(pattern="*.py", path=self.tmpdir.name)
        self.assertTrue(result.success)
        self.assertIn("file1.py", result.content)
        self.assertIn("file2.py", result.content)
        self.assertEqual(result.metadata.get("match_count"), 2)

    async def test_glob_no_matches(self):
        result = await self.tool.execute(pattern="*.xyz", path=self.tmpdir.name)
        self.assertTrue(result.success)
        self.assertIn("no files", result.content.lower())

    async def test_glob_recursive(self):
        result = await self.tool.execute(pattern="**/*.py", path=self.tmpdir.name)
        self.assertTrue(result.success)
        self.assertIn("file1.py", result.content)
        self.assertIn("file2.py", result.content)
        self.assertIn("sub_file.py", result.content)
        self.assertEqual(result.metadata.get("match_count"), 3)

    async def test_glob_json_files(self):
        result = await self.tool.execute(pattern="*.json", path=self.tmpdir.name)
        self.assertTrue(result.success)
        self.assertIn("data.json", result.content)
        # Should not include sub/config.json since **/ not used
        self.assertNotIn("config.json", result.content)

    async def test_glob_recursive_json(self):
        result = await self.tool.execute(pattern="**/*.json", path=self.tmpdir.name)
        self.assertTrue(result.success)
        self.assertIn("data.json", result.content)
        self.assertIn("config.json", result.content)
        self.assertEqual(result.metadata.get("match_count"), 2)

    async def test_glob_empty_pattern(self):
        result = await self.tool.execute(pattern="", path=self.tmpdir.name)
        self.assertFalse(result.success)
        self.assertIn("required", result.error.lower())

    async def test_glob_nonexistent_path(self):
        result = await self.tool.execute(pattern="*.py", path="/nonexistent_path_xyz")
        self.assertFalse(result.success)
        self.assertIn("not found", result.error.lower())

    async def test_glob_path_is_file(self):
        file_path = os.path.join(self.tmpdir.name, "file1.py")
        result = await self.tool.execute(pattern="*.py", path=file_path)
        self.assertFalse(result.success)
        self.assertIn("not a directory", result.error.lower())

    async def test_glob_with_default_path(self):
        # Should use current working directory when path is None
        result = await self.tool.execute(pattern="nonexistent_pattern_xyz.py")
        self.assertTrue(result.success)
        self.assertIn("no files", result.content.lower())

    async def test_glob_filters_only_files(self):
        # Create a directory that would match the pattern
        matching_dir = os.path.join(self.tmpdir.name, "directory.py")
        os.makedirs(matching_dir)

        result = await self.tool.execute(pattern="*.py", path=self.tmpdir.name)
        self.assertTrue(result.success)
        self.assertNotIn("directory.py", result.content)


class TestGlobWorkspace(IsolatedAsyncioTestCase):
    async def test_model_glob_stays_inside_workspace(self):
        from sanityops_agent.tools.context import (
            ToolContext,
            ToolInvocation,
            current_tool_invocation,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "workspace"
            root.mkdir()
            outside = Path(temp) / "elsewhere"
            outside.mkdir()
            (outside / "outside-fixture.txt").touch()
            (root / "inside-fixture.txt").touch()
            (root / "linked").symlink_to(outside)
            context = ToolContext()
            context.set("workdir", root)
            tool = GlobTool(context)
            token = current_tool_invocation.set(
                ToolInvocation(agent=None, allowed_tools=frozenset())
            )
            self.addCleanup(current_tool_invocation.reset, token)
            result = await tool.execute(pattern="**/*fixture.txt")
            self.assertTrue(result.success, result.error)
            self.assertIn("inside-fixture.txt", result.content)
            self.assertNotIn("outside-fixture.txt", result.content)
            self.assertFalse((await tool.execute(pattern="../elsewhere/*.txt")).success)
            self.assertFalse(
                (await tool.execute(pattern="*.txt", path=str(outside))).success
            )
            self.assertFalse(
                (await tool.execute(pattern="*.txt", path="linked")).success
            )
            context.remove("workdir")
            self.assertFalse(
                (await tool.execute(pattern="*.txt", path=str(outside))).success
            )

    async def test_entry_and_result_limits(self):
        from unittest.mock import patch

        from sanityops_agent.tools import workspace

        with tempfile.TemporaryDirectory() as temp:
            for name in ["a", "b", "c"]:
                (Path(temp) / name).touch()
            tool = GlobTool()
            with patch.object(workspace, "MAX_ENTRIES", 2):
                self.assertFalse(
                    (await tool.execute(pattern="*.txt", path=temp)).success
                )
            with patch.object(workspace, "MAX_MATCHES", 2):
                result = await tool.execute(pattern="*", path=temp)
                self.assertEqual(result.metadata["match_count"], 2)
                self.assertTrue(result.metadata["truncated"])


if __name__ == "__main__":
    main()
