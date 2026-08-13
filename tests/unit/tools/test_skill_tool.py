"""Unit tests for sanityops_agent.tools.builtins.skill - SkillTool"""
from __future__ import annotations

import pytest

from sanityops_agent.managers.skill import SkillManager
from sanityops_agent.tools.builtins.skill import SkillTool
from sanityops_agent.tools.context import ToolContext


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory with test skill directories."""
    d = tmp_path / "skills"
    d.mkdir()
    # Skill with frontmatter - now uses directory structure
    code_review_dir = d / "code_review"
    code_review_dir.mkdir()
    (code_review_dir / "SKILL.md").write_text(
        "---\ndescription: Review code for issues\ncategory: coding\n---\n# Code Review\n\nReview the code..."
    )
    data_analysis_dir = d / "data_analysis"
    data_analysis_dir.mkdir()
    (data_analysis_dir / "SKILL.md").write_text(
        "---\ndescription: Analyze data patterns\ncategory: analysis\n---\n# Data Analysis\n\nAnalyze patterns..."
    )
    return d


class TestSkillToolWithoutManager:
    """Test SkillTool when skill_manager is not configured."""

    @pytest.mark.asyncio
    async def test_execute_without_manager_returns_error(self):
        tool = SkillTool()
        result = await tool.execute(operation="list")
        assert result.success is False
        assert "skill_manager is not configured" in result.content
        assert result.error == "skill_manager is not configured"


class TestSkillToolList:
    """Test SkillTool list operation."""

    @pytest.mark.asyncio
    async def test_list_with_skills(self, skills_dir):
        ctx = ToolContext()
        ctx.set("skill_manager", SkillManager(skills_dir=str(skills_dir)))
        tool = SkillTool(context=ctx)

        result = await tool.execute(operation="list")
        assert result.success is True
        assert "code_review" in result.content
        assert "data_analysis" in result.content

    @pytest.mark.asyncio
    async def test_list_no_skills(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        ctx = ToolContext()
        ctx.set("skill_manager", SkillManager(skills_dir=str(empty_dir)))
        tool = SkillTool(context=ctx)

        result = await tool.execute(operation="list")
        assert result.success is True
        assert "No skills available" in result.content


class TestSkillToolLoad:
    """Test SkillTool load operation."""

    @pytest.mark.asyncio
    async def test_load_success(self, skills_dir):
        ctx = ToolContext()
        ctx.set("skill_manager", SkillManager(skills_dir=str(skills_dir)))
        tool = SkillTool(context=ctx)

        result = await tool.execute(operation="load", name="code_review")
        assert result.success is True
        assert "Code Review" in result.content
        assert "---" in result.content

    @pytest.mark.asyncio
    async def test_load_nonexistent_skill(self, skills_dir):
        ctx = ToolContext()
        ctx.set("skill_manager", SkillManager(skills_dir=str(skills_dir)))
        tool = SkillTool(context=ctx)

        result = await tool.execute(operation="load", name="nonexistent")
        assert result.success is False
        assert "not found" in result.content

    @pytest.mark.asyncio
    async def test_load_without_name(self, skills_dir):
        ctx = ToolContext()
        ctx.set("skill_manager", SkillManager(skills_dir=str(skills_dir)))
        tool = SkillTool(context=ctx)

        result = await tool.execute(operation="load")
        assert result.success is False
        assert "name is required" in result.content


class TestSkillToolUnknownOperation:
    """Test SkillTool with unknown operation."""

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_error(self, skills_dir):
        ctx = ToolContext()
        ctx.set("skill_manager", SkillManager(skills_dir=str(skills_dir)))
        tool = SkillTool(context=ctx)

        result = await tool.execute(operation="delete")
        assert result.success is False
        assert "unknown operation" in result.content


class TestSkillToolDefinition:
    """Test SkillTool metadata."""

    def test_name(self):
        assert SkillTool.name == "skill"

    def test_description(self):
        # Check for partial match since description may vary
        assert "skill" in SkillTool.description.lower()

    def test_tags(self):
        assert "skill" in SkillTool.tags

    def test_to_tool_definition(self):
        tool = SkillTool()
        definition = tool.to_tool_definition()
        assert definition.name == "skill"
        assert "operation" in definition.input_schema["properties"]


class TestSkillToolRun:
    """Test SkillTool via the run template method."""

    @pytest.mark.asyncio
    async def test_run_adds_metadata(self, skills_dir):
        ctx = ToolContext()
        ctx.set("skill_manager", SkillManager(skills_dir=str(skills_dir)))
        tool = SkillTool(context=ctx)

        result = await tool.run(operation="list")
        assert result.success is True
        assert "execution_time" in result.metadata
        assert result.metadata["tool_name"] == "skill"
