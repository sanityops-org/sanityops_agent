"""AgentFactory - creates ParentAgent and SubAgent instances."""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import TYPE_CHECKING, Any

from sanityops_agent.agents.parent import ParentAgent
from sanityops_agent.agents.sub import SubAgent
from sanityops_agent.config import AgentConfig as EnvironmentAgentConfig
from sanityops_agent.core.agent import AgentConfig
from sanityops_agent.hooks.base import HookExecutor
from sanityops_agent.llm.base import LLMProvider
from sanityops_agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from sanityops_agent.managers.skill import SkillManager


class AgentFactory:
    """Factory for creating agent instances.

    Provides:
    - create_parent_agent: full-featured parent agent with all tools
    - create_sub_agent: lightweight sub-agent with filtered tools (no parent_only)
    """

    def __init__(
        self,
        provider: LLMProvider,
        config: AgentConfig | EnvironmentAgentConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        hook_executor: HookExecutor | None = None,
        inject_skill_info: bool = True,
    ):
        self.provider = provider
        self.config = (
            config.to_runtime_config()
            if isinstance(config, EnvironmentAgentConfig)
            else config or AgentConfig()
        )
        # Use explicit None check to avoid creating new registry when empty
        self.tool_registry = (
            tool_registry if tool_registry is not None else ToolRegistry()
        )
        self.hook_executor = hook_executor
        self.inject_skill_info = inject_skill_info

        # Intentionally store a strong reference in ToolContext so TaskTool can
        # still spawn sub-agents even if the caller only retains the parent agent.
        # Call close() if you need to break the factory <-> registry cycle early.
        self.tool_registry.context.set("agent_factory", self)

    def close(self) -> None:
        """Break the reference cycle with ToolContext when disposing the factory."""
        if self.tool_registry.context.get("agent_factory") is self:
            self.tool_registry.context.remove("agent_factory")

    @staticmethod
    def _build_skill_prompt(skill_manager: SkillManager | None) -> str:
        """Build a Markdown-formatted skill section for system prompts.

        Args:
            skill_manager: SkillManager instance to get skill info from.
                           If None or has no skills, returns empty string.

        Returns:
            Markdown-formatted string describing available skills.
            Empty string if no skills available.
        """
        if skill_manager is None:
            return ""

        skills = skill_manager.describe_available()
        valid_skills = [
            skill
            for skill in skills
            if isinstance(skill.get("name"), str) and skill["name"].strip()
        ]
        if not valid_skills:
            return ""

        lines = [
            "",
            "# Available Skills",
            "",
            "You have access to the following skills:",
            "",
        ]

        for skill in valid_skills:
            name = skill["name"].strip()
            description = skill.get("description", "No description")
            lines.append(f"- **{name}**: {description}")
            lines.append(f'  Usage: `skill(operation="load", name="{name}")`')
            lines.append("")

        lines.append(
            'Use `skill(operation="load", name="<skill_name>")` to load a skill\'s full documentation.'
        )
        lines.append("")

        return "\n".join(lines)

    def _inject_skill_prompt(self, system_prompt: str) -> str:
        """Inject skill information into system prompt if enabled.

        Args:
            system_prompt: Original system prompt from user.

        Returns:
            Enhanced system prompt with skill information appended.
        """
        if not self.inject_skill_info:
            return system_prompt

        skill_manager = self.tool_registry.context.skill_manager
        skill_section = self._build_skill_prompt(skill_manager)

        if not skill_section:
            return system_prompt

        return system_prompt + skill_section

    def _build_config(
        self,
        system_prompt: str,
        defaults: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> AgentConfig:
        """Clone the factory's AgentConfig and apply role defaults and overrides."""
        config_fields = fields(AgentConfig)
        valid_fields = {f.name for f in config_fields}
        config_values: dict[str, Any] = {
            f.name: copy.deepcopy(getattr(self.config, f.name)) for f in config_fields
        }
        if defaults:
            config_values.update(defaults)
        config_values["system_prompt"] = system_prompt
        for k, v in overrides.items():
            if k in valid_fields:
                config_values[k] = v
        return AgentConfig(**config_values)

    def create_parent_agent(self, system_prompt: str = "", **overrides) -> ParentAgent:
        """Create a ParentAgent with full tool access.

        Skill metadata is automatically injected into the system prompt
        when inject_skill_info=True (default). Set inject_skill_info=False
        to disable this behavior.

        Args:
            system_prompt: System prompt for the parent agent.
            **overrides: Override any AgentConfig fields.

        Returns:
            A configured ParentAgent instance.
        """
        # Inject skill metadata into system prompt
        enhanced_prompt = self._inject_skill_prompt(system_prompt)
        cfg = self._build_config(enhanced_prompt, **overrides)
        return ParentAgent(
            cfg,
            self.provider,
            self.tool_registry,
            self.hook_executor,
        )

    def create_sub_agent(self, system_prompt: str = "", **overrides) -> SubAgent:
        """Create a SubAgent with filtered tool access.

        Sub-agents inherit factory config with tighter loop/timeout caps
        (max_loops<=10, total_timeout<=120s unless overridden) and cannot
        access tools tagged with 'parent_only'.

        Args:
            system_prompt: System prompt for the sub-agent.
            **overrides: Override any AgentConfig fields.

        Returns:
            A configured SubAgent instance.
        """
        cfg = self._build_config(
            system_prompt,
            defaults={
                "max_loops": min(self.config.max_loops, 10),
                "total_timeout": min(self.config.total_timeout, 120),
            },
            **overrides,
        )

        # Filter out parent_only tools for sub-agents
        sub_registry = self._filter_registry(exclude_tags=["parent_only"])

        return SubAgent(
            cfg,
            self.provider,
            sub_registry,
            self.hook_executor,
        )

    def _filter_registry(self, exclude_tags: list[str]) -> ToolRegistry:
        """Create a new registry excluding tools with the given tags.

        Intentionally shares ToolContext with the parent registry so filtered
        sub-agent tools can access shared managers (e.g., skill_manager).
        """
        new_registry = ToolRegistry(context=self.tool_registry.context)
        for tool in self.tool_registry.list_all():
            tags = getattr(tool, "tags", None) or ()
            if not any(tag in exclude_tags for tag in tags):
                new_registry.register(tool)
        return new_registry
