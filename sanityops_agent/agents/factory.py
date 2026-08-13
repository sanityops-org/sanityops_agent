"""AgentFactory - creates ParentAgent and SubAgent instances."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sanityops_agent.agents.parent import ParentAgent
from sanityops_agent.agents.sub import SubAgent
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
        config: AgentConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        hook_executor: HookExecutor | None = None,
        inject_skill_info: bool = True,
    ):
        self.provider = provider
        self.config = config or AgentConfig()
        # Use explicit None check to avoid creating new registry when empty
        self.tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self.hook_executor = hook_executor
        self.inject_skill_info = inject_skill_info

        # Inject self into ToolContext so TaskTool can create sub-agents
        self.tool_registry.context.set("agent_factory", self)

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
        if not skills:
            return ""

        lines = [
            "",
            "# Available Skills",
            "",
            "You have access to the following skills:",
            "",
        ]

        for skill in skills:
            name = skill.get("name", "")
            description = skill.get("description", "No description")
            lines.append(f"- **{name}**: {description}")
            lines.append(f"  Usage: `skill(operation=\"load\", name=\"{name}\")`")
            lines.append("")

        lines.append(
            "Use `skill(operation=\"load\", name=\"<skill_name>\")` to load a skill's full documentation."
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

    def create_parent_agent(self, system_prompt: str = "", **overrides) -> ParentAgent:
        """Create a ParentAgent with full tool access.

        Skill metadata is automatically injected into the system prompt
        when inject_skill_info=True (default). Set inject_skill_info=False
        to disable this behavior.

        Args:
            system_prompt: System prompt for the
            **overrides: Override any AgentConfig fields.

        Returns:
            A configured ParentAgent instance.
        """
        # Inject skill metadata into system prompt
        enhanced_prompt = self._inject_skill_prompt(system_prompt)

        # Start from the factory's config, then apply system_prompt and overrides
        cfg = AgentConfig(**{
            k: v
            for k, v in self.config.__dict__.items()
            if k in AgentConfig.__dataclass_fields__
        })
        cfg.system_prompt = enhanced_prompt
        for k, v in overrides.items():
            if k in AgentConfig.__dataclass_fields__:
                setattr(cfg, k, v)
        return ParentAgent(
            cfg,
            self.provider,
            self.tool_registry,
            self.hook_executor,
        )

    def create_sub_agent(self, system_prompt: str = "", **overrides) -> SubAgent:
        """Create a SubAgent with filtered tool access.

        Sub-agents have tighter limits (max_loops=10, timeout=120s) and
        cannot access tools tagged with 'parent_only'.

        Args:
            system_prompt: System prompt for the
            **overrides: Override any AgentConfig fields.

        Returns:
            A configured SubAgent instance.
        """
        # Build sub-agent config with tighter limits
        sub_overrides: dict = {
            "max_loops": 10,
            "total_timeout": 120,
            "system_prompt": system_prompt,
        }
        # Apply user overrides (can override the defaults above)
        for k, v in overrides.items():
            if k in AgentConfig.__dataclass_fields__:
                sub_overrides[k] = v

        cfg = AgentConfig(**sub_overrides)

        # Filter out parent_only tools for sub-agents
        sub_registry = self._filter_registry(exclude_tags=["parent_only"])

        return SubAgent(
            cfg,
            self.provider,
            sub_registry,
            self.hook_executor,
        )

    def _filter_registry(self, exclude_tags: list[str]) -> ToolRegistry:
        """Create a new registry excluding tools with the given tags."""
        new_registry = ToolRegistry(context=self.tool_registry.context)
        for tool in self.tool_registry.list_all():
            if not any(tag in exclude_tags for tag in tool.tags):
                new_registry.register(tool)
        return new_registry
