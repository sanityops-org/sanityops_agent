"""Integration tests for state persistence."""

import pytest

from sanityops_agent.core.context import AgentContext
from sanityops_agent.core.message import (
    Message,
    RedactedThinkingBlock,
    Role,
    ThinkingBlock,
)
from sanityops_agent.managers.state import StateManager
from sanityops_agent.managers.todo import TodoManager


class TestStatePersist:
    """Test state save/load cycle."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        """State should survive a save/load cycle."""
        state_file = str(tmp_path / "state.json")
        manager = StateManager(state_path=state_file)

        ctx = AgentContext()
        await ctx.add_message(Message.from_text(Role.USER, "Hello"))
        await ctx.add_message(Message.from_text(Role.ASSISTANT, "Hi there!"))
        ctx.total_tokens = 100

        await manager.save(ctx)

        # Load into a new context
        manager2 = StateManager(state_path=state_file)
        loaded_ctx = await manager2.load()

        assert len(loaded_ctx.messages) == 2
        assert loaded_ctx.total_tokens == 100
        assert loaded_ctx.messages[0].role == Role.USER

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises(self, tmp_path):
        """Loading from a non-existent file should raise FileNotFoundError."""
        state_file = str(tmp_path / "nonexistent.json")
        manager = StateManager(state_path=state_file)
        with pytest.raises(FileNotFoundError):
            await manager.load()

    @pytest.mark.asyncio
    async def test_save_and_load_empty_context(self, tmp_path):
        """Save and load an empty context."""
        state_file = str(tmp_path / "empty.json")
        manager = StateManager(state_path=state_file)

        ctx = AgentContext()
        await manager.save(ctx)

        manager2 = StateManager(state_path=state_file)
        loaded_ctx = await manager2.load()

        assert len(loaded_ctx.messages) == 0


async def test_state_keeps_opaque_thinking_fields(tmp_path):
    manager = StateManager(tmp_path / "state.json")
    content = [
        ThinkingBlock("thinking", signature="fixture-signature"),
        RedactedThinkingBlock("fixture-block"),
    ]
    await manager.save(AgentContext(messages=[Message(Role.ASSISTANT, content)]))
    assert (await manager.load()).messages[0].content == content


async def test_managed_state_cannot_be_loaded_for_a_different_session(tmp_path):
    tmp_path.chmod(0o700)
    state_path = tmp_path / "managed.json"
    first = StateManager(state_path, session_id="one")
    await first.save(
        AgentContext(messages=[Message.from_text(Role.USER, "Private turn")])
    )

    with pytest.raises(PermissionError, match="session"):
        await StateManager(state_path, session_id="two").load()


def test_todo_persist_creates_private_state(tmp_path):
    state_path = tmp_path / "todo.json"
    TodoManager(state_path).create("Test item")
    assert state_path.stat().st_mode & 0o777 == 0o600
