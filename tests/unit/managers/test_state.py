"""Tests for agent/managers/state.py - StateManager."""

import json
import os

import pytest

from sanityops_agent.core.context import AgentContext
from sanityops_agent.core.message import (
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.managers.state import StateManager


@pytest.fixture
def tmp_state_path(tmp_path):
    return str(tmp_path / "agent_state.json")


class TestStateManagerSave:
    """Test saving context to JSON."""

    @pytest.mark.asyncio
    async def test_save_writes_json_file(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        await manager.save(ctx)
        assert os.path.exists(tmp_state_path)

    @pytest.mark.asyncio
    async def test_save_writes_valid_json(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        await manager.save(ctx)
        # Should not raise
        data = json.loads(open(tmp_state_path).read())
        assert "messages" in data
        assert "total_tokens" in data
        assert "has_compacted" in data
        assert "last_summary" in data

    @pytest.mark.asyncio
    async def test_save_with_messages(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        msg = Message.from_text(Role.USER, "hello world")
        ctx.messages.append(msg)
        ctx.total_tokens = 100
        await manager.save(ctx)

        data = json.loads(open(tmp_state_path).read())
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["blocks"] == [
            {"type": "text", "text": "hello world"}
        ]
        assert data["total_tokens"] == 100

    @pytest.mark.asyncio
    async def test_save_with_compaction_state(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext(
            has_compacted=True,
            last_summary="Conversation summarized",
        )
        await manager.save(ctx)

        data = json.loads(open(tmp_state_path).read())
        assert data["has_compacted"] is True
        assert data["last_summary"] == "Conversation summarized"


class TestStateManagerLoad:
    """Test loading context from JSON."""

    @pytest.mark.asyncio
    async def test_load_file_not_found(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        with pytest.raises(FileNotFoundError, match="State file not found"):
            await manager.load()

    @pytest.mark.asyncio
    async def test_load_reconstructs_context(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        # First save
        ctx = AgentContext(
            messages=[Message.from_text(Role.USER, "test message")],
            total_tokens=50,
            has_compacted=True,
            last_summary="summary text",
        )
        await manager.save(ctx)

        # Then load
        loaded = await manager.load()
        assert len(loaded.messages) == 1
        assert loaded.messages[0].role == Role.USER
        assert loaded.messages[0].content[0].text == "test message"
        assert loaded.total_tokens == 50
        assert loaded.has_compacted is True
        assert loaded.last_summary == "summary text"

    @pytest.mark.asyncio
    async def test_load_multiple_messages(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(Message.from_text(Role.USER, "hello"))
        ctx.messages.append(Message.from_text(Role.ASSISTANT, "hi there"))
        ctx.total_tokens = 200
        await manager.save(ctx)

        loaded = await manager.load()
        assert len(loaded.messages) == 2
        assert loaded.messages[0].role == Role.USER
        assert loaded.messages[1].role == Role.ASSISTANT

    @pytest.mark.asyncio
    async def test_load_with_provider(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[TextBlock(text="response")],
                provider="openai",
            )
        )
        await manager.save(ctx)

        loaded = await manager.load()
        assert loaded.messages[0].provider == "openai"

    @pytest.mark.asyncio
    async def test_load_empty_messages(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        data = {
            "messages": [],
            "total_tokens": 0,
            "has_compacted": False,
            "last_summary": "",
        }
        open(tmp_state_path, "w").write(json.dumps(data))

        loaded = await manager.load()
        assert loaded.messages == []
        assert loaded.total_tokens == 0


class TestStateManagerRoundTrip:
    """Test save-then-load round trips."""

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_state(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        original = AgentContext(
            messages=[Message.from_text(Role.USER, "query")],
            total_tokens=75,
            has_compacted=False,
            last_summary="",
        )
        await manager.save(original)
        loaded = await manager.load()

        assert len(loaded.messages) == len(original.messages)
        assert loaded.messages[0].role == original.messages[0].role
        assert loaded.total_tokens == original.total_tokens
        assert loaded.has_compacted == original.has_compacted
        assert loaded.last_summary == original.last_summary


class TestStateManagerToolBlocks:
    """Tool blocks must survive a save/load cycle.

    Dropping them left the restored history with a tool_result rendered as
    plain prose and no matching tool_use, which providers reject.
    """

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_tool_use_block(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[
                    TextBlock(text="Let me check."),
                    ToolUseBlock(
                        tool_call_id="call_1",
                        tool_name="search",
                        tool_input={"q": "weather", "limit": 3},
                    ),
                ],
            )
        )

        await manager.save(ctx)
        loaded = await manager.load()

        blocks = loaded.messages[0].content
        assert [type(b).__name__ for b in blocks] == ["TextBlock", "ToolUseBlock"]
        tool_use = blocks[1]
        assert tool_use.tool_call_id == "call_1"
        assert tool_use.tool_name == "search"
        assert tool_use.tool_input == {"q": "weather", "limit": 3}

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_tool_result_block(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[
                    ToolResultBlock(
                        tool_call_id="call_1",
                        tool_result_content="Sunny, 75F",
                        is_error=True,
                        history_key="hist_1",
                    )
                ],
            )
        )

        await manager.save(ctx)
        loaded = await manager.load()

        block = loaded.messages[0].content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.tool_call_id == "call_1"
        assert block.tool_result_content == "Sunny, 75F"
        assert block.is_error is True
        assert block.history_key == "hist_1"

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_thinking_block(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[ThinkingBlock(thinking="Weighing options.")],
            )
        )

        await manager.save(ctx)
        loaded = await manager.load()

        block = loaded.messages[0].content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.thinking == "Weighing options."

    @pytest.mark.asyncio
    async def test_roundtrip_keeps_tool_pairing(self, tmp_state_path):
        """A saved tool exchange reloads with both halves intact."""
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[
                    ToolUseBlock(
                        tool_call_id="c9", tool_name="calc", tool_input={"e": "1+1"}
                    )
                ],
            )
        )
        ctx.messages.append(
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c9", tool_result_content="2")],
            )
        )

        await manager.save(ctx)
        loaded = await manager.load()

        use_ids = {
            b.tool_call_id
            for m in loaded.messages
            for b in m.content
            if isinstance(b, ToolUseBlock)
        }
        result_ids = {
            b.tool_call_id
            for m in loaded.messages
            for b in m.content
            if isinstance(b, ToolResultBlock)
        }
        assert use_ids == result_ids == {"c9"}

    @pytest.mark.asyncio
    async def test_loads_legacy_text_only_format(self, tmp_state_path):
        """Files written by the old text-only serializer still load."""
        manager = StateManager(state_path=tmp_state_path)
        legacy = {
            "messages": [
                {"role": "user", "text": "old message", "provider": None},
            ],
            "total_tokens": 10,
            "has_compacted": False,
            "last_summary": "",
        }
        open(tmp_state_path, "w").write(json.dumps(legacy))

        loaded = await manager.load()

        assert len(loaded.messages) == 1
        assert loaded.messages[0].role == Role.USER
        assert loaded.messages[0].content[0].text == "old message"


class TestStateManagerUsage:
    """Token usage must survive a save/load cycle, or the restored context's
    token accounting restarts from zero mid-conversation."""

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_usage_tokens(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=[TextBlock(text="response")],
                raw_response={"usage": {"total_tokens": 4321}},
            )
        )

        await manager.save(ctx)
        loaded = await manager.load()

        raw = loaded.messages[0].raw_response
        assert raw["usage"]["total_tokens"] == 4321

    @pytest.mark.asyncio
    async def test_messages_without_usage_load_cleanly(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(Message.from_text(Role.USER, "hi"))

        await manager.save(ctx)
        loaded = await manager.load()

        assert loaded.messages[0].raw_response == {}


class TestStateManagerRobustness:
    """Writes are atomic and malformed files raise a descriptive error."""

    @pytest.mark.asyncio
    async def test_save_is_atomic_no_tmp_leftover(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(Message.from_text(Role.USER, "hi"))

        await manager.save(ctx)

        assert os.path.exists(tmp_state_path)
        # The temp file used for the atomic rename must be gone.
        leftovers = [
            f for f in os.listdir(os.path.dirname(tmp_state_path)) if f.endswith(".tmp")
        ]
        assert leftovers == []

    @pytest.mark.asyncio
    async def test_load_malformed_json_raises_descriptive(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        open(tmp_state_path, "w").write("{not valid json")

        with pytest.raises(ValueError, match="corrupt|invalid"):
            await manager.load()

    @pytest.mark.asyncio
    async def test_load_wrong_shape_raises_descriptive(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        # messages is not a list; role keys are missing.
        open(tmp_state_path, "w").write(json.dumps({"messages": [{"nope": 1}]}))

        with pytest.raises((ValueError, KeyError)):
            await manager.load()

    @pytest.mark.asyncio
    async def test_save_write_failure_leaves_previous_state_intact(
        self, tmp_state_path, monkeypatch
    ):
        """A crash mid-write must not corrupt the previous good state."""
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(Message.from_text(Role.USER, "good state"))
        await manager.save(ctx)
        original = open(tmp_state_path).read()

        import sanityops_agent.managers.state as state_mod

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(state_mod.os, "replace", boom)

        with pytest.raises(OSError):
            await manager.save(ctx)

        # Previous state file untouched.
        assert open(tmp_state_path).read() == original


class TestStateManagerValidation:
    """Malformed state files fail loudly; unknown block types are visible."""

    @pytest.mark.asyncio
    async def test_invalid_role_raises_descriptive(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        open(tmp_state_path, "w").write(
            json.dumps({"messages": [{"role": "wizard", "text": "hi"}]})
        )

        with pytest.raises(ValueError, match="invalid or missing role.*wizard"):
            await manager.load()

    @pytest.mark.asyncio
    async def test_missing_role_raises_descriptive(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        open(tmp_state_path, "w").write(json.dumps({"messages": [{"text": "hi"}]}))

        with pytest.raises(ValueError, match="invalid or missing role"):
            await manager.load()

    @pytest.mark.asyncio
    async def test_unknown_block_type_kept_as_text_with_warning(
        self, tmp_state_path, caplog
    ):
        manager = StateManager(state_path=tmp_state_path)
        data = {
            "messages": [
                {
                    "role": "assistant",
                    "blocks": [{"type": "brand_new_block", "payload": "x"}],
                }
            ],
            "total_tokens": 0,
        }
        open(tmp_state_path, "w").write(json.dumps(data))

        with caplog.at_level("WARNING", logger="sanityops_agent.managers.state"):
            loaded = await manager.load()

        assert any("brand_new_block" in r.message for r in caplog.records)
        text = loaded.messages[0].content[0].text
        assert "brand_new_block" in text and "x" in text

    @pytest.mark.asyncio
    async def test_state_file_owner_only_permissions(self, tmp_state_path):
        import stat

        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext()
        ctx.messages.append(Message.from_text(Role.USER, "secret"))

        await manager.save(ctx)
        mode = stat.S_IMODE(os.stat(tmp_state_path).st_mode)
        assert mode == 0o600

    @pytest.mark.asyncio
    async def test_unsigned_thinking_tool_turn_cannot_resume(self, tmp_state_path):
        manager = StateManager(state_path=tmp_state_path)
        ctx = AgentContext(
            messages=[
                Message.from_text(Role.USER, "Check"),
                Message(
                    role=Role.ASSISTANT,
                    content=[
                        ThinkingBlock(thinking="Reasoning without signature"),
                        ToolUseBlock("c1", "calc", {"e": "1+1"}),
                    ],
                ),
            ]
        )
        await manager.save(ctx)
        with pytest.raises(ValueError, match="Cannot resume an unsigned"):
            await manager.load()


@pytest.mark.asyncio
async def test_load_accepts_legacy_thinking_signature_key(tmp_state_path):
    """State files written before the signature rename still reload intact."""
    with open(tmp_state_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "provider": "anthropic",
                        "blocks": [
                            {
                                "type": "thinking",
                                "thinking": "Reasoning",
                                "thinking_signature": "legacy-sig",
                            }
                        ],
                    }
                ]
            },
            handle,
        )

    loaded = await StateManager(state_path=tmp_state_path).load()

    block = loaded.messages[0].content[0]
    assert isinstance(block, ThinkingBlock)
    assert block.signature == "legacy-sig"
