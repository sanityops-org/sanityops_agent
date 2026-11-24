"""Unit tests for LLM message adapters."""

from __future__ import annotations

import json
import re

import pytest

from sanityops_agent.core.message import (
    ContentType,
    ErrorBlock,
    FinishReason,
    Message,
    Role,
    StreamChunk,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.llm.adapters.anthropic_adapter import AnthropicAdapter
from sanityops_agent.llm.adapters.base import MessageAdapter, merge_consecutive_messages
from sanityops_agent.llm.adapters.openai_adapter import OpenAIAdapter
from sanityops_agent.llm.base import LLMResponse
from sanityops_agent.llm.exceptions import (
    AuthenticationError,
    ContextWindowExceededError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)


def _parallel_tool_history() -> list[Message]:
    """The malformed history the agent loop produces for 2 parallel tool calls.

    One assistant turn with two tool_use blocks, followed by one USER message
    per tool result (see agents/parent.py:_add_tool_result). Emitted verbatim
    this is consecutive same-role messages -> provider HTTP 400.
    """
    return [
        Message(
            role=Role.ASSISTANT,
            content=[
                ToolUseBlock(tool_call_id="c1", tool_name="t", tool_input={}),
                ToolUseBlock(tool_call_id="c2", tool_name="t", tool_input={}),
            ],
        ),
        Message(
            role=Role.USER,
            content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
        ),
        Message(
            role=Role.USER,
            content=[ToolResultBlock(tool_call_id="c2", tool_result_content="r2")],
        ),
    ]


# ====================================================================
# MessageAdapter abstractness
# ====================================================================


class TestMessageAdapterAbstract:
    def test_cannot_instantiate_abstract_adapter(self):
        with pytest.raises(TypeError):
            MessageAdapter()  # type: ignore[abstract, call-arg]

    def test_concrete_subclass_can_be_instantiated(self):
        class DummyAdapter(MessageAdapter):
            def to_provider(self, messages, tools=None, **kwargs):
                return {}

            def from_provider_response(self, response):
                return LLMResponse(
                    content=[TextBlock(text="ok")],
                    stop_reason=FinishReason.STOP,
                    model="dummy",
                )

            def from_provider_chunk(self, chunk):
                return StreamChunk(type=ContentType.TEXT, delta="")

            def map_exception(self, exception):
                return LLMError(str(exception))

        adapter = DummyAdapter()
        assert isinstance(adapter, MessageAdapter)


# ====================================================================
# AnthropicAdapter – to_provider (text messages)
# ====================================================================


class TestAnthropicAdapterText:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_single_user_text_message(self, adapter: AnthropicAdapter):
        messages = [Message.from_text(Role.USER, "Hello")]
        result = adapter.to_provider(messages)

        assert "messages" in result
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == [{"type": "text", "text": "Hello"}]

    def test_single_assistant_text_message(self, adapter: AnthropicAdapter):
        messages = [Message.from_text(Role.ASSISTANT, "Hi there")]
        result = adapter.to_provider(messages)

        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "Hi there"}]

    def test_system_message_becomes_top_level_key(self, adapter: AnthropicAdapter):
        messages = [Message.from_text(Role.SYSTEM, "You are helpful")]
        result = adapter.to_provider(messages)

        assert result["system"] == "You are helpful"
        assert result["messages"] == []

    def test_system_plus_user(self, adapter: AnthropicAdapter):
        messages = [
            Message.from_text(Role.SYSTEM, "Be concise"),
            Message.from_text(Role.USER, "What is 2+2?"),
        ]
        result = adapter.to_provider(messages)

        assert result["system"] == "Be concise"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


# ====================================================================
# AnthropicAdapter – to_provider (tool use blocks)
# ====================================================================


class TestAnthropicAdapterToolUse:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_tool_use_block_conversion(self, adapter: AnthropicAdapter):
        tool_call = ToolUseBlock(
            tool_call_id="call_abc",
            tool_name="search",
            tool_input={"query": "weather"},
        )
        msg = Message(role=Role.ASSISTANT, content=[tool_call])
        result = adapter.to_provider([msg])

        content = result["messages"][0]["content"][0]
        assert content["type"] == "tool_use"
        assert content["id"] == "call_abc"
        assert content["name"] == "search"
        assert content["input"] == {"query": "weather"}

    def test_tool_result_block_conversion(self, adapter: AnthropicAdapter):
        result_block = ToolResultBlock(
            tool_call_id="call_abc",
            tool_result_content="Sunny, 75F",
            is_error=False,
        )
        msg = Message(role=Role.USER, content=[result_block])
        out = adapter.to_provider([msg])

        content = out["messages"][0]["content"][0]
        assert content["type"] == "tool_result"
        assert content["tool_use_id"] == "call_abc"
        assert content["content"] == "Sunny, 75F"

    def test_tools_parameter(self, adapter: AnthropicAdapter):
        tool_def = ToolDefinition(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        messages = [Message.from_text(Role.USER, "Search for cats")]
        result = adapter.to_provider(messages, tools=[tool_def])

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "search"
        assert result["tools"][0]["description"] == "Search the web"


class TestAnthropicAdapterConsecutiveMessages:
    """Consecutive same-role messages must be merged (Anthropic 400 fix)."""

    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_parallel_tool_results_merge_into_single_user_message(
        self, adapter: AnthropicAdapter
    ):
        out = adapter.to_provider(_parallel_tool_history())
        messages = out["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"

        # No two consecutive entries may share a role.
        roles = [m["role"] for m in messages]
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False))

        content = messages[1]["content"]
        assert [b["type"] for b in content] == ["tool_result", "tool_result"]
        assert [b["tool_use_id"] for b in content] == ["c1", "c2"]

    def test_tool_result_ordered_before_text(self, adapter: AnthropicAdapter):
        history = [
            Message(role=Role.USER, content=[TextBlock(text="note")]),
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
            ),
        ]
        content = adapter.to_provider(history)["messages"][0]["content"]

        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "c1"
        assert content[1] == {"type": "text", "text": "note"}

    def test_tool_result_ordered_before_text_reversed_input(
        self, adapter: AnthropicAdapter
    ):
        history = [
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
            ),
            Message(role=Role.USER, content=[TextBlock(text="note")]),
        ]
        content = adapter.to_provider(history)["messages"][0]["content"]

        assert content[0]["type"] == "tool_result"
        assert content[1] == {"type": "text", "text": "note"}

    def test_empty_assistant_placeholder_dropped(self, adapter: AnthropicAdapter):
        history = [
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(tool_call_id="c1", tool_name="t", tool_input={})],
            ),
            Message(role=Role.ASSISTANT, content=[]),  # sub.py placeholder
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
            ),
        ]
        out = adapter.to_provider(history)
        messages = out["messages"]

        assert [m["role"] for m in messages] == ["assistant", "user"]
        assert all(m["content"] for m in messages)
        assert messages[0]["content"][0]["type"] == "tool_use"

    def test_empty_message_dropped(self, adapter: AnthropicAdapter):
        history = [
            Message(role=Role.USER, content=[TextBlock(text="hi")]),
            Message(role=Role.ASSISTANT, content=[]),
        ]
        out = adapter.to_provider(history)

        assert len(out["messages"]) == 1

    def test_message_converting_to_empty_content_is_skipped(
        self, adapter: AnthropicAdapter
    ):
        # An ErrorBlock has no Anthropic wire form; the message would otherwise
        # emit an empty content array, which the API rejects.
        history = [
            Message.from_text(Role.USER, "hello"),
            Message(
                role=Role.ASSISTANT,
                content=[ErrorBlock(error_code="x", error_message="boom")],
            ),
        ]
        out = adapter.to_provider(history)

        assert [m["role"] for m in out["messages"]] == ["user"]
        assert all(m["content"] for m in out["messages"])

    def test_no_consecutive_roles_when_middle_message_converts_to_nothing(
        self, adapter: AnthropicAdapter
    ):
        # The ErrorBlock assistant message converts to no wire blocks and is
        # skipped AFTER coalescing; the surrounding user messages must not end
        # up adjacent in the emitted payload.
        history = [
            Message(role=Role.USER, content=[TextBlock(text="hi")]),
            Message(
                role=Role.ASSISTANT,
                content=[ErrorBlock(error_code="x", error_message="boom")],
            ),
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r")],
            ),
        ]
        out = adapter.to_provider(history)["messages"]

        roles = [m["role"] for m in out]
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False))
        assert roles == ["user"]

    def test_user_and_tool_messages_coalesce(self, adapter: AnthropicAdapter):
        # Role.USER and Role.TOOL both emit role:"user", so adjacent ones must
        # merge even though the internal roles differ.
        history = [
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(tool_call_id="c1", tool_name="t", tool_input={})],
            ),
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
            ),
            Message(
                role=Role.TOOL,
                content=[ToolResultBlock(tool_call_id="c2", tool_result_content="r2")],
            ),
        ]
        out = adapter.to_provider(history)["messages"]

        assert [m["role"] for m in out] == ["assistant", "user"]
        assert [b["tool_use_id"] for b in out[1]["content"]] == ["c1", "c2"]

    def test_no_merge_across_role_boundary(self, adapter: AnthropicAdapter):
        history = [
            Message(role=Role.USER, content=[TextBlock(text="hi")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="ok")]),
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
            ),
        ]
        out = adapter.to_provider(history)

        assert [m["role"] for m in out["messages"]] == [
            "user",
            "assistant",
            "user",
        ]


# ====================================================================
# AnthropicAdapter – from_provider_response
# ====================================================================


class TestAnthropicAdapterFromResponse:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_parse_text_response(self, adapter: AnthropicAdapter):
        response = {
            "content": [{"type": "text", "text": "Hello world"}],
            "stop_reason": "end_turn",
            "model": "claude-3-5-sonnet",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = adapter.from_provider_response(response)

        assert isinstance(result, LLMResponse)
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Hello world"
        assert result.stop_reason == FinishReason.END_TURN
        assert result.model == "claude-3-5-sonnet"
        assert result.usage == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    def test_parse_tool_use_response(self, adapter: AnthropicAdapter):
        response = {
            "content": [
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "search",
                    "input": {"query": "test"},
                },
            ],
            "stop_reason": "tool_use",
            "model": "claude-3",
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 2
        assert isinstance(result.content[0], TextBlock)
        assert isinstance(result.content[1], ToolUseBlock)
        assert result.content[1].tool_call_id == "call_1"
        assert result.content[1].tool_name == "search"
        assert result.stop_reason == FinishReason.TOOL_USE

    def test_parse_thinking_response(self, adapter: AnthropicAdapter):
        response = {
            "content": [
                {"type": "thinking", "thinking": "Hmm, let me think..."},
                {"type": "text", "text": "The answer is 42"},
            ],
            "stop_reason": "end_turn",
            "model": "claude-3",
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 2
        assert isinstance(result.content[0], ThinkingBlock)
        assert result.content[0].thinking == "Hmm, let me think..."


# ====================================================================
# AnthropicAdapter – from_provider_chunk (streaming)
# ====================================================================


class TestAnthropicAdapterStreaming:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_text_delta(self, adapter: AnthropicAdapter):
        chunk = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
            "index": 0,
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TEXT
        assert result.delta == "Hello"
        assert result.index == 0

    def test_thinking_delta(self, adapter: AnthropicAdapter):
        chunk = {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "Thinking..."},
            "index": 0,
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.THINKING
        assert result.delta == "Thinking..."

    def test_tool_use_start(self, adapter: AnthropicAdapter):
        chunk = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_1",
                "name": "search",
                "input": {},
            },
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TOOL_USE
        assert result.tool_call_id == "call_1"

    def test_message_delta_stop(self, adapter: AnthropicAdapter):
        chunk = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.finish_reason == FinishReason.END_TURN


class TestAnthropicAdapterSdkStreamEvent:
    """from_sdk_stream_event: the typed path AnthropicProvider.stream() uses.

    Real streams yield raw content_block_* events; the SDK also yields parsed
    convenience events (text/thinking/input_json) with the same payloads, which
    must NOT also be handled or every fragment is emitted twice.
    """

    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    @staticmethod
    def _start(index: int, block_type: str, **fields):
        from anthropic.types import RawContentBlockStartEvent

        return RawContentBlockStartEvent(
            type="content_block_start",
            index=index,
            content_block={"type": block_type, **fields},
        )

    @staticmethod
    def _delta(index: int, delta):
        from anthropic.types import RawContentBlockDeltaEvent

        return RawContentBlockDeltaEvent(
            type="content_block_delta", index=index, delta=delta
        )

    def test_tool_use_start_carries_name_id_and_index(self, adapter: AnthropicAdapter):
        """The tool name is only available on the start event."""
        from anthropic.types import ToolUseBlock

        event = self._start(1, "tool_use", id="call_1", name="calculator", input={})
        assert isinstance(event.content_block, ToolUseBlock)
        result = adapter.from_sdk_stream_event(event)

        assert result.type == ContentType.TOOL_USE
        assert result.tool_call_id == "call_1"
        assert result.delta == {"name": "calculator", "input": {}}
        assert result.index == 1

    def test_input_json_delta_uses_matching_index(self, adapter: AnthropicAdapter):
        """Arguments must carry the same index as the start event, else the
        consumer opens a second buffer and loses the name."""
        from anthropic.types import InputJSONDelta

        result = adapter.from_sdk_stream_event(
            self._delta(1, InputJSONDelta(type="input_json_delta", partial_json='{"a"'))
        )

        assert result.type == ContentType.TOOL_USE
        assert result.delta == {"arguments": '{"a"'}
        assert result.index == 1
        assert result.tool_call_id is None

    def test_text_delta_via_raw_event(self, adapter: AnthropicAdapter):
        from anthropic.types import TextDelta

        result = adapter.from_sdk_stream_event(
            self._delta(0, TextDelta(type="text_delta", text="Hello"))
        )

        assert result.type == ContentType.TEXT
        assert result.delta == "Hello"

    def test_parsed_convenience_events_are_ignored(self, adapter: AnthropicAdapter):
        """text/thinking/input_json events duplicate the raw events above."""
        from anthropic.lib.streaming import InputJsonEvent, TextEvent

        text_event = TextEvent(type="text", text="Hello", snapshot="Hello")
        assert adapter.from_sdk_stream_event(text_event) is None

        json_event = InputJsonEvent(
            type="input_json", partial_json='{"a"', snapshot='{"a"'
        )
        assert adapter.from_sdk_stream_event(json_event) is None


# ====================================================================
# AnthropicAdapter – map_exception
# ====================================================================


class TestAnthropicAdapterMapException:
    @pytest.fixture
    def adapter(self):
        return AnthropicAdapter()

    def test_authentication_error(self, adapter: AnthropicAdapter):
        exc = self._fake_exception("anthropic.AuthenticationError", "Invalid API key")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, AuthenticationError)
        assert mapped.recoverable is False

    def test_rate_limit_error(self, adapter: AnthropicAdapter):
        exc = self._fake_exception("anthropic.RateLimitError", "Rate limited")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, RateLimitError)
        assert mapped.recoverable is True

    def test_api_error(self, adapter: AnthropicAdapter):
        exc = self._fake_exception("anthropic.APIError", "Server error")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ProviderUnavailableError)

    def test_context_window_exceeded(self, adapter: AnthropicAdapter):
        exc = self._fake_exception(
            "anthropic.BadRequestError", "prompt is too long: context window exceeded"
        )
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ContextWindowExceededError)

    def test_unknown_error(self, adapter: AnthropicAdapter):
        exc = ValueError("something weird")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, LLMError)

    @staticmethod
    def _fake_exception(full_name: str, msg: str) -> Exception:
        mod, cls_name = full_name.rsplit(".", 1)
        exc_cls = type(cls_name, (Exception,), {})
        return exc_cls(msg)


# ====================================================================
# OpenAIAdapter – to_provider (text messages)
# ====================================================================


class TestOpenAIAdapterText:
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_single_user_text_message(self, adapter: OpenAIAdapter):
        messages = [Message.from_text(Role.USER, "Hello")]
        result = adapter.to_provider(messages)

        assert "messages" in result
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == [{"type": "text", "text": "Hello"}]

    def test_single_assistant_text_message(self, adapter: OpenAIAdapter):
        messages = [Message.from_text(Role.ASSISTANT, "Hi there")]
        result = adapter.to_provider(messages)

        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "Hi there"}]

    def test_system_message(self, adapter: OpenAIAdapter):
        messages = [Message.from_text(Role.SYSTEM, "Be concise")]
        result = adapter.to_provider(messages)

        msg = result["messages"][0]
        assert msg["role"] == "system"
        assert msg["content"] == "Be concise"

    def test_multiple_messages(self, adapter: OpenAIAdapter):
        messages = [
            Message.from_text(Role.SYSTEM, "You are helpful"),
            Message.from_text(Role.USER, "What is 2+2?"),
            Message.from_text(Role.ASSISTANT, "It's 4"),
        ]
        result = adapter.to_provider(messages)

        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"


# ====================================================================
# OpenAIAdapter – to_provider (tool use blocks)
# ====================================================================


class TestOpenAIAdapterToolUse:
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_tool_use_in_assistant_message(self, adapter: OpenAIAdapter):
        tool_call = ToolUseBlock(
            tool_call_id="call_xyz",
            tool_name="search",
            tool_input={"query": "weather"},
        )
        msg = Message(role=Role.ASSISTANT, content=[tool_call])
        result = adapter.to_provider([msg])

        assistant_msg = result["messages"][0]
        assert "tool_calls" in assistant_msg
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "call_xyz"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "weather"}

    def test_tool_result_becomes_role_tool(self, adapter: OpenAIAdapter):
        result_block = ToolResultBlock(
            tool_call_id="call_xyz",
            tool_result_content="Sunny",
        )
        # ToolResultBlock in an assistant message still gets emitted as a
        # separate tool-role message by the OpenAI adapter.
        msg = Message(role=Role.ASSISTANT, content=[result_block])
        result = adapter.to_provider([msg])

        tool_msg = result["messages"][0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_xyz"
        assert tool_msg["content"] == "Sunny"


class TestOpenAIAdapterConsecutiveMessages:
    """Empty placeholders dropped; N results stay N role:tool messages."""

    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_parallel_tool_results_become_sequential_tool_messages(
        self, adapter: OpenAIAdapter
    ):
        out = adapter.to_provider(_parallel_tool_history())
        messages = out["messages"]

        assert [m["role"] for m in messages] == ["assistant", "tool", "tool"]
        assert [tc["id"] for tc in messages[0]["tool_calls"]] == ["c1", "c2"]
        assert messages[1]["tool_call_id"] == "c1"
        assert messages[1]["content"] == "r1"
        assert messages[2]["tool_call_id"] == "c2"
        assert messages[2]["content"] == "r2"

    def test_empty_assistant_placeholder_dropped(self, adapter: OpenAIAdapter):
        history = [
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(tool_call_id="c1", tool_name="t", tool_input={})],
            ),
            Message(role=Role.ASSISTANT, content=[]),  # sub.py placeholder
            Message(
                role=Role.USER,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
            ),
        ]
        out = adapter.to_provider(history)
        messages = out["messages"]

        assert [m["role"] for m in messages] == ["assistant", "tool"]
        assert not any(
            m["role"] == "assistant" and not m.get("tool_calls") for m in messages
        )

    def test_consecutive_assistant_messages_merged(self, adapter: OpenAIAdapter):
        history = [
            Message(role=Role.ASSISTANT, content=[TextBlock(text="ok")]),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(tool_call_id="c1", tool_name="t", tool_input={})],
            ),
        ]
        out = adapter.to_provider(history)
        messages = out["messages"]

        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == [{"type": "text", "text": "ok"}]
        assert [tc["id"] for tc in messages[0]["tool_calls"]] == ["c1"]

    def test_consecutive_user_messages_not_merged(self, adapter: OpenAIAdapter):
        # Deliberate asymmetry with Anthropic: each user tool-result message
        # must stay its own role:tool message.
        history = [
            Message(role=Role.USER, content=[TextBlock(text="a")]),
            Message(role=Role.USER, content=[TextBlock(text="b")]),
        ]
        out = adapter.to_provider(history)

        assert [m["role"] for m in out["messages"]] == ["user", "user"]

    def test_unconvertible_assistant_not_emitted_as_empty(self, adapter: OpenAIAdapter):
        # An ErrorBlock-only assistant message must not become
        # {"role":"assistant","content":""}, which is invalid between
        # tool_calls and its results.
        history = [
            Message(role=Role.USER, content=[TextBlock(text="hi")]),
            Message(
                role=Role.ASSISTANT,
                content=[ErrorBlock(error_code="x", error_message="boom")],
            ),
        ]
        out = adapter.to_provider(history)["messages"]

        assert [m["role"] for m in out] == ["user"]
        assert all(m["role"] != "assistant" for m in out)

    def test_assistant_tool_results_kept_alongside_other_blocks(
        self, adapter: OpenAIAdapter
    ):
        # A tool_result in an assistant message must still become role:"tool"
        # even when the same message carries a tool_use, otherwise the matching
        # tool_call_id is orphaned.
        history = [
            Message(
                role=Role.ASSISTANT,
                content=[ToolResultBlock(tool_call_id="c1", tool_result_content="r1")],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[ToolUseBlock(tool_call_id="c2", tool_name="t", tool_input={})],
            ),
        ]
        out = adapter.to_provider(history)["messages"]

        tool_msgs = [m for m in out if m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["c1"]
        assert tool_msgs[0]["content"] == "r1"
        assert any(m.get("tool_calls") for m in out)

    def test_tools_parameter(self, adapter: OpenAIAdapter):
        tool_def = ToolDefinition(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        messages = [Message.from_text(Role.USER, "Search")]
        result = adapter.to_provider(messages, tools=[tool_def])

        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "search"


# ====================================================================
# merge_consecutive_messages helper
# ====================================================================


class TestMergeConsecutiveMessages:
    def test_empty_messages_dropped(self):
        history = [
            Message(role=Role.USER, content=[TextBlock(text="hi")]),
            Message(role=Role.ASSISTANT, content=[]),
            Message(role=Role.USER, content=[TextBlock(text="bye")]),
        ]
        out = merge_consecutive_messages(history, frozenset({Role.USER}))

        # The empty assistant is dropped, leaving the two user messages
        # adjacent, so they merge into one.
        assert [m.role for m in out] == [Role.USER]
        assert [b.text for b in out[0].content] == ["hi", "bye"]

    def test_no_merge_across_role_boundary(self):
        history = [
            Message(role=Role.USER, content=[TextBlock(text="a")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="b")]),
            Message(role=Role.USER, content=[TextBlock(text="c")]),
        ]
        out = merge_consecutive_messages(
            history, frozenset({Role.USER, Role.ASSISTANT})
        )

        assert [m.role for m in out] == [Role.USER, Role.ASSISTANT, Role.USER]

    def test_only_listed_roles_are_merged(self):
        history = [
            Message(role=Role.ASSISTANT, content=[TextBlock(text="a")]),
            Message(role=Role.ASSISTANT, content=[TextBlock(text="b")]),
        ]
        out = merge_consecutive_messages(history, frozenset({Role.USER}))

        assert [m.role for m in out] == [Role.ASSISTANT, Role.ASSISTANT]

    def test_merge_is_pure_and_idempotent(self):
        history = _parallel_tool_history()
        before = [len(m.content) for m in history]
        before_ids = [id(m.content) for m in history]

        out = merge_consecutive_messages(
            history, frozenset({Role.USER, Role.ASSISTANT})
        )

        # Inputs untouched: same block counts, same content-list objects.
        assert [len(m.content) for m in history] == before
        assert [id(m.content) for m in history] == before_ids

        # Idempotent: a second pass over the output changes nothing.
        again = merge_consecutive_messages(out, frozenset({Role.USER, Role.ASSISTANT}))
        assert [(m.role, m.content) for m in again] == [
            (m.role, m.content) for m in out
        ]

    def test_blocks_shared_by_reference(self):
        # Merging copies the list, not the blocks: adapters only read them.
        history = _parallel_tool_history()
        out = merge_consecutive_messages(
            history, frozenset({Role.USER, Role.ASSISTANT})
        )
        assert out[1].content[0] is history[1].content[0]
        assert out[1].content[1] is history[2].content[0]


# ====================================================================
# OpenAIAdapter – from_provider_response
# ====================================================================


class TestOpenAIAdapterFromResponse:
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_parse_text_response(self, adapter: OpenAIAdapter):
        response = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = adapter.from_provider_response(response)

        assert isinstance(result, LLMResponse)
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Hello world"
        assert result.stop_reason == FinishReason.STOP
        assert result.model == "gpt-4o"
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_parse_tool_use_response(self, adapter: OpenAIAdapter):
        response = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "test"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolUseBlock)
        assert result.content[0].tool_call_id == "call_1"
        assert result.content[0].tool_name == "search"
        assert result.content[0].tool_input == {"query": "test"}
        assert result.stop_reason == FinishReason.TOOL_USE

    def test_parse_text_and_tools(self, adapter: OpenAIAdapter):
        response = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me look that up.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "cats"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 2
        assert isinstance(result.content[0], TextBlock)
        assert isinstance(result.content[1], ToolUseBlock)


# ====================================================================
# OpenAIAdapter – from_provider_chunk (streaming)
# ====================================================================


class TestOpenAIAdapterStreaming:
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_text_delta(self, adapter: OpenAIAdapter):
        chunk = {
            "choices": [
                {
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                    "index": 0,
                }
            ]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TEXT
        assert result.delta == "Hello"

    def test_tool_call_delta(self, adapter: OpenAIAdapter):
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TOOL_USE
        assert result.tool_call_id == "call_1"
        assert result.delta["name"] == "search"

    def test_finish_delta(self, adapter: OpenAIAdapter):
        chunk = {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.finish_reason == FinishReason.STOP
        assert result.delta is None


# ====================================================================
# OpenAIAdapter – map_exception
# ====================================================================


class TestOpenAIAdapterMapException:
    @pytest.fixture
    def adapter(self):
        return OpenAIAdapter()

    def test_authentication_error(self, adapter: OpenAIAdapter):
        exc = self._fake_exception("openai.AuthenticationError", "Invalid key")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, AuthenticationError)

    def test_rate_limit_error(self, adapter: OpenAIAdapter):
        exc = self._fake_exception("openai.RateLimitError", "Too many requests")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, RateLimitError)

    def test_api_error(self, adapter: OpenAIAdapter):
        exc = self._fake_exception("openai.APIError", "Server down")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ProviderUnavailableError)

    def test_api_connection_error(self, adapter: OpenAIAdapter):
        exc = self._fake_exception("openai.APIConnectionError", "Connection refused")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ProviderUnavailableError)

    def test_context_window_exceeded(self, adapter: OpenAIAdapter):
        exc = self._fake_exception(
            "openai.BadRequestError", "maximum context length exceeded"
        )
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ContextWindowExceededError)

    def test_unknown_error(self, adapter: OpenAIAdapter):
        exc = ValueError("weird")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, LLMError)

    @staticmethod
    def _fake_exception(full_name: str, msg: str) -> Exception:
        mod, cls_name = full_name.rsplit(".", 1)
        exc_cls = type(cls_name, (Exception,), {})
        return exc_cls(msg)


class TestOpenAIAdapterFlattenWarning:
    """Flattening a multi-block user message degrades structure; it must be
    visible via a warning log."""

    def test_flatten_logs_warning(self, caplog):
        adapter = OpenAIAdapter()
        history = [
            Message(
                role=Role.USER,
                content=[TextBlock(text="part one"), TextBlock(text="part two")],
            )
        ]
        with caplog.at_level(
            "WARNING", logger="sanityops_agent.llm.adapters.openai_adapter"
        ):
            out = adapter.to_provider(history)

        assert out["messages"][0]["content"] == "part one\npart two"
        assert any("Flattening" in r.message for r in caplog.records)


class TestExceptionMappingOverloaded:
    """OverloadedError (529) means the provider is busy: it must map to a
    recoverable ProviderUnavailableError, not a generic LLMError."""

    def test_anthropic_overloaded_maps_to_provider_unavailable(self):
        class OverloadedError(Exception):
            pass

        mapped = AnthropicAdapter.map_exception(OverloadedError("overloaded"))
        assert isinstance(mapped, ProviderUnavailableError)

    def test_openai_overloaded_maps_to_provider_unavailable(self):
        class OverloadedError(Exception):
            pass

        mapped = OpenAIAdapter.map_exception(OverloadedError("overloaded"))
        assert isinstance(mapped, ProviderUnavailableError)


class TestOpenAIAdapterToolInputSerialization:
    def test_non_serializable_tool_input_is_stringified(self):
        adapter = OpenAIAdapter()
        history = [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ToolUseBlock(
                        tool_call_id="c1",
                        tool_name="t",
                        tool_input={"bad": object()},
                    )
                ],
            )
        ]
        out = adapter.to_provider(history)["messages"]

        args = out[0]["tool_calls"][0]["function"]["arguments"]
        assert "raw" in json.loads(args)


class TestToolsToProviderStrictness:
    """OpenAI strictness must be explicitly requested."""

    def test_openai_strict_defaults_false(self):
        adapter = OpenAIAdapter()
        td = ToolDefinition(
            name="t", description="d", input_schema={"type": "object", "properties": {}}
        )
        out = adapter.tools_to_provider([td])[0]
        assert out["function"].get("strict", False) is False

    def test_openai_strict_opt_out_via_provider_config(self):
        adapter = OpenAIAdapter()
        td = ToolDefinition(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            provider_config={"strict": False},
        )
        out = adapter.tools_to_provider([td])[0]
        assert out["function"]["strict"] is False

    def test_openai_additional_properties_preserved(self):
        adapter = OpenAIAdapter()
        td = ToolDefinition(
            name="t",
            description="d",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": {"type": "string"},
            },
        )
        out = adapter.tools_to_provider([td])[0]
        assert out["function"]["parameters"]["additionalProperties"] == {
            "type": "string"
        }

    def test_openai_additional_properties_defaults_false(self):
        adapter = OpenAIAdapter()
        td = ToolDefinition(
            name="t", description="d", input_schema={"type": "object", "properties": {}}
        )
        out = adapter.tools_to_provider([td])[0]
        assert out["function"]["parameters"]["additionalProperties"] is False

    def test_anthropic_strict_opt_out_via_provider_config(self):
        adapter = AnthropicAdapter()
        td = ToolDefinition(
            name="t",
            description="d",
            input_schema={"type": "object", "properties": {}},
            provider_config={"strict": False},
        )
        out = adapter.tools_to_provider([td])[0]
        assert out["strict"] is False


class TestContextWindowMapping:
    """Context-window errors must be recognized structurally where possible."""

    @staticmethod
    def _bad_request(message: str, status: int | None):
        class BadRequestError(Exception):
            pass

        exc = BadRequestError(message)
        if status is not None:
            exc.response = type("R", (), {"status_code": status})()
        return exc

    def test_anthropic_context_window_with_400(self):
        exc = self._bad_request("prompt is too long", 400)
        mapped = AnthropicAdapter.map_exception(exc)
        assert isinstance(mapped, ContextWindowExceededError)

    def test_anthropic_context_window_without_response_attr(self):
        exc = self._bad_request("prompt is too long", None)
        mapped = AnthropicAdapter.map_exception(exc)
        assert isinstance(mapped, ContextWindowExceededError)

    def test_anthropic_non_context_bad_request_not_misclassified(self):
        exc = self._bad_request("invalid model", 400)
        mapped = AnthropicAdapter.map_exception(exc)
        assert not isinstance(mapped, ContextWindowExceededError)

    def test_openai_context_window_with_400(self):
        exc = self._bad_request("maximum context length exceeded", 400)
        mapped = OpenAIAdapter.map_exception(exc)
        assert isinstance(mapped, ContextWindowExceededError)

    def test_openai_non_context_bad_request_not_misclassified(self):
        exc = self._bad_request("invalid api key format", 400)
        mapped = OpenAIAdapter.map_exception(exc)
        assert not isinstance(mapped, ContextWindowExceededError)


# ---------------------------------------------------------------------------
# GeminiAdapter tests
# ---------------------------------------------------------------------------


# GeminiAdapter – text message conversion
# ====================================================================


class TestGeminiAdapterText:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()

    def test_single_user_text_message(self, adapter):
        messages = [Message.from_text(Role.USER, "Hello")]
        result = adapter.to_provider(messages)

        assert "contents" in result
        assert len(result["contents"]) == 1
        content = result["contents"][0]
        assert content["role"] == "user"
        assert len(content["parts"]) == 1
        assert content["parts"][0]["text"] == "Hello"

    def test_assistant_role_becomes_model(self, adapter):
        messages = [Message.from_text(Role.ASSISTANT, "Hi there")]
        result = adapter.to_provider(messages)

        content = result["contents"][0]
        assert content["role"] == "model"
        assert content["parts"][0]["text"] == "Hi there"

    def test_system_message_becomes_system_instruction(self, adapter):
        messages = [Message.from_text(Role.SYSTEM, "You are helpful")]
        result = adapter.to_provider(messages)

        assert "systemInstruction" in result
        assert result["systemInstruction"]["parts"][0]["text"] == "You are helpful"
        assert result["contents"] == []

    def test_system_plus_user(self, adapter):
        messages = [
            Message.from_text(Role.SYSTEM, "Be concise"),
            Message.from_text(Role.USER, "What is 2+2?"),
        ]
        result = adapter.to_provider(messages)

        assert result["systemInstruction"]["parts"][0]["text"] == "Be concise"
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"


# ====================================================================
# GeminiAdapter – tool use blocks
# ====================================================================


class TestGeminiAdapterToolUse:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()

    def test_tool_use_block_conversion(self, adapter):
        tool_call = ToolUseBlock(
            tool_call_id="call_abc",
            tool_name="search",
            tool_input={"query": "weather"},
        )
        msg = Message(role=Role.ASSISTANT, content=[tool_call])
        result = adapter.to_provider([msg])

        part = result["contents"][0]["parts"][0]
        assert "functionCall" in part
        assert part["functionCall"]["name"] == "search"
        assert part["functionCall"]["args"] == {"query": "weather"}

    def test_tool_result_block_conversion(self, adapter):
        result_block = ToolResultBlock(
            tool_call_id="call_abc",
            tool_result_content="Sunny, 75F",
            is_error=False,
        )
        msg = Message(role=Role.USER, content=[result_block])
        out = adapter.to_provider([msg])

        part = out["contents"][0]["parts"][0]
        assert "functionResponse" in part
        # Unpaired fallback: name falls back to the call id
        assert part["functionResponse"]["name"] == "call_abc"
        assert part["functionResponse"]["response"]["result"] == "Sunny, 75F"

    def test_tool_result_uses_function_name_from_prior_tool_use(self, adapter):
        """The real agent flow sends ToolUse then ToolResult. Gemini's
        FunctionResponse requires the FUNCTION NAME, not the call id, so the
        adapter must look it up from the earlier ToolUseBlock."""
        tool_use = ToolUseBlock(
            tool_call_id="call_abc",
            tool_name="search",
            tool_input={"query": "weather"},
        )
        tool_result = ToolResultBlock(
            tool_call_id="call_abc",
            tool_result_content="Sunny, 75F",
        )
        messages = [
            Message(role=Role.ASSISTANT, content=[tool_use]),
            Message(role=Role.USER, content=[tool_result]),
        ]
        out = adapter.to_provider(messages)

        call_part = out["contents"][0]["parts"][0]["functionCall"]
        assert call_part["id"] == "call_abc"
        assert call_part["name"] == "search"

        resp_part = out["contents"][1]["parts"][0]["functionResponse"]
        assert resp_part["id"] == "call_abc"
        assert resp_part["name"] == "search"
        assert resp_part["response"]["result"] == "Sunny, 75F"

    def test_tools_parameter(self, adapter):
        tool_def = ToolDefinition(
            name="search",
            description="Search the web",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        messages = [Message.from_text(Role.USER, "Search for cats")]
        result = adapter.to_provider(messages, tools=[tool_def])

        assert "tools" in result
        assert "functionDeclarations" in result["tools"][0]
        assert result["tools"][0]["functionDeclarations"][0]["name"] == "search"

    def test_parse_tool_use_response(self, adapter):
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"query": "test"},
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash-lite",
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolUseBlock)
        assert result.content[0].tool_name == "search"
        assert result.model == "gemini-3.5-flash-lite"
        assert result.content[0].tool_input == {"query": "test"}


# ====================================================================
# GeminiAdapter – from_provider_chunk (streaming)
# ====================================================================


class TestGeminiAdapterStreaming:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter(store_thinking=True)

    def test_text_delta(self, adapter):
        chunk = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TEXT
        assert result.delta == "Hello"

    def test_thinking_delta(self, adapter):
        # Real Gemini thought parts carry BOTH text and thought=True
        chunk = {
            "candidates": [
                {"content": {"parts": [{"text": "Let me think...", "thought": True}]}}
            ]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.THINKING
        assert result.delta == "Let me think..."

    def test_thinking_not_stored(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        adapter = GeminiAdapter(store_thinking=False)
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Hmm...", "thought": True},
                            {"text": "The answer is 42"},
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash-lite",
        }
        result = adapter.from_provider_response(response)

        # Only text, no thinking block
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "The answer is 42"

    def test_tool_call_delta(self, adapter):
        chunk = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"query": "test"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TOOL_USE
        assert result.tool_call_id == "search"
        assert result.delta["name"] == "search"

    def test_finish_reason(self, adapter):
        chunk = {
            "candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.finish_reason == FinishReason.MAX_TOKENS


# ====================================================================
# GeminiAdapter – map_exception
# ====================================================================


class TestGeminiAdapterMapException:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()

    def test_authentication_error_401(self, adapter):
        """401 status code should map to AuthenticationError with recoverable=False."""
        exc = self._fake_api_error(401, "Unauthorized")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, AuthenticationError)
        assert mapped.recoverable is False

    def test_authentication_error_403(self, adapter):
        """403 status code should map to AuthenticationError with recoverable=False."""
        exc = self._fake_api_error(403, "Permission denied")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, AuthenticationError)
        assert mapped.recoverable is False

    def test_rate_limit_error(self, adapter):
        """429 status code should map to RateLimitError with retry_after."""
        exc = self._fake_api_error(429, "Rate limited", headers={"retry-after": "60"})
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, RateLimitError)
        assert mapped.recoverable is True
        # retry_after should be extracted from headers

    def test_api_error_500(self, adapter):
        """500+ status codes should map to ProviderUnavailableError."""
        exc = self._fake_api_error(500, "Internal server error")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ProviderUnavailableError)

    def test_api_error_503(self, adapter):
        """503 status code should map to ProviderUnavailableError."""
        exc = self._fake_api_error(503, "Service unavailable")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ProviderUnavailableError)

    def test_context_window_exceeded_error(self, adapter):
        """400 status with 'context' in message should map to ContextWindowExceededError."""
        exc = self._fake_api_error(
            400, "Request payload size exceeds the limit: context window exceeded"
        )
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ContextWindowExceededError)

    def test_context_window_exceeded_error_token_keyword(self, adapter):
        """400 status with 'token' in message should map to ContextWindowExceededError."""
        exc = self._fake_api_error(400, "Token count exceeds the maximum limit")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ContextWindowExceededError)

    def test_llm_error_400_no_context(self, adapter):
        """400 status without context keywords should map to LLMError."""
        exc = self._fake_api_error(400, "Invalid request format")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, LLMError)
        assert not isinstance(mapped, ContextWindowExceededError)

    def test_unknown_sdk_exception_by_name_resource_exhausted(self, adapter):
        """ResourceExhausted by name should map to RateLimitError."""
        exc = self._fake_exception("ResourceExhausted", "Resource exhausted")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, RateLimitError)
        assert mapped.recoverable is True

    def test_unknown_sdk_exception_by_name_service_unavailable(self, adapter):
        """ServiceUnavailable by name should map to ProviderUnavailableError."""
        exc = self._fake_exception("ServiceUnavailable", "Service unavailable")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, ProviderUnavailableError)

    def test_plain_unrelated_exception(self, adapter):
        """Plain unrelated exception should map to LLMError."""
        exc = ValueError("Something went wrong")
        mapped = adapter.map_exception(exc)
        assert isinstance(mapped, LLMError)

    @staticmethod
    def _fake_api_error(code: int, msg: str, headers: dict | None = None) -> Exception:
        """Create a fake google.genai.errors.APIError-like exception."""
        # Import the actual errors module to use the real APIError class
        import httpx
        from google.genai import errors as genai_errors

        # Create a mock httpx.Response with the given headers
        mock_response = httpx.Response(
            status_code=code,
            headers=headers or {},
            text=msg,
        )

        # Create the actual APIError with proper arguments
        return genai_errors.APIError(
            code=code,
            response_json={"error": {"message": msg}},
            response=mock_response,
        )

    @staticmethod
    def _fake_exception(cls_name: str, msg: str) -> Exception:
        """Create a generic exception with the given class name."""
        exc_cls = type(cls_name, (Exception,), {})
        return exc_cls(msg)


# ====================================================================
# GeminiAdapter – thoughtSignature handling
# ====================================================================


class TestGeminiAdapterThoughtSignature:
    """Tests for thoughtSignature handling in GeminiAdapter."""

    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter(store_thinking=True, model_id="gemini-2.5-flash")

    def test_thinking_block_with_signature_same_model(self, adapter):
        """When same provider/model, thoughtSignature should be preserved."""
        thinking = ThinkingBlock(
            thinking="Let me think...",
            signature="abc123==",
        )
        msg = Message(
            role=Role.ASSISTANT,
            content=[thinking],
            provider="google",
            model="gemini-2.5-flash",
        )
        result = adapter.to_provider(
            [msg],
            current_provider="google",
            current_model="gemini-2.5-flash",
        )

        part = result["contents"][0]["parts"][0]
        assert part["thought"] is True
        assert part["text"] == "Let me think..."
        assert part["thoughtSignature"] == "abc123=="

    def test_thinking_block_cross_model_drops_signature(self, adapter):
        """Cross-model thinking block should convert to plain text, no signature."""
        thinking = ThinkingBlock(
            thinking="Let me think...",
            signature="abc123==",
        )
        msg = Message(
            role=Role.ASSISTANT,
            content=[thinking],
            provider="anthropic",  # Different provider
            model="claude-3-sonnet",
        )
        result = adapter.to_provider(
            [msg],
            current_provider="google",
            current_model="gemini-2.5-flash",
        )

        part = result["contents"][0]["parts"][0]
        # Cross-model: becomes plain text, no thought flag or signature
        assert "thought" not in part
        assert part["text"] == "Let me think..."
        assert "thoughtSignature" not in part

    def test_text_block_with_signature_same_model(self, adapter):
        """Text blocks can carry thoughtSignature for Gemini reasoning replay."""
        text = TextBlock(
            text="The answer is 42",
            text_signature="sig456==",
        )
        msg = Message(
            role=Role.ASSISTANT,
            content=[text],
            provider="google",
            model="gemini-2.5-flash",
        )
        result = adapter.to_provider(
            [msg],
            current_provider="google",
            current_model="gemini-2.5-flash",
        )

        part = result["contents"][0]["parts"][0]
        assert part["text"] == "The answer is 42"
        assert part["thoughtSignature"] == "sig456=="

    def test_tool_use_with_thought_signature(self, adapter):
        """Tool calls can carry thoughtSignature."""
        tool_use = ToolUseBlock(
            tool_call_id="call_abc",
            tool_name="search",
            tool_input={"query": "test"},
            thought_signature="toolSig==",
        )
        msg = Message(
            role=Role.ASSISTANT,
            content=[tool_use],
            provider="google",
            model="gemini-2.5-flash",
        )
        result = adapter.to_provider(
            [msg],
            current_provider="google",
            current_model="gemini-2.5-flash",
        )

        part = result["contents"][0]["parts"][0]
        assert part["functionCall"]["name"] == "search"
        assert part["thoughtSignature"] == "toolSig=="

    def test_from_provider_response_extracts_thinking_signature(self, adapter):
        """Response parsing should extract thoughtSignature."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "thought": True,
                                "text": "Hmm...",
                                "thoughtSignature": "respSig==",
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-2.5-flash",
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 1
        block = result.content[0]
        assert isinstance(block, ThinkingBlock)
        assert block.thinking == "Hmm..."
        assert block.signature == "respSig=="

    def test_from_provider_response_extracts_text_signature(self, adapter):
        """Response parsing should extract thoughtSignature from text parts."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "The answer",
                                "thoughtSignature": "textSig==",
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-2.5-flash",
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 1
        block = result.content[0]
        assert isinstance(block, TextBlock)
        assert block.text == "The answer"
        assert block.text_signature == "textSig=="

    def test_from_provider_response_extracts_tool_signature(self, adapter):
        """Response parsing should extract thoughtSignature from tool calls."""
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"q": "test"},
                                },
                                "thoughtSignature": "toolRespSig==",
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-2.5-flash",
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 1
        block = result.content[0]
        assert isinstance(block, ToolUseBlock)
        assert block.tool_name == "search"
        assert block.thought_signature == "toolRespSig=="

    def test_thinking_block_with_none_provider_drops_signature(self, adapter):
        """If message provider is None, signature should be dropped."""
        thinking = ThinkingBlock(
            thinking="Hmm...",
            signature="sig==",
        )
        msg = Message(
            role=Role.ASSISTANT,
            content=[thinking],
            provider=None,  # Legacy message without provider
            model="gemini-2.5-flash",
        )
        result = adapter.to_provider(
            [msg],
            current_provider="google",
            current_model="gemini-2.5-flash",
        )

        # Should NOT have thoughtSignature (dropped due to None provider)
        part = result["contents"][0]["parts"][0]
        assert "thoughtSignature" not in part
        assert part["text"] == "Hmm..."

    def test_thinking_block_with_none_current_model_drops_signature(self, adapter):
        """If current_model is None, signature should be dropped."""
        thinking = ThinkingBlock(
            thinking="Hmm...",
            signature="sig==",
        )
        msg = Message(
            role=Role.ASSISTANT,
            content=[thinking],
            provider="google",
            model="gemini-2.5-flash",
        )
        result = adapter.to_provider(
            [msg],
            current_provider="google",
            current_model=None,  # Unknown current model
        )

        # Should NOT have thoughtSignature (dropped due to None current_model)
        part = result["contents"][0]["parts"][0]
        assert "thoughtSignature" not in part


# ====================================================================
# GeminiAdapter – tool call ID normalization
# ====================================================================


class TestGeminiAdapterToolCallIdNormalization:
    """Tests for tool call ID normalization."""

    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter(model_id="gemini-3.5-flash")

    def test_normalize_special_characters(self, adapter):
        """Special characters should be replaced with underscore."""
        normalized = adapter._normalize_tool_call_id("call|with|pipes")
        assert normalized == "call_with_pipes"

    def test_normalize_truncates_long_id(self, adapter):
        """IDs longer than 64 chars should be truncated with hash suffix to avoid collisions."""
        long_id = "a" * 100
        normalized = adapter._normalize_tool_call_id(long_id)
        assert len(normalized) == 64
        # First 53 chars are 'a', then underscore, then 10-char hash
        assert normalized[:53] == "a" * 53
        assert normalized[53] == "_"
        assert len(normalized[54:]) == 10  # hash suffix

    def test_normalize_preserves_valid_id(self, adapter):
        """Valid IDs should pass through unchanged."""
        valid_id = "call_abc-123_XYZ"
        normalized = adapter._normalize_tool_call_id(valid_id)
        assert normalized == valid_id

    def test_tool_use_block_normalizes_id(self, adapter):
        """Tool use blocks should have normalized IDs in output."""
        tool_use = ToolUseBlock(
            tool_call_id="call|with|special|chars",
            tool_name="search",
            tool_input={"query": "test"},
        )
        msg = Message(role=Role.ASSISTANT, content=[tool_use])
        result = adapter.to_provider([msg])

        fc = result["contents"][0]["parts"][0]["functionCall"]
        assert fc["id"] == "call_with_special_chars"

    def test_normalize_exactly_64_chars(self, adapter):
        """IDs exactly 64 chars should pass through unchanged."""
        exact_id = "a" * 64
        normalized = adapter._normalize_tool_call_id(exact_id)
        assert normalized == exact_id
        assert len(normalized) == 64

    def test_normalize_special_chars_at_end(self, adapter):
        """Special characters at the end should be replaced."""
        normalized = adapter._normalize_tool_call_id("call_123|@#$")
        assert normalized == "call_123____"
        assert re.match(r"^[a-zA-Z0-9_-]+$", normalized)

    def test_normalize_empty_string(self, adapter):
        """Empty string should return a placeholder."""
        normalized = adapter._normalize_tool_call_id("")
        assert normalized == "empty_call"

    def test_normalize_whitespace_only(self, adapter):
        """Whitespace-only string should normalize to underscores then placeholder if empty."""
        # All spaces become underscores, so not empty after normalization
        normalized = adapter._normalize_tool_call_id("   ")
        assert normalized == "___"
        # Empty after normalization would hit the empty check
        normalized = adapter._normalize_tool_call_id("")
        assert normalized == "empty_call"


# ====================================================================
# GeminiAdapter – tool result merging
# ====================================================================


class TestGeminiAdapterToolResultMerging:
    """Tests for consecutive tool result merging."""

    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()

    def test_consecutive_tool_results_merged(self, adapter):
        """Multiple tool results should be merged into one user turn."""
        tool_use_1 = ToolUseBlock(
            tool_call_id="call_1",
            tool_name="search",
            tool_input={"q": "a"},
        )
        tool_use_2 = ToolUseBlock(
            tool_call_id="call_2",
            tool_name="search",
            tool_input={"q": "b"},
        )
        tool_result_1 = ToolResultBlock(
            tool_call_id="call_1",
            tool_result_content="Result A",
        )
        tool_result_2 = ToolResultBlock(
            tool_call_id="call_2",
            tool_result_content="Result B",
        )
        messages = [
            Message(role=Role.ASSISTANT, content=[tool_use_1, tool_use_2]),
            Message(role=Role.USER, content=[tool_result_1, tool_result_2]),
        ]
        result = adapter.to_provider(messages)

        # Should have 2 contents: 1 model turn + 1 merged user turn
        assert len(result["contents"]) == 2
        assert result["contents"][0]["role"] == "model"
        assert result["contents"][1]["role"] == "user"

        # The user turn should have both function responses
        user_parts = result["contents"][1]["parts"]
        assert len(user_parts) == 2
        assert user_parts[0]["functionResponse"]["name"] == "search"
        assert user_parts[1]["functionResponse"]["name"] == "search"

    def test_single_tool_result_not_merged(self, adapter):
        """Single tool result should create its own user turn."""
        tool_use = ToolUseBlock(
            tool_call_id="call_1",
            tool_name="search",
            tool_input={"q": "test"},
        )
        tool_result = ToolResultBlock(
            tool_call_id="call_1",
            tool_result_content="Result",
        )
        messages = [
            Message(role=Role.ASSISTANT, content=[tool_use]),
            Message(role=Role.USER, content=[tool_result]),
        ]
        result = adapter.to_provider(messages)

        # Single tool result creates its own turn
        assert len(result["contents"]) == 2
        assert result["contents"][1]["role"] == "user"
        assert len(result["contents"][1]["parts"]) == 1

    def test_tool_result_in_non_user_role_raises(self, adapter):
        """ToolResultBlock in non-user role should raise ValueError."""
        tool_result = ToolResultBlock(
            tool_call_id="call_1",
            tool_result_content="Result",
        )
        # ToolResultBlock in ASSISTANT role is invalid for Gemini
        messages = [
            Message(role=Role.ASSISTANT, content=[tool_result]),
        ]
        with pytest.raises(
            ValueError, match="ToolResultBlock must be in a user-role message"
        ):
            adapter.to_provider(messages)


class TestGeminiAdapterStreamAssembly:
    """Streamed function calls must survive assembly into a tool-use turn."""

    @staticmethod
    def _adapter():
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()

    def test_function_call_finish_maps_to_tool_use(self):
        raw = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "search", "args": {"q": "x"}}}
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        chunks = list(self._adapter().from_provider_chunks(raw))

        assert [chunk.type for chunk in chunks] == [
            ContentType.TOOL_USE,
            ContentType.TEXT,
        ]
        assert chunks[1].finish_reason == FinishReason.TOOL_USE

    def test_text_and_finish_in_one_chunk_are_both_emitted(self):
        raw = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "done"}]},
                    "finishReason": "STOP",
                }
            ]
        }
        chunks = list(self._adapter().from_provider_chunks(raw))

        assert [chunk.type for chunk in chunks] == [
            ContentType.TEXT,
            ContentType.TEXT,
        ]
        assert chunks[0].delta == "done"
        assert chunks[1].finish_reason == FinishReason.STOP

    @pytest.mark.asyncio
    async def test_collect_stream_keeps_function_arguments(self):
        from sanityops_agent.llm.stream import collect_stream

        adapter = self._adapter()

        async def events():
            for raw in (
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "search",
                                            "args": {"q": "x"},
                                        }
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
            ):
                for chunk in adapter.from_provider_chunks(raw):
                    yield chunk

        response = await collect_stream(events(), model="gemini-2.5-flash")

        assert response.stop_reason == FinishReason.TOOL_USE
        call = response.get_tool_calls()[0]
        assert call.tool_name == "search"
        assert call.tool_input == {"q": "x"}


class TestGeminiAdapterToolArgumentValidation:
    """Function-call arguments must be JSON objects."""

    @staticmethod
    def _adapter():
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()

    def test_non_object_arguments_raise(self):
        raw = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "search", "args": [1, 2]}}],
                    }
                }
            ]
        }
        with pytest.raises(LLMError, match="search"):
            self._adapter().from_provider_response(raw)

    def test_missing_arguments_default_to_empty_object(self):
        raw = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "clock"}}],
                    }
                }
            ]
        }
        response = self._adapter().from_provider_response(raw)

        call = response.get_tool_calls()[0]
        assert call.tool_name == "clock"
        assert call.tool_input == {}
