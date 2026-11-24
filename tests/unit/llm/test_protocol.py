import json
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message as ClaudeMessage
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from sanityops_agent.core.message import (
    Message,
    Role,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
)
from sanityops_agent.llm.adapters.anthropic_adapter import AnthropicAdapter
from sanityops_agent.llm.adapters.openai_adapter import OpenAIAdapter
from sanityops_agent.llm.exceptions import LLMError
from sanityops_agent.llm.providers.anthropic import AnthropicProvider
from sanityops_agent.llm.providers.openai import OpenAIProvider


class LocalStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def __aiter__(self):
        for event in self.events:
            yield event

    async def iter_lines(self):
        for event in self.events:
            raw = event.model_dump(exclude_none=True)
            yield f"data: {json.dumps(raw)}"
        yield "data: [DONE]"


def function(index, name=None, arguments=None, id=None):
    return {
        "index": index,
        "id": id,
        "function": {"name": name, "arguments": arguments},
    }


def event(*calls, text=None, finish=None, usage=None):
    choices = (
        []
        if usage
        else [
            {
                "index": 0,
                "delta": {"content": text, "tool_calls": calls},
                "finish_reason": finish,
            }
        ]
    )
    return ChatCompletionChunk.model_validate(
        {
            "choices": choices,
            "usage": usage,
            "created": 0,
            "model": "fixture",
            "object": "chat.completion.chunk",
            "id": "fixture",
        }
    )


def openai_stream(events, *, base_url=None):
    provider = OpenAIProvider(api_key="fixture", base_url=base_url)
    stream = LocalStream(events)
    client = MagicMock()
    client.is_closed.return_value = False
    client.chat.completions.with_streaming_response.create = MagicMock(
        return_value=stream
    )
    provider._client = client
    return provider, stream


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,request_usage", [(None, True), ("https://example.test/v1", False)]
)
async def test_collects_parallel_calls_and_final_usage(url, request_usage):
    from sanityops_agent.llm.stream import collect_stream

    provider, stream = openai_stream(
        [
            event(
                function(0, "search", '{"q":'),
                function(1, "clock", "{", "b"),
                text="Checking",
            ),
            event(
                function(1, arguments="}"),
                function(0, arguments='"hi"}', id="a"),
                finish="tool_calls",
            ),
            event(
                usage={"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 9}
            ),
        ],
        base_url=url,
    )
    response = await collect_stream(provider.stream([]), model="fixture")
    assert response.content == [
        TextBlock("Checking"),
        ToolUseBlock("a", "search", {"q": "hi"}),
        ToolUseBlock("b", "clock", {}),
    ]
    assert response.usage["total_tokens"] == 9
    kwargs = provider._client.chat.completions.with_streaming_response.create.call_args.kwargs
    assert kwargs.get("stream_options", {}).get("include_usage", False) is request_usage
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events",
    [
        [event(function(0, "clock", "{}", "a"))],
        [event(function(0, "clock", "{", "a"), finish="tool_calls")],
        [event(function(0, "clock", "[]", "a"), finish="tool_calls")],
        [event(function(0, "clock", "", "a"), finish="tool_calls")],
        [event(function(0, "clock", "{}", "a"), finish="length")],
        [
            event(function(0, "clock", "{}", "a")),
            event(function(0, id="b"), finish="tool_calls"),
        ],
        [
            event(
                function(0, "clock", "{}", "a"),
                function(1, "clock", "{}", "a"),
                finish="tool_calls",
            )
        ],
    ],
)
async def test_incomplete_stream_cannot_return_tool_calls(events):
    from sanityops_agent.llm.stream import collect_stream

    provider, stream = openai_stream(events)
    with pytest.raises(LLMError):
        await collect_stream(provider.stream([]), model="fixture")
    assert stream.closed


@pytest.mark.parametrize("args", ["{", "[]", "", "null", '{"count":NaN}'])
def test_sync_openai_rejects_invalid_arguments(args):
    raw = {
        "id": "test",
        "created": 0,
        "object": "chat.completion",
        "model": "fixture",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "call-a",
                            "function": {"name": "clock", "arguments": args},
                        }
                    ],
                },
            }
        ],
    }
    adapter = OpenAIAdapter()
    with pytest.raises(LLMError, match="call-a"):
        adapter.from_provider_response(raw)
    with pytest.raises(LLMError, match="call-a"):
        adapter.from_sdk_response(ChatCompletion.model_validate(raw))


def test_strict_schema_checks_nested_objects_and_preserves_optional_fields():
    adapter = OpenAIAdapter()
    leaf = {
        "type": "object",
        "properties": {"count": {"type": "integer", "default": 3}},
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/leaf"}}},
        "required": ["items"],
        "additionalProperties": False,
        "$defs": {"leaf": leaf},
    }
    tool = ToolDefinition("counter", "", schema)
    default = adapter.tools_to_provider([tool])[0]["function"]
    assert default["parameters"] == schema
    assert not default.get("strict")
    tool.provider_config = {"strict": True}
    with pytest.raises(ValueError, match="counter"):
        adapter.tools_to_provider([tool])
    leaf["required"] = ["count"]
    assert adapter.tools_to_provider([tool])[0]["function"]["strict"] is True
    del leaf["additionalProperties"]
    with pytest.raises(ValueError, match="counter"):
        adapter.tools_to_provider([tool])


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 0, -2, True, "5"])
@pytest.mark.parametrize("streaming", [True, False])
async def test_invalid_claude_limit_rejected_before_client(limit, streaming):
    provider = AnthropicProvider(api_key="fixture")
    provider._get_client = MagicMock(
        side_effect=AssertionError("unexpected client creation")
    )
    with pytest.raises(ValueError, match="max_tokens"):
        if streaming:
            await anext(provider.stream([], max_tokens=limit))
        else:
            await provider.create([], max_tokens=limit)


def test_claude_extra_tokens_priority_and_caller_copy():
    default = {"max_tokens": 16, "metadata": {"test": "default"}}
    call_extra = {"max_tokens": 32, "metadata": {"test": "call"}}
    provider = AnthropicProvider(api_key="fixture", max_tokens=4, extra_body=default)
    for kwargs, expected in [
        ({}, 16),
        ({"extra_body": call_extra}, 32),
        ({"max_tokens": 64, "extra_body": call_extra}, 64),
        ({"extra_body": None}, 4),
    ]:
        body = provider._build_request_body([], None, kwargs)
        assert body["max_tokens"] == expected
        assert "max_tokens" not in body.get("extra_body", {})
        if "extra_body" in body:
            body["extra_body"]["metadata"]["test"] = "changed"
    assert default["metadata"]["test"] == "default"
    assert call_extra == {"max_tokens": 32, "metadata": {"test": "call"}}


CLAUDE_BLOCKS = [
    {"type": "text", "text": "first"},
    {"type": "thinking", "thinking": "step", "signature": "abc"},
    {"type": "redacted_thinking", "data": "opaque-fixture"},
    {"type": "tool_use", "id": "a", "name": "clock", "input": {}},
]
CLAUDE_USAGE = {
    "input_tokens": 12,
    "cache_read_input_tokens": 7,
    "cache_creation_input_tokens": 3,
    "output_tokens": 5,
}


def claude_message(content, usage, stop_reason):
    return {
        "id": "m",
        "role": "assistant",
        "model": "fixture",
        "type": "message",
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage,
    }


def assert_claude_round_trip(response):
    history = [Message(Role.ASSISTANT, response.content)]
    assert (
        AnthropicAdapter().to_provider(history)["messages"][0]["content"]
        == CLAUDE_BLOCKS
    )
    assert response.usage["total_tokens"] == 27
    for key, count in CLAUDE_USAGE.items():
        assert response.usage[key] == count


def test_claude_sync_round_trip():
    adapter = AnthropicAdapter()
    raw = claude_message(CLAUDE_BLOCKS, CLAUDE_USAGE, "tool_use")
    assert_claude_round_trip(adapter.from_provider_response(raw))
    assert_claude_round_trip(
        adapter.from_sdk_response(ClaudeMessage.model_validate(raw))
    )


def claude_events():
    from types import SimpleNamespace

    from anthropic.types import (
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
    )

    yield RawMessageStartEvent(
        type="message_start",
        message=claude_message([], {**CLAUDE_USAGE, "output_tokens": 1}, None),
    )
    for index, block in enumerate(CLAUDE_BLOCKS):
        initial = {**block, "thinking": "", "signature": ""} if index == 1 else block
        yield RawContentBlockStartEvent(
            type="content_block_start", index=index, content_block=initial
        )
        if index == 1:
            for delta in [
                {"type": "thinking_delta", "thinking": "step"},
                {"type": "signature_delta", "signature": "a"},
                {"type": "signature_delta", "signature": "bc"},
            ]:
                yield RawContentBlockDeltaEvent(
                    type="content_block_delta", index=index, delta=delta
                )
            yield SimpleNamespace(type="thinking", thinking="step", snapshot="step")
        yield RawContentBlockStopEvent(type="content_block_stop", index=index)
    for count in (3, 5):
        yield RawMessageDeltaEvent(
            type="message_delta",
            delta={"stop_reason": "tool_use" if count == 5 else None},
            usage={"output_tokens": count},
        )
    yield RawMessageStopEvent(type="message_stop")


@pytest.mark.asyncio
@pytest.mark.parametrize("truncate", ["message", "block", None])
async def test_claude_stream_round_trip(truncate):
    from sanityops_agent.llm.stream import collect_stream

    events = list(claude_events())
    if truncate == "message":
        events = events[:-1]
    elif truncate == "block":
        events = [e for e in events if e.type != "content_block_stop" or e.index != 3]
    stream = LocalStream(events)
    provider = AnthropicProvider(api_key="fixture", max_tokens=16)
    client = MagicMock()
    client.messages.stream.return_value = stream
    provider._get_client = lambda: client
    if truncate:
        with pytest.raises(LLMError):
            await collect_stream(provider.stream([]), model="fixture")
    else:
        response = await collect_stream(provider.stream([]), model="fixture")
        assert_claude_round_trip(response)
    assert stream.closed


def test_strict_tool_rejects_unsupported_composition_in_an_array_item():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {"type": "array", "items": {"allOf": [{"type": "integer"}]}}
        },
        "required": ["items"],
    }
    tool = ToolDefinition("items", "", schema, provider_config={"strict": True})
    with pytest.raises(ValueError, match="items"):
        OpenAIAdapter().tools_to_provider([tool])


@pytest.mark.parametrize("value", [None, [], "text"])
def test_claude_rejects_non_object_input(value):
    raw = claude_message(
        [{"type": "tool_use", "name": "clock", "id": "c", "input": value}],
        CLAUDE_USAGE,
        "tool_use",
    )
    with pytest.raises(LLMError):
        AnthropicAdapter().from_provider_response(raw)


@pytest.mark.asyncio
async def test_final_text_and_finish_share_event_without_reported_usage():
    from sanityops_agent.llm.stream import collect_stream

    provider, stream = openai_stream([event(text="Done", finish="stop")])
    observed = []
    response = await collect_stream(
        provider.stream([]), model="fixture", on_chunk=observed.append
    )
    assert response.content == [TextBlock("Done")]
    assert response.usage == {}
    assert observed[-1].finish_reason == response.stop_reason
    assert stream.closed


@pytest.mark.parametrize("adapter", [AnthropicAdapter(), OpenAIAdapter()])
def test_adapter_errors_exclude_response_payload(adapter):
    class BadRequestError(Exception):
        pass

    error = BadRequestError("fixture-private-text")
    converted = adapter.map_exception(error)
    assert isinstance(converted, LLMError)
    assert "fixture-private-text" not in str(converted)


def test_strict_schema_rejects_excessive_nesting():
    node = {"type": "object", "properties": {}, "additionalProperties": False}
    for _ in range(25):
        node = {
            "type": "object",
            "properties": {"child": node},
            "required": ["child"],
            "additionalProperties": False,
        }
    tool = ToolDefinition("deep", "", node)
    tool.provider_config = {"strict": True}

    with pytest.raises(ValueError, match="nested too deeply"):
        OpenAIAdapter().tools_to_provider([tool])
