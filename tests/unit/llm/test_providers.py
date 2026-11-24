from __future__ import annotations

import importlib
import json
from contextlib import aclosing
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sanityops_agent.core.message import FinishReason, Message, Role, TextBlock
from sanityops_agent.llm.base import LLMResponse
from sanityops_agent.llm.factory import ProviderFactory
from sanityops_agent.llm.providers.anthropic import AnthropicProvider
from sanityops_agent.llm.providers.openai import OpenAIProvider


def _make_httpx_response(
    status_code: int, json_body: dict | None = None
) -> httpx.Response:
    """Create a minimal httpx.Response for SDK exception constructors."""
    return httpx.Response(
        status_code=status_code,
        json=json_body or {},
        request=httpx.Request("POST", "https://example.com"),
    )


def _raw_text_delta(text: str):
    """A real Anthropic content_block_delta event carrying a text fragment."""
    from anthropic.types import RawContentBlockDeltaEvent, TextDelta

    return RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=TextDelta(type="text_delta", text=text),
    )


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicProviderInstantiation:
    """Tests for AnthropicProvider instantiation."""

    def test_can_be_instantiated_with_defaults(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.api_key == "sk-ant-test"
        assert provider.base_url is None
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.adapter is not None

    def test_can_be_instantiated_with_all_args(self):
        provider = AnthropicProvider(
            api_key="sk-ant-123",
            base_url="https://api.anthropic.com",
            model="claude-opus-4-20250514",
        )
        assert provider.api_key == "sk-ant-123"
        assert provider.base_url == "https://api.anthropic.com"
        assert provider.model == "claude-opus-4-20250514"

    def test_has_adapter(self):
        from sanityops_agent.llm.adapters.anthropic_adapter import AnthropicAdapter

        provider = AnthropicProvider(api_key="sk-ant")
        assert isinstance(provider.adapter, AnthropicAdapter)


class TestAnthropicProviderCreate:
    """Tests for AnthropicProvider.create()."""

    @pytest.mark.asyncio
    async def test_create_returns_llm_response(self):
        provider = AnthropicProvider(api_key="sk-ant")

        from anthropic.types import Message as AnthropicMessage

        mock_message = AnthropicMessage.model_validate(
            {
                "id": "msg_test",
                "type": "message",
                "model": "claude-sonnet-4-20250514",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello, world!"}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            }
        )

        async def fake_create(**kwargs):
            return mock_message

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            result = await provider.create(messages, max_tokens=128)

        assert isinstance(result, LLMResponse)
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.content) == 1

    @pytest.mark.asyncio
    async def test_create_raises_on_auth_error(self):
        provider = AnthropicProvider(api_key="sk-ant")

        import anthropic

        resp = _make_httpx_response(401, {"error": {"type": "authentication_error"}})
        err = anthropic.AuthenticationError(
            message="Invalid key", response=resp, body={}
        )

        async def fake_create(**kwargs):
            raise err

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            from sanityops_agent.llm.exceptions import AuthenticationError

            with pytest.raises(AuthenticationError):
                await provider.create(messages, max_tokens=128)

    @pytest.mark.asyncio
    async def test_create_propagates_retry_after(self):
        provider = AnthropicProvider(api_key="sk-ant")

        import anthropic

        resp = _make_httpx_response(429, {"error": {"type": "rate_limit_error"}})
        resp.headers["retry-after"] = "5"
        err = anthropic.RateLimitError(message="Rate limited", response=resp, body={})

        async def fake_create(**kwargs):
            raise err

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            from sanityops_agent.llm.exceptions import RateLimitError

            with pytest.raises(RateLimitError) as exc_info:
                await provider.create(messages, max_tokens=128)
            assert exc_info.value.retry_after == 5

    @pytest.mark.asyncio
    async def test_create_no_extra_body_by_default(self):
        provider = AnthropicProvider(api_key="sk-ant")
        assert provider.extra_body is None

        passed_kwargs = {}

        async def fake_create(**kwargs):
            nonlocal passed_kwargs
            passed_kwargs = kwargs
            mock_msg = MagicMock()
            mock_msg.model = "claude-sonnet-4-20250514"
            mock_msg.stop_reason = "end_turn"
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "hello"
            mock_msg.content = [mock_block]
            mock_msg.usage = None
            mock_msg.model_dump.return_value = {}
            return mock_msg

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            await provider.create(messages, max_tokens=128)

        assert "thinking" not in passed_kwargs
        assert "extra_body" not in passed_kwargs
        assert passed_kwargs["max_tokens"] == 128

    @pytest.mark.asyncio
    async def test_create_respects_init_and_call_max_tokens(self):
        provider = AnthropicProvider(api_key="sk-ant", max_tokens=16384)
        captured: list[dict] = []

        async def fake_create(**kwargs):
            captured.append(kwargs)
            mock_msg = MagicMock()
            mock_msg.model = "claude-sonnet-4-20250514"
            mock_msg.stop_reason = "end_turn"
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "hello"
            mock_msg.content = [mock_block]
            mock_msg.usage = None
            mock_msg.model_dump.return_value = {}
            return mock_msg

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
        with patch.object(provider, "_get_client", return_value=mock_client):
            await provider.create(messages)
            await provider.create(messages, max_tokens=4096)
            await provider.create(messages, max_tokens=None)

        assert [c["max_tokens"] for c in captured] == [16384, 4096, 16384]

    @pytest.mark.asyncio
    async def test_create_forwards_extra_body_as_deep_copy(self):
        extra = {"thinking": {"type": "disabled"}}
        provider = AnthropicProvider(api_key="sk-ant", extra_body=extra)

        passed_kwargs = {}

        async def fake_create(**kwargs):
            nonlocal passed_kwargs
            passed_kwargs = kwargs
            passed_kwargs["extra_body"]["thinking"]["type"] = "mutated"
            mock_msg = MagicMock()
            mock_msg.model = "claude-sonnet-4-20250514"
            mock_msg.stop_reason = "end_turn"
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "hello"
            mock_msg.content = [mock_block]
            mock_msg.usage = None
            mock_msg.model_dump.return_value = {}
            return mock_msg

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            await provider.create(messages, max_tokens=128)

        assert "thinking" not in passed_kwargs
        assert provider.extra_body == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_create_call_override_extra_body(self):
        provider = AnthropicProvider(
            api_key="sk-ant", extra_body={"thinking": {"type": "disabled"}}
        )

        passed_kwargs = {}

        async def fake_create(**kwargs):
            nonlocal passed_kwargs
            passed_kwargs = kwargs
            mock_msg = MagicMock()
            mock_msg.model = "claude-sonnet-4-20250514"
            mock_msg.stop_reason = "end_turn"
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "hello"
            mock_msg.content = [mock_block]
            mock_msg.usage = None
            mock_msg.model_dump.return_value = {}
            return mock_msg

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            await provider.create(messages, extra_body=None, max_tokens=128)

        assert "extra_body" not in passed_kwargs

    @pytest.mark.asyncio
    async def test_create_raises_llm_error_without_retry_on_bad_request(self, caplog):
        import anthropic

        from sanityops_agent.llm.exceptions import LLMError

        provider = AnthropicProvider(
            api_key="sk-ant", extra_body={"thinking": {"type": "disabled"}}
        )
        resp = _make_httpx_response(400, {"error": {"message": "unsupported thinking"}})
        err = anthropic.BadRequestError(
            message="unsupported thinking", response=resp, body={}
        )
        calls = 0

        async def fake_create(**kwargs):
            nonlocal calls
            calls += 1
            raise err

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            with pytest.raises(LLMError, match="400") as exc:
                await provider.create(messages, max_tokens=128)

        assert "unsupported thinking" not in str(exc.value)
        assert "unsupported thinking" not in caplog.text
        assert exc.value.__suppress_context__
        assert calls == 1


class TestAnthropicProviderStream:
    """Tests for AnthropicProvider.stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        provider = AnthropicProvider(api_key="sk-ant")

        # Real streams yield raw content_block_delta events for text. Parsed
        # convenience events (TextEvent) carry the same payload and are not
        # handled separately, to avoid emitting every fragment twice.
        events = [
            _raw_text_delta("Hel"),
            _raw_text_delta("lo"),
        ]

        class MockStream:
            def __init__(self, evts):
                self._evts = iter(evts)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._evts)
                except StopIteration:
                    raise StopAsyncIteration

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = MockStream(events)
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            chunks = []
            async for chunk in provider.stream(messages, max_tokens=128):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo"
        stream_kwargs = mock_client.messages.stream.call_args.kwargs
        assert stream_kwargs["max_tokens"] == 128


# ---------------------------------------------------------------------------
# OpenAIProvider tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderInstantiation:
    """Tests for OpenAIProvider instantiation."""

    def test_can_be_instantiated_with_defaults(self):
        provider = OpenAIProvider(api_key="sk-openai-test")
        assert provider.api_key == "sk-openai-test"
        assert provider.base_url is None
        assert provider.model == "gpt-4o"
        assert provider.adapter is not None

    def test_can_be_instantiated_with_all_args(self):
        provider = OpenAIProvider(
            api_key="sk-123",
            base_url="https://api.openai.com/v1",
            model="gpt-4o-mini",
        )
        assert provider.api_key == "sk-123"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.model == "gpt-4o-mini"

    def test_has_adapter(self):
        from sanityops_agent.llm.adapters.openai_adapter import OpenAIAdapter

        provider = OpenAIProvider(api_key="sk-123")
        assert isinstance(provider.adapter, OpenAIAdapter)


class TestOpenAIProviderCreate:
    """Tests for OpenAIProvider.create()."""

    @pytest.mark.asyncio
    async def test_create_returns_llm_response(self):
        provider = OpenAIProvider(api_key="sk-123")

        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        mock_completion = ChatCompletion(
            id="chatcmpl-test",
            created=1,
            model="gpt-4o",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="Hello, world!",
                    ),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

        passed_kwargs = {}

        async def fake_create(**kwargs):
            nonlocal passed_kwargs
            passed_kwargs = kwargs
            return mock_completion

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            result = await provider.create(messages)

        assert isinstance(result, LLMResponse)
        assert result.model == "gpt-4o"
        assert "max_tokens" not in passed_kwargs

    @pytest.mark.asyncio
    async def test_create_raises_on_auth_error(self):
        provider = OpenAIProvider(api_key="sk-123")

        import openai

        resp = _make_httpx_response(401, {"error": {"message": "Invalid API key"}})
        err = openai.AuthenticationError(
            message="Invalid key", response=resp, body=None
        )

        async def fake_create(**kwargs):
            raise err

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            from sanityops_agent.llm.exceptions import AuthenticationError

            with pytest.raises(AuthenticationError):
                await provider.create(messages)

    @pytest.mark.asyncio
    async def test_create_propagates_retry_after(self):
        provider = OpenAIProvider(api_key="sk-123")

        import openai

        resp = _make_httpx_response(429, {"error": {"message": "Rate limited"}})
        resp.headers["retry-after"] = "5"
        err = openai.RateLimitError(message="Rate limited", response=resp, body=None)

        async def fake_create(**kwargs):
            raise err

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            from sanityops_agent.llm.exceptions import RateLimitError

            with pytest.raises(RateLimitError) as exc_info:
                await provider.create(messages)
            assert exc_info.value.retry_after == 5

    @pytest.mark.asyncio
    async def test_create_no_extra_body_by_default(self):
        provider = OpenAIProvider(api_key="sk-123")
        assert provider.extra_body is None

        passed_kwargs = {}

        async def fake_create(**kwargs):
            nonlocal passed_kwargs
            passed_kwargs = kwargs
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.finish_reason = "stop"
            mock_choice.message.content = "hello"
            mock_choice.message.tool_calls = None
            mock_resp.choices = [mock_choice]
            mock_resp.usage = None
            mock_resp.model = "gpt-4o"
            mock_resp.model_dump.return_value = {}
            return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            await provider.create(messages)

        assert "extra_body" not in passed_kwargs

    @pytest.mark.asyncio
    async def test_create_forwards_extra_body(self):
        provider = OpenAIProvider(
            api_key="sk-123", extra_body={"thinking": {"type": "disabled"}}
        )

        passed_kwargs = {}

        async def fake_create(**kwargs):
            nonlocal passed_kwargs
            passed_kwargs = kwargs
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.finish_reason = "stop"
            mock_choice.message.content = "hello"
            mock_choice.message.tool_calls = None
            mock_resp.choices = [mock_choice]
            mock_resp.usage = None
            mock_resp.model = "gpt-4o"
            mock_resp.model_dump.return_value = {}
            return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            await provider.create(messages)

        assert passed_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_create_call_override_extra_body(self):
        provider = OpenAIProvider(api_key="sk-123", extra_body={"custom": 1})

        passed_kwargs = {}

        async def fake_create(**kwargs):
            nonlocal passed_kwargs
            passed_kwargs = kwargs
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.finish_reason = "stop"
            mock_choice.message.content = "hello"
            mock_choice.message.tool_calls = None
            mock_resp.choices = [mock_choice]
            mock_resp.usage = None
            mock_resp.model = "gpt-4o"
            mock_resp.model_dump.return_value = {}
            return mock_resp

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            await provider.create(
                messages, extra_body={"thinking": {"type": "disabled"}}
            )

        assert passed_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    @pytest.mark.asyncio
    async def test_create_raises_llm_error_without_retry_on_bad_request(self, caplog):
        import openai

        from sanityops_agent.llm.exceptions import LLMError

        provider = OpenAIProvider(
            api_key="sk-123", extra_body={"thinking": {"type": "disabled"}}
        )
        resp = _make_httpx_response(400, {"error": {"message": "bad extra_body"}})
        err = openai.BadRequestError(message="bad extra_body", response=resp, body=None)
        calls = 0

        async def fake_create(**kwargs):
            nonlocal calls
            calls += 1
            raise err

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            with pytest.raises(LLMError, match="400") as exc:
                await provider.create(messages)

        assert "bad extra_body" not in str(exc.value)
        assert "bad extra_body" not in caplog.text
        assert exc.value.__suppress_context__
        assert calls == 1


class _FakeStreamingResponse:
    """Minimal streaming-response double tracking context-manager cleanup."""

    def __init__(self, events, error_after: int | None = None):
        self._events = iter(events)
        self._error_after = error_after
        self._yielded = 0
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True

    async def iter_lines(self):
        for event in self._events:
            if self._error_after is not None and self._yielded >= self._error_after:
                raise RuntimeError("stream interrupted")
            self._yielded += 1
            yield f"data: {json.dumps(event)}"
        yield "data: [DONE]"


class TestOpenAIProviderStream:
    """Tests for OpenAIProvider.stream()."""

    @staticmethod
    def _make_chunk(content: str | None = None, finish_reason: str | None = None):
        delta = {"content": content} if content is not None else {}
        return {
            "id": "chunk-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    @staticmethod
    def _make_provider_with_stream(fake_stream: _FakeStreamingResponse):
        provider = OpenAIProvider(api_key="sk-123")
        mock_client = MagicMock()
        mock_client.chat.completions.with_streaming_response.create.return_value = (
            fake_stream
        )
        mock_client.is_closed.return_value = False
        provider._client = mock_client
        return provider

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        fake_stream = _FakeStreamingResponse(
            [self._make_chunk(content="Hel"), self._make_chunk(content="lo")]
        )
        provider = self._make_provider_with_stream(fake_stream)
        messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]

        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo"
        assert fake_stream.entered is True
        assert fake_stream.closed is True

    @pytest.mark.asyncio
    async def test_stream_closes_on_exception(self):
        fake_stream = _FakeStreamingResponse(
            [self._make_chunk(content="Hel"), self._make_chunk(content="lo")],
            error_after=1,
        )
        provider = self._make_provider_with_stream(fake_stream)
        messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]

        chunks = []
        from sanityops_agent.llm.exceptions import LLMError

        with pytest.raises(LLMError, match="OpenAI request failed"):
            async for chunk in provider.stream(messages):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert fake_stream.entered is True
        assert fake_stream.closed is True

    @pytest.mark.asyncio
    async def test_stream_closes_on_early_aclose(self):
        fake_stream = _FakeStreamingResponse(
            [self._make_chunk(content="Hel"), self._make_chunk(content="lo")]
        )
        provider = self._make_provider_with_stream(fake_stream)
        messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]

        chunks = []
        async with aclosing(provider.stream(messages)) as stream:
            async for chunk in stream:
                chunks.append(chunk)
                break

        assert len(chunks) == 1
        assert fake_stream.entered is True
        assert fake_stream.closed is True

    @pytest.mark.asyncio
    async def test_stream_surfaces_finish_and_usage(self):
        fake_stream = _FakeStreamingResponse(
            [
                self._make_chunk(content="done", finish_reason="stop"),
                {
                    "id": "chunk-usage",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            ]
        )
        provider = self._make_provider_with_stream(fake_stream)
        messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]

        chunks = [chunk async for chunk in provider.stream(messages)]

        assert chunks[0].delta == "done"
        assert chunks[1].finish_reason is FinishReason.STOP
        assert chunks[2].usage == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }


# ---------------------------------------------------------------------------
# Client lifecycle regression tests
# ---------------------------------------------------------------------------
# The openai/anthropic SDKs expose ``is_closed`` as a *method*, not an
# attribute.  A bound method is always truthy, so reading it as an attribute
# leaked a fresh client per ``_get_client()`` call and made ``close()`` a
# silent no-op (surfacing later as "Task exception was never retrieved /
# RuntimeError: Event loop is closed").  The mocks used elsewhere in this file
# set ``is_closed`` as a plain attribute, which hides this bug; these tests use
# a method-style fake matching the real SDK clients.


class _MethodStyleSdkClient:
    """Fake SDK client whose ``is_closed`` is a method (like the real
    AsyncOpenAI / AsyncAnthropic clients), not a plain attribute.
    """

    def __init__(self, **kwargs):
        self._closed = False

    def is_closed(self):
        return self._closed

    async def close(self):
        self._closed = True


class _RaisingCloseClient:
    """Fake SDK client whose ``close()`` raises — like a broken transport.

    ``provider.close()`` is best-effort and must not propagate the error.
    """

    def __init__(self):
        self._closed = False

    def is_closed(self):
        return self._closed

    async def close(self):
        raise RuntimeError("transport failed to close")


class _RaisingIsClosedClient:
    """Fake SDK client whose ``is_closed()`` raises.

    ``_get_client()`` intentionally lets this propagate: an unexpected error
    while inspecting the existing client must not be papered over by silently
    constructing a fresh one.
    """

    def __init__(self):
        # Present for interface parity with the other fakes; is_closed() always
        # raises, so this is never read.
        self._closed = False

    def is_closed(self):
        raise RuntimeError("is_closed failed")

    async def close(self):
        pass


PROVIDER_CLASSES = [OpenAIProvider, AnthropicProvider]


def _patch_sdk_class(provider_cls, monkeypatch, fake_cls):
    """Point the provider module's SDK class at a fake so ``_get_client()``
    never instantiates a real (network-lazy) SDK client in these unit tests.
    """
    module = importlib.import_module(provider_cls.__module__)
    sdk_name = "AsyncOpenAI" if provider_cls == OpenAIProvider else "AsyncAnthropic"
    monkeypatch.setattr(module, sdk_name, fake_cls)


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES, ids=["openai", "anthropic"])
@pytest.mark.asyncio
async def test_close_closes_method_style_client(provider_cls):
    """close() must actually close the SDK client and drop the reference.

    Regression: before the fix the guard ``self._client and not
    self._client.is_closed`` evaluated the bound method (truthy) so ``not
    truthy`` was False and close() never ran.
    """
    provider = provider_cls(api_key="sk-test")
    # The fake is injected directly, not obtained via _get_client(), so no
    # SDK class patch is needed: the test exercises close() in isolation.
    client = _MethodStyleSdkClient()
    provider._client = client
    await provider.close()
    assert client._closed is True, "provider.close() must close the SDK client"
    assert provider._client is None, "provider.close() must clear its reference"


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES, ids=["openai", "anthropic"])
@pytest.mark.asyncio
async def test_close_is_best_effort_when_client_close_raises(provider_cls, caplog):
    """provider.close() must not raise when the SDK client's close() fails.

    close() is best-effort: an error from the underlying client must not
    propagate to the caller, and the provider must still drop its reference.
    """
    provider = provider_cls(api_key="sk-test")
    provider._client = _RaisingCloseClient()
    await provider.close()  # must not raise
    assert "transport failed to close" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)
    assert provider._client is None, "provider.close() must clear its reference"


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES, ids=["openai", "anthropic"])
@pytest.mark.asyncio
async def test_close_is_best_effort_when_is_closed_raises(provider_cls):
    """provider.close() must not raise when the client's is_closed() fails.

    is_closed() is evaluated inside the try block, so an error there is
    swallowed too and the reference is still cleared by the finally.
    """
    provider = provider_cls(api_key="sk-test")
    provider._client = _RaisingIsClosedClient()
    await provider.close()  # must not raise
    assert provider._client is None, "provider.close() must clear its reference"


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES, ids=["openai", "anthropic"])
def test_get_client_propagates_is_closed_error(provider_cls):
    """_get_client() must propagate an is_closed() error, not hide it.

    An unexpected failure while inspecting the existing client is surfaced to
    the caller rather than silently replaced with a fresh client.
    """
    provider = provider_cls(api_key="sk-test")
    provider._client = _RaisingIsClosedClient()
    with pytest.raises(RuntimeError, match="is_closed failed"):
        provider._get_client()
    assert isinstance(provider._client, _RaisingIsClosedClient)


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES, ids=["openai", "anthropic"])
@pytest.mark.asyncio
async def test_get_client_reuses_open_client(provider_cls, monkeypatch):
    """_get_client() must reuse the existing client, not leak a new one per call.

    Regression: before the fix the guard ``self._client is None or
    self._client.is_closed`` was always true for an existing client (bound
    method is truthy), so every call created a fresh SDK client.
    """
    provider = provider_cls(api_key="sk-test")
    _patch_sdk_class(provider_cls, monkeypatch, _MethodStyleSdkClient)
    try:
        first = provider._get_client()
        second = provider._get_client()
        assert first is second, "_get_client() must reuse the existing client"
    finally:
        await provider.close()


@pytest.mark.parametrize("provider_cls", PROVIDER_CLASSES, ids=["openai", "anthropic"])
@pytest.mark.asyncio
async def test_get_client_creates_fresh_after_close(provider_cls, monkeypatch):
    """After close(), _get_client() must create a fresh client."""
    provider = provider_cls(api_key="sk-test")
    _patch_sdk_class(provider_cls, monkeypatch, _MethodStyleSdkClient)
    try:
        first = provider._get_client()
        await provider.close()
        assert provider._client is None
        second = provider._get_client()
        assert second is not first
    finally:
        await provider.close()


# ---------------------------------------------------------------------------
# ProviderFactory integration tests
# ---------------------------------------------------------------------------


class TestProviderFactoryIntegration:
    """Tests for ProviderFactory with concrete providers."""

    @pytest.fixture(autouse=True)
    def _setup_registry(self):
        """Ensure concrete providers are registered, then clean up."""
        ProviderFactory.register("anthropic", AnthropicProvider)
        ProviderFactory.register("openai", OpenAIProvider)
        yield
        ProviderFactory._registry.pop("anthropic", None)
        ProviderFactory._registry.pop("openai", None)

    def _make_config(
        self,
        provider: str = "anthropic",
        api_key: str = "sk-test",
        base_url: str | None = None,
        model: str = "default-model",
    ):
        from sanityops_agent.config import ProviderConfig

        return ProviderConfig(
            PROVIDER=provider,
            API_KEY=api_key,
            BASE_URL=base_url,
            MODEL_ID=model,
        )

    def test_factory_creates_anthropic_provider(self):
        config = self._make_config(
            provider="anthropic", api_key="sk-ant", model="claude-test"
        )
        provider = ProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "sk-ant"
        assert provider.model == "claude-test"

    def test_factory_creates_openai_provider(self):
        config = self._make_config(
            provider="openai", api_key="sk-oai", model="gpt-4o-mini"
        )
        provider = ProviderFactory.create(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.api_key == "sk-oai"
        assert provider.model == "gpt-4o-mini"

    def test_factory_lists_registered_providers(self):
        providers = ProviderFactory.list_providers()
        assert "anthropic" in providers
        assert "openai" in providers

    def test_factory_anthropic_has_correct_default_base_url(self):
        from sanityops_agent.config import ProviderConfig

        config = ProviderConfig(PROVIDER="anthropic", API_KEY="sk-ant")
        provider = ProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)
        assert provider.base_url is None

    def test_factory_openai_has_correct_default_base_url(self):
        from sanityops_agent.config import ProviderConfig

        config = ProviderConfig(PROVIDER="openai", API_KEY="sk-oai")
        provider = ProviderFactory.create(config)
        assert isinstance(provider, OpenAIProvider)
        assert provider.base_url is None


# GeminiProvider tests
# ---------------------------------------------------------------------------


class TestGeminiProviderInstantiation:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

        return GeminiAdapter()

    def test_can_be_instantiated_with_defaults(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gemini-3.5-flash-lite"
        assert provider.adapter is not None

    def test_can_be_instantiated_with_vertex_ai_config(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider(
            api_key="test-key",
            project_id="my-project",
            location="us-central1",
        )
        assert provider.project_id == "my-project"
        assert provider.location == "us-central1"

    def test_thinking_config(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider(
            api_key="test-key",
            thinking_budget=2048,
            store_thinking=True,
        )
        assert provider.thinking_budget == 2048
        assert provider.store_thinking is True


class _FakeGeminiClient:
    """Fake google-genai client: ``close()`` on the sync side, ``aclose()``
    on the ``aio`` side — matching the real SDK's split."""

    def __init__(self):
        self.sync_closed = False
        self.async_closed = False

    def close(self):
        self.sync_closed = True

    @property
    def aio(self):
        return self

    async def aclose(self):
        self.async_closed = True


class _RaisingGeminiClient(_FakeGeminiClient):
    async def aclose(self):
        raise RuntimeError("transport failed to close")


class TestGeminiProviderClose:
    @pytest.mark.asyncio
    async def test_close_closes_async_client_and_clears_reference(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="k")
        client = _FakeGeminiClient()
        provider._client = client
        await provider.close()
        assert client.async_closed is True
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_is_best_effort_when_aclose_raises(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="k")
        provider._client = _RaisingGeminiClient()
        await provider.close()  # must not raise
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_is_noop_when_no_client(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        provider = GeminiProvider(api_key="k")
        await provider.close()  # must not raise
        assert provider._client is None


class TestGeminiProviderCreate:
    @pytest.fixture
    def provider(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_create_returns_llm_response(self, provider):
        from unittest.mock import AsyncMock, MagicMock, patch

        from sanityops_agent.core.message import Role, TextBlock

        # Mock the SDK response with a real dict from model_dump(by_alias=True)
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello, world!"}]},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-3.5-flash-lite",
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        # generate_content is awaited, so it must be an AsyncMock
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_models

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            result = await provider.create(messages)

        assert isinstance(result, LLMResponse)
        assert result.model == "gemini-3.5-flash-lite"
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)

    @pytest.mark.asyncio
    async def test_create_maps_max_tokens_to_max_output_tokens(self, provider):
        from unittest.mock import AsyncMock, MagicMock, patch

        from sanityops_agent.core.message import Role, TextBlock

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
            ],
            "modelVersion": "gemini-3.5-flash-lite",
        }

        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(return_value=mock_response)
        mock_client = MagicMock()
        mock_client.aio.models = mock_models

        messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
        with patch.object(provider, "_get_client", return_value=mock_client):
            await provider.create(messages, max_tokens=4096)

        config = mock_models.generate_content.call_args.kwargs["config"]
        assert config.max_output_tokens == 4096


class TestGeminiProviderStream:
    @pytest.fixture
    def provider(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider

        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider):
        from unittest.mock import MagicMock, patch

        from sanityops_agent.core.message import Role, TextBlock

        # Mock streaming response chunks
        class MockChunk:
            def __init__(self, text):
                self._text = text

            def model_dump(self, **kwargs):
                return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}

        chunks = [MockChunk("Hel"), MockChunk("lo")]

        class MockAsyncStream:
            def __init__(self, items):
                self._items = iter(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

        # generate_content_stream is a coroutine that RETURNS an async iterator,
        # so the mock must be an async function returning MockAsyncStream.
        async def fake_stream(**kwargs):
            return MockAsyncStream(chunks)

        mock_models = MagicMock()
        mock_models.generate_content_stream = fake_stream

        mock_client = MagicMock()
        mock_client.aio.models = mock_models

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            results = []
            async for chunk in provider.stream(messages):
                results.append(chunk)

        assert len(results) == 2
        assert results[0].delta == "Hel"
        assert results[1].delta == "lo"
