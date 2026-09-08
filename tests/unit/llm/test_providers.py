from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sanityops_agent.core.message import Message, Role, TextBlock
from sanityops_agent.llm.base import LLMResponse
from sanityops_agent.llm.factory import ProviderFactory
from sanityops_agent.llm.providers.anthropic import AnthropicProvider
from sanityops_agent.llm.providers.openai import OpenAIProvider


def _make_httpx_response(status_code: int, json_body: dict | None = None) -> httpx.Response:
    """Create a minimal httpx.Response for SDK exception constructors."""
    return httpx.Response(
        status_code=status_code,
        json=json_body or {},
        request=httpx.Request("POST", "https://example.com"),
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
        mock_message = AnthropicMessage.model_validate({
            "id": "msg_test",
            "type": "message",
            "model": "claude-sonnet-4-20250514",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello, world!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
        })

        async def fake_create(**kwargs):
            return mock_message

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            result = await provider.create(messages)

        assert isinstance(result, LLMResponse)
        assert result.model == "claude-sonnet-4-20250514"
        assert len(result.content) == 1

    @pytest.mark.asyncio
    async def test_create_raises_on_auth_error(self):
        provider = AnthropicProvider(api_key="sk-ant")

        import anthropic
        resp = _make_httpx_response(401, {"error": {"type": "authentication_error"}})
        err = anthropic.AuthenticationError(message="Invalid key", response=resp, body={})

        async def fake_create(**kwargs):
            raise err

        mock_client = MagicMock()
        mock_client.messages.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            from sanityops_agent.llm.exceptions import AuthenticationError
            with pytest.raises(AuthenticationError):
                await provider.create(messages)


class TestAnthropicProviderStream:
    """Tests for AnthropicProvider.stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        provider = AnthropicProvider(api_key="sk-ant")

        from anthropic.lib.streaming import TextEvent

        events = [
            TextEvent(type="text", text="Hel", snapshot="Hel"),
            TextEvent(type="text", text="lo", snapshot="Hello"),
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
            async for chunk in provider.stream(messages):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo"


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
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

        async def fake_create(**kwargs):
            return mock_completion

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            result = await provider.create(messages)

        assert isinstance(result, LLMResponse)
        assert result.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_create_raises_on_auth_error(self):
        provider = OpenAIProvider(api_key="sk-123")

        import openai
        resp = _make_httpx_response(401, {"error": {"message": "Invalid API key"}})
        err = openai.AuthenticationError(message="Invalid key", response=resp, body=None)

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


class TestOpenAIProviderStream:
    """Tests for OpenAIProvider.stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        provider = OpenAIProvider(api_key="sk-123")

        from openai.types.chat import ChatCompletionChunk
        from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

        def make_chunk(content=None, finish_reason=None):
            delta = ChoiceDelta(content=content)
            choice = Choice(delta=delta, index=0, finish_reason=finish_reason)
            chunk = ChatCompletionChunk(
                id="chunk-test",
                created=1,
                model="gpt-4o",
                object="chat.completion.chunk",
                choices=[choice],
            )
            return chunk

        async def fake_create(**kwargs):
            chunks = [
                make_chunk(content="Hel"),
                make_chunk(content="lo"),
            ]

            class AsyncIter:
                def __init__(self, items):
                    self._items = iter(items)

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    try:
                        return next(self._items)
                    except StopIteration:
                        raise StopAsyncIteration

            stream = AsyncIter(chunks)
            # Mimic AsyncStream.response for close() in cleanup
            stream.response = MagicMock()
            return stream

        mock_client = MagicMock()
        mock_client.chat.completions.create = fake_create
        mock_client.is_closed = False

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            chunks = []
            async for chunk in provider.stream(messages):
                chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo"


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
async def test_close_is_best_effort_when_client_close_raises(provider_cls):
    """provider.close() must not raise when the SDK client's close() fails.

    close() is best-effort: an error from the underlying client must not
    propagate to the caller, and the provider must still drop its reference.
    """
    provider = provider_cls(api_key="sk-test")
    provider._client = _RaisingCloseClient()
    await provider.close()  # must not raise
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

    def _make_config(self, provider: str = "anthropic", api_key: str = "sk-test",
                     base_url: str | None = None, model: str = "default-model"):
        from sanityops_agent.config import ProviderConfig
        return ProviderConfig(
            PROVIDER=provider,
            API_KEY=api_key,
            BASE_URL=base_url,
            MODEL_ID=model,
        )

    def test_factory_creates_anthropic_provider(self):
        config = self._make_config(provider="anthropic", api_key="sk-ant", model="claude-test")
        provider = ProviderFactory.create(config)
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "sk-ant"
        assert provider.model == "claude-test"

    def test_factory_creates_openai_provider(self):
        config = self._make_config(provider="openai", api_key="sk-oai", model="gpt-4o-mini")
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
