from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import cast

from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    AsyncOpenAI,
    AsyncStream,
)
from openai.types.chat import ChatCompletionChunk

from sanityops_agent.core.message import (
    Message,
    StreamChunk,
    ToolDefinition,
)
from sanityops_agent.llm.adapters.base import (
    _extract_retry_after as _extract_retry_after,
)
from sanityops_agent.llm.adapters.openai_adapter import OpenAIAdapter
from sanityops_agent.llm.base import LLMProvider, LLMResponse, prepare_request_kwargs
from sanityops_agent.llm.exceptions import (
    APIConnectionError,
    LLMError,
)

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider (works with OpenAI, Azure, vLLM, etc.).

    Uses the official OpenAI SDK. Compatible with any server that
    implements the OpenAI chat completions API.

    Not thread-safe and not safe for concurrent asyncio access: the client
    is created lazily and closed by the agent that owns it.  Intended for
    sequential use within a single event loop.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "gpt-4o",
        timeout: float = 120.0,
        extra_body: dict | None = None,
        *,
        follow_redirects: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.extra_body = extra_body
        self.follow_redirects = follow_redirects
        self._client: AsyncOpenAI | None = None
        self.adapter = OpenAIAdapter()

    def _build_request_kwargs(self, kwargs: dict) -> dict:
        return prepare_request_kwargs(kwargs, self.extra_body)

    def _get_client(self) -> AsyncOpenAI:
        # A closed client is replaced with a fresh one; it is already closed
        # so needs no cleanup here (single-context use, see class docstring).
        if self._client is None or self._client.is_closed():
            if self.follow_redirects:
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url or self.DEFAULT_BASE_URL,
                    timeout=float(self.timeout),  # type: ignore[arg-type]
                    max_retries=0,  # agent layer handles retries
                )
            else:
                from openai import DefaultAsyncHttpxClient

                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url or self.DEFAULT_BASE_URL,
                    timeout=float(self.timeout),  # type: ignore[arg-type]
                    max_retries=0,  # agent layer handles retries
                    http_client=DefaultAsyncHttpxClient(
                        follow_redirects=False,
                        trust_env=False,
                        timeout=self.timeout,
                    ),
                )
        return self._client

    async def close(self):
        # Best-effort and idempotent: never raise and never keep a stale
        # reference, even if the client was already closed or its
        # close()/is_closed() raised.  This is cleanup (typically from a
        # finally block), so a failing client close must not propagate.
        try:
            if self._client is not None and not self._client.is_closed():
                await self._client.close()
        except Exception:
            logger.warning("Failed to close OpenAI client")
        finally:
            self._client = None

    async def create(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        payload = self.adapter.to_provider(messages, tools)
        request_kwargs = self._build_request_kwargs(kwargs)

        try:
            client = self._get_client()
            completion = await client.chat.completions.create(
                model=self.model,
                messages=payload["messages"],
                tools=payload.get("tools"),  # type: ignore[arg-type]
                **request_kwargs,
            )
            return self.adapter.from_sdk_response(completion)

        except LLMError:
            raise
        except OpenAIAPIConnectionError:
            raise APIConnectionError("OpenAI connection failed") from None
        except Exception as exc:
            raise self.adapter.map_exception(exc) from None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        payload = self.adapter.to_provider(messages, tools)
        request_kwargs = self._build_request_kwargs(kwargs)

        if (self.base_url or self.DEFAULT_BASE_URL).rstrip(
            "/"
        ) == self.DEFAULT_BASE_URL:
            request_kwargs.setdefault("stream_options", {"include_usage": True})

        try:
            client = self._get_client()
            stream = await client.chat.completions.create(
                model=self.model,
                messages=payload["messages"],
                tools=payload.get("tools"),  # type: ignore[arg-type]
                stream=True,
                **request_kwargs,
            )
            # With stream=True, the return type is AsyncStream[ChatCompletionChunk]
            # mypy sees a union; we use a cast to narrow the type
            async with cast(AsyncStream[ChatCompletionChunk], stream) as stream_iter:
                async for chunk in stream_iter:
                    for part in self.adapter.from_sdk_stream_events(chunk):
                        yield part

        except LLMError:
            raise
        except OpenAIAPIConnectionError:
            raise APIConnectionError("OpenAI connection failed") from None
        except Exception as exc:
            raise self.adapter.map_exception(exc) from None
