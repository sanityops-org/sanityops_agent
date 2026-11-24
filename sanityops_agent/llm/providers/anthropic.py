from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
)
from anthropic import (
    AsyncAnthropic,
)

from sanityops_agent.core.message import (
    Message,
    StreamChunk,
    ToolDefinition,
)
from sanityops_agent.llm.adapters.anthropic_adapter import AnthropicAdapter
from sanityops_agent.llm.adapters.base import (
    _extract_retry_after as _extract_retry_after,
)
from sanityops_agent.llm.base import LLMProvider, LLMResponse, prepare_request_kwargs
from sanityops_agent.llm.exceptions import (
    APIConnectionError,
    LLMError,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider.

    Uses the official Anthropic SDK Messages API.
    Supports: text, thinking, tool_use content blocks.

    Not thread-safe and not safe for concurrent asyncio access: the client
    is created lazily and closed by the agent that owns it.  Intended for
    sequential use within a single event loop.
    """

    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_MAX_TOKENS = 32768

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        api_version: str = "2023-06-01",
        timeout: float = 120.0,
        extra_body: dict | None = None,
        max_tokens: int | None = None,
        *,
        follow_redirects: bool = True,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.api_version = api_version
        self.timeout = timeout
        self.extra_body = extra_body
        self.max_tokens = max_tokens
        self.follow_redirects = follow_redirects
        self._client: AsyncAnthropic | None = None
        self.adapter = AnthropicAdapter()

    def _build_request_kwargs(self, kwargs: dict) -> dict:
        return prepare_request_kwargs(kwargs, self.extra_body)

    def _build_request_body(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None,
        kwargs: dict,
    ) -> dict:
        payload = self.adapter.to_provider(messages, tools)
        request_kwargs = self._build_request_kwargs(kwargs)
        max_tokens = request_kwargs.pop("max_tokens", None)
        extra_body = request_kwargs.get("extra_body", {})
        extra_limit = extra_body.pop("max_tokens", None)
        if not extra_body:
            request_kwargs.pop("extra_body", None)
        if max_tokens is None:
            max_tokens = self.max_tokens if extra_limit is None else extra_limit
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError(
                "Anthropic requires an explicit positive integer max_tokens"
            )

        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": payload.get("messages", []),
        }
        if "system" in payload:
            body["system"] = payload["system"]
        if "tools" in payload:
            body["tools"] = payload["tools"]
            body.setdefault("tool_choice", {"type": "auto"})

        body.update(request_kwargs)
        return body

    def _get_client(self) -> AsyncAnthropic:
        # Replace a closed client with a fresh one; a closed client is already
        # cleaned up (that is what is_closed() == True means), so no explicit
        # close here — single-context use, see the class docstring.
        if self._client is None or self._client.is_closed():
            timeout = httpx.Timeout(
                connect=10.0, read=self.timeout, write=self.timeout, pool=10.0
            )
            if self.follow_redirects:
                self._client = AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url or self.DEFAULT_BASE_URL,
                    timeout=timeout,
                    max_retries=0,
                )
            else:
                self._client = AsyncAnthropic(
                    api_key=self.api_key,
                    base_url=self.base_url or self.DEFAULT_BASE_URL,
                    timeout=timeout,
                    max_retries=0,
                    http_client=httpx.AsyncClient(
                        follow_redirects=False, trust_env=False, timeout=timeout
                    ),
                )
        return self._client

    async def close(self):
        # Best-effort and idempotent: never raise and never keep a stale
        # reference, even if the client was already closed or its
        # close()/is_closed() raised.  This is cleanup, so a failing client
        # close must not propagate to the caller (e.g. a finally block).
        try:
            if self._client is not None and not self._client.is_closed():
                await self._client.close()
        except Exception:
            logger.warning("Failed to close Anthropic client")
        finally:
            self._client = None

    async def create(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        body = self._build_request_body(messages, tools, kwargs)

        try:
            client = self._get_client()
            message = await client.messages.create(**body)
            return self.adapter.from_sdk_response(message)

        except LLMError:
            raise
        except AnthropicAPIConnectionError:
            raise APIConnectionError("Anthropic connection failed") from None
        except Exception as exc:
            raise self.adapter.map_exception(exc) from None

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        body = self._build_request_body(messages, tools, kwargs)

        try:
            client = self._get_client()
            # message_delta announces the reason, message_stop confirms the message
            # completed. Never expose a tool-use finish before that confirmation.
            finish = None
            open_blocks: set[int] = set()
            async with client.messages.stream(**body) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        open_blocks.add(event.index)
                    elif event.type == "content_block_stop":
                        open_blocks.discard(event.index)
                    elif event.type == "message_stop":
                        if open_blocks:
                            raise LLMError(
                                "Anthropic stream ended during a content block"
                            )
                        if finish is None:
                            raise LLMError(
                                "Anthropic stream ended without a stop reason"
                            )
                        yield finish
                        finish = None
                        continue
                    chunk = self.adapter.from_sdk_stream_event(event)
                    if chunk is None:
                        continue
                    if chunk.finish_reason is not None:
                        finish = chunk
                    else:
                        yield chunk
            if finish is not None:
                raise LLMError("Anthropic stream ended before message_stop")

        except LLMError:
            raise
        except AnthropicAPIConnectionError:
            raise APIConnectionError("Anthropic connection failed") from None
        except Exception as exc:
            raise self.adapter.map_exception(exc) from None
