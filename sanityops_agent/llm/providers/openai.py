from __future__ import annotations

import logging
from collections.abc import AsyncIterator

import httpx
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
)
from openai import (
    APIStatusError as OpenAIAPIStatusError,
)
from openai import (
    AsyncOpenAI,
    BadRequestError,
)
from openai import (
    AuthenticationError as OpenAIAuthError,
)
from openai import (
    RateLimitError as OpenAIRateLimit,
)

from sanityops_agent.core.message import (
    Message,
    StreamChunk,
    ToolDefinition,
)
from sanityops_agent.llm.adapters.openai_adapter import OpenAIAdapter
from sanityops_agent.llm.base import LLMProvider, LLMResponse
from sanityops_agent.llm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    ContextWindowExceededError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    extract_retry_after,
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
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client: AsyncOpenAI | None = None
        self.adapter = OpenAIAdapter()

    def _get_client(self) -> AsyncOpenAI:
        # A closed client is replaced with a fresh one; it is already closed
        # so needs no cleanup here (single-context use, see class docstring).
        if self._client is None or self._client.is_closed():
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or self.DEFAULT_BASE_URL,
                timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=self.timeout, pool=10.0),
                max_retries=0,  # agent layer handles retries
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
        except Exception as e:
            logger.warning("Failed to close LLM client: %s", e, exc_info=True)
        finally:
            self._client = None

    async def create(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_client()
        payload = self.adapter.to_provider(messages, tools)

        try:
            completion = await client.chat.completions.create(
                model=self.model,
                messages=payload["messages"],
                tools=payload.get("tools"),
                **kwargs,
            )
            return self.adapter.from_sdk_response(completion)

        except OpenAIAuthError as e:
            raise AuthenticationError(str(e))
        except OpenAIRateLimit as e:
            raise RateLimitError(str(e), retry_after=extract_retry_after(e))
        except BadRequestError as e:
            msg = str(e).lower()
            if "context" in msg and ("exceed" in msg or "maximum" in msg or "too_long" in msg):
                raise ContextWindowExceededError(str(e))
            raise LLMError(str(e))
        except OpenAIAPIConnectionError as e:
            raise APIConnectionError(str(e))
        except OpenAIAPIStatusError as e:
            if e.status_code >= 500:
                raise ProviderUnavailableError(str(e))
            raise LLMError(str(e))

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        client = self._get_client()
        payload = self.adapter.to_provider(messages, tools)

        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=payload["messages"],
                tools=payload.get("tools"),
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                sc = self.adapter.from_sdk_stream_event(chunk)
                if sc:
                    yield sc

        except OpenAIAuthError as e:
            raise AuthenticationError(str(e))
        except OpenAIRateLimit as e:
            raise RateLimitError(str(e), retry_after=extract_retry_after(e))
        except BadRequestError as e:
            msg = str(e).lower()
            if "context" in msg and ("exceed" in msg or "maximum" in msg or "too_long" in msg):
                raise ContextWindowExceededError(str(e))
            raise LLMError(str(e))
        except OpenAIAPIConnectionError as e:
            raise APIConnectionError(str(e))
        except OpenAIAPIStatusError as e:
            if e.status_code >= 500:
                raise ProviderUnavailableError(str(e))
            raise LLMError(str(e))
