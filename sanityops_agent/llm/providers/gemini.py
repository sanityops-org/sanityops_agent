from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from sanityops_agent.core.message import (
    Message,
    StreamChunk,
    ToolDefinition,
)
from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter
from sanityops_agent.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the native google-genai SDK.

    Supports both AI Studio and Vertex AI endpoints.
    Supports: text, thinking, tool_use content blocks.

    Not thread-safe and not safe for concurrent asyncio access: the client
    is created lazily and closed by the agent that owns it. Intended for
    sequential use within a single event loop.
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,  # Not currently used; kept for signature compatibility
        model: str = DEFAULT_MODEL,
        # Vertex AI configuration
        project_id: str | None = None,
        location: str | None = None,
        # Thinking configuration
        thinking_budget: int | None = None,
        store_thinking: bool = False,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.project_id = project_id
        self.location = location
        self.thinking_budget = thinking_budget
        self.store_thinking = store_thinking
        self.timeout = timeout
        self._client: genai.Client | None = None
        self.adapter = GeminiAdapter(store_thinking=store_thinking, model_id=model)

    def _build_config(
        self, payload: dict, kwargs: dict
    ) -> types.GenerateContentConfig:
        """Build a GenerateContentConfig from the adapter payload and kwargs.

        - systemInstruction goes into ``system_instruction``
        - tools (list of {"functionDeclarations": [...]} dicts) are converted
          to ``types.Tool`` objects
        - thinking_budget (when set) enables thinking; include_thoughts=True is
          REQUIRED for the API to return thought parts at all
        """
        config_kwargs: dict = {}

        if payload.get("systemInstruction"):
            config_kwargs["system_instruction"] = types.Content.model_validate(
                payload["systemInstruction"]
            )

        if payload.get("tools"):
            tool_objs = []
            for tool_dict in payload["tools"]:
                declarations = [
                    types.FunctionDeclaration(
                        name=d["name"],
                        description=d.get("description"),
                        parameters_json_schema=d.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    )
                    for d in tool_dict.get("functionDeclarations", [])
                ]
                tool_objs.append(types.Tool(function_declarations=declarations))
            config_kwargs["tools"] = tool_objs

        if self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            )

        # Caller-supplied overrides (temperature, max_output_tokens, ...)
        config_kwargs.update(kwargs)

        return types.GenerateContentConfig(**config_kwargs)

    def _get_client(self) -> genai.Client:
        """Get or create the SDK client.

        Uses Vertex AI mode if project_id is set, otherwise AI Studio mode.
        """
        if self._client is None:
            if self.project_id:
                # Vertex AI mode (uses Application Default Credentials)
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location or "us-central1",
                )
            else:
                # AI Studio mode
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def close(self):
        """Close the underlying client connection.

        Best-effort and idempotent: google-genai's ``Client.close()`` closes
        only the sync client, so ``aio.aclose()`` is used for the async client
        this provider actually calls. Never raises and always clears the
        reference, even if closing fails.
        """
        try:
            if self._client is not None:
                await self._client.aio.aclose()
        except Exception as e:
            logger.warning("Failed to close Gemini client: %s", e, exc_info=True)
        finally:
            self._client = None

    async def create(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Non-streaming call to Gemini API."""
        client = self._get_client()
        payload = self.adapter.to_provider(
            messages,
            tools,
            current_provider="google",
            current_model=self.model,
        )
        config = self._build_config(payload, kwargs)

        contents = [types.Content.model_validate(c) for c in payload["contents"]]

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            raw = response.model_dump(by_alias=True, exclude_none=True)
            return self.adapter.from_provider_response(raw)

        except Exception as e:
            raise self.adapter.map_exception(e)

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming call to Gemini API."""
        client = self._get_client()
        payload = self.adapter.to_provider(
            messages,
            tools,
            current_provider="google",
            current_model=self.model,
        )
        config = self._build_config(payload, kwargs)

        contents = [types.Content.model_validate(c) for c in payload["contents"]]

        try:
            # generate_content_stream is a coroutine that returns an async
            # iterator, so it must be awaited before iteration.
            stream = await client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            async for chunk in stream:
                raw = chunk.model_dump(by_alias=True, exclude_none=True)
                result = self.adapter.from_provider_chunk(raw)
                if result:
                    yield result

        except Exception as e:
            raise self.adapter.map_exception(e)
