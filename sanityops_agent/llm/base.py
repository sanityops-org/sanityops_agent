from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sanityops_agent.core.message import (
    ContentBlockUnion,
    FinishReason,
    Message,
    StreamChunk,
    ToolDefinition,
    ToolUseBlock,
)


class LLMProvider(ABC):
    """LLM Provider abstract base class - async only, no retry logic"""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "",
        **kwargs,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @abstractmethod
    async def create(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Non-streaming call"""
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming call - returns async iterator, must support cleanup on error"""
        pass

    async def close(self):
        """Close underlying connections, release resources"""
        pass


@dataclass
class LLMResponse:
    content: list[ContentBlockUnion]
    stop_reason: FinishReason
    model: str
    provider: str = "unknown"  # Provider name for thoughtSignature handling
    usage: dict = field(default_factory=dict)
    raw_response: dict = field(default_factory=dict, repr=False)

    def get_tool_calls(self) -> list[ToolUseBlock]:
        """Return tool use blocks (specific types)"""
        return [block for block in self.content if isinstance(block, ToolUseBlock)]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.get_tool_calls()) > 0


def prepare_request_kwargs(
    kwargs: dict[str, Any], default_extra_body: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Copy per-call kwargs and inject provider default extra_body when omitted."""
    request_kwargs = dict(kwargs)
    if "extra_body" not in request_kwargs:
        if default_extra_body:
            request_kwargs["extra_body"] = copy.deepcopy(default_extra_body)
    elif not request_kwargs["extra_body"]:
        request_kwargs.pop("extra_body", None)
    else:
        request_kwargs["extra_body"] = copy.deepcopy(request_kwargs["extra_body"])
    return request_kwargs


def normalize_usage(usage: dict | None) -> dict:
    """Preserve upstream counters and derive a total only for complete usage."""
    result = dict(usage or {})
    if (
        result.get("input_tokens") is not None
        and result.get("output_tokens") is not None
    ):
        result["total_tokens"] = sum(
            result.get(key) or 0
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            )
        )
    elif (
        result.get("total_tokens") is None
        and result.get("prompt_tokens") is not None
        and result.get("completion_tokens") is not None
    ):
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result
