from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from sanityops_agent.core.message import (
    FinishReason,
    Message,
    Role,
    StreamChunk,
    ToolDefinition,
)
from sanityops_agent.llm.base import LLMResponse


def merge_consecutive_messages(
    messages: list[Message],
    merge_roles: frozenset[Role],
) -> list[Message]:
    """Drop empty-content messages and collapse adjacent same-role messages.

    The agent loop stores one internal Message per tool result, so a turn with
    N parallel tool calls produces N consecutive same-role messages. Providers
    require a specific grouping of those results, so each adapter normalizes
    the history before emitting its payload.

    - Messages with empty ``content`` are dropped. This removes the bare
      assistant placeholder that SubAgent appends, which would otherwise emit
      an invalid empty assistant turn.
    - Adjacent messages whose role is in ``merge_roles`` are concatenated in
      order; a role change always starts a new message, so the alternation of
      the emitted payload is preserved. ``merge_roles`` is typed as a
      ``frozenset[Role]`` — pass a literal set and it is coerced by the
      annotation-checking caller; membership tests on it are O(1).

    Pure: returns new Message objects with new content lists. The input
    messages and their block lists are never mutated, and the content blocks
    are shared by reference (adapters only read them). Idempotent.
    """
    out: list[Message] = []
    for msg in messages:
        if not msg.content:
            continue
        if out and out[-1].role == msg.role and msg.role in merge_roles:
            out[-1] = Message(
                role=out[-1].role,
                content=out[-1].content + list(msg.content),
                provider=out[-1].provider,
                model=out[-1].model,
                raw_response=out[-1].raw_response,
            )
        else:
            out.append(
                Message(
                    role=msg.role,
                    content=list(msg.content),
                    provider=msg.provider,
                    model=msg.model,
                    raw_response=msg.raw_response,
                )
            )
    return out


class MessageAdapter(ABC):
    """Abstract base for message <-> provider format adapters.

    Subclasses must implement:
    - to_provider(messages, tools=None, **kwargs) -> dict
    - from_provider_response(response) -> LLMResponse
    - from_provider_chunk(chunk) -> StreamChunk | None
    - map_exception(exception) -> Exception
    """

    @abstractmethod
    def to_provider(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Convert internal messages to provider API format."""
        ...

    @abstractmethod
    def from_provider_response(self, response: dict) -> LLMResponse:
        """Convert provider API response to LLMResponse."""
        ...

    @abstractmethod
    def from_provider_chunk(self, chunk: dict) -> StreamChunk | None:
        """Convert a single provider streaming chunk to StreamChunk."""
        ...

    @abstractmethod
    def map_exception(self, exception: Exception) -> Exception:
        """Map provider-specific exceptions to internal LLMError types."""
        ...

    def from_sdk_response(self, response: Any) -> LLMResponse:
        """Convert a typed SDK response object to LLMResponse.

        Concrete subclasses should implement this with specific type annotations.
        Default raises NotImplementedError for backward compatibility.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.from_sdk_response is not implemented"
        )

    def from_sdk_stream_event(self, event: Any) -> StreamChunk | None:
        """Convert a typed SDK streaming event to StreamChunk.

        Concrete subclasses should implement this with specific type annotations.
        Default raises NotImplementedError for backward compatibility.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.from_sdk_stream_event is not implemented"
        )

    @staticmethod
    def _map_stop_reason(reason: str | None) -> FinishReason:
        """Map provider stop reason to FinishReason enum."""
        if reason is None:
            return FinishReason.END_TURN
        mapping: dict[str, FinishReason] = {
            "stop": FinishReason.STOP,
            "end_turn": FinishReason.END_TURN,
            "stop_sequence": FinishReason.STOP_SEQUENCE,
            "tool_use": FinishReason.TOOL_USE,
            "tool_calls": FinishReason.TOOL_USE,
            "function_call": FinishReason.TOOL_USE,
            "length": FinishReason.MAX_TOKENS,
            "max_tokens": FinishReason.MAX_TOKENS,
            "content_filter": FinishReason.CONTENT_FILTER,
            "error": FinishReason.ERROR,
        }
        return mapping.get(reason, FinishReason.END_TURN)
