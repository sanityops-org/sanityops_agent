from __future__ import annotations

import logging
from typing import Any

from anthropic.lib.streaming import ParsedMessageStreamEvent
from anthropic.types import Message as AnthropicMessage

from sanityops_agent.core.message import (
    ContentType,
    Message,
    Role,
    StreamChunk,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.llm.adapters.base import MessageAdapter
from sanityops_agent.llm.base import LLMResponse
from sanityops_agent.llm.exceptions import (
    AuthenticationError,
    ContextWindowExceededError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    extract_retry_after,
)

logger = logging.getLogger(__name__)


def _coalesce(result: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    """Append an entry, merging it into the previous one when roles coincide.

    Anthropic requires roles to alternate: every tool_result for an assistant
    turn must sit in the SINGLE user message that follows it. The agent loop
    stores one internal message per tool result, so several entries collapse to
    the same wire role and must be joined. Coalescing on the emitted role (not
    the internal one) also merges Role.USER with Role.TOOL and drops messages
    whose blocks converted to nothing, either of which would otherwise leave
    two adjacent same-role entries.

    When merging user entries, tool_result blocks are moved to the FRONT of
    the combined content, per Anthropic's requirement that tool_result blocks
    precede any text. Examples:

        [user(tool_result c1), user(tool_result c2)]
            -> [user(tool_result c1, tool_result c2)]

        [user(tool_result c1), user(text "next...")]
            -> [user(tool_result c1, text "next...")]   # result first

        [user(text), assistant(ErrorBlock-only), user(tool_result c1)]
            -> [user(text), user(tool_result c1)]  # joined: the empty
                                                   # assistant vanished
    """
    if not blocks:
        return
    if result and result[-1]["role"] == role:
        combined = result[-1]["content"] + blocks
        if role == "user":
            # tool_result blocks must come FIRST in the content array.
            tool_results = [b for b in combined if b["type"] == "tool_result"]
            others = [b for b in combined if b["type"] != "tool_result"]
            result[-1]["content"] = tool_results + others
        else:
            result[-1]["content"] = combined
    else:
        result.append({"role": role, "content": blocks})


class AnthropicAdapter(MessageAdapter):
    """Anthropic Claude API message adapter.

    Format mappings:
    - system is a separate top-level key: {\"system\": \"...\", \"messages\": [...]}
    - user messages: {\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"...\"}]}
    - assistant messages: {\"role\": \"assistant\", \"content\": [...]}
    - tool use blocks: {\"type\": \"tool_use\", \"id\": \"...\", \"name\": \"...\", \"input\": {...}}
    - tool result blocks: {\"type\": \"tool_result\", \"tool_use_id\": \"...\", \"content\": \"...\"}
    - thinking blocks: {\"type\": \"thinking\", \"thinking\": \"...\"}
    """

    def to_provider(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Convert messages to Anthropic format.

        Returns: {\"system\": \"...\", \"messages\": [...], \"tools\": [...]}
        """
        system_parts: list[str] = []
        result: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                for block in msg.content:
                    if hasattr(block, "text"):
                        system_parts.append(block.text)
                continue

            # User (and internally equivalent TOOL) messages carry tool results.
            if msg.role in (Role.USER, Role.TOOL):
                blocks: list[dict[str, Any]] = []
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        blocks.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.tool_call_id,
                                "content": block.tool_result_content,
                                "is_error": block.is_error,
                            }
                        )
                    elif isinstance(block, TextBlock):
                        blocks.append({"type": "text", "text": block.text})
                _coalesce(result, "user", blocks)
                continue

            # Assistant messages
            if msg.role == Role.ASSISTANT:
                blocks = []
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        blocks.append({"type": "text", "text": block.text})
                    elif isinstance(block, ThinkingBlock):
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.thinking,
                            }
                        )
                    elif isinstance(block, ToolUseBlock):
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": block.tool_call_id,
                                "name": block.tool_name,
                                "input": block.tool_input,
                            }
                        )
                _coalesce(result, "assistant", blocks)
                continue

            # Fallback
            logger.warning(
                "Falling back to text for unhandled message role %s", msg.role
            )
            _coalesce(
                result,
                "user",
                [{"type": "text", "text": str(msg)}],
            )

        output: dict[str, Any] = {"messages": result}
        if system_parts:
            output["system"] = "\n\n".join(system_parts)
        if tools:
            output["tools"] = self.tools_to_provider(tools)
        return output

    def tools_to_provider(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema or {"type": "object", "properties": {}},
                # Strict schemas must declare every field; tools that do not
                # conform can opt out via provider_config.
                "strict": (t.provider_config or {}).get("strict", True),
            }
            for t in tools
        ]

    def from_provider_response(self, raw: dict) -> LLMResponse:
        content_block = raw.get("content", [])
        stop_reason = self._map_stop_reason(raw.get("stop_reason"))

        content: list = []
        for block in content_block:
            block_type = block.get("type", "text")

            if block_type == "text":
                content.append(TextBlock(text=block.get("text", "")))
            elif block_type == "tool_use":
                content.append(
                    ToolUseBlock(
                        tool_call_id=block.get("id", ""),
                        tool_name=block.get("name", ""),
                        tool_input=block.get("input", {}),
                    )
                )
            elif block_type == "thinking":
                content.append(
                    ThinkingBlock(
                        thinking=block.get("thinking", ""),
                    )
                )

        usage = raw.get("usage")
        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            model=raw.get("model", "unknown"),
            provider="anthropic",
            usage=dict(usage) if usage else {},
            raw_response=raw,
        )

    def from_provider_chunk(self, raw: dict) -> StreamChunk | None:
        event_type = raw.get("type", "")

        if event_type == "message_start":
            return None

        if event_type == "content_block_start":
            block = raw.get("content_block", {})
            block_type = block.get("type", "")
            index = raw.get("index", 0)

            if block_type == "tool_use":
                return StreamChunk(
                    type=ContentType.TOOL_USE,
                    delta={"name": block.get("name", "")},
                    tool_call_id=block.get("id", ""),
                    index=index,
                )
            if block_type == "thinking":
                return StreamChunk(
                    type=ContentType.THINKING,
                    delta="",
                    index=index,
                )
            return None

        if event_type == "content_block_delta":
            delta = raw.get("delta", {})
            delta_type = delta.get("type", "")
            index = raw.get("index", 0)

            if delta_type == "text_delta":
                return StreamChunk(
                    type=ContentType.TEXT,
                    delta=delta.get("text", ""),
                    index=index,
                )
            if delta_type == "thinking_delta":
                return StreamChunk(
                    type=ContentType.THINKING,
                    delta=delta.get("thinking", ""),
                    index=index,
                )
            if delta_type == "input_json_delta":
                # Match the OpenAI chunk shape so one consumer handles both.
                return StreamChunk(
                    type=ContentType.TOOL_USE,
                    delta={"arguments": delta.get("partial_json", "")},
                    index=index,
                )
            return None

        if event_type == "message_delta":
            delta = raw.get("delta", {})
            stop_reason = delta.get("stop_reason")
            # usage arrives on this event alongside the stop reason.
            usage = raw.get("usage")
            if stop_reason or usage:
                return StreamChunk(
                    type=ContentType.TEXT,
                    finish_reason=(
                        self._map_stop_reason(stop_reason) if stop_reason else None
                    ),
                    usage=dict(usage) if usage else None,
                )
            return None

        return None

    @staticmethod
    def map_exception(exception: Exception) -> Exception:
        """Map provider-specific exceptions to internal LLMError types."""
        exc_type_name = type(exception).__name__
        msg = str(exception).lower()

        if exc_type_name in ("AuthenticationError",):
            err: LLMError = AuthenticationError(str(exception))
            err.recoverable = False
            return err
        if exc_type_name in ("RateLimitError",):
            err = RateLimitError(
                str(exception),
                retry_after=extract_retry_after(exception),
            )
            err.recoverable = True
            return err
        if exc_type_name in ("APIError", "APIConnectionError", "OverloadedError"):
            return ProviderUnavailableError(str(exception))
        if exc_type_name in ("BadRequestError",):
            # Prefer a structural signal (HTTP 400) alongside message keywords,
            # which providers change freely. Known Anthropic phrasings:
            # "prompt is too long: ..." and "... context ... exceeds ..." .
            status = getattr(getattr(exception, "response", None), "status_code", None)
            if status is None or status == 400:
                context_window = "too long" in msg or (
                    "context" in msg and "exceed" in msg
                )
                if context_window:
                    return ContextWindowExceededError(str(exception))
        return LLMError(str(exception))

    def from_sdk_response(self, response: AnthropicMessage) -> LLMResponse:
        """Convert Anthropic SDK Message to LLMResponse."""
        content: list = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUseBlock(
                        tool_call_id=block.id or "",
                        tool_name=block.name or "",
                        tool_input=block.input or {},
                    )
                )
            elif block.type == "thinking":
                content.append(
                    ThinkingBlock(
                        thinking=block.thinking or "",
                    )
                )

        usage = response.usage.model_dump() if response.usage else {}
        return LLMResponse(
            content=content,
            stop_reason=self._map_stop_reason(response.stop_reason),
            model=response.model or "unknown",
            provider="anthropic",
            usage=usage,
            raw_response=response.model_dump(),
        )

    def from_sdk_stream_event(
        self, event: ParsedMessageStreamEvent
    ) -> StreamChunk | None:
        """Convert Anthropic SDK MessageStreamEvent to StreamChunk."""
        event_type = event.type

        # Only the raw events (content_block_start / content_block_delta /
        # message_delta) are handled. messages.stream() additionally yields
        # parsed convenience events (text / thinking / input_json) that carry
        # the SAME payloads; handling both emitted every fragment twice, which
        # doubled thinking text and corrupted accumulated tool arguments.

        if event_type == "content_block_start":
            block = getattr(event, "content_block", None)
            if block is None:
                return None
            index = getattr(event, "index", 0)
            if block.type == "tool_use":
                # The tool name is only known here, and the index must match the
                # one later content_block_delta events carry, or the consumer
                # opens a second buffer and drops the name.
                return StreamChunk(
                    type=ContentType.TOOL_USE,
                    delta={"name": getattr(block, "name", "") or ""},
                    tool_call_id=getattr(block, "id", ""),
                    index=index,
                )
            if block.type == "thinking":
                # Block-start marker: the empty delta carries no content, it
                # only opens the buffer at this index so the thinking_delta
                # events that follow land in the right one.
                return StreamChunk(
                    type=ContentType.THINKING,
                    delta="",
                    index=index,
                )
            return None

        if event_type == "content_block_delta":
            block_delta = getattr(event, "delta", None)
            if block_delta is None:
                return None
            delta_type = getattr(block_delta, "type", "")
            index = getattr(event, "index", 0)

            if delta_type == "text_delta":
                return StreamChunk(
                    type=ContentType.TEXT,
                    delta=getattr(block_delta, "text", ""),
                    index=index,
                )
            if delta_type == "thinking_delta":
                return StreamChunk(
                    type=ContentType.THINKING,
                    delta=getattr(block_delta, "thinking", ""),
                    index=index,
                )
            if delta_type == "input_json_delta":
                # Match the OpenAI chunk shape so one consumer handles both.
                return StreamChunk(
                    type=ContentType.TOOL_USE,
                    delta={"arguments": getattr(block_delta, "partial_json", "")},
                    index=index,
                )
            return None

        if event_type == "message_delta":
            stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)
            # usage arrives on this event alongside the stop reason.
            usage = getattr(event, "usage", None)
            usage_dict: dict | None = None
            if usage is not None:
                try:
                    usage_dict = dict(usage)
                except (TypeError, ValueError):
                    # Never expected from the SDK; surface it instead of
                    # silently losing the usage report.
                    logger.warning(
                        "Could not convert message_delta usage %r to dict",
                        usage,
                    )
                    usage_dict = None
            if stop_reason or usage_dict:
                return StreamChunk(
                    type=ContentType.TEXT,
                    finish_reason=(
                        self._map_stop_reason(stop_reason) if stop_reason else None
                    ),
                    usage=usage_dict,
                )
            return None

        # content_block_stop, message_start, etc. — no content to yield
        return None
