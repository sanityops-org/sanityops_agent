from __future__ import annotations

import logging
from typing import Any

from anthropic.lib.streaming import ParsedMessageStreamEvent
from anthropic.types import Message as AnthropicMessage

from sanityops_agent.core.message import (
    ContentType,
    Message,
    RedactedThinkingBlock,
    Role,
    StreamChunk,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.llm.adapters.base import MessageAdapter, map_sdk_error
from sanityops_agent.llm.base import LLMResponse, normalize_usage
from sanityops_agent.llm.exceptions import (
    LLMError,
)

logger = logging.getLogger(__name__)


def _coalesce(result: list[dict], role: str, blocks: list[dict]) -> None:
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
                                **(
                                    {"signature": block.signature}
                                    if block.signature is not None
                                    else {}
                                ),
                            }
                        )
                    elif isinstance(block, RedactedThinkingBlock):
                        blocks.append({"type": "redacted_thinking", "data": block.data})
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
                if not isinstance(block.get("input"), dict):
                    raise LLMError(
                        f"Expected object arguments for tool call {block.get('id', '')}"
                    )
                content.append(
                    ToolUseBlock(
                        tool_call_id=block.get("id", ""),
                        tool_name=block.get("name", ""),
                        tool_input=block["input"],
                    )
                )
            elif block_type == "thinking":
                content.append(
                    ThinkingBlock(
                        thinking=block.get("thinking", ""),
                        signature=block.get("signature"),
                    )
                )
            elif block_type == "redacted_thinking":
                content.append(RedactedThinkingBlock(data=block["data"]))

        usage = raw.get("usage")
        return LLMResponse(
            content=content,
            stop_reason=stop_reason,
            model=raw.get("model", "unknown"),
            usage=normalize_usage(usage),
            raw_response=raw,
        )

    def from_provider_chunk(self, raw: dict) -> StreamChunk | None:
        event_type = raw.get("type", "")

        if event_type == "message_start":
            usage = raw.get("message", {}).get("usage")
            return StreamChunk(ContentType.TEXT, usage=usage) if usage else None

        if event_type == "content_block_start":
            block = raw.get("content_block", {})
            block_type = block.get("type", "")
            index = raw.get("index", 0)

            if block_type == "tool_use":
                return StreamChunk(
                    type=ContentType.TOOL_USE,
                    tool_call_id=block.get("id", ""),
                    delta={"name": block.get("name"), "input": block.get("input", {})},
                    index=index,
                )
            if block_type == "thinking":
                return StreamChunk(
                    type=ContentType.THINKING,
                    delta=block.get("thinking", ""),
                    signature=block.get("signature"),
                    index=index,
                )
            if block_type == "redacted_thinking":
                return StreamChunk(
                    ContentType.REDACTED_THINKING, delta=block["data"], index=index
                )
            if block_type == "text":
                return StreamChunk(
                    ContentType.TEXT, delta=block.get("text", ""), index=index
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
            if delta_type == "signature_delta":
                return StreamChunk(
                    ContentType.THINKING, signature=delta["signature"], index=index
                )
            if delta_type == "input_json_delta":
                return StreamChunk(
                    type=ContentType.TOOL_USE,
                    delta=delta.get("partial_json", ""),
                    index=index,
                )
            return None

        if event_type == "message_delta":
            delta = raw.get("delta", {})
            stop_reason = delta.get("stop_reason")
            if stop_reason or raw.get("usage"):
                return StreamChunk(
                    type=ContentType.TEXT,
                    usage=raw.get("usage"),
                    finish_reason=self._map_stop_reason(stop_reason)
                    if stop_reason
                    else None,
                )
            return None

        return None

    @staticmethod
    def map_exception(exception: Exception) -> LLMError:
        return map_sdk_error(exception, "Anthropic")

    def from_sdk_response(self, response: AnthropicMessage) -> LLMResponse:
        """Convert Anthropic SDK Message to LLMResponse."""
        return self.from_provider_response(response.model_dump())

    def from_sdk_stream_event(
        self, event: ParsedMessageStreamEvent
    ) -> StreamChunk | None:
        # Parsed convenience events duplicate the raw content deltas.
        if event.type not in (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "message_delta",
        ):
            return None
        chunk = self.from_provider_chunk(event.model_dump(exclude_unset=True))
        if (
            chunk
            and chunk.type == ContentType.TOOL_USE
            and isinstance(chunk.delta, str)
        ):
            chunk.delta = {"arguments": chunk.delta}
        return chunk
