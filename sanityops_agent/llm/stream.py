"""Assemble a complete provider stream before exposing executable tool calls."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from sanityops_agent.core.message import (
    ContentBlockUnion,
    ContentType,
    FinishReason,
    RedactedThinkingBlock,
    StreamChunk,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from sanityops_agent.llm.base import LLMResponse, normalize_usage
from sanityops_agent.llm.exceptions import LLMError


def _reject_non_json_number(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def parse_tool_arguments(arguments: str | None, call_id: str) -> dict:
    if not isinstance(arguments, str):
        raise LLMError(f"Missing arguments for tool call {call_id}")
    try:
        parsed = json.loads(arguments, parse_constant=_reject_non_json_number)
    except ValueError:
        raise LLMError(f"Invalid JSON arguments for tool call {call_id}") from None
    if not isinstance(parsed, dict):
        raise LLMError(f"Expected object arguments for tool call {call_id}")
    return parsed


@dataclass
class _ToolCall:
    call_id: str | None = None
    name: list[str] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)
    initial_input: dict | None = None
    signature: str | None = None

    def add(self, chunk: StreamChunk) -> None:
        if chunk.tool_call_id:
            if self.call_id and self.call_id != chunk.tool_call_id:
                raise LLMError("Conflicting tool call IDs at a stream index")
            self.call_id = chunk.tool_call_id
        if chunk.signature is not None:
            self.signature = chunk.signature
        delta = chunk.delta
        if isinstance(delta, str):
            self.arguments.append(delta)
        elif isinstance(delta, dict):
            for key, parts in (("name", self.name), ("arguments", self.arguments)):
                if delta.get(key) is not None:
                    if not isinstance(delta[key], str):
                        raise LLMError(f"Invalid streamed tool {key}")
                    parts.append(delta[key])
            for key in ("input", "args"):
                if key in delta:
                    if not isinstance(delta[key], dict):
                        raise LLMError("Expected a tool input object")
                    self.initial_input = delta[key]

    def finish(self) -> ToolUseBlock:
        name = "".join(self.name)
        if not self.call_id or not name:
            raise LLMError("Streamed tool call is missing its ID or name")
        if self.arguments or self.initial_input is None:
            arguments = parse_tool_arguments("".join(self.arguments), self.call_id)
        else:
            arguments = self.initial_input
        return ToolUseBlock(
            self.call_id, name, arguments, thought_signature=self.signature
        )


async def collect_stream(
    stream: AsyncIterator[StreamChunk],
    *,
    model: str,
    on_chunk: Callable[[StreamChunk], None] | None = None,
) -> LLMResponse:
    """Consume all chunks, including trailing usage, and close the source on exit."""
    calls: dict[int, _ToolCall] = {}
    fragments: dict[tuple[int, ContentType], list[str]] = {}
    signatures: dict[int, list[str]] = {}
    usage: dict = {}
    finish: FinishReason | None = None
    try:
        async for chunk in stream:
            if on_chunk is not None:
                on_chunk(chunk)
            if chunk.usage:
                usage.update({k: v for k, v in chunk.usage.items() if v is not None})
            if chunk.type == ContentType.ERROR:
                raise LLMError("Provider reported a stream error")
            has_content = (
                chunk.delta is not None
                or chunk.signature is not None
                or chunk.tool_call_id is not None
            )
            if finish is not None and (has_content or chunk.finish_reason is not None):
                raise LLMError("Unexpected content or finish after stream termination")
            if has_content:
                parts = fragments.setdefault((chunk.index, chunk.type), [])
                if chunk.type == ContentType.TOOL_USE:
                    calls.setdefault(chunk.index, _ToolCall()).add(chunk)
                elif isinstance(chunk.delta, str):
                    parts.append(chunk.delta)
                elif chunk.delta is not None:
                    raise LLMError("Invalid streamed content delta")
                if chunk.signature is not None:
                    signatures.setdefault(chunk.index, []).append(chunk.signature)
            if chunk.finish_reason is not None:
                finish = chunk.finish_reason
    finally:
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()

    if finish is None or finish == FinishReason.ERROR:
        raise LLMError("Stream did not finish successfully")
    if bool(calls) != (finish == FinishReason.TOOL_USE):
        raise LLMError("Stream finish reason does not match tool calls")

    content: list[ContentBlockUnion] = []
    seen_ids: set[str] = set()
    # Claude indexes all content blocks. OpenAI indexes only parallel tool calls;
    # at equal indices preserve arrival order for its text and reasoning deltas.
    for (index, kind), parts in sorted(fragments.items(), key=lambda item: item[0][0]):
        value = "".join(parts)
        if kind == ContentType.TEXT:
            signature = "".join(signatures[index]) if index in signatures else None
            content.append(TextBlock(value, text_signature=signature))
        elif kind == ContentType.THINKING:
            signature = "".join(signatures[index]) if index in signatures else None
            content.append(ThinkingBlock(value, signature=signature))
        elif kind == ContentType.REDACTED_THINKING:
            content.append(RedactedThinkingBlock(value))
        elif kind == ContentType.TOOL_USE:
            call = calls[index].finish()
            if call.tool_call_id in seen_ids:
                raise LLMError("Duplicate streamed tool call ID")
            seen_ids.add(call.tool_call_id)
            content.append(call)
    return LLMResponse(content, finish, model, usage=normalize_usage(usage))
