from __future__ import annotations

import json
import logging
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from sanityops_agent.core.message import (
    ContentType,
    FinishReason,
    Message,
    Role,
    StreamChunk,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.llm.adapters.base import MessageAdapter, merge_consecutive_messages
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


def _dump_tool_input(tool_input: dict) -> str:
    """Serialize tool arguments; a non-JSON-serializable input is stringified
    rather than raising mid-conversation."""
    try:
        return json.dumps(tool_input)
    except (TypeError, ValueError):
        logger.warning("Tool input is not JSON-serializable; stringifying")
        return json.dumps({"raw": str(tool_input)})


class OpenAIAdapter(MessageAdapter):
    """OpenAI API message adapter.

    Format mappings:
    - system messages: {\"role\": \"system\", \"content\": \"...\"}
    - user messages: {\"role\": \"user\", \"content\": [...]} (array for multi-content)
    - assistant messages: {\"role\": \"assistant\", \"content\": \"...\" or [blocks]}
    - tool calls in message: {\"role\": \"assistant\", \"tool_calls\": [{\"id\": \"...\", \"function\": {...}}]}
    - tool results: {\"role\": \"tool\", \"tool_call_id\": \"...\", \"content\": \"...\"}

    Known limitation: a user message with several text blocks is flattened to
    plain text for compatibility (logged as a warning); tool results in user
    messages are unaffected, but any other non-text block in a user message is
    downgraded to its repr. Multi-block user content should be joined by the
    caller when it matters.
    """

    def to_provider(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict:
        result: list[dict] = []

        # OpenAI needs an assistant message carrying tool_calls, followed by one
        # role:"tool" message per call, so the tool-result messages must stay
        # separate (unlike Anthropic). Merge assistant messages only: that
        # absorbs the bare assistant placeholder SubAgent appends, which would
        # otherwise emit an empty assistant turn between tool_calls and results.
        messages = merge_consecutive_messages(
            messages,
            frozenset({Role.ASSISTANT}),
        )

        for msg in messages:
            role = self._map_role(msg.role)

            if role == "assistant":
                # OpenAI requires one role:"tool" message PER tool_call_id,
                # each answering the assistant message that carries tool_calls.
                # ToolResultBlocks arriving on an assistant message (the agent
                # loop stores results as user messages, but other callers may
                # not) are therefore extracted here, BEFORE the assistant turn
                # is emitted, so the tool messages always follow it.
                for tr in (b for b in msg.content if isinstance(b, ToolResultBlock)):
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id,
                            "content": tr.tool_result_content,
                        }
                    )

                texts = [b for b in msg.content if isinstance(b, TextBlock)]
                tool_calls = [b for b in msg.content if isinstance(b, ToolUseBlock)]

                content_parts: list[dict] = []
                for t in texts:
                    content_parts.append({"type": "text", "text": t.text})

                # A message with no convertible content must not be emitted as an
                # assistant turn: {"role":"assistant","content":""} with no
                # tool_calls sits invalidly between tool_calls and its results.
                # The tool messages extracted above are already appended, so
                # skipping the assistant shell loses nothing.
                if not content_parts and not tool_calls:
                    continue

                entry: dict[str, Any] = {"role": role}
                entry["content"] = content_parts if content_parts else None
                if tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": _dump_tool_input(tc.tool_input),
                            },
                        }
                        for tc in tool_calls
                    ]
                result.append(entry)

            elif role == "tool":
                # tool result blocks (explicit Role.TOOL)
                entry = {"role": "tool"}
                blocks = []
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        entry["tool_call_id"] = block.tool_call_id
                        blocks.append(block.tool_result_content)
                    elif isinstance(block, TextBlock):
                        blocks.append(block.text)
                entry["content"] = "\n".join(str(b) for b in blocks)
                result.append(entry)
            else:
                # user or system messages
                entry = {"role": role}
                content_list: list[dict] = []
                tool_results = []

                for block in msg.content:
                    if isinstance(block, TextBlock):
                        content_list.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseBlock):
                        content_list.append(
                            {
                                "type": "tool_call",
                                "tool_call_id": block.tool_call_id,
                                "name": block.tool_name,
                                "input": block.tool_input,
                            }
                        )
                    elif isinstance(block, ToolResultBlock):
                        tool_results.append(block)
                    else:
                        content_list.append({"type": "text", "text": str(block)})

                # OpenAI: user messages containing only ToolResultBlock(s)
                # must be converted to role: "tool" messages
                if role == "user" and tool_results and not content_list:
                    for tr in tool_results:
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": tr.tool_call_id,
                                "content": tr.tool_result_content,
                            }
                        )
                    continue

                # System messages: plain string; user messages: array-of-blocks
                if (
                    role == "system"
                    and len(content_list) == 1
                    and content_list[0]["type"] == "text"
                ):
                    entry["content"] = content_list[0]["text"]
                else:
                    # OPENAI fix: user messages with multiple blocks
                    # Some versions may not support array format, flatten to text
                    if role == "user" and len(content_list) > 1:
                        # Flatten to single text for compatibility.
                        #
                        # Known limitation: text block structure is lost. Text
                        # blocks survive as joined text; tool results in user
                        # messages are handled separately (emitted as
                        # role:"tool") and are NOT affected, but any other
                        # non-text block in a user message is downgraded to its
                        # repr. Callers relying on structured multi-part
                        # content on OpenAI need to join text themselves.
                        non_text = sum(1 for i in content_list if i["type"] != "text")
                        logger.warning(
                            "Flattening user message with %d blocks (%d "
                            "non-text) to plain text; text block structure is "
                            'lost (tool results unaffected — emitted as "tool")',
                            len(content_list),
                            non_text,
                        )
                        text_parts = []
                        for item in content_list:
                            if item["type"] == "text":
                                text_parts.append(item["text"])
                            else:
                                text_parts.append(str(item))
                        entry["content"] = "\n".join(text_parts)
                    else:
                        entry["content"] = content_list if content_list else ""
                result.append(entry)

        output: dict[str, Any] = {"messages": result}
        if tools:
            output["tools"] = self.tools_to_provider(tools)
        return output

    def tools_to_provider(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        **(t.input_schema or {}),
                        # A schema may declare additionalProperties itself;
                        # otherwise strict-mode compatibility defaults it off.
                        "additionalProperties": (t.input_schema or {}).get(
                            "additionalProperties",
                            (t.provider_config or {}).get(
                                "additionalProperties", False
                            ),
                        ),
                    },
                    "strict": (t.provider_config or {}).get("strict", True),
                },
            }
            for t in tools
        ]

    def from_provider_response(self, raw: dict) -> LLMResponse:
        choice = raw.get("choices", [{}])[0]
        finish_reason = self._map_stop_reason(choice.get("finish_reason"))

        message = choice.get("message", {})
        content: list = []

        text = message.get("content", "")
        if text:
            content.append(TextBlock(text=text))

        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            content.append(
                ToolUseBlock(
                    tool_call_id=tc.get("id", ""),
                    tool_name=fn.get("name", ""),
                    tool_input=arguments,
                )
            )

        usage = raw.get("usage")
        return LLMResponse(
            content=content,
            stop_reason=finish_reason,
            model=raw.get("model", "unknown"),
            provider="openai",
            usage=dict(usage) if usage else {},
            raw_response=raw,
        )

    def from_provider_chunk(self, raw: dict) -> StreamChunk | None:
        choices = raw.get("choices", [])
        if not choices:
            # Usage-only chunk (include_usage=True) — no content to yield
            return None
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = self._map_stop_reason(choice.get("finish_reason"))

        # Check for tool call delta first
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            fn = tc.get("function", {})
            return StreamChunk(
                type=ContentType.TOOL_USE,
                delta={"name": fn.get("name", ""), "arguments": fn.get("arguments", "")},
                tool_call_id=tc.get("id"),
            )

        # Thinking/reasoning content (some OpenAI-compatible APIs send it)
        thinking = delta.get("reasoning_content") or delta.get("reasoning")
        if thinking is not None and thinking != "":
            return StreamChunk(
                type=ContentType.THINKING,
                delta=thinking,
            )

        text = delta.get("content")
        if text is not None and text != "":
            return StreamChunk(
                type=ContentType.TEXT,
                delta=text,
            )

        # If only finish reason, emit it
        if finish_reason != FinishReason.END_TURN:
            return StreamChunk(
                type=ContentType.TEXT,
                delta=None,
                finish_reason=finish_reason,
            )

        return None

    @staticmethod
    def _map_role(role: Role) -> str:
        mapping = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.TOOL: "tool",
        }
        return mapping.get(role, "user")

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
        if exc_type_name in ("APIError", "APIConnectionError"):
            return ProviderUnavailableError(str(exception))
        if exc_type_name in ("BadRequestError",):
            # Prefer a structural signal (HTTP 400) alongside the message
            # keywords, which providers change freely.
            status = getattr(getattr(exception, "response", None), "status_code", None)
            if status is None or status == 400:
                context_window = (
                    "context" in msg and ("exceed" in msg or "too_long" in msg)
                ) or ("maximum context" in msg)
                if context_window:
                    return ContextWindowExceededError(str(exception))
        return LLMError(str(exception))

    def from_sdk_response(self, response: ChatCompletion) -> LLMResponse:
        """Convert OpenAI SDK ChatCompletion to LLMResponse."""
        choice = response.choices[0]
        finish_reason = self._map_stop_reason(choice.finish_reason)

        message = choice.message
        content: list = []

        if message.content:
            content.append(TextBlock(text=message.content))

        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.type != "function":
                    continue
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                content.append(
                    ToolUseBlock(
                        tool_call_id=tc.id or "",
                        tool_name=tc.function.name or "",
                        tool_input=arguments,
                    )
                )

        usage = response.usage.model_dump() if response.usage else {}
        return LLMResponse(
            content=content,
            stop_reason=finish_reason,
            model=response.model or "unknown",
            provider="openai",
            usage=usage,
            raw_response=response.model_dump(),
        )

    def from_sdk_stream_event(self, chunk: ChatCompletionChunk) -> StreamChunk | None:
        """Convert OpenAI SDK ChatCompletionChunk to StreamChunk."""
        choices = chunk.choices
        if not choices:
            return None

        choice = choices[0]
        delta = choice.delta
        finish_reason = self._map_stop_reason(choice.finish_reason)

        # Tool call delta
        if delta.tool_calls:
            tc = delta.tool_calls[0]
            fn = tc.function
            if fn is not None:
                return StreamChunk(
                    type=ContentType.TOOL_USE,
                    delta={"name": fn.name or "", "arguments": fn.arguments or ""},
                    tool_call_id=tc.id,
                )
            return StreamChunk(
                type=ContentType.TOOL_USE,
                delta={},
                tool_call_id=tc.id,
            )

        # Thinking/reasoning content
        reasoning = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        if reasoning is not None and reasoning != "":
            return StreamChunk(
                type=ContentType.THINKING,
                delta=reasoning,
            )

        # Text delta
        if delta.content is not None and delta.content != "":
            return StreamChunk(
                type=ContentType.TEXT,
                delta=delta.content,
            )

        # Finish reason only
        if finish_reason != FinishReason.END_TURN:
            return StreamChunk(
                type=ContentType.TEXT,
                delta=None,
                finish_reason=finish_reason,
            )

        return None
