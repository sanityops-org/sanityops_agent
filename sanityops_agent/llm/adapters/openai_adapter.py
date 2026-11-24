from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from sanityops_agent.core.message import (
    ContentType,
    Message,
    Role,
    StreamChunk,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.llm.adapters.base import (
    MessageAdapter,
    map_sdk_error,
    merge_consecutive_messages,
)
from sanityops_agent.llm.base import LLMResponse, normalize_usage
from sanityops_agent.llm.exceptions import (
    LLMError,
)
from sanityops_agent.llm.stream import parse_tool_arguments

logger = logging.getLogger(__name__)


def _dump_tool_input(tool_input: dict) -> str:
    """Serialize tool arguments; a non-JSON-serializable input is stringified
    rather than raising mid-conversation."""
    try:
        return json.dumps(tool_input)
    except (TypeError, ValueError):
        logger.warning("Tool input is not JSON-serializable; stringifying")
        return json.dumps({"raw": str(tool_input)})


def _check_strict_schema(schema: dict, tool_name: str) -> None:
    """Check object constraints in schema positions, including local definitions."""

    max_depth = 20
    unsupported = {
        "allOf",
        "oneOf",
        "not",
        "dependentRequired",
        "dependentSchemas",
        "if",
        "then",
        "else",
    }

    def visit(node: dict, depth: int) -> None:
        if depth > max_depth:
            raise ValueError(f"Strict tool {tool_name} schema is nested too deeply")
        if disallowed := (unsupported & node.keys()):
            keys = ", ".join(sorted(disallowed))
            raise ValueError(
                f"Strict tool {tool_name} uses unsupported schema keywords: {keys}"
            )
        kind = node.get("type")
        is_object = kind == "object" or isinstance(kind, list) and "object" in kind
        if is_object or "properties" in node:
            properties = node.get("properties", {})
            required = node.get("required", [])
            if (
                node.get("additionalProperties") is not False
                or not isinstance(properties, dict)
                or not isinstance(required, list)
                or not all(isinstance(key, str) for key in required)
                or set(required) != set(properties)
            ):
                raise ValueError(
                    f"Strict tool {tool_name} requires all object fields and additionalProperties=false"
                )
        for key in ("properties", "$defs", "definitions", "patternProperties"):
            for child in node.get(key, {}).values():
                if isinstance(child, dict):
                    visit(child, depth + 1)
        for key in ("items", "anyOf", "prefixItems"):
            children = node.get(key, [])
            for child in children if isinstance(children, list) else [children]:
                if isinstance(child, dict):
                    visit(child, depth + 1)

    if schema.get("type") != "object" or "anyOf" in schema:
        raise ValueError(f"Strict tool {tool_name} requires a root object")
    visit(schema, 0)


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
        result = []
        for tool in tools:
            config = tool.provider_config or {}
            schema = dict(tool.input_schema or {})
            strict = config.get("strict", False)
            if type(strict) is not bool:
                raise ValueError(f"Tool {tool.name} requires a boolean strict setting")
            if "additionalProperties" in config:
                schema.setdefault(
                    "additionalProperties", config["additionalProperties"]
                )
            if strict:
                _check_strict_schema(schema, tool.name)
            else:
                schema.setdefault("additionalProperties", False)
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                        "strict": strict,
                    },
                }
            )
        return result

    def from_provider_response(self, raw: dict) -> LLMResponse:
        choice = raw.get("choices", [{}])[0]
        finish_reason = self._map_stop_reason(choice.get("finish_reason"))

        message = choice.get("message", {})
        content: list = []

        text = message.get("content", "")
        if text:
            content.append(TextBlock(text=text))

        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            if tc.get("type", "function") != "function":
                continue
            arguments = parse_tool_arguments(fn.get("arguments"), tc.get("id", ""))
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
            usage=normalize_usage(usage),
            raw_response=raw,
        )

    def from_provider_chunk(self, raw: dict) -> StreamChunk | None:
        return next(self.from_provider_chunks(raw), None)

    def from_provider_chunks(self, raw: dict) -> Iterator[StreamChunk]:
        choices = raw.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}
            thinking = delta.get("reasoning_content") or delta.get("reasoning")
            if thinking:
                yield StreamChunk(ContentType.THINKING, delta=thinking)
            if delta.get("content"):
                yield StreamChunk(ContentType.TEXT, delta=delta["content"])
            for call in delta.get("tool_calls") or []:
                index = call.get("index")
                if type(index) is not int or index < 0:
                    raise LLMError("OpenAI streamed a tool call without a valid index")
                yield StreamChunk(
                    ContentType.TOOL_USE,
                    delta=call.get("function") or {},
                    tool_call_id=call.get("id"),
                    index=index,
                )
            if choice.get("finish_reason") is not None:
                yield StreamChunk(
                    ContentType.TEXT,
                    finish_reason=self._map_stop_reason(choice["finish_reason"]),
                )
        if raw.get("usage"):
            yield StreamChunk(ContentType.TEXT, usage=raw["usage"])

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
    def map_exception(exception: Exception) -> LLMError:
        return map_sdk_error(exception, "OpenAI")

    def from_sdk_response(self, response: ChatCompletion) -> LLMResponse:
        """Convert OpenAI SDK ChatCompletion to LLMResponse."""
        return self.from_provider_response(response.model_dump())

    def from_sdk_stream_events(
        self, chunk: ChatCompletionChunk
    ) -> Iterator[StreamChunk]:
        """Yield each semantic part of an SDK event without dropping siblings."""
        return self.from_provider_chunks(chunk.model_dump(exclude_none=True))

    def from_sdk_stream_event(self, chunk: ChatCompletionChunk) -> StreamChunk | None:
        return next(self.from_sdk_stream_events(chunk), None)
