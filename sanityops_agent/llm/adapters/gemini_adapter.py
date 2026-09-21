from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from sanityops_agent.core.message import (
    ContentType,
    FinishReason,
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


class GeminiAdapter(MessageAdapter):
    """Google Gemini API message adapter.

    Format mappings:
    - system instruction: {"systemInstruction": {"parts": [{"text": "..."}]}}
    - contents: [{"role": "user" | "model", "parts": [...]}]
    - role: ASSISTANT -> "model"
    - TextBlock -> {"text": "..."} (with optional thoughtSignature)
    - ThinkingBlock -> {"thought": "..."} (Gemini 2.5+ native format, with optional thoughtSignature)
    - ToolUseBlock -> {"functionCall": {"id": "...", "name": "...", "args": {...}}}
    - ToolResultBlock -> {"functionResponse": {"name": "...", "response": {...}}}
    """

    def __init__(self, store_thinking: bool = False, model_id: str | None = None):
        self.store_thinking = store_thinking
        self.model_id = model_id

    def _normalize_tool_call_id(self, tool_call_id: str) -> str:
        """Normalize tool call ID to Gemini-compatible format.

        Gemini requires IDs matching ^[a-zA-Z0-9_-]{1,64}$

        If truncation occurs, we keep the first 56 chars and append a 7-char hash
        to reduce collision risk while staying within 64 char limit.
        """
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_call_id)
        if len(normalized) <= 64:
            return normalized
        # Truncate and add hash suffix to reduce collision risk
        # Use SHA-256 (not MD5) to avoid security scanner warnings
        hash_suffix = hashlib.sha256(tool_call_id.encode()).hexdigest()[:7]
        result = f"{normalized[:56]}_{hash_suffix}"
        logger.debug("Tool call ID truncated: %s -> %s", tool_call_id[:30], result)
        return result

    def _is_same_provider_and_model(
        self, msg: Message, current_provider: str | None, current_model: str | None
    ) -> bool:
        """Check if a message is from the same provider and model.

        Only keep thoughtSignature when provider AND model match exactly.
        """
        return msg.provider == current_provider and msg.model == current_model

    def to_provider(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Convert messages to Gemini API format.

        Supports:
        - thoughtSignature preservation (same provider/model only)
        - Tool call ID normalization for Gemini 3+ compatibility
        - Consecutive function response merging

        Returns: {"systemInstruction": {...}, "contents": [...], "tools": [...]}
        """
        # Extract model info for cross-model signature handling
        current_provider: str | None = kwargs.get("current_provider", "google")
        current_model: str | None = kwargs.get("current_model", self.model_id)

        # First pass: build tool_call_id -> tool_name mapping
        # Gemini requires function NAME in the response, but ToolResultBlock only has tool_call_id
        call_to_name: dict[str, str] = {}
        for msg in messages:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    call_to_name[block.tool_call_id] = block.tool_name

        system_parts: list[dict] = []
        contents: list[dict] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        system_parts.append({"text": block.text})
                    elif isinstance(block, ThinkingBlock):
                        system_parts.append({"thought": block.thinking})
                continue

            # Convert role: ASSISTANT -> "model"
            role = "model" if msg.role == Role.ASSISTANT else "user"
            parts: list[dict] = []

            # Check if message is from same provider/model for signature handling
            is_same_model = self._is_same_provider_and_model(
                msg, current_provider, current_model
            )

            for block in msg.content:
                if isinstance(block, TextBlock):
                    # Handle text with optional thoughtSignature
                    if is_same_model and block.text_signature:
                        parts.append({
                            "text": block.text,
                            "thoughtSignature": block.text_signature,
                        })
                    else:
                        parts.append({"text": block.text})

                elif isinstance(block, ThinkingBlock):
                    # Handle thinking with optional thoughtSignature
                    # Cross-model: convert to plain text, drop signature
                    if is_same_model and block.thinking_signature:
                        parts.append({
                            "thought": True,
                            "text": block.thinking,
                            "thoughtSignature": block.thinking_signature,
                        })
                    elif is_same_model:
                        parts.append({"thought": True, "text": block.thinking})
                    else:
                        # Cross-provider/model: signature unusable, emit as text
                        logger.warning(
                            "Cross-model ThinkingBlock converted to text (signature dropped). "
                            "Thinking semantics may be lost for model %s.",
                            current_model,
                        )
                        parts.append({"text": block.thinking})

                elif isinstance(block, ToolUseBlock):
                    # Normalize tool call ID for Gemini compatibility
                    normalized_id = self._normalize_tool_call_id(block.tool_call_id)
                    part: dict = {
                        "functionCall": {
                            "id": normalized_id,
                            "name": block.tool_name,
                            "args": block.tool_input,
                        }
                    }
                    # Add thoughtSignature if present and same model
                    if is_same_model and block.thought_signature:
                        part["thoughtSignature"] = block.thought_signature
                    parts.append(part)

                elif isinstance(block, ToolResultBlock):
                    # Gemini FunctionResponse requires both id (call ID) and name (function name)
                    # We look up the function name from preceding ToolUseBlock
                    # ToolResultBlocks must be in user-role messages for Gemini
                    if role != "user":
                        logger.warning(
                            "ToolResultBlock in non-user role (%s) - this indicates a bug in message construction",
                            role,
                        )
                        raise ValueError(
                            f"ToolResultBlock must be in a user-role message for Gemini, "
                            f"got role={role!r}. Check message construction logic."
                        )

                    fn_name = call_to_name.get(block.tool_call_id, block.tool_call_id)
                    normalized_id = self._normalize_tool_call_id(block.tool_call_id)
                    part = {
                        "functionResponse": {
                            "id": normalized_id,
                            "name": fn_name,
                            "response": {
                                "result": block.tool_result_content,
                            },
                        }
                    }

                    # Merge consecutive function responses into same user turn
                    # (required for Cloud Code Assist API compatibility)
                    last_content = contents[-1] if contents else None
                    if (
                        last_content
                        and last_content["role"] == "user"
                        and any("functionResponse" in p for p in last_content["parts"])
                    ):
                        last_content["parts"].append(part)
                    else:
                        parts.append(part)

            if parts:
                contents.append({"role": role, "parts": parts})

        result: dict[str, Any] = {"contents": contents}
        if system_parts:
            result["systemInstruction"] = {"parts": system_parts}
        if tools:
            result["tools"] = self.tools_to_provider(tools)
        return result

    def tools_to_provider(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert tools to Gemini function declaration dicts.

        Returns dicts (not types.Tool) to keep this adapter SDK-agnostic and
        unit-testable without importing google.genai.
        """
        return [
            {
                "functionDeclarations": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema
                        or {"type": "object", "properties": {}},
                    }
                    for t in tools
                ]
            }
        ]

    def from_provider_response(self, raw: dict) -> LLMResponse:
        """Convert a Gemini API response dict to LLMResponse.

        The caller passes ``response.model_dump(by_alias=True, exclude_none=True)``
        so keys are camelCase (``functionCall``, ``finishReason``, ``modelVersion``).

        Extracts thoughtSignature from parts when present.
        """
        candidates = raw.get("candidates", [])
        if not candidates:
            return LLMResponse(
                content=[],
                stop_reason=FinishReason.END_TURN,
                model=raw.get("modelVersion", "unknown"),
                provider="google",
                usage={},
                raw_response=raw,
            )

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])
        finish_reason = self._map_stop_reason(candidate.get("finishReason"))

        content: list = []
        for part in content_parts:
            # Thought parts also carry a `text` field, so the thought flag
            # must be checked BEFORE the text branch.
            if part.get("thought") is True:
                if self.store_thinking and part.get("text"):
                    content.append(ThinkingBlock(
                        thinking=part["text"],
                        thinking_signature=part.get("thoughtSignature"),
                    ))
                continue
            if "text" in part and part.get("text") is not None:
                content.append(TextBlock(
                    text=part["text"],
                    text_signature=part.get("thoughtSignature"),
                ))
            elif "functionCall" in part:
                fc = part["functionCall"]
                content.append(
                    ToolUseBlock(
                        tool_call_id=fc.get("id") or fc.get("name", ""),
                        tool_name=fc.get("name", ""),
                        tool_input=fc.get("args") or {},
                        thought_signature=part.get("thoughtSignature"),
                    )
                )

        usage = raw.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            stop_reason=finish_reason,
            model=raw.get("modelVersion", "unknown"),
            provider="google",
            usage=usage,
            raw_response=raw,
        )

    def from_provider_chunk(self, raw: dict) -> StreamChunk | None:
        """Convert a Gemini streaming chunk dict to StreamChunk.

        The caller passes ``chunk.model_dump(by_alias=True, exclude_none=True)``.
        """
        candidates = raw.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        finish_reason = self._map_stop_reason(candidate.get("finishReason"))

        if not parts:
            if finish_reason != FinishReason.END_TURN:
                return StreamChunk(
                    type=ContentType.TEXT,
                    delta=None,
                    finish_reason=finish_reason,
                )
            return None

        part = parts[0]

        # Thought parts also carry `text` — check the flag first.
        if part.get("thought") is True:
            return StreamChunk(
                type=ContentType.THINKING,
                delta=part.get("text") or "",
            )

        if "text" in part and part.get("text") is not None:
            return StreamChunk(
                type=ContentType.TEXT,
                delta=part["text"],
            )

        if "functionCall" in part:
            fc = part["functionCall"]
            return StreamChunk(
                type=ContentType.TOOL_USE,
                delta={"name": fc.get("name"), "args": fc.get("args") or {}},
                tool_call_id=fc.get("id") or fc.get("name"),
            )

        return None

    @staticmethod
    def map_exception(exception: Exception) -> Exception:
        """Map google-genai SDK exceptions to internal LLMError types.

        google-genai raises ``errors.APIError`` (and its subclasses
        ``ClientError`` for 4xx, ``ServerError`` for 5xx). ``APIError`` exposes
        the HTTP status as ``.code`` and the raw response as ``.response``.
        """
        from google.genai import errors as genai_errors

        msg = str(exception).lower()

        if isinstance(exception, genai_errors.APIError):
            code = getattr(exception, "code", None)
            if code == 401 or code == 403:
                err: LLMError = AuthenticationError(str(exception))
                err.recoverable = False
                return err
            if code == 429:
                err = RateLimitError(
                    str(exception),
                    retry_after=extract_retry_after(exception),
                )
                err.recoverable = True
                return err
            if code is not None and code >= 500:
                return ProviderUnavailableError(str(exception))
            if code == 400:
                if "context" in msg or "token" in msg or "too long" in msg:
                    return ContextWindowExceededError(str(exception))
                return LLMError(str(exception))
            return LLMError(str(exception))

        # Name-based fallback for SDK exception types we do not import directly
        exc_type_name = type(exception).__name__
        if exc_type_name in ("AuthenticationError", "PermissionDeniedError"):
            err = AuthenticationError(str(exception))
            err.recoverable = False
            return err
        if exc_type_name in ("ResourceExhausted", "RateLimitError"):
            err = RateLimitError(
                str(exception),
                retry_after=extract_retry_after(exception),
            )
            err.recoverable = True
            return err
        if exc_type_name in ("ServiceUnavailable", "InternalServerError"):
            return ProviderUnavailableError(str(exception))
        if exc_type_name in ("InvalidArgument", "BadRequestError"):
            if "context" in msg or "token" in msg:
                return ContextWindowExceededError(str(exception))
            return LLMError(str(exception))
        return LLMError(str(exception))

    @staticmethod
    def _map_stop_reason(reason: object) -> FinishReason:
        """Map a Gemini finish reason to the internal FinishReason enum.

        Accepts either a plain string or a ``types.FinishReason`` enum member
        (which is what ``model_dump()`` yields).
        """
        if reason is None:
            return FinishReason.END_TURN
        # types.FinishReason is a str-enum: unwrap to its name for lookup
        name = getattr(reason, "name", None) or str(reason)
        mapping: dict[str, FinishReason] = {
            "STOP": FinishReason.STOP,
            "END_TURN": FinishReason.END_TURN,
            "MAX_TOKENS": FinishReason.MAX_TOKENS,
            "SAFETY": FinishReason.CONTENT_FILTER,
            "RECITATION": FinishReason.CONTENT_FILTER,
            "BLOCKLIST": FinishReason.CONTENT_FILTER,
            "PROHIBITED_CONTENT": FinishReason.CONTENT_FILTER,
            "SPII": FinishReason.CONTENT_FILTER,
            "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
            "IMAGE_SAFETY": FinishReason.CONTENT_FILTER,
            "OTHER": FinishReason.ERROR,
        }
        return mapping.get(name.upper(), FinishReason.END_TURN)
