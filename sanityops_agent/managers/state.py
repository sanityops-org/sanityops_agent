from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from sanityops_agent.core.context import AgentContext
from sanityops_agent.core.message import (
    ContentBlockUnion,
    ContentType,
    ErrorBlock,
    Message,
    RedactedThinkingBlock,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from sanityops_agent.managers._files import write_private_json

logger = logging.getLogger(__name__)


class StateManager:
    """JSON format save/load context, compatible with old JSONL format.

    Content blocks are round-tripped by type so tool_use/tool_result pairing
    survives a save/load cycle; the previous text-only serialization dropped
    every tool block and made restored histories unusable with tools.

    Saves are atomic (temp file + os.replace), so a crash mid-write leaves the
    previous state file intact rather than corrupting it.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        state_path: str | Path = "agent_state.json",
        *,
        session_id: str | None = None,
        system_prompt: str = "",
    ):
        self.state_path = Path(state_path)
        self.session_id = session_id
        self.system_prompt = system_prompt

    async def save(self, context: AgentContext) -> None:
        """Save context to JSON"""
        data = {
            "version": self.STATE_VERSION,
            "session_id": self.session_id,
            "messages": self._serialize_messages(context.messages),
            "total_tokens": context.total_tokens,
            "has_compacted": context.has_compacted,
            "last_summary": context.last_summary,
        }
        self._check_directory()
        write_private_json(self.state_path, data)

    def _check_directory(self) -> None:
        if self.session_id is None or os.name != "posix":
            return
        parent = self.state_path.parent.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.geteuid()
            or parent.st_mode & 0o077
        ):
            raise PermissionError(
                "Managed session state requires an owner-only directory"
            )

    async def load(self) -> AgentContext:
        """Load context from JSON"""
        self._check_directory()
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(self.state_path, flags)
            with os.fdopen(fd, encoding="utf-8") as source:
                attributes = os.fstat(source.fileno())
                if not stat.S_ISREG(attributes.st_mode):
                    raise ValueError("State must be a regular file")
                if (
                    self.session_id is not None
                    and os.name == "posix"
                    and (
                        attributes.st_uid != os.geteuid() or attributes.st_mode & 0o077
                    )
                ):
                    raise PermissionError("Managed session state must be owner-only")
                data = json.load(source)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"State file not found: {self.state_path}") from exc
        except json.JSONDecodeError as e:
            raise ValueError(
                f"State file {self.state_path} is corrupt or not valid JSON: {e}"
            ) from e
        if not isinstance(data, dict):
            raise ValueError(
                f"State file {self.state_path} has an unexpected structure: "
                "expected a JSON object with a 'messages' list"
            )
        if data.get("version", 0) not in (0, self.STATE_VERSION):
            raise ValueError("Unsupported conversation state version")
        if data.get("session_id") != self.session_id:
            raise PermissionError("Conversation state belongs to a different session")
        messages_data = data.get("messages", [])
        if not isinstance(messages_data, list):
            raise ValueError(
                f"State file {self.state_path} is invalid: 'messages' must be a list"
            )
        messages = []
        for msg_data in messages_data:
            if not isinstance(msg_data, dict):
                raise ValueError("Each saved message must be an object")
            try:
                role = Role(msg_data["role"])
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"State file {self.state_path} has a message with an "
                    f"invalid or missing role: {msg_data.get('role')!r}"
                ) from e
            # Usage is persisted (raw_response itself is not — it can hold the
            # full provider payload) so token accounting survives a restore.
            raw_response: dict = {}
            usage = msg_data.get("usage")
            if usage:
                raw_response["usage"] = usage
            messages.append(
                Message(
                    role=role,
                    content=self._deserialize_blocks(msg_data),
                    provider=msg_data.get("provider"),
                    model=msg_data.get("model"),
                    raw_response=raw_response,
                )
            )

        context = AgentContext(
            messages=messages,
            total_tokens=data.get("total_tokens", 0),
            has_compacted=data.get("has_compacted", False),
            last_summary=data.get("last_summary", ""),
        )
        if self.session_id is not None:
            previous = [
                block.text
                for msg in messages
                if msg.role is Role.SYSTEM
                for block in msg.content
                if isinstance(block, TextBlock)
            ]
            if (
                context.has_signed_turn(0)
                and "\n\n".join(previous) != self.system_prompt
            ):
                raise ValueError(
                    "Signed history requires the original trusted system prompt"
                )
            context.messages = [msg for msg in messages if msg.role is not Role.SYSTEM]
            if self.system_prompt:
                context.messages.insert(
                    0, Message.from_text(Role.SYSTEM, self.system_prompt)
                )
        assistant = next(
            (msg for msg in reversed(messages) if msg.role is Role.ASSISTANT), None
        )
        # Anthropic rejects an unsigned thinking/tool turn on resume. Other
        # providers (e.g. Gemini) can replay the turn without a signature.
        if assistant and assistant.provider in (None, "anthropic"):
            if any(
                isinstance(block, ToolUseBlock) for block in assistant.content
            ) and any(
                isinstance(block, ThinkingBlock) and not block.signature
                for block in assistant.content
            ):
                raise ValueError(
                    "Cannot resume an unsigned thinking/tool turn; start a new turn from a completed checkpoint"
                )
        return context

    def _serialize_messages(self, messages) -> list[dict]:
        result = []
        for msg in messages:
            usage = (msg.raw_response or {}).get("usage")
            result.append(
                {
                    "role": msg.role.value,
                    "blocks": [self._serialize_block(b) for b in msg.content],
                    "provider": msg.provider,
                    "model": msg.model,
                    # Only the usage subset of raw_response is persisted; the
                    # rest of the provider payload is not needed after restore.
                    **({"usage": usage} if usage else {}),
                }
            )
        return result

    @staticmethod
    def _serialize_block(block: ContentBlockUnion) -> dict:
        """Serialize a content block by its concrete type.

        Falls back to dataclass fields for any block type added later, so a
        new block class never silently degrades to text on save.
        """
        if isinstance(block, TextBlock):
            data = {"type": block.type.value, "text": block.text}
            if block.text_signature is not None:
                data["text_signature"] = block.text_signature
            return data
        if isinstance(block, ThinkingBlock):
            data = {"type": block.type.value, "thinking": block.thinking}
            if block.signature is not None:
                data["signature"] = block.signature
            return data
        if isinstance(block, RedactedThinkingBlock):
            return {"type": block.type.value, "data": block.data}
        if isinstance(block, ToolResultBlock):
            result_data: dict[str, Any] = {
                "type": block.type.value,
                "tool_call_id": block.tool_call_id,
                "tool_result_content": block.tool_result_content,
                "is_error": block.is_error,
            }
            if block.history_key is not None:
                result_data["history_key"] = block.history_key
            return result_data
        if isinstance(block, ToolUseBlock):
            data = {
                "type": block.type.value,
                "tool_call_id": block.tool_call_id,
                "tool_name": block.tool_name,
                "tool_input": block.tool_input,
            }
            if block.thought_signature is not None:
                data["thought_signature"] = block.thought_signature
            return data
        if isinstance(block, ErrorBlock):
            return {
                "type": block.type.value,
                "error_code": block.error_code,
                "error_message": block.error_message,
            }
        if is_dataclass(block):
            data = {"type": block.type.value}
            for f in fields(block):
                if f.name != "type":
                    data[f.name] = getattr(block, f.name)
            return data
        return {"type": ContentType.TEXT.value, "text": str(block)}

    @staticmethod
    def _deserialize_blocks(msg_data: dict) -> list[ContentBlockUnion]:
        """Rebuild content blocks.

        The current format stores a ``blocks`` list. The legacy format stored a
        single ``text`` string per message; old files still load as one
        TextBlock.
        """
        raw_blocks = msg_data.get("blocks")
        if raw_blocks is not None:
            blocks: list[ContentBlockUnion] = []
            for b in raw_blocks:
                block_type = b.get("type", ContentType.TEXT.value)
                if block_type == ContentType.TOOL_USE.value:
                    blocks.append(
                        ToolUseBlock(
                            tool_call_id=b["tool_call_id"],
                            tool_name=b.get("tool_name", ""),
                            tool_input=b.get("tool_input", {}),
                            thought_signature=b.get("thought_signature"),
                        )
                    )
                elif block_type == ContentType.TOOL_RESULT.value:
                    blocks.append(
                        ToolResultBlock(
                            tool_call_id=b["tool_call_id"],
                            tool_result_content=b.get("tool_result_content", ""),
                            is_error=b.get("is_error", False),
                            history_key=b.get("history_key"),
                        )
                    )
                elif block_type == ContentType.THINKING.value:
                    blocks.append(
                        ThinkingBlock(
                            thinking=b.get("thinking", ""),
                            # Older state files store the signature under the
                            # pre-rename key.
                            signature=b.get("signature") or b.get("thinking_signature"),
                        )
                    )
                elif block_type == ContentType.REDACTED_THINKING.value:
                    blocks.append(RedactedThinkingBlock(data=b["data"]))
                elif block_type == ContentType.ERROR.value:
                    blocks.append(
                        ErrorBlock(
                            error_code=b.get("error_code", ""),
                            error_message=b.get("error_message", ""),
                        )
                    )
                elif block_type == ContentType.TEXT.value:
                    blocks.append(
                        TextBlock(
                            text=b.get("text", ""),
                            text_signature=b.get("text_signature"),
                        )
                    )
                else:
                    # An unknown block type means the file was written by a
                    # newer version. Keep the raw payload as text so nothing
                    # is silently dropped, and make the downgrade visible.
                    logger.warning(
                        "Unknown block type %r in state file; keeping raw "
                        "payload as text",
                        block_type,
                    )
                    blocks.append(TextBlock(text=json.dumps(b, ensure_ascii=False)))
            return blocks
        return [TextBlock(text=msg_data.get("text", ""))]
