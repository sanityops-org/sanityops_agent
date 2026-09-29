from __future__ import annotations

import json
import logging
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from sanityops_agent.core.context import AgentContext
from sanityops_agent.core.message import (
    ContentBlockUnion,
    ContentType,
    ErrorBlock,
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


class StateManager:
    """JSON format save/load context, compatible with old JSONL format.

    Content blocks are round-tripped by type so tool_use/tool_result pairing
    survives a save/load cycle; the previous text-only serialization dropped
    every tool block and made restored histories unusable with tools.

    Saves are atomic (temp file + os.replace), so a crash mid-write leaves the
    previous state file intact rather than corrupting it.
    """

    def __init__(self, state_path: str = "agent_state.json"):
        self.state_path = Path(state_path)

    async def save(self, context: AgentContext) -> None:
        """Save context to JSON"""
        data = {
            "messages": self._serialize_messages(context.messages),
            "total_tokens": context.total_tokens,
            "has_compacted": context.has_compacted,
            "last_summary": context.last_summary,
        }
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = json.dumps(data, indent=2)
        # Create the parent eagerly: os.open would otherwise fail when the
        # caller passed a path under a directory that does not exist yet.
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # Create the temp file owner-only from the start: write_text() would
        # briefly create it with umask permissions, exposing conversation
        # content to other users before a later chmod could run.
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
        finally:
            os.close(fd)
        try:
            os.replace(tmp_path, self.state_path)
        except OSError:
            # Cleanup is best-effort: a failing unlink must not mask the
            # replace error, and a stale .tmp file is harmless (the next
            # save overwrites it).
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    async def load(self) -> AgentContext:
        """Load context from JSON"""
        if not self.state_path.exists():
            raise FileNotFoundError(f"State file not found: {self.state_path}")

        try:
            data = json.loads(self.state_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(
                f"State file {self.state_path} is corrupt or not valid JSON: {e}"
            ) from e
        if not isinstance(data, dict):
            raise ValueError(
                f"State file {self.state_path} has an unexpected structure: "
                "expected a JSON object with a 'messages' list"
            )
        messages_data = data.get("messages", [])
        if not isinstance(messages_data, list):
            raise ValueError(
                f"State file {self.state_path} is invalid: 'messages' must be a list"
            )
        messages = []
        for msg_data in messages_data:
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

        return AgentContext(
            messages=messages,
            total_tokens=data.get("total_tokens", 0),
            has_compacted=data.get("has_compacted", False),
            last_summary=data.get("last_summary", ""),
        )

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

        Gemini thoughtSignature fields (text_signature, thinking_signature,
        thought_signature) are carried through so signature replay still works
        after a restore.
        """
        if isinstance(block, TextBlock):
            return {
                "type": block.type.value,
                "text": block.text,
                **(
                    {"text_signature": block.text_signature}
                    if block.text_signature is not None
                    else {}
                ),
            }
        if isinstance(block, ThinkingBlock):
            return {
                "type": block.type.value,
                "thinking": block.thinking,
                **(
                    {"thinking_signature": block.thinking_signature}
                    if block.thinking_signature is not None
                    else {}
                ),
            }
        if isinstance(block, ToolResultBlock):
            data: dict[str, Any] = {
                "type": block.type.value,
                "tool_call_id": block.tool_call_id,
                "tool_result_content": block.tool_result_content,
                "is_error": block.is_error,
            }
            if block.history_key is not None:
                data["history_key"] = block.history_key
            return data
        if isinstance(block, ToolUseBlock):
            return {
                "type": block.type.value,
                "tool_call_id": block.tool_call_id,
                "tool_name": block.tool_name,
                "tool_input": block.tool_input,
                **(
                    {"thought_signature": block.thought_signature}
                    if block.thought_signature is not None
                    else {}
                ),
            }
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
                if block_type in (
                    ContentType.TOOL_USE.value,
                    ContentType.TOOL_RESULT.value,
                ) and not b.get("tool_call_id"):
                    # tool_call_id is the required linking key for both tool
                    # block types; `not` rejects both None and "" (an empty
                    # id would leave the block unpaired). Raise the ValueError
                    # the load path documents instead of a bare KeyError (or a
                    # silently unpaired block that would break tool pairing).
                    raise ValueError(
                        f"Tool block of type {block_type!r} is missing a "
                        f"'tool_call_id'; the state file is corrupt"
                    )
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
                            thinking_signature=b.get("thinking_signature"),
                        )
                    )
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
