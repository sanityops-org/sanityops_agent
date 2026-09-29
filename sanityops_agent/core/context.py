from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

from sanityops_agent.core.message import Message, ToolResultBlock

# Conservative chars-per-token ratio for token estimation (English prose).
CHARS_PER_TOKEN_ESTIMATE: int = 4


@dataclass
class CompactConfig:
    """Compaction configuration"""

    trigger_token_threshold: int = 80000  # Token threshold to trigger compaction
    preserve_recent_count: int = 5  # Preserve last N tool results in full


@dataclass
class ToolResultSummary:
    tool_name: str
    tool_call_id: str
    success: bool
    brief: str
    history_key: str | None = None

    def to_compact_string(self) -> str:
        return f"[Tool: {self.tool_name}, Success: {self.success}, Brief: {self.brief}]"


@dataclass
class AgentContext:
    """Context manager - single writer model (feedback point 15)

    Agent loop executes sequentially: add_message -> check_compact -> continue
    No concurrent write risk, no locks needed.
    """

    messages: list[Message] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)

    # Token counter (feedback point 14)
    total_tokens: int = 0

    # Compaction state
    has_compacted: bool = False
    last_summary: str = ""

    # Config
    compact_config: CompactConfig = field(default_factory=CompactConfig)
    history_store: object | None = None  # ToolHistoryStore | None

    # Compaction serialization. asyncio.Lock() has been loop-independent at
    # construction since Python 3.10 and this package requires >=3.11, so
    # creating it here cannot fail outside an event loop.
    # Caveat: lock objects are shared if an AgentContext is copied via
    # dataclasses.replace() (the field default is reused). Do not replace()
    # an AgentContext; mutate it in place (the agent loop does).
    _compact_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )
    _compaction_in_flight: bool = field(default=False, repr=False, compare=False)

    async def add_message(self, message: Message):
        """Add message, update token count"""
        self.messages.append(message)
        # Update token count (from message.raw_response usage or LLMResponse.usage)
        if message.raw_response and "usage" in message.raw_response:
            self.total_tokens += message.raw_response["usage"].get("total_tokens", 0)

    def tool_result_content(self, tool_call_id: str) -> str | None:
        """Content of the tool result for this call, or None if absent.

        Compaction shortens the content in place, so this reads the current
        (possibly summarized) text.
        """
        for msg in self.messages:
            for block in msg.content:
                if (
                    isinstance(block, ToolResultBlock)
                    and block.tool_call_id == tool_call_id
                ):
                    return block.tool_result_content
        return None

    def estimate_tokens(self) -> int:
        """Estimate context size from the messages as they are now.

        add_message() accumulates provider usage numbers, which never shrink —
        after compaction the accumulated total no longer reflects the history,
        so the compaction trigger would re-fire on every call. This estimate
        (chars/4) is what the trigger compares against.

        This is deliberately approximate: real tokenizers yield ~3-5 chars per
        token for English prose, and substantially fewer for code, JSON, or CJK
        text (~1-3). Dense content is therefore UNDERestimated and compaction
        triggers LATER than the threshold suggests — the conservative failure
        mode (a late compaction only means a larger context; an early one would
        truncate history the model still needs). Tune
        ``trigger_token_threshold`` with that in mind; plugging in a provider
        tokenizer would tighten it if ever needed.

        Example: a 20k-character tool result estimates to 5,000 tokens; the
        default threshold (80,000) is crossed at roughly 320k characters of
        English prose, sooner for code-heavy histories.

        Only text-bearing fields are counted (text, thinking,
        tool_result_content). Tool call names/inputs are ignored, so a history
        dominated by large structured tool inputs is underestimated and
        compacts later than the threshold suggests.
        """
        chars = 0
        for msg in self.messages:
            for block in msg.content:
                chars += len(getattr(block, "text", "") or "")
                chars += len(getattr(block, "thinking", "") or "")
                chars += len(getattr(block, "tool_result_content", "") or "")
        return chars // CHARS_PER_TOKEN_ESTIMATE

    async def compact_if_needed(self) -> bool:
        """Trigger compaction based on the current token estimate (feedback
        point 14). The accumulated usage total never shrinks, so comparing
        against it would re-fire the trigger on every call after compaction.

        The lock serializes concurrent callers. The in-flight flag is checked
        BEFORE the lock so a reentrant caller (e.g. a hook invoked from within
        ``_do_compact``) returns False instead of deadlocking: asyncio.Lock is
        not reentrant, so such a call would block forever waiting on a lock it
        already holds. The second check after acquiring is for the ordinary
        concurrent case. Neither check awaits in between, so overlapping calls
        either compact once or return False — never double-compact.
        """
        if self._compaction_in_flight:
            return False
        async with self._compact_lock:
            if self._compaction_in_flight:
                return False
            if self.estimate_tokens() < self.compact_config.trigger_token_threshold:
                return False

            self._compaction_in_flight = True
            try:
                await self._do_compact()
                self.has_compacted = True
                return True
            finally:
                self._compaction_in_flight = False

    async def _do_compact(self):
        """Execute compaction.

        Older tool results are shortened to a one-line summary. The
        ``ToolResultBlock`` is KEPT (only its content is replaced): providers
        reject a ``tool_use`` with no matching ``tool_result`` block, so
        removing the block would orphan the assistant turn that requested it.
        """
        # 1. Collect all tool results and identify which to preserve vs compact
        all_tool_results: list[tuple[Message, int, ToolResultBlock]] = []

        for msg in self.messages:
            for idx, block in enumerate(msg.content):
                if isinstance(block, ToolResultBlock):
                    all_tool_results.append((msg, idx, block))

        total_tool_results = len(all_tool_results)
        preserve_count = self.compact_config.preserve_recent_count

        if total_tool_results <= preserve_count:
            return

        # 2. Determine which to compact (oldest ones) and which to preserve (recent ones)
        num_to_compact = total_tool_results - preserve_count
        to_compact = all_tool_results[:num_to_compact]

        # 3. Shorten compacted tool results in place, preserving tool_call_id
        for msg, idx, block in to_compact:
            summary = ToolResultSummary(
                tool_name=block.metadata.get("tool_name", "unknown"),
                tool_call_id=block.tool_call_id,
                success=not block.is_error,
                brief=f"[Result: {len(block.tool_result_content)} chars]",
                # history_key is a direct ToolResultBlock attribute; metadata
                # may or may not mirror it, so prefer the attribute.
                history_key=block.history_key or block.metadata.get("history_key"),
            )
            msg.content[idx] = replace(
                block,
                tool_result_content=summary.to_compact_string(),
                metadata={**block.metadata, "compacted": True},
            )
