from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolHistoryEntry:
    tool_name: str
    content: str
    timestamp: float
    tool_call_id: str | None = None


class ToolHistoryStore:
    """Per-session tool result storage, supports query by tool_call_id"""

    def __init__(
        self,
        session_id: str = "default",
        *,
        max_entries: int = 256,
        max_entry_chars: int = 50_000,
        retention_seconds: float = 86_400,
    ):
        if min(max_entries, max_entry_chars, retention_seconds) <= 0:
            raise ValueError("History limits must be positive")
        self.session_id = session_id
        self.max_entries = max_entries
        self.max_entry_chars = max_entry_chars
        self.retention_seconds = retention_seconds
        self._store: dict[str, ToolHistoryEntry] = {}
        self._lock = asyncio.Lock()

    async def store(
        self,
        tool_name: str,
        content: str,
        tool_call_id: str | None = None,
        *,
        turn_id: str | None = None,
    ) -> str:
        """Store a bounded result and return its history key."""
        parts = [self.session_id]
        if turn_id is not None:
            parts.append(turn_id)
        parts.append(str(tool_call_id or id(content)))
        key = ":".join(parts)
        async with self._lock:
            now = time.time()
            self._prune_expired(now)
            self._store.pop(key, None)
            stored = content
            if len(content) > self.max_entry_chars:
                stored = content[: self.max_entry_chars]
                logger.warning(
                    "Tool history entry for %s truncated to %d characters",
                    tool_name,
                    self.max_entry_chars,
                )
            self._store[key] = ToolHistoryEntry(
                tool_name=tool_name,
                content=stored,
                timestamp=now,
                tool_call_id=tool_call_id,
            )
            while len(self._store) > self.max_entries:
                dropped = self._store.pop(next(iter(self._store)))
                logger.debug(
                    "Tool history entry for %s evicted (capacity %d)",
                    dropped.tool_name,
                    self.max_entries,
                )
        return key

    async def get(self, key: str) -> ToolHistoryEntry | None:
        async with self._lock:
            self._prune_expired(time.time())
            return self._store.get(key)

    async def search(self, query: str, limit: int) -> list[ToolHistoryEntry]:
        search = query.casefold()
        async with self._lock:
            self._prune_expired(time.time())
            results = [
                entry
                for entry in reversed(self._store.values())
                if not search
                or search in entry.tool_name.casefold()
                or search in entry.content.casefold()
            ]
            return results[: max(0, min(limit, self.max_entries))]

    def _prune_expired(self, now: float) -> None:
        cutoff = now - self.retention_seconds
        for key in [
            key for key, entry in self._store.items() if entry.timestamp <= cutoff
        ]:
            del self._store[key]

    async def clear(self):
        async with self._lock:
            self._store.clear()

    def __len__(self):
        return len(self._store)
