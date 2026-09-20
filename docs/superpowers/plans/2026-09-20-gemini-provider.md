# Gemini Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gemini provider with native SDK support for text, tool calling, streaming, and thinking mode.

**Architecture:** Provider + Adapter pattern matching existing Anthropic/OpenAI implementations. Uses `google-genai` unified SDK supporting both AI Studio and Vertex AI endpoints.

**Tech Stack:** Python 3.11+, google-genai SDK, pydantic, httpx

## Global Constraints

- Default model: `gemini-3.5-flash-lite`
- SDK: `google-genai` (unified SDK for AI Studio and Vertex AI)
- Thinking storage: configurable via `GEMINI_STORE_THINKING` (default: `False`)
- Endpoint selection: `GEMINI_PROJECT_ID` set → Vertex AI; unset → AI Studio
- Follow existing provider patterns exactly (AnthropicProvider, OpenAIProvider)

---

## File Structure

**New files:**
- `sanityops_agent/llm/providers/gemini.py` — GeminiProvider class
- `sanityops_agent/llm/adapters/gemini_adapter.py` — GeminiAdapter class

**Modified files:**
- `sanityops_agent/config.py` — Add Gemini-specific config fields
- `sanityops_agent/llm/factory.py` — Register GeminiProvider
- `sanityops_agent/llm/providers/__init__.py` — Export GeminiProvider
- `sanityops_agent/llm/adapters/__init__.py` — Export GeminiAdapter
- `pyproject.toml` — Add `google-genai` dependency

**Test files:**
- `tests/unit/llm/test_adapters.py` — Add GeminiAdapter tests
- `tests/unit/llm/test_providers.py` — Add GeminiProvider tests

---

### Task 1: Add google-genai Dependency

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: N/A
- Produces: `google-genai` package available for import

- [ ] **Step 1: Add dependency to pyproject.toml**

```python
# In dependencies array, add:
"google-genai>=1.0",
```

- [ ] **Step 2: Verify dependency can be installed**

```bash
pip install -e ".[dev]"
```

Expected: Installation succeeds without errors

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add google-genai dependency for Gemini provider"
```

---

### Task 2: Extend ProviderConfig with Gemini Fields

**Files:**
- Modify: `sanityops_agent/config.py`
- Create: `tests/unit/test_config_gemini.py`

**Interfaces:**
- Consumes: N/A
- Produces: `ProviderConfig.GEMINI_PROJECT_ID`, `GEMINI_LOCATION`, `GEMINI_THINKING_BUDGET`, `GEMINI_STORE_THINKING`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config_gemini.py

from __future__ import annotations

from sanityops_agent.config import ProviderConfig


def test_gemini_fields_default_to_disabled():
    """Without explicit values, Gemini extras stay unset and thinking is off."""
    config = ProviderConfig(PROVIDER="gemini", API_KEY="test-key")
    assert config.GEMINI_PROJECT_ID is None
    assert config.GEMINI_LOCATION is None
    assert config.GEMINI_THINKING_BUDGET is None
    assert config.GEMINI_STORE_THINKING is False


def test_gemini_fields_accept_explicit_values():
    config = ProviderConfig(
        PROVIDER="gemini",
        API_KEY="test-key",
        GEMINI_PROJECT_ID="my-project",
        GEMINI_LOCATION="us-central1",
        GEMINI_THINKING_BUDGET=2048,
        GEMINI_STORE_THINKING=True,
    )
    assert config.GEMINI_PROJECT_ID == "my-project"
    assert config.GEMINI_LOCATION == "us-central1"
    assert config.GEMINI_THINKING_BUDGET == 2048
    assert config.GEMINI_STORE_THINKING is True


def test_gemini_fields_read_from_env(monkeypatch):
    """Env vars use the LLM_ prefix from ProviderConfig.model_config."""
    monkeypatch.setenv("LLM_GEMINI_PROJECT_ID", "env-project")
    monkeypatch.setenv("LLM_GEMINI_THINKING_BUDGET", "1024")
    monkeypatch.setenv("LLM_GEMINI_STORE_THINKING", "true")
    config = ProviderConfig(PROVIDER="gemini", API_KEY="test-key")
    assert config.GEMINI_PROJECT_ID == "env-project"
    assert config.GEMINI_THINKING_BUDGET == 1024
    assert config.GEMINI_STORE_THINKING is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_config_gemini.py -v
```
Expected: FAIL with `ValidationError` (extra fields rejected) or `AttributeError`

- [ ] **Step 3: Add Gemini fields to ProviderConfig**

```python
# In sanityops_agent/config.py, inside ProviderConfig class (after MODEL_ID):

    # Gemini-specific fields
    GEMINI_PROJECT_ID: str | None = None      # Vertex AI project ID
    GEMINI_LOCATION: str | None = None        # Vertex AI region
    GEMINI_THINKING_BUDGET: int | None = None # thinking token budget (None=disabled)
    GEMINI_STORE_THINKING: bool = False       # whether to store thinking in history
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_config_gemini.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sanityops_agent/config.py tests/unit/test_config_gemini.py
git commit -m "feat(config): add Gemini-specific provider configuration fields"
```

---

### Task 3: Implement GeminiAdapter Core

**Files:**
- Create: `sanityops_agent/llm/adapters/gemini_adapter.py`

**Interfaces:**
- Consumes: `Message`, `ToolDefinition`, `ContentBlockUnion`, `StreamChunk` from `sanityops_agent.core.message`
- Consumes: `MessageAdapter` from `sanityops_agent.llm.adapters.base`
- Consumes: `LLMResponse` from `sanityops_agent.llm.base`
- Produces: `GeminiAdapter` class with `to_provider()`, `from_provider_response()`, `from_provider_chunk()`, `map_exception()`

- [ ] **Step 1: Write the failing test for to_provider text messages**

```python
# In tests/unit/llm/test_adapters.py

class TestGeminiAdapterText:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter
        return GeminiAdapter()

    def test_single_user_text_message(self, adapter):
        messages = [Message.from_text(Role.USER, "Hello")]
        result = adapter.to_provider(messages)

        assert "contents" in result
        assert len(result["contents"]) == 1
        content = result["contents"][0]
        assert content["role"] == "user"
        assert len(content["parts"]) == 1
        assert content["parts"][0]["text"] == "Hello"

    def test_assistant_role_becomes_model(self, adapter):
        messages = [Message.from_text(Role.ASSISTANT, "Hi there")]
        result = adapter.to_provider(messages)

        content = result["contents"][0]
        assert content["role"] == "model"
        assert content["parts"][0]["text"] == "Hi there"

    def test_system_message_becomes_system_instruction(self, adapter):
        messages = [Message.from_text(Role.SYSTEM, "You are helpful")]
        result = adapter.to_provider(messages)

        assert "systemInstruction" in result
        assert result["systemInstruction"]["parts"][0]["text"] == "You are helpful"
        assert result["contents"] == []

    def test_system_plus_user(self, adapter):
        messages = [
            Message.from_text(Role.SYSTEM, "Be concise"),
            Message.from_text(Role.USER, "What is 2+2?"),
        ]
        result = adapter.to_provider(messages)

        assert result["systemInstruction"]["parts"][0]["text"] == "Be concise"
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/llm/test_adapters.py::TestGeminiAdapterText -v
```
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement GeminiAdapter.to_provider for text messages**

```python
# sanityops_agent/llm/adapters/gemini_adapter.py

from __future__ import annotations

import logging
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
    - TextBlock -> {"text": "..."}
    - ThinkingBlock -> {"thought": "..."} (Gemini 2.5+ native format)
    - ToolUseBlock -> {"functionCall": {"name": "...", "args": {...}}}
    - ToolResultBlock -> {"functionResponse": {"name": "...", "response": {...}}}
    """

    def __init__(self, store_thinking: bool = False):
        self.store_thinking = store_thinking

    def to_provider(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Convert messages to Gemini API format.

        Returns: {"systemInstruction": {...}, "contents": [...], "tools": [...]}
        """
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

            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append({"text": block.text})
                elif isinstance(block, ThinkingBlock):
                    parts.append({"thought": block.thinking})
                elif isinstance(block, ToolUseBlock):
                    parts.append({
                        "functionCall": {
                            "id": block.tool_call_id,
                            "name": block.tool_name,
                            "args": block.tool_input,
                        }
                    })
                elif isinstance(block, ToolResultBlock):
                    # Gemini FunctionResponse requires both id (call ID) and name (function name)
                    # We look up the function name from preceding ToolUseBlock
                    fn_name = call_to_name.get(block.tool_call_id, block.tool_call_id)
                    parts.append({
                        "functionResponse": {
                            "id": block.tool_call_id,
                            "name": fn_name,
                            "response": {
                                "result": block.tool_result_content,
                            },
                        }
                    })

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
        """
        candidates = raw.get("candidates", [])
        if not candidates:
            return LLMResponse(
                content=[],
                stop_reason=FinishReason.END_TURN,
                model=raw.get("modelVersion", "unknown"),
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
                    content.append(ThinkingBlock(thinking=part["text"]))
                continue
            if "text" in part and part.get("text") is not None:
                content.append(TextBlock(text=part["text"]))
            elif "functionCall" in part:
                fc = part["functionCall"]
                content.append(
                    ToolUseBlock(
                        tool_call_id=fc.get("id") or fc.get("name", ""),
                        tool_name=fc.get("name", ""),
                        tool_input=fc.get("args") or {},
                    )
                )

        usage = raw.get("usageMetadata", {})
        return LLMResponse(
            content=content,
            stop_reason=finish_reason,
            model=raw.get("modelVersion", "unknown"),
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/llm/test_adapters.py::TestGeminiAdapterText -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sanityops_agent/llm/adapters/gemini_adapter.py tests/unit/llm/test_adapters.py
git commit -m "feat(adapter): add GeminiAdapter with text message conversion"
```

---

### Task 4: Implement GeminiAdapter Tool Support

**Files:**
- Modify: `sanityops_agent/llm/adapters/gemini_adapter.py`
- Modify: `tests/unit/llm/test_adapters.py`

**Interfaces:**
- Consumes: `ToolUseBlock`, `ToolResultBlock`, `ToolDefinition`
- Produces: Tool use/result conversion in `to_provider()`, tool response parsing in `from_provider_response()`

- [ ] **Step 1: Write the failing tests for tool use**

```python
# In tests/unit/llm/test_adapters.py

class TestGeminiAdapterToolUse:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter
        return GeminiAdapter()

    def test_tool_use_block_conversion(self, adapter):
        tool_call = ToolUseBlock(
            tool_call_id="call_abc",
            tool_name="search",
            tool_input={"query": "weather"},
        )
        msg = Message(role=Role.ASSISTANT, content=[tool_call])
        result = adapter.to_provider([msg])

        part = result["contents"][0]["parts"][0]
        assert "functionCall" in part
        assert part["functionCall"]["name"] == "search"
        assert part["functionCall"]["args"] == {"query": "weather"}

    def test_tool_result_block_conversion(self, adapter):
        result_block = ToolResultBlock(
            tool_call_id="call_abc",
            tool_result_content="Sunny, 75F",
            is_error=False,
        )
        msg = Message(role=Role.USER, content=[result_block])
        out = adapter.to_provider([msg])

        part = out["contents"][0]["parts"][0]
        assert "functionResponse" in part
        # Unpaired fallback: name falls back to the call id
        assert part["functionResponse"]["name"] == "call_abc"
        assert part["functionResponse"]["response"]["result"] == "Sunny, 75F"

    def test_tool_result_uses_function_name_from_prior_tool_use(self, adapter):
        """The real agent flow sends ToolUse then ToolResult. Gemini's
        FunctionResponse requires the FUNCTION NAME, not the call id, so the
        adapter must look it up from the earlier ToolUseBlock."""
        tool_use = ToolUseBlock(
            tool_call_id="call_abc",
            tool_name="search",
            tool_input={"query": "weather"},
        )
        tool_result = ToolResultBlock(
            tool_call_id="call_abc",
            tool_result_content="Sunny, 75F",
        )
        messages = [
            Message(role=Role.ASSISTANT, content=[tool_use]),
            Message(role=Role.USER, content=[tool_result]),
        ]
        out = adapter.to_provider(messages)

        call_part = out["contents"][0]["parts"][0]["functionCall"]
        assert call_part["id"] == "call_abc"
        assert call_part["name"] == "search"

        resp_part = out["contents"][1]["parts"][0]["functionResponse"]
        assert resp_part["id"] == "call_abc"
        assert resp_part["name"] == "search"
        assert resp_part["response"]["result"] == "Sunny, 75F"

    def test_tools_parameter(self, adapter):
        tool_def = ToolDefinition(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        messages = [Message.from_text(Role.USER, "Search for cats")]
        result = adapter.to_provider(messages, tools=[tool_def])

        assert "tools" in result
        assert "functionDeclarations" in result["tools"][0]
        assert result["tools"][0]["functionDeclarations"][0]["name"] == "search"

    def test_parse_tool_use_response(self, adapter):
        response = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "test"},
                        }
                    }]
                },
                "finishReason": "STOP",
            }],
            "model": "gemini-3.5-flash-lite",
        }
        result = adapter.from_provider_response(response)

        assert len(result.content) == 1
        assert isinstance(result.content[0], ToolUseBlock)
        assert result.content[0].tool_name == "search"
        assert result.content[0].tool_input == {"query": "test"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/llm/test_adapters.py::TestGeminiAdapterToolUse -v
```
Expected: FAIL (methods may be incomplete)

- [ ] **Step 3: Verify implementation covers tool cases**

The `to_provider()` method in Task 3 already includes tool use/result conversion. If tests fail, debug and fix the implementation.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/llm/test_adapters.py::TestGeminiAdapterToolUse -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sanityops_agent/llm/adapters/gemini_adapter.py tests/unit/llm/test_adapters.py
git commit -m "feat(adapter): add Gemini tool use and result conversion"
```

---

### Task 5: Implement GeminiAdapter Streaming and Thinking

**Files:**
- Modify: `sanityops_agent/llm/adapters/gemini_adapter.py`
- Modify: `tests/unit/llm/test_adapters.py`

**Interfaces:**
- Consumes: `StreamChunk`, `ContentType`, `ThinkingBlock`
- Produces: `from_provider_chunk()` with text/thought/tool support, thinking filtering based on `store_thinking`

- [ ] **Step 1: Write the failing tests for streaming**

```python
# In tests/unit/llm/test_adapters.py

class TestGeminiAdapterStreaming:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter
        return GeminiAdapter(store_thinking=True)

    def test_text_delta(self, adapter):
        chunk = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello"}]
                }
            }]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TEXT
        assert result.delta == "Hello"

    def test_thinking_delta(self, adapter):
        # Real Gemini thought parts carry BOTH text and thought=True
        chunk = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Let me think...", "thought": True}]
                }
            }]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.THINKING
        assert result.delta == "Let me think..."

    def test_thinking_not_stored(self):
        adapter = GeminiAdapter(store_thinking=False)
        response = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Hmm...", "thought": True},
                        {"text": "The answer is 42"},
                    ]
                },
                "finishReason": "STOP",
            }],
            "modelVersion": "gemini-3.5-flash-lite",
        }
        result = adapter.from_provider_response(response)

        # Only text, no thinking block
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "The answer is 42"

    def test_tool_call_delta(self, adapter):
        chunk = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "test"},
                        }
                    }]
                }
            }]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.type == ContentType.TOOL_USE
        assert result.tool_call_id == "search"
        assert result.delta["name"] == "search"

    def test_finish_reason(self, adapter):
        chunk = {
            "candidates": [{
                "finishReason": "MAX_TOKENS",
                "content": {"parts": []}
            }]
        }
        result = adapter.from_provider_chunk(chunk)

        assert result.finish_reason == FinishReason.MAX_TOKENS
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pytest tests/unit/llm/test_adapters.py::TestGeminiAdapterStreaming -v
```
Expected: PASS (implementation from Task 3 should cover this)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/llm/test_adapters.py
git commit -m "test(adapter): add Gemini streaming and thinking tests"
```

---

### Task 6: Implement GeminiProvider Core

**Files:**
- Create: `sanityops_agent/llm/providers/gemini.py`

**Interfaces:**
- Consumes: `LLMProvider` from `sanityops_agent.llm.base`
- Consumes: `GeminiAdapter` from `sanityops_agent.llm.adapters.gemini_adapter`
- Consumes: `google.genai.Client` from google-genai SDK
- Produces: `GeminiProvider` with `__init__()`, `_get_client()`, `close()`, `create()`, `stream()`

- [ ] **Step 1: Write the failing test for instantiation**

```python
# In tests/unit/llm/test_providers.py

class TestGeminiProviderInstantiation:
    @pytest.fixture
    def adapter(self):
        from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter
        return GeminiAdapter()

    def test_can_be_instantiated_with_defaults(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        provider = GeminiProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "gemini-3.5-flash-lite"
        assert provider.adapter is not None

    def test_can_be_instantiated_with_vertex_ai_config(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        provider = GeminiProvider(
            api_key="test-key",
            project_id="my-project",
            location="us-central1",
        )
        assert provider.project_id == "my-project"
        assert provider.location == "us-central1"

    def test_thinking_config(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        provider = GeminiProvider(
            api_key="test-key",
            thinking_budget=2048,
            store_thinking=True,
        )
        assert provider.thinking_budget == 2048
        assert provider.store_thinking is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/llm/test_providers.py::TestGeminiProviderInstantiation -v
```
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement GeminiProvider skeleton**

```python
# sanityops_agent/llm/providers/gemini.py

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from sanityops_agent.core.message import (
    Message,
    StreamChunk,
    ToolDefinition,
)
from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter
from sanityops_agent.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini provider using the native google-genai SDK.

    Supports both AI Studio and Vertex AI endpoints.
    Supports: text, thinking, tool_use content blocks.

    Not thread-safe and not safe for concurrent asyncio access: the client
    is created lazily and closed by the agent that owns it. Intended for
    sequential use within a single event loop.
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        # Vertex AI configuration
        project_id: str | None = None,
        location: str | None = None,
        # Thinking configuration
        thinking_budget: int | None = None,
        store_thinking: bool = False,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.project_id = project_id
        self.location = location
        self.thinking_budget = thinking_budget
        self.store_thinking = store_thinking
        self.timeout = timeout
        self._client: genai.Client | None = None
        self.adapter = GeminiAdapter(store_thinking=store_thinking)

    def _build_config(
        self, payload: dict, kwargs: dict
    ) -> types.GenerateContentConfig:
        """Build a GenerateContentConfig from the adapter payload and kwargs.

        - systemInstruction goes into ``system_instruction``
        - tools (list of {"functionDeclarations": [...]} dicts) are converted
          to ``types.Tool`` objects
        - thinking_budget (when set) enables thinking; include_thoughts=True is
          REQUIRED for the API to return thought parts at all
        """
        config_kwargs: dict = {}

        if payload.get("systemInstruction"):
            config_kwargs["system_instruction"] = types.Content.model_validate(
                payload["systemInstruction"]
            )

        if payload.get("tools"):
            tool_objs = []
            for tool_dict in payload["tools"]:
                declarations = [
                    types.FunctionDeclaration(
                        name=d["name"],
                        description=d.get("description"),
                        parameters_json_schema=d.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    )
                    for d in tool_dict.get("functionDeclarations", [])
                ]
                tool_objs.append(types.Tool(function_declarations=declarations))
            config_kwargs["tools"] = tool_objs

        if self.thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            )

        # Caller-supplied overrides (temperature, max_output_tokens, ...)
        config_kwargs.update(kwargs)

        return types.GenerateContentConfig(**config_kwargs)

    def _get_client(self) -> genai.Client:
        """Get or create the SDK client.

        Uses Vertex AI mode if project_id is set, otherwise AI Studio mode.
        """
        if self._client is None:
            if self.project_id:
                # Vertex AI mode (uses Application Default Credentials)
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location or "us-central1",
                )
            else:
                # AI Studio mode
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def close(self):
        """Close the underlying client connection.

        Best-effort and idempotent: google-genai's ``Client.close()`` closes
        only the sync client, so ``aio.aclose()`` is used for the async client
        this provider actually calls. Never raises and always clears the
        reference, even if closing fails.
        """
        try:
            if self._client is not None:
                await self._client.aio.aclose()
        except Exception as e:
            logger.warning("Failed to close Gemini client: %s", e, exc_info=True)
        finally:
            self._client = None

    async def create(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Non-streaming call. Implemented in Task 7."""
        raise NotImplementedError("create() implemented in Task 7")

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming call. Implemented in Task 7."""
        raise NotImplementedError("stream() implemented in Task 7")
```

- [ ] **Step 4: Add dedicated close() tests (note: do NOT add Gemini to PROVIDER_CLASSES)**

**Important:** the existing lifecycle tests at the bottom of `test_providers.py`
parameterize over `PROVIDER_CLASSES = [OpenAIProvider, AnthropicProvider]` and
assert that `provider.close()` calls the client's `close()`. Gemini closes via
`aio.aclose()` instead, so adding `GeminiProvider` to that list would break
those tests. Add these separate tests instead:

```python
# In tests/unit/llm/test_providers.py, in its own section

class _FakeGeminiClient:
    """Fake google-genai client: ``close()`` on the sync side, ``aclose()``
    on the ``aio`` side — matching the real SDK's split."""

    def __init__(self):
        self.sync_closed = False
        self.async_closed = False

    def close(self):
        self.sync_closed = True

    @property
    def aio(self):
        return self

    async def aclose(self):
        self.async_closed = True


class _RaisingGeminiClient(_FakeGeminiClient):
    async def aclose(self):
        raise RuntimeError("transport failed to close")


class TestGeminiProviderClose:
    @pytest.mark.asyncio
    async def test_close_closes_async_client_and_clears_reference(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        provider = GeminiProvider(api_key="k")
        client = _FakeGeminiClient()
        provider._client = client
        await provider.close()
        assert client.async_closed is True
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_is_best_effort_when_aclose_raises(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        provider = GeminiProvider(api_key="k")
        provider._client = _RaisingGeminiClient()
        await provider.close()  # must not raise
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_is_noop_when_no_client(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        provider = GeminiProvider(api_key="k")
        await provider.close()  # must not raise
        assert provider._client is None
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/unit/llm/test_providers.py::TestGeminiProviderInstantiation tests/unit/llm/test_providers.py::TestGeminiProviderClose -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add sanityops_agent/llm/providers/gemini.py tests/unit/llm/test_providers.py
git commit -m "feat(provider): add GeminiProvider skeleton with instantiation"
```

---

### Task 7: Implement GeminiProvider API Calls

**Files:**
- Modify: `sanityops_agent/llm/providers/gemini.py`
- Modify: `tests/unit/llm/test_providers.py`

**Interfaces:**
- Consumes: `google.genai.types.GenerateContentConfig`, `types.Content`, `types.Part`
- Produces: Working `create()` and `stream()` methods

- [ ] **Step 1: Write the failing test for create()**

```python
# In tests/unit/llm/test_providers.py

class TestGeminiProviderCreate:
    @pytest.fixture
    def provider(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_create_returns_llm_response(self, provider):
        from unittest.mock import AsyncMock, MagicMock, patch
        from sanityops_agent.core.message import Role, TextBlock

        # Mock the SDK response with a real dict from model_dump(by_alias=True)
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello, world!"}]
                },
                "finishReason": "STOP",
            }],
            "modelVersion": "gemini-3.5-flash-lite",
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
        }

        # generate_content is awaited, so it must be an AsyncMock
        mock_models = MagicMock()
        mock_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_models

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            result = await provider.create(messages)

        assert isinstance(result, LLMResponse)
        assert result.model == "gemini-3.5-flash-lite"
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/llm/test_providers.py::TestGeminiProviderCreate -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement create() method**

```python
# In GeminiProvider class, replace create() stub:

    async def create(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Non-streaming call to Gemini API."""
        client = self._get_client()
        payload = self.adapter.to_provider(messages, tools)
        config = self._build_config(payload, kwargs)

        contents = [types.Content.model_validate(c) for c in payload["contents"]]

        try:
            response = await client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            raw = response.model_dump(by_alias=True, exclude_none=True)
            return self.adapter.from_provider_response(raw)

        except Exception as e:
            raise self.adapter.map_exception(e)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/unit/llm/test_providers.py::TestGeminiProviderCreate -v
```
Expected: PASS

- [ ] **Step 5: Write the failing test for stream()**

```python
# In tests/unit/llm/test_providers.py

class TestGeminiProviderStream:
    @pytest.fixture
    def provider(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        return GeminiProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider):
        from unittest.mock import MagicMock, patch
        from sanityops_agent.core.message import Role, TextBlock

        # Mock streaming response chunks
        class MockChunk:
            def __init__(self, text):
                self._text = text

            def model_dump(self, **kwargs):
                return {
                    "candidates": [{
                        "content": {"parts": [{"text": self._text}]}
                    }]
                }

        chunks = [MockChunk("Hel"), MockChunk("lo")]

        class MockAsyncStream:
            def __init__(self, items):
                self._items = iter(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._items)
                except StopIteration:
                    raise StopAsyncIteration

        # generate_content_stream is a coroutine that RETURNS an async iterator,
        # so the mock must be an async function returning MockAsyncStream.
        async def fake_stream(**kwargs):
            return MockAsyncStream(chunks)

        mock_models = MagicMock()
        mock_models.generate_content_stream = fake_stream

        mock_client = MagicMock()
        mock_client.aio.models = mock_models

        with patch.object(provider, "_get_client", return_value=mock_client):
            messages = [Message(role=Role.USER, content=[TextBlock(text="Hi")])]
            results = []
            async for chunk in provider.stream(messages):
                results.append(chunk)

        assert len(results) == 2
        assert results[0].delta == "Hel"
        assert results[1].delta == "lo"
```

- [ ] **Step 6: Run test to verify it fails**

```bash
pytest tests/unit/llm/test_providers.py::TestGeminiProviderStream -v
```
Expected: FAIL with `NotImplementedError`

- [ ] **Step 7: Implement stream() method**

```python
# In GeminiProvider class, replace stream() stub:

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming call to Gemini API."""
        client = self._get_client()
        payload = self.adapter.to_provider(messages, tools)
        config = self._build_config(payload, kwargs)

        contents = [types.Content.model_validate(c) for c in payload["contents"]]

        try:
            # generate_content_stream is a coroutine that returns an async
            # iterator, so it must be awaited before iteration.
            stream = await client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            async for chunk in stream:
                raw = chunk.model_dump(by_alias=True, exclude_none=True)
                result = self.adapter.from_provider_chunk(raw)
                if result:
                    yield result

        except Exception as e:
            raise self.adapter.map_exception(e)
```

- [ ] **Step 8: Run test to verify it passes**

```bash
pytest tests/unit/llm/test_providers.py::TestGeminiProviderStream -v
```
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add sanityops_agent/llm/providers/gemini.py tests/unit/llm/test_providers.py
git commit -m "feat(provider): implement GeminiProvider create() and stream()"
```

---

### Task 8: Register GeminiProvider in Factory

**Files:**
- Modify: `sanityops_agent/llm/factory.py`
- Modify: `sanityops_agent/llm/providers/__init__.py`
- Modify: `sanityops_agent/llm/adapters/__init__.py`
- Modify: `tests/unit/llm/test_factory.py`

**Interfaces:**
- Consumes: `GeminiProvider` from `sanityops_agent.llm.providers.gemini`
- Produces: `ProviderFactory.create()` supports `provider="gemini"`

- [ ] **Step 1: Write the failing test**

```python
# In tests/unit/llm/test_factory.py

class TestProviderFactoryGemini:
    @pytest.fixture(autouse=True)
    def _setup_registry(self):
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        ProviderFactory.register("gemini", GeminiProvider)
        yield
        ProviderFactory._registry.pop("gemini", None)

    def _make_config(self, **kwargs):
        from sanityops_agent.config import ProviderConfig
        defaults = {"PROVIDER": "gemini", "API_KEY": "test-key"}
        defaults.update(kwargs)
        return ProviderConfig(**defaults)

    def test_factory_creates_gemini_provider(self):
        config = self._make_config()
        provider = ProviderFactory.create(config)
        from sanityops_agent.llm.providers.gemini import GeminiProvider
        assert isinstance(provider, GeminiProvider)
        assert provider.api_key == "test-key"
        assert provider.model == "gemini-3.5-flash-lite"

    def test_factory_gemini_with_vertex_ai(self):
        config = self._make_config(
            GEMINI_PROJECT_ID="my-project",
            GEMINI_LOCATION="us-central1",
        )
        provider = ProviderFactory.create(config)
        assert provider.project_id == "my-project"
        assert provider.location == "us-central1"

    def test_factory_gemini_with_thinking(self):
        config = self._make_config(
            GEMINI_THINKING_BUDGET=2048,
            GEMINI_STORE_THINKING=True,
        )
        provider = ProviderFactory.create(config)
        assert provider.thinking_budget == 2048
        assert provider.store_thinking is True

    def test_factory_lists_gemini_provider(self):
        providers = ProviderFactory.list_providers()
        assert "gemini" in providers
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/llm/test_factory.py::TestProviderFactoryGemini -v
```
Expected: FAIL (gemini not registered)

- [ ] **Step 3: Update factory.py to register GeminiProvider**

```python
# In sanityops_agent/llm/factory.py, add import:

from sanityops_agent.llm.providers.gemini import GeminiProvider

# Add registration after other registrations:
ProviderFactory.register("gemini", GeminiProvider)
```

- [ ] **Step 4: Update factory.create() to pass Gemini config**

```python
# In ProviderFactory.create() method, update the provider instantiation:

    @classmethod
    def create(cls, config: ProviderConfig) -> LLMProvider:
        """Create provider from config."""
        provider_name = config.PROVIDER.lower()

        if provider_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown provider '{config.PROVIDER}'. Available: {available}")

        provider_cls = cls._registry[provider_name]
        
        if provider_name == "gemini":
            return provider_cls(
                api_key=config.API_KEY,
                base_url=config.BASE_URL,
                model=config.MODEL_ID,
                project_id=config.GEMINI_PROJECT_ID,
                location=config.GEMINI_LOCATION,
                thinking_budget=config.GEMINI_THINKING_BUDGET,
                store_thinking=config.GEMINI_STORE_THINKING,
            )
        else:
            return provider_cls(
                api_key=config.API_KEY,
                base_url=config.BASE_URL,
                model=config.MODEL_ID,
            )
```

- [ ] **Step 5: Update providers/__init__.py**

```python
# In sanityops_agent/llm/providers/__init__.py:

from sanityops_agent.llm.providers.anthropic import AnthropicProvider
from sanityops_agent.llm.providers.openai import OpenAIProvider
from sanityops_agent.llm.providers.gemini import GeminiProvider

__all__ = ["AnthropicProvider", "OpenAIProvider", "GeminiProvider"]
```

- [ ] **Step 6: Update adapters/__init__.py**

```python
# In sanityops_agent/llm/adapters/__init__.py:

from sanityops_agent.llm.adapters.anthropic_adapter import AnthropicAdapter
from sanityops_agent.llm.adapters.openai_adapter import OpenAIAdapter
from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

__all__ = ["AnthropicAdapter", "OpenAIAdapter", "GeminiAdapter"]
```

- [ ] **Step 7: Run test to verify it passes**

```bash
pytest tests/unit/llm/test_factory.py::TestProviderFactoryGemini -v
```
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add sanityops_agent/llm/factory.py \
        sanityops_agent/llm/providers/__init__.py \
        sanityops_agent/llm/adapters/__init__.py \
        tests/unit/llm/test_factory.py
git commit -m "feat(factory): register GeminiProvider and pass Gemini config"
```

---

### Task 9: Run Full Test Suite

**Files:**
- N/A (verification task)

- [ ] **Step 1: Run all LLM tests**

```bash
pytest tests/unit/llm/ -v
```
Expected: All tests pass

- [ ] **Step 2: Run ruff linting**

```bash
ruff check sanityops_agent/ tests/
```
Expected: No errors

- [ ] **Step 3: Fix any lint errors**

If ruff reports errors, fix them and re-run.

- [ ] **Step 4: Final commit (if fixes were needed)**

```bash
git add -A
git commit -m "fix: resolve lint errors in Gemini implementation"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** All spec requirements covered:
  - ✓ Provider + Adapter architecture
  - ✓ Dual endpoint support (AI Studio / Vertex AI)
  - ✓ Text, tool calling, streaming
  - ✓ Thinking mode with configurable storage
  - ✓ Exception mapping
  - ✓ Factory registration

- [ ] **Placeholder scan:** No TBD, TODO, or vague instructions

- [ ] **Type consistency:** All method signatures match across tasks