# Gemini Provider Design Document

## Overview

Add a Gemini provider to sanityops_agent to support calling Google Gemini models. Uses the native Gemini SDK, supporting core features (text, tool calling, streaming) and Thinking mode.

## Architecture

Follows the existing Provider + Adapter pattern:

```
llm/
├── providers/
│   ├── anthropic.py      # existing
│   ├── openai.py         # existing
│   └── gemini.py         # new
├── adapters/
│   ├── anthropic_adapter.py  # existing
│   ├── openai_adapter.py     # existing
│   └── gemini_adapter.py     # new
└── factory.py            # add gemini registration
```

## Configuration Extension

### ProviderConfig New Fields

```python
class ProviderConfig(BaseSettings):
    # existing fields...

    # Gemini-specific fields
    GEMINI_PROJECT_ID: str | None = None      # Vertex AI project ID
    GEMINI_LOCATION: str | None = None        # Vertex AI region
    GEMINI_THINKING_BUDGET: int | None = None # thinking token budget (None=disabled)
    GEMINI_STORE_THINKING: bool = False       # whether to store thinking in history
```

### Endpoint Selection Logic

- `GEMINI_PROJECT_ID` is set → use Vertex AI endpoint
- `GEMINI_PROJECT_ID` is empty → use Google AI Studio endpoint

## GeminiAdapter

Responsible for bidirectional conversion between internal message format and Gemini API format.

### Gemini API Message Format Characteristics

- Uses `contents` instead of `messages`
- Each message contains a `parts` array
- `role` only has `user` and `model` (assistant is called model)
- System prompt is passed separately via `systemInstruction` field
- Tool calling uses `functionCall` / `functionResponse` structure

### Core Conversion Logic

**to_provider method conversion rules:**

1. Extract system messages → `systemInstruction`
2. Convert role: `ASSISTANT` → `"model"`
3. Convert content blocks → parts:
   - `TextBlock` → `{"text": ...}`
   - `ThinkingBlock` → `{"thought": ...}`
   - `ToolUseBlock` → `{"functionCall": {...}}`
   - `ToolResultBlock` → `{"functionResponse": {...}}`
4. Convert tools → `functionDeclarations`

**from_provider_response method conversion rules:**

Reverse conversion: Gemini response → `LLMResponse`

**from_provider_chunk method:**

Streaming chunk conversion, supporting text, thought, and functionCall types.

### Thinking Handling

- Based on `store_thinking` config, decide whether to keep thinking blocks in response
- When not stored, thinking is only displayed during streaming, not written to message history

## GeminiProvider

Responsible for actual API calls.

### Initialization

```python
class GeminiProvider(LLMProvider):
    DEFAULT_MODEL = "gemini-3.5-flash-lite"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        # Vertex AI config
        project_id: str | None = None,
        location: str | None = None,
        # Thinking config
        thinking_budget: int | None = None,
        store_thinking: bool = False,
        timeout: float = 120.0,
    ):
        self._client: Client | None = None
```

### Endpoint Initialization

```python
# AI Studio mode
client = Client(api_key=api_key)

# Vertex AI mode
client = Client(
    vertexai=True,
    project=project_id,
    location=location,
)
```

### Method Implementations

- `create()` — non-streaming call
- `stream()` — streaming call
- `close()` — close client connection

### Thinking Configuration Passing

Set `thinking_budget` via `generation_config`, parse `thought` blocks in response and convert to `ThinkingBlock`.

## Exception Handling

### Exception Mapping

| Gemini SDK Exception | Internal Exception |
|---------------------|-------------------|
| `AuthenticationError` | `AuthenticationError` |
| `RateLimitError` | `RateLimitError` |
| `APIError` (5xx) | `ProviderUnavailableError` |
| `APIError` (4xx) | `LLMError` |
| `InvalidRequestError` (context related) | `ContextWindowExceededError` |
| `InvalidRequestError` (other) | `LLMError` |

### Retry Information

Extract `retry_after` from `RateLimitError` response headers.

## Factory Registration

### factory.py

```python
from sanityops_agent.llm.providers.gemini import GeminiProvider

ProviderFactory.register("gemini", GeminiProvider)
```

### llm/providers/__init__.py

```python
from sanityops_agent.llm.providers.gemini import GeminiProvider

__all__ = ["AnthropicProvider", "OpenAIProvider", "GeminiProvider"]
```

### llm/adapters/__init__.py

```python
from sanityops_agent.llm.adapters.gemini_adapter import GeminiAdapter

__all__ = ["AnthropicAdapter", "OpenAIAdapter", "GeminiAdapter"]
```

## Dependencies

New dependency: `google-genai` — Google's new unified SDK (supports both AI Studio and Vertex AI)

## Usage Examples

### AI Studio Mode

```python
# .env
LLM_PROVIDER=gemini
LLM_API_KEY=your-api-key
LLM_MODEL_ID=gemini-3.5-flash-lite

# Python
config = ProviderConfig()
provider = ProviderFactory.create(config)
```

### Vertex AI Mode

```python
# .env
LLM_PROVIDER=gemini
LLM_API_KEY=your-api-key
LLM_MODEL_ID=gemini-3.5-flash-lite
LLM_GEMINI_PROJECT_ID=your-project-id
LLM_GEMINI_LOCATION=us-central1

# Python
config = ProviderConfig()
provider = ProviderFactory.create(config)
```

### Enable Thinking

```python
# .env
LLM_GEMINI_THINKING_BUDGET=2048
LLM_GEMINI_STORE_THINKING=true
```