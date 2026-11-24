# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-11-24

### Added
- `RuntimeAgentConfig` export and `RedactedThinkingBlock`.
- Per-run deadlines, results and token budgets; delegated work shares the
  parent's remaining time and budget, and `AgentResult.usage_complete`
  reports incomplete provider usage.
- Shared tool dispatch in `ToolAgent` with execution-time permission checks
  and JSON Schema validation of tool arguments.
- Workspace-bounded file, glob and bash tools, and bounded tool history.
- Stream assembly that preserves every streamed tool call and signed
  thinking block.

### Changed
- **Breaking:** Anthropic requests require an explicit positive `max_tokens`;
  there is no bundled default.
- **Breaking:** `ThinkingBlock.thinking_signature` is renamed to `signature`;
  a read-only alias remains.
- **Breaking:** the `ProviderConfig.LLM_PROVIDER` attribute is removed. The
  `LLM_PROVIDER` environment variable still sets `PROVIDER`.
- **Breaking:** model-requested file tools require a host-configured
  workspace, and bash requires a host-provided shell runner.
- **Breaking:** `AgentConfig.total_timeout` is now a float.
- OpenAI tool definitions opt into strict mode explicitly; OpenAI streams
  read the response body to EOF instead of stopping at `[DONE]`.

### Fixed
- State files written with the legacy `thinking_signature` key restore their
  signatures.
- Legacy saved conversations keep signed thinking turns resumable.

## [0.0.3] - 2025-08-13

### Added
- Open source essentials: CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- Issue and PR templates for GitHub
- CI workflow with ruff and pytest
- Enhanced Ruff configuration for code quality (bugbear, pyupgrade)
- PyPI badges in README

### Changed
- License identifier in pyproject.toml to SPDX format ("Apache 2.0" → "Apache-2.0")

## [0.0.2] - 2025-08-10

### Changed
- Renamed project to Sanityops Agent
- Restructured package under `sanityops_agent` namespace

## [0.0.1] - 2025-08-10

### Added
- Initial release
- Unified Message Model with subclassed content blocks
- LLM Provider Abstraction (OpenAI, Anthropic)
- Plugin Tools system with JSON Schema parameters
- Multi-Agent support (Parent/Sub agents)
- Hook System for tool and LLM interception
- State Persistence for conversation context
- Streaming support with StreamChunk
- Retry logic with exponential backoff
- Built-in tools: bash, file_ops, todo, history, skill, glob
