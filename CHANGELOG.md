# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
