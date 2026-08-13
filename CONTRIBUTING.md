# Contributing to Sanityops Agent

Thank you for your interest in contributing to Sanityops Agent! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Development Setup

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Getting Started

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/your-username/sanityops-agent.git
   cd sanityops-agent
   ```

2. **Create a virtual environment and install dependencies**

   Using uv (recommended):
   ```bash
   uv sync --all-extras
   ```

   Using pip:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   uv run ruff check . --fix
   ```

### Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your API credentials for testing.

## Project Structure

```
sanityops_agent/
├── core/          # Message model, Agent base class, Context
├── llm/           # Provider abstraction, adapters, exceptions
│   └── providers/ # OpenAI, Anthropic implementations
├── tools/         # Tool base class, registry, context
│   └── builtins/  # Built-in tools (bash, file_ops, todo, etc.)
├── agents/        # Parent agent, Sub-agent, Factory
├── hooks/         # Hook system (before/after tool, LLM, etc.)
└── managers/      # State, skill, todo, history persistence

tests/
├── unit/          # Isolated component tests
├── integration/   # Multi-component workflow tests
├── mocks/         # MockLLMProvider, mock tools
└── conftest.py    # Pytest fixtures
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The configuration is in `pyproject.toml`.

### Key Guidelines

- **Line length**: 88 characters
- **Imports**: Sorted with isort, first-party imports grouped
- **Type hints**: Encouraged for public APIs
- **Docstrings**: Use for modules, classes, and public functions

### Running Linting

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix
```

## Testing

We use pytest with pytest-asyncio for testing.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=sanityops_agent

# Run specific test file
uv run pytest tests/unit/test_message.py
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use the mocks in `tests/mocks/` for LLM providers
- Mark async tests with `@pytest.mark.asyncio`

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run checks locally**
   ```bash
   uv run ruff check .
   uv run pytest
   ```

4. **Commit your changes**
   - Use clear, descriptive commit messages
   - Reference any related issues

5. **Push and create a pull request**
   - Fill out the PR template completely
   - Link any related issues
   - Request review from maintainers

### PR Requirements

- All tests must pass
- No linting errors
- New code must have tests
- Breaking changes must be documented

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.
