"""Tests for the Retry-After header extraction helper."""

import pytest

from sanityops_agent.llm.providers.anthropic import (
    _extract_retry_after as anthropic_extract,
)
from sanityops_agent.llm.providers.openai import _extract_retry_after as openai_extract


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers


class _FakeExc:
    def __init__(self, headers=None, no_response=False):
        self.response = None if no_response else _FakeResponse(headers or {})


@pytest.mark.parametrize("extract", [anthropic_extract, openai_extract])
class TestExtractRetryAfter:
    def test_returns_int_from_header(self, extract):
        exc = _FakeExc({"retry-after": "30"})
        assert extract(exc) == 30

    def test_returns_none_when_header_absent(self, extract):
        exc = _FakeExc({})
        assert extract(exc) is None

    def test_returns_none_when_no_response(self, extract):
        exc = _FakeExc(no_response=True)
        assert extract(exc) is None

    def test_returns_none_when_header_non_numeric(self, extract):
        exc = _FakeExc({"retry-after": "abc"})
        assert extract(exc) is None

    def test_returns_none_for_non_positive(self, extract):
        exc = _FakeExc({"retry-after": "-5"})
        assert extract(exc) is None

    def test_handles_fractional_seconds(self, extract):
        exc = _FakeExc({"retry-after": "2.5"})
        assert extract(exc) == 2
