"""Phase 6.0 Tavily tests. Fully offline — HTTP is faked, never live."""

import json
import socket
import urllib.error

import pytest

from worker.models.search import SearchResult
from worker.pipelines.search_pipeline import run_search_discovery
from worker.search import config as search_config
from worker.search import registry as search_registry
from worker.search.base import SearchProvider
from worker.search.mock_provider import MockSearchProvider
from worker.search.readiness import format_readiness, get_readiness
from worker.search.tavily import (
    MalformedError,
    NetworkError,
    NotConfiguredError,
    RateLimitedError,
    ServerError,
    TavilyError,
    TavilySearchProvider,
    UnauthorizedError,
)


def _payload(*items):
    return {"query": "java developer intern", "response_time": 1.2, "results": list(items)}


def _hit(**overrides):
    base = {
        "title": "Java Developer Intern — Acme",
        "url": "https://boards.greenhouse.io/acme/jobs/42",
        "content": "Java internship in Chennai. Freshers welcome.",
        "score": 0.91,
        "published_date": "2026-09-20T10:00:00.000+00:00",
    }
    base.update(overrides)
    return base


class FakeResponse:
    """Minimal urlopen stand-in: status + headers + body."""

    def __init__(self, body: bytes, status: int = 200, headers: dict = None) -> None:
        self._body = body
        self.status = status
        self.headers = headers or {}

    def read(self, *args):
        return self._body


def _opener(behavior):
    """Fixed behavior: value (status, headers, body) | exception."""

    def opener(url: str, timeout: float, *, body: bytes, headers: dict):
        if isinstance(behavior, BaseException):
            raise behavior
        status, response_headers, response_body = behavior
        return FakeResponse(response_body, status=status, headers=response_headers)

    return opener


def _provider(behavior, **kwargs):
    kwargs.setdefault("api_key", "tvly-test-key")
    return TavilySearchProvider(opener=_opener(behavior), **kwargs)


# --- 1. configured provider -----------------------------------------------------


def test_configured_provider_reports_ready():
    provider = TavilySearchProvider(api_key="tvly-test-key")
    assert provider.is_configured() is True
    health = provider.health_check()
    assert health.ok is True and "tvly-test-key" not in health.message


# --- 2. missing key --------------------------------------------------------------


def test_missing_key_clean_error_and_unhealthy():
    provider = TavilySearchProvider(api_key="")
    assert provider.is_configured() is False
    assert provider.health_check().ok is False
    with pytest.raises(NotConfiguredError):
        provider.search("java developer")


# --- 3. successful normalized response ----------------------------------------------


def test_successful_response_normalized():
    provider = _provider((200, {}, json.dumps(_payload(_hit())).encode()))
    results = provider.search("java developer intern", max_results=5)
    assert len(results) == 1
    job = results[0]
    assert isinstance(job, SearchResult)
    assert job.title == "Java Developer Intern — Acme"
    assert job.url == "https://boards.greenhouse.io/acme/jobs/42"
    assert "Freshers welcome" in job.snippet
    assert job.source_name == "tavily"
    assert job.raw_data["score"] == 0.91
    assert job.raw_data["published_date"] == "2026-09-20T10:00:00.000+00:00"
    assert job.raw_data["query"] == "java developer intern"


# --- 4. malformed responses ------------------------------------------------------------


def test_malformed_responses_rejected():
    for bad in (
        (200, {}, b"not json"),
        (200, {}, json.dumps({"nope": []}).encode()),
        (200, {}, json.dumps(["a", "list"]).encode()),
        (400, {}, b"bad request"),
    ):
        with pytest.raises(TavilyError):
            _provider(bad).search("java")


def test_url_less_items_skipped():
    provider = _provider((200, {}, json.dumps(_payload(_hit(), {"title": "No URL"}, "junk")).encode()))
    results = provider.search("java")
    assert len(results) == 1 and results[0].url.startswith("https://")


# --- 5. timeout / network ---------------------------------------------------------------


def test_timeout_and_network_failure():
    with pytest.raises(NetworkError):
        _provider(socket.timeout("timed out")).search("java")
    with pytest.raises(NetworkError):
        _provider(urllib.error.URLError("dns fail")).search("java")


# --- 6/7. unauthorized + rate limit ------------------------------------------------------------


def test_unauthorized_hides_key():
    with pytest.raises(UnauthorizedError) as caught:
        _provider((401, {}, b"unauthorized")).search("java")
    assert "tvly-test-key" not in str(caught.value)


def test_rate_limited_with_retry_after():
    with pytest.raises(RateLimitedError) as caught:
        _provider((429, {"Retry-After": "45"}, b"slow")).search("java")
    assert caught.value.retry_after == 45
    with pytest.raises(ServerError):
        _provider((503, {}, b"oops")).search("java")


# --- 8. registry -------------------------------------------------------------------------------


def test_registry_selects_tavily_and_mock():
    assert isinstance(search_registry.get_provider("mock"), MockSearchProvider)
    tavily = search_registry.get_provider("tavily")
    assert isinstance(tavily, TavilySearchProvider)
    assert tavily.provider_name == "tavily"
    assert {p.provider_name for p in search_registry.get_all_providers()} >= {"mock", "tavily"}


# --- 9. mock untouched ------------------------------------------------------------------------------


def test_mock_provider_still_offline():
    results = MockSearchProvider().search("java developer intern")
    assert len(results) == 5
    assert all(r.raw_data.get("mock") is True for r in results)


# --- 10. readiness --------------------------------------------------------------------------------------


def test_readiness_reports_without_network_or_secrets(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    readiness = get_readiness()
    assert readiness.tavily_configured is True
    assert readiness.gemini_configured is False
    # A safe default model is always present even with no env override.
    assert readiness.gemini_model_set is True
    assert readiness.database_configured is False
    assert readiness.search_provider == "tavily"
    text = format_readiness(readiness)
    assert "tvly-test-key" not in text and "YES" in text

    monkeypatch.delenv("TAVILY_API_KEY")
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    readiness = get_readiness()
    assert readiness.tavily_configured is False
    assert readiness.search_provider == "tavily"
    assert "not silently fall back" in format_readiness(readiness)

    monkeypatch.setenv("SEARCH_PROVIDER", "mock")
    assert "not silently fall back" not in format_readiness(get_readiness())


def test_tavily_selected_without_key_errors_cleanly_in_pipeline(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilySearchProvider(api_key="")
    result = run_search_discovery(provider=provider, max_queries=2)
    assert result.results_found == 0
    assert any("not_configured" in e or "NotConfigured" in e for e in result.errors)


# --- 11. no secret leakage -------------------------------------------------------------------------------


def test_errors_never_contain_key():
    key = "tvly-super-secret-12345"
    cases = [
        (401, {}, b"nope"),
        (429, {}, b"slow"),
        (500, {}, b"boom"),
        (200, {}, b"garbage"),
        socket.timeout("x"),
    ]
    for behavior in cases:
        try:
            _provider(behavior, api_key=key).search("java")
            raise AssertionError("expected TavilyError")
        except TavilyError as exc:
            assert key not in str(exc), behavior
            assert key not in repr(exc), behavior
