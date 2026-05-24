"""Phase 6 A'.1 — Tavily web_search with DDG fallback tests.

Covers the fallback paths added when Tavily returns quota / auth /
transient errors, plus the DDG-missing edge case.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# Stub minimal deerflow.config.get_app_config so importing tools.py
# doesn't drag the full app boot.
if "deerflow.config" not in sys.modules:
    _cfg_mod = types.ModuleType("deerflow.config")
    _cfg_mod.get_app_config = lambda: MagicMock(
        get_tool_config=lambda _: None
    )
    sys.modules["deerflow.config"] = _cfg_mod

from deerflow.community.tavily import tools  # noqa: E402

from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)


def _tavily_response(n=2):
    return {
        "results": [
            {
                "title": f"Title {i}",
                "url": f"https://example.com/{i}",
                "content": f"Snippet {i}",
            }
            for i in range(n)
        ]
    }


class TavilyHappyPathTest(unittest.TestCase):
    """Phase 6 A'.1 regression — Tavily success path unchanged."""

    @patch.object(tools, "_get_tavily_client")
    def test_tavily_success_returns_normalized_json(self, mock_client):
        mock_client.return_value.search.return_value = _tavily_response(3)
        out = tools.web_search_tool.invoke({"query": "anything"})
        data = json.loads(out)
        self.assertEqual(len(data), 3)
        self.assertEqual(
            set(data[0].keys()), {"title", "url", "snippet"}
        )
        self.assertEqual(data[0]["title"], "Title 0")
        self.assertEqual(data[0]["url"], "https://example.com/0")
        self.assertEqual(data[0]["snippet"], "Snippet 0")


class TavilyFallbackToDdgTest(unittest.TestCase):
    """A'.1 core — quota/auth errors should not blow up, they should
    transparently route through DDG. Lead agent + fact-checker see
    a JSON list of results in the same shape, just from a different
    source. Sentinel string lets downstream count fallbacks."""

    def _patch_ddg(self, ddg_results):
        """Install a fake ddgs module with the given results."""
        fake_ddg_module = types.ModuleType("ddgs")

        class _FakeDDGS:
            def __init__(self, *args, **kwargs):
                pass

            def text(self, query, **kwargs):
                return ddg_results

        fake_ddg_module.DDGS = _FakeDDGS
        return patch.dict(sys.modules, {"ddgs": fake_ddg_module})

    def _run_with_tavily_error(self, exc):
        ddg_results = [
            {"title": "DDG Title", "href": "https://ddg.example/1",
             "body": "DDG snippet"},
        ]
        with patch.object(tools, "_get_tavily_client") as mock_client, \
                self._patch_ddg(ddg_results):
            mock_client.return_value.search.side_effect = exc
            out = tools.web_search_tool.invoke({"query": "anything"})
        return json.loads(out)

    def test_forbidden_falls_back(self):
        """Phase 6 A'.1 main trigger: ForbiddenError = quota."""
        data = self._run_with_tavily_error(
            ForbiddenError("This request exceeds your plan's set "
                           "usage limit.")
        )
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["title"], "DDG Title")
        self.assertEqual(data[0]["url"], "https://ddg.example/1")
        self.assertEqual(data[0]["snippet"], "DDG snippet")

    def test_usage_limit_falls_back(self):
        data = self._run_with_tavily_error(
            UsageLimitExceededError("quota out")
        )
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["title"], "DDG Title")

    def test_invalid_api_key_falls_back(self):
        data = self._run_with_tavily_error(
            InvalidAPIKeyError("bad key")
        )
        self.assertIsInstance(data, list)

    def test_missing_api_key_falls_back(self):
        # MissingAPIKeyError takes no args (see tavily/errors.py)
        data = self._run_with_tavily_error(MissingAPIKeyError())
        self.assertIsInstance(data, list)

    def test_bad_request_falls_back(self):
        """Transient errors also fall back rather than 100% failing."""
        data = self._run_with_tavily_error(BadRequestError("bad req"))
        self.assertIsInstance(data, list)

    def test_timeout_falls_back(self):
        data = self._run_with_tavily_error(TimeoutError("slow"))
        self.assertIsInstance(data, list)

    def test_unexpected_exception_still_falls_back(self):
        """ANY exception triggers fallback — we'd rather have a degraded
        result than no result, since fact-checker depends on this."""
        data = self._run_with_tavily_error(
            RuntimeError("never seen this before")
        )
        self.assertIsInstance(data, list)


class DdgMissingEdgeCaseTest(unittest.TestCase):
    """If DDG library isn't installed, return a clear error JSON
    rather than letting the exception propagate. Fact-checker can
    then log this as 'unverifiable_no_source' downstream."""

    def test_ddg_import_failure_returns_error_json(self):
        # Force ImportError by removing ddgs from sys.modules if there
        # and intercepting via meta_path
        sys.modules.pop("ddgs", None)
        # Inject a meta path finder that raises on ddgs import
        import importlib.abc

        class _BlockDDG(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "ddgs":
                    raise ImportError("blocked for test")
                return None

        blocker = _BlockDDG()
        sys.meta_path.insert(0, blocker)
        try:
            with patch.object(tools, "_get_tavily_client") as mock_client:
                mock_client.return_value.search.side_effect = ForbiddenError(
                    "quota"
                )
                out = tools.web_search_tool.invoke({"query": "x"})
            data = json.loads(out)
            self.assertIn("error", data)
            self.assertIn("DDG fallback", data["error"])
            self.assertEqual(data["query"], "x")
        finally:
            sys.meta_path.remove(blocker)


class DdgFailsAfterTavilyTest(unittest.TestCase):
    """Both backends down: return error JSON, do not raise.

    Without this guard, ToolErrorHandlingMiddleware would catch the
    raise but the fact-checker still gets the orphan-risk path."""

    def test_ddg_runtime_error_after_tavily_quota(self):
        fake_ddg_module = types.ModuleType("ddgs")

        class _BrokenDDGS:
            def __init__(self, *args, **kwargs):
                pass

            def text(self, *a, **k):
                raise RuntimeError("ddg server down")

        fake_ddg_module.DDGS = _BrokenDDGS
        with patch.object(tools, "_get_tavily_client") as mock_client, \
                patch.dict(sys.modules, {"ddgs": fake_ddg_module}):
            mock_client.return_value.search.side_effect = ForbiddenError("q")
            out = tools.web_search_tool.invoke({"query": "x"})
        data = json.loads(out)
        self.assertIn("error", data)
        self.assertIn("DDG fallback failed", data["error"])


class SentinelTaggingTest(unittest.TestCase):
    """The sentinel is the wrapper-side observability handle. Phase 6
    closeout pipeline parses it from gateway.log to count fallback
    invocations per run."""

    def test_sentinel_constant_defined(self):
        self.assertEqual(
            tools.TAVILY_FALLBACK_SENTINEL,
            "[phase6-tavily-fallback]",
        )


if __name__ == "__main__":
    unittest.main()
