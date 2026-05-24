import json
import logging

from langchain.tools import tool
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)

from deerflow.config import get_app_config

logger = logging.getLogger(__name__)

# Phase 6 A'.1: sentinel for downstream observability (wrapper sentinel /
# closeout). Fallback path may degrade fact-checker quality vs Tavily, so
# we want to count how often it fires per run.
TAVILY_FALLBACK_SENTINEL = "[phase6-tavily-fallback]"


def _get_tavily_client() -> TavilyClient:
    config = get_app_config().get_tool_config("web_search")
    api_key = None
    if config is not None and "api_key" in config.model_extra:
        api_key = config.model_extra.get("api_key")
    return TavilyClient(api_key=api_key)


def _normalize_tavily(results: list[dict]) -> list[dict]:
    return [
        {
            "title": r["title"],
            "url": r["url"],
            "snippet": r["content"],
        }
        for r in results
    ]


def _ddg_fallback(query: str, max_results: int) -> str:
    """Phase 6 A'.1 fallback when Tavily quota / auth / transient fails.

    DDG has no quota and no key. Quality is lower than Tavily, so fact-
    checker results MAY degrade — flagged via TAVILY_FALLBACK_SENTINEL
    in logs for the wrapper sentinel pipeline to count.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.error(
            f"{TAVILY_FALLBACK_SENTINEL} DDG library not installed — "
            f"web_search is fully degraded for query={query[:80]!r}"
        )
        return json.dumps(
            {
                "error": "Tavily unavailable and DDG fallback library not "
                         "installed. Install ddgs to enable fallback.",
                "query": query,
            },
            ensure_ascii=False,
        )

    try:
        ddgs = DDGS(timeout=30)
        raw = ddgs.text(
            query,
            region="wt-wt",
            safesearch="moderate",
            max_results=max_results,
        )
        results = list(raw) if raw else []
    except Exception as e:
        logger.error(
            f"{TAVILY_FALLBACK_SENTINEL} DDG fallback failed: "
            f"{type(e).__name__}: {e}"
        )
        return json.dumps(
            {"error": f"DDG fallback failed: {e}", "query": query},
            ensure_ascii=False,
        )

    normalized = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "snippet": r.get("body", r.get("snippet", "")),
        }
        for r in results
    ]
    logger.info(
        f"{TAVILY_FALLBACK_SENTINEL} via DDG: query={query[:50]!r} "
        f"results={len(normalized)}"
    )
    return json.dumps(normalized, indent=2, ensure_ascii=False)


@tool("web_search", parse_docstring=True)
def web_search_tool(query: str) -> str:
    """Search the web.

    Args:
        query: The query to search for.
    """
    config = get_app_config().get_tool_config("web_search")
    max_results = 5
    if config is not None and "max_results" in config.model_extra:
        max_results = config.model_extra.get("max_results")

    # Try Tavily first
    try:
        client = _get_tavily_client()
        res = client.search(query, max_results=max_results)
        normalized = _normalize_tavily(res["results"])
        return json.dumps(normalized, indent=2, ensure_ascii=False)
    except (
        ForbiddenError,
        UsageLimitExceededError,
        InvalidAPIKeyError,
        MissingAPIKeyError,
    ) as e:
        # Quota / auth — Tavily not coming back this run. Fall back to DDG.
        logger.warning(
            f"{TAVILY_FALLBACK_SENTINEL} Tavily unavailable "
            f"({type(e).__name__}: {e}); switching to DDG for "
            f"query={query[:80]!r}"
        )
        return _ddg_fallback(query, max_results)
    except (BadRequestError, TimeoutError) as e:
        # Transient — also fall back rather than 100% fail, since the
        # query is still likely answerable via DDG.
        logger.warning(
            f"{TAVILY_FALLBACK_SENTINEL} Tavily transient "
            f"({type(e).__name__}: {e}); switching to DDG for "
            f"query={query[:80]!r}"
        )
        return _ddg_fallback(query, max_results)
    except Exception as e:
        # Truly unexpected — log loudly, still try DDG so the fact-check
        # path has SOMETHING to work with.
        logger.exception(
            f"{TAVILY_FALLBACK_SENTINEL} Tavily unexpected error "
            f"({type(e).__name__}: {e}); attempting DDG"
        )
        return _ddg_fallback(query, max_results)


@tool("web_fetch", parse_docstring=True)
def web_fetch_tool(url: str) -> str:
    """Fetch the contents of a web page at a given URL.
    Only fetch EXACT URLs that have been provided directly by the user or have been returned in results from the web_search and web_fetch tools.
    This tool can NOT access content that requires authentication, such as private Google Docs or pages behind login walls.
    Do NOT add www. to URLs that do NOT have them.
    URLs must include the schema: https://example.com is a valid URL while example.com is an invalid URL.

    Args:
        url: The URL to fetch the contents of.
    """
    client = _get_tavily_client()
    res = client.extract([url])
    if "failed_results" in res and len(res["failed_results"]) > 0:
        return f"Error: {res['failed_results'][0]['error']}"
    elif "results" in res and len(res["results"]) > 0:
        result = res["results"][0]
        return f"# {result['title']}\n\n{result['raw_content'][:4096]}"
    else:
        return "Error: No results found"
