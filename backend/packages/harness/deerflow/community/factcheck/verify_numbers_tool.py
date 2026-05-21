"""Phase 1 fact-check: deterministic claim verification tool.

The `verify_numbers` tool is called by fact-checker subagents (sonnet / gpt-4o)
to verify a specific claim against a cited source URL.

Pipeline:
    1. Extract numbers + units from claim_text (number_extractor)
    2. Fetch source_url via Jina Reader (reused infrastructure)
    3. Level B: check each number is present in source content
    4. Level A: for numbers that ARE in source, compare direction words
       in claim window vs source window near that number

Returns structured JSON verdict — subagent uses it instead of doing
NLI inference itself (which PoC #4 case 3 showed is unreliable).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from langchain.tools import tool

from deerflow.community.factcheck.direction_classifier import (
    Direction,
    classify_direction,
    is_direction_conflict,
)
from deerflow.community.factcheck.number_extractor import (
    NumberClaim,
    extract_numbers,
    number_in_source,
)

logger = logging.getLogger(__name__)

_JINA_URL = "https://r.jina.ai/"
_FETCH_TIMEOUT = 60.0
_SOURCE_TRUNCATE_CHARS = 8000   # cap to avoid huge LLM input downstream


async def _fetch_source(url: str) -> tuple[str, str | None]:
    """Fetch URL via Jina Reader.

    Returns (content, error_message_or_None).
    """
    headers = {
        "Content-Type": "application/json",
        "X-Return-Format": "markdown",
        # NOTE: Deliberately NO X-Timeout header (causes Jina to hold connection
        # for the full window — see jina_client.py comment).
    }
    if os.getenv("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _JINA_URL,
                headers=headers,
                json={"url": url},
                timeout=_FETCH_TIMEOUT,
            )
        if response.status_code != 200:
            return "", f"Jina HTTP {response.status_code}: {response.text[:150]}"
        text = response.text or ""
        if not text.strip():
            return "", "Jina returned empty response"
        return text[:_SOURCE_TRUNCATE_CHARS], None
    except Exception as e:
        return "", f"Request to Jina failed: {type(e).__name__}: {e}"


def _find_window_around_value(value_str: str, source_text: str, window_chars: int = 200) -> str:
    """Locate value_str in source_text, return ±window_chars context.

    Used for Level A direction analysis: pull the sentence(s) around where
    the number appears in source.
    """
    idx = source_text.find(value_str)
    if idx == -1:
        return ""
    start = max(0, idx - window_chars)
    end = min(len(source_text), idx + len(value_str) + window_chars)
    return source_text[start:end]


def _verify_single_claim(
    claim_number: NumberClaim,
    source_text: str,
) -> dict[str, Any]:
    """Run Level B + Level A on a single number.

    Returns:
        {
            "raw": <claim text>,
            "level_b_present": bool,
            "level_a_applicable": bool,
            "level_a_claim_dir": Direction,
            "level_a_source_dir": Direction,
            "level_a_conflict": bool,
            "fail_reasons": [str],
        }
    """
    result: dict[str, Any] = {
        "raw": claim_number.raw,
        "level_b_present": False,
        "level_a_applicable": False,
        "level_a_claim_dir": "none",
        "level_a_source_dir": "none",
        "level_a_conflict": False,
        "fail_reasons": [],
    }

    # Level B
    present = number_in_source(claim_number, source_text)
    result["level_b_present"] = present
    if not present:
        result["fail_reasons"].append("number_missing_in_source")
        return result

    # Level A — only applicable if number IS in source
    import re as _re
    val_match = _re.search(r"\d+(?:\.\d+)?", claim_number.raw)
    if val_match:
        value_str = val_match.group()
        source_window = _find_window_around_value(value_str, source_text)
        claim_dir = classify_direction(claim_number.window)
        source_dir = classify_direction(source_window) if source_window else "none"
        result["level_a_applicable"] = True
        result["level_a_claim_dir"] = claim_dir
        result["level_a_source_dir"] = source_dir
        if is_direction_conflict(claim_dir, source_dir):
            result["level_a_conflict"] = True
            result["fail_reasons"].append("direction_conflict")

    return result


@tool("verify_numbers", parse_docstring=True)
async def verify_numbers_tool(
    description: str,
    claim_text: str,
    source_url: str,
) -> str:
    """Verify whether a claim's specific numerical assertions are supported by the cited source URL.

    Performs deterministic two-stage check:
      Level B: extract every (number + unit) from the claim, fetch source via
               Jina, check each number is present.
      Level A: for numbers that ARE in source, compare direction words
               (rises/drops/+/-/上升/下降) in the claim's local context
               vs the source's local context around that number.

    Returns a JSON-serialized verdict that the calling fact-checker subagent
    should respect verbatim — do NOT second-guess the deterministic result
    based on your own reading.

    Args:
        description: Why you are verifying this claim. ALWAYS PROVIDE FIRST.
        claim_text: The full sentence (or paragraph) containing the claim.
        source_url: The URL cited as support for the claim.
    """
    # Step 1: extract numbers from claim
    numbers = extract_numbers(claim_text)
    if not numbers:
        return json.dumps({
            "verdict": "unverifiable",
            "summary": "no specific numerical claim found in claim_text",
            "claim_text": claim_text,
            "source_url": source_url,
            "checks": [],
            "fail_reasons": [],
        })

    # Step 2: fetch source
    source_text, fetch_error = await _fetch_source(source_url)
    if fetch_error:
        return json.dumps({
            "verdict": "unverifiable",
            "summary": f"could not fetch source: {fetch_error}",
            "claim_text": claim_text,
            "source_url": source_url,
            "fetch_error": fetch_error,
            "checks": [],
            "fail_reasons": ["source_fetch_failed"],
        })

    # Step 3: verify each number
    checks = [_verify_single_claim(n, source_text) for n in numbers]

    # Step 4: aggregate verdict
    all_fail_reasons: list[str] = []
    for c in checks:
        all_fail_reasons.extend(c["fail_reasons"])

    has_missing = any("number_missing_in_source" in c["fail_reasons"] for c in checks)
    has_direction = any("direction_conflict" in c["fail_reasons"] for c in checks)

    if has_missing or has_direction:
        verdict = "unsupported"
        # Build summary
        if has_direction:
            summary = "Level A direction conflict detected"
        else:
            summary = f"{sum(1 for c in checks if not c['level_b_present'])} number(s) missing in source"
    else:
        verdict = "supported"
        summary = f"all {len(checks)} numerical claims found in source"

    return json.dumps({
        "verdict": verdict,
        "summary": summary,
        "claim_text": claim_text,
        "source_url": source_url,
        "checks": checks,
        "fail_reasons": all_fail_reasons,
    }, ensure_ascii=False)
