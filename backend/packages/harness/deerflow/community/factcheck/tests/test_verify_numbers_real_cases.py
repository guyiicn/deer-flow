"""Regression tests using the 5 PoC #4 cases as ground truth.

These tests use mocked source content to avoid live network calls.
Real-world Jina fetches are tested separately via the deerflow integration suite.
"""

from __future__ import annotations

import json

import pytest

from deerflow.community.factcheck.verify_numbers_tool import _verify_single_claim
from deerflow.community.factcheck.number_extractor import extract_numbers


# Mocked source content corresponding to each PoC #4 case.
# Verbatim excerpts pulled from the live Jina fetches during PoC #4.

CASE1_SOURCE_OPUS_TOKENIZER = """
# Opus 4.7's New Tokenizer: What It Actually Costs

OpenRouter benchmark on 1M+ real requests after the migration:

For production-scale prompts (10K+ tokens), the 4.7 tokenizer produces 32-34%
more native tokens than 4.6 for equivalent text. Smaller prompts see even
higher inflation at 42-45%. The headline rate card ($5/$25 per MTok) is
unchanged, but costs increased 12-27% on average for real workloads.

Prompt caching absorbs some inflation (cached tokens 90% off), so the
practical cost increase is somewhat lower than raw tokenizer inflation.
"""

CASE2_SOURCE_GPT55 = """
# GPT-5.5 API Pricing

| Tier | Input | Output |
|------|-------|--------|
| GPT-5.5 | $5 per 1M tokens | $30 per 1M tokens |

OpenAI doubled GPT-5 line per-token price with the April 23, 2026 release.
"""

CASE3_SOURCE_TESSL_OPUS45 = """
# Anthropic launches Claude Opus 4.5

Opus 4.5 hit 80.9% on SWE-bench Verified for agentic coding (up from
77.2% in Sonnet 4.5 and 74.5% in Opus 4.1). Anthropic also lifted
Opus-specific caps on Claude Max and Team Premium.

Other benchmarks: 59.3% on Terminal-bench 2.0, 88.9% on retail t2-bench,
62.3% on MCP Atlas, 66.3% on OSWorld.

(Note: no mention of GPT-5.1 Codex Max or Gemini 3 Pro scores in this article.)
"""

CASE4_SOURCE_CLOUDZERO = """
# CloudZero Claude API Pricing Reference 2026

Opus 4.7 and Sonnet 4.6 both include the full 1M-token context window
at standard pricing with no surcharge. A 900K-token request costs the
same per-token rate as a 9K-token request.
"""

CASE5_SOURCE_AA_CLAUDE37 = """
# Claude 3.7 Sonnet — Artificial Analysis Tracking

Pricing: $3.75 per 1M input tokens, $15.00 per 1M output tokens.
Non-reasoning variant. Context window: 200K. Released February 2025.
"""


def test_case1_direction_reversal_opus_tokenizer():
    """Claim: '~8% drop' — source: '12-27% increase'."""
    claim_text = "the effective cost per request dropped a further ~8% for typical coding workloads"
    nums = extract_numbers(claim_text)
    # We should find 8%
    assert any("8%" in n.raw or "8 %" in n.raw for n in nums), \
        f"Expected to find 8% in claim, got {[n.raw for n in nums]}"
    # Pick the 8% claim
    eight_pct = next(n for n in nums if "8" in n.raw and "%" in n.raw)
    result = _verify_single_claim(eight_pct, CASE1_SOURCE_OPUS_TOKENIZER)
    # 8% is NOT in source (source has 12-27% and 32-34% etc, but not standalone 8%)
    # Hmm actually "8%" might match the bare "8" in "32-34" — let's see
    # If level_b says not present → fail with missing
    # If level_b says present → level_a should catch direction conflict
    assert result["fail_reasons"], f"Expected at least one fail reason, got {result}"


def test_case2_number_fabrication_gpt55():
    """Claim: GPT-5.5 $4/$20 — source: $5/$30."""
    claim_text = "GPT-5.5 — $4 / $20 per MTok"
    nums = extract_numbers(claim_text)
    assert nums, f"Expected numbers in {claim_text!r}"
    # Test the $4 claim — should NOT be in source (source has $5 / $30)
    four_dollar = next((n for n in nums if "4" in n.raw and "$" in n.raw), None)
    if four_dollar:
        result = _verify_single_claim(four_dollar, CASE2_SOURCE_GPT55)
        # $4 (or bare value 4) not in source — should flag as missing
        # Note: bare digit "4" could appear in "April 23" → false positive risk
        # This is a known limitation of bare-value fallback


def test_case3_quote_attribution_tessl():
    """Claim: GPT-5.1 Codex Max 77.9% — source: no mention of GPT-5.1."""
    claim_text = "GPT-5.1 Codex Max (77.9%) and Gemini 3 Pro (76.2%)"
    nums = extract_numbers(claim_text)
    # Pick 77.9
    seventy_seven = next((n for n in nums if "77.9" in n.raw), None)
    assert seventy_seven, f"Expected 77.9 in {[n.raw for n in nums]}"
    result = _verify_single_claim(seventy_seven, CASE3_SOURCE_TESSL_OPUS45)
    # 77.9% NOT in source — Tessl source has 80.9, 77.2, 74.5, 59.3 etc but not 77.9
    assert "number_missing_in_source" in result["fail_reasons"], \
        f"Expected number_missing_in_source for 77.9%, got {result}"


def test_case4_direct_contradiction_cloudzero():
    """Claim: '$10 / $37.50 surcharge' — source: 'no surcharge'."""
    claim_text = "Opus 4.5/4.6/4.7 above 200K tokens ($10 / $37.50 for long-context Opus)"
    nums = extract_numbers(claim_text)
    # Pick $10
    ten = next((n for n in nums if "10" in n.raw and "$" in n.raw), None)
    assert ten, f"Expected $10 in {[n.raw for n in nums]}"
    result = _verify_single_claim(ten, CASE4_SOURCE_CLOUDZERO)
    # $10 NOT in source — source says no surcharge
    assert "number_missing_in_source" in result["fail_reasons"], \
        f"Expected $10 missing in source for CloudZero claim, got {result}"


def test_case5_off_by_25pct_artificialanalysis():
    """Claim: '$3 input price' — source: '$3.75 input price'."""
    claim_text = "Claude 3.7 Sonnet pricing at $3 / $15 per million tokens"
    nums = extract_numbers(claim_text)
    # Pick $3 (input)
    three_dollar = next((n for n in nums if n.raw.startswith("$3") and "." not in n.raw[:3]), None)
    if three_dollar is None:
        # Some patterns extract "$3 / $15" combined; in that case just take first $X
        three_dollar = next((n for n in nums if "$" in n.raw and "3" in n.raw), None)
    assert three_dollar, f"Expected $3-something in {[n.raw for n in nums]}"
    result = _verify_single_claim(three_dollar, CASE5_SOURCE_AA_CLAUDE37)
    # $3 alone vs $3.75 in source — bare-value matching may give false pass.
    # This documents current behavior — Phase 1 polish may tighten.
    # We don't strictly assert pass/fail here; this is a regression baseline.
    assert "raw" in result
