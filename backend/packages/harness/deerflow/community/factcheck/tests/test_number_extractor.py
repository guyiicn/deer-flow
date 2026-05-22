"""Unit tests for number_extractor.

Covers the regex patterns + extract_numbers + number_in_source logic.
"""

from __future__ import annotations

from deerflow.community.factcheck.number_extractor import (
    extract_numbers,
    number_in_source,
)


def test_extract_simple_currency():
    nums = extract_numbers("The price is $5 per million tokens.")
    raws = [n.raw for n in nums]
    # Expect to find "$5 per million" or similar
    assert any("$5" in r for r in raws), f"Expected to find $5 in {raws}"


def test_extract_percentage():
    nums = extract_numbers("Costs dropped ~8% for typical workloads.")
    raws = [n.raw for n in nums]
    assert any("8%" in r for r in raws), f"Expected to find 8% in {raws}"


def test_extract_range_percentage():
    nums = extract_numbers("Costs increased 12-27% on average.")
    raws = [n.raw for n in nums]
    # Range or individual numbers OK as long as 12 and 27 captured somehow
    flat = " ".join(raws)
    assert "12" in flat and "27" in flat, f"Expected to find 12 and 27 in {raws}"


def test_extract_token_count():
    nums = extract_numbers("Context window of 200K tokens.")
    raws = [n.raw for n in nums]
    assert any("200K" in r.replace(" ", "") or "200k" in r.lower().replace(" ", "") for r in raws), \
        f"Expected to find 200K in {raws}"


def test_extract_benchmark_score():
    nums = extract_numbers("Opus 4.5 hit 80.9% on SWE-bench Verified.")
    raws = [n.raw for n in nums]
    flat = " ".join(raws)
    assert "80.9" in flat, f"Expected 80.9 in {raws}"


def test_extract_price_ratio():
    nums = extract_numbers("GPT-5.5 priced at $5 / $30 per MTok.")
    raws = [n.raw for n in nums]
    flat = " ".join(raws)
    # Either captured as combined "$5 / $30" or individually
    assert "5" in flat and "30" in flat, f"Expected 5 and 30 in {raws}"


def test_extract_no_numbers():
    nums = extract_numbers("This sentence has no specific numbers.")
    # May find nothing, or may find "no" depending on greediness — main thing is no crash
    # Filter to only ones that contain a digit
    digit_nums = [n for n in nums if any(c.isdigit() for c in n.raw)]
    assert len(digit_nums) == 0, f"Did not expect digit-numbers but found {digit_nums}"


def test_window_includes_direction_words():
    text = "The new tokenizer caused costs to rise by 27% on average for production workloads."
    nums = extract_numbers(text, window_chars=50)
    # The 27% claim should have a window with "rise" in it
    found = [n for n in nums if "27" in n.raw]
    assert found, "Should find 27%"
    assert "rise" in found[0].window.lower(), \
        f"Window {found[0].window!r} should contain 'rise'"


def test_number_in_source_exact():
    claim = extract_numbers("Effective cost ~8% drop.")[0]
    source = "The new system delivered an 8% improvement in efficiency."
    assert number_in_source(claim, source), "8% should match 8%"


def test_number_in_source_missing():
    claim = extract_numbers("Pricing at $4 per million.")[0]
    source = "Pricing is $5 per million tokens (April 2026)."
    # $4 should NOT match a source that has $5
    assert not number_in_source(claim, source), \
        "$4 should NOT match source containing only $5"


def test_number_in_source_close_but_wrong():
    """PoC #4 case 5 — $3 claim vs $3.75 source.

    With Phase 1 strict matching: "$3" pattern uses word boundary that
    rejects matching inside "$3.75" → returns False (correctly flagged).
    """
    claim = extract_numbers("Pricing $3 input.")[0]
    source_with_375 = "Actual pricing is $3.75 input."
    assert number_in_source(claim, source_with_375) is False, \
        "$3 should NOT match $3.75 source under strict matching"


def test_number_in_source_dollar_4_not_in_5():
    """PoC #4 case 2 — $4 claim vs $5 source.

    Real failure observed in Phase 1 Day 1 smoke test: bare-value fallback
    matched digit 4 anywhere in source. Strict fix: $4 must not match
    a source whose only dollar value is $5 (even if "4" appears elsewhere
    in dates / IDs).
    """
    claim = extract_numbers("GPT-5.5 priced at $4 per MTok.")[0]
    source_with_5 = "Pricing is $5 per 1M tokens. Released April 23 2026 with 4x multiplier."
    # Source has "4x" but no "$4" — strict matching must return False
    assert number_in_source(claim, source_with_5) is False, \
        "$4 must NOT match a source containing only $5 (even with bare 4 elsewhere)"


def test_number_in_source_percentage_strict():
    """8% claim vs source with 80%, 18%, 12-27% — must not false-positive."""
    claim = extract_numbers("Cost dropped ~8%.")[0]
    source_no_8 = "Costs increased 12-27% on average. Some workloads saw 80% gains."
    assert number_in_source(claim, source_no_8) is False, \
        "8% must not match 80% or 12-27% or 18%"


def test_number_in_source_percentage_match():
    """8% claim vs source with explicit 8% — must match."""
    claim = extract_numbers("Cost dropped ~8%.")[0]
    source_with_8 = "The discount averaged 8% across all customers."
    assert number_in_source(claim, source_with_8) is True, \
        "8% should match source containing standalone 8%"


# ─── Day 5 Bug A regression: display-equivalent value variants ──────────────
def test_dollar_8_dot_00_matches_dollar_8_in_source():
    """Bug A from Day 5 sanity: claim '$8.00 per million' must match source
    that says '$8/million'. The trailing zeros are display, not semantic."""
    claim = extract_numbers("Pricing was $8.00 per million input tokens.")[0]
    source_with_int = "API pricing: $8/million input, $24/million output."
    assert number_in_source(claim, source_with_int) is True, \
        "$8.00 with /M unit must match source containing $8/million"


def test_dollar_22_dot_50_matches_22_50_per_million():
    """Sanity Day 5 second case: $22.50/MTok claim vs $22.50/million source."""
    claim = extract_numbers("Output: $22.50 per MTok.")[0]
    source = "For >200K prompts, output rates rise to $22.50/million tokens."
    assert number_in_source(claim, source) is True


def test_dollar_3_dot_00_matches_dollar_3():
    claim = extract_numbers("Input: $3.00 / MTok")[0]
    source = "$3/M input, $15/M output"
    assert number_in_source(claim, source) is True


def test_value_variants_still_blocks_3_vs_3_75():
    """Regression guard: the variant change must NOT regress PoC #4 case 5.
    Claim $3 (no decimal) must still NOT match source containing only $3.75."""
    claim = extract_numbers("Pricing $3 input.")[0]
    source = "Actual pricing is $3.75 input."
    assert number_in_source(claim, source) is False, \
        "$3 must NOT match $3.75 (word-boundary protection)"


def test_percent_8_dot_0_matches_8_percent():
    """Variants apply to percent suffix too: 8.0% matches 8%."""
    claim = extract_numbers("Cost dropped 8.0%.")[0]
    source = "The discount averaged 8% across all customers."
    assert number_in_source(claim, source) is True


def test_per_mtok_matches_per_million():
    """Unit equivalence: 'per MTok' should match 'per million' source."""
    claim = extract_numbers("Pricing $5 per MTok")[0]
    source_per_million = "API at $5 per million tokens"
    assert number_in_source(claim, source_per_million) is True
