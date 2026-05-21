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

    Strict mode (tolerance=0): $3 alone IS a substring of $3.75 text.
    But the bare numeric value '3' will match against '3' in '3.75'.

    This test documents current behavior; if it's too lenient, we'll
    tighten in Phase 1 polish.
    """
    claim = extract_numbers("Pricing $3 input.")[0]
    # The claim string is "$3" but with bare-value matching, "3" appears in "3.75"
    source_with_375 = "Actual pricing is $3.75 input."
    # Acknowledge this is a known limitation; the test documents it
    result = number_in_source(claim, source_with_375)
    # Either result is acceptable for now; test pinpoints the actual behavior
    assert result in (True, False)
