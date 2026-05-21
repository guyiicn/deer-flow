"""Phase 1 fact-check: extract specific numerical claims from text.

Extracts (number + unit) tokens that the fact-check pipeline should verify
against the cited source. Units include: currency ($/¥), percentages,
token counts (K/M/B), price-per-MTok, multipliers.

Output: list of `NumberClaim` records with the raw substring, normalized
value, unit, and a ±50-char window around the claim (used by Level A
direction analysis).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Regex patterns — order matters (more specific first to avoid greedy mis-matches)
_PATTERNS = [
    # Price per million tokens with USD/CNY:  $5/$30 per MTok, $0.15/M
    re.compile(
        r"(?P<full>\$\d+(?:\.\d+)?\s*/\s*\$?\d+(?:\.\d+)?\s*(?:per\s+(?:M|million)\s*(?:tok|tokens)?|/\s*M)?)",
        re.IGNORECASE,
    ),
    # Currency with explicit unit: $5 per MTok, $0.60/MTok, ¥49/month
    re.compile(
        r"(?P<full>(?:\$|¥|US\$|CNY\s+)\d+(?:\.\d+)?\s*(?:/\s*(?:M|MTok|million|month|year|day|user|seat)|per\s+(?:M|MTok|million|month|year|day|user|seat))?)",
        re.IGNORECASE,
    ),
    # Standalone currency: $5, ¥49, US$200
    re.compile(r"(?P<full>(?:\$|¥|US\$)\s?\d+(?:[,\.]\d+)*)"),
    # Percentage: 8%, 12-27%, ~30%
    re.compile(r"(?P<full>(?:~|approximately\s+|around\s+|about\s+)?\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?\s*%)"),
    # Token count with unit: 200K, 1M tokens, 128K ctx
    re.compile(
        r"(?P<full>\d+(?:\.\d+)?\s*[KMB]\s*(?:tokens?|ctx|context|window)?)",
        re.IGNORECASE,
    ),
    # Per-million pricing without dollar sign: "0.028 per million", "$0.15 per M"
    re.compile(
        r"(?P<full>\d+(?:\.\d+)?\s*(?:per\s+(?:M|MTok|million)|/\s*M|/\s*million))",
        re.IGNORECASE,
    ),
    # Multiplier: 17x cheaper, 2x faster, 10× larger
    re.compile(r"(?P<full>\d+(?:\.\d+)?\s*[xX×]\s*(?:cheaper|faster|larger|smaller|more|less|increase|decrease)?)"),
    # Benchmark score with %: 80.9% SWE-bench, 67% improvement
    re.compile(r"(?P<full>\d+\.\d+\s*%\s+(?:on\s+)?(?:SWE-bench|HumanEval|MMLU|GSM8K|HellaSwag|ARC|TruthfulQA))", re.IGNORECASE),
]


@dataclass
class NumberClaim:
    """A specific numerical claim extracted from a sentence."""
    raw: str            # the matched substring as it appears in text
    start: int          # char offset in source text
    end: int            # char offset in source text
    window: str         # ±50 chars around the claim (for Level A direction check)
    normalized: str     # lowercased + whitespace-collapsed for comparison

    def __repr__(self) -> str:
        return f"NumberClaim(raw={self.raw!r}, normalized={self.normalized!r})"


def _normalize(s: str) -> str:
    """Collapse whitespace + lowercase for fuzzy comparison."""
    return re.sub(r"\s+", "", s.lower())


def extract_numbers(text: str, window_chars: int = 50) -> list[NumberClaim]:
    """Extract all (number + unit) claims from text.

    Args:
        text: input text to scan
        window_chars: ±N chars of context for direction analysis

    Returns:
        list of NumberClaim, sorted by position, deduplicated by (start, end)
    """
    found: list[NumberClaim] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            span = match.span("full") if "full" in match.groupdict() else match.span()
            if span in seen_spans:
                continue
            # Drop spans that are subsumed by an already-found longer span
            subsumed = False
            for existing in seen_spans:
                if existing[0] <= span[0] and existing[1] >= span[1]:
                    subsumed = True
                    break
            if subsumed:
                continue
            seen_spans.add(span)
            raw = text[span[0]:span[1]].strip()
            wstart = max(0, span[0] - window_chars)
            wend = min(len(text), span[1] + window_chars)
            window = text[wstart:wend]
            found.append(NumberClaim(
                raw=raw,
                start=span[0],
                end=span[1],
                window=window,
                normalized=_normalize(raw),
            ))
    found.sort(key=lambda c: c.start)
    return found


def number_in_source(number_claim: NumberClaim, source_text: str, tolerance: float = 0.0) -> bool:
    """Check if the number from claim appears in source_text.

    By default exact-match on normalized form. With tolerance>0, allow numeric
    fuzzy match (e.g., $3 vs $3.05 with tolerance=0.05 ≈ 1.7% diff).

    For Phase 1: tolerance=0 (strict). PoC #4 case 5 showed $3 vs $3.75
    must be flagged as mismatch, not tolerated.
    """
    norm_source = _normalize(source_text)
    if number_claim.normalized in norm_source:
        return True
    # Extract bare numeric value (first number in the claim) and search by value
    val_match = re.search(r"\d+(?:\.\d+)?", number_claim.raw)
    if val_match:
        val_str = val_match.group()
        # Look for that bare value anywhere in source (loose check)
        if val_str in source_text:
            return True
    return False
