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
    """Check if the claim's number + unit appears in source_text.

    Algorithm (Phase 1):
        1. Build a regex from claim.raw that requires WORD BOUNDARIES on both sides
           of the number, with the currency/unit affix attached.
           - "$4"   → r"\\$4(?!\\d|\\.\\d)"      (no extending digits or decimal)
           - "8%"   → r"\\b8\\s*%"
           - "12-27%" → r"\\b12\\s*[-–~]\\s*27\\s*%"
        2. If word-boundary regex finds match → True.
        3. NO bare-digit fallback — that produced false positives in Phase 1
           Day 1 smoke test ($4 matched bare digit 4 in source containing $5).

    For PoC #4 case 5 ($3 vs $3.75): "$3" pattern is r"\\$3(?!\\d|\\.\\d)" — refuses
    to match inside "$3.75", correctly flags as missing.
    """
    val_match = re.search(r"(\d+(?:\.\d+)?(?:\s*[-–~]\s*\d+(?:\.\d+)?)?)", number_claim.raw)
    if val_match is None:
        return False
    value_str = val_match.group(1)
    # Detect prefix (currency) and suffix (unit) in claim
    raw = number_claim.raw
    prefix = ""
    if raw.startswith("$") or raw.startswith("US$"):
        prefix = r"US?\$"
    elif raw.startswith("¥"):
        prefix = "¥"
    elif raw.startswith("~") or raw.lower().startswith("approximately") or raw.lower().startswith("around"):
        prefix = ""  # ignore approximation prefix for matching

    # Find suffix (token after the numeric value in the claim)
    after_value = raw[val_match.end():].strip()
    suffix_pattern = ""
    if after_value.startswith("%"):
        suffix_pattern = r"\s*%"
    elif re.match(r"^\s*(?:K|M|B)(?:\s*(?:tokens?|ctx|context|window)?)?\b", after_value, re.IGNORECASE):
        # Token count like "200K tokens"
        m = re.match(r"^\s*([KMB])", after_value, re.IGNORECASE)
        if m:
            suffix_pattern = r"\s*" + m.group(1) + r"\b"
    elif re.match(r"^\s*[xX×]", after_value):
        suffix_pattern = r"\s*[xX×]"
    elif re.match(r"^\s*(?:/|per)\s*(?:M|MTok|million|month|year)", after_value, re.IGNORECASE):
        # Per-M pricing — matching the value with /M or per-M context
        suffix_pattern = r"\s*(?:/\s*M|per\s+M)"

    # Build the value regex: escape special chars in value_str, allow flex on hyphen/spaces
    value_re = re.escape(value_str).replace(r"-", r"\s*[-–~]\s*").replace(r"\ ", r"\s*")

    pattern_str = prefix + value_re + suffix_pattern
    # Word boundary on the right unless suffix already consumed it
    if not suffix_pattern:
        # Prevent matching as a prefix of a longer number (e.g., "3" inside "3.75")
        pattern_str += r"(?!\d|\.\d)"

    try:
        return bool(re.search(pattern_str, source_text, re.IGNORECASE))
    except re.error:
        # If regex compilation fails for weird input, fall back to normalized substring
        return number_claim.normalized in _normalize(source_text)
