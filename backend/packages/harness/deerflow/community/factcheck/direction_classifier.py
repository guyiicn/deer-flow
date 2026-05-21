"""Phase 1 fact-check: classify directional language near numerical claims.

Used by verify_numbers Level A (subject-matched direction conflict detection):
given a window of text around a number, classify the direction word:
  - up    : "rises", "increase", "+", "上升", ...
  - down  : "drops", "decrease", "-", "下降", ...
  - neutral: "unchanged", "持平", ...
  - none  : no direction word in window

If claim says "drops 8%" and source says "rises 8%" at the same number,
that's a Level A direction conflict.
"""

from __future__ import annotations

import re
from typing import Literal

Direction = Literal["up", "down", "neutral", "none"]

# Direction word vocab — case-insensitive substring match
_UP_WORDS = [
    # English
    "rise", "rises", "rising", "rose", "increase", "increased", "increasing",
    "grew", "grow", "growth", "growing", "higher", "more", "raise", "raised",
    "up by", "go up", "went up", "jump", "jumped", "soar", "soared",
    "expansion", "expand", "expanded",
    # Chinese
    "上升", "增加", "上涨", "上调", "提高", "更多", "涨", "增长", "提升",
]

_DOWN_WORDS = [
    "drop", "drops", "dropped", "dropping",
    "fall", "fell", "falls", "falling",
    "decline", "declined", "declines", "declining",
    "decrease", "decreased", "decreases", "decreasing",
    "reduce", "reduced", "reduces", "reducing",
    "lower", "fewer", "less", "down by", "go down", "went down",
    "shrink", "shrank", "shrunk", "shrinking",
    "cut", "cuts", "trim", "trimmed",
    # Chinese
    "下降", "减少", "降低", "下跌", "更少", "跌", "下调", "削减",
]

_NEUTRAL_WORDS = [
    "unchanged", "same", "stable", "preserved", "held flat", "no change",
    "kept", "maintain", "maintained", "持平", "不变", "稳定", "保持",
]


def _contains_any(text_lower: str, words: list[str]) -> bool:
    return any(w in text_lower for w in words)


def classify_direction(window_text: str) -> Direction:
    """Determine the dominant direction word in window_text.

    If both up and down words appear (e.g., "increased then decreased"),
    return whichever appears closer to the start (proxy for primary verb).
    If neutral words present and no up/down, return neutral.
    """
    lower = window_text.lower()
    has_up = _contains_any(lower, _UP_WORDS)
    has_down = _contains_any(lower, _DOWN_WORDS)
    has_neutral = _contains_any(lower, _NEUTRAL_WORDS)

    if has_up and has_down:
        # Find first occurrence position of any up vs any down word
        up_pos = min(
            (lower.find(w) for w in _UP_WORDS if w in lower), default=-1
        )
        down_pos = min(
            (lower.find(w) for w in _DOWN_WORDS if w in lower), default=-1
        )
        if up_pos == -1:
            return "down"
        if down_pos == -1:
            return "up"
        return "up" if up_pos < down_pos else "down"

    if has_up:
        return "up"
    if has_down:
        return "down"
    if has_neutral:
        return "neutral"
    return "none"


def is_direction_conflict(claim_direction: Direction, source_direction: Direction) -> bool:
    """Whether the two directions are in semantic conflict.

    Conflicts:
        up vs down → conflict
        down vs up → conflict
        neutral vs up/down → NOT conflict (claim says "stable", source elaborates direction)
        none → never conflict (no direction word in claim or source)
    """
    if claim_direction == "none" or source_direction == "none":
        return False
    if claim_direction == "neutral" or source_direction == "neutral":
        return False
    return claim_direction != source_direction
