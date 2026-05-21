"""Unit tests for direction_classifier."""

from deerflow.community.factcheck.direction_classifier import (
    classify_direction,
    is_direction_conflict,
)


def test_up_words_english():
    assert classify_direction("The cost increased by 27%.") == "up"
    assert classify_direction("Prices rose 10%.") == "up"
    assert classify_direction("Higher usage observed.") == "up"


def test_down_words_english():
    assert classify_direction("Cost dropped ~8% for coding workloads.") == "down"
    assert classify_direction("Prices fell to $5.") == "down"
    assert classify_direction("Reduced spending overall.") == "down"


def test_up_words_chinese():
    assert classify_direction("成本上升了 27%。") == "up"
    assert classify_direction("价格增加了。") == "up"


def test_down_words_chinese():
    assert classify_direction("成本下降了 8%。") == "down"
    assert classify_direction("价格降低了。") == "down"


def test_neutral():
    assert classify_direction("The rate card remained unchanged.") == "neutral"
    assert classify_direction("Price held flat through Q3.") == "neutral"
    assert classify_direction("价格保持不变。") == "neutral"


def test_none():
    assert classify_direction("The new model was released on Tuesday.") == "none"
    assert classify_direction("发布了新版本。") == "none"


def test_mixed_takes_first():
    """When both directions appear, prefer the one closer to start."""
    # "increased" appears first → up
    text = "Costs increased significantly, though some workloads saw decreases."
    assert classify_direction(text) == "up"
    # "dropped" appears first → down
    text2 = "Spending dropped overall, even as some categories grew."
    assert classify_direction(text2) == "down"


def test_conflict_up_vs_down():
    assert is_direction_conflict("up", "down") is True
    assert is_direction_conflict("down", "up") is True


def test_no_conflict_same():
    assert is_direction_conflict("up", "up") is False
    assert is_direction_conflict("down", "down") is False


def test_no_conflict_with_none():
    """If either side has no direction word, no conflict."""
    assert is_direction_conflict("up", "none") is False
    assert is_direction_conflict("none", "down") is False
    assert is_direction_conflict("none", "none") is False


def test_no_conflict_with_neutral():
    """Neutral never conflicts with a direction (claim says stable,
    source elaborates a direction — could be sloppy but not contradictory)."""
    assert is_direction_conflict("up", "neutral") is False
    assert is_direction_conflict("neutral", "down") is False
