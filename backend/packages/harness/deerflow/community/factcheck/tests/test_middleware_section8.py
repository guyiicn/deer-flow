"""Phase 3 G1 Section8EnforcementMiddleware tests.

Wrapper-side §8 enforcement via system message injection.
NOT synthetic ToolMessage (would pollute conversation with fabricated
fact-checker verdict — Phase 3 design v2 §3.2.3).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from deerflow.community.factcheck.middleware import (
    FACT_CHECKER_SONNET_SUBAGENT,
    G1_EXEMPT_SECTION_IDS,
    SECTION_8_ENFORCEMENT_SENTINEL,
    Section8EnforcementMiddleware,
)


# ─── Helpers ───────────────────────────────────────────────────────────────
def _ai_write_file(path, content, tc_id="wf_1"):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "write_file",
            "id": tc_id,
            "args": {"path": path, "content": content},
        }],
    )


def _tool_msg(tc_id="wf_1", content="OK"):
    return ToolMessage(content=content, tool_call_id=tc_id, name="write_file")


def _ai_fact_check(section_id, tc_id="fc_1", subagent=FACT_CHECKER_SONNET_SUBAGENT):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "task",
            "id": tc_id,
            "args": {
                "subagent_type": subagent,
                "prompt": f"Verify section. section_id={section_id}",
            },
        }],
    )


def _request_with(messages):
    req = MagicMock()
    req.messages = messages
    def _override(messages=None):
        new = MagicMock()
        new.messages = messages if messages is not None else req.messages
        new.override = req.override
        return new
    req.override = _override
    return req


# ─── _extract_write_file_calls ─────────────────────────────────────────────
def test_extract_no_writes():
    assert Section8EnforcementMiddleware._extract_write_file_calls([]) == []


def test_extract_single_write_with_headings():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md",
                       "## Section A\nfoo\n\n## Section B\nbar"),
    ]
    result = Section8EnforcementMiddleware._extract_write_file_calls(msgs)
    assert len(result) == 1
    idx, slugs = result[0]
    assert sorted(slugs) == ["section-a", "section-b"]


def test_extract_filters_exempt_sections():
    """Sources / References must be filtered out."""
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md",
                       "## Pricing\nbody\n\n## Sources\n- url1\n\n## References\nrefs"),
    ]
    result = Section8EnforcementMiddleware._extract_write_file_calls(msgs)
    _, slugs = result[0]
    assert slugs == ["pricing"]


def test_extract_filters_non_md():
    msgs = [_ai_write_file("/mnt/user-data/outputs/data.json", "## A")]
    assert Section8EnforcementMiddleware._extract_write_file_calls(msgs) == []


def test_extract_filters_non_outputs_path():
    msgs = [_ai_write_file("/mnt/user-data/workspace/scratch.md", "## A")]
    assert Section8EnforcementMiddleware._extract_write_file_calls(msgs) == []


def test_extract_ignores_non_ai_messages():
    msgs = [HumanMessage(content="write something")]
    assert Section8EnforcementMiddleware._extract_write_file_calls(msgs) == []


# ─── _extract_fact_checked_section_ids ─────────────────────────────────────
def test_fact_checked_empty():
    assert Section8EnforcementMiddleware._extract_fact_checked_section_ids([], 0) == set()


def test_fact_checked_from_prompt_kv():
    msgs = [_ai_fact_check("section-a", "fc1")]
    verified = Section8EnforcementMiddleware._extract_fact_checked_section_ids(msgs, 0)
    assert verified == {"section-a"}


def test_fact_checked_from_explicit_section_id_arg():
    msgs = [AIMessage(content="", tool_calls=[{
        "name": "task", "id": "fc1",
        "args": {"subagent_type": FACT_CHECKER_SONNET_SUBAGENT,
                 "section_id": "section-b",
                 "prompt": "verify"},
    }])]
    verified = Section8EnforcementMiddleware._extract_fact_checked_section_ids(msgs, 0)
    assert verified == {"section-b"}


def test_fact_checked_ignores_non_factchecker_subagent():
    msgs = [AIMessage(content="", tool_calls=[{
        "name": "task", "id": "fc1",
        "args": {"subagent_type": "general-purpose",
                 "prompt": "verify section_id=section-a"},
    }])]
    verified = Section8EnforcementMiddleware._extract_fact_checked_section_ids(msgs, 0)
    assert verified == set()


def test_fact_checked_multiple_sections():
    msgs = [
        _ai_fact_check("section-a", "fc1"),
        HumanMessage(content="continue"),
        _ai_fact_check("section-b", "fc2"),
    ]
    verified = Section8EnforcementMiddleware._extract_fact_checked_section_ids(msgs, 0)
    assert verified == {"section-a", "section-b"}


# ─── _find_orphan_sections ─────────────────────────────────────────────────
def test_orphans_when_all_verified():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md", "## A\nfoo"),
        _tool_msg("wf_1"),
        _ai_fact_check("a", "fc1"),
    ]
    orphans = Section8EnforcementMiddleware._find_orphan_sections(msgs)
    assert orphans == []


def test_orphans_detected_when_unverified():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md",
                       "## A\nfoo\n\n## B\nbar"),
        _tool_msg("wf_1"),
        _ai_fact_check("a", "fc1"),
        # B written but never fact-checked
    ]
    orphans = Section8EnforcementMiddleware._find_orphan_sections(msgs)
    assert orphans == ["b"]


def test_orphans_includes_all_when_none_verified():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md",
                       "## A\nfoo\n\n## B\nbar\n\n## C\nbaz"),
        _tool_msg("wf_1"),
    ]
    orphans = Section8EnforcementMiddleware._find_orphan_sections(msgs)
    assert orphans == ["a", "b", "c"]


def test_orphans_with_cjk_section_slugs():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md", "## 价格分析\nfoo"),
        _tool_msg("wf_1"),
    ]
    orphans = Section8EnforcementMiddleware._find_orphan_sections(msgs)
    assert orphans == ["价格分析"]


# ─── _already_nagged_recently ──────────────────────────────────────────────
def test_nag_loop_guard_no_prior_nag():
    msgs = [HumanMessage(content="hi"), AIMessage(content="ok")]
    assert Section8EnforcementMiddleware._already_nagged_recently(msgs) is False


def test_nag_loop_guard_detects_sentinel_in_recent():
    msgs = [
        HumanMessage(content="hi"),
        SystemMessage(content=f"{SECTION_8_ENFORCEMENT_SENTINEL} call task for [a]"),
        AIMessage(content="OK calling now"),
    ]
    assert Section8EnforcementMiddleware._already_nagged_recently(msgs) is True


def test_nag_loop_guard_ignores_old_sentinel():
    """Sentinel beyond NAG_LOOKBACK (4) messages back should not block."""
    msgs = [
        SystemMessage(content=f"{SECTION_8_ENFORCEMENT_SENTINEL} old nag"),
        HumanMessage(content="m1"),
        AIMessage(content="r1"),
        HumanMessage(content="m2"),
        AIMessage(content="r2"),
        HumanMessage(content="m3"),
    ]
    # NAG_LOOKBACK=4 → only last 4 msgs checked; sentinel at index 0 is out of range
    assert Section8EnforcementMiddleware._already_nagged_recently(msgs) is False


# ─── wrap_model_call inject ────────────────────────────────────────────────
def test_wrap_model_call_no_orphans_passthrough():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md", "## A\nfoo"),
        _tool_msg("wf_1"),
        _ai_fact_check("a", "fc1"),
    ]
    req = _request_with(msgs)
    handler = MagicMock(return_value="result")
    mw = Section8EnforcementMiddleware()
    result = mw.wrap_model_call(req, handler)
    handler.assert_called_once()
    # No nag injected — messages unchanged
    assert len(handler.call_args[0][0].messages) == len(msgs)
    assert result == "result"


def test_wrap_model_call_injects_system_message_on_orphan():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md",
                       "## A\nfoo\n\n## B\nbar"),
        _tool_msg("wf_1"),
        # No fact-check for either
    ]
    req = _request_with(msgs)
    handler = MagicMock(return_value="result")
    mw = Section8EnforcementMiddleware()
    mw.wrap_model_call(req, handler)
    handler.assert_called_once()
    received = handler.call_args[0][0]
    # Prepended a SystemMessage at front
    assert len(received.messages) == len(msgs) + 1
    first = received.messages[0]
    assert isinstance(first, SystemMessage)
    assert SECTION_8_ENFORCEMENT_SENTINEL in first.content
    assert "a" in first.content and "b" in first.content


def test_wrap_model_call_does_not_inject_synthetic_tool_message():
    """v2 P0-1 requirement: NEVER inject ToolMessage (would pollute
    conversation with fake fact-checker verdict)."""
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md", "## A\nfoo"),
        _tool_msg("wf_1"),
    ]
    req = _request_with(msgs)
    handler = MagicMock(return_value="result")
    mw = Section8EnforcementMiddleware()
    mw.wrap_model_call(req, handler)
    received = handler.call_args[0][0]
    # NO ToolMessage in injected output
    for m in received.messages:
        assert not isinstance(m, ToolMessage) or m.tool_call_id in {"wf_1"}, \
            "Synthetic ToolMessage injection forbidden — must use SystemMessage"


def test_wrap_model_call_respects_nag_loop_guard():
    """If sentinel appears in last 4 messages, no new inject."""
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md", "## A\nfoo"),
        _tool_msg("wf_1"),
        SystemMessage(content=f"{SECTION_8_ENFORCEMENT_SENTINEL} prior nag"),
        HumanMessage(content="continue"),
    ]
    req = _request_with(msgs)
    handler = MagicMock(return_value="result")
    mw = Section8EnforcementMiddleware()
    mw.wrap_model_call(req, handler)
    received = handler.call_args[0][0]
    # Messages length unchanged — no new nag inject
    assert len(received.messages) == len(msgs)


# ─── Async path ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_awrap_model_call_injects_async():
    msgs = [
        _ai_write_file("/mnt/user-data/outputs/x.md",
                       "## A\nfoo"),
        _tool_msg("wf_1"),
    ]
    req = _request_with(msgs)
    captured = {}
    async def handler(r):
        captured["req"] = r
        return "async-result"
    mw = Section8EnforcementMiddleware()
    result = await mw.awrap_model_call(req, handler)
    assert result == "async-result"
    received = captured["req"]
    assert len(received.messages) == len(msgs) + 1
    assert isinstance(received.messages[0], SystemMessage)


# ─── Constants stability ──────────────────────────────────────────────────
def test_sentinel_string_stable():
    """Wrapper greps for this — must not change without coordination."""
    assert SECTION_8_ENFORCEMENT_SENTINEL == "[wrapper-p3-g1]"


def test_exempt_sections_set():
    """Wrapper and middleware must agree on exempt list."""
    assert "sources" in G1_EXEMPT_SECTION_IDS
    assert "references" in G1_EXEMPT_SECTION_IDS
    assert "bibliography" in G1_EXEMPT_SECTION_IDS
    # frozenset so callers can't mutate
    assert isinstance(G1_EXEMPT_SECTION_IDS, frozenset)
