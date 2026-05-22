"""Unit tests for Phase 2 P0-1 Tier 2 OrphanRetryFailFastMiddleware.

The middleware is a safety net — runs AFTER DanglingToolCallMiddleware.
If orphans are present (rare), raises OrphanToolUseAfterPatchError so
the wrapper can retry with a fresh thread instead of letting the
request hit Anthropic and 400.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.community.factcheck.middleware import (
    ORPHAN_AFTER_PATCH_SENTINEL,
    OrphanRetryFailFastMiddleware,
    OrphanToolUseAfterPatchError,
)


def _ai_with_tool_calls(tool_calls):
    return AIMessage(content="", tool_calls=tool_calls)


def _tool_msg(tool_call_id, name="test_tool"):
    return ToolMessage(content="result", tool_call_id=tool_call_id, name=name)


def _tc(name="bash", tc_id="call_1"):
    return {"name": name, "id": tc_id, "args": {}}


# ─── _collect_orphans ───────────────────────────────────────────────────────
def test_empty_messages_no_orphans():
    assert OrphanRetryFailFastMiddleware._collect_orphans([]) == []


def test_no_ai_no_orphans():
    msgs = [HumanMessage(content="hi")]
    assert OrphanRetryFailFastMiddleware._collect_orphans(msgs) == []


def test_ai_without_tool_calls_no_orphans():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert OrphanRetryFailFastMiddleware._collect_orphans(msgs) == []


def test_all_paired_no_orphans():
    msgs = [
        _ai_with_tool_calls([_tc("bash", "call_1"), _tc("read", "call_2")]),
        _tool_msg("call_1"),
        _tool_msg("call_2"),
    ]
    assert OrphanRetryFailFastMiddleware._collect_orphans(msgs) == []


def test_single_orphan_detected():
    msgs = [
        _ai_with_tool_calls([_tc("bash", "call_1")]),
        # no ToolMessage for call_1 → orphan
    ]
    assert OrphanRetryFailFastMiddleware._collect_orphans(msgs) == ["call_1"]


def test_partial_orphan_detected():
    msgs = [
        _ai_with_tool_calls([_tc("bash", "call_1"), _tc("read", "call_2")]),
        _tool_msg("call_1"),
        # call_2 has no ToolMessage → orphan
    ]
    assert OrphanRetryFailFastMiddleware._collect_orphans(msgs) == ["call_2"]


def test_multiple_ai_with_orphans():
    msgs = [
        _ai_with_tool_calls([_tc("bash", "call_1")]),
        _tool_msg("call_1"),
        _ai_with_tool_calls([_tc("read", "call_2"), _tc("write", "call_3")]),
        # call_2 + call_3 both orphan
    ]
    assert sorted(OrphanRetryFailFastMiddleware._collect_orphans(msgs)) == ["call_2", "call_3"]


def test_orphan_handles_object_style_tool_call():
    """Some tool_call entries are objects, not dicts."""
    tc_obj = MagicMock()
    tc_obj.id = "call_obj"
    ai = AIMessage(content="", tool_calls=[])
    ai.tool_calls = [tc_obj]  # bypass type check, raw form
    assert OrphanRetryFailFastMiddleware._collect_orphans([ai]) == ["call_obj"]


# ─── wrap_model_call ────────────────────────────────────────────────────────
def test_wrap_model_call_passthrough_when_paired():
    mw = OrphanRetryFailFastMiddleware()
    request = MagicMock()
    request.messages = [
        _ai_with_tool_calls([_tc("bash", "call_1")]),
        _tool_msg("call_1"),
    ]
    handler = MagicMock(return_value="handler-result")
    result = mw.wrap_model_call(request, handler)
    assert result == "handler-result"
    handler.assert_called_once_with(request)


def test_wrap_model_call_raises_on_orphan():
    mw = OrphanRetryFailFastMiddleware()
    request = MagicMock()
    request.messages = [
        _ai_with_tool_calls([_tc("bash", "call_orphan")]),
        # no tool result
    ]
    handler = MagicMock()
    with pytest.raises(OrphanToolUseAfterPatchError) as exc:
        mw.wrap_model_call(request, handler)
    assert "call_orphan" in str(exc.value)
    assert ORPHAN_AFTER_PATCH_SENTINEL in str(exc.value)
    handler.assert_not_called()  # request must NOT have hit the LLM


# ─── awrap_model_call (async path) ─────────────────────────────────────────
@pytest.mark.asyncio
async def test_awrap_model_call_passthrough_when_paired():
    mw = OrphanRetryFailFastMiddleware()
    request = MagicMock()
    request.messages = [
        _ai_with_tool_calls([_tc("bash", "call_1")]),
        _tool_msg("call_1"),
    ]

    async def handler(req):
        return "async-handler-result"

    result = await mw.awrap_model_call(request, handler)
    assert result == "async-handler-result"


@pytest.mark.asyncio
async def test_awrap_model_call_raises_on_orphan():
    mw = OrphanRetryFailFastMiddleware()
    request = MagicMock()
    request.messages = [_ai_with_tool_calls([_tc("bash", "call_orphan_async")])]

    async def handler(req):
        return "should not reach"

    with pytest.raises(OrphanToolUseAfterPatchError) as exc:
        await mw.awrap_model_call(request, handler)
    assert "call_orphan_async" in str(exc.value)


# ─── Error class contract ──────────────────────────────────────────────────
def test_error_carries_orphan_ids():
    err = OrphanToolUseAfterPatchError(["a", "b"])
    assert err.orphan_tool_call_ids == ["a", "b"]
    assert ORPHAN_AFTER_PATCH_SENTINEL in str(err)
    assert "a" in str(err) and "b" in str(err)


def test_sentinel_string_is_stable():
    """The wrapper greps for this sentinel — don't change it without
    coordinating with the wrapper side."""
    assert ORPHAN_AFTER_PATCH_SENTINEL == "[p0-1-tier2-orphan-after-patch]"
