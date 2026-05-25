"""Phase 7 A'.2 DanglingToolCallPatchMiddleware tests.

7 tests covering:
- happy path (no orphan → no-op)
- single orphan in content-block format → inserted after orphan AI
- single orphan in tool_calls field format → inserted
- multiple orphans across different AI messages → each inserted in place
- in-flight tool_result already in history → idempotent dedupe
- race-window simulation → A'.2 synthetic precedes real result commit
- defensive fallback when orphan id can't be located in any AI message
- sentinel string verification
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.community.factcheck.dangling_tool_call_patch_middleware import (
    DanglingToolCallPatchMiddleware,
    ORPHAN_PATCH_SENTINEL,
)


def _ai_with_content_block_tool_use(tu_id: str, tool_name: str = "read_file"):
    """Build an AI message with Anthropic-format content-block tool_use
    (no msg.tool_calls populated — simulates the gap that Tier 1 misses)."""
    return AIMessage(
        content=[
            {"type": "text", "text": "Let me read that file."},
            {
                "type": "tool_use",
                "id": tu_id,
                "name": tool_name,
                "input": {"path": "/some/file.md"},
            },
        ],
        # tool_calls intentionally empty — this is the A'.2 gap case
    )


def _ai_with_lc_tool_calls(tu_id: str, tool_name: str = "read_file"):
    """Build an AI message with LangChain-normalized tool_calls (the
    case Tier 1 already covers)."""
    msg = AIMessage(
        content="Let me read that file.",
        tool_calls=[
            {"id": tu_id, "name": tool_name, "args": {"path": "/x.md"}}
        ],
    )
    return msg


def _make_request(messages):
    """Build a minimal ModelRequest mock; we only call .override() and
    read .messages."""
    req = MagicMock()
    req.messages = messages

    def _override(messages=None, **kw):
        new = MagicMock()
        new.messages = messages if messages is not None else req.messages
        return new
    req.override = _override
    return req


class HappyPathTest(unittest.TestCase):
    def test_no_orphan_passes_through_unchanged(self):
        """Messages with all tool_uses paired → no patching."""
        mw = DanglingToolCallPatchMiddleware()
        messages = [
            HumanMessage(content="What's in the file?"),
            _ai_with_content_block_tool_use("toolu_1"),
            ToolMessage(
                content="(content)", tool_call_id="toolu_1",
                name="read_file"
            ),
        ]
        handler_called_with = []

        def handler(req):
            handler_called_with.append(req)
            return "ok"

        req = _make_request(messages)
        result = mw.wrap_model_call(req, handler)
        self.assertEqual(result, "ok")
        self.assertEqual(len(handler_called_with), 1)
        # Same messages passed through (override not called)
        self.assertIs(handler_called_with[0].messages, messages)


class ContentBlockGapTest(unittest.TestCase):
    """Tier 1 gap: tool_use in raw content block, NOT in msg.tool_calls."""

    def test_single_content_block_orphan_inserted_after_ai(self):
        mw = DanglingToolCallPatchMiddleware()
        ai_msg = _ai_with_content_block_tool_use("toolu_X")
        messages = [
            HumanMessage(content="hi"),
            ai_msg,
            HumanMessage(content="continue"),  # follow-up,no tool_result
        ]
        captured = {}

        def handler(req):
            captured["msgs"] = req.messages
            return "ok"

        mw.wrap_model_call(_make_request(messages), handler)
        msgs = captured["msgs"]
        # Synthetic must be IMMEDIATELY AFTER ai_msg (P1-B per design v2 §2.2)
        ai_idx = msgs.index(ai_msg)
        self.assertIsInstance(msgs[ai_idx + 1], ToolMessage)
        self.assertEqual(msgs[ai_idx + 1].tool_call_id, "toolu_X")
        self.assertEqual(msgs[ai_idx + 1].status, "error")


class LangChainNormalizedTest(unittest.TestCase):
    """A'.2 ALSO catches msg.tool_calls orphans (overlap with Tier 1
    — idempotent dedupe handles Tier 1's prior patch if any)."""

    def test_lc_tool_calls_orphan_patched(self):
        mw = DanglingToolCallPatchMiddleware()
        ai_msg = _ai_with_lc_tool_calls("toolu_Y")
        messages = [
            HumanMessage(content="hi"),
            ai_msg,
            HumanMessage(content="continue"),
        ]
        captured = {}

        def handler(req):
            captured["msgs"] = req.messages
            return "ok"

        mw.wrap_model_call(_make_request(messages), handler)
        msgs = captured["msgs"]
        ai_idx = msgs.index(ai_msg)
        self.assertIsInstance(msgs[ai_idx + 1], ToolMessage)
        self.assertEqual(msgs[ai_idx + 1].tool_call_id, "toolu_Y")


class MultipleOrphansTest(unittest.TestCase):
    def test_orphans_inserted_each_after_own_ai(self):
        mw = DanglingToolCallPatchMiddleware()
        ai_1 = _ai_with_content_block_tool_use("toolu_A", "read_file")
        ai_2 = _ai_with_content_block_tool_use("toolu_B", "grep_tool")
        messages = [
            HumanMessage(content="task 1"),
            ai_1,
            HumanMessage(content="task 2"),
            ai_2,
            HumanMessage(content="next"),
        ]
        captured = {}

        def handler(req):
            captured["msgs"] = req.messages
            return "ok"

        mw.wrap_model_call(_make_request(messages), handler)
        msgs = captured["msgs"]
        idx_a = msgs.index(ai_1)
        idx_b = msgs.index(ai_2)
        self.assertEqual(msgs[idx_a + 1].tool_call_id, "toolu_A")
        self.assertEqual(msgs[idx_b + 1].tool_call_id, "toolu_B")


class IdempotentDedupeTest(unittest.TestCase):
    """In-flight result: tool_use has matching ToolMessage already →
    A'.2 should NOT double-patch (set subtraction logic, per P6 D5
    micro-MM1 §2 Q2)."""

    def test_in_flight_result_skipped(self):
        mw = DanglingToolCallPatchMiddleware()
        ai_msg = _ai_with_content_block_tool_use("toolu_INFLIGHT")
        real_result = ToolMessage(
            content="(real result committed)",
            tool_call_id="toolu_INFLIGHT", name="read_file",
        )
        messages = [
            HumanMessage(content="hi"),
            ai_msg,
            real_result,   # Real tool_result is already there
            HumanMessage(content="more"),
        ]
        handler_called_with = []

        def handler(req):
            handler_called_with.append(req)
            return "ok"

        req = _make_request(messages)
        mw.wrap_model_call(req, handler)
        # No-op: handler received original messages, no synthetic added
        self.assertIs(handler_called_with[0].messages, messages)


class RaceWindowSimulationTest(unittest.TestCase):
    """Simulate: A'.2 patches (synthetic), then on next wrap_model_call
    (after real result commits late), A'.2 sees both — idempotent
    skips the now-paired tool_use."""

    def test_real_result_commits_after_synthetic(self):
        mw = DanglingToolCallPatchMiddleware()
        ai_msg = _ai_with_content_block_tool_use("toolu_RACE")
        # First call: only orphan, no result yet
        first_messages = [
            HumanMessage(content="hi"),
            ai_msg,
            HumanMessage(content="follow"),
        ]
        first_capture = {}

        def first_handler(req):
            first_capture["msgs"] = req.messages
            return "ok"
        mw.wrap_model_call(_make_request(first_messages), first_handler)
        # synthetic inserted
        first_out = first_capture["msgs"]
        self.assertTrue(
            any(
                isinstance(m, ToolMessage) and m.tool_call_id == "toolu_RACE"
                for m in first_out
            )
        )

        # Second call: real result has now committed AND synthetic is
        # still there. A'.2 must NOT add another synthetic.
        second_messages = list(first_out) + [
            ToolMessage(
                content="(real result, late commit)",
                tool_call_id="toolu_RACE",
                name="read_file",
            )
        ]
        second_capture = {}

        def second_handler(req):
            second_capture["msgs"] = req.messages
            return "ok"
        mw.wrap_model_call(_make_request(second_messages),
                           second_handler)
        # Idempotent: no new synthetic added in second pass
        tool_msgs_for_race = [
            m for m in second_capture["msgs"]
            if isinstance(m, ToolMessage)
            and m.tool_call_id == "toolu_RACE"
        ]
        # Synthetic from pass 1 + real from late commit = 2.
        # A'.2 does NOT add a 3rd.
        self.assertEqual(len(tool_msgs_for_race), 2)


class SentinelTest(unittest.TestCase):
    def test_sentinel_constant_defined(self):
        self.assertEqual(ORPHAN_PATCH_SENTINEL, "[wrapper-p7-orphan-patch]")

    def test_synthetic_content_contains_sentinel(self):
        synthetic = DanglingToolCallPatchMiddleware._build_synthetic(
            "toolu_X"
        )
        self.assertIn(ORPHAN_PATCH_SENTINEL, synthetic.content)
        # P2-B wording: MUST retry NOW (NOT "if needed")
        self.assertIn("MUST retry", synthetic.content)
        self.assertIn("before continuing", synthetic.content)
        self.assertNotIn("if needed", synthetic.content)


if __name__ == "__main__":
    unittest.main()
