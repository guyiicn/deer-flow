"""Phase 2 P1-1 EscalationBudgetEnforcementMiddleware tests.

Budget enforcement at tool-call boundary. No ThreadState extension needed
— count is derived live from message history (spike result).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.community.factcheck.middleware import (
    DEFAULT_ESCALATION_BUDGET,
    ESCALATION_BUDGET_WARNING_SENTINEL,
    FACT_CHECKER_GPT_SUBAGENT,
    FACT_CHECKER_SONNET_SUBAGENT,
    EscalationBudgetEnforcementMiddleware,
)


def _ai_with_gpt_task(tc_id="call_gpt_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "task", "id": tc_id,
                     "args": {"subagent_type": FACT_CHECKER_GPT_SUBAGENT,
                              "prompt": "verify section X"}}],
    )


def _ai_with_sonnet_task(tc_id="call_sonnet_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "task", "id": tc_id,
                     "args": {"subagent_type": FACT_CHECKER_SONNET_SUBAGENT,
                              "prompt": "verify section X"}}],
    )


def _ai_with_other_tool(tc_id="call_other_1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "id": tc_id, "args": {"query": "..."}}],
    )


def _tool_request(tool_call, state):
    """Mock ToolCallRequest with state."""
    req = MagicMock()
    req.tool_call = tool_call
    req.state = state
    # override() should produce a new mock with tool_call replaced
    def _override(tool_call=None):
        new_req = MagicMock()
        new_req.tool_call = tool_call if tool_call is not None else req.tool_call
        new_req.state = state
        new_req.override = req.override
        return new_req
    req.override = _override
    return req


# ─── _is_gpt_escalation_call ────────────────────────────────────────────────
def test_detects_gpt_task_call():
    tc = {"name": "task", "args": {"subagent_type": FACT_CHECKER_GPT_SUBAGENT, "prompt": "x"}}
    assert EscalationBudgetEnforcementMiddleware._is_gpt_escalation_call(tc) is True


def test_rejects_sonnet_task_call():
    tc = {"name": "task", "args": {"subagent_type": FACT_CHECKER_SONNET_SUBAGENT}}
    assert EscalationBudgetEnforcementMiddleware._is_gpt_escalation_call(tc) is False


def test_rejects_non_task_tool():
    tc = {"name": "web_search", "args": {"query": "x"}}
    assert EscalationBudgetEnforcementMiddleware._is_gpt_escalation_call(tc) is False


def test_rejects_task_without_subagent_type():
    tc = {"name": "task", "args": {"prompt": "x"}}
    assert EscalationBudgetEnforcementMiddleware._is_gpt_escalation_call(tc) is False


def test_rejects_task_with_non_dict_args():
    tc = {"name": "task", "args": "not a dict"}
    assert EscalationBudgetEnforcementMiddleware._is_gpt_escalation_call(tc) is False


# ─── _count_gpt_calls_in_history ───────────────────────────────────────────
def test_count_empty_history():
    assert EscalationBudgetEnforcementMiddleware._count_gpt_calls_in_history([]) == 0


def test_count_no_ai_messages():
    msgs = [HumanMessage(content="hi")]
    assert EscalationBudgetEnforcementMiddleware._count_gpt_calls_in_history(msgs) == 0


def test_count_only_sonnet_calls():
    msgs = [_ai_with_sonnet_task("s1"), _ai_with_sonnet_task("s2")]
    assert EscalationBudgetEnforcementMiddleware._count_gpt_calls_in_history(msgs) == 0


def test_count_one_gpt_call():
    msgs = [_ai_with_gpt_task("g1")]
    assert EscalationBudgetEnforcementMiddleware._count_gpt_calls_in_history(msgs) == 1


def test_count_multiple_gpt_calls():
    msgs = [
        _ai_with_gpt_task("g1"),
        _ai_with_sonnet_task("s1"),
        _ai_with_gpt_task("g2"),
        _ai_with_other_tool("o1"),
        _ai_with_gpt_task("g3"),
    ]
    assert EscalationBudgetEnforcementMiddleware._count_gpt_calls_in_history(msgs) == 3


def test_count_ignores_non_task_tools():
    msgs = [_ai_with_other_tool("o1"), _ai_with_other_tool("o2")]
    assert EscalationBudgetEnforcementMiddleware._count_gpt_calls_in_history(msgs) == 0


# ─── _maybe_substitute ─────────────────────────────────────────────────────
def test_passthrough_for_non_gpt_call():
    mw = EscalationBudgetEnforcementMiddleware(default_budget=5)
    req = _tool_request(
        {"name": "web_search", "args": {"query": "x"}},
        {"messages": [_ai_with_gpt_task("g1")] * 10},  # over budget but not gpt call
    )
    result = mw._maybe_substitute(req)
    assert result.tool_call["name"] == "web_search"


def test_passthrough_for_gpt_under_budget():
    mw = EscalationBudgetEnforcementMiddleware(default_budget=12)
    # 5 previous gpt calls — under default budget of 12
    history = [_ai_with_gpt_task(f"g{i}") for i in range(5)]
    req = _tool_request(
        {"name": "task", "args": {"subagent_type": FACT_CHECKER_GPT_SUBAGENT, "prompt": "x"}},
        {"messages": history},
    )
    result = mw._maybe_substitute(req)
    assert result.tool_call["args"]["subagent_type"] == FACT_CHECKER_GPT_SUBAGENT


def test_substitute_when_over_budget():
    mw = EscalationBudgetEnforcementMiddleware(default_budget=3)
    history = [_ai_with_gpt_task(f"g{i}") for i in range(3)]
    req = _tool_request(
        {"name": "task", "args": {"subagent_type": FACT_CHECKER_GPT_SUBAGENT,
                                  "prompt": "verify this"}},
        {"messages": history},
    )
    result = mw._maybe_substitute(req)
    new_args = result.tool_call["args"]
    assert new_args["subagent_type"] == FACT_CHECKER_SONNET_SUBAGENT
    assert ESCALATION_BUDGET_WARNING_SENTINEL in new_args["prompt"]
    assert "verify this" in new_args["prompt"]  # original prompt preserved


def test_substitute_respects_state_budget_override():
    mw = EscalationBudgetEnforcementMiddleware(default_budget=20)
    history = [_ai_with_gpt_task(f"g{i}") for i in range(2)]
    # state budget=2 overrides default 20
    req = _tool_request(
        {"name": "task", "args": {"subagent_type": FACT_CHECKER_GPT_SUBAGENT, "prompt": "x"}},
        {"messages": history, "escalation_budget": 2},
    )
    result = mw._maybe_substitute(req)
    assert result.tool_call["args"]["subagent_type"] == FACT_CHECKER_SONNET_SUBAGENT


def test_substitute_at_exactly_budget():
    """Used == budget: substitute (the count is what's ALREADY happened;
    this call would be the over-budget one)."""
    mw = EscalationBudgetEnforcementMiddleware(default_budget=3)
    history = [_ai_with_gpt_task(f"g{i}") for i in range(3)]
    req = _tool_request(
        {"name": "task", "args": {"subagent_type": FACT_CHECKER_GPT_SUBAGENT, "prompt": "x"}},
        {"messages": history},
    )
    result = mw._maybe_substitute(req)
    assert result.tool_call["args"]["subagent_type"] == FACT_CHECKER_SONNET_SUBAGENT


# ─── wrap_tool_call ────────────────────────────────────────────────────────
def test_wrap_tool_call_invokes_handler_with_modified_request():
    mw = EscalationBudgetEnforcementMiddleware(default_budget=2)
    history = [_ai_with_gpt_task(f"g{i}") for i in range(2)]
    req = _tool_request(
        {"name": "task", "args": {"subagent_type": FACT_CHECKER_GPT_SUBAGENT, "prompt": "x"}},
        {"messages": history},
    )

    captured = {}
    def handler(r):
        captured["req"] = r
        return MagicMock(spec=ToolMessage)

    mw.wrap_tool_call(req, handler)
    assert captured["req"].tool_call["args"]["subagent_type"] == FACT_CHECKER_SONNET_SUBAGENT


def test_wrap_tool_call_passthrough_for_non_gpt():
    mw = EscalationBudgetEnforcementMiddleware(default_budget=2)
    req = _tool_request(
        {"name": "web_search", "args": {"query": "x"}},
        {"messages": []},
    )
    handler = MagicMock(return_value="result")
    result = mw.wrap_tool_call(req, handler)
    handler.assert_called_once()
    assert handler.call_args[0][0].tool_call["name"] == "web_search"
    assert result == "result"


# ─── Constants ─────────────────────────────────────────────────────────────
def test_sentinel_constant_stable():
    """Wrapper greps for this — don't change without coordinating."""
    assert ESCALATION_BUDGET_WARNING_SENTINEL == "[wrapper-p1-1-budget-exhausted]"


def test_default_budget_constant():
    assert DEFAULT_ESCALATION_BUDGET == 12
