"""Phase 2 P0-1 Tier 2 + P1-1 escalation budget middleware.

P0-1 Tier 2: fail-fast safety net for orphan tool_use blocks
that escape DanglingToolCallMiddleware.

DeerFlow's DanglingToolCallMiddleware (in agents/middlewares/) already
patches orphan tool_use blocks before model calls — and works correctly
for the common case (verified by Phase 2 Day 0.5 investigation with
debug logging on a 132s sanity run, where it fired on every model call
and judged messages well-paired).

However, Phase 1 sanity v2 (an 8-minute long-conversation run) hit a
mid-stream Anthropic 400 "tool_use ids were found without tool_result
blocks immediately after". That suggests a rare deep-conversation edge
case where DanglingToolCallMiddleware doesn't catch the orphan.

This middleware runs AFTER DanglingToolCallMiddleware in the chain. If
orphans are STILL present at this point, we raise a structured error
instead of letting the request hit Anthropic and 400. The wrapper
catches the error and retries the run with a fresh thread (graceful
degrade — we don't try to preserve mid-conversation state).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


# Sentinel string the wrapper greps for. Keep it stable — wrapper-side
# detection is brittle to phrasing changes.
ORPHAN_AFTER_PATCH_SENTINEL = "[p0-1-tier2-orphan-after-patch]"

# P1-1 escalation budget enforcement
ESCALATION_BUDGET_WARNING_SENTINEL = "[wrapper-p1-1-budget-exhausted]"
DEFAULT_ESCALATION_BUDGET = 12   # default per-run cap on fact-checker-gpt calls
FACT_CHECKER_GPT_SUBAGENT = "fact-checker-gpt"
FACT_CHECKER_SONNET_SUBAGENT = "fact-checker-sonnet"


class OrphanToolUseAfterPatchError(RuntimeError):
    """Raised when orphan tool_use blocks are still present despite
    DanglingToolCallMiddleware running upstream. Wrapper catches via
    sentinel string in the streamed error and retries with fresh thread.
    """

    def __init__(self, orphan_tool_call_ids: list[str]) -> None:
        self.orphan_tool_call_ids = orphan_tool_call_ids
        super().__init__(
            f"{ORPHAN_AFTER_PATCH_SENTINEL} orphan tool_use blocks after "
            f"DanglingToolCallMiddleware: {orphan_tool_call_ids}"
        )


class OrphanRetryFailFastMiddleware(AgentMiddleware[AgentState]):
    """Phase 2 P0-1 Tier 2 — must run AFTER DanglingToolCallMiddleware.

    Inspects request.messages for AIMessages with tool_calls that lack a
    matching ToolMessage anywhere in the history. If found, raises
    OrphanToolUseAfterPatchError so the wrapper-level retry can take
    over with a fresh thread.
    """

    @staticmethod
    def _collect_orphans(messages) -> list[str]:
        """Return list of tool_call_ids that have an AIMessage reference
        but no corresponding ToolMessage anywhere in the message list."""
        tool_result_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tcid = getattr(msg, "tool_call_id", None)
                if tcid:
                    tool_result_ids.add(tcid)

        orphans: list[str] = []
        for msg in messages:
            if getattr(msg, "type", None) != "ai":
                continue
            tcs = getattr(msg, "tool_calls", None) or []
            for tc in tcs:
                # Handle both dict-style and object-style tool_calls
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id and tc_id not in tool_result_ids:
                    orphans.append(tc_id)
        return orphans

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        orphans = self._collect_orphans(request.messages)
        if orphans:
            logger.warning(
                f"{ORPHAN_AFTER_PATCH_SENTINEL} orphan tool_call_ids={orphans} "
                f"escaped DanglingToolCallMiddleware — raising for wrapper retry"
            )
            raise OrphanToolUseAfterPatchError(orphans)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        orphans = self._collect_orphans(request.messages)
        if orphans:
            logger.warning(
                f"{ORPHAN_AFTER_PATCH_SENTINEL} orphan tool_call_ids={orphans} "
                f"escaped DanglingToolCallMiddleware — raising for wrapper retry"
            )
            raise OrphanToolUseAfterPatchError(orphans)
        return await handler(request)


class EscalationBudgetEnforcementMiddleware(AgentMiddleware[AgentState]):
    """Phase 2 P1-1: enforce escalation budget by intercepting task() calls
    to fact-checker-gpt and substituting fact-checker-sonnet when the count
    of prior gpt escalations meets or exceeds the budget.

    Spike result: no ThreadState extension needed. Count is derived live
    from message history (each AIMessage with a task tool_call to
    fact-checker-gpt counts as 1). Budget is a class default or
    state['escalation_budget'] if provided by caller.

    Soft enforcement: the substituted call carries the warning sentinel
    string in its prompt so the agent sees inline that budget was hit.
    """

    def __init__(self, default_budget: int = DEFAULT_ESCALATION_BUDGET) -> None:
        super().__init__()
        self._default_budget = default_budget

    @staticmethod
    def _is_gpt_escalation_call(tool_call: dict) -> bool:
        if tool_call.get("name") != "task":
            return False
        args = tool_call.get("args") or {}
        if not isinstance(args, dict):
            return False
        return args.get("subagent_type") == FACT_CHECKER_GPT_SUBAGENT

    @staticmethod
    def _count_gpt_calls_in_history(messages) -> int:
        """Count completed fact-checker-gpt task() invocations in the
        message history. Each AIMessage with such a tool_call counts as 1
        (regardless of whether the result was returned)."""
        count = 0
        for msg in messages or []:
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in (getattr(msg, "tool_calls", None) or []):
                # Each tc is dict-like {"name": ..., "args": {...}, ...}
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args") or {}
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", None) or {}
                if name == "task" and isinstance(args, dict) and \
                   args.get("subagent_type") == FACT_CHECKER_GPT_SUBAGENT:
                    count += 1
        return count

    def _budget(self, state) -> int:
        if state and isinstance(state, dict):
            v = state.get("escalation_budget")
            if isinstance(v, int) and v > 0:
                return v
        return self._default_budget

    def _maybe_substitute(self, request: ToolCallRequest) -> ToolCallRequest:
        """Return a (possibly modified) request — substitutes gpt with
        sonnet + warning prefix if budget exceeded."""
        if not self._is_gpt_escalation_call(request.tool_call):
            return request
        state_dict = getattr(request, "state", None)
        if state_dict is None:
            return request
        messages = state_dict.get("messages", []) if isinstance(state_dict, dict) else []
        used = self._count_gpt_calls_in_history(messages)
        budget = self._budget(state_dict)
        if used < budget:
            return request   # under budget, let it through

        original_args = request.tool_call.get("args") or {}
        original_prompt = original_args.get("prompt", "")
        warning = (
            f"{ESCALATION_BUDGET_WARNING_SENTINEL} Escalation budget "
            f"exhausted ({used}/{budget} gpt calls used). The wrapper "
            f"redirected this audit to fact-checker-sonnet instead. "
            f"Treat the verdict accordingly — sonnet may miss what gpt "
            f"would have caught (cross-provider blind spots).\n\n"
        )
        modified_args = {
            **original_args,
            "subagent_type": FACT_CHECKER_SONNET_SUBAGENT,
            "prompt": warning + original_prompt,
        }
        modified_call = {**request.tool_call, "args": modified_args}
        logger.warning(
            f"{ESCALATION_BUDGET_WARNING_SENTINEL} substituted gpt→sonnet "
            f"(history shows {used} gpt calls already, budget={budget})"
        )
        return request.override(tool_call=modified_call)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], object],
    ) -> object:
        return handler(self._maybe_substitute(request))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[object]],
    ) -> object:
        return await handler(self._maybe_substitute(request))
