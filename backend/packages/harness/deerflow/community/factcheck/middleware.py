"""Phase 2 P0-1 Tier 2: fail-fast safety net for orphan tool_use blocks
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
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


# Sentinel string the wrapper greps for. Keep it stable — wrapper-side
# detection is brittle to phrasing changes.
ORPHAN_AFTER_PATCH_SENTINEL = "[p0-1-tier2-orphan-after-patch]"


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
