"""Phase 7 A'.2: Generic DanglingToolCallPatch middleware.

Runs between Tier 1 (DanglingToolCallMiddleware) and Tier 2
(OrphanRetryFailFastMiddleware) in the chain. Catches orphan
tool_use blocks that Tier 1 misses (specifically: raw Anthropic
content-block tool_use entries that aren't normalized into
msg.tool_calls).

Day 0 audit finding (Phase 7 Day 5):
  DanglingToolCallMiddleware (Tier 1) reads `msg.tool_calls` field
  and `additional_kwargs.tool_calls`. It does NOT scan raw content
  blocks of type `tool_use`. Gateway log evidence: Tier 1 has 0
  placeholder injections across the entire log history, yet Calib
  v3 still hit Anthropic 400 with orphan toolu_015f7FMLuLNeSahiPpxHotap
  at messages.9 — confirming the gap.

Algorithm:
  1. Scan ALL AI messages for tool_use blocks in `content` (raw
     Anthropic format) AND in `msg.tool_calls` (LangChain normalized).
  2. Collect all tool_call_ids that ToolMessages have already
     accounted for.
  3. Orphans = tool_use_ids that have no matching tool_result.
  4. For each orphan, synthesize ToolMessage(status='error', content
     with retry directive) and insert IMMEDIATELY AFTER the AI
     message that contains the orphan tool_use (P1-B insert-after-
     orphan-ai per design v2 §2.2).
  5. Idempotent dedupe via set subtraction (P6 Day 5 micro-MM1 §2
     Q2): if a tool_call_id already has a ToolMessage, skip — won't
     double-patch when real result commits late.

Per Phase 7 design v2 §2.4 ordering:
    EscalationBudgetEnforcementMiddleware
    DanglingToolCallMiddleware              # Tier 1 (existing)
    DanglingToolCallPatchMiddleware         # A'.2 (this file)
    OrphanRetryFailFastMiddleware           # Tier 2 (existing)
    Section8EnforcementMiddleware           # G1

A'.2 runs after Tier 1 so it catches the content-block gap, and
before Tier 2 so Tier 2 sees a clean history (no escalate to fresh-
thread retry unless A'.2 also failed).
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
)
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


# Sentinel for wrapper-side observability + per-run counting.
# Phase 7 wrapper sentinel pipeline parses this from gateway.log.
ORPHAN_PATCH_SENTINEL = "[wrapper-p7-orphan-patch]"


class DanglingToolCallPatchMiddleware(AgentMiddleware[AgentState]):
    """Phase 7 A'.2: Generic safety net for orphan tool_use blocks
    that escape DanglingToolCallMiddleware (Tier 1).

    Tier 1 reads msg.tool_calls / additional_kwargs.tool_calls but
    not raw content-block tool_use. This middleware fills that gap
    by directly inspecting Anthropic-format content blocks too.
    """

    @staticmethod
    def _collect_tool_uses(messages: list) -> dict[str, object]:
        """Return mapping tool_use_id -> the AI message containing it.

        Scans BOTH representations:
          1. Anthropic raw content blocks: content=[{type:'tool_use',
             id:'toolu_xxx', name:'...', input:{...}}, ...]
          2. LangChain normalized: msg.tool_calls = [{id, name, args}]
          3. OpenAI raw: msg.additional_kwargs.tool_calls

        If both populated for the same id, the AI message in (1)
        owners the slot (used for insertion ordering).
        """
        tool_use_to_ai: dict[str, object] = {}
        for msg in messages:
            if getattr(msg, "type", None) != "ai":
                continue
            # 1. Raw content blocks (Anthropic)
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) \
                            and block.get("type") == "tool_use":
                        tu_id = block.get("id")
                        if tu_id and tu_id not in tool_use_to_ai:
                            tool_use_to_ai[tu_id] = msg
            # 2. LangChain normalized tool_calls
            for tc in (getattr(msg, "tool_calls", None) or []):
                tc_id = tc.get("id") if isinstance(tc, dict) else \
                        getattr(tc, "id", None)
                if tc_id and tc_id not in tool_use_to_ai:
                    tool_use_to_ai[tc_id] = msg
            # 3. OpenAI raw additional_kwargs (defensive)
            raw_tcs = (getattr(msg, "additional_kwargs", None) or {}) \
                .get("tool_calls") or []
            for raw_tc in raw_tcs:
                if not isinstance(raw_tc, dict):
                    continue
                tc_id = raw_tc.get("id")
                if tc_id and tc_id not in tool_use_to_ai:
                    tool_use_to_ai[tc_id] = msg
        return tool_use_to_ai

    @staticmethod
    def _collect_tool_result_ids(messages: list) -> set[str]:
        """Return tool_call_ids that have a matching ToolMessage."""
        ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tc_id = getattr(msg, "tool_call_id", None)
                if tc_id:
                    ids.add(tc_id)
        return ids

    @classmethod
    def _find_orphan_tool_use_ids(cls, messages: list) -> set[str]:
        """Idempotent dedupe via set subtraction (P6 D5 micro-MM1 Q2).

        tool_use_ids without matching tool_result_ids. Already-paired
        tool_uses are skipped — won't double-patch when real result
        commits late.
        """
        all_tool_uses = set(cls._collect_tool_uses(messages).keys())
        all_results = cls._collect_tool_result_ids(messages)
        return all_tool_uses - all_results

    @staticmethod
    def _build_synthetic(tu_id: str) -> ToolMessage:
        """Synthetic error ToolMessage with STRONG retry directive.

        Per Phase 7 design v2 P2-B: 'MUST retry NOW before continuing
        with subsequent steps' — strict wording reduces R5 risk of
        agent treating synthetic error as a permanent failure and
        skipping the tool entirely (R8: agent abandons batch).
        """
        return ToolMessage(
            content=(
                f"{ORPHAN_PATCH_SENTINEL} Synthetic error: tool "
                f"execution result was not committed to state. This "
                f"is a transient infrastructure error. If this tool "
                f"call was part of your planned workflow, you MUST "
                f"retry the tool call NOW before continuing with "
                f"subsequent steps."
            ),
            tool_call_id=tu_id,
            name="unknown_tool",
            status="error",
        )

    @classmethod
    def _insert_synthetic_after_orphan_ai(
        cls, messages: list, orphan_ids: set[str]
    ) -> list:
        """Insert synthetic ToolMessage IMMEDIATELY after the AI
        message containing each orphan tool_use (P1-B per design v2
        §2.2). Append-at-end risks Anthropic API rejection when the
        history shape places tool_result far from its tool_use.

        Edge: if an orphan id is somehow not found in any AI message
        (shouldn't happen with set logic, but defensive), it's
        appended at the end with a warning.
        """
        tool_use_to_ai = cls._collect_tool_uses(messages)
        remaining = set(orphan_ids)
        result: list = []
        for m in messages:
            result.append(m)
            if not remaining:
                continue
            if getattr(m, "type", None) != "ai":
                continue
            # Find orphans whose owning AI message is THIS one
            for tu_id in list(remaining):
                if tool_use_to_ai.get(tu_id) is m:
                    result.append(cls._build_synthetic(tu_id))
                    remaining.discard(tu_id)
        for stray in remaining:
            logger.warning(
                f"{ORPHAN_PATCH_SENTINEL} orphan id {stray} not found "
                f"in any AI message; appending at end as fallback"
            )
            result.append(cls._build_synthetic(stray))
        return result

    @classmethod
    def _maybe_patch(cls, request: ModelRequest) -> ModelRequest:
        orphans = cls._find_orphan_tool_use_ids(request.messages)
        if not orphans:
            return request
        new_messages = cls._insert_synthetic_after_orphan_ai(
            request.messages, orphans
        )
        logger.warning(
            f"{ORPHAN_PATCH_SENTINEL} patched {len(orphans)} orphan "
            f"tool_use(s): {sorted(orphans)}"
        )
        return request.override(messages=new_messages)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        # Use self._maybe_patch — Python auto-resolves to classmethod.
        # (cls is not defined in instance methods.)
        return handler(self._maybe_patch(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._maybe_patch(request))
