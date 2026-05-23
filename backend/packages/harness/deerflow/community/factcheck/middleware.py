"""Phase 2/3 factcheck middlewares (3 classes in this module).

P0-1 Tier 2 (Phase 2): fail-fast safety net for orphan tool_use blocks
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

import re

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


# P0-1 Tier 2 sentinel (Phase 2). Wrapper greps for this.
ORPHAN_AFTER_PATCH_SENTINEL = "[p0-1-tier2-orphan-after-patch]"

# P1-1 escalation budget enforcement (Phase 2)
ESCALATION_BUDGET_WARNING_SENTINEL = "[wrapper-p1-1-budget-exhausted]"
DEFAULT_ESCALATION_BUDGET = 12   # default per-run cap on fact-checker-gpt calls
FACT_CHECKER_GPT_SUBAGENT = "fact-checker-gpt"
FACT_CHECKER_SONNET_SUBAGENT = "fact-checker-sonnet"

# G1 §8 enforcement (Phase 3). Sentinel string for nag-loop detection
# and for wrapper-side debug visibility.
SECTION_8_ENFORCEMENT_SENTINEL = "[wrapper-p3-g1]"
FACT_CHECKER_SUBAGENT_PREFIX = "fact-checker-"   # matches sonnet / gpt
# Sections that legitimately don't need fact-check (URL bibliographies,
# table of contents, etc.). Matches wrapper's EXEMPT_SECTION_IDS.
G1_EXEMPT_SECTION_IDS = frozenset({
    "sources", "references", "bibliography", "citations",
    "appendix", "footnotes", "acknowledgements", "acknowledgments",
    "table-of-contents", "toc", "index",
})


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


# Regex to extract `## Heading` slugs from markdown content
_G1_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_G1_SLUG_TRIM_RE = re.compile(r"[^a-z0-9一-鿿]+")


def _g1_slugify_heading(text: str) -> str:
    """Phase 3 G1: deterministic slug — must match wrapper's _slugify_heading
    (deer-flow-wrapper/deerflow). CJK characters preserved (Phase 2 P3-1)."""
    s = re.sub(r"[*`_~]", "", text or "").strip().lower()
    s = _G1_SLUG_TRIM_RE.sub("-", s).strip("-")
    return s or "untitled"


class Section8EnforcementMiddleware(AgentMiddleware[AgentState]):
    """Phase 3 G1: wrapper-side §8 protocol enforcement.

    Detects orphan write_file(outputs/*.md) events in message history
    where the main agent wrote sections but did NOT follow up with a
    task(fact-checker-*) call within LOOKAHEAD_TURNS model turns.

    Enforcement is via SYSTEM MESSAGE injection (NOT synthetic ToolMessage).
    Phase 3 design v2 §3.2.3 explicitly rejects fake ToolMessage approach
    because it would pollute conversation with fabricated fact-checker
    verdict — agent would make decisions on counterfeit data.

    Instead we inject a system reminder telling the agent to call
    fact-checker-sonnet NOW for each unverified section. Agent then makes
    the real task() call itself and gets real verdict data.

    Nag-loop guard (_already_nagged_recently): if our sentinel string
    appears in the last 4 messages, don't inject again — let the agent
    act on the previous reminder first.

    EXEMPT_SECTION_IDS handled (Sources/References etc. — same as wrapper).
    """

    LOOKAHEAD_TURNS = 3
    NAG_LOOKBACK = 4   # messages — should cover 1-2 model turns of context

    @staticmethod
    def _extract_write_file_calls(messages: list) -> list[tuple[int, list[str]]]:
        """Find all write_file tool_call invocations in the message history
        targeting outputs/*.md, return list of (message_index, [section_slugs]).
        """
        out = []
        for i, msg in enumerate(messages or []):
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in (getattr(msg, "tool_calls", None) or []):
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args") or {}
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", None) or {}
                if name != "write_file":
                    continue
                if not isinstance(args, dict):
                    continue
                path = args.get("path") or args.get("file_path") or ""
                if not isinstance(path, str) or not path.endswith(".md"):
                    continue
                if "/outputs/" not in path.lower() and not path.lower().startswith("outputs/"):
                    continue
                content = args.get("content") or ""
                slugs = [
                    _g1_slugify_heading(h) for h in _G1_HEADING_RE.findall(content or "")
                ]
                # Filter exempt sections (Sources, References, etc.)
                slugs = [s for s in slugs if s not in G1_EXEMPT_SECTION_IDS]
                if slugs:
                    out.append((i, slugs))
        return out

    @staticmethod
    def _extract_fact_checked_section_ids(messages: list, from_idx: int) -> set[str]:
        """Return section_ids that received a task(fact-checker-*) call
        AFTER from_idx in the message history. Slug extracted from args.prompt
        text via `section_id=<slug>` regex or from args.section_id field."""
        kv_re = re.compile(r"section_id\s*=\s*([A-Za-z0-9一-鿿](?:[\w\-]|\.(?=[A-Za-z0-9一-鿿]))*)")
        verified: set[str] = set()
        for msg in messages[from_idx:]:
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in (getattr(msg, "tool_calls", None) or []):
                if isinstance(tc, dict):
                    name = tc.get("name")
                    args = tc.get("args") or {}
                else:
                    name = getattr(tc, "name", None)
                    args = getattr(tc, "args", None) or {}
                if name != "task" or not isinstance(args, dict):
                    continue
                sub_t = args.get("subagent_type", "")
                if not isinstance(sub_t, str) or not sub_t.startswith(FACT_CHECKER_SUBAGENT_PREFIX):
                    continue
                # Try explicit section_id arg first, then prompt regex
                explicit = args.get("section_id")
                if isinstance(explicit, str) and explicit:
                    verified.add(_g1_slugify_heading(explicit))
                    continue
                prompt = args.get("prompt", "")
                if isinstance(prompt, str):
                    for m in kv_re.finditer(prompt):
                        verified.add(_g1_slugify_heading(m.group(1)))
        return verified

    @classmethod
    def _find_orphan_sections(cls, messages: list) -> list[str]:
        """Identify sections that have been written to outputs/*.md but not
        yet fact-checked within LOOKAHEAD_TURNS of their write_file event."""
        write_events = cls._extract_write_file_calls(messages)
        if not write_events:
            return []
        # All distinct section slugs ever written
        all_written = set()
        for _, slugs in write_events:
            all_written.update(slugs)
        # All sections fact-checked so far (from any write_file forward)
        verified = cls._extract_fact_checked_section_ids(messages, 0)
        # Orphans: written but never verified
        return sorted(all_written - verified)

    @classmethod
    def _already_nagged_recently(cls, messages: list) -> bool:
        """Nag-loop guard: if our sentinel appears in the last NAG_LOOKBACK
        messages, don't inject again — let the agent act on the previous
        reminder first."""
        for msg in (messages or [])[-cls.NAG_LOOKBACK:]:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and SECTION_8_ENFORCEMENT_SENTINEL in content:
                return True
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and SECTION_8_ENFORCEMENT_SENTINEL in str(block.get("text", "")):
                        return True
        return False

    @classmethod
    def _build_nag_message(cls, orphan_sections: list[str]) -> SystemMessage:
        section_list = ", ".join(orphan_sections)
        return SystemMessage(content=(
            f"{SECTION_8_ENFORCEMENT_SENTINEL} You wrote section(s) "
            f"[{section_list}] but did NOT call task(subagent_type="
            f"'fact-checker-sonnet', ...) for them. Per OUTPUT_POLICY §8, "
            f"fact-checking EVERY section is MANDATORY before finalizing. "
            f"Call task() for each unverified section NOW. Skipping = "
            f"exit 33 = run invalid. Do not summarize / finalize / pause "
            f"to ask user permission until every section has been "
            f"fact-checked."
        ))

    @classmethod
    def _maybe_inject(cls, request: ModelRequest) -> ModelRequest:
        messages = list(request.messages or [])
        if cls._already_nagged_recently(messages):
            return request   # let prior nag take effect first
        orphans = cls._find_orphan_sections(messages)
        if not orphans:
            return request   # all sections verified, nothing to do
        nag = cls._build_nag_message(orphans)
        # Prepend so model sees it before user-facing context
        new_messages = [nag] + messages
        logger.warning(
            f"{SECTION_8_ENFORCEMENT_SENTINEL} injecting nag for unverified "
            f"sections {orphans}"
        )
        return request.override(messages=new_messages)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._maybe_inject(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._maybe_inject(request))
