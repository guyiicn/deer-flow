"""PoC #3 — Minimal CitationMiddleware to observe agent reaction to deny.

Phase 0 PoC ONLY. Not for production use.

Behavior: any write_file call whose `content` arg contains an `openrouter.ai`
URL is DENIED with a structured error message instructing the agent to
fetch the URL first.

Purpose: empirically measure what main agent does when it gets a citation-related
deny ToolMessage:
- Does it modify just the offending URL?
- Does it remove the section entirely?
- Does it loop until cap?
- Does it bail?

This data informs Phase 2 (full fork) feasibility.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)

# Pattern to find URLs in markdown content. Mock target: openrouter.ai
_MOCK_TARGET = "openrouter.ai"


class CitationMiddlewarePoC(AgentMiddleware[AgentState]):
    """Phase 0 PoC: deny write_file when content contains a target-domain URL.

    The denial returns a structured ToolMessage so the agent can see why
    and (hopefully) adapt by removing the offending citation or fetching
    the URL first.
    """

    def _scan_content(self, content: str) -> list[str]:
        """Find URLs in content that match the mock 'unverified' criterion."""
        if not isinstance(content, str):
            return []
        url_pattern = re.compile(r"https?://[^\s)\"\]]+")
        urls = url_pattern.findall(content)
        return [u for u in urls if _MOCK_TARGET in u]

    def _build_denial_message(
        self,
        request: ToolCallRequest,
        offending_urls: list[str],
    ) -> ToolMessage:
        tool_name = str(request.tool_call.get("name") or "write_file")
        tool_call_id = str(request.tool_call.get("id") or "")
        msg = (
            f"[CitationMiddleware-PoC] write_file DENIED. The content includes "
            f"{len(offending_urls)} citation URL(s) under the mock-unverified "
            f"domain '{_MOCK_TARGET}':\n  "
            + "\n  ".join(offending_urls[:5])
            + f"\n\nReason: in this PoC the wrapper enforces 'fetch-before-cite' "
            f"for {_MOCK_TARGET} URLs. Please web_fetch each URL above to verify "
            f"it actually exists and matches the claim's context, OR remove the "
            f"citation. Then retry write_file with the corrected content."
        )
        logger.warning(
            "CitationMiddlewarePoC denied write_file with %d offending URLs",
            len(offending_urls),
        )
        return ToolMessage(
            content=msg,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    def _check(self, request: ToolCallRequest) -> ToolMessage | None:
        """Return a denial ToolMessage if the call should be denied, else None."""
        if request.tool_call.get("name") != "write_file":
            return None
        args = request.tool_call.get("args") or {}
        content = args.get("content", "")
        offending = self._scan_content(content)
        if offending:
            return self._build_denial_message(request, offending)
        return None

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        try:
            denial = self._check(request)
            if denial is not None:
                return denial
            return handler(request)
        except GraphBubbleUp:
            raise

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        try:
            denial = self._check(request)
            if denial is not None:
                return denial
            return await handler(request)
        except GraphBubbleUp:
            raise
