import logging
import os

import httpx

logger = logging.getLogger(__name__)

_api_key_warned = False


class JinaClient:
    async def crawl(self, url: str, return_format: str = "html", timeout: int = 10) -> str:
        global _api_key_warned
        # NOTE: Do NOT send X-Timeout header. Empirically (2026-05-20) Jina holds the
        # connection open for the full X-Timeout value rather than returning as soon
        # as the page is fetched, causing client-side ReadTimeout. The `timeout=`
        # parameter on the httpx request below is what enforces the upper bound.
        headers = {
            "Content-Type": "application/json",
            "X-Return-Format": return_format,
        }
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        elif not _api_key_warned:
            _api_key_warned = True
            logger.warning("Jina API key is not set. Provide your own key to access a higher rate limit. See https://jina.ai/reader for more information.")
        data = {"url": url}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post("https://r.jina.ai/", headers=headers, json=data, timeout=timeout)

            if response.status_code != 200:
                error_message = f"Jina API returned status {response.status_code}: {response.text}"
                logger.error(error_message)
                return f"Error: {error_message}"

            if not response.text or not response.text.strip():
                error_message = "Jina API returned empty response"
                logger.error(error_message)
                return f"Error: {error_message}"

            return response.text
        except Exception as e:
            error_message = f"Request to Jina API failed: {type(e).__name__}: {e}"
            logger.warning(error_message)
            return f"Error: {error_message}"
