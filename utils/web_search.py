"""
utils/web_search.py — Live web search integration.

Primary:  Serper.dev  (Google Search API — https://serper.dev)
Fallback: Tavily      (https://tavily.com)

API keys are managed in config/config.py via environment variables.
"""

import logging
import requests

from config.config import SERPER_API_KEY, TAVILY_API_KEY

logger = logging.getLogger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
REQUEST_TIMEOUT = 10  # seconds


# ── Serper (Primary) ───────────────────────────────────────────────────────────

def search_serper(query: str, num_results: int = 5) -> list[dict]:
    """
    Google Search via Serper.dev API.

    Returns:
        List of {"title": str, "snippet": str, "url": str}
    """
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not configured — skipping.")
        return []
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}
        resp = requests.post(SERPER_ENDPOINT, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url":     item.get("link", ""),
            }
            for item in data.get("organic", [])[:num_results]
        ]
    except Exception as exc:
        logger.error("Serper search error: %s", exc)
        return []


# ── Tavily (Fallback) ──────────────────────────────────────────────────────────

def search_tavily(query: str, num_results: int = 5) -> list[dict]:
    """
    Web search via Tavily API (fallback).

    Returns:
        List of {"title": str, "snippet": str, "url": str}
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not configured — skipping.")
        return []
    try:
        payload = {
            "api_key":      TAVILY_API_KEY,
            "query":        query,
            "search_depth": "basic",
            "max_results":  num_results,
        }
        resp = requests.post(TAVILY_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "title":   item.get("title", ""),
                "snippet": item.get("content", ""),
                "url":     item.get("url", ""),
            }
            for item in data.get("results", [])[:num_results]
        ]
    except Exception as exc:
        logger.error("Tavily search error: %s", exc)
        return []


# ── Unified Search ─────────────────────────────────────────────────────────────

def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Perform web search: Serper primary → Tavily fallback.

    Args:
        query:       Search query string.
        num_results: Max results to return.

    Returns:
        List of result dicts with keys: title, snippet, url.
    """
    results = search_serper(query, num_results)
    if not results:
        logger.info("Serper returned nothing — trying Tavily fallback.")
        results = search_tavily(query, num_results)
    return results


def format_search_results(results: list[dict]) -> str:
    """
    Format search results into a readable context block for LLM injection.

    Args:
        results: Output of web_search()

    Returns:
        Multi-result formatted string.
    """
    if not results:
        return "No web search results were found for this query."
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Web Result {i}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"{r['snippet']}"
        )
    return "\n\n".join(parts)
