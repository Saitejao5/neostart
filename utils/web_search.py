"""
utils/web_search.py - Live web search.
Priority: Serper -> Tavily -> DuckDuckGo (free fallback, no key needed)
"""
import os
import logging
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)
SERPER_ENDPOINT = "https://google.serper.dev/search"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TIMEOUT = 10

def _search_serper(query: str, n: int = 5) -> List[Dict]:
    key = os.getenv("SERPER_API_KEY", "").strip()
    if not key:
        return []
    try:
        r = requests.post(SERPER_ENDPOINT, json={"q": query, "num": n},
                          headers={"X-API-KEY": key, "Content-Type": "application/json"}, timeout=TIMEOUT)
        r.raise_for_status()
        return [{"title": x.get("title",""), "snippet": x.get("snippet",""), "url": x.get("link","")}
                for x in r.json().get("organic", [])[:n]]
    except Exception as e:
        logger.warning("Serper error: %s", e)
        return []

def _search_tavily(query: str, n: int = 5) -> List[Dict]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return []
    try:
        r = requests.post(TAVILY_ENDPOINT, json={"api_key": key, "query": query,
                          "search_depth": "basic", "max_results": n}, timeout=TIMEOUT)
        r.raise_for_status()
        return [{"title": x.get("title",""), "snippet": x.get("content",""), "url": x.get("url","")}
                for x in r.json().get("results", [])[:n]]
    except Exception as e:
        logger.warning("Tavily error: %s", e)
        return []

def _search_ddg(query: str, n: int = 5) -> List[Dict]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as d:
            return [{"title": r.get("title",""), "snippet": r.get("body",""), "url": r.get("href","")}
                    for r in d.text(query, max_results=n)]
    except Exception as e:
        logger.error("DuckDuckGo error: %s", e)
        return []

def web_search(query: str, num_results: int = 5) -> List[Dict]:
    r = _search_serper(query, num_results)
    if r:
        return r
    r = _search_tavily(query, num_results)
    if r:
        return r
    return _search_ddg(query, num_results)

def format_search_results(results: List[Dict]) -> str:
    if not results:
        return "No web search results found."
    return "\n\n".join(
        f"[Web Result {i}] {r.get('title','')}\nURL: {r.get('url','')}\n{r.get('snippet','')}"
        for i, r in enumerate(results, 1)
    )
