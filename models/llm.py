"""
models/llm.py — OpenRouter LLM interface.

This module provides a unified interface to interact with
OpenRouter models while keeping the same function signature
used throughout the application.
"""

import logging
from typing import Optional

import requests

from config.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ── OpenRouter Call ────────────────────────────────────────────────────────────
def get_openrouter_response(
    messages: list[dict],
    model: str = OPENROUTER_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Send messages to OpenRouter and return the assistant reply."""

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY.strip() == "":
        raise ValueError("OPENROUTER_API_KEY is not set. Check your .env file.")

    try:
        url = f"{OPENROUTER_BASE_URL}/chat/completions"

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://scholarbot.local",  # OpenRouter requires a referer
            "X-Title": "ScholarBot",
        }

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": model,
            "messages": full_messages,
            "temperature": TEMPERATURE,
            "max_tokens": 2048,
        }

        logger.info(f"Calling OpenRouter: {model}")
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 401:
            logger.error(
                "OpenRouter auth failed (401). Check your API key in .env file. "
                f"Key starts with: {OPENROUTER_API_KEY[:20]}..."
            )
        
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as exc:
        logger.error("OpenRouter HTTP error: %s - %s", exc.response.status_code, exc.response.text)
        raise
    except Exception as exc:
        logger.error("OpenRouter API error: %s", exc)
        raise


# ── Unified Interface (kept for compatibility) ─────────────────────────────────
def get_llm_response(
    messages: list[dict],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    Main entry point used by the application.

    The 'provider' argument is kept only for backward compatibility
    but is ignored because OpenRouter is the only backend now.
    """

    try:
        return get_openrouter_response(
            messages=messages,
            model=model or OPENROUTER_MODEL,
            system_prompt=system_prompt,
        )

    except Exception as exc:
        logger.error("LLM response error: %s", exc)
        raise