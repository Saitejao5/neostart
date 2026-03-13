"""
utils/chat.py — Chat history management and prompt construction.

Keeps the message history within context limits and injects
RAG/web-search context + response mode instruction into each user turn.
"""

import logging

from config.config import RESPONSE_MODES, MAX_HISTORY_TURNS, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Keep only the most recent N conversation turns.
    Each turn = 1 user message + 1 assistant message (2 dicts).

    Args:
        history:   Full list of {"role": ..., "content": ...} dicts.
        max_turns: Maximum conversation turns to retain.

    Returns:
        Trimmed history list.
    """
    max_messages = max_turns * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def build_prompt_with_context(
    user_query: str,
    rag_context:  str = "",
    web_context:  str = "",
    response_mode: str = "Concise",
) -> str:
    """
    Construct the enriched user message with injected context and mode instruction.

    Context injection order:
      1. Response mode instruction
      2. RAG document excerpts (if any)
      3. Web search results (if any)
      4. The actual user question

    Args:
        user_query:    The raw question from the user.
        rag_context:   Retrieved document chunks (empty if RAG unused).
        web_context:   Web search result text (empty if search unused).
        response_mode: "Concise" | "Detailed"

    Returns:
        Enhanced user message string ready to send to the LLM.
    """
    mode_instruction = RESPONSE_MODES.get(response_mode, RESPONSE_MODES["Concise"])
    parts = [f"[Response Mode — {response_mode}]: {mode_instruction}\n"]

    if rag_context:
        parts.append(
            "### Relevant Document Context\n"
            "Use these excerpts from the user's uploaded documents to ground your answer:\n\n"
            f"{rag_context}\n"
        )

    if web_context:
        parts.append(
            "### Live Web Search Results\n"
            "Supplement your answer with these current findings from the web:\n\n"
            f"{web_context}\n"
        )

    parts.append(f"### User Question\n{user_query}")
    return "\n".join(parts)


def build_system_prompt(response_mode: str = "Concise") -> str:
    """
    Build the final system prompt, embedding the current response mode.

    Args:
        response_mode: "Concise" | "Detailed"

    Returns:
        Complete system prompt string.
    """
    mode_note = (
        "Be brief and direct (2-4 sentences max)."
        if response_mode == "Concise"
        else "Be thorough and well-structured with headings and examples."
    )
    return SYSTEM_PROMPT + f"\n\n[Active response mode: {response_mode}] {mode_note}"
