"""
utils/chat_utils.py - Chat helpers matching app.py API.
build_system_prompt(behavior, response_mode, rag_enabled, web_search_enabled)
trim_history(messages, max_turns)
get_temperature(behavior)
"""
import logging
from typing import List, Dict
from config.config import BEHAVIOR_PRESETS, RESPONSE_MODES, SYSTEM_PROMPT, MAX_HISTORY_TURNS

logger = logging.getLogger(__name__)

def build_system_prompt(
    behavior: str,
    response_mode: str,
    rag_enabled: bool = False,
    web_search_enabled: bool = False,
) -> str:
    parts = [SYSTEM_PROMPT.strip()]
    preset = BEHAVIOR_PRESETS.get(behavior, {})
    if preset.get("system_suffix"):
        parts.append(preset["system_suffix"].strip())
    mode_instr = RESPONSE_MODES.get(response_mode)
    if mode_instr:
        parts.append(f"Response style: {mode_instr.strip()}")
    if rag_enabled:
        parts.append("The user has uploaded documents. Cite the source file name when referencing them.")
    if web_search_enabled:
        parts.append("Live web search results may be in the context. Mention when using them.")
    return "\n\n".join(parts)

def trim_history(messages: List[Dict], max_turns: int = MAX_HISTORY_TURNS) -> List[Dict]:
    max_msgs = max_turns * 2
    if len(messages) <= max_msgs:
        return messages
    trimmed = messages[-max_msgs:]
    if trimmed and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]
    return trimmed

def get_temperature(behavior: str) -> float:
    preset = BEHAVIOR_PRESETS.get(behavior, {})
    return float(preset.get("temperature", 0.7))
