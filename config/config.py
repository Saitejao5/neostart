"""
config/config.py — Application configuration.

All sensitive values are loaded from environment variables (.env).
Never commit real API keys to version control.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root (robust path handling) ──────────────────────────
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# ── LLM Provider (OpenRouter Only) ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Validate API key exists
if not OPENROUTER_API_KEY:
    raise ValueError(
        "ERROR: OPENROUTER_API_KEY not found in .env file. "
        "Please ensure .env exists in the project root with your OpenRouter API key."
    )

# Default model used through OpenRouter
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct"
)

# OpenRouter endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Web Search Keys ────────────────────────────────────────────────────────────
SERPER_API_KEY  = os.getenv("SERPER_API_KEY", "")   # https://serper.dev
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY", "")   # https://tavily.com (fallback)

# ── RAG Settings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 80
TOP_K_RESULTS    = 4
VECTOR_DB_PATH   = "data/vectorstore"

# ── App Settings ───────────────────────────────────────────────────────────────
APP_TITLE         = "ScholarBot — AI Research Assistant"
APP_ICON          = "🎓"
MAX_HISTORY_TURNS = 20
TEMPERATURE       = 0.3

# ── Response Modes ─────────────────────────────────────────────────────────────
RESPONSE_MODES = {
    "Concise": (
        "Be brief and direct. Answer in 2–4 sentences maximum. "
        "No bullet points unless absolutely needed. Get straight to the point."
    ),
    "Detailed": (
        "Provide a thorough, well-structured answer. Use headings, bullet points, "
        "and examples where helpful. Explain reasoning clearly, cite sources, "
        "and include relevant caveats or further reading suggestions."
    ),
}

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ScholarBot, an intelligent AI research and education assistant built for NeoStats.
Your purpose is to help students, researchers, and lifelong learners understand complex topics,
summarize academic content, explore research papers, and find reliable information.

Guidelines:
- Be accurate, clear, and intellectually rigorous.
- Always cite the document or source when referencing uploaded materials.
- Encourage critical thinking — explain *why*, not just *what*.
- When uncertain, say so explicitly rather than guessing.
- Suggest follow-up questions or related topics when appropriate.
- Adapt your language to the user's apparent level of expertise.
"""