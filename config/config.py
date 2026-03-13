"""
config/config.py — Application configuration.

All sensitive values are loaded from environment variables (.env).
Never commit real API keys to version control.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider (OpenRouter Only) ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

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