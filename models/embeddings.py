"""
models/embeddings.py — Embedding model wrapper for the RAG pipeline.
Uses HuggingFace sentence-transformers (runs fully locally, no API key needed).
Model is cached as a singleton after first load to avoid repeated downloads.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Singleton cache — loaded once per Streamlit session
_model_cache: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Load (or return cached) sentence-transformer model."""
    global _model_cache
    if _model_cache is None:
        try:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            _model_cache = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model ready.")
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise
    return _model_cache


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into embedding vectors.

    Args:
        texts: List of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    try:
        model = get_embedding_model()
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception as exc:
        logger.error("Batch embed error: %s", exc)
        raise


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Args:
        query: The user's question or search phrase.

    Returns:
        np.ndarray of shape (embedding_dim,).
    """
    try:
        return embed_texts([query])[0]
    except Exception as exc:
        logger.error("Query embed error: %s", exc)
        raise
