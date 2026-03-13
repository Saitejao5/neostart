"""
utils/rag.py — Full RAG (Retrieval-Augmented Generation) pipeline.

Handles:
  - Multi-format document text extraction (PDF, DOCX, TXT, MD)
  - Overlapping text chunking
  - FAISS-based vector indexing and retrieval
  - Context block formatting for LLM prompts
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, VECTOR_DB_PATH
from models.embeddings import embed_texts, embed_query

logger = logging.getLogger(__name__)


# ── Text Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as exc:
        logger.error("PDF extraction failed [%s]: %s", file_path, exc)
        raise


def extract_text_from_docx(file_path: str) -> str:
    """Extract raw text from a .docx Word document."""
    try:
        import docx
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as exc:
        logger.error("DOCX extraction failed [%s]: %s", file_path, exc)
        raise


def extract_text_from_txt(file_path: str) -> str:
    """Read a plain text or Markdown file."""
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error("TXT read failed [%s]: %s", file_path, exc)
        raise


def extract_text(file_path: str) -> str:
    """
    Auto-detect file type and extract text content.
    Supports: .pdf, .docx, .txt, .md
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'. Use PDF, DOCX, TXT, or MD.")


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    Overlap ensures context isn't lost at chunk boundaries,
    which improves retrieval quality for long documents.

    Args:
        text:       Raw document text.
        chunk_size: Max characters per chunk.
        overlap:    Overlap characters between consecutive chunks.

    Returns:
        List of non-empty text chunk strings.
    """
    chunks = []
    text = text.strip()
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Vector Store ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory FAISS vector store with optional disk persistence.

    Stores document chunks as L2-indexed float32 embeddings.
    Retrieves the top-K most semantically similar chunks for a query.
    """

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunks:   list[str]  = []
        self.metadata: list[dict] = []   # {"source": filename, "chunk_id": int}

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_documents(self, file_paths: list[str]) -> int:
        """
        Ingest a list of files: extract → chunk → embed → index.

        Args:
            file_paths: Absolute paths to document files.

        Returns:
            Total number of chunks successfully indexed.
        """
        all_chunks, all_meta = [], []

        for fp in file_paths:
            try:
                text = extract_text(fp)
                chunks = chunk_text(text)
                name = Path(fp).name
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_meta.append({"source": name, "chunk_id": i})
                logger.info("Indexed %d chunks from '%s'", len(chunks), name)
            except Exception as exc:
                logger.warning("Skipping '%s': %s", fp, exc)

        if not all_chunks:
            logger.warning("No chunks were produced from provided files.")
            return 0

        embeddings = embed_texts(all_chunks).astype(np.float32)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.chunks.extend(all_chunks)
        self.metadata.extend(all_meta)
        return len(all_chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
        """
        Retrieve the most semantically relevant chunks for a query.

        Args:
            query: User's question string.
            top_k: Number of results to return.

        Returns:
            List of dicts: {"text": ..., "source": ..., "score": float}
        """
        if self.index is None or self.index.ntotal == 0:
            logger.info("Vector store is empty — no retrieval performed.")
            return []
        try:
            q_vec = embed_query(query).astype(np.float32).reshape(1, -1)
            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(q_vec, k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                results.append({
                    "text":   self.chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "score":  float(dist),
                })
            return results
        except Exception as exc:
            logger.error("Retrieval error: %s", exc)
            return []

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str = VECTOR_DB_PATH) -> None:
        """Persist index and chunks to disk."""
        try:
            os.makedirs(path, exist_ok=True)
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
            with open(os.path.join(path, "chunks.pkl"), "wb") as f:
                pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
            logger.info("Vector store saved to '%s'", path)
        except Exception as exc:
            logger.error("Save failed: %s", exc)

    def load(self, path: str = VECTOR_DB_PATH) -> bool:
        """Load a previously saved vector store. Returns True on success."""
        try:
            idx_path = os.path.join(path, "index.faiss")
            pkl_path = os.path.join(path, "chunks.pkl")
            if not (os.path.exists(idx_path) and os.path.exists(pkl_path)):
                return False
            self.index = faiss.read_index(idx_path)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.chunks   = data["chunks"]
            self.metadata = data["metadata"]
            logger.info("Vector store loaded: %d chunks", len(self.chunks))
            return True
        except Exception as exc:
            logger.error("Load failed: %s", exc)
            return False

    def clear(self) -> None:
        """Reset the vector store entirely."""
        self.index    = None
        self.chunks   = []
        self.metadata = []

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        return sorted(set(m["source"] for m in self.metadata))


# ── Context Formatter ──────────────────────────────────────────────────────────

def build_rag_context(retrieved: list[dict]) -> str:
    """
    Format retrieved chunks into a readable context block for LLM injection.

    Args:
        retrieved: Output of VectorStore.retrieve()

    Returns:
        Formatted multi-chunk string.
    """
    if not retrieved:
        return ""
    parts = []
    for i, r in enumerate(retrieved, 1):
        parts.append(f"[Excerpt {i} — Source: {r['source']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)
