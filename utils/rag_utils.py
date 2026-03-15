"""
utils/rag_utils.py - Adapter layer.
app.py imports from here. We wrap utils/rag.py + add bytes-based extract_text
that app.py calls with (file_bytes, filename) from st.file_uploader.
"""
import io
import logging
import re
import numpy as np
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ── Lazy embedding model ────────────────────────────────────────────────────
def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError("sentence-transformers required. Add to requirements.txt")

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    return _get_model(model_name)

# ── Text extraction (bytes-based, for st.file_uploader) ─────────────────────
def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            return "\n\n".join(p.extract_text() or "" for p in reader.pages).strip()
        except ImportError:
            raise ImportError("pypdf required for PDF support.")
    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported: .{ext}. Use pdf, txt, md.")

# ── Chunking ────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 80) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
            else:
                b2 = text.rfind(" ", start, end)
                if b2 > start:
                    end = b2
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

# ── VectorStore ─────────────────────────────────────────────────────────────
class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._chunks: List[str] = []
        self._sources: List[str] = []
        self._embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        if self._model is None:
            self._model = _get_model(self._model_name)

    def _embed(self, texts):
        self._load_model()
        return self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    @staticmethod
    def _cosine_sim(q, matrix):
        q = q / (np.linalg.norm(q) + 1e-10)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        return (matrix / norms) @ q

    def add_documents(self, chunks: List[str], source: str = "unknown") -> None:
        if not chunks:
            return
        emb = self._embed(chunks)
        self._chunks.extend(chunks)
        self._sources.extend([source] * len(chunks))
        self._embeddings = emb if self._embeddings is None else np.vstack([self._embeddings, emb])

    def search(self, query: str, top_k: int = 4, score_threshold: float = 0.20) -> List[Dict[str, Any]]:
        if self.is_empty():
            return []
        q_vec = self._embed([query])[0]
        scores = self._cosine_sim(q_vec, self._embeddings)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self._chunks[i], "source": self._sources[i], "score": round(float(scores[i]), 4)}
            for i in top_idx if float(scores[i]) >= score_threshold
        ]

    def is_empty(self) -> bool:
        return len(self._chunks) == 0

    def clear(self) -> None:
        self._chunks, self._sources, self._embeddings = [], [], None

    def stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self._chunks),
            "sources": list(dict.fromkeys(self._sources)),
        }

# ── Context formatter ────────────────────────────────────────────────────────
def format_context(results: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    if not results:
        return ""
    lines, total = [], 0
    for i, r in enumerate(results, 1):
        entry = f"[{i}] Source: {r.get('source','?')} (relevance: {r.get('score',0):.2f})\n{r['text'].strip()}"
        if total + len(entry) > max_chars:
            break
        lines.append(entry)
        total += len(entry)
    return "\n\n".join(lines)
