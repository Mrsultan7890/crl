import logging
import re
import threading
from typing import Dict, List, Literal, Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

ModeType = Literal["keyword", "semantic", "both"]

# ── Tokenizer ─────────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^\w\s]")
_STOP_WORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "was", "are",
    "be", "as", "at", "so", "we", "he", "she", "they", "you", "i",
])


def _tokenize(text: str) -> List[str]:
    """Use bridge zero-copy tokenizer — faster than naive split."""
    from .bridge import tokenize as _bridge_tokenize
    return _bridge_tokenize(text)


# ── Sentence Transformer singleton (thread-safe lazy load) ────────────────────

class _ModelRegistry:
    _lock = threading.Lock()
    _models: Dict[str, object] = {}

    @classmethod
    def get(cls, model_name: str):
        if model_name not in cls._models:
            with cls._lock:
                if model_name not in cls._models:
                    try:
                        from sentence_transformers import SentenceTransformer
                        logger.info("Loading semantic model '%s' (one-time)...", model_name)
                        cls._models[model_name] = SentenceTransformer(model_name)
                        logger.info("Model '%s' loaded.", model_name)
                    except ImportError:
                        logger.warning(
                            "sentence-transformers not installed. "
                            "Semantic scoring unavailable. Install with: "
                            "pip install sentence-transformers"
                        )
                        cls._models[model_name] = None
        return cls._models[model_name]


# ── Scoring ───────────────────────────────────────────────────────────────────

def keyword_score(pages: List[Dict], query: str) -> List[Dict]:
    """BM25 keyword scoring with stopword-filtered tokenization."""
    corpus = [_tokenize(p["text"]) for p in pages]
    if not any(corpus):
        for page in pages:
            page["keyword_score"] = 0.0
        return pages

    bm25 = BM25Okapi(corpus)
    raw_scores = bm25.get_scores(_tokenize(query))
    normalized = _minmax_normalize(raw_scores)
    for page, score in zip(pages, normalized):
        page["keyword_score"] = round(float(score), 6)
    return pages


def semantic_score(pages: List[Dict], query: str, model_name: str = "all-MiniLM-L6-v2") -> List[Dict]:
    """
    Semantic cosine similarity scoring using sentence-transformers.
    Model is loaded once and reused across all calls (singleton).
    Text is chunked and max-pooled for long documents.
    """
    model = _ModelRegistry.get(model_name)
    if model is None:
        for page in pages:
            page["semantic_score"] = 0.0
        return pages

    from sentence_transformers import util

    texts = [_chunk_and_pool(p["text"], model) for p in pages]
    query_emb = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    scores = [float(util.cos_sim(query_emb, t)[0][0]) for t in texts]
    normalized = _minmax_normalize(np.array(scores))
    for page, score in zip(pages, normalized):
        page["semantic_score"] = round(float(score), 6)
    return pages


def rank(
    pages: List[Dict],
    query: str,
    top_k: Optional[int] = None,
    mode: ModeType = "both",
    semantic_weight: float = 0.5,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict]:
    """
    Rank pages by relevance to query.

    Args:
        pages: Parsed page dicts (must have 'text' key).
        query: Search query string.
        top_k: Return only top K results. None returns all.
        mode: 'keyword', 'semantic', or 'both'.
        semantic_weight: Weight for semantic score in 'both' mode (0.0–1.0).
                         keyword_weight = 1 - semantic_weight.
        model_name: Sentence-transformers model to use for semantic scoring.

    Returns:
        Pages sorted by relevance_score descending.
    """
    if not pages:
        return []
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if mode not in ("keyword", "semantic", "both"):
        raise ValueError(f"Invalid mode '{mode}'. Choose from: keyword, semantic, both.")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weight must be between 0.0 and 1.0.")

    pages = [dict(p) for p in pages]  # avoid mutating caller's data

    if mode in ("keyword", "both"):
        pages = keyword_score(pages, query)
    if mode in ("semantic", "both"):
        pages = semantic_score(pages, query, model_name=model_name)

    kw = 1.0 - semantic_weight
    for page in pages:
        k = page.get("keyword_score", 0.0)
        s = page.get("semantic_score", 0.0)
        if mode == "both":
            page["relevance_score"] = round(kw * k + semantic_weight * s, 6)
        elif mode == "keyword":
            page["relevance_score"] = k
        else:
            page["relevance_score"] = s

    pages.sort(key=lambda x: x["relevance_score"], reverse=True)
    return pages[:top_k] if top_k else pages


# ── Helpers ───────────────────────────────────────────────────────────────────

def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


def _chunk_and_pool(text: str, model, chunk_size: int = 256) -> object:
    """
    Split long text into overlapping chunks, encode each, return max-pooled embedding.
    This ensures long documents are fully represented, not truncated.
    """
    from sentence_transformers import util
    words = text.split()
    if not words:
        return model.encode("", convert_to_tensor=True, show_progress_bar=False)

    step = chunk_size // 2  # 50% overlap
    chunks = [
        " ".join(words[i: i + chunk_size])
        for i in range(0, max(1, len(words) - chunk_size + 1), step)
    ]
    if not chunks:
        chunks = [text]

    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    # Max pooling across chunks — captures the most relevant chunk
    pooled, _ = embeddings.max(dim=0)
    return pooled.unsqueeze(0)
