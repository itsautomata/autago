"""
local embedding model for RAG memory retrieval.
loads once, embeds on demand, caches results.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
_cache = {}
_max_cache = 1000


def init(model_name="all-MiniLM-L6-v2", max_cache=1000):
    """load the embedding model. runs locally, no API calls.

    model options:
      - all-MiniLM-L6-v2: 80MB, fast, good enough for retrieval
      - BAAI/bge-small-en-v1.5: 130MB, better quality
      - BAAI/bge-large-en-v1.5: 1.3GB, original AgentNet (heavy)
    """
    global _model, _max_cache
    _max_cache = max_cache
    print(f"  loading embedding model: {model_name}...")
    _model = SentenceTransformer(model_name)
    print("  embedding model ready.")
    return _model


def embed(text):
    """embed a single text string. cached."""
    if text in _cache:
        return _cache[text]

    if _model is None:
        init()

    vec = _model.encode(text, normalize_embeddings=True)

    if len(_cache) >= _max_cache:
        oldest = next(iter(_cache))
        del _cache[oldest]

    _cache[text] = vec
    return vec


def similarity(text_a, text_b):
    """cosine similarity between two texts."""
    vec_a = embed(text_a)
    vec_b = embed(text_b)
    return float(np.dot(vec_a, vec_b))


def find_similar(query, candidates, top_k=3, threshold=0.7):
    """find the most similar candidates to a query.

    candidates: list of {"text": ..., ...} dicts
    returns: list of (candidate, score) tuples, sorted by score descending
    """
    query_vec = embed(query)
    scored = []

    for candidate in candidates:
        cand_vec = embed(candidate["text"])
        score = float(np.dot(query_vec, cand_vec))
        if score >= threshold:
            scored.append((candidate, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]
