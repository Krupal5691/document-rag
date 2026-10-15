"""Cross-encoder reranking.

Embedding search is a bi-encoder: the question and each chunk are embedded separately and compared
by vector similarity - fast, but approximate. A cross-encoder reads the question and the chunk
TOGETHER through a transformer (attention across both), producing a direct relevance score. It is
too slow to run over every chunk, so we use it on the ~20-30 candidates from hybrid search and keep
the best few. Model: ms-marco-MiniLM-L-6 (runs locally on CPU via ONNX, no API needed).
"""
from __future__ import annotations

from functools import lru_cache

from . import config
from .store import Hit


@lru_cache(maxsize=1)
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(config.RERANK_MODEL)


def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    if not hits:
        return []
    scores = list(_model().rerank(query, [h.metadata.get("embed_text", h.text) for h in hits]))
    for h, s in zip(hits, scores, strict=True):
        h.scores["rerank"] = float(s)
    ranked = sorted(hits, key=lambda h: h.scores["rerank"], reverse=True)[:top_k]
    for i, h in enumerate(ranked):
        h.rank, h.score = i + 1, h.scores["rerank"]
    return ranked
