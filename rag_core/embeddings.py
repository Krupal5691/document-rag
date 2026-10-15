"""Dense (meaning) and sparse (keyword/BM25) embedders.

Dense: Gemini embeddings via API, or a local open-source model (bge-small) via fastembed when no
key is available or data must stay on the machine. Sparse: BM25 term weights from fastembed,
stored in Qdrant as a sparse vector; Qdrant applies the IDF part at query time.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import config, llm


class DenseEmbedder:
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder(DenseEmbedder):
    def __init__(self):
        self.dim = config.GEMINI_EMBED_DIM
        self.name = f"gemini{self.dim}"

    def embed_documents(self, texts):
        return [_unit(v) for v in llm.embed(texts, "RETRIEVAL_DOCUMENT")]

    def embed_query(self, text):
        return _unit(llm.embed([text], "RETRIEVAL_QUERY")[0])


class LocalEmbedder(DenseEmbedder):
    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(config.LOCAL_EMBED_MODEL)
        self.dim = self.model.embedding_size
        self.name = "local-" + config.LOCAL_EMBED_MODEL.split("/")[-1].lower()

    def embed_documents(self, texts):
        return [v.tolist() for v in self.model.embed(texts, batch_size=32)]

    def embed_query(self, text):
        return next(iter(self.model.query_embed(text))).tolist()


def _unit(v: list[float]) -> list[float]:
    # Gemini vectors with reduced dimensionality are not normalised; cosine needs unit length.
    a = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return (a / n).tolist() if n else a.tolist()


@lru_cache(maxsize=2)
def dense(provider: str | None = None) -> DenseEmbedder:
    provider = provider or config.embed_provider()
    return GeminiEmbedder() if provider == "gemini" else LocalEmbedder()


class SparseEmbedder:
    def __init__(self):
        from fastembed import SparseTextEmbedding
        self.model = SparseTextEmbedding(config.SPARSE_MODEL)

    def embed_documents(self, texts: list[str]):
        return [(e.indices.tolist(), e.values.tolist()) for e in self.model.embed(texts, batch_size=64)]

    def embed_query(self, text: str):
        e = next(iter(self.model.query_embed(text)))
        return e.indices.tolist(), e.values.tolist()


@lru_cache(maxsize=1)
def sparse() -> SparseEmbedder:
    return SparseEmbedder()
