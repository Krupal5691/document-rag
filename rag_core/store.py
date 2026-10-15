"""Qdrant vector index with dense + sparse (BM25) vectors, hybrid search fused with RRF,
and a small JSON docstore for parent chunks."""
from __future__ import annotations

import atexit
import json
import uuid
from dataclasses import dataclass, field
from functools import lru_cache

from qdrant_client import QdrantClient, models

from . import config
from .chunking import Chunk
from .embeddings import DenseEmbedder, sparse

DENSE, SPARSE = "dense", "bm25"
FILTER_FIELDS = ("doc_id", "region", "chapter", "kind", "label")


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float
    rank: int
    scores: dict = field(default_factory=dict)      # per-stage scores for the trace


@lru_cache(maxsize=1)
def qdrant() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(config.INDEX_DIR / "qdrant"))
    atexit.register(client.close)
    return client


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _filter(filters: dict | None) -> models.Filter | None:
    if not filters:
        return None
    return models.Filter(must=[
        models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in filters.items() if v
    ])


class VectorIndex:
    def __init__(self, name: str, embedder: DenseEmbedder, use_sparse: bool = True):
        self.name = name
        self.embedder = embedder
        self.use_sparse = use_sparse
        self.parents_path = config.INDEX_DIR / "parents" / f"{name}.json"
        self._parents: dict | None = None

    # ------------------------------------------------------------------ build

    def exists(self) -> bool:
        db = qdrant()
        return db.collection_exists(self.name) and db.count(self.name).count > 0

    def build(self, chunks: list[Chunk], parents: dict[str, dict] | None = None, progress=print) -> None:
        db = qdrant()
        if db.collection_exists(self.name):
            db.delete_collection(self.name)
        db.create_collection(
            self.name,
            vectors_config={DENSE: models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)}
            if self.use_sparse else None,
        )
        texts = [c.embed_text for c in chunks]
        progress(f"Embedding {len(texts)} chunks with {self.embedder.name}...")
        dense_vecs = self.embedder.embed_documents(texts)
        sparse_vecs = sparse().embed_documents(texts) if self.use_sparse else [None] * len(texts)
        points = []
        for c, dv, sv in zip(chunks, dense_vecs, sparse_vecs, strict=True):
            vector = {DENSE: dv}
            if sv:
                vector[SPARSE] = models.SparseVector(indices=sv[0], values=sv[1])
            points.append(models.PointStruct(id=_point_id(c.id), vector=vector,
                                             payload={"chunk_id": c.id, "text": c.text,
                                                      "embed_text": c.embed_text, **c.metadata}))
        for start in range(0, len(points), 256):
            db.upsert(self.name, points[start:start + 256])
        if config.QDRANT_URL:          # payload indexes keep filtered search fast on a server (no-op locally)
            for field_name in FILTER_FIELDS:
                db.create_payload_index(self.name, field_name, field_schema=models.PayloadSchemaType.KEYWORD)
        self.parents_path.parent.mkdir(parents=True, exist_ok=True)
        self.parents_path.write_text(json.dumps(parents or {}))
        self._parents = parents or {}
        progress(f"Indexed {len(points)} chunks into '{self.name}'.")

    # ------------------------------------------------------------------ search

    def _hits(self, points, stage: str) -> list[Hit]:
        return [Hit(id=p.payload["chunk_id"], text=p.payload["text"], metadata=p.payload, score=getattr(p, "score", 1.0),
                    rank=i + 1, scores={stage: getattr(p, "score", 1.0)}) for i, p in enumerate(points)]

    def search_dense(self, query: str, k: int, filters: dict | None = None, qvec=None) -> list[Hit]:
        qvec = qvec or self.embedder.embed_query(query)
        res = qdrant().query_points(self.name, query=qvec, using=DENSE, limit=k, query_filter=_filter(filters),
                                    with_payload=True)
        return self._hits(res.points, "dense")

    def search_sparse(self, query: str, k: int, filters: dict | None = None) -> list[Hit]:
        idx, val = sparse().embed_query(query)
        res = qdrant().query_points(self.name, query=models.SparseVector(indices=idx, values=val), using=SPARSE,
                                    limit=k, query_filter=_filter(filters), with_payload=True)
        return self._hits(res.points, "bm25")

    def search_hybrid(self, query: str, k: int, filters: dict | None = None, prefetch_k: int = 30,
                      qvec=None) -> list[Hit]:
        """Dense and BM25 searches run inside Qdrant, then Reciprocal Rank Fusion merges the lists:
        score(doc) = sum over lists of 1 / (60 + rank). A chunk ranked well by both wins; a chunk
        found by only one list is pushed down."""
        qvec = qvec or self.embedder.embed_query(query)
        idx, val = sparse().embed_query(query)
        flt = _filter(filters)
        res = qdrant().query_points(
            self.name,
            prefetch=[
                models.Prefetch(query=qvec, using=DENSE, limit=prefetch_k, filter=flt),
                models.Prefetch(query=models.SparseVector(indices=idx, values=val), using=SPARSE,
                                limit=prefetch_k, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=k, with_payload=True,
        )
        return self._hits(res.points, "rrf")

    def facet_values(self, key: str) -> list[str]:
        values, offset = set(), None
        while True:
            pts, offset = qdrant().scroll(self.name, limit=512, offset=offset, with_payload=[key])
            values.update(p.payload.get(key) for p in pts if p.payload.get(key))
            if offset is None:
                return sorted(values)

    def by_label(self, label: str) -> list[Hit]:
        pts, _ = qdrant().scroll(self.name, limit=20, with_payload=True, scroll_filter=_filter({"label": label}))
        return self._hits(pts, "link") if pts else []

    # ------------------------------------------------------------------ parents

    @property
    def parents(self) -> dict:
        if self._parents is None:
            self._parents = json.loads(self.parents_path.read_text()) if self.parents_path.exists() else {}
        return self._parents
