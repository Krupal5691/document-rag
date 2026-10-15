"""Central configuration, read from environment / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"              # committed: parsed docs + vision transcriptions (saves API quota)
INDEX_DIR = ROOT / ".index"             # not committed: Qdrant local storage + parent docstore
EVAL_DIR = ROOT / "eval"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Models
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.5-flash")              # answers
QUERY_MODEL = os.getenv("QUERY_MODEL", "gemini-3.5-flash-lite")     # small, fast model for query rewriting
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.5-flash")        # charts / tables / call-outs
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash")          # evaluation judge
# Tried in order when a model is overloaded (503) or retired (404)
FALLBACK_MODELS = [m for m in os.getenv(
    "FALLBACK_MODELS",
    "gemini-flash-latest,gemini-3.7-flash,gemini-3-flash-preview,gemini-3.8-flash,"
    "gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-3.1-flash-lite,gemini-2.5-flash-lite",
).split(",") if m]

# Embeddings: "local" (fastembed, runs offline - default) or "gemini" (API). "auto" = gemini when a key
# exists. Local is the default because the Gemini free tier allows ~1,000 embedding requests per day and
# counts every chunk as one request (this corpus has ~7,000 chunks); it also keeps document text local.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# Vector database: embedded (local folder) by default; set QDRANT_URL to use a Qdrant server.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def embed_provider() -> str:
    if EMBED_PROVIDER == "auto":
        return "gemini" if GEMINI_API_KEY else "local"
    return EMBED_PROVIDER
