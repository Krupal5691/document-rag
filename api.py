"""REST API for the RAG pipelines - the production entry point (the Streamlit app is the demo).

Run:  uvicorn api:app --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_core import citations, config, documents, embeddings, llm, pipeline
from rag_core.llm import QuotaExhaustedError
from rag_core.pipeline import Options

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("rag.api")
QUERY_LOG = config.ROOT / "logs" / "queries.jsonl"

app = FastAPI(title="Document RAG API", version="1.0.0",
              description="Grounded question answering over the GPO, World Bank and Apple iMac reports.")


class AskRequest(BaseModel):
    doc: str = Field(description="Document id: " + ", ".join(documents.DOCUMENTS))
    question: str = Field(min_length=3, max_length=1000)
    variant: str = Field("advanced", description="'naive', 'advanced' or an ablation like 'advanced-no-rerank'")
    top_k: int = Field(6, ge=1, le=12)
    include_trace: bool = False


class Evidence(BaseModel):
    page: int = Field(description="1-based PDF page where the supporting text was found")
    sentence: str = Field(description="The passage sentence that supports the cited claim")
    rects: list[list[float]] = Field(description="Highlight boxes [x0, y0, x1, y1] in PDF points")
    match: str = Field(description="cells | exact | partial | numbers | none")
    label_rects: list[list[float]] = Field(default_factory=list,
                                           description="Row-label boxes for table cells (match = cells)")


class Source(BaseModel):
    n: int
    pages: list[int]
    section: str
    label: str | None
    text: str
    cited: bool
    evidence: Evidence | None = None


class AskResponse(BaseModel):
    request_id: str
    answer: str | None
    sources: list[Source]
    timings: dict
    trace: dict | None = None


@lru_cache(maxsize=32)
def _rag(doc: str, variant: str, top_k: int):
    if variant == "naive":
        return pipeline.NaiveRAG(doc, top_k=min(top_k, 5))
    opts = Options.parse(variant)
    opts.top_k = top_k
    return pipeline.AdvancedRAG(doc, opts)


@app.get("/health")
def health():
    return {"status": "ok", "llm": llm.has_key(), "embeddings": embeddings.dense().name,
            "models": {"answer": config.GEN_MODEL, "query": config.QUERY_MODEL, "vision": config.VISION_MODEL}}


@app.get("/documents")
def list_documents():
    from rag_core.store import VectorIndex
    emb = embeddings.dense()
    return [{"id": d.id, "title": d.title, "description": d.description,
             "indexed": VectorIndex(pipeline._index_name(d.id, "advanced"), emb).exists()}
            for d in documents.DOCUMENTS.values()]


@app.get("/documents/{doc}/pages/{page}/highlight.pdf")
def highlighted_page(doc: str, page: int, rects: str = ""):
    """One PDF page with highlight boxes, e.g. ?rects=36,692,260,703;207,680,259,691 (from /ask evidence)."""
    from fastapi.responses import Response
    if doc not in documents.DOCUMENTS:
        raise HTTPException(404, f"Unknown document '{doc}'")
    try:
        boxes = [tuple(float(v) for v in r.split(",")) for r in rects.split(";") if r]
        assert all(len(b) == 4 for b in boxes)
    except (ValueError, AssertionError) as e:
        raise HTTPException(422, "rects must be 'x0,y0,x1,y1;...'") from e
    h = citations.Highlight(n=0, page=page, sentence="", rects=boxes)
    try:
        pdf = citations.page_pdf(documents.get(doc).path, h)
    except (IndexError, ValueError) as e:
        raise HTTPException(404, f"Page {page} not found") from e
    return Response(pdf, media_type="application/pdf")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.doc not in documents.DOCUMENTS:
        raise HTTPException(404, f"Unknown document '{req.doc}'")
    if req.variant != "naive" and not req.variant.startswith("advanced"):
        raise HTTPException(422, "variant must be 'naive' or start with 'advanced'")
    rid = uuid.uuid4().hex[:12]
    t = time.perf_counter()
    try:
        res = _rag(req.doc, req.variant, req.top_k).ask(req.question, generate_answer=llm.has_key())
    except QuotaExhaustedError as e:
        raise HTTPException(429, f"LLM quota exhausted: {e}") from e
    except Exception as e:                                   # never leak a stack trace to clients
        log.exception("ask failed rid=%s", rid)
        raise HTTPException(500, f"Internal error (request {rid})") from e
    total = time.perf_counter() - t

    hls = citations.locate(res.answer, res.contexts, documents.get(req.doc).path) if res.answer else {}
    sources = [Source(n=i, pages=[p for p in (c.get("pages") or [c.get("page")]) if p is not None],
                      section=" > ".join(c.get("section_path") or []), label=c.get("label"),
                      text=c["text"], cited=i in res.cited,
                      evidence=Evidence(page=hls[i].page, sentence=hls[i].sentence,
                                        rects=[list(r) for r in hls[i].rects], match=hls[i].match,
                                        label_rects=[list(r) for r in hls[i].label_rects])
                      if i in hls else None)
               for i, c in enumerate(res.contexts, 1)]
    timings = {k: round(v, 3) for k, v in res.timings.items()} | {"total": round(total, 3)}

    QUERY_LOG.parent.mkdir(exist_ok=True)
    with QUERY_LOG.open("a") as f:                           # observability: one line per request
        f.write(json.dumps({"rid": rid, "ts": time.time(), "doc": req.doc, "variant": req.variant,
                            "question": req.question, "filters": res.trace.get("plan", {}).get("applied_filters"),
                            "pages": [s.pages for s in sources], "cited": res.cited, "timings": timings}) + "\n")
    log.info("ask rid=%s doc=%s variant=%s total=%.2fs", rid, req.doc, req.variant, total)
    return AskResponse(request_id=rid, answer=res.answer or None, sources=sources, timings=timings,
                       trace=res.trace if req.include_trace else None)
