"""Ingestion and the two RAG pipelines.

NaiveRAG     plain text -> fixed 500-char chunks -> dense search top-5 -> answer
AdvancedRAG  layout-aware parsing -> structured parent/child chunks (+ vision for visual pages)
             -> query rewriting + metadata filters -> hybrid dense+BM25 search fused with RRF
             -> cross-encoder rerank -> small-to-big parent expansion + linked tables/figures
             -> grounded answer with citations

Both use the same LLM and answer prompt, so differences in quality come from retrieval.
Every advanced step can be switched off (Options) to measure its contribution.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from . import documents, embeddings, generate, llm, query, rerank, vision
from .chunking import naive_chunks, structured_chunks
from .documents import DocConfig
from .parsing.structure import ParsedDocument, parse_pdf
from .store import Hit, VectorIndex

CONTEXT_BUDGET = 16000      # characters of context sent to the LLM
PARENT_MAX = 4000           # a single parent is trimmed around its matching child beyond this


@dataclass
class Options:
    rewrite: bool = True        # query rewriting with a small LLM (+ rule-based filters)
    filters: bool = True        # apply metadata filters (region / chapter)
    hybrid: bool = True         # dense + BM25 with RRF; False = dense only
    rerank: bool = True         # cross-encoder rerank of the candidates
    parent: bool = True         # small-to-big: send the parent section instead of the child
    links: bool = True          # add tables/figures the retrieved text refers to
    candidates: int = 30
    top_k: int = 6

    @classmethod
    def parse(cls, spec: str) -> "Options":
        """'advanced' or 'advanced-no-rerank-no-hybrid' style names used by the evaluator."""
        opts = cls()
        for part in spec.split("-no-")[1:]:
            setattr(opts, part.replace("-", "_"), False)
        return opts


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[dict]
    cited: list[int] = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _index_name(doc_id: str, variant: str) -> str:
    return f"{doc_id}-{variant}-{embeddings.dense().name}"


# ------------------------------------------------------------------------------------ ingest

def parse_with_vision(cfg: DocConfig, progress=print) -> tuple[ParsedDocument, dict]:
    parsed = parse_pdf(cfg.path, cfg.id)
    status = {}
    if cfg.vision == "all":
        import pymupdf
        doc = pymupdf.open(cfg.path)
        by_page: dict[int, list] = {}
        for el in parsed.elements:
            by_page.setdefault(el.page, []).append(el)
        merged = []
        for page in doc:
            page_no = page.number + 1
            own = by_page.get(page_no, [])
            path = own[-1].section_path if own else (merged[-1].section_path if merged else [])
            extra, st = vision.page_elements(cfg, page, path)
            status[st] = status.get(st, 0) + 1
            merged.extend(own + extra)
        parsed.elements = merged
        parsed.stats["vision_pages"] = status
        progress(f"Vision: {status}")
        if status.get("skipped"):
            progress("Vision skipped for some pages (no GEMINI_API_KEY and no cached transcription).")
    return parsed, status


def ingest(doc_id: str, variants=("naive", "advanced"), progress=print) -> dict:
    cfg = documents.get(doc_id)
    embedder = embeddings.dense()
    stats = {"doc": doc_id, "embedder": embedder.name}
    if "naive" in variants:
        chunks = naive_chunks(cfg)
        VectorIndex(_index_name(doc_id, "naive"), embedder, use_sparse=False).build(chunks, progress=progress)
        stats["naive_chunks"] = len(chunks)
    if "advanced" in variants:
        parsed, _ = parse_with_vision(cfg, progress)
        children, parents = structured_chunks(cfg, parsed)
        VectorIndex(_index_name(doc_id, "advanced"), embedder).build(children, parents, progress=progress)
        stats.update(parse=parsed.stats, children=len(children), parents=len(parents))
    return stats


# ------------------------------------------------------------------------------------ naive

class NaiveRAG:
    name = "naive"

    def __init__(self, doc_id: str, top_k: int = 5):
        self.cfg = documents.get(doc_id)
        self.index = VectorIndex(_index_name(doc_id, "naive"), embeddings.dense(), use_sparse=False)
        self.top_k = top_k

    def retrieve(self, question: str) -> tuple[list[dict], dict, dict]:
        t = time.perf_counter()
        hits = self.index.search_dense(question, self.top_k)
        contexts = [{"text": h.text, "page": h.metadata["page"], "pages": [h.metadata["page"]],
                     "section_path": [], "chunk_id": h.id, "score": h.score} for h in hits]
        return contexts, {"dense": _brief(hits)}, {"retrieve": time.perf_counter() - t}

    def ask(self, question: str, generate_answer: bool = True) -> RAGResult:
        contexts, trace, timings = self.retrieve(question)
        result = RAGResult(question, "", contexts, trace=trace, timings=timings)
        if generate_answer:
            t = time.perf_counter()
            gen = generate.answer(question, contexts)
            result.answer, result.cited = gen.text, gen.cited
            result.timings["generate"] = time.perf_counter() - t
        return result


# --------------------------------------------------------------------------------- advanced

def _brief(hits: list[Hit], n: int = 10) -> list[dict]:
    return [{"rank": h.rank, "page": h.metadata.get("page"), "kind": h.metadata.get("kind"),
             "section": h.metadata.get("section"), "label": h.metadata.get("label"),
             "scores": {k: round(v, 4) for k, v in h.scores.items()}, "text": h.text[:220]} for h in hits[:n]]


def _trim(text: str, anchor: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    pos = max(text.find(anchor[:60]), 0)
    start = max(0, min(pos - limit // 3, len(text) - limit))
    return ("..." if start else "") + text[start:start + limit] + ("..." if start + limit < len(text) else "")


class AdvancedRAG:
    name = "advanced"

    def __init__(self, doc_id: str, options: Options | None = None):
        self.cfg = documents.get(doc_id)
        self.index = VectorIndex(_index_name(doc_id, "advanced"), embeddings.dense())
        self.options = options or Options()
        self._chapters: list[str] | None = None

    @property
    def chapters(self) -> list[str]:
        if self._chapters is None:
            self._chapters = self.index.facet_values("chapter")
        return self._chapters

    def retrieve(self, question: str, trace_all: bool = True) -> tuple[list[dict], dict, dict]:
        o, timings, trace = self.options, {}, {}

        t = time.perf_counter()
        plan = query.plan(question, self.cfg, self.chapters, use_llm=o.rewrite)
        search_q = plan.search_query if o.rewrite else question
        filters = plan.filters if o.filters else {}
        trace["plan"] = asdict(plan) | {"applied_filters": filters, "search_query_used": search_q}
        timings["plan"] = time.perf_counter() - t

        t = time.perf_counter()
        qvec = self.index.embedder.embed_query(search_q)
        if trace_all:
            trace["dense"] = _brief(self.index.search_dense(search_q, 10, filters, qvec=qvec))
            if o.hybrid:
                trace["bm25"] = _brief(self.index.search_sparse(search_q, 10, filters))

        def search(f):
            if o.hybrid:
                return self.index.search_hybrid(search_q, o.candidates, f, qvec=qvec)
            return self.index.search_dense(search_q, o.candidates, f, qvec=qvec)

        candidates = search(filters)
        if filters and len(candidates) < 3:                 # filter too strict: fall back to no filter
            trace["filter_fallback"] = f"only {len(candidates)} results with {filters}; searched without filters"
            candidates = search({})
        trace["fused" if o.hybrid else "candidates"] = _brief(candidates)
        timings["search"] = time.perf_counter() - t

        t = time.perf_counter()
        if o.rerank:
            # the cross-encoder judges relevance to what the user actually asked
            top = rerank.rerank(question, candidates, o.top_k)
            trace["reranked"] = _brief(top)
        else:
            top = candidates[:o.top_k]
        timings["rerank"] = time.perf_counter() - t

        t = time.perf_counter()
        contexts = self._build_context(top, trace)
        timings["context"] = time.perf_counter() - t
        return contexts, trace, timings

    def _build_context(self, top: list[Hit], trace: dict) -> list[dict]:
        o, parents = self.options, self.index.parents
        contexts, seen, used = [], set(), 0

        def add(text, meta, origin, hit):
            nonlocal used
            if used + len(text) > CONTEXT_BUDGET and contexts:
                return
            contexts.append({"text": text, "page": meta.get("page"), "pages": meta.get("pages") or [meta.get("page")],
                             "section_path": meta.get("section_path"), "label": meta.get("label"),
                             "kind": meta.get("kind"), "origin": origin, "chunk_id": hit.id,
                             "score": round(hit.score, 4)})
            used += len(text)

        for h in top:
            pid = h.metadata.get("parent_id")
            if o.parent and pid in parents:
                if pid in seen:
                    continue
                seen.add(pid)
                p = parents[pid]
                text = _trim(p["text"], h.text, PARENT_MAX)
                add(text, {**h.metadata, "pages": p["pages"], "page": p["page"]}, "parent", h)
            else:
                add(h.text, h.metadata, "child", h)

        if o.links:
            linked = []
            have = {c.get("label") for c in contexts}
            for h in top[:3]:
                for ref in h.metadata.get("refs", [])[:2]:
                    if ref in have or len(linked) >= 2:
                        continue
                    found = self.index.by_label(ref)
                    if found:
                        pid = found[0].metadata.get("parent_id")
                        p = parents.get(pid)
                        text = p["text"] if p else found[0].text
                        add(_trim(text, "", PARENT_MAX), found[0].metadata, f"linked from '{ref}'", found[0])
                        have.add(ref)
                        linked.append(ref)
            if linked:
                trace["linked"] = linked
        return contexts

    def ask(self, question: str, generate_answer: bool = True) -> RAGResult:
        contexts, trace, timings = self.retrieve(question)
        result = RAGResult(question, "", contexts, trace=trace, timings=timings)
        if generate_answer:
            t = time.perf_counter()
            gen = generate.answer(question, contexts)
            result.answer, result.cited = gen.text, gen.cited
            result.timings["generate"] = time.perf_counter() - t
        return result


def build(doc_id: str, variant: str):
    """'naive', 'advanced', or an ablation like 'advanced-no-rerank'."""
    if variant == "naive":
        return NaiveRAG(doc_id)
    return AdvancedRAG(doc_id, Options.parse(variant))


def can_generate() -> bool:
    return llm.has_key()
