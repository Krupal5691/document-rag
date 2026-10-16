"""Streamlit demo: naive vs advanced RAG over three differently shaped documents.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import warnings

import pandas as pd
import pymupdf
import streamlit as st

from rag_core import citations, config, documents, embeddings, llm, pipeline
from rag_core.pipeline import Options

warnings.filterwarnings("ignore")
logging.getLogger("pymupdf").setLevel(logging.ERROR)

st.set_page_config(page_title="Document RAG: naive vs advanced", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------- cached objects

@st.cache_resource(show_spinner=False)
def get_rag(doc_id: str, variant: str, opts_json: str = ""):
    if variant == "naive":
        return pipeline.NaiveRAG(doc_id)
    return pipeline.AdvancedRAG(doc_id, Options(**json.loads(opts_json)) if opts_json else Options())


@st.cache_data(show_spinner=False)
def parsed_doc(doc_id: str):
    cfg = documents.get(doc_id)
    parsed, _ = pipeline.parse_with_vision(cfg, progress=lambda m: None)
    return parsed.to_dict()


def index_ready(doc_id: str) -> bool:
    from rag_core.store import VectorIndex
    emb = embeddings.dense()
    return all(VectorIndex(pipeline._index_name(doc_id, v), emb).exists() for v in ("naive", "advanced"))


def questions(doc_id: str) -> list[dict]:
    path = config.EVAL_DIR / "questions" / f"{doc_id}.json"
    return json.loads(path.read_text()) if path.exists() else []


# ----------------------------------------------------------------------------- sidebar

st.sidebar.title("📄 Document RAG")
doc_id = st.sidebar.radio("Document", list(documents.DOCUMENTS), format_func=lambda d: documents.get(d).title)
cfg = documents.get(doc_id)

has_key = llm.has_key()
st.sidebar.caption(
    f"**LLM:** {'`' + config.GEN_MODEL + '`' if has_key else '⚠️ no GEMINI_API_KEY — retrieval only'}  \n"
    f"**Query rewriting:** `{config.QUERY_MODEL if has_key else 'rules only'}`  \n"
    f"**Embeddings:** `{embeddings.dense().name}`  \n"
    f"**Keyword index:** `{config.SPARSE_MODEL}` · **Reranker:** `{config.RERANK_MODEL.split('/')[-1]}`  \n"
    f"**Vector DB:** Qdrant ({'server' if config.QDRANT_URL else 'embedded'})"
)

with st.sidebar.expander("Advanced pipeline steps", expanded=False):
    opts = Options(
        rewrite=st.checkbox("Query rewriting (LLM)", True, help="Small LLM rewrites the question into a search query"),
        filters=st.checkbox("Metadata filters", True, help="Region / chapter filters detected from the question"),
        hybrid=st.checkbox("Hybrid search (dense + BM25, RRF)", True),
        rerank=st.checkbox("Cross-encoder rerank", True),
        parent=st.checkbox("Parent expansion (small-to-big)", True),
        links=st.checkbox("Follow table/figure references", True),
        top_k=st.slider("Passages sent to the LLM", 3, 10, 6),
    )

if not index_ready(doc_id):
    st.sidebar.warning("Index not built yet for this document.")
if st.sidebar.button("Build / rebuild index", use_container_width=True):
    with st.spinner(f"Parsing, chunking and embedding {cfg.title}..."):
        log = st.empty()
        stats = pipeline.ingest(doc_id, progress=lambda m: log.info(m))
        st.cache_resource.clear()
    st.sidebar.success(f"Indexed: {stats.get('children')} chunks ({stats.get('naive_chunks')} naive)")

st.title(cfg.title)
st.caption(cfg.description)

tab_ask, tab_parse, tab_eval, tab_how = st.tabs(["💬 Ask", "🔬 Parsing inspector", "📊 Evaluation", "🧭 How it works"])


# ----------------------------------------------------------------------------- ask

HL_DIR = config.ROOT / "static" / "hl"


@st.cache_data(show_spinner=False, max_entries=200)
def cached_ask(doc_id: str, variant: str, opts_json: str, question: str, with_answer: bool) -> dict:
    """Streamlit re-runs the whole script on every click; caching the result means the LLM is called
    once per question, not once per interaction."""
    return get_rag(doc_id, variant, opts_json).ask(question, generate_answer=with_answer).to_dict()


@st.cache_data(show_spinner=False, max_entries=200)
def evidence(doc_id: str, answer: str, contexts_json: str) -> dict[int, dict]:
    hl = citations.locate(answer, json.loads(contexts_json), documents.get(doc_id).path)
    return {n: h.__dict__ for n, h in hl.items()}


@st.cache_data(show_spinner=False, max_entries=200)
def page_image(doc_id: str, h: dict) -> bytes:
    return citations.page_png(documents.get(doc_id).path, citations.Highlight(**h))


def pdf_link(doc_id: str, h: dict) -> str:
    """Write the highlighted page once and return its static URL (served by Streamlit)."""
    hl = citations.Highlight(**h)
    name = f"{doc_id}-p{hl.page}-{hl.key}.pdf"
    path = HL_DIR / name
    if not path.exists():
        HL_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(citations.page_pdf(documents.get(doc_id).path, hl))
    return f"app/static/hl/{name}"


def show_answer(doc_id: str, res: dict) -> dict[int, dict]:
    """Answer with every [n] turned into a link to the highlighted PDF page."""
    if not res["answer"]:
        return {}
    hls = evidence(doc_id, res["answer"], json.dumps(res["contexts"]))
    urls = {n: pdf_link(doc_id, h) for n, h in hls.items()}
    titles = {n: f"Open page {h['page']} with the evidence highlighted" for n, h in hls.items()}
    st.markdown(citations.link_citations(res["answer"], urls, titles))
    st.caption("Click a citation like [1] to open the source page with the supporting text highlighted.")
    return hls


def show_contexts(doc_id: str, contexts: list[dict], cited: list[int], hls: dict[int, dict] | None = None):
    hls = hls or {}
    for i, c in enumerate(contexts, 1):
        where = " > ".join(c.get("section_path") or []) or "—"
        tag = " ✅ cited" if i in cited else ""
        origin = f" · {c['origin']}" if c.get("origin") else ""
        with st.expander(f"[{i}] p.{', '.join(map(str, c.get('pages') or [c.get('page')]))} · "
                         f"{c.get('label') or c.get('kind') or ''}{origin}{tag}"):
            st.caption(where)
            h = hls.get(i)
            if h:
                found = {"exact": "exact sentence found", "partial": "sentence found (approximate match)",
                         "numbers": "cited numbers found", "none": "page only - text not matched"}[h["match"]]
                st.markdown(f"**Evidence** (page {h['page']}, {found}):  \n> {h['sentence']}")
                st.markdown(f"[Open highlighted page ↗]({pdf_link(doc_id, h)})")
                st.image(page_image(doc_id, h), use_container_width=True)
                st.markdown("**Full passage**")
            st.markdown(c["text"])


def hits_table(rows: list[dict]):
    if not rows:
        st.caption("—")
        return
    df = pd.DataFrame([{"rank": r["rank"], "page": r["page"], "section": r.get("section"), "kind": r.get("kind"),
                        **{k: v for k, v in r["scores"].items()}, "text": r["text"][:140]} for r in rows])
    st.dataframe(df, hide_index=True, use_container_width=True)


with tab_ask:
    if not index_ready(doc_id):
        st.info("Build the index from the sidebar first (takes ~10 s for iMac, ~4 min for the World Bank report "
                "with local embeddings).")
    else:
        qs = [q for q in questions(doc_id)]
        c1, c2 = st.columns([3, 1])
        sample = c1.selectbox("Sample questions", ["— type your own below —"] + [q["question"] for q in qs])
        mode = c2.radio("Mode", ["Compare", "Advanced only"], horizontal=False)
        question = st.text_input("Question", value="" if sample.startswith("—") else sample)

        if question:
            opts_json = json.dumps(opts.__dict__)
            cols = st.columns(2) if mode == "Compare" else [None, st.container()]
            if mode == "Compare":
                with cols[0]:
                    st.subheader("Naive RAG")
                    st.caption("page text · 500-char chunks · dense top-5")
                    with st.spinner("Naive..."):
                        nres = cached_ask(doc_id, "naive", "", question, has_key)
                    show_contexts(doc_id, nres["contexts"], nres["cited"], show_answer(doc_id, nres))
            with cols[1]:
                st.subheader("Advanced RAG")
                st.caption("layout parsing · structured chunks · rewrite · hybrid+RRF · rerank · parent expansion")
                with st.spinner("Advanced..."):
                    ares = cached_ask(doc_id, "advanced", opts_json, question, has_key)
                show_contexts(doc_id, ares["contexts"], ares["cited"], show_answer(doc_id, ares))

            st.divider()
            st.subheader("How the advanced pipeline got there")
            t = ares["trace"]
            plan = t.get("plan", {})
            s1, s2, s3 = st.columns(3)
            s1.markdown(f"**Original question**  \n{plan.get('original')}")
            s2.markdown(f"**Search query used** ({plan.get('source')})  \n`{plan.get('search_query_used')}`")
            s3.markdown(f"**Filters**  \n`{plan.get('applied_filters') or 'none'}`")
            if t.get("filter_fallback"):
                st.warning(t["filter_fallback"])
            steps = [("1 · Dense (semantic) search", "dense"), ("2 · BM25 keyword search", "bm25"),
                     ("3 · Reciprocal Rank Fusion", "fused"), ("3 · Dense candidates", "candidates"),
                     ("4 · Cross-encoder rerank", "reranked")]
            for title, key in steps:
                if key in t:
                    with st.expander(title, expanded=key == "reranked"):
                        hits_table(t[key])
            if t.get("linked"):
                st.info(f"Followed references to: {', '.join(t['linked'])}")
            st.caption("Timings (s): " + ", ".join(f"{k} {v:.2f}" for k, v in ares["timings"].items())
                       + (f" · answered by `{t['answer_model']}`" if t.get("answer_model") else ""))


# ----------------------------------------------------------------------------- parsing inspector

with tab_parse:
    st.markdown("Compare what a **naive text extractor** sees with what our **layout-aware parser** produces.")
    doc = pymupdf.open(cfg.path)
    page_no = st.number_input("Page", 1, doc.page_count, value={"gpo": 4, "worldbank": 98, "imac": 10}[doc_id])
    data = parsed_doc(doc_id)
    st.caption(f"Parse stats: {data['stats']}")
    a, b, c = st.columns([1, 1, 1])
    with a:
        st.markdown("**Page image**")
        st.image(doc[page_no - 1].get_pixmap(dpi=90).tobytes("png"))
    with b:
        st.markdown("**Naive: top-to-bottom, left-to-right text**")
        st.code(doc[page_no - 1].get_text(sort=True)[:5000], language=None)
    with c:
        st.markdown("**Parsed elements (reading order)**")
        for e in [e for e in data["elements"] if e["page"] == page_no]:
            badge = e["kind"] + (f" L{e['level']}" if e.get("level") else "") + (f" · {e['label']}" if e.get("label") else "")
            st.markdown(f"`{badge}` <small>{' > '.join(e['section_path'])}</small>", unsafe_allow_html=True)
            st.text(e["text"][:1200])


# ----------------------------------------------------------------------------- evaluation

with tab_eval:
    res_md = config.EVAL_DIR / "results" / f"{doc_id}.md"
    if res_md.exists():
        st.markdown(res_md.read_text())
    else:
        st.info("No evaluation results yet. Run `python -m rag_core eval --doc " + doc_id + "`.")
    st.caption("Metrics: hit@k = an expected page is in the context · MRR = 1/rank of first relevant passage · "
               "context recall = required facts present in the context · accuracy = required facts in the answer "
               "(unanswerable questions must be refused).")


# ----------------------------------------------------------------------------- how it works

with tab_how:
    st.markdown((config.ROOT / "docs" / "HOW_IT_WORKS.md").read_text()
                if (config.ROOT / "docs" / "HOW_IT_WORKS.md").exists() else "See README.md")
