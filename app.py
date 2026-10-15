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

from rag_core import config, documents, embeddings, llm, pipeline
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

def show_contexts(contexts: list[dict], cited: list[int]):
    for i, c in enumerate(contexts, 1):
        where = " > ".join(c.get("section_path") or []) or "—"
        tag = " ✅ cited" if i in cited else ""
        origin = f" · {c['origin']}" if c.get("origin") else ""
        with st.expander(f"[{i}] p.{', '.join(map(str, c.get('pages') or [c.get('page')]))} · "
                         f"{c.get('label') or c.get('kind') or ''}{origin}{tag}"):
            st.caption(where)
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
            adv = get_rag(doc_id, "advanced", json.dumps(opts.__dict__))
            cols = st.columns(2) if mode == "Compare" else [None, st.container()]
            if mode == "Compare":
                with cols[0]:
                    st.subheader("Naive RAG")
                    st.caption("page text · 500-char chunks · dense top-5")
                    with st.spinner("Naive..."):
                        nres = get_rag(doc_id, "naive").ask(question, generate_answer=has_key)
                    if has_key:
                        st.markdown(nres.answer)
                    show_contexts(nres.contexts, nres.cited)
            with cols[1]:
                st.subheader("Advanced RAG")
                st.caption("layout parsing · structured chunks · rewrite · hybrid+RRF · rerank · parent expansion")
                with st.spinner("Advanced..."):
                    ares = adv.ask(question, generate_answer=has_key)
                if has_key:
                    st.markdown(ares.answer)
                show_contexts(ares.contexts, ares.cited)

            st.divider()
            st.subheader("How the advanced pipeline got there")
            t = ares.trace
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
            st.caption("Timings (s): " + ", ".join(f"{k} {v:.2f}" for k, v in ares.timings.items()))


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
