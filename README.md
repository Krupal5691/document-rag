# Document RAG: naive vs advanced, on three document shapes

Question answering over three PDFs that each break naive RAG in a different way, with a side-by-side
demo and an evaluation that measures what each technique adds.

| # | Task | Document | What breaks naive RAG |
|---|---|---|---|
| 1 | Baseline RAG | GPO 2018 Annual Report (82 pp) | Two-column layout, sentences cut by fixed-size chunks |
| 2 | Advanced document RAG | World Bank Global Economic Prospects, June 2026 (200 pp) | Length, similar regional sections, borderless tables, dropped ligatures |
| 3 | Multimodal / complex RAG | Apple iMac Product Environmental Report (12 pp) | Facts in charts, call-outs, small tables and footnotes |

**Stack:** Python · PyMuPDF · Qdrant (dense + BM25 hybrid, RRF) · fastembed (BM25, cross-encoder,
optional local embeddings) · Gemini (answers, query rewriting, vision, embeddings) · Streamlit.

## Results

Retrieval quality with local embeddings, no LLM in the loop (`python -m rag_core eval --no-answers`).
*hit@k* = an expected page is in the context sent to the LLM; *context recall* = share of the required
facts present in that context.

| Document | Naive hit@k | Advanced hit@k | Naive context recall | Advanced context recall |
|---|---|---|---|---|
| GPO | 56% | **100%** | 78% | **100%** |
| World Bank | 89% | **100%** | 89% | **100%** |
| iMac | 100% | 100% | 93% | **100%** |

Ablations (`advanced-no-rerank`, `advanced-no-hybrid`, ...) and answer accuracy with Gemini are in
[`eval/results/`](eval/results). Answer accuracy and the iMac vision step need a `GEMINI_API_KEY`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # add GEMINI_API_KEY (works without it: retrieval only, local embeddings)
python -m rag_core ingest          # parse, chunk and index all three documents
streamlit run app.py               # demo at http://localhost:8501
```

Other commands:

```bash
python -m rag_core ask --doc worldbank --trace "What is the growth outlook for South Asia?"
python -m rag_core eval --doc all --variants naive,advanced,advanced-no-rerank,advanced-no-hybrid
python -m rag_core eval --doc imac --judge          # adds LLM-judged faithfulness
python -m rag_core inspect --doc worldbank --page 98 # see how a page is parsed
pytest -q                                            # parsing / chunking / query tests (no key needed)
```

The local Qdrant database allows one process at a time: stop the Streamlit app before running CLI
commands, or set `QDRANT_URL` to a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`).

## The demo

- **Ask**: naive and advanced answers side by side with their sources, plus a step-by-step trace of
  the advanced pipeline (rewritten query and filters → dense → BM25 → RRF → cross-encoder rerank →
  final context).
- **Parsing inspector**: page image vs naive left-to-right text vs the parsed elements
  (reading order, headings, section paths, tables, footnotes).
- **Evaluation**: the latest results for the selected document.
- **Advanced pipeline steps** (sidebar): switch each technique on or off and see the effect.

## How it works

See [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). In short:

**Ingestion:** layout blocks → header/footer removal → XY-cut reading order → ligature and hyphen
repair → headings from font sizes + PDF outline → caption-anchored table reconstruction → footnote
linking → (iMac) vision transcription of charts and call-outs → parent sections + sentence-aware
child chunks with section/region metadata → Qdrant (dense + BM25).

**Retrieval:** rule + LLM query rewriting with region/chapter filters → hybrid search fused with
Reciprocal Rank Fusion → cross-encoder rerank → parent expansion (small-to-big) → follow
"see table 2.9" references → grounded answer with [n] citations.

## Project layout

```
rag_core/
  parsing/layout.py      blocks, superscripts, running headers, XY-cut reading order
  parsing/normalize.py   ligatures, hyphenation, vocabulary-based word repair
  parsing/tables.py      table reconstruction from word positions
  parsing/structure.py   headings, section paths, tables, figures, footnotes
  vision.py              Gemini page transcription for visual elements (cached)
  chunking.py            naive and structured (parent/child, sentence-aware) chunking
  embeddings.py          Gemini or local dense embeddings; BM25 sparse vectors
  store.py               Qdrant index: dense, BM25, hybrid RRF, filters, parent docstore
  query.py               query rewriting and metadata filters
  rerank.py              cross-encoder reranker
  generate.py            grounded answer prompt with citations
  pipeline.py            ingestion, NaiveRAG, AdvancedRAG (each step switchable)
  evaluation.py          hit@k, MRR, context recall, accuracy, refusal, faithfulness
  documents.py           per-document settings (regions, aliases, vision, chunk sizes)
eval/questions/          question sets with expected facts and pages (incl. unanswerable)
eval/results/            latest evaluation output
cache/vision/            vision transcriptions (committed, so the demo works without re-calling the API)
tests/                   pytest suite
app.py                   Streamlit demo
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Answers, query rewriting, vision, Gemini embeddings |
| `EMBED_PROVIDER` | `auto` | `gemini`, `local` (offline bge-small) or `auto` (gemini if key) |
| `GEN_MODEL` / `QUERY_MODEL` / `VISION_MODEL` | `gemini-2.5-flash` / `gemini-2.5-flash-lite` / `gemini-2.5-flash` | Models per job |
| `QDRANT_URL` | — | Use a Qdrant server instead of the embedded database |

Changing the embedding provider creates separate collections, so re-run `ingest` after switching.

## Limitations

- Table reconstruction targets caption-anchored, borderless tables; complex merged-cell tables may
  lose structure. Charts in the World Bank report are indexed by caption and notes, not values
  (vision on 200 pages is possible via `vision="all"` but costly on the free tier).
- Heading detection is heuristic (font size/weight); unusual layouts may need per-document tuning
  in `documents.py`.
- The evaluation sets are small (10 questions per document) and check facts by keyword.
