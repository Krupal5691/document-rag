# Document RAG: naive vs advanced, on three document shapes

Question answering over three PDFs that each break naive RAG in a different way, with a side-by-side
demo and an evaluation that measures what each technique adds.

| # | Task | Document | What breaks naive RAG |
|---|---|---|---|
| 1 | Baseline RAG | GPO 2018 Annual Report (82 pp) | Two-column layout, sentences cut by fixed-size chunks |
| 2 | Advanced document RAG | World Bank Global Economic Prospects, June 2026 (200 pp) | Length, similar regional sections, borderless tables, dropped ligatures |
| 3 | Multimodal / complex RAG | Apple iMac Product Environmental Report (12 pp) | Facts in charts, call-outs, small tables and footnotes |

**Stack:** Python · PyMuPDF · Qdrant (dense + BM25 hybrid, RRF) · fastembed (local embeddings, BM25,
cross-encoder reranker) · Gemini (answers, query rewriting, vision) · FastAPI · Streamlit · Docker.

## Results

Evaluated on 11 questions per document (the brief's sample questions, harder extras such as table
rows, footnotes and keyword-only queries, and one unanswerable question that must be refused).
Answers by Gemini, embeddings local (`bge-small`), same LLM and prompt for both pipelines.

| Document | Naive: correct answers | **Advanced: correct answers** | Naive: right page in context | Advanced: right page in context |
|---|---|---|---|---|
| GPO Annual Report | 91% | **100%** | 90% | **100%** |
| World Bank GEP | 73% | **100%** | 90% | **100%** |
| Apple iMac report | 82% | **100%** | 100% | **100%** |

Where naive fails: a keyword-only query ("procurement orders vendors"), the executive-summary risks
(wrong section retrieved), Bangladesh's FY2026/27 forecast (right page, but the table text is
scrambled), the iMac emissions breakdown that only exists in a chart.

**Ablation** (retrieval only, MRR = how high the first relevant passage ranks; 1.0 = always first):

| Variant | GPO MRR | World Bank MRR | iMac MRR |
|---|---|---|---|
| naive | 0.68 | 0.71 | 0.93 |
| advanced (all steps) | **0.95** | **0.90** | **1.00** |
| without cross-encoder rerank | 0.82 (and 10% of facts lost) | 0.79 | 1.00 |
| without query rewriting / filters | 0.95 | 0.88 | 0.93 |
| without hybrid BM25 | 0.95 | 0.90 | 1.00 |
| without parent expansion | 0.95 | 0.88 | 1.00 |

Reranking contributes most on this question set; hybrid search and parent expansion show no
measurable difference here (they matter more for exact-term and long-context questions than these 33
probe). Full per-question results, answers and the model used for each are in
[`eval/results/`](eval/results).

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

The same tasks are in the `Makefile` (`make ingest`, `make demo`, `make api`, `make eval`, `make test`).

The local Qdrant database allows one process at a time: stop the Streamlit app before running CLI
commands, or set `QDRANT_URL` to a Qdrant server (`docker run -p 6333:6333 qdrant/qdrant`).

## REST API

```bash
uvicorn api:app --port 8000          # interactive docs at http://localhost:8000/docs
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"doc": "worldbank", "question": "What is the growth outlook for South Asia?"}'
```

`GET /health`, `GET /documents`, `POST /ask` (answer, numbered sources with pages and section, and for
each cited source the evidence sentence, page and highlight boxes in PDF coordinates; per-stage
timings; optional trace), `GET /documents/{doc}/pages/{page}/highlight.pdf?rects=...` (the page with
those boxes highlighted). Inputs are validated, errors never expose stack traces, a
daily-quota failure returns HTTP 429, and every request is logged to `logs/queries.jsonl`.

## Docker

```bash
docker compose up -d qdrant
docker compose run --rm api python -m rag_core ingest
docker compose up -d api demo        # API on :8000, demo on :8501, Qdrant on :6333
```

The image downloads the local models at build time, runs as a non-root user and has a health check.

## The demo

- **Verifiable citations**: every [n] in an answer is a link that opens the source page as a PDF with
  the supporting sentence highlighted; each cited source also shows the highlighted page inline. The
  evidence sentence is matched to the claim (numbers weighted), located on the page with PyMuPDF
  (exact, then fuzzy 5-word windows, then the cited numbers for chart transcriptions).
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
  citations.py           evidence sentence per citation, located and highlighted in the PDF
  generate.py            grounded answer prompt with citations
  pipeline.py            ingestion, NaiveRAG, AdvancedRAG (each step switchable)
  evaluation.py          hit@k, MRR, context recall, accuracy, refusal, faithfulness
  documents.py           per-document settings (regions, aliases, vision, chunk sizes)
eval/questions/          question sets with expected facts and pages (incl. unanswerable)
eval/results/            latest evaluation output
cache/vision/            vision transcriptions (committed, so the demo works without re-calling the API)
tests/                   pytest suite
app.py                   Streamlit demo
api.py                   FastAPI service
Dockerfile, docker-compose.yml, Makefile, pyproject.toml (ruff + pytest config)
```

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Answers, query rewriting, vision, Gemini embeddings |
| `EMBED_PROVIDER` | `local` | `local` (offline bge-small, no quota) or `gemini` (free tier: ~1,000 chunk embeddings/day) |
| `GEN_MODEL` / `QUERY_MODEL` / `VISION_MODEL` | `gemini-3.5-flash` / `gemini-3.5-flash-lite` / `gemini-3.5-flash` | Models per job |
| `QDRANT_URL` | — | Use a Qdrant server instead of the embedded database |

| `FALLBACK_MODELS` | 8 flash models | Tried in order when a model is overloaded, retired or out of quota |

Changing the embedding provider creates separate collections, so re-run `ingest` after switching.

## Production notes

- **Quotas.** The Gemini free tier allows about 20 generation requests per model per day and ~1,000
  embedding requests per day (one per chunk). So embeddings run locally by default; answers rotate
  through `FALLBACK_MODELS` when a model's quota runs out; a daily-quota error stops immediately
  instead of retrying for hours; vision transcriptions, query plans and evaluation answers are cached
  on disk so reruns cost nothing. A paid tier removes these limits without code changes.
- **Resilience.** Retries with backoff on rate limits, server errors and dropped connections;
  `ingest --skip-existing` resumes a partial build; a too-strict metadata filter falls back to an
  unfiltered search; query rewriting failures fall back to rule-based queries.
- **Grounding and safety.** Answers only from retrieved passages with [n] citations, an explicit
  "not in the document" answer (tested with unanswerable questions), and document text is treated as
  data, never as instructions (prompt-injection guard). Secrets live only in `.env`, which is git-ignored.
- **Observability.** Every answer carries per-stage timings and a full trace (rewritten query,
  filters, dense / BM25 / fused / reranked lists, which model answered); the API logs each request.
- **Quality gates.** 26 pytest tests (parsing, chunking, query rules, API contract, LLM paths with a
  fake Gemini) run in GitHub Actions on every push; `ruff` lint; evaluation with ablations.
- **Scaling.** Point `QDRANT_URL` at a Qdrant server or cluster; add documents in `documents.py`; for
  many documents add payload indexes on `doc_id`/`region`/`chapter` and a queue for ingestion.

## Limitations

- Table reconstruction targets caption-anchored, borderless tables; complex merged-cell tables may
  lose structure. Charts in the World Bank report are indexed by caption and notes, not values
  (vision on 200 pages is possible via `vision="all"` but costly on the free tier).
- Heading detection is heuristic (font size/weight); unusual layouts may need per-document tuning
  in `documents.py`.
- The evaluation sets are small (10 questions per document) and check facts by keyword.
