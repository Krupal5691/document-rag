# How it works

Every RAG pipeline has three stages: **ingest** (read and chunk documents, store them in a vector
database), **retrieve** (find the right chunks for a question) and **generate** (answer from them).
RAG usually fails in the first two stages, even when it looks like the LLM is "hallucinating".
This project solves one failure mode per document, then compares the result against a naive baseline.

## Where naive RAG breaks

| Document | What breaks | Fix in this project |
|---|---|---|
| GPO Annual Report (82 pp, two-column text) | Left-to-right reading mixes the two columns, so sentences from different paragraphs are glued together. Fixed 500-character chunks cut sentences in half. | Column-aware reading order (XY-cut), header/footer removal, sentence-aware chunks |
| World Bank GEP (200 pp, chapters/regions/tables) | Six regional sections all talk about "growth", so search returns the wrong region. Borderless tables lose their row labels. The "Th" ligature is dropped ("Te global economy"). | Heading hierarchy from font sizes + PDF outline, region/chapter metadata and filters, table reconstruction, ligature repair, hybrid search, reranking, parent expansion |
| Apple iMac report (12 pp, visual) | Facts live in charts, icon call-outs, small tables and numbered footnotes, not in paragraphs. | Vision model transcribes each page's visuals; superscript footnote markers are linked to the endnotes and attached to chunks |

## Ingestion (advanced)

1. **Layout extraction** (`parsing/layout.py`): text blocks with position and font information.
   Superscript digits become footnote markers `[^18]`; subscripts (CO₂) are left alone.
2. **Running text removal**: lines repeated in the page margins on many pages (headers, footers,
   page numbers) are dropped.
3. **Reading order**: recursive **XY-cut** – split the page at vertical gutters first (columns), then
   at horizontal gaps; full-width titles block the vertical cut, so they are separated first.
4. **Normalisation** (`parsing/normalize.py`): NFKC turns ligature glyphs (ﬁ, ﬀ) into letters,
   hyphenated line breaks are re-joined, and **dictionary-based ligature repair** uses the document's
   own vocabulary to fix words broken by dropped glyphs ("signi cant" → "significant",
   "Te" → "The" when "the" is common and "te" never appears in lower case).
5. **Structure** (`parsing/structure.py`): headings are detected from font size, weight and family
   relative to the page's body text; the PDF outline (table of contents) gives chapter names. Every
   element gets a **section path**, e.g. `Ch. 2 Regional Outlooks > South Asia > Outlook`.
6. **Tables** (`parsing/tables.py`): a `TABLE x.y` caption anchors a region; the grid is rebuilt from
   word positions (rows by baseline, header row by year-like tokens) and written both as a Markdown
   table and as one sentence per row ("India: 2026/27f = 6.6; ...").
7. **Figures**: caption + panel titles + source/notes (chart tick labels are dropped as noise).
8. **Footnotes**: endnote sections and page footnotes are collected; each chunk gets the text of the
   footnotes its markers point to.
9. **Vision** (`vision.py`, iMac): each page is rendered and sent to Gemini, which transcribes every
   chart, table and call-out with all numbers. Cached in `cache/vision/` so the API is called once.
10. **Chunking** (`chunking.py`): **parents** = a section's paragraphs (≤ 2,400 chars) or one whole
    table/figure; **children** = sentence-aware pieces (≤ 450 chars, never cutting a sentence,
    one-sentence overlap). Children are embedded with a context header
    (`[document > section path | label]`) and carry metadata: page, section, chapter, region, kind,
    label, references to tables/figures, footnotes.
11. **Indexing** (`store.py`): Qdrant with two vectors per chunk: a **dense** embedding
    (Gemini or local bge-small) and a **sparse BM25** vector.

## Retrieval (advanced)

1. **Query understanding** (`query.py`): rules detect region names, abbreviations (SAR → South Asia)
   and chapter references; a small LLM (`gemini-2.5-flash-lite`) rewrites the question into a
   keyword-rich search query and expands keyword-only questions. Filter values are validated against
   real values in the index.
2. **Hybrid search**: dense and BM25 searches run in Qdrant (with metadata filters) and are merged
   with **Reciprocal Rank Fusion**: `score = Σ 1 / (60 + rank)`. If a filter leaves fewer than 3
   results, the search is repeated without it.
3. **Cross-encoder rerank** (`rerank.py`): the top 30 candidates are rescored by a cross-encoder that
   reads question and chunk together (attention across both), and the best 6 are kept.
4. **Small-to-big**: each child is replaced by its parent section (de-duplicated, trimmed around
   the match), giving the LLM full context.
5. **Reference linking**: if the text says "(table 2.9)", that table is added to the context.

## Generation

One prompt for both pipelines: answer only from the numbered passages, copy numbers exactly with
units and periods, cite passages as [n], keep regions/years apart, and reply
"The document does not contain this information." when it doesn't.

## Naive baseline

Page text read top-to-bottom, left-to-right → 500-character chunks (50 overlap) → dense search
top-5 → the same answer prompt. Same LLM, same embeddings: differences come from retrieval.
