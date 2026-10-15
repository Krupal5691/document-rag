PY ?= .venv/bin/python

.PHONY: setup ingest demo api eval eval-retrieval test lint

setup:            ## create the virtualenv and install dependencies
	python3 -m venv .venv && $(PY) -m pip install -r requirements.txt

ingest:           ## parse, chunk and index all documents (naive + advanced)
	$(PY) -m rag_core ingest --doc all

demo:             ## Streamlit demo on http://localhost:8501
	$(PY) -m streamlit run app.py

api:              ## REST API on http://localhost:8000/docs
	$(PY) -m uvicorn api:app --port 8000

eval:             ## full evaluation with answers (needs GEMINI_API_KEY)
	$(PY) -m rag_core eval --doc all --variants naive,advanced --judge

eval-retrieval:   ## retrieval-only evaluation with ablations (no API calls)
	$(PY) -m rag_core eval --doc all --no-answers \
		--variants naive,advanced-no-rewrite,advanced-no-rewrite-no-rerank,advanced-no-rewrite-no-hybrid

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check rag_core tests app.py api.py
