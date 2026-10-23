FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.models FASTEMBED_CACHE_PATH=/app/.models
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_core rag_core
COPY data data
COPY cache cache
COPY eval eval
COPY docs docs
COPY .streamlit .streamlit
COPY app.py api.py ./

# Download the local models at build time so containers start fast and work offline
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# Folders the app writes to must exist (and belong to the app user) before volumes are mounted on them
RUN mkdir -p /app/.index /app/logs /app/static/hl && useradd -m app && chown -R app /app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
