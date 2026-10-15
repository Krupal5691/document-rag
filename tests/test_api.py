"""REST API contract tests (fake Gemini, real local index)."""
import pytest
from fastapi.testclient import TestClient

from rag_core import config, embeddings, llm

needs_index = pytest.mark.skipif(
    not (config.INDEX_DIR / "parents" / "gpo-advanced-local-bge-small-en-v1.5.json").exists(),
    reason="run `python -m rag_core ingest` with local embeddings first")


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "local")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "has_key", lambda: True)
    monkeypatch.setattr(llm, "generate", lambda *a, **k: "166 libraries [1].")
    monkeypatch.setattr(llm, "generate_json", lambda *a, **k: {"search_query": "depository library visits FY 2018",
                                                               "keywords": ["visits"]})
    embeddings.dense.cache_clear()
    import api
    monkeypatch.setattr(api, "QUERY_LOG", tmp_path / "q.jsonl")
    api._rag.cache_clear()
    return TestClient(api.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["embeddings"].startswith("local")


def test_validation_errors(client):
    assert client.post("/ask", json={"doc": "nope", "question": "hello there"}).status_code == 404
    assert client.post("/ask", json={"doc": "gpo", "question": "hi"}).status_code == 422          # too short
    assert client.post("/ask", json={"doc": "gpo", "question": "hello", "variant": "x"}).status_code == 422


@needs_index
def test_ask_returns_answer_sources_and_logs(client, tmp_path):
    r = client.post("/ask", json={"doc": "gpo", "question": "How many depository libraries did GPO staff visit?",
                                  "include_trace": True})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("166") and body["sources"][0]["cited"]
    assert any("166 depository libraries" in s["text"] for s in body["sources"])
    assert "plan" in body["trace"] and "total" in body["timings"]
    assert (tmp_path / "q.jsonl").read_text().count("\n") == 1
