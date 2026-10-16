"""Exercise the Gemini-dependent code paths with a fake Gemini, so they are tested without an API key."""
import json

import pymupdf
import pytest

from rag_core import config, documents, embeddings, evaluation, generate, llm, pipeline, query, vision


@pytest.fixture
def fake_gemini(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "local")           # keep using the local index
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "has_key", lambda: True)
    calls = {"generate": [], "json": [], "image": 0}

    def fake_generate(prompt, system, model=None, temperature=0.0, cache=None):
        calls["generate"].append(prompt)
        return "GPO staff visited 166 depository libraries in 25 states [1]."

    def fake_json(prompt, system, schema, model=None, cache=None):
        calls["json"].append(system)
        if "supported" in schema["properties"]:
            return {"supported": True}
        return {"search_query": "South Asia SAR growth forecast 2026", "keywords": ["South Asia", "growth"],
                "chapter": "Not a real chapter", "region": "South Asia", "kind": None}

    def fake_image(png, prompt, schema, model=None):
        calls["image"] += 1
        return {"elements": [{"type": "chart", "title": "Carbon footprint",
                              "content": "Total 346 kg CO2e: production 50%, use 45%", "footnotes": ["10"]}]}

    monkeypatch.setattr(llm, "generate", fake_generate)
    monkeypatch.setattr(llm, "generate_json", fake_json)
    monkeypatch.setattr(llm, "describe_image", fake_image)
    return calls


def test_query_plan_merges_llm_and_validates_filters(fake_gemini):
    plan = query.plan("growth outlook?", documents.get("worldbank"), ["Ch. 1 Global Outlook"])
    assert plan.source == "llm+rules"
    assert plan.search_query.startswith("South Asia")
    assert plan.filters == {"region": "South Asia"}                 # invented chapter rejected


def test_generate_extracts_citations(fake_gemini):
    out = generate.answer("q", [{"text": "a", "page": 5}, {"text": "b", "page": 21}])
    assert out.cited == [1]


def test_vision_elements_and_cache(fake_gemini):
    cfg = documents.get("imac")
    page = pymupdf.open(cfg.path)[9]
    els, status = vision.page_elements(cfg, page, ["Carbon Footprint"])
    assert status == "called" and els[0].kind == "figure" and els[0].footnote_refs == ["10"]
    assert "346 kg" in els[0].text
    els2, status2 = vision.page_elements(cfg, page, ["Carbon Footprint"])
    assert status2 == "cached" and fake_gemini["image"] == 1        # second call served from cache


needs_index = pytest.mark.skipif(
    not (config.INDEX_DIR / "parents" / "gpo-advanced-local-bge-small-en-v1.5.json").exists(),
    reason="run `python -m rag_core ingest` with local embeddings first")


@needs_index
def test_advanced_pipeline_answers_with_mocked_llm(fake_gemini):
    embeddings.dense.cache_clear()
    res = pipeline.AdvancedRAG("gpo").ask("How many depository libraries did GPO staff visit in FY 2018?")
    assert "166" in res.answer and res.cited == [1]
    assert any("166 depository libraries" in c["text"] for c in res.contexts)
    assert "Context passages" in fake_gemini["generate"][0]


@needs_index
def test_evaluation_with_answers(fake_gemini, monkeypatch, tmp_path):
    embeddings.dense.cache_clear()
    monkeypatch.setattr(config, "EVAL_DIR", tmp_path)
    (tmp_path / "questions").mkdir()
    (tmp_path / "questions" / "gpo.json").write_text(json.dumps([
        {"id": "a", "question": "How many depository libraries did GPO staff visit in FY 2018?",
         "facts": [["166"]], "pages": [5, 21]},
        {"id": "b", "question": "Price of tea?", "answerable": False}]))
    res = evaluation.evaluate("gpo", ["advanced"], with_answers=True, with_judge=True, progress=lambda m: None)
    s = res["variants"]["advanced"]["summary"]
    assert s["hit@k"] == 1 and s["accuracy"] == 0.5 and s["faithfulness"] == 1   # fake never refuses
    assert (tmp_path / "results" / "gpo.md").exists()


def test_fallback_rotates_past_exhausted_models(monkeypatch):
    monkeypatch.setattr(config, "FALLBACK_MODELS", ["b", "c"])
    monkeypatch.setattr(llm, "EXHAUSTED", set())
    tried = []

    def call(m):
        tried.append(m)
        if m in ("a", "b"):
            raise llm.QuotaExhaustedError(f"{m} out of quota")
        return "ok"

    assert llm._with_fallback("a", call) == "ok" and llm.LAST_MODEL == "c"
    assert llm._with_fallback("a", call) == "ok" and tried == ["a", "b", "c", "c"]   # exhausted models skipped
    monkeypatch.setattr(config, "FALLBACK_MODELS", [])
    with pytest.raises(llm.QuotaExhaustedError):
        llm._with_fallback("a", call)


def test_citation_evidence_is_found_and_highlighted():
    from rag_core import citations
    from rag_core.parsing.structure import parse_pdf
    cfg = documents.get("worldbank")
    page19 = " ".join(e.text for e in parse_pdf(cfg.path, "worldbank").elements if e.page == 19)
    hl = citations.locate("Global growth is projected to slow to 2.5 percent in 2026 [1].",
                          [{"text": page19, "pages": [19]}], cfg.path)[1]
    assert hl.page == 19 and hl.match == "exact" and hl.rects          # despite "Te"/"The" and ligatures
    assert "2.5 percent" in hl.sentence
    assert citations.page_pdf(cfg.path, hl).startswith(b"%PDF")
    linked = citations.link_citations("Growth slows [1].", {1: "app/static/hl/x.pdf"})
    assert linked == "Growth slows [[1]](app/static/hl/x.pdf)."


def test_table_citations_highlight_exact_cells():
    from rag_core import citations
    from rag_core.parsing.structure import parse_pdf
    cfg = documents.get("worldbank")
    table = next(e for e in parse_pdf(cfg.path, "worldbank").elements if e.label == "Table 2.10")
    hl = citations.locate("Bangladesh's growth is forecast at 4.6 percent in fiscal year 2026/27 [1].",
                          [{"text": table.text, "pages": [98], "kind": "table"}], cfg.path)[1]
    assert hl.match == "cells" and len(hl.rects) == 1                  # one cell, not the whole table
    page = pymupdf.open(cfg.path)[97]
    assert page.get_textbox(pymupdf.Rect(hl.rects[0])).strip() == "4.6"
    label = " ".join(page.get_textbox(pymupdf.Rect(r)).strip() for r in hl.label_rects)
    assert "Bangladesh" in label and "Excluding" not in label          # Table 2.10 row, not Table 2.9

    cfg = documents.get("imac")
    page10 = " ".join(e.text for e in parse_pdf(cfg.path, "imac").elements if e.page == 10)
    hl = citations.locate("The total product footprint is 346 kg CO2e [1]. Product use accounts for 45% [1].",
                          [{"text": page10, "pages": [10]}], cfg.path)[1]
    page = pymupdf.open(cfg.path)[9]
    labels = " ".join(page.get_textbox(pymupdf.Rect(r)).strip() for r in hl.label_rects)
    assert hl.match == "cells" and "Total product footprint" in labels and "Life cycle" not in labels
