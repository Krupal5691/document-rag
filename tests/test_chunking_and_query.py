from rag_core import documents
from rag_core.chunking import sentence_chunks, split_sentences
from rag_core.pipeline import Options
from rag_core.query import keyword_query, rule_filters


def test_sentence_splitter_keeps_abbreviations():
    s = split_sentences("The U.S. Government Publishing Office grew. Growth is 6.3 percent. Iran, Islamic Rep. is listed.")
    assert s == ["The U.S. Government Publishing Office grew.", "Growth is 6.3 percent.",
                 "Iran, Islamic Rep. is listed."]


def test_sentence_chunks_never_cut_sentences():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(30))
    chunks = sentence_chunks(text, max_chars=120, overlap=1)
    assert all(c.endswith(".") for c in chunks)
    assert all(len(c) <= 120 for c in chunks)
    # one-sentence overlap: each chunk starts with the previous chunk's last sentence
    assert chunks[1].startswith(split_sentences(chunks[0])[-1])


def test_region_filters_from_names_and_abbreviations():
    cfg = documents.get("worldbank")
    assert rule_filters("What is the growth outlook for South Asia?", cfg, []) == {"region": "South Asia"}
    assert rule_filters("SAR growth 2026", cfg, []) == {"region": "South Asia"}
    assert rule_filters("growth in sub-saharan africa", cfg, []) == {"region": "Sub-Saharan Africa"}


def test_chapter_filter():
    cfg = documents.get("worldbank")
    chapters = ["Ch. 1 Global Outlook", "Ch. 3 A Rising Challenge: Sovereign Debt Levels and Interest Rates in EMDEs",
                "Executive Summary"]
    assert rule_filters("What does Chapter 3 say about debt?", cfg, chapters)["chapter"].startswith("Ch. 3")
    assert rule_filters("According to the executive summary, what are the risks?", cfg, chapters)["chapter"] == "Executive Summary"


def test_keyword_query_drops_filler():
    assert keyword_query("What is the growth outlook for South Asia in this report?") == "growth outlook South Asia"


def test_options_parse_ablation_names():
    o = Options.parse("advanced-no-rerank-no-hybrid")
    assert (o.rerank, o.hybrid, o.rewrite) == (False, False, True)


def test_llm_chapter_guess_needs_lexical_support():
    from rag_core.query import _mentions
    assert not _mentions("What were GPO's total revenues in FY 2018?", "FINANCIAL STATEMENTS")
    assert _mentions("What does the sovereign debt chapter conclude?", "Ch. 3 A Rising Challenge: Sovereign Debt Levels")
