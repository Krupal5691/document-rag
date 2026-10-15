"""Query understanding: turn the user's raw question into a better search query plus filters.

Users ask vague, long or keyword-only questions ("south asia growth?"). Before searching we:
  1. detect filter values with cheap rules (region names, abbreviations like SAR, "Chapter 3")
  2. ask a small, fast LLM to rewrite the question into a keyword-rich search query, expand short
     queries, and pick filters - it is told what the document contains and which filter values exist
Rules and LLM are merged; LLM filter values are only accepted if they are real values in the index.
Without an API key the rule-based plan is used on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, llm
from .documents import DocConfig

STOPWORDS = set("""a an the of in on for to and or is are was were be been what which who whom whose how
why when where does do did according this that these those report document say says tell me please about
with by from as at it its into than then there their they can could would should will shall""".split())

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "search_query": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "chapter": {"type": "string", "nullable": True},
        "region": {"type": "string", "nullable": True},
        "kind": {"type": "string", "nullable": True, "enum": ["table", "figure", "text"]},
    },
    "required": ["search_query", "keywords"],
}


@dataclass
class QueryPlan:
    original: str
    search_query: str
    keywords: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    source: str = "rules"            # "rules" | "llm+rules"
    notes: list[str] = field(default_factory=list)


def rule_filters(question: str, cfg: DocConfig, chapters: list[str]) -> dict:
    q = question.lower()
    filters = {}
    for facet, values in cfg.facets.items():
        for v in sorted(values, key=len, reverse=True):
            if re.search(rf"\b{re.escape(v.lower())}\b", q):
                filters[facet] = v
                break
        if facet not in filters:
            for alias, v in cfg.aliases.items():
                if v in values and re.search(rf"\b{re.escape(alias)}\b", question if alias.isupper() else q,
                                             0 if alias.isupper() else re.I):
                    filters[facet] = v
                    break
    m = re.search(r"\bchapter\s+(\d+)\b|\bch\.?\s*(\d+)\b", q)
    if m:
        n = m.group(1) or m.group(2)
        match = [c for c in chapters if re.match(rf"(ch\.?|chapter)\s*{n}\b", c.lower())]
        if match:
            filters["chapter"] = match[0]
    elif "executive summary" in q:
        match = [c for c in chapters if c.lower().startswith("executive summary")]
        if match:
            filters["chapter"] = match[0]
    return filters


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS and w != "chapter"}


def _mentions(question: str, title: str) -> bool:
    return bool(_words(question) & _words(title))


def keyword_query(question: str) -> str:
    words = re.findall(r"[A-Za-z0-9$%.\-/]+", question)
    kept = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(kept) or question


def plan(question: str, cfg: DocConfig, chapters: list[str], use_llm: bool = True) -> QueryPlan:
    filters = rule_filters(question, cfg, chapters)
    result = QueryPlan(original=question, search_query=keyword_query(question), filters=dict(filters),
                       keywords=keyword_query(question).split())
    if not (use_llm and llm.has_key()):
        result.notes.append("rule-based plan (no LLM)")
        return result

    regions = cfg.facets.get("region", [])
    system = (
        "You prepare search queries for a retrieval system over ONE document.\n"
        f"Document: {cfg.title}. Contents: {cfg.description}\n"
        "Rewrite the user's question into a concise, keyword-rich search query that would match the "
        "wording used in the document: keep names, numbers, years and units; expand abbreviations; drop "
        "filler words. If the question is only a few keywords, expand it into a full descriptive query. "
        "Return 3-8 important keywords.\n"
        "Set `chapter` only if the question clearly targets one of these chapters: "
        f"{chapters}. "
        + (f"Set `region` only if the question is about one of these regions: {regions}. " if regions else "")
        + "Set `kind` to 'table' only if the question asks for a specific number likely in a table, otherwise null. "
        "Use exact values from the lists, or null."
    )
    try:
        out = llm.generate_json(f"Question: {question}", system, PLAN_SCHEMA, model=config.QUERY_MODEL,
                                cache="query_plan")
    except Exception as e:                         # never let query rewriting break retrieval
        result.notes.append(f"LLM rewrite failed ({type(e).__name__}); used rules")
        return result

    result.source = "llm+rules"
    result.search_query = out.get("search_query") or result.search_query
    result.keywords = out.get("keywords") or result.keywords
    llm_chapter = out.get("chapter")
    if llm_chapter in chapters and "chapter" not in filters:
        # A wrong chapter filter hides the answer completely, so only accept the LLM's choice when the
        # question itself names that chapter (shares a meaningful word with its title).
        if _mentions(question, llm_chapter):
            filters["chapter"] = llm_chapter
        else:
            result.notes.append(f"ignored LLM chapter guess '{llm_chapter}' (not named in the question)")
    if out.get("region") in regions and "region" not in filters:
        filters["region"] = out["region"]
    result.filters = filters
    return result
