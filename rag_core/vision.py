"""Vision step for visually dense pages (charts, icon call-outs, small tables).

Each page is rendered to an image and sent to a vision LLM (Gemini), which transcribes every
visual element into self-contained text with all numbers, labels and units. Those descriptions
become extra elements that are chunked and indexed like normal text, so a question about a bar in
a chart can match the chart's transcription.

Results are cached in cache/vision/<doc>/ (committed to git), so the API is called once per page.
"""
from __future__ import annotations

import json

import pymupdf

from . import config, llm
from .documents import DocConfig
from .parsing.structure import Element

SCHEMA = {
    "type": "object",
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["chart", "table", "callout", "diagram", "text"]},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "footnotes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["type", "title", "content"],
            },
        }
    },
    "required": ["elements"],
}

PROMPT = """You are transcribing one page of a product environmental report for a search index.
List every VISUAL element on the page: charts, tables, icon call-outs / big-number statements, diagrams.
Skip plain body paragraphs (they are extracted separately) unless they are inside a visual element.

For each element:
- type: chart | table | callout | diagram | text
- title: a short descriptive title (use the printed title if there is one)
- content: a self-contained factual description that someone could answer questions from WITHOUT the
  image. Include EVERY number exactly as printed with its unit and label (e.g. "Production: 50% of
  the 346 kg CO2e total"). For charts, state what each bar/segment/arrow shows and any totals or
  reductions. For tables, write each row as "row label - column: value; column: value". Keep the
  product name (iMac) and configuration in the text.
- footnotes: the footnote/endnote numbers printed next to the element's claims, as strings.
Do not invent anything that is not on the page."""


def _cache_file(cfg: DocConfig, page_no: int):
    return config.CACHE_DIR / "vision" / cfg.id / f"page-{page_no:03d}.json"


def page_elements(cfg: DocConfig, page: pymupdf.Page, section_path: list[str]) -> tuple[list[Element], str]:
    """Returns (elements, status) where status is cached | called | skipped."""
    page_no = page.number + 1
    path = _cache_file(cfg, page_no)
    if path.exists():
        data, status = json.loads(path.read_text()), "cached"
    elif llm.has_key():
        png = page.get_pixmap(dpi=150).tobytes("png")
        data = llm.describe_image(png, PROMPT, SCHEMA)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        status = "called"
    else:
        return [], "skipped"

    elements = []
    for i, item in enumerate(data.get("elements", [])):
        refs = [f for f in item.get("footnotes", []) if f.isdigit()]
        marks = "".join(f"[^{n}]" for n in refs)
        text = f"{item['title']}: {item['content']}{marks}"
        kind = {"chart": "figure", "diagram": "figure"}.get(item["type"], item["type"])
        elements.append(Element(kind if kind in ("figure", "table") else "callout", text, page_no,
                                section_path, label=f"Page {page_no} visual {i + 1}: {item['title']}",
                                footnote_refs=refs, extra={"source": "vision", "visual_type": item["type"]}))
    return elements, status
