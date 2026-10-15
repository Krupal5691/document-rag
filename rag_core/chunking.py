"""Chunking strategies.

naive_chunks      - the baseline: page text read top-to-bottom, left-to-right (columns get mixed),
                    cut every N characters, mid-word and mid-sentence.
structured_chunks - the advanced strategy:
    * parents  = a section's paragraphs (bounded by headings), or one whole table / figure
    * children = sentence-aware pieces of a parent (max N chars, never cutting a sentence,
                 one-sentence overlap); table children = caption + a few linearised rows
    * every child carries its section path, page, kind, labels it references and footnote text
Children are what we embed and search (small = precise); parents are what we hand to the LLM
(big = enough context). This is "small-to-big" / parent-child retrieval.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import pymupdf

from .documents import DocConfig
from .parsing.structure import Element, ParsedDocument

ABBREVIATIONS = {"u.s", "e.g", "i.e", "etc", "rep", "dem", "fig", "no", "vs", "mr", "mrs", "dr", "st", "jan",
                 "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "inc", "co",
                 "corp", "ltd", "approx", "est", "p", "pp", "al", "cf"}
SENT_END = re.compile(r"(?<=[.!?])[\"”’)\]]*\s+(?=[\"“(\[]?[A-Z0-9])")
FOOTNOTE_MARK = re.compile(r"\[\^(\d{1,3})\]")


@dataclass
class Chunk:
    id: str
    text: str                  # what the LLM / user sees
    embed_text: str            # what gets embedded and keyword-indexed (text + context header)
    metadata: dict = field(default_factory=dict)


def _id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def split_sentences(text: str) -> list[str]:
    sentences, start = [], 0
    for m in SENT_END.finditer(text):
        before = text[start:m.start()].rstrip()
        last_word = re.findall(r"([A-Za-z.]+)\.$", before)
        if last_word and (last_word[0].lower().rstrip(".") in ABBREVIATIONS or re.fullmatch(r"[A-Z]", last_word[0])):
            continue                                   # "U.S. Government", "Rep. of", "A. Smith"
        sentences.append(text[start:m.start()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def sentence_chunks(text: str, max_chars: int, overlap: int = 1) -> list[str]:
    """Pack whole sentences into chunks of at most max_chars. A sentence longer than max_chars
    becomes its own chunk (we never cut inside a sentence). `overlap` sentences are repeated at the
    start of the next chunk so a fact spanning a boundary is still whole somewhere."""
    sentences = split_sentences(text)
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            if cur and len(" ".join(cur + [s])) > max_chars:
                cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ------------------------------------------------------------------------------------ naive

def naive_chunks(cfg: DocConfig, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Baseline: plain text extraction, fixed-size character windows. No layout, no structure."""
    doc = pymupdf.open(cfg.path)
    chunks = []
    for page in doc:
        text = page.get_text(sort=True)          # plain top-to-bottom, left-to-right reading
        for start in range(0, max(len(text) - overlap, 1), size - overlap):
            piece = text[start:start + size]
            if piece.strip():
                chunks.append(Chunk(
                    id=_id(cfg.id, "naive", page.number, start), text=piece, embed_text=piece,
                    metadata={"doc_id": cfg.id, "page": page.number + 1, "kind": "text", "section_path": []},
                ))
    return chunks


# ------------------------------------------------------------------------------- structured

def _facets(cfg: DocConfig, path: list[str], text: str) -> dict:
    out = {}
    haystack = " | ".join(path).lower()
    for facet, values in cfg.facets.items():
        for v in values:
            if v.lower() in haystack:
                out[facet] = v
    return out


def _footnotes(text: str, page: int, parsed: ParsedDocument) -> list[str]:
    notes = []
    for n in dict.fromkeys(FOOTNOTE_MARK.findall(text)):
        note = parsed.page_footnotes.get(f"{page}:{n}") or parsed.footnotes.get(n)
        if note:
            notes.append(f"[^{n}] {note}")
    return notes


def _context_header(cfg: DocConfig, path: list[str], kind: str, label: str | None) -> str:
    where = " > ".join(path) if path else "Front matter"
    what = f" | {label}" if label else (f" | {kind}" if kind not in ("paragraph",) else "")
    return f"[{cfg.title} > {where}{what}]"


def structured_chunks(cfg: DocConfig, parsed: ParsedDocument) -> tuple[list[Chunk], dict[str, dict]]:
    """Returns (children to index, parents by id)."""
    children: list[Chunk] = []
    parents: dict[str, dict] = {}

    def add_parent(text: str, elements: list[Element], kind: str, label: str | None = None) -> str:
        first = elements[0]
        pid = _id(cfg.id, "parent", len(parents), first.page, text[:50])
        pages = sorted({e.page for e in elements})
        parents[pid] = {
            "id": pid, "text": text, "kind": kind, "label": label, "pages": pages, "page": pages[0],
            "section_path": first.section_path,
        }
        return pid

    def add_child(text: str, pid: str, el: Element, kind: str, label: str | None, pages: list[int]):
        notes = _footnotes(text, el.page, parsed)
        display = text + ("\n\nFootnotes: " + " ".join(notes) if notes else "")
        header = _context_header(cfg, el.section_path, kind, label)
        meta = {
            "doc_id": cfg.id, "parent_id": pid, "page": pages[0], "pages": pages, "kind": kind,
            "label": label, "section_path": el.section_path,
            "chapter": el.section_path[0] if el.section_path else "Front matter",
            "section": el.section_path[-1] if el.section_path else "Front matter",
            "refs": sorted(set(el.refs) - ({label} if label else set())),
            "footnotes": notes, **_facets(cfg, el.section_path, text),
        }
        children.append(Chunk(id=_id(cfg.id, pid, len(children)), text=display,
                              embed_text=f"{header}\n{display}", metadata=meta))

    # 1) group running text by section into parents
    buffer: list[Element] = []

    def flush():
        if not buffer:
            return
        group, size = [], 0
        for el in buffer + [None]:
            if el is None or (group and size + len(el.text) > cfg.parent_chars):
                text = "\n\n".join(e.text for e in group)
                pid = add_parent(text, group, "text")
                pages = sorted({e.page for e in group})
                for piece in sentence_chunks(text.replace("\n\n", " "), cfg.child_chars, cfg.overlap_sentences):
                    # attribute the child to the element (page, refs) where its first words appear
                    src = next((e for e in group if piece[:40] in e.text.replace("\n\n", " ")), group[0])
                    merged = Element(src.kind, src.text, src.page, src.section_path,
                                     refs=sorted({r for e in group for r in e.refs}))
                    add_child(piece, pid, merged, "text", None, pages)
                group, size = [], 0
            if el is not None:
                group.append(el)
                size += len(el.text)
        buffer.clear()

    for el in parsed.elements:
        if el.kind in ("paragraph", "note", "footnote"):
            if buffer and buffer[-1].section_path != el.section_path:
                flush()
            buffer.append(el)
            continue
        flush()
        if el.kind == "heading":
            continue
        if el.kind == "table":
            pid = add_parent(el.text, [el], "table", el.label)
            caption = el.extra.get("caption", el.label)
            rows = el.extra.get("rows", [])
            group: list[str] = []
            for row in rows + [None]:
                if row is None or (group and len(" ".join(group)) + len(row) > cfg.child_chars * 2):
                    if group:
                        add_child(f"{caption}\n" + "\n".join(group), pid, el, "table", el.label, [el.page])
                    group = []
                if row is not None:
                    group.append(row)
            if el.extra.get("notes"):
                add_child(f"{caption}\n{el.extra['notes']}", pid, el, "table", el.label, [el.page])
        else:                                   # figure, vision element, call-out
            pid = add_parent(el.text, [el], el.kind, el.label)
            for piece in sentence_chunks(el.text.replace("\n", " "), cfg.child_chars * 2, 0):
                add_child(piece, pid, el, el.kind, el.label, [el.page])
    flush()
    return children, parents
