"""Turn ordered PDF blocks into typed document elements with a section hierarchy.

Element kinds:
  heading    - detected from font size / weight / family relative to the body text
  paragraph  - normal body text
  table      - caption-anchored region, rebuilt into rows (see tables.py)
  figure     - caption-anchored chart region: caption, panel titles, source and notes
  note       - small-print text outside tables/figures (sources, sidebars)
  footnote   - numbered notes, from page bottoms or an "Endnotes" section

Every element carries `section_path`, e.g. ["Ch. 2 Regional Outlooks", "South Asia", "Outlook"],
built from the PDF outline (table of contents) plus the heading stack. That path is what lets the
retriever tell the South Asia section from the East Asia one.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

import pymupdf

from .layout import Block, read_pdf
from .normalize import LigatureRepair, normalize_text
from .tables import reconstruct_table

CAPTION_RE = re.compile(r"^(TABLE|FIGURE|Table|Figure)\s+([A-Z]?\d+(?:\.\d+)*(?:\.[A-Z])?)\b\.?\s*(.*)", re.S)
BOX_RE = re.compile(r"^(BOX|Box)\s+(\d+(?:\.\d+)*)\b\.?\s*(.*)", re.S)
REF_RE = re.compile(r"\b(Figure|Table|Box)s?\s+((?:[A-Z]?\d+(?:\.\d+)*))", re.I)
ENDNOTES_RE = re.compile(r"^(end\s*notes|notes|footnotes)$", re.I)
TOC_LINE_RE = re.compile(r"\.{6,}|(\. ){5,}")
SOURCE_RE = re.compile(r"^(Sources?|Note|Notes)\s*:", re.I)


@dataclass
class Element:
    kind: str
    text: str
    page: int
    section_path: list[str] = field(default_factory=list)
    level: int | None = None              # headings only
    label: str | None = None              # "Table 2.10", "Figure 1.1"
    footnote_refs: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)    # labels of tables/figures this text points to
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    pages: int
    elements: list[Element]
    footnotes: dict[str, str]             # "18" -> text   (document endnotes)
    page_footnotes: dict[str, str]        # "12:3" -> text (page-level footnotes)
    stats: dict

    def to_dict(self) -> dict:
        return {**asdict(self), "elements": [e.to_dict() for e in self.elements]}

    @classmethod
    def from_dict(cls, d: dict) -> ParsedDocument:
        return cls(**{**d, "elements": [Element(**e) for e in d["elements"]]})


def _family(font: str) -> str:
    return re.split(r"[-,]", font)[0].lower()


def _refs(text: str) -> list[str]:
    return sorted({f"{m.group(1).title()} {m.group(2)}" for m in REF_RE.finditer(text)})


def _mostly_numbers(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(c.isdigit() or c in ".,%-–()$" for c in chars) / len(chars) > 0.5


def _chapter_map(doc: pymupdf.Document) -> dict[int, str]:
    """page -> chapter title from the PDF outline. Box and region entries are skipped because they
    sit inside chapters; the in-text headings handle those."""
    toc = [(lvl, re.sub(r"\s+", " ", title).strip(), page) for lvl, title, page in doc.get_toc()]
    chapters = [(t, p) for lvl, t, p in toc if lvl == 1 and not t.lower().startswith(("box", "--", "contents", "summary of"))
                and "--" not in t]
    mapping, current = {}, None
    for page in range(1, doc.page_count + 1):
        for title, start in chapters:
            if start == page:
                current = title
        if current:
            mapping[page] = current
    return mapping


class StructureBuilder:
    def __init__(self, doc: pymupdf.Document, pages: list[list[Block]], heading_max_words: int = 14):
        self.doc = doc
        self.pages = pages
        self.heading_max_words = heading_max_words
        sizes, fonts = Counter(), Counter()
        for blocks in pages:
            for b in blocks:
                sizes[b.size] += len(b.text)
                fonts[_family(b.font)] += len(b.text)
        self.body_size = sizes.most_common(1)[0][0]
        self.body_family = fonts.most_common(1)[0][0]
        # Body size can change between parts of a document (narrative vs financial statements), so
        # headings are judged against the page's own body text when the page has enough of it.
        self.page_body: dict[int, float] = {}
        for blocks in pages:
            local = Counter()
            for b in blocks:
                local[b.size] += len(b.text)
            if blocks and sum(local.values()) > 1200:
                self.page_body[blocks[0].page] = local.most_common(1)[0][0]

    def _body(self, b: Block) -> float:
        return self.page_body.get(b.page, self.body_size)

    # ---------------------------------------------------------------- classification

    def _is_heading(self, b: Block) -> bool:
        text = normalize_text(b.text)
        words = text.split()
        if not words or len(words) > self.heading_max_words or text.endswith((".", ",", ";")) or _mostly_numbers(text):
            return False
        if re.search(r":\s*[$\d]", text) or b.footnote_refs and len(words) > 6:
            return False
        if sum(c.isalpha() for c in text) < 3:
            return False
        body = self._body(b)
        bigger = b.size >= body * 1.15
        emphasised = b.size >= body * 0.95 and (b.bold or _family(b.font) != self.body_family)
        return bigger or emphasised

    def _level(self, b: Block) -> int:
        """Heading level from size relative to body text: chapter titles are much bigger than body,
        sub-section headings are body-sized but bold or in a different font."""
        ratio = b.size / self._body(b)
        if ratio >= 1.9:
            return 1
        if ratio >= 1.35:
            return 2
        if ratio >= 1.1:
            return 3
        return 4

    # ---------------------------------------------------------------- caption regions

    def _region(self, blocks: list[Block], start: int) -> tuple[list[Block], int]:
        """Blocks belonging to the table/figure whose caption is blocks[start]: everything up to and
        including the Source/Note lines, stopping at body paragraphs, headings or another caption."""
        region, i = [], start + 1
        while i < len(blocks) and len(region) < 120:
            b = blocks[i]
            text = b.text.strip()
            if CAPTION_RE.match(text) or BOX_RE.match(text):
                break
            is_body = b.size >= self.body_size * 0.97 and len(text) > 120 and not _mostly_numbers(text)
            if is_body:
                break
            region.append(b)
            i += 1
            if SOURCE_RE.match(text):
                # keep a directly following Note block too
                while i < len(blocks) and SOURCE_RE.match(blocks[i].text.strip()):
                    region.append(blocks[i])
                    i += 1
                break
        return region, i

    # ---------------------------------------------------------------- main pass

    def build(self) -> tuple[list[Element], dict, dict]:
        chapters = _chapter_map(self.doc)
        elements: list[Element] = []
        endnotes: dict[str, str] = {}
        page_notes: dict[str, str] = {}
        stack: list[tuple[int, str]] = []
        in_endnotes = False
        last_heading: Block | None = None
        page_height = self.doc[0].rect.height

        for page_idx, blocks in enumerate(self.pages):
            page_no = page_idx + 1
            page = self.doc[page_idx]
            chapter = chapters.get(page_no)

            def path() -> list[str]:  # noqa: B023 - always called within the current iteration
                p = [t for _, t in stack]  # noqa: B023
                return ([chapter] if chapter and (not p or p[0] != chapter) else []) + p  # noqa: B023

            i = 0
            pending_prefix = ""
            while i < len(blocks):
                b = blocks[i]
                text = normalize_text(b.text)
                if not text:
                    i += 1
                    continue

                cap = CAPTION_RE.match(text)
                if cap:
                    kind = "table" if cap.group(1).lower() == "table" else "figure"
                    label = f"{cap.group(1).title()} {cap.group(2)}"
                    region, i = self._region(blocks, i)
                    elements.append(self._caption_element(page, kind, label, text, b, region, path()))
                    continue

                box = BOX_RE.match(text)
                if box and len(text.split()) <= 25:
                    lvl = (self._level(b) if self._is_heading(b) else 2)
                    stack = [(lv, t) for lv, t in stack if lv < lvl] + [(lvl, text)]
                    elements.append(Element("heading", text, page_no, path(), level=lvl, label=f"Box {box.group(2)}"))
                    i += 1
                    continue

                if TOC_LINE_RE.search(text):          # "Chapter 2 ........ 69" lines of a contents page
                    i += 1
                    continue

                if self._is_heading(b):
                    lvl = self._level(b)
                    text = re.sub(r"\[\^\d+\]", "", text).strip()
                    if any(lv == lvl and t == text for lv, t in stack):
                        i += 1                         # running title repeated on every page of a box
                        continue
                    prev = elements[-1] if elements else None
                    if (prev and prev.kind == "heading" and prev.page == page_no and prev.level == lvl
                            and last_heading is not None and abs(b.y0 - last_heading.y1) < b.size
                            and b.size == last_heading.size):
                        # one heading broken over two blocks ("Progress toward" / "our 2030 goal")
                        prev.text = f"{prev.text} {text}"
                        stack[-1] = (lvl, prev.text)
                        prev.section_path = path()
                        last_heading = b
                        i += 1
                        continue
                    last_heading = b
                    in_endnotes = bool(ENDNOTES_RE.match(text))
                    stack = [(lv, t) for lv, t in stack if lv < lvl] + [(lvl, text)]
                    elements.append(Element("heading", text, page_no, path(), level=lvl))
                    i += 1
                    continue

                if in_endnotes:
                    m = re.match(r"^\[?\^?(\d{1,3})\]?\s*(\S.*)", text, re.S)
                    if m:
                        endnotes[m.group(1)] = m.group(2).strip()
                        last_note = m.group(1)
                    elif endnotes:
                        endnotes[last_note] += " " + text
                    i += 1
                    continue

                small = b.size < self.body_size * 0.85
                at_bottom = b.y0 > page_height * 0.8
                fn = re.match(r"^(\d{1,3})[.\s]\s*(\S.*)", text, re.S)
                if small and at_bottom and fn:
                    page_notes[f"{page_no}:{fn.group(1)}"] = fn.group(2).strip()
                    elements.append(Element("footnote", text, page_no, path(), label=f"Note {fn.group(1)}"))
                    i += 1
                    continue

                # Drop cap: a lone capital letter printed big before "he U.S. Government ..."
                if re.fullmatch(r"[A-Z]", text) and i + 1 < len(blocks) and blocks[i + 1].text[:1].islower():
                    pending_prefix = text
                    i += 1
                    continue

                # A big stand-alone number ("100%", "76%") is a call-out: glue it to the next block.
                if _mostly_numbers(text) and len(text) <= 8 and b.size > self.body_size:
                    pending_prefix = text + " "
                    i += 1
                    continue

                kind = "note" if small and (SOURCE_RE.match(text) or _mostly_numbers(text)) else "paragraph"
                elements.append(Element(kind, pending_prefix + text, page_no, path(),
                                        footnote_refs=b.footnote_refs, refs=_refs(text)))
                pending_prefix = ""
                i += 1
        return elements, endnotes, page_notes

    def _caption_element(self, page, kind, label, caption, cap_block, region, path) -> Element:
        notes = " ".join(normalize_text(r.text) for r in region if SOURCE_RE.match(r.text.strip()))
        body = [r for r in region if not SOURCE_RE.match(r.text.strip())]
        if kind == "table" and body:
            rect = pymupdf.Rect(min(r.x0 for r in body) - 1, min(r.y0 for r in body) - 1,
                                max(r.x1 for r in body) + 1, max(r.y1 for r in body) + 1)
            table = reconstruct_table(page, rect, caption, notes)
            if table:
                text = f"{caption}\n\n{table.to_markdown()}" + (f"\n\n{notes}" if notes else "")
                return Element("table", text, page.number + 1, path, label=label, refs=[label],
                               extra={"rows": table.row_sentences(), "caption": caption, "notes": notes})
        # figure (or a table we could not rebuild): caption + panel titles + notes, chart tick labels dropped
        lines = [normalize_text(r.text) for r in body if not _mostly_numbers(r.text) and len(r.text.strip()) > 2]
        text = "\n".join([caption, *lines] + ([notes] if notes else []))
        return Element(kind, text, page.number + 1, path, label=label, refs=[label],
                       extra={"caption": caption, "notes": notes})


def parse_pdf(path, doc_id: str) -> ParsedDocument:
    doc, pages = read_pdf(path)
    elements, endnotes, page_notes = StructureBuilder(doc, pages).build()

    # Ligature repair with the document's own vocabulary
    repair = LigatureRepair(" ".join(e.text for e in elements))
    fixes = 0
    for e in elements:
        e.text, n = repair.repair(e.text)
        fixes += n

    counts = Counter(e.kind for e in elements)
    return ParsedDocument(
        doc_id=doc_id,
        title=doc.metadata.get("title") or str(path),
        pages=doc.page_count,
        elements=elements,
        footnotes=endnotes,
        page_footnotes=page_notes,
        stats={"elements": dict(counts), "ligature_fixes": fixes, "endnotes": len(endnotes)},
    )
