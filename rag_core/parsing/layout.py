"""Layout-aware PDF reading.

Naive extractors read a page top-to-bottom, left-to-right, which interleaves the lines of
two-column layouts ("The United States ... Among the many highlights ..."). This module:

  1. extracts text blocks with their position and font information,
  2. marks superscript footnote numbers as [^n],
  3. removes running headers / footers / page numbers (text repeated in the page margins),
  4. orders blocks with a recursive XY-cut so columns are read one after another.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

SUPERSCRIPT_FLAG = 1


@dataclass
class Block:
    text: str
    bbox: tuple[float, float, float, float]
    page: int                   # 1-based
    size: float                 # dominant font size
    font: str                   # dominant font name
    bold: bool
    footnote_refs: list[str] = field(default_factory=list)

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]


BOLD_HINTS = ("bold", "semibold", "medium", "-md", "heavy", "black")


def _is_bold(font: str, flags: int) -> bool:
    return bool(flags & 16) or any(h in font.lower() for h in BOLD_HINTS)


def extract_blocks(page: pymupdf.Page) -> list[Block]:
    """Text blocks with dominant font info; superscript digits become footnote markers [^n]."""
    blocks: list[Block] = []
    for raw in page.get_text("dict", sort=False)["blocks"]:
        if raw.get("type") != 0:
            continue
        lines_text, refs = [], []
        size_chars: Counter = Counter()
        fonts: Counter = Counter()
        bold_chars = total_chars = 0
        for line in raw["lines"]:
            parts = []
            main = max(line["spans"], key=lambda s: s["size"])
            for span in line["spans"]:
                text = span["text"]
                stripped = text.strip()
                if not stripped:
                    parts.append(text)
                    continue
                # Superscript = smaller AND raised above the line's baseline. A subscript (CO2) is
                # smaller but lowered, so it stays as text.
                raised = span["origin"][1] < main["origin"][1] - 0.5
                is_sup = raised and (span["flags"] & SUPERSCRIPT_FLAG or span["size"] < 0.85 * main["size"])
                if is_sup and re.fullmatch(r"\d{1,3}(,\d{1,3})*", stripped):
                    for n in stripped.split(","):
                        parts.append(f"[^{n}]")
                        refs.append(n)
                    continue
                parts.append(text)
                n = len(stripped)
                size_chars[round(span["size"], 1)] += n
                fonts[span["font"]] += n
                total_chars += n
                if _is_bold(span["font"], span["flags"]):
                    bold_chars += n
            lines_text.append("".join(parts))
        text = "\n".join(lines_text).strip()
        if not text or not total_chars:
            continue
        blocks.append(Block(
            text=text,
            bbox=tuple(raw["bbox"]),
            page=page.number + 1,
            size=size_chars.most_common(1)[0][0],
            font=fonts.most_common(1)[0][0],
            bold=bold_chars > total_chars / 2,
            footnote_refs=refs,
        ))
    return blocks


# ---------------------------------------------------------------- running headers / footers

def _signature(line: str) -> str:
    """Normalise so 'Report 12' and 'Report 13' compare equal; empty for bare numbers."""
    return re.sub(r"[\d\s]+", "", line.lower())


def remove_running_text(pages: list[list[Block]], page_height: float,
                        margin: float = 0.09, min_share: float = 0.1) -> list[list[Block]]:
    """Remove lines in the top/bottom margin that repeat on many pages (running headers and
    footers), and bare page numbers. Works per line because PDFs often merge the page number
    and one or two header lines into a single block."""
    def in_margin(b: Block) -> bool:
        return b.y1 < page_height * margin or b.y0 > page_height * (1 - margin)

    counts = Counter()
    for blocks in pages:
        counts.update({_signature(line) for b in blocks if in_margin(b) for line in b.text.split("\n")})
    threshold = max(3, min_share * len(pages))
    repeated = {sig for sig, n in counts.items() if n >= threshold and sig}

    cleaned = []
    for blocks in pages:
        keep = []
        for b in blocks:
            if in_margin(b):
                lines = [ln for ln in b.text.split("\n") if _signature(ln) and _signature(ln) not in repeated]
                if not lines:
                    continue
                b.text = "\n".join(lines)
            keep.append(b)
        cleaned.append(keep)
    return cleaned


# ---------------------------------------------------------------- reading order (XY-cut)

def _gaps(blocks: list[Block], axis: str, min_gap: float) -> list[list[Block]]:
    """Split blocks into groups separated by empty bands along an axis."""
    lo, hi = (lambda b: b.x0, lambda b: b.x1) if axis == "x" else (lambda b: b.y0, lambda b: b.y1)
    ordered = sorted(blocks, key=lo)
    groups, current, reach = [], [ordered[0]], hi(ordered[0])
    for b in ordered[1:]:
        if lo(b) - reach >= min_gap:
            groups.append(current)
            current, reach = [b], hi(b)
        else:
            current.append(b)
            reach = max(reach, hi(b))
    groups.append(current)
    return groups


def xy_cut(blocks: list[Block], min_col_gap: float = 8.0, min_row_gap: float = 1.0) -> list[Block]:
    """Recursive XY-cut. Vertical cuts (columns) are tried first so a two-column page is read
    column by column; a full-width title blocks the vertical cut, so a horizontal cut separates
    it first and the columns underneath are then cut on the next level."""
    if len(blocks) <= 1:
        return blocks
    cols = _gaps(blocks, "x", min_col_gap)
    if len(cols) > 1:
        return [b for g in cols for b in xy_cut(g, min_col_gap, min_row_gap)]
    rows = _gaps(blocks, "y", min_row_gap)
    if len(rows) > 1:
        return [b for g in rows for b in xy_cut(g, min_col_gap, min_row_gap)]
    return sorted(blocks, key=lambda b: (round(b.y0), b.x0))


def read_pdf(path) -> tuple[pymupdf.Document, list[list[Block]]]:
    """Blocks for every page, cleaned of running text and in reading order."""
    doc = pymupdf.open(path)
    pages = [extract_blocks(p) for p in doc]
    pages = remove_running_text(pages, doc[0].rect.height)
    return doc, [xy_cut(blocks) if blocks else [] for blocks in pages]
