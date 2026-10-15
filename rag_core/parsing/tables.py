"""Table reconstruction from word positions.

Borderless report tables (World Bank style) defeat line-based table detectors: row labels get
merged into numbers and tables split into fragments. Instead we take the words inside the table
region and rebuild the grid ourselves:

  * words on the same baseline form a row
  * words closer than a small gap form one cell
  * a row of year-like tokens (2025, 2026f, 2025/26e) is a header row; a new header row mid-table
    starts a new block of the table (e.g. "Fiscal-year basis")
  * values are aligned to header columns from the right, since numbers are right-aligned

The result is rendered two ways: a Markdown grid (for the LLM to read) and one linearised sentence
per row ("India: 2025e = 7.6; 2026f = 6.6; ...") so each row is self-contained when chunked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

HEADER_TOKEN = re.compile(r"^(\d{4}(/\d{2})?[a-z]?|Q[1-4]|[12]\d{3}[–-]\d{2,4}[a-z]?)$")
NUMBER = re.compile(r"^[-–−+]?\(?[\d.,]+\)?%?$|^\.\.$|^n\.?a\.?$|^-$")


@dataclass
class TableData:
    caption: str
    header: list[str]
    rows: list[list[str]]                       # [label, v1, v2, ...]
    notes: str = ""
    sections: list[tuple[int, list[str]]] = field(default_factory=list)  # (row index, header) switches

    def to_markdown(self) -> str:
        out, header = [], self.header
        switches = dict(self.sections)
        for i, row in enumerate(self.rows):
            if i in switches or i == 0:
                header = switches.get(i, header)
                out.append("| " + " | ".join([""] + header) + " |")
                out.append("|" + " --- |" * (len(header) + 1))
            out.append("| " + " | ".join(row + [""] * (len(header) + 1 - len(row))) + " |")
        return "\n".join(out)

    def row_sentences(self) -> list[str]:
        sentences, header = [], self.header
        switches = dict(self.sections)
        for i, row in enumerate(self.rows):
            header = switches.get(i, header)
            label, values = row[0], row[1:]
            if not values:
                sentences.append(f"{label}:")
                continue
            pairs = [f"{h} = {v}" for h, v in zip(header, values) if v]
            sentences.append(f"{label}: " + "; ".join(pairs))
        return sentences


def _rows_from_words(words: list[tuple], y_tol: float = 2.5, cell_gap: float = 4.5) -> list[list[str]]:
    words = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list[tuple]] = []
    for w in words:
        yc = (w[1] + w[3]) / 2
        if lines and abs(((lines[-1][0][1] + lines[-1][0][3]) / 2) - yc) <= y_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    rows = []
    for line in lines:
        line.sort(key=lambda w: w[0])
        cells, cur, last_x1 = [], [], None
        for w in line:
            if last_x1 is not None and w[0] - last_x1 > cell_gap:
                cells.append(" ".join(cur))
                cur = []
            cur.append(w[4])
            last_x1 = w[2]
        cells.append(" ".join(cur))
        rows.append(cells)
    return rows


def _dedupe_header(header: list[str]) -> list[str]:
    seen, out = {}, []
    for h in header:
        seen[h] = seen.get(h, 0) + 1
        out.append(h if seen[h] == 1 else f"{h} (change vs. previous forecast)")
    return out


def _split_label(cells: list[str]) -> tuple[str, list[str]]:
    """Leading non-numeric cells form the row label; the rest are values (one token per value)."""
    label_parts, values = [], []
    for c in cells:
        tokens = c.split()
        if not values and not all(NUMBER.match(t) for t in tokens):
            # a cell may hold "Iran, Islamic Rep. 5.3" when the gap was small: peel numbers off the end
            while tokens and NUMBER.match(tokens[-1]):
                values.insert(0, tokens.pop())
            label_parts.append(" ".join(tokens))
            if values:
                continue
        else:
            values.extend(tokens)
    return " ".join(p for p in label_parts if p).strip(), values


def reconstruct_table(page: pymupdf.Page, rect: pymupdf.Rect, caption: str, notes: str = "") -> TableData | None:
    words = [w for w in page.get_text("words", clip=rect)]
    if not words:
        return None
    grid = _rows_from_words(words)
    header: list[str] = []
    rows: list[list[str]] = []
    sections: list[tuple[int, list[str]]] = []
    pending_label = ""
    for cells in grid:
        tokens = [t for c in cells for t in c.split()]
        year_tokens = [t for t in tokens if HEADER_TOKEN.match(t)]
        if len(year_tokens) >= 3 and len(year_tokens) >= len(tokens) - 3:
            label = " ".join(t for t in tokens if not HEADER_TOKEN.match(t))
            new_header = _dedupe_header(year_tokens)
            if header:
                if pending_label:
                    rows.append([pending_label])
                    pending_label = ""
                sections.append((len(rows), new_header))
            else:
                header = new_header
                pending_label = ""          # lines above the first header are the table's subtitle
            if label:
                pending_label = label       # e.g. "Fiscal-year basis" printed on the header line
            continue
        if not header:
            continue                        # caption / subtitle lines before the column header
        label, values = _split_label(cells)
        label = re.sub(r"^(?:[a-z] )+", "", label)                  # clipped descenders from the line above
        if not values:
            if pending_label and not (label[:1].islower() or pending_label.endswith((",", "-", "and"))):
                rows.append([pending_label])                       # previous line was a group title
                pending_label = ""
            pending_label = (pending_label + " " + label).strip()   # label wrapped onto 2 lines, or a group title
            continue
        if pending_label and (label[:1].islower() or pending_label.endswith((",", "-", "and"))):
            label = (pending_label + " " + label).strip()         # wrapped label
        elif pending_label:
            rows.append([pending_label])                           # group title row ("Calendar-year basis")
        pending_label = ""
        cur_header = sections[-1][1] if sections else header
        width = len(cur_header)
        if width and len(values) < width:                          # right-align to the header
            values = [""] * (width - len(values)) + values
        elif width and len(values) > width:                        # extra leading digit = footnote mark on the label
            extra, values = values[: len(values) - width], values[len(values) - width:]
            label += "".join(f"[^{e}]" for e in extra)
        rows.append([label] + values)
    if pending_label and rows:
        rows.append([pending_label])
    if not rows or not header:
        return None
    return TableData(caption=caption, header=header, rows=rows, notes=notes, sections=sections)
