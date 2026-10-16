"""Clickable, verifiable citations: find the exact sentence behind each [n] and highlight it in the PDF.

For every citation [n] in an answer:
  1. take the answer sentence(s) that cite [n]
  2. pick the sentence in passage n that best supports them (word overlap, numbers weighted heavily -
     a number is the strongest evidence of where a fact came from)
  3. locate that sentence on the passage's page(s) with PyMuPDF. Our text was cleaned during ingestion
     (ligatures repaired, line breaks joined), so it may not match the raw PDF text exactly; if the whole
     sentence is not found we search overlapping 5-word windows and keep the hits.
  4. render the page with highlight annotations, as an image (inline) or a one-page PDF (link).

Passages built by the vision model (charts) have no literal text on the page; for those we highlight
the key numbers instead, and if nothing matches we still point to the right page.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

import pymupdf

from .chunking import split_sentences

SEARCH_FLAGS = pymupdf.TEXT_DEHYPHENATE          # NOT preserving ligatures, so "fl" matches "ﬂ"
STOP = set("the a an of in on for to and or is are was were be by with as at from that this its it".split())
CITE_RE = re.compile(r"\[(\d+)\]")
NUM_RE = re.compile(r"\d[\d,.]*%?")


@dataclass
class Highlight:
    n: int                                       # citation number in the answer
    page: int                                    # 1-based page where the evidence was found
    sentence: str                                # evidence sentence from the passage
    rects: list[tuple[float, float, float, float]] = field(default_factory=list)
    match: str = "none"                          # "cells" | "exact" | "partial" | "numbers" | "none"
    label_rects: list[tuple[float, float, float, float]] = field(default_factory=list)   # table row labels

    @property
    def key(self) -> str:
        raw = f"{self.page}|{self.rects}|{self.label_rects}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]+|\d[\d,.]*", text.lower()) if t not in STOP]


def _clean(text: str) -> str:
    text = re.sub(r"\[\^\d+\]", "", text)                    # footnote markers
    text = re.sub(r"^\[[^\]]*\]\s*", "", text)               # context header
    return re.sub(r"\s+", " ", text).strip()


def _score(claim: list[str], sentence: str) -> float:
    toks = set(_tokens(sentence))
    return sum(3.0 if c[0].isdigit() else 1.0 for c in set(claim) if c in toks)


def best_sentence(claim: str, passage: str) -> str:
    sentences = [s for s in split_sentences(_clean(passage).replace("\n", " ")) if len(s) > 15] or [_clean(passage)]
    claim_toks = _tokens(CITE_RE.sub("", claim))
    return max(sentences, key=lambda s: _score(claim_toks, s))


def _numbers(text: str) -> list[str]:
    return list(dict.fromkeys(n.rstrip(".,") for n in NUM_RE.findall(text) if len(n.rstrip(".,")) >= 2))


def _search(page: pymupdf.Page, sentence: str, claim_numbers: list[str] | None = None) -> tuple[list, str]:
    hits = page.search_for(sentence, flags=SEARCH_FLAGS)
    if hits:
        return hits, "exact"
    words = sentence.split()
    rects, windows, found = [], 0, 0
    for i in range(0, max(len(words) - 4, 1), 3):            # overlapping 5-word windows
        window = " ".join(words[i:i + 5])
        if len(window) < 12:
            continue
        windows += 1
        w_hits = page.search_for(window, flags=SEARCH_FLAGS)
        if w_hits:
            found += 1
            rects.extend(w_hits)
    if windows and found / windows >= 0.4:
        return rects, "partial"
    # Last resort (e.g. a chart transcribed by the vision model): highlight the numbers the answer
    # actually cites, or every number in the evidence if the answer cites none of them.
    nums = _numbers(sentence)
    cited = [n for n in (claim_numbers or []) if n in nums]
    rects = [r for n in (cited or nums) for r in page.search_for(n, flags=SEARCH_FLAGS)]
    return (rects, "numbers") if rects else ([], "none")


# ------------------------------------------------------------------------------- table cells

YEAR_RE = re.compile(r"^(19|20)\d\d([/–-]\d{2,4})?[a-z]?$")
VALUE_RE = re.compile(r"[-−–]?\d[\d,]*(?:\.\d+)?%?(?:/\d{2,4})?[a-z]?")


def _year(tok: str) -> str:
    return re.sub(r"[a-z]$", "", tok.lower())        # "2026/27f" (forecast) and "2026/27" name the same column


def _claim_cues(claim: str) -> tuple[list[str], set[str], set[str]]:
    """Split the claim into cited values (346, 50%, 4.6), year cues that name a column (2026/27f) and
    words that may name a row (production, bangladesh). Numbers glued to letters ("256GB", "CO2e") are
    product names or units, not cited quantities."""
    claim = CITE_RE.sub("", unicodedata.normalize("NFKC", claim))
    values, years = [], set()
    for m in VALUE_RE.finditer(claim):
        tok = m.group(0).rstrip(".,").replace("−", "-").replace("–", "-")
        nxt = claim[m.end():m.end() + 1]
        if YEAR_RE.match(tok):
            years.add(_year(tok))
        elif nxt.isalpha() or (m.start() > 0 and claim[m.start() - 1].isalpha()):
            continue
        elif len(tok.lstrip("-")) >= 2 or "." in tok:
            values.append(tok)
    words = {w for w in _tokens(claim) if w.isalpha() and len(w) >= 3}
    return list(dict.fromkeys(values)), years, words


def _word(w: str) -> str:
    return unicodedata.normalize("NFKC", w).strip("()[],;:*\u2022").replace("−", "-").replace("–", "-")


def _is_value(word: str, value: str) -> bool:
    w = _word(word)
    if w == value:
        return True
    rest = w[len(value):]                       # "512GB", "346kg" - a unit glued to the number
    return w.startswith(value) and rest.isalpha() and len(rest) <= 3


def _table_cells(page: pymupdf.Page, claim_sentences: list[str]):
    """Value cells the claim cites, each paired with its row label. When a value occurs more than once
    on the page, the occurrence whose row label (words to its left) and column header (words above it)
    best match the claim wins; an ambiguous value with no matching label is skipped."""
    values, _, _ = _claim_cues(" ".join(claim_sentences))
    if not values:
        return [], []
    words = page.get_text("words", flags=SEARCH_FLAGS)
    value_rects, label_rects = [], []
    for v in values:
        # score against the sentence(s) that cite this value, not the whole answer
        own = [s for s in claim_sentences if v in _claim_cues(s)[0]] or claim_sentences
        _, years, claim_words = _claim_cues(" ".join(own))
        scored = []
        for o in (w for w in words if _is_value(w[4], v)):
            yc, h = (o[1] + o[3]) / 2, o[3] - o[1]
            row = [w for w in words if abs((w[1] + w[3]) / 2 - yc) < h * 0.6 and w[2] <= o[0] + 1
                   and o[0] - w[2] < page.rect.width * 0.85]
            label = [w for w in row if not any(ch.isdigit() for ch in w[4]) and len(_word(w[4])) > 1]
            col = [w for w in words if w[3] <= o[1] + 1 and min(w[2], o[2]) - max(w[0], o[0]) > 0]
            row_score = len({_word(w[4]).lower() for w in label} & claim_words)
            col_score = len({_year(_word(w[4])) for w in col} & years) * 2 \
                + 0.5 * len({_word(w[4]).lower() for w in col} & claim_words)
            scored.append((row_score * 2 + col_score, o, label))
        if not scored:
            continue
        best = max(s[0] for s in scored)
        keep = [s for s in scored if s[0] == best] if best > 0 else (scored if len(scored) == 1 else [])
        for _, o, label in keep:
            value_rects.append(tuple(o[:4]))
            label_rects.extend(tuple(w[:4]) for w in label)
    return value_rects, list(dict.fromkeys(label_rects))


def _tabular(ctx: dict, evidence: str) -> bool:
    return ctx.get("kind") in ("table", "figure", "callout") or len(_numbers(evidence)) >= 4


@lru_cache(maxsize=4)
def _open(path: str) -> pymupdf.Document:
    return pymupdf.open(path)


def locate(answer: str, contexts: list[dict], pdf_path) -> dict[int, Highlight]:
    """Highlight for every citation number used in the answer."""
    doc = _open(str(pdf_path))
    claims: dict[int, list[str]] = {}
    for sent in split_sentences(answer.replace("\n", " ")):
        for n in {int(x) for x in CITE_RE.findall(sent)}:
            claims.setdefault(n, []).append(sent)
    out = {}
    for n, sents in claims.items():
        if not 0 < n <= len(contexts):
            continue
        ctx = contexts[n - 1]
        evidence = best_sentence(" ".join(sents), ctx["text"])
        pages = [p for p in (ctx.get("pages") or [ctx.get("page")]) if p]
        claim = " ".join(sents)
        best_labels: list = []
        best = Highlight(n=n, page=pages[0] if pages else 1, sentence=evidence)
        rank = {"cells": 4, "exact": 3, "partial": 2, "numbers": 1, "none": 0}
        for p in pages:
            if not 0 < p <= doc.page_count:
                continue
            rects, how = _search(doc[p - 1], evidence, _numbers(CITE_RE.sub("", claim)))
            # Evidence that is a table row / chart transcription (not a prose sentence found verbatim):
            # highlight the exact cells the answer cites instead of whole table lines.
            if how != "exact" and _tabular(ctx, evidence):
                cells, labels = _table_cells(doc[p - 1], sents)
                if cells:
                    rects, how = cells, "cells"
                    best_labels = labels
            if rank[how] > rank[best.match]:
                best = Highlight(n, p, evidence, [tuple(r) for r in rects], how,
                                 best_labels if how == "cells" else [])
            if how in ("exact", "cells"):
                break
        out[n] = best
    return out


def _annotated(doc_path: str, h: Highlight) -> pymupdf.Document:
    src = _open(doc_path)
    one = pymupdf.open()
    one.insert_pdf(src, from_page=h.page - 1, to_page=h.page - 1)
    page = one[0]
    for r in h.label_rects:                                  # row labels in light blue
        annot = page.add_highlight_annot(pymupdf.Rect(r))
        annot.set_colors(stroke=(0.62, 0.82, 1.0))
        annot.update()
    for r in h.rects:                                        # evidence / value cells in yellow
        page.add_highlight_annot(pymupdf.Rect(r))
    return one


def page_png(doc_path, h: Highlight, dpi: int = 110) -> bytes:
    """The cited page as an image with the evidence highlighted."""
    return _annotated(str(doc_path), h)[0].get_pixmap(dpi=dpi).tobytes("png")


def page_pdf(doc_path, h: Highlight) -> bytes:
    """The cited page as a one-page PDF with highlight annotations (opens in any PDF viewer)."""
    return _annotated(str(doc_path), h).tobytes()


def link_citations(answer: str, urls: dict[int, str], titles: dict[int, str] | None = None) -> str:
    """Turn [n] in the answer into Markdown links."""
    titles = titles or {}

    def repl(m):
        n = int(m.group(1))
        if n not in urls:
            return m.group(0)
        title = titles.get(n, "")
        return f'[[{n}]]({urls[n]}{" " + chr(34) + title + chr(34) if title else ""})'
    return CITE_RE.sub(repl, answer)
