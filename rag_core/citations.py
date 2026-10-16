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
    match: str = "none"                          # "exact" | "partial" | "numbers" | "none"

    @property
    def key(self) -> str:
        raw = f"{self.page}|{self.rects}"
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
        best = Highlight(n=n, page=pages[0] if pages else 1, sentence=evidence)
        rank = {"exact": 3, "partial": 2, "numbers": 1, "none": 0}
        for p in pages:
            if not 0 < p <= doc.page_count:
                continue
            rects, how = _search(doc[p - 1], evidence, _numbers(CITE_RE.sub("", " ".join(sents))))
            if rank[how] > rank[best.match]:
                best = Highlight(n, p, evidence, [tuple(r) for r in rects], how)
            if how == "exact":
                break
        out[n] = best
    return out


def _annotated(doc_path: str, h: Highlight) -> pymupdf.Document:
    src = _open(doc_path)
    one = pymupdf.open()
    one.insert_pdf(src, from_page=h.page - 1, to_page=h.page - 1)
    page = one[0]
    for r in h.rects:
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
