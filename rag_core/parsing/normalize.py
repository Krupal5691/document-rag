"""Text clean-up: ligatures, hyphenation, whitespace, and repair of words broken by dropped ligatures.

Many PDFs draw "fi", "fl", "ff", "ffi" as a single ligature glyph. Good extractors return the
ligature character (U+FB01 "ﬁ"), which NFKC normalisation turns back into "fi". Weaker extractors
drop the glyph, giving "signi cant" or "e ect". `LigatureRepair` fixes those by checking candidate
spellings against a vocabulary built from the document itself: if "significant" appears intact
elsewhere in the report, "signi cant" is confidently repaired.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

LIGATURES = ("ffi", "ffl", "ff", "fi", "fl", "ft", "th")
WORD_RE = re.compile(r"[A-Za-z]+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)                  # ﬁ -> fi, non-breaking space -> space
    text = text.replace("­", "")                           # soft hyphen
    text = re.sub(r"[ \t]*\t[ \t]*\n[ \t]*", ": ", text)        # "label<TAB>\nvalue" rows -> "label: value"
    text = re.sub(r"(\w)[-‐]\n(?=[a-z])", r"\1", text)     # re-join "hyphen-\nated" line breaks
    text = text.replace("\t", " ")
    text = re.sub(r"[  ]*\n[  ]*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)                 # single newlines are just line wraps
    text = re.sub(r"[  ]{2,}", " ", text)
    return text.strip()


class LigatureRepair:
    """Repairs 'signi cant' -> 'significant' and 'e ect' -> 'effect' using document vocabulary."""

    def __init__(self, corpus_text: str, min_count: int = 2):
        words = WORD_RE.findall(corpus_text)
        self.vocab = Counter(w.lower() for w in words)
        self.cs_vocab = Counter(words)            # case-sensitive counts
        self.min_count = min_count

    def _known(self, word: str) -> bool:
        return self.vocab.get(word.lower(), 0) >= self.min_count

    def _with_ligature(self, left: str, right: str) -> str | None:
        for lig in LIGATURES:
            if self._known(left + lig + right):
                return left + lig + right
        return None

    def repair(self, text: str) -> tuple[str, int]:
        """Returns (repaired text, number of fixes)."""
        tokens = re.split(r"(\s+)", text)           # words at even indexes, whitespace at odd
        out, fixes, i = [], 0, 0
        while i < len(tokens):
            tok = tokens[i]
            word = re.sub(r"^(\[\^\d+\])+", "", tok).strip(".,;:!?()[]\"'“”‘’—–-")
            # Case 0: "Te global economy" - a capital "Th" ligature lost its "h". The broken form is
            # frequent (it appears wherever the ligature was used), so vocabulary counts alone can't
            # catch it. But the ligature only affects capitalised words: if "te" / "tis" / "tese" never
            # (almost) never occur in lower case while "the" / "this" / "these" do, "Te" is a broken "The".
            # Real words like "Ten" survive because "ten" also appears in lower case.
            if len(word) >= 2 and word[0] == "T" and word[1:].islower() and word.isalpha():
                fixed = "Th" + word[1:]
                lower, fixed_lower = self.cs_vocab.get(word.lower(), 0), self.cs_vocab.get(fixed.lower(), 0)
                if fixed_lower >= 3 and lower * 50 < fixed_lower:
                    out.append(tok.replace(word, fixed))
                    fixes += 1
                    i += 1
                    continue
            if WORD_RE.fullmatch(word or "-") and not self._known(word):
                # Case 1: "signi cant" - the glyph was dropped and left a gap
                if i + 2 < len(tokens) and tokens[i + 1] == " ":
                    nxt = tokens[i + 2]
                    nxt_word = nxt.rstrip(".,;:!?)\"'")
                    joined = WORD_RE.fullmatch(nxt_word or "-") and self._with_ligature(word, nxt_word)
                    if joined and tok.endswith(word):
                        out.append(tok[: len(tok) - len(word)] + joined + nxt[len(nxt_word):])
                        fixes += 1
                        i += 3
                        continue
                # Case 2: "nancial" - the glyph was dropped at the start of a word, no gap
                if len(word) >= 3 and word.islower():
                    fixed = self._with_ligature("", word)
                    if fixed:
                        out.append(tok.replace(word, fixed))
                        fixes += 1
                        i += 1
                        continue
            out.append(tok)
            i += 1
        return "".join(out), fixes
