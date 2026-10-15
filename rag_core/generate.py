"""Grounded answer generation with numbered source citations."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import llm

NOT_FOUND = "The document does not contain this information."

SYSTEM = f"""You answer questions about a single document using ONLY the numbered context passages provided.

Rules:
1. Use only facts stated in the context. Never use outside knowledge, even if you know the answer.
2. Copy numbers exactly as written, with their units, qualifiers ("more than", "about") and time period
   (e.g. "2026f", "FY 2018", "fiscal year 2026/27").
3. Cite every fact with the passage number in square brackets, e.g. [2] or [1][3].
4. If passages disagree or cover different regions, years or configurations, say so and keep them apart.
   Never combine numbers from different regions, sections or tables.
5. If the context does not contain the answer, reply exactly: "{NOT_FOUND}"
6. Answer directly in 1-4 sentences, or a short list when the question asks for several items.
   Lead with the specific answer (the number, name or date), then brief supporting detail."""


@dataclass
class Generated:
    text: str
    cited: list[int]


def format_context(contexts: list[dict]) -> str:
    parts = []
    for i, c in enumerate(contexts, 1):
        where = " > ".join(c.get("section_path") or []) or "Front matter"
        pages = c.get("pages") or [c.get("page")]
        label = f" | {c['label']}" if c.get("label") else ""
        parts.append(f"[{i}] (page {', '.join(map(str, pages))} | {where}{label})\n{c['text']}")
    return "\n\n".join(parts)


def answer(question: str, contexts: list[dict]) -> Generated:
    prompt = f"Context passages:\n\n{format_context(contexts)}\n\nQuestion: {question}\nAnswer:"
    text = llm.generate(prompt, SYSTEM)
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(contexts)})
    return Generated(text=text, cited=cited)
