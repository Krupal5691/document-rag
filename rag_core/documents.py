"""Per-document settings. Everything document-specific lives here; the pipeline code is generic."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config


@dataclass(frozen=True)
class DocConfig:
    id: str
    title: str
    file: str
    description: str                    # told to the query-rewriting LLM so it knows what the corpus holds
    facets: dict[str, list[str]] = field(default_factory=dict)       # filterable values, e.g. region
    aliases: dict[str, str] = field(default_factory=dict)            # "SAR" -> "South Asia"
    vision: str | None = None           # None | "all" (send every page image to the vision model)
    child_chars: int = 450              # max characters per child chunk (sentence-aware)
    parent_chars: int = 2400            # max characters per parent section chunk
    overlap_sentences: int = 1

    @property
    def path(self):
        return config.DATA_DIR / self.file


REGIONS = [
    "East Asia and Pacific",
    "Europe and Central Asia",
    "Latin America and the Caribbean",
    "Middle East, North Africa, Afghanistan, and Pakistan",
    "South Asia",
    "Sub-Saharan Africa",
]

DOCUMENTS: dict[str, DocConfig] = {
    "gpo": DocConfig(
        id="gpo",
        title="GPO 2018 Annual Report",
        file="GPO-Anual-report.pdf",
        description=(
            "U.S. Government Publishing Office (GPO) annual report for fiscal year 2018: message from the "
            "Acting Deputy Director, work for Congress and federal agencies, the Federal Depository Library "
            "Program (FDLP), govinfo (successor to FDsys), 'GPO by the Numbers' statistics, finances, and "
            "audited financial statements."
        ),
    ),
    "worldbank": DocConfig(
        id="worldbank",
        title="World Bank Global Economic Prospects, June 2026",
        file="GEP-Jun-2026.pdf",
        description=(
            "World Bank Global Economic Prospects report, June 2026 (200 pages). Executive Summary; Chapter 1 "
            "Global Outlook (with Box 1.1 on AI and growth, Box 1.2 on private investment); Chapter 2 Regional "
            "Outlooks for six regions; Chapter 3 sovereign debt levels and interest rates in emerging market "
            "and developing economies (EMDEs), Box 3.1 hidden debt; Chapter 4 fiscal policy and commodity "
            "price swings; Statistical Annex. Forecasts use e = estimate, f = forecast."
        ),
        facets={"region": REGIONS},
        aliases={
            "SAR": "South Asia", "EAP": "East Asia and Pacific", "ECA": "Europe and Central Asia",
            "LAC": "Latin America and the Caribbean", "SSA": "Sub-Saharan Africa",
            "MNA": "Middle East, North Africa, Afghanistan, and Pakistan",
            "MENA": "Middle East, North Africa, Afghanistan, and Pakistan",
            "MENAAP": "Middle East, North Africa, Afghanistan, and Pakistan",
            "Southeast Asia": "East Asia and Pacific", "Africa": "Sub-Saharan Africa",
            "Latin America": "Latin America and the Caribbean", "Middle East": "Middle East, North Africa, Afghanistan, and Pakistan",
        },
    ),
    "imac": DocConfig(
        id="imac",
        title="Apple iMac Product Environmental Report (Oct 2024)",
        file="iMac_PER_Oct2024.pdf",
        description=(
            "Apple iMac Product Environmental Report, October 2024 (12 pages): recycled materials (aluminum, "
            "copper, gold, tin, plastic), carbon footprint and emissions breakdown (production, transport, "
            "use, end of life), energy efficiency vs ENERGY STAR, power consumption table, packaging, "
            "supplier programs, smarter chemistry, recycling and Apple Trade In, definitions and endnotes. "
            "Most facts sit in charts, icon call-outs, small tables and numbered footnotes."
        ),
        vision="all",
        child_chars=400,
        parent_chars=1800,
    ),
}


def get(doc_id: str) -> DocConfig:
    if doc_id not in DOCUMENTS:
        raise KeyError(f"Unknown document '{doc_id}'. Choose from: {', '.join(DOCUMENTS)}")
    return DOCUMENTS[doc_id]
