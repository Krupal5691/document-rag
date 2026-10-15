"""Parsing tests run against the real PDFs in data/ - no API key needed."""
import pymupdf
import pytest

from rag_core import documents
from rag_core.parsing.layout import read_pdf
from rag_core.parsing.normalize import LigatureRepair, normalize_text
from rag_core.parsing.structure import parse_pdf


@pytest.fixture(scope="module")
def gpo_pages():
    return read_pdf(documents.get("gpo").path)[1]


@pytest.fixture(scope="module")
def worldbank():
    return parse_pdf(documents.get("worldbank").path, "worldbank")


def test_two_column_reading_order(gpo_pages):
    """Naive extraction interleaves the two columns of page 4; XY-cut reads the left column first."""
    page4 = [b.text for b in gpo_pages[3]]
    first_body = next(t for t in page4 if len(t) > 80)
    assert first_body.startswith("The United States Government")
    naive = pymupdf.open(documents.get("gpo").path)[3].get_text(sort=True)   # top-to-bottom, left-to-right
    assert naive.index("Among the many highlights") < naive.index("posted")   # the problem we fix


def test_running_headers_removed(gpo_pages):
    text = "\n".join(b.text for page in gpo_pages for b in page)
    assert "A N N U A L R E P O R T" not in text


def test_normalize_ligatures_and_hyphenation():
    assert normalize_text("signiﬁcant eﬀect of in-\nflation") == "significant effect of inflation"


def test_ligature_repair_uses_document_vocabulary():
    repair = LigatureRepair("the " * 50 + "significant effect financial flows " * 3 + "ten " * 5 + "then " * 30)
    text, fixes = repair.repair("Te signi cant e ect on nancial ows. Ten men.")
    assert text == "The significant effect on financial flows. Ten men."
    assert fixes == 5


def test_th_ligature_fixed_in_world_bank(worldbank):
    summary = next(e for e in worldbank.elements if e.page == 19 and e.kind == "paragraph")
    assert summary.text.startswith("The global economy")


def test_region_section_path(worldbank):
    para = next(e for e in worldbank.elements if e.page == 96 and "Growth in SAR is expected" in e.text)
    assert "South Asia" in para.section_path
    assert para.section_path[-1] == "Outlook"


def test_table_rebuilt_with_rows(worldbank):
    table = next(e for e in worldbank.elements if e.kind == "table" and e.label == "Table 2.10")
    india = next(r for r in table.extra["rows"] if r.startswith("India"))
    assert "2026/27f = 6.6" in india


def test_imac_footnote_markers_and_endnotes():
    parsed = parse_pdf(documents.get("imac").path, "imac")
    energy = next(e for e in parsed.elements if "58 percent less energy" in e.text)
    assert "18" in energy.footnote_refs
    assert parsed.footnotes["18"].startswith("Energy consumption")
    assert not any("CO[^2]" in e.text for e in parsed.elements)        # subscript is not a footnote


def test_parent_sections_carry_footnotes():
    from rag_core.chunking import structured_chunks
    cfg = documents.get("imac")
    _, parents = structured_chunks(cfg, parse_pdf(cfg.path, "imac"))
    energy = next(p for p in parents.values() if "58 percent less energy" in p["text"])
    assert "143W Power Adapter" in energy["text"]          # footnote 18 travels with the section
