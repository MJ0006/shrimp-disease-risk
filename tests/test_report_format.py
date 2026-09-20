"""Tests for format, typography, structural integrity, and word count of the official report.

The official VIT Academic Project Report (Report_19MIS1127.docx) is a formal examination
deliverable for Review 3. It must satisfy strict academic and institutional invariants:

1. **Document Sections & Margins**:
   - Exactly 3 Word document sections:
     - Section 1: Unnumbered front matter (Cover, Title, Declaration, Certificate).
     - Section 2: Roman page numbering (i, ii, ...: Abstract, Ack, TOC, LOF, LOT, Acronyms).
     - Section 3: Arabic page numbering (1, 2, ...: Chapters 1-6, Appendices 1-4, References).
   - Strict VIT thesis margins on all sections:
     - Left margin: 3.81 cm (1.5 inches) for binding gutter.
     - Top, Right, Bottom margins: 2.54 cm (1.0 inch).

2. **Asset Sequence & Monotonicity**:
   - Exactly 14 figures (Figure 1 through Figure 14) appearing in strictly monotonic order in the text.
   - Exactly 24 empirical tables (Table 1 through Table 24) appearing in strictly monotonic order in the text.

3. **Academic Density & Volume**:
   - Body paragraph words >= 20,000 words.
   - Total document words (body + tables) >= 22,500 words.
   - Page budget heuristic P = round(words / 380 + 20.4) within [60, 100] pages (target: 75-85 pages).

4. **Institutional Identity**:
   - Candidate: G MOHAN JAYANTH (19MIS1127).
   - Guide: Dr. M. Premalatha (mandatory 'Dr.' prefix).
   - Verbatim registered VTOP title:
     'An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp Disease Detection Using Image'
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from docx.oxml.ns import qn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "report" / "Report_19MIS1127.docx"

REGISTERED_TITLE = (
    "An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp "
    "Disease Detection Using Image"
)


@pytest.fixture(scope="module")
def doc():
    if not REPORT_PATH.exists():
        pytest.skip(f"Report not built at {REPORT_PATH}; run scripts/build_report_docx.py")
    pytest.importorskip("docx")
    import docx
    return docx.Document(str(REPORT_PATH))


def test_document_sections_count(doc):
    """The document must contain exactly 3 sections representing front matter and body."""
    assert len(doc.sections) == 3, f"Expected exactly 3 sections, found {len(doc.sections)}"


def test_document_margins(doc):
    """All sections must adhere to VIT thesis margin requirements (Left 3.81cm, others 2.54cm)."""
    for idx, s in enumerate(doc.sections, start=1):
        # Left margin: 1.5 in (3.81 cm)
        assert abs(s.left_margin.inches - 1.5) < 0.05, (
            f"Section {idx} left margin is {s.left_margin.inches:.2f} in, expected 1.50 in (3.81 cm)"
        )
        # Top margin: 1.0 in (2.54 cm)
        assert abs(s.top_margin.inches - 1.0) < 0.05, (
            f"Section {idx} top margin is {s.top_margin.inches:.2f} in, expected 1.00 in (2.54 cm)"
        )
        # Right margin: 1.0 in (2.54 cm)
        assert abs(s.right_margin.inches - 1.0) < 0.05, (
            f"Section {idx} right margin is {s.right_margin.inches:.2f} in, expected 1.00 in (2.54 cm)"
        )
        # Bottom margin: 1.0 in (2.54 cm)
        assert abs(s.bottom_margin.inches - 1.0) < 0.05, (
            f"Section {idx} bottom margin is {s.bottom_margin.inches:.2f} in, expected 1.00 in (2.54 cm)"
        )


def test_figure_count_and_monotonicity(doc):
    """Exactly 14 figures must appear in strict monotonic sequence in body and List of Figures."""
    # 1. Body captions in doc.paragraphs
    body_fig_captions = [p.text.strip() for p in doc.paragraphs if re.match(r"^Figure\s+\d+:", p.text.strip())]
    assert len(body_fig_captions) == 14, f"Expected 14 body figure captions, found {len(body_fig_captions)}"

    fig_numbers = [int(re.search(r"^Figure\s+(\d+):", c).group(1)) for c in body_fig_captions]
    expected_numbers = list(range(1, 15))
    assert fig_numbers == expected_numbers, (
        f"Figure sequence in report body is non-monotonic or incomplete: {fig_numbers}"
    )

    # 2. List of Figures in doc.tables
    lof_table = None
    for tbl in doc.tables:
        hdr = [cell.text.strip() for cell in tbl.rows[0].cells]
        if hdr == ["Figure No.", "Figure Title / Caption Description", "Page No."]:
            lof_table = tbl
            break
    assert lof_table is not None, "List of Figures table not found in doc.tables"
    assert len(lof_table.rows) - 1 == 14, f"Expected 14 LOF rows, found {len(lof_table.rows) - 1}"
    lof_fig_numbers = [
        int(re.search(r"^Figure\s+(\d+)", r.cells[0].text.strip()).group(1))
        for r in lof_table.rows[1:]
    ]
    assert lof_fig_numbers == expected_numbers, (
        f"Figure sequence in LOF table is non-monotonic or incomplete: {lof_fig_numbers}"
    )


def test_table_count_and_monotonicity(doc):
    """Exactly 24 empirical tables must appear in strict monotonic sequence in body and List of Tables."""
    # 1. Body captions in doc.paragraphs
    body_tbl_captions = [p.text.strip() for p in doc.paragraphs if re.match(r"^Table\s+\d+:", p.text.strip())]
    assert len(body_tbl_captions) == 24, f"Expected 24 body table captions, found {len(body_tbl_captions)}"

    tbl_numbers = [int(re.search(r"^Table\s+(\d+):", c).group(1)) for c in body_tbl_captions]
    expected_numbers = list(range(1, 25))
    assert tbl_numbers == expected_numbers, (
        f"Table sequence in report body is non-monotonic or incomplete: {tbl_numbers}"
    )

    # 2. List of Tables in doc.tables
    lot_table = None
    for tbl in doc.tables:
        hdr = [cell.text.strip() for cell in tbl.rows[0].cells]
        if hdr == ["Table No.", "Table Title / Caption Description", "Page No."]:
            lot_table = tbl
            break
    assert lot_table is not None, "List of Tables table not found in doc.tables"
    assert len(lot_table.rows) - 1 == 24, f"Expected 24 LOT rows, found {len(lot_table.rows) - 1}"
    lot_tbl_numbers = [
        int(re.search(r"^Table\s+(\d+)", r.cells[0].text.strip()).group(1))
        for r in lot_table.rows[1:]
    ]
    assert lot_tbl_numbers == expected_numbers, (
        f"Table sequence in LOT table is non-monotonic or incomplete: {lot_tbl_numbers}"
    )


def test_frontmatter_tables_structure(doc):
    """All 4 front-matter sections must exist as formal tables with correct schemas and repeating headers."""
    expected_headers = {
        "TOC": ["Chapter / Section", "Title / Content Description", "Page No."],
        "LOF": ["Figure No.", "Figure Title / Caption Description", "Page No."],
        "LOT": ["Table No.", "Table Title / Caption Description", "Page No."],
        "Acronyms": ["Sl. No.", "Acronym / Symbol", "Expansion / Standard Definition"],
    }
    found = {}
    for tbl in doc.tables:
        hdr = [c.text.strip() for c in tbl.rows[0].cells]
        for key, exp_hdr in expected_headers.items():
            if hdr == exp_hdr:
                found[key] = tbl
                break

    assert len(found) == 4, f"Expected 4 front-matter tables, found {len(found)}: {list(found.keys())}"

    # Verify header properties
    for key, tbl in found.items():
        trPr = tbl.rows[0]._tr.get_or_add_trPr()
        tbl_headers = trPr.findall(qn("w:tblHeader"))
        assert len(tbl_headers) >= 1, f"Table '{key}' header row lacks w:tblHeader"

        cant_splits = trPr.findall(qn("w:cantSplit"))
        assert len(cant_splits) >= 1, f"Table '{key}' header row lacks w:cantSplit"

    # Specific assertions per front-matter table
    assert len(found["TOC"].rows) >= 50, f"TOC table has fewer rows than expected: {len(found['TOC'].rows)}"
    assert len(found["LOF"].rows) == 15, f"LOF table must have 1 header + 14 items, got {len(found['LOF'].rows)}"
    assert len(found["LOT"].rows) == 25, f"LOT table must have 1 header + 24 items, got {len(found['LOT'].rows)}"
    assert len(found["Acronyms"].rows) == 34, f"Acronyms table must have 1 header + 33 items, got {len(found['Acronyms'].rows)}"


def test_body_word_count(doc):
    """Paragraph body words must meet or exceed the 20,000 word threshold."""
    body_words = sum(len(p.text.split()) for p in doc.paragraphs)
    assert body_words >= 20000, (
        f"Report body word count {body_words} is below the mandatory 20,000 threshold."
    )


def test_total_word_count(doc):
    """Total document words (body + tables) must exceed 22,500 words."""
    body_words = sum(len(p.text.split()) for p in doc.paragraphs)
    table_words = 0
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                table_words += len(cell.text.split())
    total_words = body_words + table_words
    assert total_words >= 22500, (
        f"Total word count {total_words} is below the required 22,500 words."
    )


def test_page_budget_heuristic(doc):
    """Estimated page budget P = round(words / 380 + 20.4) must fall within [60, 100]."""
    body_words = sum(len(p.text.split()) for p in doc.paragraphs)
    est_pages = round(body_words / 380 + 20.4)
    assert 60 <= est_pages <= 100, (
        f"Estimated page count {est_pages} is outside the acceptable range [60, 100]."
    )


def test_identity_and_vtop_title(doc):
    """The document must accurately record student identity, guide title, and VTOP project title."""
    full_text = "\n".join(p.text for p in doc.paragraphs)

    assert "G MOHAN JAYANTH" in full_text or "G Mohan Jayanth" in full_text, (
        "Candidate name 'G Mohan Jayanth' missing from report."
    )
    assert "19MIS1127" in full_text, "Candidate registration number '19MIS1127' missing from report."
    assert "Dr. M. Premalatha" in full_text, (
        "Guide name 'Dr. M. Premalatha' (with mandatory 'Dr.' prefix) missing from report."
    )
    assert (REGISTERED_TITLE in full_text) or (REGISTERED_TITLE.upper() in full_text), (
        f"Verbatim registered VTOP title missing or altered: '{REGISTERED_TITLE}'"
    )
    assert "SCOPE" in full_text, "School acronym 'SCOPE' missing from report."
    assert "M.Tech" in full_text, "Degree 'M.Tech' missing from report."


# ---------------------------------------------------------------------------------------
# Pagination guards added 2026-09-20.
#
# The Table of Contents, List of Figures and List of Tables carried hand-typed page numbers
# that had never been measured: the TOC put References on page 68 in a document that Word
# paginates to 86. `python-docx` cannot paginate -- it writes a placeholder `Pages: 1` into
# docProps/app.xml -- so nothing in the build could have caught it. The numbers are now read
# out of Word by `.scratch/final/extract_pages.py` and injected into the builder.
#
# These tests cannot re-paginate either, so they check the two properties that would break
# first if the document grew or shrank: the page-number FORMAT, and internal consistency of
# the TOC's own ordering.
# ---------------------------------------------------------------------------------------

def test_front_matter_is_numbered_in_lowercase_roman(doc):
    """Section 2 must print i, ii, iii -- which is what the TOC claims for the front matter."""
    from docx.oxml.ns import qn

    styles = []
    for section in doc.sections:
        pg = section._sectPr.find(qn("w:pgNumType"))
        styles.append(None if pg is None else pg.get(qn("w:fmt")))

    assert styles[1] == "lowerRoman", (
        f"front matter is numbered {styles[1] or 'decimal (Word default)'}, but the Table of "
        f"Contents lists Abstract as page i. Word defaults every section to Arabic, so the "
        f"format has to be set explicitly."
    )
    assert styles[2] == "decimal", f"body should be Arabic, found {styles[2]}"


def test_toc_page_numbers_are_monotonic(doc):
    """A reader turning to a page number must not find the document going backwards."""
    import re

    toc = None
    for tbl in doc.tables:
        if [c.text.strip() for c in tbl.rows[0].cells] == [
            "Chapter / Section", "Title / Content Description", "Page No."
        ]:
            toc = tbl
            break
    assert toc is not None, "Table of Contents not found"

    arabic = []
    for row in toc.rows[1:]:
        raw = row.cells[2].text.strip()
        if re.fullmatch(r"\d+", raw):
            arabic.append((row.cells[0].text.strip(), int(raw)))

    assert len(arabic) >= 40, f"expected the whole body in the TOC, found {len(arabic)} entries"
    regressions = [
        f"{arabic[i][0]} (p{arabic[i][1]}) -> {arabic[i + 1][0]} (p{arabic[i + 1][1]})"
        for i in range(len(arabic) - 1)
        if arabic[i + 1][1] < arabic[i][1]
    ]
    assert not regressions, "TOC page numbers go backwards at: " + "; ".join(regressions)

    # The old hand-typed numbers stopped at 68 for References. If the last entry is still
    # below the count of body pages the document actually has, they were never re-measured.
    assert arabic[-1][1] >= 60, (
        f"the last TOC entry is page {arabic[-1][1]}; the report is ~86 pages, so these "
        f"numbers look like the unmeasured originals"
    )


def test_figure_and_table_lists_are_monotonic(doc):
    """List of Figures and List of Tables must run forwards too."""
    import re

    for header, label in (
        (["Figure No.", "Figure Title / Caption Description", "Page No."], "List of Figures"),
        (["Table No.", "Table Title / Caption Description", "Page No."], "List of Tables"),
    ):
        table = next(
            (t for t in doc.tables
             if [c.text.strip() for c in t.rows[0].cells] == header),
            None,
        )
        if table is None:
            continue
        pages = [
            int(r.cells[2].text.strip())
            for r in table.rows[1:]
            if re.fullmatch(r"\d+", r.cells[2].text.strip())
        ]
        assert pages == sorted(pages), f"{label} page numbers are not monotonic: {pages}"
