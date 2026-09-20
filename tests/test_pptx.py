"""Is the generated deck the deck the content file describes?

`Review2_19MIS1127.pptx` is the only artefact in this project that a panel marks directly, and it
is the one nobody can check while it is on screen. Three failure modes matter:

1. **It goes stale.** The content file is corrected, the deck is not rebuilt, and the slide says
   something `reports/` no longer supports. The headline test rebuilds the key figures from the
   CSVs and looks for them in the rendered text.
2. **Slide 1 is wrong.** The registered VTOP title must appear verbatim. `D-004` is explicit that a
   slide-1 mismatch is the single error that can cost all 25 marks, and it is the easiest thing in
   the deck to get subtly wrong.
3. **Content is lost in rendering.** The builder parses markdown; a parser bug silently drops a
   table or a diagram. Counts are asserted against the content file rather than trusted.

The deck is a build artefact. If it is absent these tests skip, because "not built here" is not
"broken" -- `scripts/build_pptx.py` regenerates it in about two seconds.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECK = PROJECT_ROOT / "Review2_19MIS1127.pptx"
CONTENT = PROJECT_ROOT / "docs" / "presentation" / "ppt_content.md"
TABLES = PROJECT_ROOT / "reports" / "tables"

# From CLAUDE.md, confirmed against VTOP by the student on 2026-08-26. Hard-coded deliberately:
# if the deck stops matching this, the test must fail rather than adopt whatever the deck says.
REGISTERED_TITLE = ("An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp "
                    "Disease Detection Using Image")


@pytest.fixture(scope="module")
def deck():
    if not DECK.exists():
        pytest.skip("deck not built; run scripts/build_pptx.py")
    pytest.importorskip("pptx")
    from pptx import Presentation
    return Presentation(DECK)


@pytest.fixture(scope="module")
def deck_text(deck) -> str:
    chunks = []
    for slide in deck.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                chunks.append(shape.text_frame.text)
            if shape.has_table:
                for row in shape.table.rows:
                    chunks.extend(cell.text for cell in row.cells)
    return "\n".join(chunks)


def _table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
# Slide 1 -- the one that can cost every mark
# --------------------------------------------------------------------------------------

def test_slide_one_carries_the_registered_title_verbatim(deck):
    first = " ".join(shape.text_frame.text for shape in deck.slides[0].shapes
                     if shape.has_text_frame)
    assert REGISTERED_TITLE in first, (
        "slide 1 does not carry the VTOP-registered title verbatim (D-004)"
    )


def test_slide_one_identifies_the_student_and_the_guide(deck):
    """`CLAUDE.md`: the "Dr." prefix is required on every deliverable."""
    first = " ".join(shape.text_frame.text for shape in deck.slides[0].shapes
                     if shape.has_text_frame)
    for required in ("19MIS1127", "G Mohan Jayanth", "M. Premalatha"):
        assert required in first, f"slide 1 is missing {required}"
    assert re.search(r"Dr\.?\s*M\.\s*Premalatha", first), "the guide's Dr. prefix is missing"


# --------------------------------------------------------------------------------------
# The deck matches the content file it was rendered from
# --------------------------------------------------------------------------------------

def test_every_slide_in_the_content_file_reached_the_deck(deck):
    """A parser that silently skips a heading loses a required section of the prescribed
    format. Continuation slides are permitted, so the deck may be longer -- never shorter."""
    content = CONTENT.read_text(encoding="utf-8")
    headings = [h for h in re.findall(r"^## (Slide [^\n]*)$", content, flags=re.M)
                if re.match(r"Slide\s+A?\d+", h)]
    assert len(deck.slides) >= len(headings), (
        f"{len(headings)} slides in the content file, {len(deck.slides)} in the deck"
    )


def test_every_slide_has_speaker_notes(deck):
    """Master Prompt Section 27 requires notes for every slide. An empty pane during a rehearsal reads
    as an oversight, so continuation slides carry a pointer rather than nothing."""
    empty = [i + 1 for i, slide in enumerate(deck.slides)
             if not (slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip())]
    assert not empty, f"slides without speaker notes: {empty}"


def test_the_tables_and_figures_survived_rendering(deck):
    """18 markdown tables and 7 figure markers are in the content file. A parser regression
    would drop them silently and the deck would still open."""
    tables = sum(1 for slide in deck.slides for shape in slide.shapes if shape.has_table)
    images = sum(1 for slide in deck.slides for shape in slide.shapes if shape.shape_type == 13)
    assert tables >= 18, f"only {tables} tables rendered"
    assert images >= 6, f"only {images} figures embedded"


# --------------------------------------------------------------------------------------
# The deck is current
# --------------------------------------------------------------------------------------

def test_headline_numbers_on_the_slides_match_the_tables(deck_text: str):
    """Rebuild each headline from `reports/tables/` and find it in the rendered deck. This is
    what fails when an experiment is re-run and the deck is not rebuilt."""
    exp01 = _table("exp01_leakage_summary.csv")
    exp02 = _table("exp02_summary.csv")
    exp04 = _table("exp04_summary.csv")

    exposure = exp01.loc[exp01["protocol"] == "image_level", "pct_test_leaked_mean"].iloc[0]
    b = exp02.loc[exp02["condition"] == "B_image_level", "test_f1_macro_mean"].iloc[0]
    c = exp02.loc[exp02["condition"] == "C_specimen_level", "test_f1_macro_mean"].iloc[0]
    best = exp04.sort_values("f1_macro_mean", ascending=False).iloc[0]

    for expected in (f"{exposure:.2f}", f"{b:.4f}", f"{c:.4f}", f"{b - c:.4f}",
                     f"{best['f1_macro_mean']:.4f}"):
        assert expected in deck_text, (
            f"{expected} is in reports/tables/ but not on any slide. "
            "Rebuild: scripts/build_pptx.py"
        )


def test_the_withdrawn_significance_claim_is_not_on_any_slide(deck_text: str):
    """`D-027` withdrew the "orders of magnitude above noise" inference. It must not survive in
    a rendered deck after being removed from the content file."""
    assert "orders of magnitude" not in deck_text.lower()


def test_no_slide_claims_sample_level_fusion(deck_text: str):
    """`D-008` is a binding constraint: no dataset pairs a photograph with that farm's
    covariates, so the fusion is decision-level and saying otherwise is false."""
    low = deck_text.lower()
    # The phrase is *allowed* where the deck rules it out -- Slide 10 lists it under "out of
    # scope, deliberately", which is the project stating its own limit and is worth more than
    # silence. The test therefore requires a disclaimer nearby, not the absence of the words.
    denials = ("not", "never", "impossible", "cannot", "out of scope", "no dataset",
               "decision-level", "rather than", "excluded")
    for index in (m.start() for m in re.finditer("sample-level", low)):
        window = low[max(0, index - 160):index]
        assert any(word in window for word in denials), (
            "a slide appears to assert sample-level fusion (D-008): " + repr(window[-90:])
        )


def test_every_literature_paper_reached_a_slide(deck):
    """The prescribed format requires 20-25 papers in a five-column table. Counting tables is not
    enough: twice during this deck's construction a whole literature table was dropped by the
    renderer while the deck-wide table count still cleared its threshold. This checks the S.No
    values themselves, so a missing table is a missing paper and the test says which."""
    # Only rows of the literature tables -- other slides carry numeric tables of their own.
    content = CONTENT.read_text(encoding="utf-8")
    expected, in_literature = set(), False
    for line in content.splitlines():
        if line.startswith("## Slide"):
            in_literature = "literature review" in line.lower()
        if in_literature and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0].isdigit():
                expected.add(cells[0])

    found = set()
    for slide in deck.slides:
        for shape in slide.shapes:
            if not shape.has_table:
                continue
            header = [cell.text.strip().lower() for cell in shape.table.rows[0].cells]
            if header and header[0].startswith("s.no"):
                found.update(row.cells[0].text.strip() for row in list(shape.table.rows)[1:])

    missing = sorted(expected - found, key=int)
    assert not missing, f"literature rows in the content file but not on any slide: {missing}"
    assert 20 <= len(found) <= 25, (
        f"the instructions require 20-25 papers; the deck shows {len(found)}"
    )


def test_the_prescribed_section_order_is_followed(deck):
    """`Review 2 Important Instructions V1.docx`: Title, Outline, then Introduction, Literature
    Review, Scope & Problem Statement, Challenges, Objectives, Architecture, Results, Conclusion,
    Limitations, References -- in that order."""
    titles = []
    for slide in deck.slides:
        # The slide-number tag shares the header band, so take the longer of the two.
        tops = sorted((sh for sh in slide.shapes
                       if sh.has_text_frame and sh.top is not None and sh.top < 914400
                       and sh.text_frame.text.strip()),
                      key=lambda sh: len(sh.text_frame.text), reverse=True)
        titles.append(tops[0].text_frame.text.lower() if tops else "")

    def first_index(needle: str) -> int:
        return next((i for i, t in enumerate(titles) if needle in t), -1)

    outline = first_index("outline")
    introduction = first_index("introduction")
    literature = first_index("literature review")
    problem = first_index("problem statement")
    objectives = first_index("research objectives")
    results = first_index("result 1")
    conclusion = first_index("conclusion")
    references = first_index("references")

    order = [outline, introduction, literature, problem, objectives, results,
             conclusion, references]
    assert all(i >= 0 for i in order), f"a prescribed section is missing: {order}"
    assert order == sorted(order), (
        "slides are not in the order the instructions prescribe: "
        f"outline={outline}, introduction={introduction}, literature={literature}, "
        f"problem={problem}, objectives={objectives}, results={results}, "
        f"conclusion={conclusion}, references={references}"
    )
