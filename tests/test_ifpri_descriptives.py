"""Do the IFPRI survey figures quoted on Slide 4 still hold?

Slide 4 is the evidence slide: it is where "shrimp disease costs money" stops being an assertion
and becomes a measurement. Three of its numbers are labelled *computed by me*, and until
`scripts/profile_ifpri.py` existed they were prose in a stage document that no code produced.

These tests pin the published claims against the regenerated table. If the survey extract is ever
replaced and a figure moves, the slide is wrong and this fails — which is the whole point.

The `dataset` marker applies: the IFPRI `.tab` modules are ~9 MB and are not redistributed with the
code, so these skip rather than fail when the data is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE = PROJECT_ROOT / "reports" / "tables" / "exp03a_ifpri_descriptives.csv"

# The values quoted in docs/stages/stage_03a_ifpri_profiling.md, ppt_content.md Slide 4,
# CLAUDE.md and docs/faculty_challenge/index.md. Hard-coded on purpose: if the data stops
# matching the documentation, the test must fail rather than quietly adopt the new value.
PUBLISHED = {
    "farms_in_survey": 1221,
    "farms_reporting_disease": 918,
    "farms_reporting_no_disease": 303,
    "farms_white_spot_disease": 492,
    "farms_black_gill": 54,
    "wsd_loss_mean_bdt": 23308,
    "wsd_loss_median_bdt": 15000,
    "wsd_loss_max_bdt": 250000,
    "mortality_sentinel_rows": 94,
}


@pytest.fixture(scope="module")
def descriptives() -> dict[str, float]:
    if not TABLE.exists():
        pytest.skip("run scripts/profile_ifpri.py (needs the IFPRI modules in data/raw/ifpri)")
    table = pd.read_csv(TABLE)
    return dict(zip(table["quantity"], table["value"]))


@pytest.mark.dataset
@pytest.mark.parametrize("quantity,expected", PUBLISHED.items())
def test_published_figure_still_holds(descriptives, quantity: str, expected: float):
    assert quantity in descriptives, f"{quantity} is quoted in the documentation but not computed"
    assert descriptives[quantity] == pytest.approx(expected), (
        f"{quantity}: documentation says {expected}, the data now says {descriptives[quantity]}"
    )


@pytest.mark.dataset
def test_the_disease_shares_are_internally_consistent(descriptives):
    """918 + 303 must be the whole survey, or one of the three numbers is wrong."""
    assert (descriptives["farms_reporting_disease"]
            + descriptives["farms_reporting_no_disease"]
            == descriptives["farms_in_survey"])
    assert descriptives["share_reporting_disease"] == pytest.approx(75.18, abs=0.01)


@pytest.mark.dataset
def test_the_sentinel_diagnosis_still_stands(descriptives):
    """`D-002`'s trap: 9997 in `c904` is a "don't know" code, not a mortality percentage.

    The evidence for calling it a code rather than a measurement is the **gap** — nothing lies
    between 101 and 9996. If a value ever appears in that range, 9997 might be a real reading and
    the cleaning rule would have to be revisited, so this is the load-bearing assertion.
    """
    assert descriptives["mortality_rows_between_101_and_9996"] == 0
    assert descriptives["mortality_sentinel_rows"] > 0
    # The size of the trap, which the documentation quotes as 32.5% versus 819.6%.
    assert descriptives["mortality_mean_pct_cleaned"] == pytest.approx(32.5, abs=0.1)
    assert descriptives["mortality_mean_pct_if_sentinel_kept"] == pytest.approx(819.6, abs=0.1)


@pytest.mark.dataset
def test_the_white_spot_subset_matches_the_track_a_sample(descriptives):
    """Slide 22's n=463 is a subset of these 492 farms; the two must not drift apart.

    492 farms report White Spot; 463 survive complete-case on the outcome and covariates. If the
    first number ever fell below the second, one of the two pipelines would be wrong.
    """
    diagnostics = PROJECT_ROOT / "reports" / "tables" / "exp03_diagnostics.csv"
    if not diagnostics.exists():
        pytest.skip("exp03 has not been run")
    n_modelled = pd.read_csv(diagnostics).query("model == 'disease'")["n"].iloc[0]
    assert n_modelled <= descriptives["farms_white_spot_disease"]


@pytest.mark.dataset
def test_every_row_names_its_source(descriptives):
    """A number in `reports/` that cannot say where it came from is not evidence.

    These descriptives are quoted on Slide 4 as the project's own measurement of the problem. If a
    panel member asks which survey question produced "75.2% of farms reported disease", the answer
    has to be in the table, not reconstructed from memory. Every row must name its source column.
    """
    table = pd.read_csv(TABLE)
    assert table["source"].notna().all()
    assert (table["source"].str.len() > 0).all()
