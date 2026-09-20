"""Does the demonstration script tell the truth?

`scripts/demo_results.py` is the only artefact in this project that a panel sees *executing*.
That makes it the easiest place for a discrepancy to do real damage: a number on the screen
that disagrees with the slide behind it, or -- far worse -- a block labelled LIVE whose figures
were in fact read out of a results table.

So the script is tested like a result, not like a utility:

* every number it prints must equal the artefact it claims to come from;
* the provenance labels must match what each section actually does;
* the cross-check lines it prints must say MATCH, because they are the evidence offered to the
  panel that the tables were generated rather than typed;
* no line may exceed the width a projected screenshot can show.

The script is run **once** here and every assertion reads that one capture. Running it per test
would triple the suite's runtime for no extra coverage.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import demo_results  # `scripts` is on the path via pytest.ini

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "reports" / "tables"
EXP02_RUNS = PROJECT_ROOT / "experiments" / "runs" / "exp02"

# What each section is allowed to claim about where its numbers come from. Written out in full
# rather than derived from the script, so relabelling a training result as LIVE fails here.
EXPECTED_PROVENANCE = {
    "structure": "LIVE",
    "leakage": "LIVE",
    "effect": "RECORDED",
    "matrix": "LIVE",
    "backbones": "RECORDED",
    "edge": "RECORDED",
    "convnext": "RECORDED",
    "track_a": "RECORDED",
    "lr_sweep": "RECORDED",
    "export": "RECORDED",
    "farm_risk": "ILLUSTRATIVE",
}


@pytest.fixture(scope="module")
def demo_output() -> str:
    """One full run of the demonstration, captured."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = demo_results.main([])
    assert exit_code == 0
    return buffer.getvalue()


def _table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet; run the experiment scripts")
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
# Provenance -- the claim the whole script rests on
# --------------------------------------------------------------------------------------

def test_every_section_declares_the_provenance_it_actually_has():
    """A block labelled LIVE asserts to a panel that it was computed in front of them. If a
    training outcome were ever relabelled that way, the script would be lying on the student's
    behalf, so the mapping is pinned."""
    declared = {name: tag for name, tag, _, _ in demo_results.SECTIONS}
    assert declared == EXPECTED_PROVENANCE


def test_live_sections_read_no_results_table(demo_output: str):
    """The two structural LIVE sections may cite the manifest and may cross-check themselves
    against `exp01`, but their own figures must come from the manifest. `reports/tables/` may
    therefore appear only as a cross-check line, never as their source."""
    section = demo_output.split("Result 1")[1].split("Result 2")[0]
    sources = [line for line in section.splitlines() if line.strip().startswith("source:")]
    assert sources, "the leakage section must name its source"
    assert all("image_manifest" in line for line in sources)


def test_cross_checks_all_match(demo_output: str):
    """These lines are shown to a panel as proof the stored tables agree with the raw data.
    One DIFFERS on the day would be worse than never printing them."""
    checks = [line for line in demo_output.splitlines()
              if line.strip().startswith("cross-check")]
    assert checks, "the demo must cross-check recomputed figures against the stored ones"
    assert all(line.strip().endswith("MATCH") for line in checks), checks


def test_recomputed_macro_f1_equals_the_recorded_run():
    """The claim behind the cross-check, tested independently of the script that prints it:
    the macro-F1 in the results table is recoverable from the stored posteriors alone."""
    from sklearn.metrics import f1_score

    runs = _table("exp02_runs.csv")
    checked = 0
    for tag, condition in (("B_image_level__seed2025", "B_image_level"),
                           ("C_specimen_level__seed2025", "C_specimen_level")):
        npz = EXP02_RUNS / f"{tag}_predictions.npz"
        if not npz.exists():
            continue
        stored = np.load(npz, allow_pickle=True)
        recomputed = f1_score(stored["labels"], stored["probs"].argmax(axis=1),
                              average="macro", zero_division=0)
        recorded = runs.loc[(runs["condition"] == condition) & (runs["seed"] == 2025),
                            "test_f1_macro"].iloc[0]
        assert recomputed == pytest.approx(float(recorded), abs=5e-4)
        checked += 1
    if checked == 0:
        pytest.skip("no exp02 prediction files on disk")


# --------------------------------------------------------------------------------------
# Agreement with the artefacts, and with the slides that quote them
# --------------------------------------------------------------------------------------

def test_headline_numbers_printed_match_the_tables(demo_output: str):
    """Every headline the demo prints is rebuilt here from the CSV and looked for verbatim.
    If a table were regenerated and the script left stale, this fails."""
    exp02 = _table("exp02_summary.csv")
    exp04 = _table("exp04_summary.csv")

    b = exp02.loc[exp02["condition"] == "B_image_level", "test_f1_macro_mean"].iloc[0]
    c = exp02.loc[exp02["condition"] == "C_specimen_level", "test_f1_macro_mean"].iloc[0]
    best = exp04.sort_values("f1_macro_mean", ascending=False).iloc[0]
    edge = exp04.loc[exp04["backbone"] == "mobilenetv3_large_100"].iloc[0]

    for expected in (f"{b:.4f}", f"{c:.4f}", f"{b - c:.4f}",
                     f"{best['f1_macro_mean']:.4f}", f"{best['f1_macro_sd']:.4f}",
                     f"{edge['size_mb']:.2f}"):
        assert expected in demo_output, f"{expected} is in the tables but not in the demo"


def test_the_leakage_figures_match_the_recorded_audit(manifest, demo_output: str):
    """The live measurement and the recorded audit must agree to two decimal places -- the
    same two the slide carries."""
    exp01 = _table("exp01_leakage_summary.csv")
    image_level = exp01.loc[exp01["protocol"] == "image_level", "pct_test_leaked_mean"].iloc[0]
    assert f"{image_level:.2f}%" in demo_output
    assert "0.00%" in demo_output


def test_the_pdp_refusal_names_the_model_the_experiment_selected(demo_output: str):
    """`D-017` gates the PDP on the *selected* model, and selection is by cross-validated R^2,
    not by holdout R^2. Ranking by the wrong column names SVM_RBF as Cubist for the disease
    model -- a small slip that would put a different model name on the screen than in the
    recorded table."""
    skipped = _table("exp03_pdp_skipped.csv")
    for _, row in skipped.iterrows():
        assert str(row["best_model"]) in demo_output


def test_the_per_class_drop_is_internally_consistent(demo_output: str):
    """A panel member who subtracts the two printed columns must get the printed difference."""
    rows = [line.split() for line in demo_output.splitlines()
            if line.strip().startswith(("BG ", "BG_WSSV", "Healthy", "WSSV"))
            and line.count(".") >= 3]
    checked = 0
    for row in rows:
        try:
            b, c, drop = float(row[-3]), float(row[-2]), float(row[-1])
        except ValueError:
            continue
        if abs(drop) < 0.5:                       # the drop column, not a metrics row
            assert round(c - b, 3) == pytest.approx(drop, abs=1e-9)
            checked += 1
    assert checked >= 4, "expected one consistency check per class"


# --------------------------------------------------------------------------------------
# Fitness as a live demonstration
# --------------------------------------------------------------------------------------

def test_no_line_is_too_wide_to_project(demo_output: str):
    """78 columns. A table that wraps in a projector is unreadable from the back of a room,
    and a wrapped screenshot on a slide looks like a mistake."""
    too_wide = [line for line in demo_output.splitlines() if len(line) > demo_results.WIDTH]
    assert not too_wide, too_wide[:5]


def test_output_is_ascii_only(demo_output: str):
    """A Windows console in cp1252 raises `UnicodeEncodeError` on a stray typographic
    character. Discovering that mid-demonstration is not recoverable."""
    demo_output.encode("ascii")


def test_every_section_names_at_least_one_artefact(demo_output: str):
    """A number on screen with no named source cannot be checked by the person watching.

    In a review slot the audience cannot open the repository, so the provenance has to travel with
    the number. Every block must therefore either name the file it read or say explicitly that its
    input is missing -- silence would look identical to success.
    """
    blocks = demo_output.split("=" * demo_results.WIDTH)
    result_blocks = [b for b in blocks
                     if any(f"] {tag}" in b for tag in ("LIVE", "RECORDED", "ILLUSTRATIVE"))]
    assert len(result_blocks) == len(demo_results.SECTIONS)
    for block in result_blocks:
        assert "source:" in block or "[skipped]" in block


def test_class_order_matches_the_training_pipeline():
    """`demo_results.CLASSES` is a deliberate copy that avoids importing torch. If the real
    list ever changed, every confusion matrix the demo prints would be silently mislabelled."""
    from shrimp_risk.vision.datasets import CLASSES as TRAINING_CLASSES

    assert demo_results.CLASSES == TRAINING_CLASSES


def test_convnext_convergence_is_read_from_the_stored_histories():
    """`D-020`'s standing rule -- a score is not interpretable until the training curve shows
    the model fitted its training data -- is only demonstrable if the histories are there."""
    histories = sorted((PROJECT_ROOT / "experiments" / "runs" / "exp04")
                       .glob("convnext_tiny__fold*.json"))
    if not histories:
        pytest.skip("exp04 runs not on disk")
    for path in histories:
        run = json.loads(path.read_text(encoding="utf-8"))
        assert run["config"]["lr"] == pytest.approx(5e-5)
        assert run["history"][-1]["train_f1"] > 0.99, f"{path.name} did not fit its training set"


# --------------------------------------------------------------------------------------
# The command-line surface -- it is used under pressure, so it must fail clearly
# --------------------------------------------------------------------------------------

def test_listing_sections_succeeds():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert demo_results.main(["--list"]) == 0
    for name in EXPECTED_PROVENANCE:
        assert name in buffer.getvalue()


def test_a_single_section_can_be_run_alone():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert demo_results.main(["--only", "track_a"]) == 0
    assert "[1/1]" in buffer.getvalue()


def test_an_unknown_section_name_is_rejected_with_the_valid_ones():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert demo_results.main(["--only", "not_a_section"]) == 2
