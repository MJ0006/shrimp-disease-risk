"""Do the reported results still match the artefacts that produced them?

Every figure that will appear on a Review-2 slide is read out of `reports/tables/`. Those
tables are derived from the per-run JSON files in `experiments/runs/`. Nothing enforces that
the two stay in step -- a table could be edited, or regenerated from a partial set of runs,
and the slide would still render.

So the agreement is tested. Master Plan section 8 also states the project's evidence rule:

    "Every run records seed, library versions, GPU and config hash. A result without a
     recorded environment is treated as not obtained."

`test_every_run_records_its_environment` is that rule as an executable check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "reports" / "tables"
EXP02_RUNS = PROJECT_ROOT / "experiments" / "runs" / "exp02"

CONDITIONS = {"A_image_level_noaug", "B_image_level", "C_specimen_level"}
SEEDS = {2025, 7, 42}


def _table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet; run the experiment scripts")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def runs() -> pd.DataFrame:
    return _table("exp02_runs.csv")


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return _table("exp02_summary.csv")


# --------------------------------------------------------------------------------------
# Naming discipline -- D-010
# --------------------------------------------------------------------------------------

def test_condition_a_is_not_called_a_reproduction(runs, summary):
    """`D-010`: condition A differs from the inherited notebook in six factors and scores
    0.74 against its 0.638. It must not be labelled a reproduction anywhere in reporting."""
    for df in (runs, summary):
        assert "A_notebook_repro" not in set(df["condition"])
    assert set(runs["condition"]) == CONDITIONS
    assert set(summary["condition"]) == CONDITIONS


def test_every_row_points_at_an_artefact_that_exists(runs):
    """Traceability: a number with no artefact behind it cannot be defended."""
    assert "run_artifact" in runs.columns
    for rel in runs["run_artifact"]:
        assert (PROJECT_ROOT / rel).exists(), f"missing artefact {rel}"


# --------------------------------------------------------------------------------------
# The tables agree with the runs
# --------------------------------------------------------------------------------------

def test_all_nine_runs_are_present(runs):
    assert len(runs) == 9
    for cond in CONDITIONS:
        assert set(runs.loc[runs.condition == cond, "seed"]) == SEEDS


def test_summary_means_are_recomputable_from_the_run_table(runs, summary):
    recomputed = runs.groupby("condition")["test_f1_macro"].agg(["mean", "std"]).round(4)
    for _, row in summary.iterrows():
        got = recomputed.loc[row["condition"]]
        assert abs(got["mean"] - row["test_f1_macro_mean"]) < 5e-4
        assert abs(got["std"] - row["test_f1_macro_sd"]) < 5e-4


def test_run_table_matches_the_per_run_json(runs):
    """Spot-check the derived table against the primary artefact for every single run."""
    for _, row in runs.iterrows():
        payload = json.loads((PROJECT_ROOT / row["run_artifact"]).read_text(encoding="utf-8"))
        assert payload["config"]["seed"] == row["seed"]
        assert payload["config"]["protocol"] == row["protocol"]
        assert bool(payload["config"]["augment"]) == bool(row["augment"])
        assert payload["config"]["model_name"] == row["backbone"]
        assert abs(payload["leakage"]["pct_test_leaked"] - row["pct_test_leaked"]) < 0.01
        assert abs(payload["per_class"]["macro avg"]["f1-score"] - row["test_f1_macro"]) < 5e-4


def test_every_run_records_its_environment():
    """A result without a recorded environment is treated as not obtained.

    Widened on 2026-08-27: this globbed only `exp02` while claiming to cover every run, so the 25
    Exp 04 artefacts were never checked. They happened to comply -- but a rule enforced on one
    experiment is not enforced.
    """
    artefacts = sorted((PROJECT_ROOT / "experiments" / "runs").glob("*/*.json"))
    if not artefacts:
        pytest.skip("no run artefacts yet")
    for path in artefacts:
        env = json.loads(path.read_text(encoding="utf-8"))["environment"]
        for key in ("python", "platform", "torch"):
            assert env.get(key), f"{path.name} does not record {key}"


def test_the_experiments_that_train_nothing_still_record_their_environment():
    """Exp 01 and Exp 03 write no run JSON -- they train nothing -- so for a long time their
    environment was printed to the console and lost. Eight of the project's result tables come
    from Exp 03, and none of them could say what produced them."""
    for name in ("exp01_environment.csv", "exp03_environment.csv"):
        path = TABLES / name
        if not path.exists():
            pytest.skip(f"{name} not written yet; re-run the experiment")
        env = pd.read_csv(path).iloc[0]
        for key in ("python", "platform", "torch"):
            assert str(env.get(key, "")).strip(), f"{name} does not record {key}"


# --------------------------------------------------------------------------------------
# The findings themselves
# --------------------------------------------------------------------------------------

def test_the_controlled_comparison_differs_in_exactly_one_factor(runs):
    """B vs C is the whole argument. If they ever differ in anything but split protocol, the
    15-point gap stops being attributable to leakage and the Stage 4 claim is void."""
    b = runs[runs.condition == "B_image_level"]
    c = runs[runs.condition == "C_specimen_level"]

    assert set(b["augment"]) == set(c["augment"]) == {True}
    assert set(b["backbone"]) == set(c["backbone"]) == {"rexnet_150"}
    assert set(b["seed"]) == set(c["seed"])
    assert set(b["protocol"]) == {"image_level"}
    assert set(c["protocol"]) == {"specimen_level"}


def test_specimen_level_runs_report_zero_leakage(runs):
    c = runs[runs.condition == "C_specimen_level"]
    assert (c["pct_test_leaked"] == 0.0).all()


def test_image_level_runs_report_heavy_leakage(runs):
    leaky = runs[runs.protocol == "image_level"]
    assert (leaky["pct_test_leaked"] > 50.0).all()


def test_leakage_gap_exceeds_seed_noise(summary):
    """The pre-registered criterion: a gap smaller than the seed-to-seed SD is inconclusive.

    The gap must clear the *noisier* arm's SD, not the quieter one. Comparing 0.1494 against the
    specimen-level SD of 0.0004 while the image-level SD is 0.0636 would make almost any gap look
    decisive, because the specimen-level protocol is stable by construction -- its 63 test
    specimens are drawn the same way every seed. The uncertainty on a *difference* is carried by
    the side that moves, and here that side moves 143x more.

    This test encodes the criterion rather than the number, so it stays meaningful if the
    experiment is re-run.
    """
    b = summary.loc[summary.condition == "B_image_level"].iloc[0]
    c = summary.loc[summary.condition == "C_specimen_level"].iloc[0]
    gap = b["test_f1_macro_mean"] - c["test_f1_macro_mean"]

    assert gap > 0, "image-level scored no higher than specimen-level"
    assert gap > max(b["test_f1_macro_sd"], c["test_f1_macro_sd"]), (
        "gap is within the noisier arm's seed spread: report as inconclusive"
    )
    assert abs(gap - 0.1494) < 5e-4, "the documented headline figure has drifted"


def test_no_document_claims_the_gap_is_orders_of_magnitude_above_noise(summary):
    """Guards the correction made on 2026-08-28 (`D-027`) against silently reverting.

    The gap was for a time described as exceeding seed noise "by more than two orders of
    magnitude". That compares it against the specimen-level SD alone. Tested properly -- paired
    across the three shared seeds -- the difference gives p = 0.056 with a 95% interval that
    crosses zero, so the *magnitude* of the inflation is not tightly bounded at three seeds even
    though its *direction* is consistent and the 87.98% exposure behind it is structural fact.

    A text test rather than a numeric one, because the defect was a sentence, not a value. Mirrors
    the forbidden-claim pattern already used in `test_notebook.py`.

    Checked per line, not per file, and a line is exempt if it also carries a retraction marker.
    A document is allowed to *quote* the wording it withdrew -- that is how the project records a
    correction, and the same shape appears where `1,222` survives only inside the sentence
    explaining why we report 1,221. What is forbidden is asserting it.
    """
    banned = ("two orders of magnitude", "orders of magnitude larger than the seed")
    retraction_markers = ("corrected", "previously", "d-027", "withdrawn", "was a mistake")
    surfaces = [
        "docs/stages/stage_04_leakage_results.md",
        "docs/MASTER_DOCUMENT.md",
        "docs/presentation/ppt_content.md",
        "docs/presentation/speaker_notes.md",
        "docs/faculty_challenge/index.md",
        "docs/frameworks/README.md",
        "scripts/demo_results.py",
        "scripts/build_notebook.py",
    ]
    # Checked over a whitespace-normalised window rather than per raw line.
    #
    # The per-line version of this test passed while `docs/MASTER_DOCUMENT.md` asserted the
    # withdrawn claim in full, because markdown had wrapped it: "...by more than two orders of"
    # ended one line and "magnitude." began the next, so the banned phrase existed on neither.
    # A guard that a line break defeats is not a guard.
    offenders = []
    for rel in surfaces:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno in range(len(lines)):
            # A three-line window, collapsed to single spaces, so a phrase split across a wrap is
            # still seen. Retraction markers are looked for over the same window.
            window = " ".join(lines[lineno:lineno + 3])
            low = " ".join(window.lower().replace(">", " ").split())
            if any(phrase in low for phrase in banned) and not any(
                marker in low for marker in retraction_markers
            ):
                offenders.append(f"{rel}:{lineno + 1}")

    assert not offenders, (
        "the gap's significance is asserted again (not merely quoted as retracted) at: "
        + "; ".join(offenders)
    )


def test_exp01_leakage_summary_matches_the_documented_figures():
    """Stage 4 section 3 quotes 87.98% +/- 1.32 and exactly 0.00%."""
    df = _table("exp01_leakage_summary.csv").set_index("protocol")
    assert abs(df.loc["image_level", "pct_test_leaked_mean"] - 87.98) < 0.01
    assert abs(df.loc["image_level", "pct_test_leaked_sd"] - 1.32) < 0.01
    assert df.loc["specimen_level", "pct_test_leaked_mean"] == 0.0
    assert df.loc["specimen_level", "shared_specimens"] == 0.0


def test_black_gill_is_the_class_leakage_flatters_most(runs):
    """Stage 4 section 4.4: BG drops 0.83 -> 0.56, the largest fall of the four classes, and
    the one that matters most in practice. Guards the per-class story on the slides."""
    b = runs[runs.condition == "B_image_level"]
    c = runs[runs.condition == "C_specimen_level"]

    drops = {cls: b[f"f1_{cls}"].mean() - c[f"f1_{cls}"].mean()
             for cls in ("BG", "BG_WSSV", "Healthy", "WSSV")}
    assert max(drops, key=drops.get) == "BG"
    assert drops["BG"] > 0.2
    assert c["f1_BG"].mean() < 0.60


# ---------------------------------------------------------------------------
# Experiment 05 -- calibration (the assumption Stage 8 rests on)
# ---------------------------------------------------------------------------

# Experiments 02 and 04 stored 34 prediction artefacts between them, and for a long time this
# file asserted that exact number. Stage 8's Experiment 06 adds ten more, so a frozen count would
# now fail for the right reason at the wrong time. The property that actually matters is
# *completeness* -- every stored run is measured -- and it is asserted directly below. The floor
# keeps the other half of the guarantee: a corpus that silently shrank would still fail, which a
# bare `len(df) == len(npz)` would not catch.
MINIMUM_PREDICTION_ARTEFACTS = 34


def test_calibration_was_measured_for_every_stored_run():
    """Stage 8's estimators assume calibrated posteriors, so the claim needs a number.

    Every run that saved posteriors must appear, because a partial measurement would let a
    reader assume the unmeasured runs were fine.
    """
    df = _table("exp05_calibration.csv")
    npz = sorted((PROJECT_ROOT / "experiments" / "runs").glob("*/*_predictions.npz"))
    assert len(npz) >= MINIMUM_PREDICTION_ARTEFACTS, (
        f"only {len(npz)} prediction artefacts on disk; {MINIMUM_PREDICTION_ARTEFACTS} were "
        "obtained in Stages 4-5 and results cannot be reproduced without them"
    )
    assert len(df) == len(npz), (
        f"{len(df)} calibration rows for {len(npz)} artefacts -- re-run "
        "scripts/exp05_calibration.py so the measurement covers every stored run"
    )
    assert df["ece"].between(0.0, 1.0).all()
    assert (df["n_test"] > 0).all()


def test_the_models_are_measurably_overconfident():
    """Pins the finding behind `D-028` so it cannot quietly stop being true.

    If a calibration step is ever added and this fails, that is the good outcome -- but the
    documents that currently say "calibration is outstanding" must be updated in the same change.
    """
    df = _table("exp05_calibration_summary.csv").set_index("group")
    specimen = df.loc["C_specimen_level"]
    assert specimen["ece_mean"] > 0.15, "the documented ECE ~ 0.20 has drifted"
    assert specimen["overconfidence"] > 0.10, "overconfidence is the direction that matters"


def test_specimen_level_splitting_did_not_deliver_calibration():
    """The correction `D-028` records: removing leakage is not sufficient for calibration.

    Several documents argued that leakage causes overconfidence and that the specimen-level
    protocol therefore prepares the posteriors for Stage 8. Measured, the specimen-level model
    is the more miscalibrated of the two. This test guards the corrected wording.
    """
    df = _table("exp05_calibration_summary.csv").set_index("group")
    assert df.loc["C_specimen_level", "ece_mean"] > df.loc["B_image_level", "ece_mean"], (
        "if this reverses, the causal story in D-028 must be revisited, not the test"
    )


def test_reliability_bins_account_for_every_prediction():
    """A reliability diagram that silently dropped predictions would flatter the model."""
    bins = _table("exp05_reliability.csv")
    runs = _table("exp05_calibration.csv")
    for group, rows in bins.groupby("group"):
        expected = runs.loc[runs["group"] == group, "n_test"].sum()
        assert rows["n"].sum() == expected, f"{group}: bins hold {rows['n'].sum()} of {expected}"


# ---------------------------------------------------------------------------------------
# Flags that appear to work and do nothing
# ---------------------------------------------------------------------------------------

def test_no_script_imports_class_weights_without_calling_it():
    """`exp04_backbone_comparison.py` imported `class_weights` and never called it.

    The effect was that `class_weighted_loss=True` trained an unweighted model and reported it as
    a weighted one. Nothing failed; the flag simply did nothing, and it sat there for the whole of
    Stage 5 while the class-weighted ablation was planned against it (`D-030`, open item 11).

    This is the general shape of the bug, so the test is general: a script that imports the helper
    but never calls it has a switch that lies. Written as a source check rather than a behavioural
    one on purpose -- the behaviour is only observable by training a model, and by then the wrong
    number is already in a table.
    """
    scripts = sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    assert scripts, "no scripts found -- the check would pass vacuously"
    for path in scripts:
        src = path.read_text(encoding="utf-8")
        if "import" not in src or "class_weights" not in src:
            continue
        imports_it = any(
            line.strip().startswith(("from ", "import ")) and "class_weights" in line
            for line in src.splitlines()
        )
        if imports_it:
            assert "class_weights(" in src, (
                f"{path.name} imports class_weights but never calls it -- "
                "class_weighted_loss would silently do nothing"
            )


def test_the_reported_backbone_comparison_is_the_unweighted_one():
    """Pins what the 25 stored Experiment 04 runs actually were.

    Now that the flag is wired, a future run with `--class-weighted` would produce genuinely
    different models. Those belong in their own experiment with their own directory; if weighted
    runs ever appear under `exp04`, the reported comparison has been overwritten by a different
    configuration wearing the same tags, and every number quoted from it is wrong.
    """
    artefacts = sorted((PROJECT_ROOT / "experiments" / "runs" / "exp04").glob("*.json"))
    if not artefacts:
        pytest.skip("no Experiment 04 artefacts present")
    for path in artefacts:
        cfg = json.loads(path.read_text(encoding="utf-8"))["config"]
        assert cfg.get("class_weighted_loss") is False, (
            f"{path.name} was trained with class weighting, but every document reports "
            "Experiment 04 as unweighted (D-030)"
        )
