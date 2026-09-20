"""Is the shipped notebook current, executed, and honest about what it ran?

`shrimp_disease_risk_project.ipynb` is the artefact most likely to be read by someone who never
opens a terminal, which makes it the one most likely to be believed without checking. Three
things can go wrong with it, and each is tested here:

1. **It goes stale.** A table is regenerated, the notebook is not rebuilt, and its printed
   numbers quietly disagree with `reports/`. The headline test rebuilds the expected figures
   from the CSVs and looks for them in the notebook's stored outputs.
2. **It ships broken.** A traceback in a cell, or a notebook saved without ever being run.
3. **It overclaims.** The notebook trains a model for real, once, on one fold at a reduced
   budget. If that run were ever presented as a reported result, the notebook would be making a
   claim the project's own experiments contradict (`D-022`).

A notebook built with `--no-execute` has no outputs at all. That is a legitimate state, so the
output-dependent tests **skip** rather than fail -- "not checkable here" is different from
"broken". The build script executes by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "shrimp_disease_risk_project.ipynb"
TABLES = PROJECT_ROOT / "reports" / "tables"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip("notebook not built; run scripts/build_notebook.py")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def outputs(code_cells: list[dict]) -> str:
    """Every stream and text/plain output in the notebook, as one string."""
    if not any(cell.get("execution_count") for cell in code_cells):
        pytest.skip("notebook has not been executed; run scripts/build_notebook.py")
    chunks = []
    for cell in code_cells:
        for output in cell.get("outputs", []):
            chunks.append("".join(output.get("text", "")))
            chunks.append("".join(output.get("data", {}).get("text/plain", "")))
    return "\n".join(chunks)


def _table(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet; run the experiment scripts")
    return pd.read_csv(path)


# --------------------------------------------------------------------------------------
# It ran, and it ran cleanly
# --------------------------------------------------------------------------------------

def test_notebook_is_valid_and_has_both_kinds_of_cell(notebook: dict):
    assert notebook["nbformat"] == 4
    kinds = {cell["cell_type"] for cell in notebook["cells"]}
    assert kinds == {"code", "markdown"}, "a notebook of code with no prose is not a document"


def test_no_cell_produced_an_error(code_cells: list[dict]):
    """A traceback sitting in a notebook a guide opens is worse than no notebook."""
    failed = [
        index for index, cell in enumerate(code_cells)
        if any(output.get("output_type") == "error" for output in cell.get("outputs", []))
    ]
    assert not failed, f"code cells with error output: {failed}"


def test_every_code_cell_was_executed(code_cells: list[dict]):
    if not any(cell.get("execution_count") for cell in code_cells):
        pytest.skip("notebook has not been executed; run scripts/build_notebook.py")
    unexecuted = [i for i, cell in enumerate(code_cells) if cell.get("execution_count") is None]
    assert not unexecuted, f"cells never run: {unexecuted}. Rebuild rather than hand-editing"


def test_cells_were_executed_in_order(code_cells: list[dict]):
    """Out-of-order execution counts mean cells were run piecemeal, so an output may depend on
    state that no longer exists. Only a clean top-to-bottom run is reproducible."""
    counts = [cell.get("execution_count") for cell in code_cells]
    if not any(counts):
        pytest.skip("notebook has not been executed; run scripts/build_notebook.py")
    assert counts == sorted(counts), "notebook was not executed top to bottom"


# --------------------------------------------------------------------------------------
# It is current
# --------------------------------------------------------------------------------------

def test_printed_headline_numbers_match_the_tables(outputs: str):
    """Rebuild each headline from `reports/tables/` and look for it in the notebook's output.
    This is what fails when an experiment is re-run and the notebook is not rebuilt."""
    exp01 = _table("exp01_leakage_summary.csv")
    exp02 = _table("exp02_summary.csv")
    exp04 = _table("exp04_summary.csv")

    exposure = exp01.loc[exp01["protocol"] == "image_level", "pct_test_leaked_mean"].iloc[0]
    b = exp02.loc[exp02["condition"] == "B_image_level", "test_f1_macro_mean"].iloc[0]
    c = exp02.loc[exp02["condition"] == "C_specimen_level", "test_f1_macro_mean"].iloc[0]
    best = exp04.sort_values("f1_macro_mean", ascending=False).iloc[0]

    for expected in (f"{exposure:.2f}", f"{b:.4f}", f"{c:.4f}", f"{b - c:.4f}",
                     f"{best['f1_macro_mean']:.4f}"):
        assert expected in outputs, (
            f"{expected} is in reports/tables/ but not in the notebook. "
            "Rebuild: scripts/build_notebook.py"
        )


def test_track_a_precondition_verdicts_match_the_diagnostics(outputs: str):
    """The Just-Pope mean equations are recomputed live in the notebook, so their diagnostics
    must still equal the recorded ones -- if they do not, the survey data has changed."""
    diagnostics = _table("exp03_diagnostics.csv")
    for _, row in diagnostics.iterrows():
        assert f"{row['r_squared']:.4f}" in outputs, f"{row['model']} R^2 missing or stale"


def test_the_test_suite_output_embedded_in_the_notebook_passed(outputs: str):
    """The notebook runs pytest and stores the result. A failing suite recorded in a document
    a guide reads would be a self-inflicted wound."""
    assert "exit code: 0" in outputs
    assert "passed" in outputs
    assert "failed" not in outputs.split("exit code:")[0][-400:]


# --------------------------------------------------------------------------------------
# It does not overclaim -- D-022
# --------------------------------------------------------------------------------------

def test_the_live_training_run_is_labelled_a_demonstration(code_cells: list[dict], outputs: str):
    """One cell genuinely trains. It must say, in the source and in the printed output, that it
    is a demonstration and not a reported result."""
    training = [cell for cell in code_cells if "train_and_evaluate" in "".join(cell["source"])]
    assert len(training) == 1, "exactly one cell should train a model"
    source = "".join(training[0]["source"])
    assert "DEMONSTRATION RUN" in source
    assert "not a reported result" in source
    assert "DEMONSTRATION RUN" in outputs


def test_the_notebook_writes_no_artefacts(code_cells: list[dict]):
    """Reading a notebook must never overwrite a result. The demonstration run in particular is
    not written to `experiments/runs/`, where it could later be mistaken for an experiment."""
    forbidden = ("to_csv(", "np.savez", "json.dump(", "torch.save", ".write_text(")
    offenders = [
        (index, token)
        for index, cell in enumerate(code_cells)
        for token in forbidden
        if token in "".join(cell["source"])
    ]
    assert not offenders, f"notebook cells write to disk: {offenders}"


def test_recorded_sections_name_the_file_they_replay(notebook: dict):
    """Every block that replays a result must name its artefact, the same rule the slides and
    the demonstration runner follow."""
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "RECORDED" in markdown and "LIVE" in markdown, "provenance labelling is missing"

    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for artefact in ("exp01_leakage_summary.csv", "exp02_summary.csv", "exp04_summary.csv",
                     "exp03_step3_candidates.csv", "exp04_folds_lr3e-4_all.csv"):
        assert artefact in code, f"{artefact} is replayed but never named"


def test_the_forbidden_claims_are_only_ever_denied(notebook: dict):
    """The claims the project has decided may never be asserted (`D-008`, the ShrimpXNet
    comparison, `D-010`).

    They are *allowed* to appear — the conclusion lists them explicitly as things this work does
    not claim, which is more useful to a reader than silence. So the test does not ban the
    words; it requires each occurrence to sit inside a denial. A phrase that ever turns up
    without one has changed from a disclaimer into a claim.
    """
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"]).lower()
    denials = ("not", "never", "cannot", "must not")

    for phrase in ("sample-level fusion", "better than shrimpxnet", "inflated by 15 points",
                   "reproduced the notebook"):
        start = 0
        while (found := text.find(phrase, start)) != -1:
            window = text[max(0, found - 160):found]
            assert any(word in window for word in denials), (
                f"'{phrase}' appears without a denial in front of it — "
                "that turns a disclaimer into a claim"
            )
            start = found + len(phrase)
