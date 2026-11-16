"""Tests for the final-review showcase page (`scripts/build_showcase.py`).

The page is shown to a panel, so the rule is the deck's rule: every number on it traces to a
file in `reports/`. These tests read the tables themselves -- not through the page builder -- and
check that the page carries exactly those values, so a wrong row selection or a typed literal in
the builder fails here. The live test runs the real pond and checks the page against the answer
key and against `predict_farm.py`'s own output.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

import build_showcase as S
from shrimp_risk.common.paths import PROJECT_ROOT, TABLES


@pytest.fixture(scope="module")
def recorded_page() -> str:
    return S.page(S.recorded(), live=None)


def _t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def test_headline_numbers_on_the_page_equal_their_tables(recorded_page):
    leak = _t("exp01_leakage_summary").set_index("protocol")
    e2 = _t("exp02_summary").set_index("condition")
    e4 = _t("exp04_summary").set_index("backbone")
    e12 = _t("exp12_export")
    onnx = e12[e12["format"] == "ONNX Runtime"].set_index("precision")
    e8 = _t("exp08_baseline").set_index("specification")
    e3 = _t("exp03_diagnostics").set_index("model")

    expected = [
        f"{leak.loc['image_level', 'pct_test_leaked_mean']:.2f}%",
        f"{e2.loc['B_image_level', 'test_f1_macro_mean']:.4f}",
        f"{e2.loc['C_specimen_level', 'test_f1_macro_mean']:.4f}",
        f"{e4.loc['convnext_tiny', 'f1_macro_mean']:.4f}",
        f"{e4.loc['mobilenetv3_large_100', 'f1_macro_mean']:.4f}",
        f"{onnx.loc['FP32', 'holdout_macro_f1']:.4f}",
        f"{onnx.loc['INT8 (Static QDQ)', 'holdout_macro_f1']:.4f}",
        f"{onnx.loc['INT8 (Static QDQ)', 'file_size_mb']:.2f} MB",
        f"{e8.loc['S2_bridge_facing', 'estimate']:.4f}",
        f"{e3.loc['income', 'r_squared']:.4f}",
    ]
    missing = [v for v in expected if v not in recorded_page]
    assert not missing, f"page does not carry these table values: {missing}"


def test_bridge_numbers_match_the_deck_configuration(recorded_page):
    """Slide 19 quotes ConvNeXt, raw posteriors, pools of 40. The page must use the same rows."""
    e7 = _t("exp07_estimators")
    sel = e7[(e7.backbone == "convnext_tiny") & (e7.calibration == "raw") & (e7.pool_size == 40)]
    by = sel.set_index("estimator")["mean_abs_bias"]
    assert f"{by['SLD-EM']:.4f}" in recorded_page
    assert f"{by['CC']:.4f}" in recorded_page


def test_validation_evidence_equals_its_tables(recorded_page):
    leak = _t("exp01_leakage_summary").set_index("protocol")
    before = _t("exp04_folds_lr3e-4_all").query("backbone == 'convnext_tiny'")["test_f1_macro"].mean()
    nb = _t("exp02b_summary").set_index("image_source")
    gcv = _t("exp10_grouped_cv").set_index(["risk_model", "fold_scheme"])
    e9 = _t("exp09_comparison").set_index("metric")

    assert "5 · How we validate" in recorded_page
    for v in [
        f"<b>{leak.loc['image_level', 'shared_specimens']:.1f}</b>",
        f"<b>{before:.4f}</b>",
        f"<b>{nb.loc['cache_256', 'f1_macro_mean']:.3f}</b>",
        f"<b>{gcv.loc[('income', 'cluster_grouped_folds'), 'holdout_rmse_vs_mean_baseline']:.3f}</b>",
        f"+{e9.loc['f1_BG', 'delta']:.4f}",
    ]:
        assert v in recorded_page, v
    # an ablation inside fold noise must read as not claimed
    assert not e9.loc["f1_BG", "beyond_fold_sd"] and "<b>not claimed</b>" in recorded_page


def test_every_experiment_is_listed_with_its_source(recorded_page):
    """Section 4 must account for every experiment that wrote a table, so none is quietly left out."""
    ids = sorted({p.name.split("_")[0] for p in TABLES.glob("exp*.csv")})
    for exp in ids:  # exp01 ... exp12, exp02b, exp03a
        label = "Exp " + exp[3:]
        if exp == "exp03a":  # descriptives of the survey, not an experiment with a finding
            continue
        assert label in recorded_page, f"{label} missing from the experiments table"
    e5 = _t("exp05_calibration_summary").query("experiment == 'exp04'")["ece_mean"]
    assert f"{e5.min():.4f}–{e5.max():.4f}" in recorded_page


def test_code_map_points_at_code_and_tests_that_exist():
    """A panel member will click these. A dead pointer in front of them is worse than none."""
    for _q, path, func, _what, test in S.CODE_MAP:
        assert (PROJECT_ROOT / path).is_file(), path
        S.locate(path, func)  # raises if the function has been renamed or moved out
        if test:
            test_file, _, selector = test.partition(" -k ")
            src = (PROJECT_ROOT / test_file).read_text(encoding="utf-8")
            if selector:
                assert any(line.startswith("def test") and selector in line for line in src.splitlines()), test


def test_every_block_is_labelled_live_or_recorded(recorded_page):
    """D-022: a replayed result must never look like one computed now."""
    assert recorded_page.count('class="tag recorded"') >= 8
    assert 'class="tag live"' not in recorded_page  # nothing ran, so nothing may claim LIVE


def test_predict_farm_json_carries_per_specimen_posteriors():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "predict_farm.py"), "--synthetic", "6", "--json"],
        capture_output=True, text=True, encoding="utf-8", cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout[proc.stdout.find('{\n  "n_images":'):])
    assert len(data["specimens"]) == data["n_specimens"] == 3
    assert data["classes"] == ["BG", "BG_WSSV", "Healthy", "WSSV"]
    for s in data["specimens"]:
        assert abs(sum(s["probs"]) - 1.0) < 1e-6


@pytest.mark.dataset
def test_live_pond_page_matches_the_tool_and_the_answer_key(tmp_path):
    if not S.POND.is_dir():
        pytest.skip("demo pond not built (scripts/make_demo_pond.py)")
    live = S.run_live(int8=False)
    page = S.page(S.recorded(), live)

    key = pd.read_csv(S.ANSWER_KEY)
    assert live["n_specimens"] == len(key)  # every shrimp in the key was diagnosed, once
    assert f"{live['prevalence']['em'] * 100:.2f}%" in page
    assert f"{key['white_spot'].mean() * 100:.1f}%" in page
    assert 'class="tag live"' in page
    assert "assumed" in page  # the Taka baseline is labelled as an assumption
