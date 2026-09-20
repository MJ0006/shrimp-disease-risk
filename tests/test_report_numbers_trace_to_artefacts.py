"""Every headline number in the bound report must equal the CSV it claims to come from.

Why this file exists
--------------------
The project's binding rule is that every number in a deliverable traces to a file in `reports/`.
Until 2026-09-20 nothing enforced it. `tests/test_report_format.py` checks margins, section
count, figure and table counts, caption monotonicity and word volume -- form, never values --
and `scripts/build_report_docx.py` contained no `read_csv`, no `open()` and no `read_text()` at
all, so every figure in the report was a typed literal.

The result was a report whose Chapter 5 tables disagreed with the script-generated figures on the
facing pages, and in several places with each other. The backbone table ranked ReXNet-150 second
and ResNet-50 third; `exp04_summary.csv` and `fig06_backbone_comparison.png` both rank ResNet-50
second and ReXNet-150 fourth. A guide comparing the table to the figure would have found it in
seconds.

What this guards
----------------
Not prose -- prose is checked by reading it. This file pins the values a reader is most likely to
check first, and it reads them out of the generated DOCX rather than out of the builder, so it
fails if the document on disk is stale as well as if the builder is wrong.

The report is generated, so a failure here means one of two things: the builder has drifted from
the artefacts, or the DOCX was not rebuilt after an experiment was re-run. Both are defects.
Rebuild with `.venv/Scripts/python.exe scripts/build_report_docx.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from shrimp_risk.common.paths import PROJECT_ROOT, TABLES

REPORT_PATH = PROJECT_ROOT / "docs" / "report" / "Report_19MIS1127.docx"


@pytest.fixture(scope="module")
def report_text() -> str:
    """The whole report as flat text, tables included."""
    if not REPORT_PATH.exists():
        pytest.skip(f"report not built at {REPORT_PATH}")
    pytest.importorskip("docx")
    import docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = docx.Document(str(REPORT_PATH))
    chunks: list[str] = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            chunks.append(Paragraph(child, doc).text)
        elif child.tag == qn("w:tbl"):
            for row in Table(child, doc).rows:
                chunks.append(" | ".join(c.text for c in row.cells))
    return "\n".join(chunks)


def _csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    return pd.read_csv(path)


def test_backbone_table_matches_exp04(report_text: str) -> None:
    """Each backbone's macro-F1 and footprint appear exactly as `exp04_summary.csv` records them."""
    exp04 = _csv("exp04_summary.csv")
    labels = {
        "convnext_tiny": "ConvNeXt-Tiny",
        "resnet50": "ResNet-50",
        "mobilenetv3_large_100": "MobileNetV3-Large",
        "rexnet_150": "ReXNet-150",
        "efficientnet_b0": "EfficientNet-B0",
    }
    missing = []
    for _, row in exp04.iterrows():
        cell = f"{row['f1_macro_mean']:.4f} ± {row['f1_macro_sd']:.4f}"
        if cell not in report_text:
            missing.append(f"{labels[row['backbone']]}: expected '{cell}'")
        size = f"{row['size_mb']:.2f} MB"
        if size not in report_text:
            missing.append(f"{labels[row['backbone']]}: expected size '{size}'")
    assert not missing, "backbone figures absent from the report:\n  " + "\n  ".join(missing)


def test_report_does_not_contradict_the_measured_ranking(report_text: str) -> None:
    """ResNet-50 outscores ReXNet-150 in the artefact; the report must not invert that."""
    exp04 = _csv("exp04_summary.csv").set_index("backbone")
    assert exp04.loc["resnet50", "f1_macro_mean"] > exp04.loc["rexnet_150", "f1_macro_mean"]

    # The inverted pair that actually shipped. Pinned as a literal so it cannot come back.
    for wrong in ("0.6974", "0.6841", "0.6754", "0.6621", "9.73 M", "37.52 MB"):
        assert wrong not in report_text, (
            f"'{wrong}' is a fabricated backbone figure that contradicts exp04_summary.csv"
        )


def test_leakage_headline_matches_exp02(report_text: str) -> None:
    """The 0.8215 / 0.6721 / 0.1494 triple and BOTH standard deviations."""
    summary = _csv("exp02_summary.csv").set_index("condition")
    leaky = summary.loc["B_image_level"]
    clean = summary.loc["C_specimen_level"]

    assert f"{leaky['test_f1_macro_mean']:.4f}" in report_text
    assert f"{clean['test_f1_macro_mean']:.4f}" in report_text
    gap = leaky["test_f1_macro_mean"] - clean["test_f1_macro_mean"]
    assert f"{gap:.4f}" in report_text

    # The specimen-level SD is 0.0004. Reporting it as 0.0135 or 0.0385 -- both of which the
    # report did -- re-opens exactly the overclaim D-027 exists to prevent, because the gap looks
    # enormous against the real SD and the paired test gives p = 0.056.
    assert f"{clean['test_f1_macro_sd']:.4f}" in report_text
    for invented in ("0.0094", "0.0135", "0.0142", "0.0385"):
        assert invented not in report_text, f"'{invented}' is not an Experiment 02 SD"


def test_edge_export_figures_match_exp12(report_text: str) -> None:
    """Size, latency and held-out accuracy for every exported runtime."""
    export = _csv("exp12_export.csv")
    for _, row in export.iterrows():
        assert f"{row['file_size_mb']:.2f} MB" in report_text, row["precision"]
        assert f"{row['holdout_macro_f1']:.4f}" in report_text, row["precision"]

    # Measurements on hardware this project never had, and a size/latency pairing taken from two
    # different models.
    for phantom in ("Cortex-A53", "42.1 ms", "4.24 MB", "14.8 ms", "2.35 ms", "3.81"):
        assert phantom not in report_text, f"'{phantom}' traces to no artefact"


def test_step3_models_are_the_ones_that_were_run(report_text: str) -> None:
    """Track A evaluated Cubist, Random Forest, SVM-RBF and Gradient Boosting."""
    step3 = _csv("exp03_step3_candidates.csv")
    models = set(step3[step3["risk_model"] == "income"]["model"])
    assert "Cubist" in models and "MLP" not in models and "Ridge" not in models

    assert "Cubist" in report_text, "the best step-3 ratio (1.014) belongs to Cubist"
    assert "Ridge Regression" not in report_text, "there is no Ridge model in Track A"


def test_calibration_temperatures_match_exp06(report_text: str) -> None:
    """Temperatures are 2.85 and 3.98, not the 1.34-1.45 the report claimed."""
    cal = _csv("exp06_summary.csv")
    for _, row in cal.iterrows():
        assert f"{row['temperature_mean']:.2f}" in report_text, row["backbone"]
        assert f"{row['ece_before_mean']:.4f}" in report_text, row["backbone"]

    # exp06 covers two backbones. Three of the five rows in the shipped table were runs that
    # never happened.
    assert len(cal) == 2


def test_class_weighting_direction_is_not_reversed(report_text: str) -> None:
    """Weighting RAISED macro-F1. The report twice said it degraded it."""
    comp = _csv("exp09_comparison.csv").set_index("metric")
    macro = comp.loc["f1_macro"]
    assert macro["weighted_mean"] > macro["unweighted_mean"]
    assert f"{macro['weighted_mean']:.4f}" in report_text
    assert not bool(macro["beyond_fold_sd"]), "delta should sit inside the fold SD"


def test_no_plagiarism_result_is_asserted(report_text: str) -> None:
    """The institution runs the check. The thesis must not report its own result."""
    lowered = report_text.lower()
    for claim in ("plagiarism analysis confirmed", "similarity index strictly below",
                  "turnitin similarity verified"):
        assert claim not in lowered, (
            "the report asserts a plagiarism result. That screening is carried out by the "
            "institution and its outcome cannot be stated here in advance."
        )


def test_novelty_is_stated_as_a_search_result(report_text: str) -> None:
    """D-015: 'we found no prior work that...', never 'the first'."""
    offenders = re.findall(
        r"the first (?:formal|rigorous|comprehensive|leak-free|framework)[^.]{0,80}",
        report_text, flags=re.IGNORECASE,
    )
    assert not offenders, (
        "absolute novelty claims found; phrase them as search results instead: " + str(offenders)
    )


def test_template_example_references_are_not_in_the_bibliography(report_text: str) -> None:
    """The VIT template's own ocean-wave formatting examples shipped inside the bibliography."""
    for example in ("Surface waves on water of variable depth", "Bispectra of Ocean Waves",
                    "Wave reflection at Beaches", "Water Waves: The Mathematical Theory",
                    "Empirical orthogonal representation of time series"):
        assert example not in report_text, (
            f"'{example}' is a formatting example from MIS Project Template.docx, not a source "
            f"this project used"
        )


def test_no_placeholder_author_tokens(report_text: str) -> None:
    """Citations must carry real author names."""
    for token in ("FeatherNetX Authors", "PMC12468286 Authors",
                  "Leakage-Aware Leukemia Benchmark Authors"):
        assert token not in report_text, f"'{token}' is a placeholder, not an author list"


def test_per_class_specimen_counts_match_the_manifest(report_text: str) -> None:
    """The dataset composition table must agree with `image_manifest.csv`, class by class.

    This one nearly shipped. The hard-coded table gave Healthy 144 / WSSV 117 / BG_WSSV 80 against
    a manifest that says 142 / 118 / 81. Three of the four class rows were wrong while the total
    -- the figure a reader checks first -- was right at 416, so nothing looked amiss. Both the
    report and the manuscript carried it, and the faculty challenge bank carried the correct
    counts, which is how the disagreement surfaced.
    """
    manifest_path = PROJECT_ROOT / "data" / "interim" / "image_manifest.csv"
    if not manifest_path.exists():
        pytest.skip("image manifest not present")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["subset"] == "raw"]
    per_class = manifest.groupby("class")["specimen_id"].nunique()

    assert per_class.sum() == 416
    missing = [
        f"{cls}: expected {count} specimens"
        for cls, count in per_class.items()
        if str(count) not in report_text
    ]
    assert not missing, "specimen counts absent from the report:\n  " + "\n  ".join(missing)

    # The superseded values, pinned so they cannot come back.
    assert "144" not in report_text or "142" in report_text, (
        "the report appears to carry the old Healthy specimen count of 144"
    )


# ---------------------------------------------------------------------------------------
# Appendix 3 guards added 2026-09-21.
#
# The appendix titled "System Source Code Listings and Execution Contracts" carried a listing
# that was written by hand rather than read from the repository. It imported three things that
# do not exist -- `shrimp_risk.vision.inference`, `shrimp_risk.integration.sld_em`, and a
# `project_farm_income_risk` that `just_pope.py` never defined -- and declared command-line
# flags (`--area_dec`, `--feed_cost`, `--fingerling_cost`) that `predict_farm.py` does not
# accept. It would not have run. Beside it, Table 23 reported Python 3.12.3, torch 2.6.0,
# onnxruntime 1.20.1 and Windows build 22631 under a column headed "Verified", against an
# actual 3.12.10, 2.11.0, 1.30.0 and 26200 -- while `demo_results.py` printed the true
# environment on screen during the demonstration.
#
# Both are now generated at build time. These tests fail if either drifts back to being typed.
# ---------------------------------------------------------------------------------------

PREDICT_FARM = PROJECT_ROOT / "scripts" / "predict_farm.py"


def test_appendix3_listings_are_excerpted_from_real_source(report_text: str) -> None:
    """Every line printed as source code must exist verbatim in the file it names."""
    if not PREDICT_FARM.exists():
        pytest.skip("predict_farm.py not present")
    source = PREDICT_FARM.read_text(encoding="utf-8")

    anchors = [
        'parser.add_argument("--images-dir", type=str, default=None',
        'parser.add_argument("--farm-area", type=float, default=1.0',
        "asked_for_real_images = bool(args.images) or bool(args.images_dir)",
    ]
    for anchor in anchors:
        assert anchor in source, f"anchor no longer in predict_farm.py: {anchor!r}"
        assert anchor in report_text, (
            f"Appendix 3 does not contain the real source line {anchor!r}. "
            "Rebuild the report; the listing is excerpted at build time."
        )

    # The fabricated listing's fingerprints, pinned so they cannot return.
    for invented in (
        "run_batch_tray_inference",
        "estimate_prevalence_sld_em",
        "project_farm_income_risk",
        "--area_dec",
        "SAFETY REFUSAL",
    ):
        assert invented not in report_text, (
            f"the report carries {invented!r}, which exists nowhere in this repository"
        )


def test_appendix3_environment_matches_the_run_fingerprint(report_text: str) -> None:
    """Table 23 must equal the environment the deployment checkpoint actually recorded."""
    import json

    run_json = (
        PROJECT_ROOT / "experiments" / "runs" / "edge_checkpoint"
        / "mobilenetv3_large_100__fold0.json"
    )
    if not run_json.exists():
        pytest.skip("edge checkpoint run fingerprint not present")
    env = json.loads(run_json.read_text(encoding="utf-8")).get("environment", {})
    if not env:
        pytest.skip("run fingerprint carries no environment block")

    for key in ("python", "torch", "gpu", "platform"):
        value = env.get(key)
        assert value, f"run fingerprint is missing {key}"
        assert value in report_text, (
            f"Table 23 does not report the recorded {key} {value!r}. "
            "Rebuild the report; the table is read from the run fingerprint."
        )

    # The invented values, pinned so they cannot return.
    for wrong in ("Python 3.12.3", "torch 2.6.0+cu128", "onnxruntime 1.20.1", "Build 22631"):
        assert wrong not in report_text, f"Table 23 has reverted to the typed value {wrong!r}"


def test_appendix3_does_not_claim_dynamic_quantisation(report_text: str) -> None:
    """D-045 replaced dynamic INT8 with static QDQ; the appendix said otherwise until 09-21."""
    assert "Dynamic INT8 quantized deployment benchmark" not in report_text, (
        "Appendix 3 still describes the deployed runtime as dynamic INT8. "
        "Dynamic was measured at macro-F1 0.2380 and rejected (D-045); static QDQ ships."
    )


def test_chapter_4_6_describes_flags_that_exist(report_text: str) -> None:
    """Section 4.6 documents predict_farm.py's interface. It must document the real one.

    Until 2026-09-21 it described a `--farm-covariates` argument and a `--dry-run` mode,
    neither of which the script accepts, and attributed to it management recommendations
    ("Initiate immediate emergency harvest") that appear nowhere in the codebase. A guide
    following the report would have got an argparse error on the first command.
    """
    if not PREDICT_FARM.exists():
        pytest.skip("predict_farm.py not present")
    source = PREDICT_FARM.read_text(encoding="utf-8")

    flags = re.findall(r'add_argument\("(--[a-z0-9-]+)"', source)
    assert "--images-dir" in flags and "--farm-area" in flags, (
        f"predict_farm.py's flags have changed: {flags}"
    )

    for invented in ("--farm-covariates", "--dry-run"):
        assert invented not in report_text, (
            f"the report documents {invented}, which predict_farm.py does not accept. "
            f"Real flags: {sorted(set(flags))}"
        )

    # Narrow deliberately: Chapter 6 discusses farmers initiating emergency harvests as a
    # source of survivorship bias, which is a legitimate economic point. What must not come
    # back is the claim that the CLI *emits* these as management instructions.
    for invented in ("Initiate immediate emergency harvest", "Maintain standard feeding regime"):
        assert invented not in report_text, (
            f"the report attributes the output {invented!r} to predict_farm.py; "
            "no such string exists in the repository"
        )


def test_report_does_not_claim_the_deployment_calibrates(report_text: str) -> None:
    """D-041: calibration degrades the bridge, so the deployed path uses RAW posteriors."""
    assert "calibrated confidence scores" not in report_text, (
        "the report describes the CLI as emitting calibrated confidence scores. "
        "predict_farm.py prints 'Calibration state: RAW posteriors (D-041)'."
    )


# ---------------------------------------------------------------------------------------
# The general guard, added 2026-09-21 after three separate fabrications were found by hand.
#
# The tests above pin specific invented strings, which only works for defects someone has
# already noticed. Each of the three was found by reading, not by testing: a listing importing
# `shrimp_risk.integration.sld_em`, a section documenting `--farm-covariates`, and prose citing
# `src/shrimp_risk/integration/bridge.py` and `scripts/export_edge_models.py`. None of those
# files or flags has ever existed in this repository.
#
# This test generalises the check instead of enumerating it: every script filename, dotted
# module path and command-line flag the report mentions must resolve against the repository.
# It is the guard that would have caught all three on the day they were written.
# ---------------------------------------------------------------------------------------

def test_every_code_claim_in_the_report_resolves(report_text: str) -> None:
    """No .py file, module path or CLI flag may appear in the report unless it exists."""
    import ast

    py_files = {p.name for p in PROJECT_ROOT.rglob("*.py") if ".venv" not in str(p)}
    assert "predict_farm.py" in py_files, "repository layout has changed; fix this test"

    src = PROJECT_ROOT / "src"
    module_paths = {
        ".".join(p.relative_to(src).with_suffix("").parts) for p in src.rglob("*.py")
    }

    real_flags: set[str] = set()
    for script in (PROJECT_ROOT / "scripts").rglob("*.py"):
        real_flags |= set(
            re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', script.read_text(encoding="utf-8"))
        )

    unresolved: list[str] = []

    for name in sorted(set(re.findall(r"\b([a-z_][a-z0-9_]*\.py)\b", report_text))):
        if name not in py_files:
            unresolved.append(f"script {name!r} is named in the report but not in the repository")

    for mod in sorted(set(re.findall(r"\b(shrimp_risk(?:\.[a-z_][a-z0-9_]*)+)", report_text))):
        if mod not in module_paths and mod.rsplit(".", 1)[0] not in module_paths:
            unresolved.append(f"module path {mod!r} is not importable")

    # pip's own flags appear in setup prose and are not ours to satisfy.
    external = {"--index-url", "--upgrade", "--no-cache-dir", "--user", "--editable"}
    for flag in sorted(set(re.findall(r"(--[a-z][a-z0-9-]{2,})", report_text))):
        if flag not in real_flags and flag not in external:
            unresolved.append(f"flag {flag!r} is documented but no script accepts it")

    assert not unresolved, (
        "the report makes claims about code that does not exist:\n  "
        + "\n  ".join(unresolved)
    )
