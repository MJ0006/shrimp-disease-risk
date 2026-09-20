"""Tests for farm-level prediction CLI (`scripts/predict_farm.py`).

Validates:
1. CLI execution across execution modes (synthetic generation, JSON output, INT8, FP32).
2. Image preprocessing to normalized tensor format.
3. Decision-level multimodal bridge math (revenue loss, measurement error attenuation).
4. Specimen-level probability aggregation (eliminating split leakage).
5. Defensibility checks: presence of D-008, D-007, and D-022 citations in output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from shrimp_risk.common.paths import PROJECT_ROOT

PREDICT_SCRIPT = PROJECT_ROOT / "scripts" / "predict_farm.py"
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def test_preprocess_image():
    """Verify image preprocessing resizes to 224x224 and normalizes correctly."""
    from scripts.predict_farm import preprocess_image

    # Create dummy 500x300 RGB image
    img = Image.new("RGB", (500, 300), color=(128, 64, 32))
    arr = preprocess_image(img, img_size=224)

    assert arr.shape == (3, 224, 224)
    assert arr.dtype == np.float32
    assert not np.isnan(arr).any()


def test_cli_synthetic_json_execution():
    """Run CLI in synthetic mode with --json and verify structured output."""
    cmd = [
        str(PYTHON_EXE),
        str(PREDICT_SCRIPT),
        "--synthetic", "6",
        "--farm-area", "2.0",
        "--int8",
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0, f"CLI exited with code {proc.returncode}:\n{proc.stderr}"

    output = proc.stdout
    assert "FARM-LEVEL SHRIMP DISEASE DIAGNOSIS & RISK PREDICTION" in output
    assert "SPECIMEN CLASSIFICATION BREAKDOWN" in output
    assert "PREVALENCE QUANTIFICATION RESULTS" in output
    assert "FARM-LEVEL ECONOMIC RISK ASSESSMENT" in output
    assert "D-008" in output
    assert "D-007" in output
    assert "D-022" in output

    # Extract JSON payload from stdout
    json_start = output.find("{\n  \"n_images\":")
    assert json_start != -1, "JSON payload start marker not found in output"
    json_str = output[json_start:].strip()
    data = json.loads(json_str)

    assert data["n_images"] == 6
    assert data["n_specimens"] == 3  # 6 images / 2 per specimen = 3 specimens
    assert "INT8" in data["backend"]
    assert "prevalence" in data
    assert "economic_risk" in data
    assert 0.0 <= data["prevalence"]["em"] <= 1.0
    assert data["economic_risk"]["pond_area_acres"] == 2.0
    assert data["economic_risk"]["baseline_revenue_bdt"] == 500000.0


def test_revenue_loss_math():
    """The loss formula matches an independently computed value, and beta matches its artefact.

    This test used to re-implement `base_rev * (1 - exp(beta * p))` from the same two constants
    it imported and then assert the result equalled itself, so it could not fail whether the
    formula or the constants were right. Both halves are pinned externally now: the coefficient
    against `reports/tables/exp08_baseline.csv`, and the arithmetic against a literal.
    """
    import pandas as pd

    from scripts.predict_farm import BETA_WSD, DEFAULT_BASELINE_REVENUE_PER_ACRE_BDT

    # beta is not a free parameter -- it is the S2 bridge-facing estimate Stage 9 reported.
    baseline = pd.read_csv(PROJECT_ROOT / "reports" / "tables" / "exp08_baseline.csv")
    row = baseline[baseline["specification"].astype(str).str.startswith("S2")]
    assert not row.empty, "exp08_baseline.csv has no S2 row to pin BETA_WSD against"
    assert abs(float(row["estimate"].iloc[0]) - BETA_WSD) < 1e-4, (
        f"BETA_WSD {BETA_WSD} has drifted from exp08_baseline.csv"
    )

    # Independently computed: 500000 * (1 - exp(-0.201695 * 0.5)) = 500000 * 0.09592911
    loss = 500_000.0 * (1.0 - np.exp(BETA_WSD * 0.50))
    assert abs(loss - 47_964.55) < 1.0, loss

    # The default baseline is an ASSUMPTION, not a survey estimate. Pinned so that if anyone
    # later sources a real per-acre revenue, this test forces the comment to be updated too.
    assert DEFAULT_BASELINE_REVENUE_PER_ACRE_BDT == 250_000.0


def test_specimen_parser_agrees_with_the_manifest_on_every_image():
    """D-007 lives or dies here: the demo CLI must group photographs by specimen, not by file.

    The previous pattern excluded hyphens and so matched 0 of 1,149 ShrimpDiseaseBD filenames,
    silently turning every photograph into its own specimen -- 1,149 "specimens" instead of 416.
    That is the leakage error the whole thesis is about, occurring inside the script the panel
    watches run.
    """
    import pandas as pd

    from scripts.predict_farm import parse_specimen_id_from_filename

    manifest_path = PROJECT_ROOT / "data" / "interim" / "image_manifest.csv"
    if not manifest_path.exists():
        pytest.skip("image manifest not present")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["subset"] == "raw"]

    mismatches = [
        (path, expected, parse_specimen_id_from_filename(Path(path).name))
        for path, expected in zip(manifest["path"], manifest["specimen_id"])
        if parse_specimen_id_from_filename(Path(path).name) != expected
    ]
    assert not mismatches, (
        f"{len(mismatches)} of {len(manifest)} filenames parse to the wrong specimen; "
        f"first three: {mismatches[:3]}"
    )

    parsed = {parse_specimen_id_from_filename(Path(p).name) for p in manifest["path"]}
    assert len(parsed) == manifest["specimen_id"].nunique() == 416


def test_underscore_filename_convention_still_parses():
    """The docstring's own example must keep working alongside the dataset convention."""
    from scripts.predict_farm import parse_specimen_id_from_filename

    assert parse_specimen_id_from_filename("IMG_012_1.jpg") == "IMG_012"
    assert parse_specimen_id_from_filename("Healthy-1-img-3.jpg") == "Healthy-1"
    assert parse_specimen_id_from_filename("no_pattern_here.jpg") == "no_pattern_here"
