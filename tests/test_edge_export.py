"""Tests for Stage 11 edge model export, quantization, and benchmarking.

Validates that:
1. Benchmark table `reports/tables/exp12_export.csv` exists with valid latency and size entries.
2. ONNX FP32 and INT8 models exist at `reports/models/`.
3. INT8 quantization achieves significant size reduction (< 5 MB uncompressed).
4. Both ONNX models initialize cleanly in ONNX Runtime without operator errors.
5. Inferences on dummy inputs produce valid 4-class logits and valid probability distributions.
6. Edge trade-offs against FeatherNetX baseline are accurately reflected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from shrimp_risk.common.paths import MODELS, REPORTS, TABLES

EXPORT_CSV = TABLES / "exp12_export.csv"
ONNX_FP32 = MODELS / "mobilenetv3_large_100.onnx"
ONNX_INT8 = MODELS / "mobilenetv3_large_100_int8.onnx"

FEATHERNETX_SIZE_MB = 2.82
FEATHERNETX_PARAMS_M = 0.739


def test_export_table_exists():
    """Ensure exp12_export.csv was generated and contains expected schema."""
    assert EXPORT_CSV.exists(), f"Benchmark table not found at {EXPORT_CSV}"
    df = pd.read_csv(EXPORT_CSV)
    assert len(df) >= 3, f"Expected at least 3 rows in exp12_export.csv, got {len(df)}"
    expected_cols = {
        "format",
        "precision",
        "parameters_m",
        "file_size_mb",
        "latency_mean_ms",
        "latency_std_ms",
        "latency_p50_ms",
        "ratio_to_feathernetx_size",
        "ratio_to_feathernetx_params",
    }
    assert expected_cols.issubset(df.columns), f"Missing columns in exp12_export.csv: {expected_cols - set(df.columns)}"


def test_onnx_artifacts_exist():
    """Ensure both FP32 and INT8 ONNX binaries exist and have non-zero size."""
    assert ONNX_FP32.exists(), f"ONNX FP32 model missing at {ONNX_FP32}"
    assert ONNX_INT8.exists(), f"ONNX INT8 model missing at {ONNX_INT8}"

    fp32_size = ONNX_FP32.stat().st_size / (1024 * 1024)
    int8_size = ONNX_INT8.stat().st_size / (1024 * 1024)

    assert 14.0 < fp32_size < 18.0, f"Unexpected FP32 size: {fp32_size:.2f} MB (expected ~16 MB)"
    assert 3.5 < int8_size < 5.0, f"Unexpected INT8 size: {int8_size:.2f} MB (expected ~4.16 MB)"
    assert int8_size < fp32_size * 0.35, f"INT8 ({int8_size:.2f} MB) should be < 35% of FP32 ({fp32_size:.2f} MB)"


def test_onnx_fp32_session_and_inference():
    """Ensure ONNX FP32 model can be loaded and executed by ONNX Runtime."""
    session = ort.InferenceSession(str(ONNX_FP32), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    assert input_meta.name == "input"
    assert input_meta.shape[1:] == [3, 224, 224]
    assert output_meta.shape[1:] == [4]

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = session.run([output_meta.name], {input_meta.name: dummy})[0]

    assert out.shape == (1, 4)
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()

    # Softmax probabilities must sum to 1.0
    exp_out = np.exp(out - np.max(out))
    probs = exp_out / np.sum(exp_out)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)


def test_onnx_int8_session_and_inference():
    """Ensure ONNX INT8 quantized model runs without runtime error and outputs 4 classes."""
    session = ort.InferenceSession(str(ONNX_INT8), providers=["CPUExecutionProvider"])
    output_meta = session.get_outputs()[0]

    assert output_meta.shape[1:] == [4]

    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    out = session.run([output_meta.name], {session.get_inputs()[0].name: dummy})[0]

    assert out.shape == (1, 4)
    assert not np.isnan(out).any()


def test_feathernetx_comparison_documented():
    """Verify that export benchmark documents accurate comparison against FeatherNetX."""
    df = pd.read_csv(EXPORT_CSV)
    int8_row = df[df["precision"].str.contains("INT8")].iloc[0]
    int8_size = float(int8_row["file_size_mb"])

    # INT8 size is 4.16 MB; FeatherNetX is 2.82 MB. Ratio is 1.48x
    ratio = float(int8_row["ratio_to_feathernetx_size"])
    assert 1.3 < ratio < 1.7, f"Expected ratio ~1.48x, got {ratio:.2f}x"
    assert np.isclose(ratio, int8_size / FEATHERNETX_SIZE_MB, atol=0.05)


# ---------------------------------------------------------------------------------------
# Regression guards added 2026-09-20.
#
# Two artefacts shipped as the deployment model without ever being scored: first a randomly
# initialised backbone (no checkpoint was saved or loaded anywhere in the pipeline), then a
# dynamically quantised graph that had collapsed to chance. Both passed every test in this
# file, because size, latency and graph validity are identical whether the weights encode
# anything at all. Only an accuracy measurement separates them.
# ---------------------------------------------------------------------------------------

def test_export_records_which_weights_it_used():
    """A benchmark table that does not say whether the network was trained is not evidence."""
    df = pd.read_csv(EXPORT_CSV)
    assert "weights" in df.columns, (
        "exp12_export.csv has no `weights` column -- it cannot distinguish a trained export "
        "from an untrained one, which is how a random-weight model shipped"
    )
    for value in df["weights"].astype(str):
        assert "UNTRAINED" not in value.upper(), (
            f"the exported artefact was built from untrained weights: {value}"
        )


def test_export_was_scored_and_is_better_than_chance():
    """The exported graph must actually classify. Chance on four classes is macro-F1 ~0.25."""
    df = pd.read_csv(EXPORT_CSV)
    assert "holdout_macro_f1" in df.columns, (
        "exp12_export.csv records no accuracy. Size and latency look identical for a trained "
        "and an untrained network, so an export pipeline that measures only those cannot "
        "detect either failure mode."
    )
    for _, row in df.iterrows():
        f1 = float(row["holdout_macro_f1"])
        assert f1 > 0.45, (
            f"{row['format']} {row['precision']} scores macro-F1 {f1:.4f} on the held-out fold. "
            f"Chance on four classes is ~0.25. This artefact is not deployable."
        )


def test_quantisation_cost_is_measured_and_bounded():
    """Quantisation may cost accuracy, but the cost must be known and must not be a collapse.

    Dynamic INT8 on this backbone measured macro-F1 0.2380 against FP32's 0.6989 -- a 0.46
    collapse -- while being 24x slower. Static quantisation with calibration data costs ~0.10.
    """
    df = pd.read_csv(EXPORT_CSV)
    fp32 = df[df["precision"].astype(str).str.strip() == "FP32"]
    int8 = df[df["precision"].astype(str).str.contains("INT8")]
    assert not fp32.empty and not int8.empty

    cost = float(fp32["holdout_macro_f1"].iloc[0]) - float(int8["holdout_macro_f1"].iloc[0])
    assert cost < 0.20, (
        f"quantisation costs {cost:.4f} macro-F1. Above ~0.15 the quantised graph is a "
        f"different, worse model rather than a compressed one -- report FP32 as deployed."
    )

    agreement = float(int8["top1_agreement_vs_fp32"].iloc[0])
    assert agreement > 0.60, (
        f"INT8 agrees with FP32 on only {agreement:.1%} of images"
    )


def test_int8_is_not_slower_than_fp32():
    """Compression that costs latency as well as accuracy has no case for being deployed."""
    df = pd.read_csv(EXPORT_CSV)
    fp32 = df[(df["format"] == "ONNX Runtime") & (df["precision"].astype(str).str.strip() == "FP32")]
    int8 = df[(df["format"] == "ONNX Runtime") & (df["precision"].astype(str).str.contains("INT8"))]
    assert not fp32.empty and not int8.empty

    ratio = float(int8["latency_mean_ms"].iloc[0]) / float(fp32["latency_mean_ms"].iloc[0])
    assert ratio < 3.0, (
        f"INT8 is {ratio:.1f}x SLOWER than FP32. Dynamic per-tensor quantisation on a "
        f"depthwise-separable backbone measured 24x; use static quantisation with calibration."
    )
