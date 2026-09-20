"""Export MobileNetV3-Large to ONNX and dynamic INT8 for edge deployment.

Stage 11 edge-deployment pipeline:
1. Instantiates `mobilenetv3_large_100` (4-class classification head for shrimp disease).
2. Exports PyTorch model to ONNX FP32 (`reports/models/mobilenetv3_large_100.onnx`).
3. Quantises ONNX model to dynamic INT8 (`reports/models/mobilenetv3_large_100_int8.onnx`).
4. Verifies numerical parity between PyTorch CPU and ONNX FP32 (< 1e-4 max absolute difference).
5. Benchmarks CPU inference latency (20 warmup runs, 200 trials) for PyTorch CPU, ONNX FP32, and
   ONNX INT8 (reporting mean, std, p50, p90, p99 in ms).
6. Compares model parameter count and on-disk size against the FeatherNetX baseline (0.739 M / 2.82 MB).
7. Emits structured benchmark findings to `reports/tables/exp12_export.csv`.

Design notes
------------
Edge deployment requires measurable runtime properties rather than aspirational claims. While
ConvNeXt-Tiny achieved the highest validation macro-F1 (0.7189), MobileNetV3-Large (0.6705 macro-F1)
was selected as the candidate for edge deployment due to its 6.6x smaller parameter footprint
(4.21 M vs 27.82 M) and 16.14 MB uncompressed footprint. Dynamic INT8 quantisation further reduces
the binary size for constrained microcontroller or edge-gateway environments.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static,
)
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shrimp_risk.common.paths import REPORTS, TABLES, resolve_image_manifest
from shrimp_risk.common.seeding import seed_everything
from shrimp_risk.vision import splits as S
from shrimp_risk.vision.datasets import CLASSES
from shrimp_risk.vision.models import count_parameters, create_model

MODELS_DIR = REPORTS / "models"
SEED = 2025
IMG_SIZE = 224
NUM_CLASSES = 4
WARMUP_RUNS = 20
BENCHMARK_RUNS = 200

# FeatherNetX published baseline (Gao et al., 2024; see Master Plan & Decision Log)
FEATHERNETX_PARAMS_M = 0.739
FEATHERNETX_SIZE_MB = 2.82


def export_onnx(model: torch.nn.Module, onnx_path: Path, img_size: int = IMG_SIZE) -> None:
    """Export PyTorch evaluation model to ONNX FP32."""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


class _CalibrationReader(CalibrationDataReader):
    """Feeds real training images to the quantiser so it can learn activation ranges.

    Calibration images come from the TRAINING specimens of the deployment fold only. Calibrating
    on the held-out fold would fit the quantisation scales to the set the accuracy is reported
    from -- the same error as image-level splitting, one level down the stack.
    """

    def __init__(self, paths: list[str], input_name: str) -> None:
        self._data = iter([{input_name: preprocess_image(p)} for p in paths])

    def get_next(self):
        return next(self._data, None)


def quantize_onnx_int8(
    onnx_fp32_path: Path,
    onnx_int8_path: Path,
    calibration_paths: list[str] | None = None,
) -> str:
    """Quantise to INT8, statically where calibration data is available.

    Why not dynamic quantisation
    ----------------------------
    This function used to call `quantize_dynamic`, and the artefact it produced was deployed and
    described in the report as carrying negligible accuracy loss. Nobody measured it. Measured on
    the held-out fold, dynamic INT8 scores **macro-F1 0.2380** against the FP32 model's 0.6989 --
    chance-level on four classes, agreeing with FP32 on only 30% of images. It was also **24x
    slower** (59 ms vs 2.45 ms), so it lost on both axes it was supposed to win.

    That is not a bug in this project; it is the documented behaviour of per-tensor dynamic
    quantisation on depthwise-separable backbones. MobileNetV3's depthwise convolutions have
    wildly different per-channel activation ranges, and a single scale per tensor cannot cover
    them. Per-channel dynamic quantisation does not rescue it either (measured: 0.2411).

    Static quantisation does work, because it observes real activation ranges on calibration data
    instead of guessing them at runtime: **4.46 MB, 3.46 ms, macro-F1 0.5951**. The accuracy cost
    is real (-0.104 macro-F1) and is reported rather than hidden.

    Falls back to dynamic only when no calibration data is available, and returns which method was
    used so the caller can record it.
    """
    if calibration_paths:
        input_name = ort.InferenceSession(
            str(onnx_fp32_path), providers=["CPUExecutionProvider"]
        ).get_inputs()[0].name
        quantize_static(
            model_input=str(onnx_fp32_path),
            model_output=str(onnx_int8_path),
            calibration_data_reader=_CalibrationReader(calibration_paths, input_name),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
        return f"static QDQ, per-channel, {len(calibration_paths)} calibration images"

    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
    )
    return "dynamic per-tensor (NO calibration data -- accuracy not defensible on this backbone)"


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Match `datasets.py`'s evaluation transform exactly: resize, scale, normalise, CHW."""
    from PIL import Image

    arr = np.asarray(
        Image.open(path).convert("RGB").resize((img_size, img_size)), dtype=np.float32
    ) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None]


def evaluate_onnx_accuracy(
    onnx_path: Path, paths: list[str], labels: list[int]
) -> dict[str, float]:
    """Score an exported graph on the held-out fold.

    This is the measurement whose absence let a randomly initialised network -- and later a
    catastrophically quantised one -- ship as the deployment artefact. Size and latency are
    properties of the graph and look identical whether the weights carry any information at all,
    so an export pipeline that checks only those two cannot detect either failure.

    The fold scored here is the one the deployment checkpoint held out, so the figure is an
    unbiased estimate for this network. It is a SINGLE fold and must never be quoted as the
    backbone's performance -- Experiment 04's 5-fold mean +/- SD remains that figure.
    """
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    preds = [int(np.argmax(sess.run(None, {name: preprocess_image(p)})[0][0])) for p in paths]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "preds": preds,
    }


def verify_numerical_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    img_size: int = IMG_SIZE,
    tolerance: float = 1e-4,
) -> float:
    """Verify PyTorch and ONNX predictions agree within tolerance on random input."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    with torch.no_grad():
        torch_logits = model(x).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"input": x.numpy()})[0]

    max_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
    if max_diff > tolerance:
        raise ValueError(
            f"Numerical parity check failed: max difference {max_diff:.6e} > tolerance {tolerance:.6e}"
        )
    return max_diff


def benchmark_pytorch_cpu(
    model: torch.nn.Module,
    img_size: int = IMG_SIZE,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCHMARK_RUNS,
) -> dict[str, float]:
    """Measure single-image CPU inference latency for PyTorch."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def benchmark_onnx_cpu(
    onnx_path: Path,
    img_size: int = IMG_SIZE,
    warmup: int = WARMUP_RUNS,
    runs: int = BENCHMARK_RUNS,
) -> dict[str, float]:
    """Measure single-image CPU inference latency for ONNX Runtime."""
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    input_name = sess.get_inputs()[0].name

    for _ in range(warmup):
        sess.run(None, {input_name: x})

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=str, default="mobilenetv3_large_100")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS)
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS)
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="trained weights to export (default: reports/models/<backbone>_best.pt)")
    parser.add_argument("--fold", type=int, default=0,
                        help="deployment fold; must match train_edge_checkpoint.py")
    parser.add_argument("--calib", type=int, default=128,
                        help="calibration images drawn from that fold's TRAINING specimens")
    parser.add_argument(
        "--allow-untrained", action="store_true",
        help="export randomly initialised weights -- benchmarks size and latency ONLY, and the "
             "resulting artefact has no diagnostic skill. Never use this for a deliverable.")
    args = parser.parse_args()

    seed_everything(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # Load the weights early stopping selected, not a fresh network.
    #
    # This script used to call `create_model(..., pretrained=False)` and export whatever came
    # back. That is a randomly initialised backbone: on four classes it returns ~0.25 for every
    # input, so the exported artefact measured the right size and the right latency while having
    # no diagnostic skill whatsoever. Size and latency do not depend on the weights, which is
    # exactly why the defect survived -- every number in `exp12_export.csv` was correct.
    #
    # The refusal below is deliberate. An export that silently falls back to random weights is
    # the failure this script already shipped once.
    ckpt_path = Path(args.checkpoint) if args.checkpoint else MODELS_DIR / f"{args.backbone}_best.pt"

    print(f"Instantiating model: {args.backbone} (classes={NUM_CLASSES})...")
    model = create_model(args.backbone, num_classes=NUM_CLASSES, pretrained=False)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        provenance = (
            f"{ckpt_path.name} (epoch {ckpt.get('best_epoch', '?')}, "
            f"val {ckpt.get('selection_metric', 'macro_f1')} {ckpt.get('best_val_f1', float('nan')):.4f})"
            if isinstance(ckpt, dict) else ckpt_path.name
        )
        print(f"  Loaded trained weights from {provenance}")
    elif args.allow_untrained:
        provenance = "UNTRAINED (random initialisation) -- size and latency only, no skill"
        print(f"  WARNING: {provenance}")
    else:
        print(
            f"\nERROR: no trained checkpoint at {ckpt_path}\n"
            f"  Exporting an untrained network produces an artefact with no diagnostic skill.\n"
            f"  Train one first:\n"
            f"    .venv/Scripts/python.exe scripts/train_edge_checkpoint.py --backbone {args.backbone}\n"
            f"  Or pass --allow-untrained to benchmark size and latency only.\n",
            file=sys.stderr,
        )
        return 1

    model.eval()
    total_params, trainable_params = count_parameters(model)
    params_m = total_params / 1e6

    onnx_fp32_path = MODELS_DIR / f"{args.backbone}.onnx"
    onnx_int8_path = MODELS_DIR / f"{args.backbone}_int8.onnx"

    print(f"Exporting PyTorch model to ONNX FP32: {onnx_fp32_path}...")
    export_onnx(model, onnx_fp32_path, img_size=args.img_size)
    size_fp32_mb = onnx_fp32_path.stat().st_size / (1024 ** 2)

    print("Verifying numerical parity (PyTorch vs ONNX FP32)...")
    max_diff = verify_numerical_parity(model, onnx_fp32_path, img_size=args.img_size)
    print(f"  Parity verified: max absolute difference = {max_diff:.6e} (< 1e-4)")

    # Calibration and evaluation both come from the deployment fold: calibration from its
    # TRAINING specimens, accuracy from its held-out specimens. Nothing the quantiser sees is
    # anything the score is reported on.
    manifest = S.load_raw_manifest(resolve_image_manifest())
    fold = S.make_specimen_folds(manifest, n_splits=5, seed=SEED)[args.fold]
    train_rows, test_rows = manifest.iloc[fold.train], manifest.iloc[fold.test]
    calib_paths = train_rows["path"].sample(n=min(args.calib, len(train_rows)),
                                            random_state=SEED).tolist()
    eval_paths = test_rows["path"].tolist()
    eval_labels = [CLASSES.index(c) for c in test_rows["class"]]
    print(f"Calibration: {len(calib_paths)} training images | "
          f"Evaluation: {len(eval_paths)} held-out images, "
          f"{test_rows['specimen_id'].nunique()} specimens (fold {fold.fold})")

    print(f"Quantising to INT8: {onnx_int8_path}...")
    quant_method = quantize_onnx_int8(onnx_fp32_path, onnx_int8_path, calib_paths)
    print(f"  Method: {quant_method}")
    size_int8_mb = onnx_int8_path.stat().st_size / (1024 ** 2)

    # Score both graphs. Without this an untrained or badly quantised export is indistinguishable
    # from a good one, because size and latency do not depend on what the weights encode.
    print("Scoring exported graphs on the held-out fold...")
    acc_fp32 = evaluate_onnx_accuracy(onnx_fp32_path, eval_paths, eval_labels)
    acc_int8 = evaluate_onnx_accuracy(onnx_int8_path, eval_paths, eval_labels)
    top1_agreement = float(np.mean(np.array(acc_fp32["preds"]) == np.array(acc_int8["preds"])))
    print(f"  ONNX FP32  acc {acc_fp32['accuracy']:.4f}  macro-F1 {acc_fp32['macro_f1']:.4f}")
    print(f"  ONNX INT8  acc {acc_int8['accuracy']:.4f}  macro-F1 {acc_int8['macro_f1']:.4f}"
          f"   (delta {acc_int8['macro_f1'] - acc_fp32['macro_f1']:+.4f})")
    print(f"  FP32 vs INT8 top-1 agreement: {top1_agreement:.2%}")
    if acc_int8["macro_f1"] < acc_fp32["macro_f1"] - 0.15:
        print("  WARNING: quantisation cost more than 0.15 macro-F1. This artefact is NOT "
              "deployable; report FP32 as the deployment model.", file=sys.stderr)

    print(f"Benchmarking inference latency on CPU ({args.warmup} warmup, {args.runs} runs)...")
    print("  1. PyTorch CPU...")
    lat_torch = benchmark_pytorch_cpu(model, img_size=args.img_size, warmup=args.warmup, runs=args.runs)
    print(f"     Mean: {lat_torch['mean_ms']:.2f} ms +/- {lat_torch['std_ms']:.2f} ms (p50: {lat_torch['p50_ms']:.2f} ms)")

    print("  2. ONNX FP32 (CPU)...")
    lat_onnx_fp32 = benchmark_onnx_cpu(onnx_fp32_path, img_size=args.img_size, warmup=args.warmup, runs=args.runs)
    print(f"     Mean: {lat_onnx_fp32['mean_ms']:.2f} ms +/- {lat_onnx_fp32['std_ms']:.2f} ms (p50: {lat_onnx_fp32['p50_ms']:.2f} ms)")

    print("  3. ONNX INT8 (CPU)...")
    lat_onnx_int8 = benchmark_onnx_cpu(onnx_int8_path, img_size=args.img_size, warmup=args.warmup, runs=args.runs)
    print(f"     Mean: {lat_onnx_int8['mean_ms']:.2f} ms +/- {lat_onnx_int8['std_ms']:.2f} ms (p50: {lat_onnx_int8['p50_ms']:.2f} ms)")

    # Build results table
    rows = [
        {
            "format": "PyTorch CPU",
            "precision": "FP32",
            "parameters_m": round(params_m, 3),
            "file_size_mb": round(size_fp32_mb, 2),
            "latency_mean_ms": round(lat_torch["mean_ms"], 2),
            "latency_std_ms": round(lat_torch["std_ms"], 2),
            "latency_p50_ms": round(lat_torch["p50_ms"], 2),
            "latency_p90_ms": round(lat_torch["p90_ms"], 2),
            "latency_p99_ms": round(lat_torch["p99_ms"], 2),
            "ratio_to_feathernetx_size": round(size_fp32_mb / FEATHERNETX_SIZE_MB, 2),
            "ratio_to_feathernetx_params": round(params_m / FEATHERNETX_PARAMS_M, 2),
            "holdout_accuracy": round(acc_fp32["accuracy"], 4),
            "holdout_macro_f1": round(acc_fp32["macro_f1"], 4),
            "quantisation": "none",
        },
        {
            "format": "ONNX Runtime",
            "precision": "FP32",
            "parameters_m": round(params_m, 3),
            "file_size_mb": round(size_fp32_mb, 2),
            "latency_mean_ms": round(lat_onnx_fp32["mean_ms"], 2),
            "latency_std_ms": round(lat_onnx_fp32["std_ms"], 2),
            "latency_p50_ms": round(lat_onnx_fp32["p50_ms"], 2),
            "latency_p90_ms": round(lat_onnx_fp32["p90_ms"], 2),
            "latency_p99_ms": round(lat_onnx_fp32["p99_ms"], 2),
            "ratio_to_feathernetx_size": round(size_fp32_mb / FEATHERNETX_SIZE_MB, 2),
            "ratio_to_feathernetx_params": round(params_m / FEATHERNETX_PARAMS_M, 2),
            "holdout_accuracy": round(acc_fp32["accuracy"], 4),
            "holdout_macro_f1": round(acc_fp32["macro_f1"], 4),
            "quantisation": "none",
        },
        {
            "format": "ONNX Runtime",
            "precision": "INT8 (Static QDQ)",
            "parameters_m": round(params_m, 3),
            "file_size_mb": round(size_int8_mb, 2),
            "latency_mean_ms": round(lat_onnx_int8["mean_ms"], 2),
            "latency_std_ms": round(lat_onnx_int8["std_ms"], 2),
            "latency_p50_ms": round(lat_onnx_int8["p50_ms"], 2),
            "latency_p90_ms": round(lat_onnx_int8["p90_ms"], 2),
            "latency_p99_ms": round(lat_onnx_int8["p99_ms"], 2),
            "ratio_to_feathernetx_size": round(size_int8_mb / FEATHERNETX_SIZE_MB, 2),
            "ratio_to_feathernetx_params": round(params_m / FEATHERNETX_PARAMS_M, 2),
            "holdout_accuracy": round(acc_int8["accuracy"], 4),
            "holdout_macro_f1": round(acc_int8["macro_f1"], 4),
            "quantisation": quant_method,
            "top1_agreement_vs_fp32": round(top1_agreement, 4),
        },
    ]

    # Every row carries the weight provenance. A benchmark table that does not say whether
    # the exported network was trained is the artefact that let an untrained export ship.
    for row in rows:
        row["weights"] = provenance

    df = pd.DataFrame(rows)
    out_table = TABLES / "exp12_export.csv"
    df.to_csv(out_table, index=False)
    print(f"\nBenchmark table saved to: {out_table}")
    print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
