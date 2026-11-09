"""Farm-level shrimp disease prediction and economic risk CLI.

Stage 11 edge deployment demonstration:
1. Loads image inputs (from file list, directory, or synthetic batch for testing).
2. Runs inference using ONNX Runtime (FP32 or dynamic INT8) or PyTorch on CPU.
3. Obtains raw uncalibrated posteriors per D-041 (temperature scaling worsened SLD-EM error in Stage 8).
4. Aggregates image posteriors to specimen-level (mean across photographs of the same animal).
5. Computes disease prevalence via both naive counting (CC) and prior-shift corrected SLD-EM.
6. Translates estimated White Spot prevalence into expected farm revenue loss (BDT) using the
   Just-Pope S2 econometric specification (beta = -0.2017, p = 0.0685).
7. Outputs formatted results with LIVE tags per D-022.

Usage:
    .venv/Scripts/python.exe scripts/predict_farm.py --synthetic 20 --farm-area 2.5
    .venv/Scripts/python.exe scripts/predict_farm.py --images data/raw/sample1.jpg data/raw/sample2.jpg
    .venv/Scripts/python.exe scripts/predict_farm.py --images-dir path/to/farm_photos/ --int8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shrimp_risk.common.paths import MODELS, REPORTS
from shrimp_risk.common.seeding import seed_everything
from shrimp_risk.integration import quantification as Q
from shrimp_risk.vision.datasets import CLASSES, IMAGENET_MEAN, IMAGENET_STD

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Class distribution across the 1,149 training images in order: [BG, BG_WSSV, Healthy, WSSV]
# [198, 220, 403, 328] / 1149
TRAIN_PRIOR = np.array([0.172324, 0.191471, 0.350740, 0.285466], dtype="float64")

# S2 Econometric bridge parameters from exp08_baseline.csv
BETA_WSD = -0.201695
SE_BETA = 0.110703
P_VALUE = 0.068463
# Illustrative per-acre revenue used to express risk in Taka.
#
# This was 250,000 BDT, annotated "IFPRI representative smallholder revenue". The only 250000 in
# `reports/` is `wsd_loss_max_bdt` in exp03a_ifpri_descriptives.csv -- the LARGEST single-farm
# White Spot loss in the survey, not a revenue, and not per-acre. Every Taka figure the panel
# would have seen was built on it.
#
# No per-acre revenue exists in the descriptives, so there is nothing to read instead. It is
# therefore declared for what it is: an assumption, labelled on screen, not an estimate. Change
# it with --baseline-revenue; the prevalence and elasticity results do not depend on it.
DEFAULT_BASELINE_REVENUE_PER_ACRE_BDT = 250000.0
BASELINE_REVENUE_IS_ASSUMPTION = True


def preprocess_image(image: Image.Image, img_size: int = 224) -> np.ndarray:
    """Resize and normalise image with ImageNet statistics (output shape: 3, H, W)."""
    img = image.convert("RGB").resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    # Normalise: (x - mean) / std
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    # Transpose to (3, H, W)
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def infer_onnx(
    model_path: Path,
    batch: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run inference using ONNX Runtime. Returns raw logits and elapsed time in ms."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    logits = sess.run(None, {input_name: batch.astype(np.float32)})[0]
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    return logits, elapsed_ms


def infer_pytorch(
    model_name: str,
    batch: np.ndarray,
    checkpoint: Path | None = None,
) -> tuple[np.ndarray, float]:
    """Run inference using PyTorch CPU model. Returns raw logits and elapsed time in ms.

    Loads trained weights. This path used to call `create_model(..., pretrained=False)` and run
    the resulting network, which is randomly initialised: it returned ~0.25 on every class for
    every input while printing a confident diagnosis. It refuses rather than falling back,
    because a silent fallback to random weights is precisely the failure this script shipped.
    """
    import torch
    from shrimp_risk.vision.models import create_model

    ckpt_path = checkpoint or (MODELS / f"{model_name}_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"no trained checkpoint at {ckpt_path}.\n"
            f"  Run: .venv/Scripts/python.exe scripts/train_edge_checkpoint.py "
            f"--backbone {model_name}"
        )

    model = create_model(model_name, num_classes=len(CLASSES), pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt)
    model.eval()
    tensor = torch.from_numpy(batch).float()
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor).cpu().numpy()
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    return logits, elapsed_ms


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def parse_specimen_id_from_filename(filename: str) -> str:
    """Recover which physical shrimp a photograph is of, from its filename.

    This is the single most load-bearing line in the script, because `D-007` -- the project's
    first binding constraint -- says every grouping is on `specimen_id`, never on image. A tray
    of 30 photographs of 10 shrimp is 10 independent observations, not 30, and the prevalence
    estimate and its bootstrap interval are both wrong if that is miscounted.

    The previous pattern was `^([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?)(?:_\\d+)?\\.[a-zA-Z]+$`, written
    against the `IMG_012_1.jpg` example in this docstring. `[A-Za-z0-9]` excludes hyphens, and
    ShrimpDiseaseBD names files `Healthy-1-img-1.jpg`, so it matched **0 of 1,149** filenames in
    the corpus and every photograph silently became its own specimen. The tray in the Stage 11
    demonstration reported "6 images across 6 unique specimens" for two shrimp photographed three
    times each.

    Both conventions are handled now, dataset-native first. Anything unrecognised falls back to
    the stem, which is the safe direction: it over-counts specimens rather than merging two
    animals into one.
    """
    stem = Path(filename).stem

    # ShrimpDiseaseBD: '<Class>-<n>-img-<k>' -> '<Class>-<n>'
    m = re.match(r"^(.+)-img-\d+$", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Underscore convention: 'IMG_012_1' -> 'IMG_012'
    m = re.match(r"^(.+)_\d+$", stem)
    if m:
        return m.group(1)

    return stem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="*", default=None, help="List of image file paths")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory containing images")
    parser.add_argument("--synthetic", type=int, default=None, help="Generate N synthetic images for testing")
    parser.add_argument("--model-path", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--backbone", type=str, default="mobilenetv3_large_100")
    parser.add_argument("--backend", choices=["onnx", "pytorch"], default="onnx")
    parser.add_argument("--int8", action="store_true", help="use the INT8 static-QDQ graph (4.46 MB, macro-F1 0.5951) instead of FP32 (16.04 MB, 0.6989)")
    parser.add_argument("--farm-area", type=float, default=1.0, help="Farm pond area in acres")
    parser.add_argument("--bootstrap", type=int, default=500, help="Number of bootstrap resamples for CI")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--baseline-revenue", type=float,
                        default=DEFAULT_BASELINE_REVENUE_PER_ACRE_BDT,
                        help="assumed gross revenue per acre in BDT; an ASSUMPTION, not an "
                             "estimate -- no per-acre revenue figure exists in the survey")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    seed_everything(args.seed)

    print("=" * 78)
    print("  [LIVE] FARM-LEVEL SHRIMP DISEASE DIAGNOSIS & RISK PREDICTION")
    print("  Multimodal Decision-Level Bridge: Edge Vision -> Just-Pope Econometrics")
    print("=" * 78)

    # 1. Gather images
    image_paths: list[Path] = []
    specimen_ids: list[str] = []

    if args.images_dir:
        dir_path = Path(args.images_dir)
        if not dir_path.is_dir():
            print(f"Error: Directory not found: {dir_path}", file=sys.stderr)
            return 1
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(sorted(dir_path.glob(ext)))

    if args.images:
        for img_str in args.images:
            p = Path(img_str)
            if p.is_file():
                image_paths.append(p)
            else:
                print(f"Warning: Image file not found: {p}", file=sys.stderr)

    # Refuse to fabricate data the caller did not ask for.
    #
    # The guard here used to read `if not image_paths and (args.synthetic is not None or not
    # image_paths)`, whose right operand is true whenever the left one is, so it collapsed to
    # `if not image_paths`. A mistyped path therefore produced a complete, confident farm
    # diagnosis in Taka built on `np.random.rand` pixels, with exit code 0 and nothing in stdout
    # saying so -- the two "file not found" warnings went to stderr and scrolled past.
    #
    # Synthetic input is now opt-in only. Asking for images and getting none is an error.
    asked_for_real_images = bool(args.images) or bool(args.images_dir)
    if not image_paths and asked_for_real_images:
        print(
            "\nError: none of the requested images could be read.\n"
            "  Check the paths, or pass --synthetic N to run on random pixels for a smoke test.\n",
            file=sys.stderr,
        )
        return 1

    using_synthetic = False
    if not image_paths:
        if args.synthetic is None:
            print(
                "\nError: no images given.\n"
                "  Pass --images <files>, --images-dir <dir>, or --synthetic N.\n",
                file=sys.stderr,
            )
            return 1
        n_synth = args.synthetic
        print(f"[*] SYNTHETIC MODE: generating {n_synth} images of random pixels.")
        print("    Nothing below is a measurement. This exercises the pipeline, not the model.")
        using_synthetic = True
        images_list = []
        for i in range(n_synth):
            spec_idx = i // 2
            specimen_ids.append(f"SPECIMEN_{spec_idx:03d}")
            # Generate dummy PIL image
            arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
            images_list.append(Image.fromarray(arr))
    else:
        print(f"[*] Loaded {len(image_paths)} images from disk.")
        images_list = []
        for p in image_paths:
            try:
                img = Image.open(p)
                images_list.append(img)
                specimen_ids.append(parse_specimen_id_from_filename(p.name))
            except Exception as e:
                print(f"Warning: Could not open {p}: {e}", file=sys.stderr)

    if not images_list:
        print("Error: No valid images to process.", file=sys.stderr)
        return 1

    # 2. Preprocess batch
    batch_np = np.stack([preprocess_image(im) for im in images_list], axis=0)
    n_images = len(images_list)
    unique_specimens = np.unique(specimen_ids)
    n_specimens = len(unique_specimens)

    print(f"[*] Sample size: {n_images} images across {n_specimens} unique specimens "
          f"({n_images / n_specimens:.2f} photos/specimen).")

    # 3. Model Inference
    default_onnx = MODELS / f"{args.backbone}{'_int8' if args.int8 else ''}.onnx"
    onnx_path = Path(args.model_path) if args.model_path else default_onnx

    if args.backend == "onnx" and onnx_path.exists():
        backend_desc = f"ONNX Runtime CPU ({'INT8' if args.int8 else 'FP32'}) -> {onnx_path.name}"
        print(f"[*] Backend: {backend_desc}")
        logits, latency_ms = infer_onnx(onnx_path, batch_np)
    else:
        if args.backend == "onnx":
            print(f"[*] ONNX model not found at {onnx_path}, falling back to PyTorch CPU.")
        backend_desc = f"PyTorch CPU -> {args.backbone}"
        print(f"[*] Backend: {backend_desc}")
        logits, latency_ms = infer_pytorch(args.backbone, batch_np)

    per_img_ms = latency_ms / n_images
    print(f"[*] Inference speed: {latency_ms:.2f} ms total ({per_img_ms:.2f} ms/image).")

    # Raw posteriors (per D-041: no temperature scaling for SLD-EM)
    probs = softmax(logits)

    # 4. Specimen Aggregation
    specimen_probs, ordered_specimens = Q.aggregate_to_specimen(probs, np.array(specimen_ids))

    # 5. Prevalence Quantification
    # WSSV Positive classes: BG_WSSV (index 1), WSSV (index 3)
    pos_indices = Q.positive_indices()

    # Estimator 1: Naive Classify-and-Count (CC)
    est_cc = Q.classify_and_count(specimen_probs, pos_indices)

    # Estimator 2: Probabilistic Classify-and-Count (PCC)
    est_pcc = Q.probabilistic_classify_and_count(specimen_probs, pos_indices)

    # Estimator 3: Saerens et al. SLD-EM (Prior shift corrected)
    est_em = Q.saerens_em(specimen_probs, TRAIN_PRIOR, pos_indices)

    # Bootstrap 95% interval for SLD-EM
    boot_res = Q.bootstrap_interval(
        lambda sp: Q.saerens_em(sp, TRAIN_PRIOR, pos_indices),
        specimen_probs,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    # 6. Econometric Risk Translation (Just-Pope S2 specification)
    # Baseline expected revenue:
    base_revenue = args.baseline_revenue * args.farm_area

    # Revenue loss formula: Baseline * (1 - exp(beta * prevalence))
    def revenue_loss(p: float) -> float:
        return float(base_revenue * (1.0 - np.exp(BETA_WSD * p)))

    # A zero-width interval is not a 95% interval. SLD-EM can drive the positive prior to
    # exactly 0 on near-uniform posteriors, at which point every resample returns the same value
    # and the percentile interval degenerates. Say so rather than printing [0.00%, 0.00%] under a
    # "95% Confidence Interval" heading.
    ci_width = float(boot_res["hi"] - boot_res["lo"])
    ci_degenerate = ci_width < 1e-9
    estimator_spread = max(est_cc.prevalence, est_pcc.prevalence, est_em.prevalence) -         min(est_cc.prevalence, est_pcc.prevalence, est_em.prevalence)

    loss_em = revenue_loss(est_em.prevalence)
    loss_lo = revenue_loss(boot_res["lo"])
    loss_hi = revenue_loss(boot_res["hi"])
    loss_cc = revenue_loss(est_cc.prevalence)

    # 7. Print Output
    print("\n" + "-" * 78)
    print("  SPECIMEN CLASSIFICATION BREAKDOWN")
    print("-" * 78)
    print(f"{'Specimen ID':<20} {'Top Class':<12} {'Confidence':<12} {'WSSV Risk':<10}")
    print("-" * 78)
    for sid, pvec in zip(ordered_specimens[:10], specimen_probs[:10]):
        top_idx = int(pvec.argmax())
        top_class = CLASSES[top_idx]
        conf = float(pvec[top_idx])
        is_wssv = top_class in ("WSSV", "BG_WSSV")
        risk_str = "POSITIVE" if is_wssv else "NEGATIVE"
        print(f"{sid:<20} {top_class:<12} {conf:>10.2%}   {risk_str:<10}")

    if len(ordered_specimens) > 10:
        print(f"... and {len(ordered_specimens) - 10} additional specimens.")

    print("\n" + "=" * 78)
    print("  PREVALENCE QUANTIFICATION RESULTS")
    print("=" * 78)
    print(f"{'Method':<25} {'Estimated Prevalence':<22} {'95% Confidence Interval':<25}")
    print("-" * 78)
    print(f"{'Naive Count (CC)':<25} {est_cc.prevalence:>18.2%}   {'[Biased: E[p] != p]':<25}")
    print(f"{'Posterior Mean (PCC)':<25} {est_pcc.prevalence:>18.2%}   {'-':<25}")
    ci_text = (
        "[degenerate -- EM collapsed to a vertex]" if ci_degenerate
        else f"[{boot_res['lo']:.2%}, {boot_res['hi']:.2%}]"
    )
    print(f"{'SLD-EM (Prior-Corrected)':<25} {est_em.prevalence:>18.2%}   {ci_text:<25}")
    if ci_degenerate:
        print("    ! The bootstrap interval has zero width. Every resample returned the same")
        print("      prevalence, so this is not a 95% interval and must not be reported as one.")
    if estimator_spread > 0.25:
        print(f"    ! CC, PCC and SLD-EM disagree by {estimator_spread:.1%}. On a sample this")
        print("      small the estimators are not interchangeable; report the spread.")
    print("-" * 78)
    print(f"[*] Saerens EM iterations: {est_em.iterations} (converged: {est_em.converged})")
    print(f"[*] Calibration state: RAW posteriors (D-041: preserves minimum EM bias)")

    print("\n" + "=" * 78)
    print(f"  FARM-LEVEL ECONOMIC RISK ASSESSMENT (Pond Area: {args.farm_area:.2f} acres)")
    print("=" * 78)
    print(f"  Assumed Baseline Revenue:         {base_revenue:12,.2f} BDT   "
          f"[ASSUMPTION: {args.baseline_revenue:,.0f} BDT/acre, not a survey estimate]")
    print(f"  Econometric Model:                Just-Pope S2 Mean Equation (beta = {BETA_WSD:.4f})")
    print("-" * 78)
    print(f"  Expected Revenue Loss (SLD-EM):   {loss_em:12,.2f} BDT ({loss_em / base_revenue:.2%})")
    print(f"  95% Confidence Interval on Loss:  [{loss_lo:,.2f} BDT, {loss_hi:,.2f} BDT]")
    print(f"  Naive Counting Prediction Loss:   {loss_cc:12,.2f} BDT ({loss_cc / base_revenue:.2%})")
    attenuation = abs(loss_cc - loss_em)
    if loss_cc < loss_em:
        print(f"  Measurement Error Attenuation:    Understates loss by {attenuation:,.2f} BDT")
    elif loss_cc > loss_em:
        print(f"  Measurement Error Distortion:     Overstates loss by {attenuation:,.2f} BDT")
    print("\n[Defense Note]")
    print("  * D-008: Decision-level fusion connects edge vision posteriors to farm risk.")
    print("  * D-007: Photos are grouped by specimen, so a shrimp photographed 3 times counts once.")
    print("  * D-022: Results are generated LIVE on local CPU using deployed edge architecture.")

    if args.json:
        result_payload = {
            "n_images": n_images,
            "n_specimens": len(ordered_specimens),
            "backend": backend_desc,
            "inference_time_ms": float(latency_ms),
            "prevalence": {
                "cc": float(est_cc.prevalence),
                "pcc": float(est_pcc.prevalence),
                "em": float(est_em.prevalence),
                "em_ci_95": [float(boot_res["lo"]), float(boot_res["hi"])],
                "em_iterations": int(est_em.iterations),
                "em_converged": bool(est_em.converged),
            },
            "economic_risk": {
                "pond_area_acres": float(args.farm_area),
                "baseline_revenue_bdt": float(base_revenue),
                "expected_loss_em_bdt": float(loss_em),
                "loss_ci_95_bdt": [float(loss_lo), float(loss_hi)],
                "loss_cc_bdt": float(loss_cc),
                "attenuation_bdt": float(abs(loss_cc - loss_em)),
            },
        }
        print("\n" + "=" * 78)
        print("  JSON RESULT OUTPUT")
        print("=" * 78)
        print(json.dumps(result_payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
