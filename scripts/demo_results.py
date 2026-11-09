"""Print every result this project has produced, in one pass, at demonstration speed.

Why this script exists
----------------------
A panel asks "show me it works". For a deep-learning project the honest answer is awkward: the
experiments behind the headline numbers take 20-45 minutes on a GPU, and nobody watches that in
a review. The dishonest answer -- a script that prints hard-coded numbers and calls it a run --
is worse, because it is a false claim made to examiners.

This script takes the third path. It separates what can genuinely be computed in front of an
audience from what cannot, labels every block with which it is, and names the file every number
came from:

    LIVE      computed in this process, now, from the image manifest or from the per-image
              posteriors stored at run time. Nothing is read from a results table.
    RECORDED  read from the artefact written when the experiment executed. The path is printed
              so it can be opened on the spot.
    ILLUSTRATIVE  computed live, but from inputs typed into this file rather than measured.
              Used once, for the decision-level bridge, where the arithmetic is the point and
              the posteriors are a worked example. Never used for a training outcome.

The distinction is not cosmetic and it is not decoration. The LIVE blocks recompute the
specimen structure, the leakage measurement and every per-class metric from raw inputs, so they
would break if the data or the splitter changed. The RECORDED blocks are training outcomes,
which cannot be reproduced in a demonstration without a GPU and three quarters of an hour.
Where both exist for the same quantity, the script cross-checks them and says so -- that is the
evidence that the tables on the slides were not typed by hand.

Design notes
------------
* Output is plain 7-bit ASCII at 78 columns, no colour codes. It is meant to be screenshotted
  onto a slide and read from the back of a room, and a Windows console in cp1252 raises on
  "+/-" written as a degree-style symbol.
* Nothing here trains, downloads or writes. It is safe to run repeatedly, including minutes
  before a review, and it cannot damage an artefact.
* Every section degrades to an explicit "skipped" line when its input is missing, rather than
  raising. A missing dataset means "not checkable here", not "broken".

Usage
-----
    .venv/Scripts/python.exe scripts/demo_results.py                  # everything, ~15 s
    .venv/Scripts/python.exe scripts/demo_results.py --list           # section names
    .venv/Scripts/python.exe scripts/demo_results.py --only leakage effect
    .venv/Scripts/python.exe scripts/demo_results.py --pause          # Enter between sections
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shrimp_risk.common.paths import (  # noqa: E402
    IMAGE_MANIFEST,
    PROJECT_ROOT,
    RUNS,
    TABLES,
)
from shrimp_risk.common.seeding import environment_fingerprint  # noqa: E402
from shrimp_risk.vision import splits as S  # noqa: E402

WIDTH = 78

# The same five seeds Experiment 01 uses. Reusing them is what makes the live measurement
# comparable to the recorded table rather than merely similar to it.
SEEDS = (2025, 7, 42, 123, 999)

# Mirrors `shrimp_risk.vision.datasets.CLASSES`. Duplicated on purpose: importing that module
# pulls in torch, which costs seconds this script does not need to spend. `tests/test_demo.py`
# asserts the two lists stay identical, so the duplication cannot drift.
CLASSES = ["BG", "BG_WSSV", "Healthy", "WSSV"]

# FeatherNetX (Scientific Reports 2026, s41598-026-44195-z) reports 0.739 M parameters. Quoted
# from the publication -- the only number in this script not produced by code in this repo, and
# it is labelled as published wherever it is printed.
FEATHERNETX_PARAMS_M = 0.739

TITLE = (
    "An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp "
    "Disease Detection Using Image"
)
STUDENT = "G Mohan Jayanth  |  Reg. 19MIS1127  |  Guide: Dr. M. Premalatha"


# ---------------------------------------------------------------------------------------
# Presentation helpers -- deliberately dumb, because a demo that crashes on formatting is
# worse than an ugly one.
# ---------------------------------------------------------------------------------------

def _rule(char: str = "=") -> None:
    print(char * WIDTH)


def _wrap(text: str, indent: str = "  ") -> None:
    """Word-wrap prose to the fixed width so a screenshot never clips a sentence."""
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and len(indent) + len(candidate) > WIDTH:
            print(indent + line)
            line = word
        else:
            line = candidate
    if line:
        print(indent + line)


def _table(headers: list[str], rows: list[list], aligns: str | None = None,
           indent: str = "  ", gap: int = 2) -> None:
    """Aligned plain-text table. `gap` narrows the column separator for wide tables, which is
    what keeps every table inside the 78 columns a projected screenshot can show."""
    srows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in srows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    aligns = aligns or "<" * len(headers)
    separator = " " * gap
    fmt = lambda cells: indent + separator.join(  # noqa: E731
        f"{c:{a}{w}}" for c, a, w in zip(cells, aligns, widths)
    )
    print(fmt(headers).rstrip())
    print(indent + separator.join("-" * w for w in widths))
    for row in srows:
        print(fmt(row).rstrip())


def _relative(path: Path | str) -> str:
    """Project-relative, forward-slashed, so a printed path matches the documentation."""
    candidate = Path(path)
    try:
        candidate = candidate.relative_to(PROJECT_ROOT)
    except (ValueError, TypeError):
        pass
    return candidate.as_posix()


def _source(*paths: Path | str, note: str = "") -> None:
    for path in paths:
        print(f"  source: {_relative(path)}")
    if note:
        _wrap(note, indent="          ")


def _skipped(path: Path, remedy: str) -> None:
    """A missing input is reported, never raised. 'Not checkable here' is not 'broken'."""
    rel = _relative(path)
    print(f"  [skipped] {rel} not found")
    print(f"            {remedy}")
    _SKIPPED.append(rel)


def _crosscheck(label: str, live: float, recorded: float, tol: float = 5e-4) -> None:
    """State whether a freshly computed number equals the one already on the slide.

    This is the single most useful line in the whole script. A panel cannot tell whether a
    results table was generated or typed; recomputing a metric from raw predictions in front
    of them and matching the stored value is the difference between a claim and a check.
    """
    verdict = "MATCH" if abs(live - recorded) <= tol else "DIFFERS"
    print(f"  cross-check {label}: recomputed {live:.4f} | stored {recorded:.4f} -> {verdict}")


_SKIPPED: list[str] = []


def _csv(name: str) -> pd.DataFrame | None:
    path = TABLES / name
    if not path.exists():
        _skipped(path, "run the experiment that writes it (see RUNBOOK.md section 5)")
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------------------
# Section 1 -- the structural fact everything else depends on
# ---------------------------------------------------------------------------------------

def section_structure() -> None:
    if not IMAGE_MANIFEST.exists():
        _skipped(IMAGE_MANIFEST, "run: .venv/Scripts/python.exe scripts/extract_images.py")
        return
    _source(IMAGE_MANIFEST, note="parsed now; no model and no results table involved")
    print()

    full = pd.read_csv(IMAGE_MANIFEST)
    raw = S.load_raw_manifest(IMAGE_MANIFEST)
    per_specimen = raw.groupby("specimen_id").size()
    n_specimens = raw["specimen_id"].nunique()

    print(f"  images in the classification corpus : {len(raw):,}")
    print(f"  individual shrimp (specimen_id)     : {n_specimens:,}")
    print(f"  photographs per specimen            : {len(raw) / n_specimens:.2f}"
          f"   (min {per_specimen.min()}, max {per_specimen.max()})")
    print()

    rows = []
    for cls in sorted(raw["class"].unique()):
        grp = raw[raw["class"] == cls]
        n_spec = grp["specimen_id"].nunique()
        rows.append([cls, f"{len(grp):,}", f"{n_spec:,}", f"{len(grp) / n_spec:.2f}"])
    _table(["class", "images", "specimens", "imgs/spec"], rows, "<>>>")
    print()

    counts = per_specimen.value_counts().sort_index(ascending=False)
    spread = ", ".join(f"{n} specimens x {k} photos" for k, n in counts.items())
    print(f"  {spread}")
    print(f"  excluded from every split: {len(full) - len(raw):,} annotated republications")
    print("  (byte-identical copies of raw images -- D-013)")
    print()
    _wrap("This is the fact the whole project rests on. The dataset is not 1,149 independent "
          "observations. It is 416, photographed 2.76 times each -- so a split that ignores "
          "which shrimp a photograph shows is not really splitting the data.")


# ---------------------------------------------------------------------------------------
# Section 2 -- how much of the test set a conventional split contaminates
# ---------------------------------------------------------------------------------------

def section_leakage() -> None:
    if not IMAGE_MANIFEST.exists():
        _skipped(IMAGE_MANIFEST, "run: .venv/Scripts/python.exe scripts/extract_images.py")
        return
    _source(IMAGE_MANIFEST,
            note="both splitters run now, over 5 seeds; splits only, nothing is trained")
    print()

    manifest = S.load_raw_manifest(IMAGE_MANIFEST)
    records = []
    for seed in SEEDS:
        for protocol, splitter in (("image_level", S.make_image_level_splits),
                                   ("specimen_level", S.make_specimen_level_splits)):
            record = S.leakage_report(manifest, splitter(manifest, seed=seed))
            record["seed"] = seed
            records.append(record)
    live = pd.DataFrame(records)

    rows = []
    for _, r in live.iterrows():
        rows.append([r["seed"], r["protocol"], f"{r['n_test_images']:.0f}",
                     f"{r['shared_train_test_specimens']:.0f}",
                     f"{r['test_images_with_train_sibling']:.0f}",
                     f"{r['pct_test_leaked']:.2f}%"])
    _table(["seed", "protocol", "test imgs", "shared spec", "leaked imgs", "% test leaked"],
           rows, "<<>>>>")
    print()

    summary = live.groupby("protocol")["pct_test_leaked"].agg(["mean", "std"])
    print("  RESULT -- share of TEST images whose specimen also appears in TRAIN")
    for protocol, r in summary.iterrows():
        print(f"    {protocol:16s} {r['mean']:6.2f}% +/- {r['std']:.2f}")
    print()

    recorded = _csv("exp01_leakage_summary.csv")
    if recorded is not None:
        for protocol, r in summary.iterrows():
            match = recorded[recorded["protocol"] == protocol]
            if not match.empty:
                _crosscheck(protocol, r["mean"],
                            float(match["pct_test_leaked_mean"].iloc[0]), tol=0.01)
        print()
    _wrap("Roughly seven of every eight test images has a photograph of the same shrimp in "
          "training. Under specimen-level grouping it is exactly zero -- which doubles as the "
          "correctness check on the splitter, because any value above zero means it is broken.")


# ---------------------------------------------------------------------------------------
# Section 3 -- does that contamination inflate the score?
# ---------------------------------------------------------------------------------------

def section_effect() -> None:
    summary = _csv("exp02_summary.csv")
    runs = _csv("exp02_runs.csv")
    if summary is None or runs is None:
        return
    _source(TABLES / "exp02_summary.csv", TABLES / "exp02_runs.csv",
            note="9 training runs: 3 conditions x 3 seeds, rexnet_150, ~20 min on an RTX 5060")
    print()

    order = ["A_image_level_noaug", "B_image_level", "C_specimen_level"]
    rows = []
    for condition in order:
        s = summary[summary["condition"] == condition].iloc[0]
        r = runs[runs["condition"] == condition].iloc[0]
        rows.append([condition, r["protocol"], "yes" if r["augment"] else "no",
                     f"{s['pct_test_leaked']:.2f}", f"{s['test_f1_macro_mean']:.4f}",
                     f"{s['test_f1_macro_sd']:.4f}"])
    _table(["condition", "split", "aug", "leak %", "test macro-F1", "seed SD"],
           rows, "<<<>>>")
    print()

    def _mean(condition: str, column: str) -> float:
        return float(summary.loc[summary["condition"] == condition, column].iloc[0])

    b = _mean("B_image_level", "test_f1_macro_mean")
    c = _mean("C_specimen_level", "test_f1_macro_mean")
    c_sd = _mean("C_specimen_level", "test_f1_macro_sd")
    print(f"  LEAKAGE-ATTRIBUTABLE OPTIMISM   B - C = {b - c:+.4f} macro-F1")
    print(f"  specimen-level seed SD                  {c_sd:.4f}")
    print()
    _wrap("The gap was positive on every one of the three matched seeds and clears the seed "
          "spread of both arms -- the pre-registered inconclusive criterion is not met. B and C "
          "differ in exactly one factor: the split protocol. Backbone, optimiser, schedule, batch "
          "size, budget, patience, augmentation and seeds are all identical, and a test enforces "
          "that. What carries the claim is the EXPOSURE printed above, which is a property of the "
          "split and involves no model; at three seeds the magnitude of the inflation is not "
          "tightly bounded (paired p = 0.056).")
    print()

    specimens = {}
    if IMAGE_MANIFEST.exists():
        raw = S.load_raw_manifest(IMAGE_MANIFEST)
        specimens = raw.groupby("class")["specimen_id"].nunique().to_dict()
    per_class = []
    for cls in sorted(CLASSES, key=lambda c: specimens.get(c, 0)):
        column = f"f1_{cls}"
        # Rounded before differencing so the printed drop is the difference between the two
        # printed columns. Differencing first gives -0.131 where the slide says -0.132, and a
        # panel that subtracts the row in their head must not find the script disagreeing.
        b_cls = round(runs.loc[runs["condition"] == "B_image_level", column].mean(), 3)
        c_cls = round(runs.loc[runs["condition"] == "C_specimen_level", column].mean(), 3)
        per_class.append([cls, specimens.get(cls, "?"), f"{b_cls:.3f}", f"{c_cls:.3f}",
                          f"{c_cls - b_cls:+.3f}"])
    _table(["class", "specimens", "B image-level", "C specimen-level", "drop"],
           per_class, "<>>>>")
    print()
    _wrap("The drop is ordered by specimen count: the fewer individuals a class contains, the "
          "more of its test set memorising those individuals covers. Black Gill -- the class a "
          "farmer most needs flagged early -- is the one that collapses.")


# ---------------------------------------------------------------------------------------
# Section 4 -- the metrics behind those numbers, recomputed from stored predictions
# ---------------------------------------------------------------------------------------

def section_matrix() -> None:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    runs = _csv("exp02_runs.csv")
    pairs = (
        ("B_image_level", "B_image_level__seed2025", "image-level, the leaky protocol"),
        ("C_specimen_level", "C_specimen_level__seed2025", "specimen-level, the honest one"),
    )

    for condition, tag, description in pairs:
        npz = RUNS / "exp02" / f"{tag}_predictions.npz"
        if not npz.exists():
            _skipped(npz, "run: .venv/Scripts/python.exe scripts/exp02_leakage_effect.py")
            continue
        stored = np.load(npz, allow_pickle=True)
        labels = stored["labels"]
        predicted = stored["probs"].argmax(axis=1)
        n_specimens = len(set(stored["specimen_ids"].tolist()))

        print(f"  {condition}  --  {description}")
        _source(npz, note=f"{len(labels)} test images from {n_specimens} specimens; the "
                          f"per-image posteriors were saved at run time and every metric "
                          f"below is recomputed from them now")
        print()
        report = classification_report(labels, predicted, labels=list(range(len(CLASSES))),
                                       target_names=CLASSES, digits=4, zero_division=0)
        for line in report.rstrip().splitlines():
            print(f"    {line}".rstrip())
        print()

        matrix = confusion_matrix(labels, predicted, labels=list(range(len(CLASSES))))
        _table(["true \\ predicted"] + CLASSES,
               [[cls] + [int(v) for v in row] for cls, row in zip(CLASSES, matrix)],
               "<" + ">" * len(CLASSES), indent="    ")
        print()

        if runs is not None:
            recorded = runs[(runs["condition"] == condition) & (runs["seed"] == 2025)]
            if not recorded.empty:
                _crosscheck("macro-F1",
                            float(f1_score(labels, predicted, average="macro", zero_division=0)),
                            float(recorded["test_f1_macro"].iloc[0]))
        print()

    _wrap("These are single runs at seed 2025, recomputed from raw posteriors; the slide "
          "figures are means over three seeds, so they will not be identical to the numbers "
          "above. The cross-check line is the point -- the stored table agrees with what the "
          "predictions themselves say, to four decimal places.")


# ---------------------------------------------------------------------------------------
# Section 5 -- which backbone, once the protocol is honest
# ---------------------------------------------------------------------------------------

def section_backbones() -> None:
    summary = _csv("exp04_summary.csv")
    folds = _csv("exp04_folds.csv")
    if summary is None:
        return
    _source(TABLES / "exp04_summary.csv", TABLES / "exp04_folds.csv",
            note="25 training runs: 5 backbones x 5 specimen-level folds, ~45 min on an "
                 "RTX 5060. Every one of the 416 specimens is tested exactly once per backbone")
    print()

    learning_rates = {}
    if folds is not None and "lr" in folds.columns:
        learning_rates = folds.groupby("backbone")["lr"].first().to_dict()

    ranked = summary.sort_values("f1_macro_mean", ascending=False)
    top = ranked.iloc[0]
    top_sd = float(top["f1_macro_sd"])

    rows = []
    for _, r in ranked.iterrows():
        lr = learning_rates.get(r["backbone"], r["lr"])
        gap = float(top["f1_macro_mean"]) - float(r["f1_macro_mean"])
        # The pre-committed rule, applied by the script rather than by eye: a gap smaller than
        # the fold-to-fold SD of either model does not support a ranking between them.
        noise = max(top_sd, float(r["f1_macro_sd"]))
        if r["backbone"] == top["backbone"]:
            verdict = "best"
        elif gap < noise:
            verdict = "within noise"
        else:
            verdict = f"{gap / noise:.1f}x noise"
        rows.append([r["backbone"], f"{lr:g}", f"{r['f1_macro_mean']:.4f}",
                     f"{r['f1_macro_sd']:.4f}", f"{gap:.4f}" if gap else "--", verdict])
    _table(["backbone", "lr", "macro-F1", "fold SD", "gap to best", "ranking rule"],
           rows, "<>>>><")
    print()
    print(f"  best: {top['backbone']} {top['f1_macro_mean']:.4f} +/- {top_sd:.4f}")
    print()
    _wrap("Black Gill F1, the weakest class for every one of them: " + " | ".join(
        f"{r['backbone']} {r['f1_BG_mean']:.4f}" for _, r in ranked.iterrows()))
    print()
    _wrap("Most of that ranking is not real, and the script says so rather than leaving it to "
          "the reader. Every adjacent pair sits inside the fold SD, so only the gap from top to "
          "bottom is resolvable: efficientnet_b0 is behind by twice the noise. No significance "
          "test was run, and five folds could not support one (D-027). "
          "About 83 test specimens per fold cannot resolve two or three points. Black Gill is "
          "the weakest class for every backbone without exception -- 75 specimens, the fewest "
          "of any class.")
    print()
    _wrap("Independent agreement worth naming: ShrimpXNet picks ConvNeXt-Tiny as its best of "
          "six backbones. The same architectural conclusion is reached here under a completely "
          "different protocol. Their absolute number is still not comparable to ours.")


# ---------------------------------------------------------------------------------------
# Section 6 -- is it actually edge-deployable?
# ---------------------------------------------------------------------------------------

def section_edge() -> None:
    deployment = _csv("exp04_deployment.csv")
    summary = _csv("exp04_summary.csv")
    if deployment is None or summary is None:
        return
    _source(TABLES / "exp04_deployment.csv",
            note="parameters and on-disk size computed from parameter and buffer bytes; "
                 "latency measured at batch size 1 with warm-up discarded")
    print()

    scores = summary.set_index("backbone")["f1_macro_mean"].to_dict()
    rows = []
    short_role = {
        "inherited baseline architecture": "inherited",
        "standard efficiency baseline": "efficiency",
        "edge-class backbone": "edge-class",
        "conventional reference point": "reference",
        "ShrimpXNet's reported best": "ShrimpXNet best",
    }
    for _, r in deployment.sort_values("size_mb").iterrows():
        rows.append([r["model_name"], f"{r['total_params']:,}", f"{r['size_mb']:.2f}",
                     f"{scores.get(r['model_name'], float('nan')):.4f}",
                     f"{r['latency_ms_mean']:.2f}", short_role.get(r["role"], r["role"])])
    _table(["backbone", "parameters", "size MB", "macro-F1", "latency ms", "role"],
           rows, "<>>>><", gap=1)
    print()

    edge = deployment.loc[deployment["model_name"] == "mobilenetv3_large_100"].iloc[0]
    best = deployment.loc[deployment["model_name"] == "convnext_tiny"].iloc[0]
    ratio = float(best["total_params"]) / float(edge["total_params"])
    cost = scores["convnext_tiny"] - scores["mobilenetv3_large_100"]
    sd = float(summary.loc[summary["backbone"] == "mobilenetv3_large_100", "f1_macro_sd"].iloc[0])
    print(f"  edge pick: mobilenetv3_large_100, {edge['size_mb']:.2f} MB")
    print(f"    {ratio:.1f}x smaller than convnext_tiny, costing {cost:.4f} macro-F1")
    print(f"    against that model's own fold SD of {sd:.4f}")
    print(f"    still {float(edge['total_params']) / 1e6 / FEATHERNETX_PARAMS_M:.1f}x "
          f"FeatherNetX's published {FEATHERNETX_PARAMS_M} M -- no parity claim")
    print()
    _wrap("Latency is printed because it was measured, but it never ranks anything (D-021). "
          "Two runs of the identical code minutes apart varied by up to 40%, and an RTX 5060 "
          "laptop GPU is not an edge device. Parameter count and model size are what transfer, "
          "so they carry the edge claim.")


# ---------------------------------------------------------------------------------------
# Section 7 -- a wrong answer, caught before it was published
# ---------------------------------------------------------------------------------------

def section_convnext() -> None:
    superseded = _csv("exp04_folds_lr3e-4_all.csv")
    if superseded is None:
        return
    rows = superseded[superseded["backbone"] == "convnext_tiny"]
    _source(TABLES / "exp04_folds_lr3e-4_all.csv",
            note="the retracted run: every backbone at lr 3e-4, kept because a failed approach "
                 "is part of the methodology")
    print()

    scores = rows["test_f1_macro"].to_numpy()
    print("  convnext_tiny at lr 3e-4 -- the result that was about to go on a slide")
    print("    per-fold macro-F1: " + "  ".join(f"{s:.4f}" for s in scores))
    print(f"    mean {scores.mean():.4f} +/- {scores.std(ddof=1):.4f}   -- last of five backbones")
    print()
    _wrap("That number would have supported a striking claim: that the architecture ShrimpXNet "
          "reports as its best is the worst under honest evaluation. It would have been false.")
    print()

    histories = sorted((RUNS / "exp04").glob("convnext_tiny__fold*.json"))
    if not histories:
        _skipped(RUNS / "exp04", "run: .venv/Scripts/python.exe scripts/exp04_backbone_comparison.py")
        return
    _source(RUNS / "exp04" / "convnext_tiny__fold*.json",
            note="corrected run at lr 5e-5; the train-F1 column below is derived from the "
                 "stored per-epoch history, not from any results table")
    print()

    table_rows = []
    for path in histories:
        run = json.loads(path.read_text(encoding="utf-8"))
        history = run["history"]
        table_rows.append([
            run["config"]["fold"], f"{run['config']['lr']:g}", len(history),
            f"{history[0]['train_f1']:.3f}", f"{history[-1]['train_f1']:.3f}",
            f"{run['per_class']['macro avg']['f1-score']:.4f},"
        ])
    _table(["fold", "lr", "epochs", "train F1 first", "train F1 last", "test macro-F1"],
           [r[:-1] + [r[-1].rstrip(",")] for r in table_rows], "<>>>>>")
    print()
    _wrap("At 3e-4 three of the five folds never fitted their own TRAINING data -- one went "
          "backwards. A model that generalises badly still fits its training set; one at "
          "near-chance training F1 never learned at all, and the score alone cannot tell those "
          "two apart. Only the curve can. At 5e-5 every fold reaches train F1 above 0.99 and "
          "the model goes from last place to first: 0.3966 to 0.7189. The per-epoch histories "
          "of the retracted run were overwritten and cannot be reprinted here; the fold scores "
          "above are what survives of it (D-020).")


# ---------------------------------------------------------------------------------------
# Section 8 -- Track A: farm-level risk on real survey data
# ---------------------------------------------------------------------------------------

def section_track_a() -> None:
    diagnostics = _csv("exp03_diagnostics.csv")
    candidates = _csv("exp03_step3_candidates.csv")
    skipped = _csv("exp03_pdp_skipped.csv")
    if diagnostics is None or candidates is None:
        return
    _source(TABLES / "exp03_diagnostics.csv",
            note="Just-Pope step 1 on IFPRI 2024 (1,221 Bangladeshi shrimp farms), "
                 "the method of the base paper applied to open data")
    print()

    rows = []
    for _, r in diagnostics.iterrows():
        holds = r["breusch_pagan_p"] < 0.05
        rows.append([r["model"], r["outcome"], f"{r['n']:,}", f"{r['r_squared']:.4f}",
                     f"{r['f_pvalue']:.1e}", f"{r['breusch_pagan_p']:.4f}",
                     "HOLDS" if holds else "NOT SHOWN"])
    _table(["risk model", "outcome", "n", "R^2", "F p", "BP p", "Just-Pope"],
           rows, "<<>>>><")
    print()
    _wrap("Just-Pope only has something to find when the variance depends on the inputs. For "
          "income it does (p = 0.0115). For disease loss that is not shown (p = 0.195): the "
          "test could not reject constant variance, so step 3 had nothing it could be expected "
          "to find (D-026).")
    print()

    _source(TABLES / "exp03_step3_candidates.csv",
            note="step 3: four learners on the log deviance, ranked by repeated-K-fold R^2 -- "
                 "the same rule the experiment used, because a ranking from a single holdout "
                 "at n=463 would be a ranking of luck")
    print()
    rows = []
    for risk_model in ("income", "disease"):
        subset = candidates[candidates["risk_model"] == risk_model]
        for _, r in subset.sort_values("cv_r2_mean", ascending=False).iterrows():
            rows.append([risk_model, r["model"], f"{r['cv_r2_mean']:.4f}",
                         f"{r['holdout_r2']:.4f}",
                         f"{r['holdout_rmse_vs_mean_baseline']:.4f}",
                         "worse than mean"
                         if r["holdout_rmse_vs_mean_baseline"] > 1 else "beats the mean"])
    _table(["risk model", "candidate", "CV R^2", "holdout", "RMSE/base", "verdict"],
           rows, "<<>>><")
    print()

    if skipped is not None and not skipped.empty:
        _source(TABLES / "exp03_pdp_skipped.csv")
        for _, r in skipped.iterrows():
            print(f"    {r['risk_model']:8s} no PDP drawn -- best was {r['best_model']}, "
                  f"holdout R^2 {r['holdout_r2']:.4f}")
        _wrap('reason recorded in the table: "' + str(skipped["reason"].iloc[0]) + '"',
              indent="    ")
        print()
    _wrap("Every candidate is worse than predicting the mean, on both outcomes. That is "
          "reported as the finding rather than tuned away (D-016), and no partial-dependence "
          "plot was drawn, because such a curve looks identical whether the model has skill or "
          "none (D-017). The gate is computed by the script, not applied by hand.")


# ---------------------------------------------------------------------------------------
# Section 9 -- Stage 10: per-backbone learning rate sweep
# ---------------------------------------------------------------------------------------

def section_lr_sweep() -> None:
    sweep = _csv("exp10_lr_sweep_summary.csv")
    if sweep is None:
        return
    _source(TABLES / "exp10_lr_sweep_summary.csv",
            note="Stage 10 systematic LR sweep: 31 runs evaluating learning-rate "
                 "sensitivity across candidate backbones")
    print()

    rows = []
    for _, r in sweep.iterrows():
        sd_str = f"{r['f1_macro_sd']:.4f}" if pd.notna(r["f1_macro_sd"]) else "-"
        bg_str = f"{r['f1_BG_mean']:.4f}" if pd.notna(r["f1_BG_mean"]) else "-"
        rows.append([
            r["backbone"], f"{r['lr']:g}", f"{int(r['folds'])}",
            f"{r['f1_macro_mean']:.4f}", sd_str, bg_str,
            f"{r['train_seconds_mean']:.1f}s"
        ])
    _table(["backbone", "lr", "folds", "macro-F1", "fold SD", "BG F1", "train time"],
           rows, "<<>>>><", gap=1)
    print()
    _wrap("The sweep is partial (R17): convnext_tiny and resnet50 were never swept, and "
          "mobilenetv3_large_100 ran one fold only. It does move the ranking: at lr 1e-3, "
          "rexnet_150 and efficientnet_b0 (rows above) both score above the edge pick "
          "mobilenetv3_large_100 in Result 4. The backbone ranking is therefore provisional, and the edge pick rests "
          "on measured size, not on a settled ordering.")


# ---------------------------------------------------------------------------------------
# Section 10 -- Stage 11: Edge export and INT8 quantization
# ---------------------------------------------------------------------------------------

def section_export() -> None:
    export = _csv("exp12_export.csv")
    if export is None:
        return
    _source(TABLES / "exp12_export.csv",
            note="Stage 11 ONNX export and static INT8 quantization benchmarked on CPU")
    print()

    rows = []
    for _, r in export.iterrows():
        rows.append([
            r["format"], r["precision"], f"{r['file_size_mb']:.2f}",
            f"{r['latency_mean_ms']:.2f}", f"{r['ratio_to_feathernetx_size']:.2f}x"
        ])
    _table(["format", "precision", "size MB", "latency ms", "vs FeatherNetX"],
           rows, "<<>>>", gap=1)
    print()
    int8_row = export[export["precision"].str.contains("INT8", na=False)]
    if not int8_row.empty:
        int8_size = float(int8_row["file_size_mb"].iloc[0])
        fp32_row = export[export["precision"].astype(str).str.strip() == "FP32"]
        fp32_size = float(fp32_row["file_size_mb"].iloc[0]) if not fp32_row.empty else float("nan")
        print(f"  INT8 edge model: {int8_size:.2f} MB "
              f"({fp32_size / int8_size:.2f}x compression from FP32 {fp32_size:.2f} MB)")
        if "holdout_macro_f1" in export.columns:
            f1_int8 = float(int8_row["holdout_macro_f1"].iloc[0])
            f1_fp32 = float(fp32_row["holdout_macro_f1"].iloc[0]) if not fp32_row.empty else float("nan")
            print(f"  Held-out macro-F1: {f1_fp32:.4f} FP32 vs {f1_int8:.4f} INT8 "
                  f"(costs {f1_fp32 - f1_int8:.4f})")
    print()
    _wrap("Static INT8 quantisation with calibration data compresses the deployable artefact "
          "below the 5 MB edge ceiling, at a measured and stated cost in macro-F1. Dynamic "
          "quantisation was tried first and rejected: it collapsed to chance (macro-F1 0.2380) "
          "and ran 24x slower, which is the documented behaviour of per-tensor dynamic "
          "quantisation on depthwise-separable backbones. Model size carries the edge claim "
          "(D-021); latency here is a laptop CPU, not an edge device.")


# ---------------------------------------------------------------------------------------
# Section 11 -- Stage 11: Live end-to-end farm risk assessment
# ---------------------------------------------------------------------------------------

def section_farm_risk() -> None:
    cli_path = PROJECT_ROOT / "scripts" / "predict_farm.py"
    _source(cli_path, note="Stage 11 decision-level multimodal pipeline: edge posteriors "
                           "aggregated to specimens, quantified via SLD-EM, mapped to risk")
    print()

    from shrimp_risk.integration import quantification as Q

    # These five posterior vectors are TYPED IN, not inferred.
    #
    # The block was labelled LIVE and headed "LIVE INFERENCE & QUANTIFICATION DEMO", while
    # loading no model and reading no image. That is the exact failure this script's own header
    # names -- "the dishonest answer: a script that prints hard-coded numbers and calls it a run"
    # -- and it breaks `D-022`, which requires every block to say which numbers were computed
    # then and which were replayed.
    #
    # The quantification below IS computed live from these inputs: SLD-EM really iterates and
    # really converges, and that is what this block demonstrates. What is illustrative is the
    # INPUT. So the block is labelled ILLUSTRATIVE and says so on screen. For a genuine
    # end-to-end run against the trained ONNX graph, use `predict_farm.py` directly, which is
    # the live demonstration Stage 11 actually ships.
    probs = np.array([
        [0.10, 0.65, 0.15, 0.10],
        [0.05, 0.05, 0.85, 0.05],
        [0.08, 0.72, 0.10, 0.10],
        [0.70, 0.10, 0.10, 0.10],
        [0.05, 0.05, 0.80, 0.10],
    ], dtype=np.float64)

    train_prior = np.array([0.172324, 0.191471, 0.350740, 0.285466], dtype=np.float64)
    pos_indices = Q.positive_indices()
    est_cc = Q.classify_and_count(probs, pos_indices)
    est_em = Q.saerens_em(probs, train_prior, pos_indices, max_iter=100)

    wssv_cc = float(est_cc.prevalence)
    wssv_em = float(est_em.prevalence)
    iters = est_em.iterations
    conv = est_em.converged

    area_acres = 2.5
    base_rev = 250000.0 * area_acres
    beta = -0.2017
    pct_loss_em = (1.0 - np.exp(beta * wssv_em)) * 100.0
    bdt_loss_em = base_rev * (pct_loss_em / 100.0)

    print("  WORKED EXAMPLE -- the five posteriors below are typed in, not inferred.")
    print("  The SLD-EM quantification IS computed live from them. For a real")
    print("  end-to-end run on the trained graph, use scripts/predict_farm.py.")
    print()
    print("  QUANTIFICATION ON THE WORKED EXAMPLE (n = 5 specimens, 2.50 acres):")
    print(f"    Naive Count (CC) White Spot prevalence : {wssv_cc * 100:.1f}%")
    print(f"    SLD-EM Prior-Corrected prevalence      : {wssv_em * 100:.1f}% "
          f"({iters} iters, conv={conv})")
    print(f"    Assumed baseline farm revenue          : {base_rev:,.2f} BDT  [ASSUMPTION]")
    print(f"    Expected revenue loss (Just-Pope S2)   : {bdt_loss_em:,.2f} BDT "
          f"({pct_loss_em:.1f}%)")
    print()
    _wrap("The multimodal bridge aggregates edge vision predictions to specimen level (D-007), "
          "corrects test-set prior distribution shift via SLD-EM without temperature distortion "
          "(D-041), and maps prevalence to farm-level revenue risk via the Just-Pope S2 "
          "specification (beta = -0.2017, p = 0.0685).")


# ---------------------------------------------------------------------------------------
# The running order. Sections are listed in the order a review should hear them: the
# structural fact, the exposure it creates, the cost of ignoring it, then everything that
# was built on top of the corrected protocol.
# ---------------------------------------------------------------------------------------

SECTIONS: list[tuple[str, str, str, object]] = [
    ("structure", "LIVE",
     "The structure the whole project rests on", section_structure),
    ("leakage", "LIVE",
     "Result 1 -- what a conventional split contaminates", section_leakage),
    ("effect", "RECORDED",
     "Result 2 -- what that contamination costs, in macro-F1", section_effect),
    ("matrix", "LIVE",
     "Result 3 -- metrics recomputed from stored predictions", section_matrix),
    ("backbones", "RECORDED",
     "Result 4 -- which backbone, once the protocol is honest", section_backbones),
    ("edge", "RECORDED",
     "Result 5 -- is it actually edge-deployable?", section_edge),
    ("convnext", "RECORDED",
     "Result 6 -- a wrong answer, caught before publishing", section_convnext),
    ("track_a", "RECORDED",
     "Result 7 -- farm-level risk on real survey data", section_track_a),
    ("lr_sweep", "RECORDED",
     "Result 8 -- Stage 10 per-backbone learning-rate sweep", section_lr_sweep),
    ("export", "RECORDED",
     "Result 9 -- Stage 11 edge export and INT8 quantization", section_export),
    ("farm_risk", "ILLUSTRATIVE",
     "Result 10 -- Stage 11 decision-level bridge on a worked example", section_farm_risk),
]


def _header() -> None:
    _rule()
    _wrap(TITLE, indent=" ")
    print(f" {STUDENT}")
    _rule()
    print(" Every block below is labelled with where its numbers come from:")
    print()
    print("   LIVE      computed in this process, now, from the image manifest or")
    print("             from the posteriors stored at run time. No table is read.")
    print("   RECORDED  read from the artefact the experiment wrote. These are")
    print("             training outcomes: re-running them costs 20-45 minutes on")
    print("             a GPU, so they are replayed and every path is printed.")
    print()
    environment = environment_fingerprint()
    fields = [f"python {environment.get('python', '?')}", f"torch {environment.get('torch', '?')}"]
    if environment.get("gpu"):
        fields.append(environment["gpu"])
    if environment.get("cuda"):
        fields.append(f"cuda {environment['cuda']}")
    _wrap(" | ".join(fields), indent=" ")
    _rule()


def _section_banner(index: int, total: int, tag: str, title: str) -> None:
    print()
    _rule()
    prefix = f" [{index}/{total}] {tag:8s} | "
    if len(prefix) + len(title) <= WIDTH:
        print(prefix + title)
    else:                                    # never let a banner run off a projected slide
        print(prefix.rstrip())
        _wrap(title, indent="   ")
    _rule("-")


def _closing(elapsed: float) -> None:
    print()
    _rule()
    print(" THE NUMBERS TO REMEMBER")
    _rule("-")

    exp01 = _csv("exp01_leakage_summary.csv")
    exp02 = _csv("exp02_summary.csv")
    exp04 = _csv("exp04_summary.csv")
    exp03 = _csv("exp03_diagnostics.csv")
    exp10 = _csv("exp10_lr_sweep_summary.csv")
    exp12 = _csv("exp12_export.csv")

    if IMAGE_MANIFEST.exists():
        raw = S.load_raw_manifest(IMAGE_MANIFEST)
        print(f"  dataset          {len(raw):,} images of {raw['specimen_id'].nunique():,} "
              f"individual shrimp")
    if exp01 is not None:
        image_level = exp01.loc[exp01["protocol"] == "image_level", "pct_test_leaked_mean"].iloc[0]
        specimen_level = exp01.loc[exp01["protocol"] == "specimen_level",
                                   "pct_test_leaked_mean"].iloc[0]
        print(f"  leakage exposure {image_level:.2f}% image-level vs {specimen_level:.2f}% "
              f"specimen-level")
    if exp02 is not None:
        b = exp02.loc[exp02["condition"] == "B_image_level", "test_f1_macro_mean"].iloc[0]
        c = exp02.loc[exp02["condition"] == "C_specimen_level", "test_f1_macro_mean"].iloc[0]
        print(f"  headline result  {b:.4f} image-level vs {c:.4f} specimen-level "
              f"-> gap {b - c:.4f}")
    if exp04 is not None:
        best = exp04.sort_values("f1_macro_mean", ascending=False).iloc[0]
        edge = exp04.loc[exp04["backbone"] == "mobilenetv3_large_100"].iloc[0]
        print(f"  best backbone    {best['backbone']} {best['f1_macro_mean']:.4f} "
              f"+/- {best['f1_macro_sd']:.4f}")
        print(f"  edge pick        {edge['backbone']} {edge['f1_macro_mean']:.4f} "
              f"at {edge['size_mb']:.2f} MB")
    if exp03 is not None:
        income = exp03.loc[exp03["model"] == "income"].iloc[0]
        disease = exp03.loc[exp03["model"] == "disease"].iloc[0]
        print(f"  Track A income   R^2 {income['r_squared']:.4f}, n={income['n']:,}, "
              f"BP p {income['breusch_pagan_p']:.4f} -- null rejected")
        print(f"  Track A disease  R^2 {disease['r_squared']:.4f}, n={disease['n']:,}, "
              f"BP p {disease['breusch_pagan_p']:.3f} -- null NOT rejected")
    if exp10 is not None and exp04 is not None:
        above = exp10[(exp10["lr"] == 1e-3) & (exp10["f1_macro_mean"] > edge["f1_macro_mean"])]
        names = " and ".join(f"{r['backbone'].split('_')[0]} ({r['f1_macro_mean']:.4f})"
                             for _, r in above.sort_values("f1_macro_mean", ascending=False).iterrows())
        print("  Stage 10 LR      sweep is PARTIAL; at lr 1e-3")
        _wrap(f"{names} score above the edge pick's {edge['f1_macro_mean']:.4f}", indent=" " * 19)
    if exp12 is not None:
        int8_row = exp12[exp12["precision"].str.contains("INT8", na=False)]
        if not int8_row.empty:
            int8_sz = int8_row["file_size_mb"].iloc[0]
            ratio = float(int8_row["ratio_to_feathernetx_size"].iloc[0])
            msg = f"  Stage 11 INT8    {int8_sz:.2f} MB ({ratio:.2f}x FeatherNetX)"
            if "holdout_macro_f1" in exp12.columns:
                msg += f", macro-F1 {float(int8_row['holdout_macro_f1'].iloc[0]):.4f}"
            print(msg)

    print()
    if _SKIPPED:
        print(f"  {len(_SKIPPED)} input(s) were missing, so those blocks were skipped:")
        for path in dict.fromkeys(_SKIPPED):
            print(f"    - {path}")
        print()
    _wrap("Every number in the tables and in this summary is read from the file named above it. "
          "The commentary under the tables repeats some of those numbers in words, and two "
          "inputs are typed in: one published comparison figure and the worked-example inputs "
          "in the ILLUSTRATIVE block, both of which say so where they appear. The "
          "test suite that guards these artefacts runs in about 100 seconds:")
    print()
    print("    .venv/Scripts/python.exe -m pytest")
    print()
    print(f"  {elapsed:.1f} s elapsed.")
    _rule()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print every result this project has produced, labelled by provenance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true",
                        help="list the section names and exit")
    parser.add_argument("--only", nargs="+", metavar="SECTION", default=None,
                        help="run only these sections, in the order given")
    parser.add_argument("--pause", action="store_true",
                        help="wait for Enter between sections, for a live demonstration")
    args = parser.parse_args(argv)

    if args.list:
        print("sections, in running order:")
        for name, tag, title, _ in SECTIONS:
            print(f"  {name:11s} {tag:8s} {title}")
        return 0

    chosen = SECTIONS
    if args.only:
        known = {name for name, _, _, _ in SECTIONS}
        unknown = [name for name in args.only if name not in known]
        if unknown:
            print(f"unknown section(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"choose from: {', '.join(sorted(known))}", file=sys.stderr)
            return 2
        by_name = {name: entry for entry in SECTIONS for name in [entry[0]]}
        chosen = [by_name[name] for name in args.only]

    started = time.perf_counter()
    _header()
    for index, (_, tag, title, runner) in enumerate(chosen, start=1):
        _section_banner(index, len(chosen), tag, title)
        runner()
        if args.pause and index < len(chosen):
            try:
                input("\n  [Enter] for the next section ")
            except EOFError:
                pass
    _closing(time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
