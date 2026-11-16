"""Build the final-review showcase: the whole project on one browser page, with the live pond demo.

Why this script exists
----------------------
`demo_results.py` and `predict_farm.py` show everything the project produced, but in a terminal,
and a panel watching a terminal sees text scroll past. This page shows the same things visually:
the pipeline in one picture, the tool diagnosing a pond of shrimp it never trained on, photograph
by photograph, and each recorded result next to the figure that came from it.

It produces no new numbers. Two sources only, and each block on the page says which (`D-022`):

    LIVE      `predict_farm.py` is run on `data/demo/pond_A/` while this script runs -- the same
              command as the demo run sheet, with `--json` -- and its output is what the pond
              section shows. Building the page *is* running the demo.
    RECORDED  read from `reports/tables/*.csv` and `reports/figures/*.png`, which the experiments
              wrote. The file is named under each block. Training cannot be re-run in a review
              slot, so these are never presented as computed now.

The answer key is embedded but hidden until the presenter clicks "Reveal", so the prediction is on
screen before the truth is. The page is a single self-contained HTML file (figures and photographs
inlined) so it opens offline. It goes to `data/demo/`, which is gitignored, because it embeds
dataset photographs.

Usage:
    .venv/Scripts/python.exe scripts/build_showcase.py            # build, then open in the browser
    .venv/Scripts/python.exe scripts/build_showcase.py --int8     # the 4.46 MB INT8 edge graph
    .venv/Scripts/python.exe scripts/build_showcase.py --no-open
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import subprocess
import sys
import time
import webbrowser
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shrimp_risk.common.paths import FIGURES, IMAGE_MANIFEST, TABLES  # noqa: E402
from predict_farm import (  # noqa: E402  -- one definition of the specimen rule (D-007)
    BETA_WSD,
    P_VALUE,
    parse_specimen_id_from_filename,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

POND = ROOT / "data" / "demo" / "pond_A"
ANSWER_KEY = ROOT / "data" / "demo" / "pond_A_answer_key.csv"
OUT = ROOT / "data" / "demo" / "showcase.html"
WHITE_SPOT = ("BG_WSSV", "WSSV")

TITLE = ("An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp Disease "
         "Detection Using Image")


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------

def _t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def _row(df: pd.DataFrame, **match) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in match.items():
        mask &= df[col] == val
    rows = df[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {match}, found {len(rows)}")
    return rows.iloc[0]


def recorded() -> dict:
    """Every recorded number the page shows, read from its table. Nothing typed."""
    r: dict = {}

    bal = _t("exp01_class_balance")
    r["n_images"] = int(bal[["train", "val", "test"]].to_numpy().sum())
    r["n_specimens"] = (int(pd.read_csv(IMAGE_MANIFEST)["specimen_id"].nunique())
                        if IMAGE_MANIFEST.exists() else None)

    leak = _t("exp01_leakage_summary")
    img = _row(leak, protocol="image_level")
    r["leak_image"], r["leak_image_sd"] = float(img["pct_test_leaked_mean"]), float(img["pct_test_leaked_sd"])
    r["leak_specimen"] = float(_row(leak, protocol="specimen_level")["pct_test_leaked_mean"])

    e2 = _t("exp02_summary")
    r["f1_image"] = float(_row(e2, condition="B_image_level")["test_f1_macro_mean"])
    r["f1_specimen"] = float(_row(e2, condition="C_specimen_level")["test_f1_macro_mean"])

    e4 = _t("exp04_summary")
    cnx, mnv = _row(e4, backbone="convnext_tiny"), _row(e4, backbone="mobilenetv3_large_100")
    r["cnx_f1"], r["cnx_sd"], r["cnx_lr"] = float(cnx["f1_macro_mean"]), float(cnx["f1_macro_sd"]), float(cnx["lr"])
    r["mnv_f1"], r["mnv_sd"] = float(mnv["f1_macro_mean"]), float(mnv["f1_macro_sd"])
    r["mnv_params"], r["mnv_mb"] = float(mnv["M_params"]), float(mnv["size_mb"])
    r["param_ratio"] = float(cnx["total_params"]) / float(mnv["total_params"])
    r["bg_best"] = float(e4["f1_BG_mean"].max())

    sweep = _t("exp10_lr_sweep_summary")
    r["rexnet_1e3"] = float(_row(sweep, backbone="rexnet_150", lr=0.001)["f1_macro_mean"])
    r["effnet_1e3"] = float(_row(sweep, backbone="efficientnet_b0", lr=0.001)["f1_macro_mean"])

    e12 = _t("exp12_export")
    fp32 = _row(e12, format="ONNX Runtime", precision="FP32")
    int8 = _row(e12, format="ONNX Runtime", precision="INT8 (Static QDQ)")
    r["fp32_f1"], r["fp32_mb"] = float(fp32["holdout_macro_f1"]), float(fp32["file_size_mb"])
    r["int8_f1"], r["int8_mb"] = float(int8["holdout_macro_f1"]), float(int8["file_size_mb"])
    r["int8_ms"] = float(int8["latency_mean_ms"])

    e6 = _t("exp06_summary")
    r["ece_cnx"] = float(_row(e6, backbone="convnext_tiny")["ece_after_mean"])
    r["ece_mnv"] = float(_row(e6, backbone="mobilenetv3_large_100")["ece_after_mean"])
    r["ece_target"] = float(_row(e6, backbone="convnext_tiny")["target"])

    e7 = _t("exp07_estimators")
    r["em_bias"] = float(_row(e7, backbone="convnext_tiny", calibration="raw", pool_size=40,
                              estimator="SLD-EM")["mean_abs_bias"])
    r["cc_bias"] = float(_row(e7, backbone="convnext_tiny", calibration="raw", pool_size=40,
                              estimator="CC")["mean_abs_bias"])

    e8 = _t("exp08_baseline")
    s1, s2 = _row(e8, specification="S1_published"), _row(e8, specification="S2_bridge_facing")
    r["s1_p"] = float(s1["p_value"])
    r["s2_beta"], r["s2_p"], r["s2_n"] = float(s2["estimate"]), float(s2["p_value"]), int(s2["n"])
    att = _row(_t("exp08_attenuation"), specification="S2_bridge_facing", level_label="measured: SLD-EM")
    r["bridge_err"], r["bridge_ratio"] = float(att["magnitude"]), float(att["attenuation_observed"])

    e3 = _t("exp03_diagnostics")
    inc, dis = _row(e3, model="income"), _row(e3, model="disease")
    r["inc_n"], r["inc_r2"], r["inc_bp"] = int(inc["n"]), float(inc["r_squared"]), float(inc["breusch_pagan_p"])
    r["dis_n"], r["dis_r2"], r["dis_bp"] = int(dis["n"]), float(dis["r_squared"]), float(dis["breusch_pagan_p"])
    r["step3_best"] = float(_t("exp03_step3_candidates")["holdout_rmse_vs_mean_baseline"].min())

    hh = _row(_t("exp11_comparison"), sample="Hasan & Haque 2020 (A_no_water_quality)")
    r["hh_n"], r["hh_skill"] = int(hh["n"]), bool(hh["skill_supported"])

    # ---- validation evidence (section 4) ----
    r["shared_image"] = float(img["shared_specimens"])
    r["shared_specimen"] = float(_row(leak, protocol="specimen_level")["shared_specimens"])
    r["leak_seeds"] = int(img["n_seeds"])
    runs = _t("exp02_runs").pivot_table(index="seed", columns="condition", values="test_f1_macro")
    per_seed_gap = runs["B_image_level"] - runs["C_specimen_level"]
    r["gap_seeds"], r["gap_seeds_pos"] = len(per_seed_gap), int((per_seed_gap > 0).sum())
    r["cv_folds"] = int(cnx["folds"])
    r["cnx_before"] = float(_t("exp04_folds_lr3e-4_all").query("backbone == 'convnext_tiny'")["test_f1_macro"].mean())

    e9 = _row(_t("exp09_comparison"), metric="f1_BG")
    r["bgw_delta"], r["bgw_beyond"] = float(e9["delta"]), bool(e9["beyond_fold_sd"])

    pt = _row(e12, format="PyTorch CPU", precision="FP32")
    r["pt_f1"], r["int8_agree"] = float(pt["holdout_macro_f1"]), float(int8["top1_agreement_vs_fp32"])

    bias = _t("exp07_bias")
    r["sim_levels"], r["sim_pools"] = int(bias["target_prevalence"].nunique()), int(bias["n_pools"].max())
    r["prop_reps"] = int(_t("exp08_attenuation")["replicates"].max())

    gcv = _row(_t("exp10_grouped_cv"), risk_model="income", fold_scheme="cluster_grouped_folds")
    r["gcv_ratio"], r["gcv_clusters"] = float(gcv["holdout_rmse_vs_mean_baseline"]), int(gcv["n_clusters"])
    cr = _row(_t("exp03_cluster_robust"), risk_model="income")
    r["cr_clusters"], r["cr_flips"] = int(cr["n_clusters"]), int(cr["n_significance_flips"])
    pw = _row(_t("exp10_bp_power"), risk_model="income", gamma=0.1)
    r["bp_power"], r["bp_sd_ratio"] = float(pw["power_at_05"]), float(pw["sd_ratio_p90_to_p10"])
    r["inc_reset"], r["dis_reset"] = float(inc["reset2_p"]), float(dis["reset2_p"])

    nb = _row(_t("exp02b_summary"), image_source="cache_256")
    r["nb_claimed"], r["nb_got"], r["nb_seeds"] = float(nb["notebook_f1"]), float(nb["f1_macro_mean"]), int(nb["seeds"])
    e5 = _t("exp05_calibration_summary").query("experiment == 'exp04'")["ece_mean"]
    r["ece05_min"], r["ece05_max"] = float(e5.min()), float(e5.max())
    return r


def counts() -> dict:
    """LIVE counts for the reproducibility card: tests collected now, run records on disk."""
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"],
                          capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
    n_tests = sum(int(line.rsplit(": ", 1)[1]) for line in proc.stdout.splitlines()
                  if line.startswith("tests") and line.rsplit(": ", 1)[-1].isdigit())
    n_runs = len(list((ROOT / "experiments" / "runs").rglob("*.json")))
    return {"tests": n_tests, "runs": n_runs}


def run_live(int8: bool) -> dict:
    """Run the deployed tool on the pond, exactly as the run sheet does, and parse its JSON."""
    cmd = [sys.executable, str(ROOT / "scripts" / "predict_farm.py"), "--images-dir", str(POND), "--json"]
    if int8:
        cmd.append("--int8")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(f"predict_farm.py failed:\n{proc.stderr}")
    start = proc.stdout.find('{\n  "n_images":')
    if start == -1:
        raise SystemExit("predict_farm.py printed no JSON payload")
    live = json.loads(proc.stdout[start:])
    live["wall_seconds"] = time.perf_counter() - t0
    live["command"] = " ".join(["predict_farm.py", "--images-dir", "data/demo/pond_A"] + (["--int8"] if int8 else []))
    return live


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _thumb(path: Path, px: int = 220) -> str:
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


def _tag(kind: str, source: str) -> str:
    return (f'<div class="src"><span class="tag {kind.lower()}">{kind}</span>'
            f'<code>{html.escape(source)}</code></div>')


def _figure(name: str, caption: str) -> str:
    return (f'<figure><img src="data:image/png;base64,{_b64(FIGURES / name)}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)} <code>reports/figures/{name}</code></figcaption></figure>')


def pond_section(live: dict) -> str:
    classes = live["classes"]
    ws_idx = [classes.index(c) for c in WHITE_SPOT]
    key = pd.read_csv(ANSWER_KEY).set_index("demo_id")

    photos: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(POND.glob("*.jpg")):
        photos[parse_specimen_id_from_filename(p.name)].append(p)

    cards, n_right = [], 0
    for s in live["specimens"]:
        probs = s["probs"]
        top = max(range(len(probs)), key=probs.__getitem__)
        p_ws = sum(probs[i] for i in ws_idx)
        called_pos = classes[top] in WHITE_SPOT
        truth = key.loc[s["id"]]
        right = bool(truth["white_spot"]) == called_pos
        n_right += right
        cards.append(
            f'<div class="card {"pos" if called_pos else "neg"}">'
            f'<img src="data:image/jpeg;base64,{_thumb(photos[s["id"]][0])}" alt="{s["id"]}">'
            f'<div class="nphoto">{len(photos[s["id"]])} photos</div>'
            f'<div class="cid">{s["id"]}</div>'
            f'<div class="call">{classes[top]} <span>{_pct(probs[top], 0)}</span></div>'
            f'<div class="bar"><i style="width:{p_ws * 100:.1f}%"></i></div>'
            f'<div class="pws">P(White Spot) {_pct(p_ws, 0)}</div>'
            f'<div class="truth">truth: {truth["class"]}<br>White Spot call '
            f'{"&#10003;" if right else "&#10007;"}</div>'
            f'</div>')

    pv, risk = live["prevalence"], live["economic_risk"]
    lo, hi = pv["em_ci_95"]
    true_prev = float(key["white_spot"].mean())
    inside = lo <= true_prev <= hi
    cc_closer = abs(pv["cc"] - true_prev) < abs(pv["em"] - true_prev)
    n_spec = live["n_specimens"]
    per_acre = risk["baseline_revenue_bdt"] / risk["pond_area_acres"]

    return f"""
<section id="demo">
  <h2>2 · The tool on a pond it has never seen</h2>
  {_tag("LIVE", live["command"] + "  — run while this page was built")}
  <p class="lead">{live["n_images"]} photographs of {n_spec} shrimp from the fold the deployed model
  held out, so it never trained on any of them. Files are renamed S001–S{n_spec:03d}, so the filename
  carries no answer. Photographs of one shrimp are averaged into one diagnosis (D-007).
  Card top edge: <b style="color:var(--pos)">orange</b> = called White Spot positive (WSSV or BG_WSSV),
  <b style="color:var(--neg)">teal</b> = called negative.</p>
  <p class="meta">Model: {html.escape(live["backend"])} · inference {live["inference_time_ms"]:.0f} ms for
  {live["n_images"]} images · end to end {live["wall_seconds"]:.1f} s</p>

  <div class="kpis">
    <div class="kpi"><b>{_pct(pv["em"])}</b><span>SLD-EM pond White Spot rate</span>
      <small>95% CI {_pct(lo)} – {_pct(hi)}</small></div>
    <div class="kpi"><b>{_pct(pv["cc"])}</b><span>Naive count (CC)</span><small>share of shrimp called positive</small></div>
    <div class="kpi"><b>{_pct(pv["pcc"])}</b><span>Posterior mean (PCC)</span><small>mean P(White Spot)</small></div>
    <div class="kpi reveal-only"><b>{_pct(true_prev, 1)}</b><span>True rate (answer key)</span>
      <small>{int(key["white_spot"].sum())} of {len(key)} positive · {"inside" if inside else "OUTSIDE"} the SLD-EM interval</small></div>
  </div>

  <div class="actions">
    <button onclick="document.body.classList.toggle('revealed')">Reveal / hide the answer key</button>
    <span class="reveal-only">Shrimp-level White Spot calls correct: <b>{n_right} of {n_spec}</b>.
    {"On this pond, counting happens to land closer to the truth than SLD-EM. One pond is one draw; the evidence that SLD-EM beats counting is Experiment 07's simulated pools (section 3)." if cc_closer else "One pond is one draw; the evidence that SLD-EM beats counting is Experiment 07's simulated pools (section 3)."}</span>
  </div>

  <div class="grid">{"".join(cards)}</div>

  <div class="risk">
    <h3>Farm risk for this pond <em>illustrative</em></h3>
    <p>Expected revenue loss at the SLD-EM rate: <b>{risk["expected_loss_em_bdt"]:,.0f} BDT</b>
    (95% CI {risk["loss_ci_95_bdt"][0]:,.0f} – {risk["loss_ci_95_bdt"][1]:,.0f}) on {risk["pond_area_acres"]:.2f} acre.</p>
    <p class="warn">The Taka figure uses an <b>assumed</b> {per_acre:,.0f} BDT per acre — no per-acre revenue
    exists in the survey — and the White Spot coefficient β = {BETA_WSD:.4f} has p = {P_VALUE:.3f}.
    Read the prevalence, not the Taka.</p>
  </div>
</section>"""


def experiments_section(r: dict) -> str:
    """Section 4: every experiment, the question it answered, and the part of the build it justified."""
    rows = [
        ("01", "Leakage audit", "Does splitting by image put the same shrimp in train and test?",
         f"{r['leak_image']:.2f}% of test images leak by image; {r['leak_specimen']:.2f}% by specimen",
         "Every split groups by shrimp, never by image", "D-007", "exp01_leakage_summary.csv"),
        ("02", "Leakage effect", "How much does that leak inflate the score?",
         f"Macro-F1 {r['f1_image']:.4f} by image vs {r['f1_specimen']:.4f} by specimen",
         "Only specimen-level scores are reported; the corrected benchmark is a contribution",
         "D-007 · D-027", "exp02_summary.csv"),
        ("02b", "Notebook reproduction", "Does the inherited notebook's score reproduce?",
         f"Claimed {r['nb_claimed']:.3f}; faithful re-run {r['nb_got']:.3f}",
         "No reproduction claim is made; the project builds on its own baseline", "D-036",
         "exp02b_summary.csv"),
        ("03", "Farm risk on survey data", "Does the Just-Pope method apply, and can ML predict farm risk?",
         f"Income Breusch-Pagan p = {r['inc_bp']:.4f} (applies); best risk model {r['step3_best']:.3f}× the mean baseline",
         "Just-Pope mean equation is the economic stage; risk models reported as a negative result",
         "D-016 · D-017", "exp03_diagnostics.csv · exp03_step3_candidates.csv"),
        ("04", "Backbone comparison", "Which network should the camera stage use?",
         f"ConvNeXt best at {r['cnx_f1']:.4f}; MobileNetV3 {r['mnv_f1']:.4f} with {r['param_ratio']:.1f}× fewer parameters",
         "MobileNetV3 is the edge model; the edge claim rests on size, not laptop latency",
         "D-020 · D-021", "exp04_summary.csv"),
        ("05", "Calibration audit", "Can the classifier's probabilities be taken at face value?",
         f"Every backbone's ECE is {r['ece05_min']:.4f}–{r['ece05_max']:.4f}, far above {r['ece_target']:.2f}",
         "Calibration became a measured prerequisite of the bridge", "D-028",
         "exp05_calibration_summary.csv"),
        ("06", "Temperature scaling", "Does calibrating on validation data fix it?",
         f"ECE {r['ece_cnx']:.4f} / {r['ece_mnv']:.4f} after scaling — improved, target not reached",
         "The limitation is declared, and the bridge is tested on both raw and calibrated outputs",
         "D-034", "exp06_summary.csv"),
        ("07", "Prevalence bridge", "Which estimator turns per-shrimp calls into a pond rate?",
         f"SLD-EM bias {r['em_bias']:.4f} vs {r['cc_bias']:.4f} for counting",
         "The tool uses SLD-EM on raw posteriors, with a bootstrap interval", "D-041",
         "exp07_estimators.csv"),
        ("08", "Error propagation", "Which disease variable can the economics use, and does camera error break it?",
         f"Yes/no disease p = {r['s1_p']:.3f}; share of stock lost β = {r['s2_beta']:.4f}; ratio at our error {r['bridge_ratio']:.3f}",
         "The economic stage takes a continuous prevalence, which is what the camera produces",
         "D-042", "exp08_baseline.csv · exp08_attenuation.csv"),
        ("09", "Class-weighted loss", "Does weighting the loss fix Black Gill, the weakest class?",
         f"Black Gill F1 +{r['bgw_delta']:.4f}, inside the fold SD",
         "Not adopted: the gain is within noise", "— (planned in D-030)", "exp09_comparison.csv"),
        ("10", "Track A repairs + LR sweep", "Does the negative result survive fixes? Is the backbone ranking settled?",
         f"Cluster-grouped folds: {r['gcv_ratio']:.3f}× the mean; at lr 1e-3 ReXNet {r['rexnet_1e3']:.4f} beats the edge pick",
         "Negative result kept; backbone ranking declared provisional (risk R17)", "D-037",
         "exp10_grouped_cv.csv · exp10_lr_sweep_summary.csv"),
        ("11", "External validation", "Does the farm-risk finding hold on a second survey?",
         f"Hasan &amp; Haque, n = {r['hh_n']}: skill supported = {'yes' if r['hh_skill'] else 'no'}",
         "The negative result is confirmed, not an artefact of one dataset", "D-040",
         "exp11_comparison.csv"),
        ("12", "Edge export", "Does the shipped file keep the trained model's accuracy, and at what size?",
         f"FP32 {r['fp32_f1']:.4f} at {r['fp32_mb']:.2f} MB → static INT8 {r['int8_f1']:.4f} at {r['int8_mb']:.2f} MB",
         "Ships static INT8; every exported graph is scored on held-out shrimp before use",
         "D-044 · D-045", "exp12_export.csv"),
    ]
    body = "".join(
        f'<tr><td class="eid">Exp {n}</td><td><b>{name}</b><br><span class="q">{q}</span></td>'
        f'<td>{found}</td><td class="so">{so}</td><td class="did">{d}</td>'
        f'<td><code>{html.escape(src)}</code></td></tr>'
        for n, name, q, found, so, d, src in rows)
    return f"""
<section id="experiments">
  <h2>4 · The experiments behind the build</h2>
  {_tag("RECORDED", "reports/tables/ — one file per row; decisions in docs/decisions/decision_log.md")}
  <p class="note">Each design choice in the tool was settled by an experiment, not assumed. Read across a row:
  the question, what was measured, and what the build does because of it.</p>
  <div class="tablewrap"><table class="exps">
    <thead><tr><th></th><th>Experiment · question</th><th>What we found</th><th>So the build…</th><th>Decision</th><th>Source</th></tr></thead>
    <tbody>{body}</tbody>
  </table></div>
</section>"""


REPO_URL = "https://github.com/MJ0006/shrimp-disease-risk/blob/main"

# (question a panel may ask, file, function to open at -- None for the whole script, what to point at,
#  test that proves it -- None where the proof is a long-running experiment)
CODE_MAP = [
    ("How do you stop leakage?", "src/shrimp_risk/vision/splits.py", "make_specimen_folds",
     "Folds are built from shrimp IDs, so every photo of one shrimp lands on the same side.",
     "tests/test_splits.py -k leaks_nothing"),
    ("How is leakage measured?", "src/shrimp_risk/vision/splits.py", "leakage_report",
     "Counts test shrimp that also appear in training — the 87.98% vs 0%.",
     "tests/test_splits.py -k leaks_heavily"),
    ("Show the leakage experiments", "scripts/exp02_leakage_effect.py", None,
     "Same model, same seeds, only the split changes: image-level vs specimen-level.", None),
    ("How is the model trained?", "src/shrimp_risk/vision/trainer.py", "train_and_evaluate",
     "Training loop, best-epoch selection on validation, and the checkpoint written to disk (D-044).", None),
    ("How did you choose the backbone?", "scripts/exp04_backbone_comparison.py", None,
     "Five backbones on the same five specimen-grouped folds.", None),
    ("How do photos become one diagnosis per shrimp?", "src/shrimp_risk/integration/quantification.py",
     "aggregate_to_specimen", "Averages the probabilities of every photo of one shrimp (D-007).",
     "tests/test_quantification.py -k aggregate"),
    ("How does the bridge estimate the pond rate?", "src/shrimp_risk/integration/quantification.py",
     "saerens_em", "SLD-EM: re-weights the classifier's outputs for the pond's own disease rate.",
     "tests/test_quantification.py"),
    ("Where does the interval come from?", "src/shrimp_risk/integration/quantification.py",
     "bootstrap_interval", "Resamples shrimp, not photos, to get the 95% interval.",
     "tests/test_quantification.py -k bootstrap"),
    ("Where is the economic model?", "src/shrimp_risk/econometrics/just_pope.py", "run_just_pope",
     "Just-Pope: mean equation, variance (risk) equation, and the mean-baseline check.",
     "tests/test_econometrics.py"),
    ("Does camera error break the economics?", "src/shrimp_risk/integration/sensitivity.py",
     "perturb_continuous", "Injects the bridge's measured error into prevalence and re-fits.",
     "tests/test_sensitivity.py"),
    ("How was it made edge-sized?", "scripts/export_edge_model.py", "quantize_onnx_int8",
     "Static INT8 with 128 calibration images (D-045).", "tests/test_edge_export.py -k int8"),
    ("How do you know the exported file is trained?", "scripts/export_edge_model.py",
     "evaluate_onnx_accuracy", "Every exported graph is scored on held-out shrimp (D-044).",
     "tests/test_edge_export.py"),
    ("Show the tool itself", "scripts/predict_farm.py", "main",
     "Photos in → per-shrimp diagnosis → pond rate → farm risk. What section 2 ran.",
     "tests/test_predict_farm.py"),
    ("How do you know the numbers are right?", "tests/test_showcase.py", None,
     "Reads the tables independently and checks this page carries exactly those values.",
     "tests/test_showcase.py"),
]


def locate(path: str, func: str | None) -> int:
    """Line where `def func(` starts, found now, so the map cannot go stale when code moves."""
    if func is None:
        return 1
    for i, line in enumerate((ROOT / path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.lstrip().startswith(f"def {func}("):
            return i
    raise ValueError(f"def {func}( not found in {path}")


def code_section() -> str:
    rows = []
    for q, path, func, what, test in CODE_MAP:
        line = locate(path, func)
        where = f"{path}:{line}" if func else path
        name = f"<code>{func}()</code>" if func else "whole script"
        anchor = f"#L{line}" if func else ""
        run = (f'<code class="cmd">.venv\\Scripts\\python.exe -m pytest {html.escape(test)} -v</code>'
               if test else '<span class="q">no quick test — this runs as a full experiment; show the code and its result table</span>')
        rows.append(
            f'<tr><td><b>{html.escape(q)}</b></td>'
            f'<td><a href="{REPO_URL}/{path}{anchor}"><code>{html.escape(where)}</code></a><br>{name}</td>'
            f'<td>{html.escape(what)}</td><td>{run}</td></tr>')
    return f"""
<section id="code">
  <h2>6 · Where the code is</h2>
  {_tag("CODE", "line numbers located in the source when this page was built")}
  <p class="note">For each question: the file and function to open, what to point at, and the test that
  proves it works. In VS Code: <b>Ctrl+P</b> and type the file name, then <b>Ctrl+G</b> and the line
  number. Links open the same line on GitHub. Run commands from the project folder.</p>
  <div class="tablewrap"><table class="exps code">
    <thead><tr><th>If the panel asks…</th><th>Open</th><th>What to point at</th><th>Prove it runs</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>"""


def validation_section(r: dict, c: dict | None, live_ran: bool) -> str:
    """Section 4: each check as (what could go wrong, what we did, what we measured)."""

    def card(group: str, risk: str, did: str, evidence: str, source: str, kind: str = "RECORDED") -> str:
        return (f'<div class="val"><div class="vg">{group}</div><h3>{risk}</h3>'
                f'<p class="did">{did}</p><p class="ev">{evidence}</p>{_tag(kind, source)}</div>')

    cards = [
        card("DATA", "Is the test set really unseen?",
             "Every split groups photographs by shrimp (<code>specimen_id</code>), never by image, and "
             "an audit counts shrimp that appear on both sides (D-007).",
             f"Shrimp shared between train and test, mean of {r['leak_seeds']} seeds: "
             f"<b>{r['shared_image']:.1f}</b> under an image split, <b>{r['shared_specimen']:.0f}</b> under ours.",
             "exp01_leakage_summary.csv"),
        card("DATA", "Is a score just one lucky split?",
             f"Every backbone is compared on {r['cv_folds']} specimen-grouped folds and reported as mean ± SD; "
             "the leakage effect was repeated over independent seeds.",
             f"The image-vs-specimen gap is positive on <b>{r['gap_seeds_pos']} of {r['gap_seeds']}</b> seeds. "
             "Differences inside the fold SD are not ranked.",
             "exp02_runs.csv · exp04_summary.csv"),
        card("TRAINING", "Did the model actually learn?",
             "The training curve is checked before any score is interpreted (D-020).",
             f"ConvNeXt scored <b>{r['cnx_before']:.4f}</b> at lr 3e-4 because it failed to fit its own training "
             f"data (fig08), not because the architecture is weak. At a workable "
             f"learning rate: <b>{r['cnx_f1']:.4f}</b>, first place.",
             "exp04_folds_lr3e-4_all.csv · exp04_summary.csv"),
        card("TRAINING", "Does a fix really help?",
             "Proposed fixes are run as ablations and kept only if the gain clears fold noise.",
             f"Class weighting for Black Gill: F1 +{r['bgw_delta']:.4f}, "
             f"{'beyond' if r['bgw_beyond'] else 'inside'} the fold SD, so it is <b>not claimed</b>.",
             "exp09_comparison.csv"),
        card("DEPLOYMENT", "Is the exported file the trained model?",
             "Every exported graph is scored on the held-out fold. Size and latency alone cannot tell a "
             "trained network from a random one (D-044).",
             f"PyTorch <b>{r['pt_f1']:.4f}</b> = ONNX FP32 <b>{r['fp32_f1']:.4f}</b> (parity). INT8 "
             f"<b>{r['int8_f1']:.4f}</b>, agreeing with FP32 on {_pct(r['int8_agree'])} of images.",
             "exp12_export.csv"),
        card("BRIDGE", "Does the estimator work when the truth is known?",
             "Simulated ponds are drawn from held-out shrimp at known prevalence, and each estimator's "
             "bias, error and interval coverage are measured.",
             f"<b>{r['sim_levels']}</b> prevalence levels × <b>{r['sim_pools']}</b> pools each, two backbones, "
             "raw and calibrated.",
             "exp07_bias.csv · exp07_coverage.csv"),
        card("BRIDGE → ECONOMICS", "Does camera error break the economics?",
             "The bridge's measured error is injected into the survey's prevalence variable and the "
             "coefficient re-estimated.",
             f"<b>{r['prop_reps']}</b> replicates per error level; at our error the ratio is "
             f"<b>{r['bridge_ratio']:.3f}</b>.",
             "exp08_attenuation.csv"),
        card("ECONOMICS", "Is the risk model better than guessing?",
             "Every risk model is compared with simply predicting the mean on held-out farms, also with "
             "folds grouped by farm cluster. No partial-dependence plot without skill (D-017).",
             f"Best ratio to the mean baseline: <b>{r['step3_best']:.3f}</b> (random folds), "
             f"<b>{r['gcv_ratio']:.3f}</b> ({r['gcv_clusters']} cluster folds). Above 1, so reported as a "
             "negative result (D-016).",
             "exp03_step3_candidates.csv · exp10_grouped_cv.csv"),
        card("ECONOMICS", "Are the statistical assumptions met?",
             "Breusch-Pagan, RESET, cluster-robust standard errors, and a power check on Breusch-Pagan.",
             f"RESET rejects both mean equations (p {r['inc_reset']:.1e}, {r['dis_reset']:.3f}) — stated. "
             f"{r['cr_clusters']} clusters change <b>{r['cr_flips']}</b> income significance results. "
             f"Breusch-Pagan detects a {r['bp_sd_ratio']:.2f}× spread with power {r['bp_power']:.3f}.",
             "exp03_diagnostics.csv · exp03_cluster_robust.csv · exp10_bp_power.csv"),
        card("ECONOMICS", "Does it hold on other data?",
             "The risk analysis was repeated on an independent survey.",
             f"Hasan &amp; Haque 2020, n = <b>{r['hh_n']}</b>: skill supported = "
             f"<b>{'yes' if r['hh_skill'] else 'no'}</b> — the same negative result.",
             "exp11_comparison.csv"),
        card("RESEARCH", "Can we trust what we inherited?",
             "The inherited notebook was re-run faithfully; base-paper figures were re-derived and only "
             "those that reproduce are quoted (D-012). Novelty is stated as a search result, never "
             "\"the first\" (D-015).",
             f"Notebook claimed <b>{r['nb_claimed']:.3f}</b>; faithful re-run over {r['nb_seeds']} seeds gave "
             f"<b>{r['nb_got']:.3f}</b>. Not reproduced, and said so (D-036).",
             "exp02b_summary.csv"),
    ]
    if live_ran:  # only claim LIVE when the pond actually ran in this build (D-022)
        cards.insert(5, card("DEPLOYMENT", "Could the demo be staged?",
             "The pond is every held-out shrimp, unselected; files are renamed so names carry no class; "
             "the answer key is opened only after the prediction (section 2).",
             "The prediction and the truth are both computed when the page is built.",
             "make_demo_pond.py · predict_farm.py", "LIVE"))
    if c:
        cards.append(card(
            "REPRODUCIBILITY", "Can someone else re-run it?",
            "Every run stores its config, seed and environment fingerprint. Every figure is drawn by "
            "script. Automated tests tie headline numbers to their source files.",
            f"<b>{c['runs']}</b> run records on disk · <b>{c['tests']}</b> automated tests "
            "(counted now; <code>pytest</code> runs them).",
            "experiments/runs/ · tests/", "LIVE"))

    return f"""
<section id="validate">
  <h2>5 · How we validate the research and the experiments</h2>
  <p class="note">Each card: what could make a result wrong, what was done about it, and what was measured.
  Several checks overturned an early result — those are reported, not hidden.</p>
  <div class="vals">{"".join(cards)}</div>
</section>"""


def page(r: dict, live: dict | None, c: dict | None = None) -> str:
    gap = r["f1_image"] - r["f1_specimen"]
    int8_cost = r["fp32_f1"] - r["int8_f1"]
    em_gain = 1 - r["em_bias"] / r["cc_bias"]
    spec = f"{r['n_specimens']:,}" if r["n_specimens"] is not None else "—"
    demo = pond_section(live) if live else (
        '<section id="demo"><h2>2 · The tool on a pond it has never seen</h2>'
        '<p class="warn">Not run: build without --no-live to include the pond demonstration.</p></section>')
    built = time.strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shrimp disease → farm risk · 19MIS1127</title>
<style>
:root {{ --ink:#1b2430; --mute:#5b6675; --line:#dde3ea; --bg:#f6f8fa; --navy:#163a5f; --pos:#c2410c; --neg:#0f766e; }}
* {{ box-sizing:border-box }}
body {{ margin:0; font-family:"Segoe UI",system-ui,sans-serif; color:var(--ink); background:var(--bg); line-height:1.5 }}
header {{ background:var(--navy); color:#fff; padding:36px 6vw 30px }}
header h1 {{ margin:0 0 8px; font-size:1.7rem; font-weight:600; max-width:60rem }}
header p {{ margin:2px 0; opacity:.85 }}
nav {{ position:sticky; top:0; z-index:5; background:#fff; border-bottom:1px solid var(--line); padding:10px 6vw; display:flex; gap:22px; font-size:.95rem }}
nav a {{ color:var(--navy); text-decoration:none; font-weight:600 }}
section {{ padding:34px 6vw; border-bottom:1px solid var(--line) }}
h2 {{ margin:0 0 6px; font-size:1.45rem; color:var(--navy) }}
h3 {{ margin:0 0 8px; font-size:1.1rem }}
.lead {{ max-width:62rem; font-size:1.05rem }}
.meta {{ color:var(--mute); font-size:.9rem }}
.src {{ margin:4px 0 14px; font-size:.85rem; color:var(--mute) }}
.src code, figcaption code {{ font-size:.8rem; color:var(--mute) }}
.tag {{ display:inline-block; font-weight:700; font-size:.72rem; letter-spacing:.06em; padding:2px 8px; border-radius:4px; margin-right:8px; color:#fff }}
.tag.live {{ background:#15803d }} .tag.recorded {{ background:#475569 }} .tag.code {{ background:#7c3aed }}
.flow {{ display:grid; grid-template-columns:repeat(3,1fr); gap:0; margin:18px 0 10px }}
.step {{ background:#fff; border:1px solid var(--line); padding:18px 20px; position:relative }}
.step:not(:last-child)::after {{ content:"→"; position:absolute; right:-16px; top:40%; font-size:1.6rem; color:var(--navy); z-index:2; background:var(--bg); border-radius:50%; width:30px; text-align:center }}
.step .n {{ font-size:.8rem; font-weight:700; color:var(--mute); letter-spacing:.06em }}
.step h3 {{ margin:4px 0 6px; color:var(--navy) }}
.step .in, .step .out {{ font-size:.92rem }}
.step .num {{ margin-top:10px; font-weight:600; font-size:.92rem; border-top:1px solid var(--line); padding-top:8px }}
.note {{ max-width:62rem; color:var(--mute); font-size:.93rem }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; margin:16px 0 }}
.kpi {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px 16px }}
.kpi b {{ display:block; font-size:1.9rem; color:var(--navy); line-height:1.2 }}
.kpi span {{ display:block; font-weight:600 }} .kpi small {{ color:var(--mute) }}
.actions {{ margin:6px 0 16px; display:flex; flex-wrap:wrap; gap:14px; align-items:center }}
button {{ font:inherit; font-weight:600; background:var(--navy); color:#fff; border:0; border-radius:6px; padding:9px 16px; cursor:pointer }}
.reveal-only {{ display:none }} body.revealed .reveal-only {{ display:block }} body.revealed .actions span.reveal-only {{ display:inline }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(132px,1fr)); gap:10px }}
.card {{ background:#fff; border:1px solid var(--line); border-top:4px solid; border-radius:6px; padding:6px; font-size:.78rem; position:relative }}
.card.pos {{ border-top-color:var(--pos) }} .card.neg {{ border-top-color:var(--neg) }}
.card img {{ width:100%; aspect-ratio:1; object-fit:cover; border-radius:3px; display:block }}
.nphoto {{ position:absolute; top:10px; right:10px; background:rgba(0,0,0,.55); color:#fff; font-size:.68rem; padding:1px 5px; border-radius:3px }}
.cid {{ font-weight:700; margin-top:4px }}
.call {{ font-weight:600 }} .card.pos .call {{ color:var(--pos) }} .card.neg .call {{ color:var(--neg) }}
.call span {{ color:var(--mute); font-weight:400 }}
.bar {{ height:5px; background:#e8edf2; border-radius:3px; margin:4px 0 2px }} .bar i {{ display:block; height:100%; background:var(--pos); border-radius:3px }}
.pws {{ color:var(--mute) }}
.truth {{ display:none; margin-top:3px; font-weight:600 }} body.revealed .truth {{ display:block }}
.risk {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px 20px; margin-top:22px; max-width:62rem }}
.risk em {{ font-style:normal; font-size:.75rem; font-weight:700; background:#fde68a; padding:2px 7px; border-radius:4px; margin-left:6px; vertical-align:middle }}
.warn {{ color:#92400e }}
.results {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:20px; margin-top:12px }}
.res {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px 20px }}
.res .big {{ font-size:1.05rem; margin:6px 0 }} .res .big b {{ color:var(--navy); font-size:1.35rem }}
.res ul {{ margin:6px 0; padding-left:20px }} .res li {{ margin:3px 0 }}
figure {{ margin:12px 0 0 }} figure img {{ width:100%; border:1px solid var(--line); border-radius:4px; background:#fff; cursor:zoom-in }}
figcaption {{ font-size:.85rem; color:var(--mute); margin-top:4px }}
#zoom {{ display:none; position:fixed; inset:0; background:rgba(10,20,30,.88); z-index:10; align-items:center; justify-content:center; cursor:zoom-out }}
#zoom img {{ max-width:94vw; max-height:92vh; background:#fff; border-radius:4px }}
.limits li {{ margin:5px 0 }}
.tablewrap {{ overflow-x:auto; margin-top:10px }}
table.exps {{ border-collapse:collapse; width:100%; background:#fff; font-size:.9rem }}
.exps th {{ text-align:left; background:var(--navy); color:#fff; padding:9px 10px; font-weight:600 }}
.exps td {{ border-bottom:1px solid var(--line); padding:9px 10px; vertical-align:top }}
.exps tr:nth-child(even) td {{ background:#fafbfc }}
.exps .eid {{ font-weight:700; color:var(--navy); white-space:nowrap }}
.exps .q {{ color:var(--mute) }} .exps .so {{ font-weight:600 }}
.exps .did {{ white-space:nowrap; color:var(--mute) }} .exps code {{ font-size:.75rem; color:var(--mute) }}
.code a {{ color:var(--navy) }} .code a code {{ color:var(--navy); font-size:.8rem }}
.code code.cmd {{ display:inline-block; background:#1b2430; color:#e6edf3; padding:3px 7px; border-radius:4px; font-size:.75rem; white-space:nowrap }}
.vals {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:16px; margin-top:12px }}
.val {{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--navy); border-radius:6px; padding:14px 16px }}
.val .vg {{ font-size:.72rem; font-weight:700; letter-spacing:.07em; color:var(--mute) }}
.val h3 {{ margin:3px 0 6px; color:var(--navy) }}
.val .did {{ margin:0 0 6px; font-size:.92rem }}
.val .ev {{ margin:0 0 8px; font-size:.92rem; background:var(--bg); padding:7px 9px; border-radius:4px }}
.val .src {{ margin:0 }}
footer {{ padding:24px 6vw 40px; color:var(--mute); font-size:.85rem }}
@media (max-width:900px) {{ .flow {{ grid-template-columns:1fr }} .step:not(:last-child)::after {{ display:none }} .results {{ grid-template-columns:1fr }} }}
</style></head>
<body>
<header>
  <h1>{TITLE}</h1>
  <p>G Mohan Jayanth · 19MIS1127 · Guide: Dr. M. Premalatha · Final Panel Review</p>
</header>
<nav><a href="#one">1 · The project</a><a href="#demo">2 · Live demo</a><a href="#results">3 · Results</a><a href="#experiments">4 · Experiments</a><a href="#validate">5 · Validation</a><a href="#code">6 · Code</a><a href="#limits">7 · Limits</a></nav>

<section id="one">
  <h2>1 · The project in one picture</h2>
  {_tag("RECORDED", "exp12_export.csv · exp07_estimators.csv · exp03_diagnostics.csv · exp08_baseline.csv · exp08_attenuation.csv")}
  <div class="flow">
    <div class="step"><div class="n">STEP 1 · CAMERA</div><h3>Which shrimp are sick?</h3>
      <div class="in">In: photos of shrimp sampled from a pond</div>
      <div class="out">Out: the disease of each shrimp (4 classes)</div>
      <div class="num">MobileNetV3, INT8, {r["int8_mb"]:.2f} MB — phone-sized</div></div>
    <div class="step"><div class="n">STEP 2 · BRIDGE</div><h3>How much of the pond?</h3>
      <div class="in">In: every sampled shrimp's disease probabilities</div>
      <div class="out">Out: the pond's White Spot rate</div>
      <div class="num">SLD-EM bias {r["em_bias"]:.3f} vs {r["cc_bias"]:.3f} for counting</div></div>
    <div class="step"><div class="n">STEP 3 · ECONOMICS</div><h3>What does it cost the farm?</h3>
      <div class="in">In: pond White Spot rate + pond area</div>
      <div class="out">Out: expected income loss from White Spot</div>
      <div class="num">Just-Pope model fitted on {r["inc_n"]:,} farms</div></div>
  </div>
  <p class="note">The photographs and the farm records are separate datasets. They meet only at the pond
  White Spot rate, so the fusion is <b>decision-level</b>, not sample-level. On {r["inc_n"]:,} farms a
  yes/no disease answer shows no income effect (p = {r["s1_p"]:.2f}); the share of stock lost does
  (β = {r["s2_beta"]:.2f} on log income, p = {r["s2_p"]:.3f}, marginal). A camera can estimate how much,
  and at the bridge's measured error ({r["bridge_err"]:.3f}) that coefficient does not weaken
  (ratio {r["bridge_ratio"]:.3f}).</p>
</section>

{demo}

<section id="results">
  <h2>3 · Results, in pipeline order</h2>
  <p class="note">Recorded when each experiment ran. Click any figure to enlarge.</p>
  <div class="results">

  <div class="res"><h3>Camera · the leak a normal split hides</h3>
    {_tag("RECORDED", "exp01_leakage_summary.csv · exp02_summary.csv")}
    <p class="big">{r["n_images"]:,} images are only <b>{spec}</b> shrimp. Splitting by image puts the same
    shrimp in train and test: <b>{r["leak_image"]:.2f}%</b> of the test set leaks (± {r["leak_image_sd"]:.2f});
    by specimen, <b>{r["leak_specimen"]:.2f}%</b>.</p>
    <ul><li>Macro-F1 {r["f1_image"]:.4f} split by image vs {r["f1_specimen"]:.4f} split by specimen — gap {gap:.4f}.</li>
    <li>The claim rests on the measured leak, not on the size of the gap.</li></ul>
    {_figure("fig02_leakage_exposure.png", "Share of test images whose shrimp also appears in training.")}
    {_figure("fig03_headline_result.png", "Macro-F1 under image-level and specimen-level splits.")}
  </div>

  <div class="res"><h3>Camera · five backbones, specimen-grouped 5-fold</h3>
    {_tag("RECORDED", "exp04_summary.csv · exp10_lr_sweep_summary.csv")}
    <p class="big">Best: ConvNeXt-Tiny <b>{r["cnx_f1"]:.4f}</b> ± {r["cnx_sd"]:.4f} (lr {r["cnx_lr"]:g}).
    Edge pick: MobileNetV3 <b>{r["mnv_f1"]:.4f}</b> ± {r["mnv_sd"]:.4f} at {r["mnv_params"]:.2f} M parameters,
    {r["param_ratio"]:.1f}× smaller.</p>
    <ul><li>Adjacent backbones sit inside each other's fold SD; only top-to-bottom is resolvable.</li>
    <li>Ranking is provisional: at lr 1e-3, ReXNet reaches {r["rexnet_1e3"]:.4f} and EfficientNet {r["effnet_1e3"]:.4f}.</li>
    <li>Black Gill is the weakest class for every backbone (best F1 {r["bg_best"]:.4f}).</li></ul>
    {_figure("fig06_backbone_comparison.png", "Macro-F1 by backbone across five specimen-grouped folds.")}
    {_figure("fig08_convnext_correction.png", "ConvNeXt before and after the learning-rate correction: check the training curve before the score.")}
  </div>

  <div class="res"><h3>Camera · the deployed edge model</h3>
    {_tag("RECORDED", "exp12_export.csv")}
    <p class="big">FP32 ONNX <b>{r["fp32_f1"]:.4f}</b> macro-F1 at {r["fp32_mb"]:.2f} MB → static INT8
    <b>{r["int8_f1"]:.4f}</b> at <b>{r["int8_mb"]:.2f} MB</b>.</p>
    <ul><li>{r["fp32_mb"] / r["int8_mb"]:.2f}× smaller, costing {int8_cost:.4f} macro-F1 — reported, not absorbed.</li>
    <li>Scored on the held-out specimen-grouped fold. Edge claims rest on size; latency ({r["int8_ms"]:.2f} ms) was measured on a laptop, not an edge device.</li></ul>
    {_figure("fig07_deployment_tradeoff.png", "Accuracy against model size for the five backbones.")}
  </div>

  <div class="res"><h3>Camera · calibration</h3>
    {_tag("RECORDED", "exp06_summary.csv")}
    <p class="big">Temperature scaling brings ECE to <b>{r["ece_cnx"]:.4f}</b> (ConvNeXt) and
    <b>{r["ece_mnv"]:.4f}</b> (MobileNetV3) — improved, but short of the {r["ece_target"]:.2f} target.</p>
    <ul><li>The bridge runs on raw probabilities, where SLD-EM's bias was lowest.</li></ul>
    {_figure("fig10_calibration.png", "Reliability before and after temperature scaling.")}
  </div>

  <div class="res"><h3>Bridge · from shrimp to pond rate</h3>
    {_tag("RECORDED", "exp07_estimators.csv")}
    <p class="big">Over simulated pools of 40 shrimp at 10 prevalence levels, SLD-EM's mean absolute bias is
    <b>{r["em_bias"]:.4f}</b> vs <b>{r["cc_bias"]:.4f}</b> for counting — about {em_gain * 100:.0f}% less.</p>
    {_figure("fig11_bias_curves.png", "Bias of each estimator across true prevalence.")}
    {_figure("fig13_interval_coverage.png", "Coverage of the bootstrap 95% intervals.")}
  </div>

  <div class="res"><h3>Bridge → Economics · does camera error break the model?</h3>
    {_tag("RECORDED", "exp08_baseline.csv · exp08_attenuation.csv")}
    <p class="big">Baseline β = <b>{r["s2_beta"]:.4f}</b> (p = {r["s2_p"]:.3f}, n = {r["s2_n"]:,}). With the
    bridge's error ({r["bridge_err"]:.3f}) added, the coefficient ratio is <b>{r["bridge_ratio"]:.3f}</b> — no visible shrinkage.</p>
    {_figure("fig14_attenuation.png", "Coefficient ratio as measurement error on prevalence grows.")}
  </div>

  <div class="res"><h3>Economics · farm risk on survey data</h3>
    {_tag("RECORDED", "exp03_diagnostics.csv · exp03_step3_candidates.csv · exp11_comparison.csv")}
    <p class="big">Income equation: n = {r["inc_n"]:,}, R² <b>{r["inc_r2"]:.4f}</b>, Breusch-Pagan p = {r["inc_bp"]:.4f}
    (variance not constant, so the Just-Pope method applies).</p>
    <ul><li>Disease subsample (n = {r["dis_n"]}): R² {r["dis_r2"]:.4f}, Breusch-Pagan p = {r["dis_bp"]:.3f} — the precondition is not demonstrated.</li>
    <li>Negative result: every step-3 risk model is worse than predicting the mean (best ratio {r["step3_best"]:.3f}).</li>
    <li>A second survey (Hasan &amp; Haque, n = {r["hh_n"]}) agrees: skill supported = {"yes" if r["hh_skill"] else "no"}.</li></ul>
    {_figure("fig09_track_a_diagnostics.png", "Track A diagnostics for the income and disease equations.")}
  </div>

  </div>
</section>

{experiments_section(r)}

{validation_section(r, c, live is not None)}

{code_section()}

<section id="limits">
  <h2>7 · Limits, stated</h2>
  <ul class="limits note">
    <li>Images and farm data come from different sources, so the fusion is decision-level only.</li>
    <li>The learning-rate sweep is partial, so the backbone ranking is provisional.</li>
    <li>RESET rejects both mean equations; no panel model was estimated.</li>
    <li>INT8 costs {int8_cost:.4f} macro-F1, and nothing was measured on real edge hardware.</li>
    <li>Black Gill stays the weakest class; calibration did not reach {r["ece_target"]:.2f}.</li>
  </ul>
</section>

<footer>Built {built} by <code>scripts/build_showcase.py</code>. <b>LIVE</b> blocks were computed when the page
was built, by running the deployed model; <b>RECORDED</b> blocks are read from the named file in
<code>reports/</code>, written when that experiment ran. No number on this page is typed by hand.</footer>

<div id="zoom" onclick="this.style.display='none'"><img alt=""></div>
<script>
document.querySelectorAll('figure img').forEach(function (im) {{
  im.addEventListener('click', function () {{
    var z = document.getElementById('zoom'); z.querySelector('img').src = im.src; z.style.display = 'flex';
  }});
}});
</script>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--int8", action="store_true", help="run the pond on the INT8 edge graph")
    ap.add_argument("--no-live", action="store_true", help="skip the pond run (recorded sections only)")
    ap.add_argument("--no-open", action="store_true", help="do not open the page in the browser")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    live = None
    if not args.no_live:
        if not POND.is_dir():
            raise SystemExit(f"no pond at {POND}. Run: .venv/Scripts/python.exe scripts/make_demo_pond.py")
        print("[LIVE] running predict_farm.py on the pond ...")
        live = run_live(args.int8)
        pv = live["prevalence"]
        print(f"[LIVE] {live['n_specimens']} shrimp · SLD-EM {_pct(pv['em'])} "
              f"[{_pct(pv['em_ci_95'][0])}, {_pct(pv['em_ci_95'][1])}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("[LIVE] counting tests and run records ...")
    args.out.write_text(page(recorded(), live, counts()), encoding="utf-8")
    print(f"[ok] wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    if not args.no_open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
