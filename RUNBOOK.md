# RUNBOOK — how to run this project

**An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp Disease Detection Using Image**
G Mohan Jayanth · 19MIS1127 · Guide: Dr. M. Premalatha

Everything here is a command you type yourself. No AI assistant required. Follow it top to bottom on
a clean machine and you will reproduce every number in `reports/`.

**Total time from zero:** ~30 min setup and data, ~75 min if you re-run every experiment.

---

## 0. What you are running

Two independent tracks plus a shared data layer.

| Track | Question | Entry point | Hardware |
|---|---|---|---|
| **B — Vision** | Can a model detect shrimp disease from a photo of a shrimp it has never seen? | `exp01`, `exp02`, `exp04` | GPU strongly preferred |
| **A — Econometrics** | Do farm and management variables predict income and disease risk? | `exp03` | CPU only |

The project's central claim is about **evaluation protocol**: the 1,149 images are photographs of
only **416 individual shrimp**, so splitting by image leaks the same animal into train and test.
Experiments 01 and 02 measure that. Everything else builds on it.

---

## 1. Requirements

| | Minimum | This project was run on |
|---|---|---|
| OS | Windows / Linux / macOS | Windows 11 |
| Python | 3.10+ | **3.12.10** |
| RAM | 8 GB | 16 GB |
| Disk | **~2.5 GB free** (data + cache + models) | — |
| GPU | optional | **NVIDIA RTX 5060 Laptop, 8 GB, CUDA 12.8** |

**GPU note that matters.** The RTX 5060 is a Blackwell card (compute capability `sm_120`). It
requires a **CUDA 12.8 or newer** PyTorch build. The default PyPI `torch` will install, import, and
then fail at the first CUDA call. Use the index URL in Section 2.

**No GPU?** Everything still runs on CPU. Experiments 02 and 04 go from ~20 and ~45 minutes to
several hours. Experiments 01 and 03 are CPU-only anyway and are unaffected.

---

## 2. First-time setup

From the project root (`mini project/`):

```bash
# 1. create the project's own Python environment
python -m venv .venv

# 2. install PyTorch FIRST, from PyTorch's index (see the GPU note above)
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. everything else
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

> **Linux / macOS:** replace `.venv/Scripts/python.exe` with `.venv/bin/python` everywhere in this
> document. That is the only difference.

**Always call the interpreter by path.** Typing bare `python` picks up the system install, which
does not have these packages. Every command below does this.

### Verify the environment

```bash
.venv/Scripts/python.exe -c "import torch, timm, sklearn, statsmodels, cubist; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected on the project machine:

```
torch 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
```

`CUDA available: False` is not fatal — it means CPU mode. It **is** a problem if you have an NVIDIA
GPU and expected otherwise: you installed the wrong torch build. Redo step 2.

---

## 3. Get the data

Three datasets. Two are manual downloads (the hosts do not allow scripted download); one has a
script.

### 3.1 Shrimp images — manual, ~471 MB

1. Open <https://data.mendeley.com/datasets/jhrtdj9txm/3> — **no login required**
2. Click **Download All**
3. Put the `.zip` in `data/raw/` **without renaming it**

The extraction script looks for it by its exact published name:
`ShrimpDiseaseImageBD An Image Dataset for Computer Vision-Based Detection of Shrimp Diseases in Bangladesh.zip`

*Citation: Islam et al. (2025), Mendeley Data, doi:10.17632/jhrtdj9txm.3, CC BY 4.0*

### 3.2 Farm disease survey — manual, ~2.7 MB

1. Open <https://data.mendeley.com/datasets/nz96v5spbf/2>
2. Download `WSSVRiskFactor_rev.xlsx`
3. Put it in `data/raw/farm_economics/`

Verify you got the right bytes:

```bash
sha256sum "data/raw/farm_economics/WSSVRiskFactor_rev.xlsx"
# expect: e7b323ffeb8b2832d876bb8ea85f56cc3a4dc0bc944cce79637b7cc7fe35b196
```

*Citation: Hasan & Haque (2020), Data in Brief 31:105936, CC BY*

### 3.3 IFPRI farm economics survey — scripted, ~8.7 MB

```bash
.venv/Scripts/python.exe scripts/fetch_ifpri.py --list            # preview: 61 files, ~8.9 MB
.venv/Scripts/python.exe scripts/fetch_ifpri.py                   # baseline + manifest
.venv/Scripts/python.exe scripts/fetch_ifpri.py --round endline   # endline: 59 files, ~11 MB
```

Resolves the DOI, checks the dataset is `RELEASED`, verifies every file's server-side MD5, and
writes a manifest.

**The baseline is what every reported result uses**, and it is the default — running the script with
no arguments reproduces the data behind every Track A table. The endline is a **second, separate
publication** with its own DOI, fetched to its own directory (`data/raw/ifpri_endline/`) so the two
rounds can never be confused. They share module *names* but not module *numbers*: the disease module
is `019_hh_modc9_c900-c901` in the baseline and `018_…` in the endline.

| Round | DOI | Collected | Directory | Farms |
|---|---|---|---|---:|
| Baseline | `doi:10.7910/DVN/TUXYXQ` | Nov–Dec 2023 | `data/raw/ifpri/` | 1,221 |
| Endline | `doi:10.7910/DVN/VQJNLG` | 7–31 May 2024 | `data/raw/ifpri_endline/` | 1,222 |

Every baseline farm reappears in the endline (1,221 of 1,221), so the two rounds form a panel. **No
analysis in this project uses the endline yet** — see `D-038` and `stage_10_completion.md` Section 6.

*Citation: IFPRI (2024), Harvard Dataverse, doi:10.7910/DVN/TUXYXQ (baseline),
doi:10.7910/DVN/VQJNLG (endline)*

---

## 4. Build the derived data

Run these **once**, in this order.

```bash
# 1. unpack images, verify counts, build the manifest with specimen ids and SHA-256 (~2 min)
.venv/Scripts/python.exe scripts/extract_images.py

# 2. pre-resize to 256 px  (511 MB -> 33 MB, ~25 s) -- makes training ~10x faster
.venv/Scripts/python.exe scripts/build_image_cache.py --size 256
```

**Step 1 must print this, exactly:**

```
raw images      : 1,149
annotated images: 746

unique specimens: 416 across 1,149 images (2.76 images per specimen)
  group sizes: {1: 24, 2: 51, 3: 341}
  -> image-level random splitting WILL leak specimens across train/test

per-class (raw):
  Healthy      403  expected   403  OK
  BG           198  expected   198  OK
  WSSV         328  expected   328  OK
  BG_WSSV      220  expected   220  OK
```

**If any line says `MISMATCH`, stop.** The archive is not the one this project was built on and no
downstream number will be valid.

Why the cache: source images are 2048×2048 (4.2 megapixels) but training uses 224×224. Without the
cache, JPEG decoding dominates and the GPU sits idle.

---

## 5. Run the tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

**Expected: `178 passed` in about 12 seconds.**

Tests that need the ~500 MB image corpus or the survey files are marked `dataset` and **skip** when
those are absent. Skipped is fine; failed is not.

Useful variants:

```bash
.venv/Scripts/python.exe -m pytest -v                          # see every test name
.venv/Scripts/python.exe -m pytest tests/test_splits.py -v      # one file
.venv/Scripts/python.exe -m pytest -k leakage -v                # match by name
```

**The one test worth knowing about:**
`tests/test_econometrics.py::test_deviance_recovers_known_variance_function` generates data whose
variance depends on the inputs *by construction* and asserts the estimator finds it. Its mirror
asserts the test does **not** fire on constant-variance data. Together they are why the negative
result in Experiment 03 is a statement about the data rather than a bug in the code.

---

## 6. Run the experiments

Run in order. Each writes to `reports/tables/` and `experiments/runs/`.

### Experiment 01 — how much leakage? (seconds, CPU)

```bash
.venv/Scripts/python.exe scripts/profile_ifpri.py             # Stage 3a descriptives
.venv/Scripts/python.exe scripts/exp01_leakage_audit.py
```

No models trained — splits only, 5 seeds, both protocols.

**Expect:**

```
image_level       87.98% +/- 1.32
specimen_level     0.00% +/- 0.00
```

That second line is also the correctness check on the splitter. Anything but `0.00%` means the
grouping logic is broken.

### Experiment 02 — does leakage inflate scores? (~20 min GPU / hours CPU)

```bash
.venv/Scripts/python.exe scripts/exp02_leakage_effect.py
```

9 training runs: 3 conditions × 3 seeds. Conditions B and C differ in **exactly one factor** — the
split protocol.

**Expect:**

```
A_image_level_noaug   0.7410  (SD 0.0515)
B_image_level         0.8215  (SD 0.0636)
C_specimen_level      0.6721  (SD 0.0004)

Leakage-attributable optimism (B - C): +0.1494 macro-F1
```

### Experiment 03 — Track A risk modelling (~3 min, CPU only)

```bash
.venv/Scripts/python.exe scripts/exp03_track_a.py
```

**Expect a negative result, and that is the finding:**

```
INCOME   R2 0.2912 · Breusch-Pagan p=0.0115 · best model RMSE/baseline 1.014
DISEASE  R2 0.1220 · Breusch-Pagan p=0.195  · best model RMSE/baseline 1.079
PDP SKIPPED: no candidate beats the mean baseline
```

Every RMSE ratio above 1.0 means the model is worse than predicting the average. The script
**refuses** to draw partial-dependence plots in that case — a PDP of a model with no skill would show
structure that is not there.

### Experiment 04 — backbone comparison (~45 min GPU / many hours CPU)

```bash
.venv/Scripts/python.exe scripts/exp04_backbone_comparison.py
```

25 runs: 5 backbones × 5 grouped folds, plus deployment cost measurement.

**Expect (macro-F1, ranked):**

```
convnext_tiny          0.7189 +/- 0.0453   27.82 M   106.14 MB
resnet50               0.6920 +/- 0.0437   23.52 M    89.91 MB
mobilenetv3_large_100  0.6705 +/- 0.0415    4.21 M    16.14 MB
rexnet_150             0.6634 +/- 0.0731    7.82 M    30.05 MB
efficientnet_b0        0.6278 +/- 0.0371    4.01 M    15.47 MB
```

Faster options while developing:

```bash
# shape check only: 1 fold, 2 epochs
.venv/Scripts/python.exe scripts/exp04_backbone_comparison.py --smoke

# one backbone
.venv/Scripts/python.exe scripts/exp04_backbone_comparison.py --backbones mobilenetv3_large_100

# fewer folds / epochs
.venv/Scripts/python.exe scripts/exp04_backbone_comparison.py --folds 3 --epochs 10
```

> **`convnext_tiny` runs at learning rate 5e-5, the others at 3e-4.** This is deliberate and declared
> in `BACKBONE_LR` at the top of the script. At 3e-4 ConvNeXt failed to fit its own training data in
> 3 of 5 folds and scored 0.3966. See `docs/decisions/decision_log.md` `D-020`.

---

## 6d. Experiment 06 — fit the calibration temperature (~25 min GPU)

```bash
.venv/Scripts/python.exe scripts/exp06_calibrate_bridge.py
.venv/Scripts/python.exe scripts/exp06_calibrate_bridge.py --smoke   # 2 folds x 2 epochs, ~2 min
```

**Expected: 10 runs — `convnext_tiny` and `mobilenetv3_large_100` across the same five specimen
folds — then a summary giving ECE before and after temperature scaling for each, and a verdict
against the 0.05 target `D-028` set.**

It retrains because a temperature has to be fitted on data that is neither training nor test, and
the 34 runs stored in Stages 4–5 saved test posteriors only. Output goes to
`experiments/runs/exp06/`; **Experiment 04's artefacts are not touched**.

Two things to check in the output:

- **`macro-F1 (unchanged by construction)`** — a positive scalar divides every logit equally and
  cannot reorder them, so accuracy and macro-F1 are identical before and after. The script asserts
  this and aborts if it is ever false.
- **`Reproducibility vs Experiment 04`** — the folds, seeds and budget are identical, so the test
  scores should match Experiment 04's. They do, to four decimals, which is the strongest evidence
  the project has that its training is deterministic.

**Re-run `exp05_calibration.py` afterwards.** It measures every stored run, and
`test_calibration_was_measured_for_every_stored_run` fails if its table does not cover the ten new
artefacts.

---

## 6e. Experiment 07 — the prevalence bridge (~3 min, CPU)

```bash
.venv/Scripts/python.exe scripts/exp07_prevalence_bridge.py
.venv/Scripts/python.exe scripts/exp07_prevalence_bridge.py --smoke   # 10 replicates, no coverage
```

Needs Experiment 06's runs. Builds pseudo-farm pools at a prevalence chosen in advance and measures
the bias of five estimators — CC, PCC, ACC, PACC and Saerens EM — against a truth that is known
because it was set. Writes `exp07_bias.csv`, `exp07_estimators.csv`, `exp07_coverage.csv` and
`exp07_feasibility.csv`.

**Read `exp07_feasibility.csv`.** A fold holds about 83 specimens, roughly 40 of them White-Spot
positive, so some combinations of pool size and prevalence cannot be built without drawing the same
shrimp twice. Those cells are **refused and listed** rather than filled, and a gap in a bias curve
means "the data could not reach here", not "not measured".

---

## 6f. Experiment 08 — sensitivity of the risk estimates (~10 min, CPU)

```bash
.venv/Scripts/python.exe scripts/exp08_sensitivity.py
.venv/Scripts/python.exe scripts/exp08_sensitivity.py --smoke        # 10 replicates, no step 3
```

Perturbs the disease regressor in the IFPRI income equation at the error magnitudes Experiment 07
measured, and reports how far the coefficient moves. Two specifications: **S1**, the published one
carrying the binary self-report `had_white_spot`; **S2**, the bridge-facing one carrying the
continuous `wsd_prevalence`.

**Expect the S1 attenuation ratio to be suppressed.** Its unperturbed coefficient is not
distinguishable from zero, so there is no effect for measurement error to attenuate, and a ratio
computed from it would be noise over noise. The script says so explicitly instead of printing a
number. That absence is a finding, not a failure of the run.

---

## 7. Regenerate the figures

```bash
.venv/Scripts/python.exe scripts/make_figures.py           # fig01-fig05  (Stages 3-4)
.venv/Scripts/python.exe scripts/make_figures_stage56.py   # fig06-fig09  (Stages 5-6)
.venv/Scripts/python.exe scripts/make_figures_stage89.py   # fig10-fig14  (Stages 8-9)
```

All three read only from `reports/tables/` and `experiments/runs/`. **No figure is ever drawn by
hand** — re-run an experiment and its figures move with it, so a slide can never disagree with its
data. `make_figures_stage89.py` skips any figure whose table is missing and says which, so it is
safe to run before the experiments behind it have finished.

---

## 6c. Measure calibration

```bash
.venv/Scripts/python.exe scripts/exp05_calibration.py
```

**Expected: a table of eight groups in about two seconds**, ending with `C_specimen_level` at
`ece_mean` ~0.2046 and the protocol comparison printed underneath.

It trains nothing. It reads the per-image posteriors every run already saved and computes Expected
Calibration Error, maximum calibration error, the Brier score, and mean confidence against accuracy.
Writes `exp05_calibration.csv`, `exp05_calibration_summary.csv` and `exp05_reliability.csv`.

**Why it matters more than its runtime suggests.** Stage 8's prevalence estimators assume calibrated
posteriors. This measures whether they are, and they are not — about twenty points overconfident. A
calibration step is therefore a prerequisite for the bridge rather than a refinement of it
(`D-028`). `--bins N` changes the confidence binning; the result is stable from 10 to 20.

---

## 7a. See every result at once

```bash
.venv/Scripts/python.exe scripts/demo_results.py
```

**Expected: eight blocks of results in about 4 seconds**, ending in a summary card that reads
`87.98% image-level vs 0.00% specimen-level`, `gap 0.1494`, `convnext_tiny 0.7189 +/- 0.0453`.

This is the fastest way to see the state of the whole project, and the script used to demonstrate
it to a review panel. It trains nothing, downloads nothing and writes nothing, so it is safe to
run at any time, including while an experiment is running.

Each block is labelled with where its numbers come from:

| Label | Meaning |
|---|---|
| `LIVE` | computed in that process, from `data/interim/image_manifest.csv` or from the stored per-image posteriors. No results table is read |
| `RECORDED` | read from the artefact the experiment wrote, because re-training costs 20-45 minutes. The path is printed |

Three blocks additionally print a `cross-check` line, recomputing a metric from the raw stored
predictions and comparing it with the value in `reports/tables/`. Those must read `MATCH`; a
`DIFFERS` means a table and its run artefact have drifted apart, and
`tests/test_demo_results.py` fails for the same reason.

Useful variants:

```bash
.venv/Scripts/python.exe scripts/demo_results.py --list             # the section names
.venv/Scripts/python.exe scripts/demo_results.py --only leakage     # one block
.venv/Scripts/python.exe scripts/demo_results.py --pause            # Enter between blocks
```

### Flags this manual previously omitted

Four real flags were missing from these pages, which meant the only way to find them was to read
`add_argument` calls. They are listed here because a runbook that is silently incomplete is worse
than one that is obviously short.

| Script | Flag | What it does |
|---|---|---|
| `extract_images.py` | `--verify` | *"skip extraction, re-verify only"* — re-checks the corpus on disk against the manifest instead of rebuilding it. Run it when you suspect the extracted files have drifted |
| `exp04_backbone_comparison.py` | `--summary-only` | Rebuilds `exp04_summary.csv` from the folds and deployment tables **without retraining**. This is the flag you want after editing a reporting column: seconds instead of 45 minutes |
| `build_image_cache.py` | `--workers N` | Parallel resize workers, default 8. Lower it on a machine that runs out of memory during the 511 MB → 33 MB pass |
| `fetch_ifpri.py` | `--all` | *"fetch community/cluster files too"* — beyond the household modules the project actually uses. Only needed when exploring variables outside Track A |

Output is fixed at 78 columns of plain ASCII so it can be screenshotted onto a slide without
wrapping. Missing inputs produce an explicit `[skipped]` line naming the script that would
create them, never a traceback.

---

## 7b. Rebuild the notebook

`shrimp_disease_risk_project.ipynb` in the project root is the whole project as a readable
document: dataset, the evaluation flaw, the correction, the backbone comparison, Track A, and a
real training run, with every output stored under the cell that produced it.

```bash
.venv/Scripts/python.exe scripts/build_notebook.py
```

**Expected: `wrote 38 cells`, then `executed 24 code cells in ~95 s, 0 with errors`.** It trains
one fold for 8 epochs on the way through, so a GPU makes it quick; on CPU expect several minutes.

| Flag | Effect |
|---|---|
| *(none)* | build the cells and execute them, storing real outputs |
| `--no-execute` | write the source cells only, no outputs |
| `--epochs 3` | shorten the live demonstration run |

**The notebook is generated. Anything edited directly in the `.ipynb` is lost on the next build**
— edit the cells in `scripts/build_notebook.py` instead (`D-023`). Rebuild after any experiment
re-run: `tests/test_notebook.py` rebuilds the headline figures from `reports/tables/` and fails
if the notebook's stored outputs have gone stale.

The notebook imports `src/shrimp_risk/` rather than restating it, so there is one implementation
of every claim in the project. It writes nothing to disk — a test asserts that — and it sets
`HF_HUB_OFFLINE=1`, so it runs with the network off using the local timm weight cache.

---

## 7c. Rebuild the presentation

```bash
.venv/Scripts/python.exe scripts/build_pptx.py
```

**Expected: `wrote 29 slides -> Review2_19MIS1127.pptx`, notes on all 29**, in about two seconds.

The deck is a rendering of `docs/presentation/ppt_content.md`, with each slide's speaker notes
pulled from `speaker_notes.md` into the notes pane. **Nothing is authored in the script** — if a
number on a slide is wrong, fix it in the content file and rebuild (`D-032`).

| Flag | Effect |
|---|---|
| *(none)* | render the full deck to `Review2_19MIS1127.pptx` |
| `--out NAME.pptx` | write to a different filename |
| `--appendix` | include the optional "it runs" appendix slide — omitted by default, because its entry in the content file is presenter guidance and a screenshot you take on the day, not slide text |

**Restyling in PowerPoint is fine — editing content there is not.** The next build overwrites the
file, so any wording change must go into `ppt_content.md` or it is lost. `tests/test_pptx.py` fails
if the deck's headline numbers stop matching `reports/tables/`.

**The deck is 38 slides for 27 sections.** Four sections carry more than one slide holds and are
continued automatically, with the script naming them each run: the two literature tables (split by
row, header repeated), Track A, and limitations. Trim the content file if you would rather present
fewer — `ppt_content.md` carries a drop order and names the six slides that must never be dropped.

**Type is one size across the whole deck** — 15pt body, 11pt tables, 9pt source lines. An earlier
version sized each slide to fit, from 18pt down to 11pt, which fitted everything and looked like
eight different decks.

---

## 8. Run everything, start to finish

```bash
.venv/Scripts/python.exe scripts/fetch_ifpri.py
.venv/Scripts/python.exe scripts/extract_images.py
.venv/Scripts/python.exe scripts/build_image_cache.py --size 256
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/exp01_leakage_audit.py
.venv/Scripts/python.exe scripts/exp02_leakage_effect.py
.venv/Scripts/python.exe scripts/exp03_track_a.py
.venv/Scripts/python.exe scripts/exp04_backbone_comparison.py
.venv/Scripts/python.exe scripts/make_figures.py
.venv/Scripts/python.exe scripts/make_figures_stage56.py
.venv/Scripts/python.exe scripts/demo_results.py           # confirm it all lines up
.venv/Scripts/python.exe scripts/build_notebook.py         # rebuild the notebook (~95 s)
```

~75 minutes on the RTX 5060, assuming the two manual downloads are already in place.

---

## 9. Where everything lands

```
data/raw/            downloaded archives, never modified
data/interim/        extracted images + manifests (image_manifest.csv carries specimen_id + SHA-256)
data/processed/      256 px training cache
reports/tables/      17 CSVs — every number that reaches a slide
reports/figures/     9 PNGs — every chart that reaches a slide
experiments/runs/    per-run JSON: full epoch history, per-class metrics, confusion matrix,
                     environment fingerprint; .npz files hold per-image posteriors
.scratch/            working files, never a deliverable
```

**Key outputs by question:**

| Question | File |
|---|---|
| How much leakage? | `reports/tables/exp01_leakage_summary.csv` |
| Does it inflate scores? | `reports/tables/exp02_summary.csv` |
| Which backbone? | `reports/tables/exp04_summary.csv` |
| Is it edge-deployable? | `reports/tables/exp04_deployment.csv` |
| Track A risk models | `reports/tables/exp03_diagnostics.csv`, `exp03_step3_candidates.csv` |
| Which variables are derived vs observed? | `reports/tables/exp03_provenance.csv` |
| All of the above, in one pass | `.venv/Scripts/python.exe scripts/demo_results.py` (Section 7a) |
| All of the above, as a readable document | `shrimp_disease_risk_project.ipynb` (Section 7b) |

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | using system Python | call `.venv/Scripts/python.exe` by path |
| `ImportError: DLL load failed while importing _C` | partial torch install | `pip uninstall torch torchvision`, redo Section 2 step 2 |
| `CUDA available: False` on an NVIDIA machine | CPU build installed | reinstall with `--index-url .../cu128` |
| `CUDA error: no kernel image is available` | torch too old for Blackwell | need CUDA 12.8+; see Section 1 |
| `authoritative archive not found` | zip missing or renamed | Section 3.1 — keep the published filename |
| `MISMATCH` in extraction | wrong archive | re-download from the Mendeley DOI |
| `N manifest rows have no specimen_id` | manifest built by an older script | re-run `extract_images.py` |
| `fold N leaks M images — refusing to run` | splitter regression | **do not bypass this.** Run `pytest tests/test_splits.py` |
| Training very slow, GPU idle | cache not built | run `build_image_cache.py` (Section 4) |
| `CUDA out of memory` | 8 GB is tight for ConvNeXt/ResNet50 | add `--backbones mobilenetv3_large_100 efficientnet_b0`, or lower `batch_size` in `TrainConfig` |
| Results differ in the 3rd decimal | non-determinism | expected; `seed_everything` bounds it, reported SDs cover it |
| Results differ a lot | different data or environment | check `pytest`, then the `environment` block in any run JSON |
| `demo_results.py` prints `DIFFERS` | a table and its run artefact have drifted apart | re-run the experiment that writes that table; do not edit the CSV |
| `demo_results.py` prints `[skipped]` | that block's input is not on disk | run the script the line names; the rest of the output is unaffected |
| `test_notebook.py` fails on a headline number | a table was regenerated and the notebook was not | `.venv/Scripts/python.exe scripts/build_notebook.py` |
| Notebook build fails on the model weights | the timm cache is empty and `HF_HUB_OFFLINE=1` blocks the fetch | run once with the flag unset, then rebuild normally |
| `NoSuchKernel: python3` | `ipykernel` missing from the venv | `.venv/Scripts/python.exe -m pip install -r requirements.txt` |
| **`OSError: [WinError 1455] The paging file is too small`** on `import torch`, usually inside a dataloader worker | **three or more torch processes at once.** Every worker imports torch and reserves virtual address space; this machine exhausts the paging file at three | **Run one training script at a time.** Two proved safe; the third killed a run mid-experiment on 2026-09-06. Windows can grow the paging file (System → Advanced → Performance → Virtual memory), but serialising the runs costs nothing and needs no system change — see Section 10a |
| A run dies but leaves its earlier seeds' artefacts | expected: every experiment writes after each run so a crash loses only the run in progress | re-run the same command; the merge keeps completed rows |

---

## 10a. Running the experiments on this machine

**One training script at a time.** All of these load PyTorch, and each dataloader worker loads it
again. On the development machine (RTX 5060 Laptop, 8 GB VRAM) three concurrent torch processes
exhaust the Windows paging file and a worker dies with `WinError 1455` — which surfaces as a
traceback from `multiprocessing.spawn`, not as anything that mentions memory.

Two concurrent processes were used routinely and are safe. The scripts that matter here are
`exp02_leakage_effect.py`, `exp02b_notebook_reproduction.py`, `exp04_backbone_comparison.py`,
`exp06_calibrate_bridge.py` and `exp09_class_weighted.py`.

**`exp07_prevalence_bridge.py` also counts**, even though it trains nothing: it imports the vision
package for `CLASSES`, so it loads torch too. It is CPU-bound and long (roughly 45-90 minutes), so
the natural order is to start it and leave the GPU idle rather than to overlap a training run with
it.

A crash costs only the run in progress. Every experiment writes its table after each run, and
Experiment 02b keys its rows by condition, so re-running the same command restores what was lost
without touching what succeeded.

---

## 11. Reading the code

Start here, in order:

1. **`src/shrimp_risk/vision/splits.py`** — the project's central claim lives here
2. **`src/shrimp_risk/econometrics/just_pope.py`** — the risk estimator and its preconditions
3. **`src/shrimp_risk/vision/trainer.py`** — training, model selection, single test evaluation

Every module opens with a docstring stating the problem it solves before the code that solves it.
Full per-module reference: **`docs/module_documentation.md`**.

**Rules the code enforces on you, on purpose:**

- Splits **group on `specimen_id`**. Image-level splitting exists only to measure its own damage.
- The test set is evaluated **once**, after model selection.
- Every run records seed, library versions and GPU. A result without an environment is not a result.
- Experiments **refuse to run** if a fold leaks.
- Figures are **generated**, never hand-made.

---

## 12. Project documentation map

| Read this | For |
|---|---|
| `CLAUDE.md` | Current state, binding constraints, key numbers — **start here** |
| `docs/MASTER_DOCUMENT.md` | Cumulative record; authoritative on the present state |
| `docs/00_MASTER_PLAN.md` | Forward-looking: stages, risks, framework conflicts |
| `docs/decisions/decision_log.md` | 33 entries under 31 ids (D-020 and D-021 are each used twice), each with a reversal condition |
| `docs/audits/` | Project audits. `nemotron-audit.md` is **EXTERNAL and UNVERIFIED** — read its header before believing anything in it, and **do not run the commands in its verification section**: two of them overwrite stored experiment artefacts (`D-031`) |
| `docs/stages/` | One document per stage, each ending in a Faculty Challenge Test |
| `docs/presentation/ppt_content.md` | Review-2 slide content (27 slides) |
| `docs/presentation/speaker_notes.md` | What to say, and what never to say |
| `docs/module_documentation.md` | Per-module reference |
| `docs/research/literature_review.md` | 24 papers with verification status |

**Not a git repository.** There is no branch, commit or PR flow.

**The `.pptx` is deliberately not generated.** Slide *content* lives in
`docs/presentation/ppt_content.md`; build the deck from it by hand.


## 12. Stage 10-12 scripts

The RUNBOOK previously stopped at Stage 9. These ten scripts a human needs to run were
undocumented; `predict_farm.py` in particular is the Review-3 demonstration itself.

| Command | What it does | Runtime |
|---|---|---|
| `scripts/exp02b_notebook_reproduction.py` | Faithful replay of the inherited notebook pipeline. `--full-res` adds the last departure. | ~10 min GPU |
| `scripts/exp09_class_weighted.py` | Black Gill cost-sensitive ablation, 5 folds. | ~1 h GPU |
| `scripts/exp10_lr_sweep.py` | Per-backbone learning-rate sweep (partial; see R17). | ~1 h GPU |
| `scripts/exp10_track_a_repairs.py` | Specification, cluster-robust SEs, Breusch-Pagan power. | ~3 min CPU |
| `scripts/exp11_external_validation.py` | Hasan & Haque external validation. | ~5 min CPU |
| `scripts/extract_source_texts.py` | Regenerates `data/interim/extracted_text/` from the supplied inputs. | seconds |
| **`scripts/train_edge_checkpoint.py`** | **Trains the deployment backbone on fold 0 and writes `reports/models/<backbone>_best.pt`. Must run BEFORE the export.** | **~90 s GPU** |
| `scripts/export_edge_model.py` | ONNX FP32 + static INT8 export, parity check, latency benchmark, and held-out accuracy scoring. Refuses to run without a checkpoint unless given `--allow-untrained`. | ~2 min CPU |
| **`scripts/predict_farm.py`** | **The Review-3 live demonstration.** Takes images or a directory, runs ONNX inference, aggregates to specimens, quantifies prevalence via SLD-EM, maps to farm risk in BDT. | seconds |
| `scripts/build_pptx_review3.py` | Renders `docs/presentation/ppt_content_review3.md` to the Review-3 deck. | seconds |
| `scripts/build_report_docx.py` | Builds the bound report, reading every table from `reports/tables/`. | seconds |

### The edge pipeline, in order

```bash
.venv/Scripts/python.exe scripts/train_edge_checkpoint.py      # writes the checkpoint
.venv/Scripts/python.exe scripts/export_edge_model.py          # exports, quantises, SCORES
.venv/Scripts/python.exe scripts/predict_farm.py --images-dir <dir> --farm-area 2.5
```

The export step scores every graph it produces on the held-out fold. That check exists because an
untrained network and a trained one have identical size and latency, so an export pipeline that
measures only those two cannot tell them apart -- which is how a randomly initialised model was
once shipped as the deployment artefact (`D-044`).

### Demonstration notes

`predict_farm.py` errors and exits non-zero if the images you named cannot be read. It generates
synthetic input only when `--synthetic N` is passed explicitly, and labels that output as random
pixels. Specimen grouping is parsed from the filename (`Healthy-1-img-3.jpg` -> `Healthy-1`), so
a tray of thirty photographs of ten shrimp is counted as ten observations, not thirty.
