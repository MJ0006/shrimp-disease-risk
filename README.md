# Shrimp disease detection and farm-level economic risk

Source code for an M.Tech Integrated mini project at VIT Chennai, linking image-based
shrimp disease classification to a farm-level production risk model.

**G Mohan Jayanth** (19MIS1127) · Guide: **Dr. M. Premalatha** · School of Computer Science
and Engineering, Vellore Institute of Technology, Chennai.

## What this is

Two independent tracks joined at the decision level:

- **Track B — vision.** A classifier over 1,149 photographs of 416 individual shrimp, split so
  that no animal appears in both training and test partitions. An INT8 ONNX graph of 4.46 MB is
  exported for edge use.
- **Track A — econometrics.** A Just–Pope production risk model fitted to a 1,221-farm survey
  panel of coastal Bangladeshi shrimp farms.
- **The bridge.** Per-image posteriors are aggregated to specimen level and converted into a
  pond-level prevalence estimate by the SLD-EM quantification algorithm, and that prevalence is
  what enters the economic model.

The two tracks are joined at the decision level, not the sample level. No dataset pairs a
photograph of a shrimp with the accounts of the farm that animal came from, so no sample-level
fusion is possible and none is claimed.

## Why the splitting matters

The image corpus contains 1,149 photographs of 416 animals, about 2.76 shots each. Splitting on
images rather than animals places roughly 88% of test specimens in the training set under a
different camera angle. Measured on this corpus, that inflates macro-F1 from 0.6721 to 0.8215.
Every split in this repository groups on specimen identity; `scripts/exp01_leakage_audit.py`
measures the exposure under both protocols.

## Running it

`RUNBOOK.md` is the operating manual: environment setup, data acquisition, every command with
its expected output, and a troubleshooting table.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/fetch_ifpri.py        # survey data
.venv/Scripts/python.exe -m pytest                     # 269 tests
.venv/Scripts/python.exe scripts/demo_results.py       # every result, ~5 s
```

`demo_results.py` labels each block `LIVE` or `RECORDED`, so it is always visible which figures
were computed at that moment and which were replayed from a stored run. Training outcomes are
replayed; they take 20–45 minutes on a GPU.

## Layout

| Path | Contents |
|---|---|
| `src/shrimp_risk/vision/` | datasets, splitting, backbones, training loop |
| `src/shrimp_risk/econometrics/` | Just–Pope estimator, step-3 models, diagnostics |
| `src/shrimp_risk/integration/` | quantification, calibration, pools, sensitivity |
| `scripts/` | one executable entry point per job, including all experiments |
| `tests/` | 269 tests: splits, determinism, econometrics, calibration, artefacts |
| `reports/tables/`, `reports/figures/` | every number and chart the project reports |
| `experiments/runs/` | per-run JSON with environment fingerprints |

## Data

Not included here. All three sources are openly licensed and are fetched from their own
repositories by the scripts above:

- ShrimpDiseaseBD photographic corpus (CC BY)
- IFPRI Bangladesh aquaculture panel survey, 2023 baseline and 2024 endline (IFPRI Terms of Use,
  unrestricted public access)
- Hasan & Haque (2020) White Spot Disease survey, *Data in Brief* (CC BY)

## What is not here

The thesis document and the draft manuscript are excluded while the report is under
institutional plagiarism screening. The notebook is excluded because it is generated;
`scripts/build_notebook.py` writes and executes it from the artefacts in this repository.

## Results worth knowing before reading the code

Several of this project's findings are negative, and the code reflects that rather than hiding it:

- No step-3 machine-learning model beats predicting the mean on the income risk equation; the
  best ratio is 1.014, where above 1.0 is worse than the mean.
- Temperature-scaling the classifier makes the prevalence estimator worse, not better.
- INT8 quantisation costs 0.104 macro-F1, from 0.6989 to 0.5951. It is reported, not absorbed.
- The per-backbone learning-rate sweep is partial, so the backbone ranking is provisional.

## Licence

Code is released under the MIT Licence. The datasets carry their own licences, named above.
