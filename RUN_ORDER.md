# Run order (verified)

Every command below was run on 2026-11-20 from a clean copy of this repository, with the raw archive `BUSBRA.zip` (MD5 `1f8b2be6476d58fc97bfb5e5a1ea9bab`) and the pinned library versions, and every one exited cleanly. Timings are from that run on one Windows 11 machine. Logs are in `reproduction_logs/`.

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
mkdir data\raw          # put BUSBRA.zip here (https://zenodo.org/records/8231412)
```

Run each command from the repository root as `.\.venv\Scripts\python.exe -m <module> ...`. The stages write into `data/processed/` and `results/`, which are git-ignored.

## Development pipeline (this is what the notebook runs)

| # | Command | Purpose | Time |
|---|---|---|---|
| 1 | `src.data` | Audit the archive (pairs, masks, labels) | seconds |
| 2 | `src.splits` | Verify the locked patient-level splits in `splits/` against the archive | seconds |
| 3 | `src.features --all-development` | 23 clean features per development image | 7 s |
| 4 | `src.models` | Logistic Regression, nested patient CV | 4 s |
| 5 | `src.svm_model` | RBF SVM, nested patient CV | 9 s |
| 6 | `src.perturbations --all-development` | Original eight-condition masks (geometry audit) | 47 s |
| 7 | `src.perturbed_features --all-development` | Features for those masks | 81 s |
| 8 | `src.robustness_eval` | Original stress test, paired development results | 3 s |
| 9a | `src.coherent_perturbations --all-development` | Coherent boundary edits (post hoc) | 42 s |
| 9b | `src.coherent_features --all-development` | Features for the coherent masks | 38 s |
| 10 | `src.coherent_eval` | Coherent stress test | 2 s |
| 11 | `src.stable_selection` | Freeze the stability-selection design and audit training scopes | 2 s |
| 12 | `src.stable_evaluation` | Nested stability-selection evaluation vs matched all-23 arm | 65 s |
| 13 | `src.patient_calibration` | Calibrated nested CV and the final development-only ensembles | 42 s |

Total about six minutes.

Check the result: the numbers printed by stages 4, 5, 12 and 13 must equal the development numbers in the report (LR 86.89% / AUC 0.9304, SVM 87.07% / 0.9352; stability arms 87.35/84.55 and 87.07/86.87 clean BA; calibrated outer CV 84.16% and 84.73%). The regenerated model file `results/robustness/calibration_step20/frozen_calibrated_models.joblib` has SHA-256 `627ef9df92e4b12b475a12c844b53e82e9dd7fefe05a9f07c3b94ca7c77f3586`, the same as `models/frozen_calibrated_models.joblib`.

## Stages that are not part of the reproduction

| Stage | Why it is not run |
|---|---|
| `src.heldout_evaluation` | **The locked one-shot test-set evaluation. Never re-run.** It was run once during the study; it enforces frozen model, protocol and source hashes and an exclusive marker file. Its saved per-patient probabilities are in `reports/final_step22_run2/patient_failure_audit.csv` |
| `src.final_report` | Builds the original report from the saved held-out outputs; needs the original local checkpoints |
| `src.robustness_uncertainty` | Optional development bootstrap intervals (needs the coherent results at a specific output name) |
| `src.visual_audit`, `src.feature_audit`, `src.outlier_review`, `src.robustness_diagnostics` | Manual or diagnostic audits from the original study; optional |

## Notebook

`notebooks/BUS_BRA_submission.ipynb` runs the whole workflow in three modes (`RUN_MODE`): `saved` (seconds, repository only), `derived` (about 3 minutes, retrains from `derived_features/`), `full` (about 8 minutes, from the raw archive). None of them runs the held-out evaluation.

## Report tables and PDF

```powershell
python analysis/build_report_tables.py --root <rebuild root> --out reports/submission_v2/tables
python analysis/build_report.py --dir reports/submission_v2
```

The first command reads development outputs only and writes every table not in the original report; the second fills the report template and prints the PDF with a locally installed Edge or Chrome.

## Original step labels

The original study used internal step numbers (14 to 22). They map to the commands above as: 14 = stage 8, 16 = stage 10, 17 = `src.robustness_uncertainty`, 18 = stage 11, 19 = stage 12, 20 = stage 13, 21 = `src.heldout_evaluation`, 22 = `src.final_report`.
