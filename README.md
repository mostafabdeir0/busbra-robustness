# BUS-BRA breast ultrasound robustness study

**Authors:** Mostafa Bdeir and Hadi Fardan

This completed research project studies how Logistic Regression and RBF SVM
classification changes under lesion-boundary uncertainty. It uses 23 handcrafted
shape, intensity and masked-texture features, patient-level splitting and nested
cross-validation. The final calibrated models were frozen before a single
held-out evaluation.

## Final report

- [Final report (PDF)](reports/submission_v2/BUS_BRA_Report.pdf)

Submitted notebook: notebooks/BUS_BRA_submission_executed.ipynb (executed in full mode).

- [Markdown version](reports/submission_v2/BUS_BRA_Report.md)
- [HTML version](reports/submission_v2/BUS_BRA_Report.html) (download to view)
- [Figures](reports/submission_v2/figures) and [report tables](reports/submission_v2/tables)
- Original run report (superseded): [PDF](reports/final_step22_run2/BUS_BRA_Final_Report.pdf), with its [reporting provenance](reports/final_step22_run2/report_manifest.json)

## Main findings

The dataset contains 1,875 images from 1,064 patients. Development used 1,514
images / 851 patients; the locked test used 361 images / 213 patients.

| Held-out clean metric | Logistic Regression | RBF SVM |
|---|---:|---:|
| Balanced accuracy | 80.42% | 81.85% |
| Sensitivity | 69.12% | 70.59% |
| Specificity | 91.72% | 93.10% |
| ROC AUC | 0.9177 | 0.9142 |
| Brier score | 0.1024 | 0.1030 |
| Log loss | 0.3296 | 0.3397 |

The primary SVM balanced-accuracy pointwise 95% interval was **75.96%-87.43%**.
Both models use a fixed calibrated-probability threshold of 0.5. SVM missed 20
of 68 malignant test patients; this is a research evaluation, not clinical
validation or a deployment recommendation.

- Original mild erosion/dilation losses were strongly sensitive to the boundary
  perturbation algorithm. A separately labelled coherent sensitivity analysis
  substantially reduced those losses on development data.
- The frozen stability-feature-selection procedure did not improve the overall
  nested development objective. Its negative result is preserved.
- SVM had slightly higher held-out clean balanced accuracy, sensitivity and
  specificity, while LR had slightly better AUC and probability losses. These
  results do not establish uniform SVM superiority.
- Some coherent edits increased balanced accuracy while AUC or probability losses
  worsened. The report includes the full paired analysis and limitations.

## Code and validation

`src/` contains the data/split checks, feature extraction, baseline classifiers,
boundary perturbations, robustness evaluation, uncertainty analysis, nested
stability selection, patient-level calibration, locked held-out evaluation and
read-only final reporting. `tests/` contains **198 passing automated tests**.

The study used Python 3.14.3. Dependency versions are pinned in
`requirements.txt`. To create an environment and run the synthetic test suite:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite uses synthetic fixtures and does not require downloading BUS-BRA.
Actual experiment reproduction requires the dataset and generated checkpoints.
The final reporting command, when those local checkpoints are available, is:

```powershell
.\.venv\Scripts\python.exe -m src.final_report
```

The reporting tool reads saved predictions; it performs no fitting or new
inference. The held-out evaluator separately enforces the frozen model/protocol
hashes and a single-evaluation marker. Do not use held-out outcomes to change
features, models or thresholds.

## Dataset and retained artifacts

Download BUS-BRA version 1.0 from the [official Zenodo record](https://zenodo.org/records/8231412)
and place the original archive at `data/raw/BUSBRA.zip` for experiment reproduction.
The repository includes the code, tests, final report, figures, report tables, the
frozen model and the derived development feature tables. The raw dataset, Python
environments and the full original checkpoint folders are not included.

Required dataset citation:

Gomez-Flores W, Gregorio-Calas MJ, Pereira WCA. **BUS-BRA: A Breast Ultrasound
Dataset for Assessing Computer-aided Diagnosis Systems.** *Medical Physics*.
2024;51:3110-3123. [DOI: 10.1002/mp.16812](https://doi.org/10.1002/mp.16812).

## Project status

All 22 planned scientific/reporting steps are complete. No additional model
experiment is planned.

## Submission supplement (2026-11-20)

New in this revision, all additive (no frozen file was changed):

| Item | Where |
|---|---|
| **Final report** (PDF, Markdown, HTML, tables, figures) | [reports/submission_v2](reports/submission_v2/BUS_BRA_Report.pdf) |
| **Trained model** (three-member calibrated LR and SVM ensembles, 0.22 MiB) and its [model card](models/MODEL_CARD.md) | [models/frozen_calibrated_models.joblib](models/frozen_calibrated_models.joblib) |
| **Notebook** covering the whole workflow, plus [an executed copy with all outputs](notebooks/BUS_BRA_submission_executed.ipynb) (full mode, run from the raw archive) | [notebooks/BUS_BRA_submission.ipynb](notebooks/BUS_BRA_submission.ipynb) |
| Verified run order and timings | [RUN_ORDER.md](RUN_ORDER.md) |
| Experiment protocol summary | [experiment_protocol.md](experiment_protocol.md) |
| Derived (non-image) development feature tables, so the notebook can retrain without the raw archive | [derived_features/](derived_features) |
| Scripts for the added development-only tables and the report build | [analysis/](analysis) |
| Logs of the independent rerun | [reproduction_logs/](reproduction_logs) |

**Model integrity.** `models/frozen_calibrated_models.joblib` has SHA-256 `627ef9df92e4b12b475a12c844b53e82e9dd7fefe05a9f07c3b94ca7c77f3586`, the digest recorded for the original frozen model in `reports/final_step22_run2/report_manifest.json`. The development pipeline was rerun from the raw archive on 2026-11-20 with the pinned versions and regenerated a byte-identical file; every deterministic development result in the report was reproduced exactly (timings vary between runs). The held-out evaluation was not rerun.

**Quick start for reviewers.** Install `requirements.txt`, then open the notebook. The submitted copy was executed in `full` mode from `data/raw/BUSBRA.zip` (about 8 minutes). Set `RUN_MODE = "saved"` to run it from this repository only (seconds), or `"derived"` to retrain from `derived_features/` (about 3 minutes). The 198 unit tests (`python -m pytest -q`) use synthetic fixtures and need no data.
