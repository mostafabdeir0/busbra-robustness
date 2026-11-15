# BUS-BRA breast ultrasound robustness study

**Authors:** Mostafa Bdeir and Hadi Fardan

This completed research project studies how Logistic Regression and RBF SVM
classification changes under lesion-boundary uncertainty. It uses 23 handcrafted
shape, intensity and masked-texture features, patient-level splitting and nested
cross-validation. The final calibrated models were frozen before a single
held-out evaluation.

## Final report

- [Final PDF report](reports/final_step22_run2/BUS_BRA_Final_Report.pdf)
- [Editable Markdown report](reports/final_step22_run2/BUS_BRA_Final_Report.md)
- [Standalone HTML report](reports/final_step22_run2/BUS_BRA_Final_Report.html) (download to view)
- [Figures in PNG and SVG](reports/final_step22_run2/figures)
- [Reporting provenance](reports/final_step22_run2/report_manifest.json)

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
The publication includes final code, tests, report, figures and report tables.
The raw dataset, Python environments, processed feature caches and full training/
evaluation checkpoint folders remain local and are not included in this upload.
Keep a separate complete project backup, including the frozen models and test marker.

Required dataset citation:

Gomez-Flores W, Gregorio-Calas MJ, Pereira WCA. **BUS-BRA: A Breast Ultrasound
Dataset for Assessing Computer-aided Diagnosis Systems.** *Medical Physics*.
2024;51:3110-3123. [DOI: 10.1002/mp.16812](https://doi.org/10.1002/mp.16812).

## Project status

All 22 planned scientific/reporting steps are complete. Remaining handoff:
author review, course-specific formatting/submission and preservation of the
complete reproducibility artifacts. No additional model experiment is planned.
