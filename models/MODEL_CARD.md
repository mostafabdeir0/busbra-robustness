# Model card: `frozen_calibrated_models.joblib`

**What it is.** The final patient-calibrated ensembles for the BUS-BRA boundary-robustness study: three members each for Logistic Regression and RBF SVM, all trained on the 851 development patients only, on all 23 handcrafted features, with a sigmoid calibrator per member and a fixed decision threshold of 0.5.

**Integrity.** SHA-256 `627ef9df92e4b12b475a12c844b53e82e9dd7fefe05a9f07c3b94ca7c77f3586`. This is the digest recorded for the original frozen model in the original report manifest (`reports/final_step22_run2/report_manifest.json`). On 2026-11-20 the development pipeline was rerun from the raw archive with the pinned library versions and regenerated a file with this identical digest.

**Companion files.** `final_evaluation_protocol.json` (frozen protocol, records the model digest and source hashes) and `calibration_summary.json` (library versions the model was created with).

**Contents.** `features` (the 23 names, in order), `design`, `development_case_sha256`, and `models` with `logreg` and `svm` ensembles. Selected settings per member. LR: (C=0.1, balanced), (C=10, unbalanced), (C=0.1, balanced). SVM: (C=1, gamma=scale, balanced), (C=10, gamma=0.01, balanced), (C=1, gamma=0.01, balanced). Final SVM ensemble: 1,303 support vectors; LR ensemble: 69 base coefficients. File size 0.22 MiB.

**Load and predict** (from the repository root, with the pinned requirements installed):

```python
import hashlib, joblib, pandas as pd
from src.patient_calibration import predict_ensemble
from src.models import load_development

path = "models/frozen_calibrated_models.joblib"
assert hashlib.file_digest(open(path, "rb"), "sha256").hexdigest() == "627ef9df92e4b12b475a12c844b53e82e9dd7fefe05a9f07c3b94ca7c77f3586"
bundle = joblib.load(path)   # only load files whose digest you have verified
dev = load_development("derived_features/development_features.csv", "splits/image_splits.csv")
rows = dev[dev.Case == dev.Case.iloc[0]]          # all images of one development patient
print(predict_ensemble(bundle["models"]["svm"], rows)[["prob_malignant", "predicted"]])
```

Input must be the 23 features for every image of a patient (one row per image); the ensemble aggregates image scores to one patient-level probability.

**Intended use.** Reproducing the study and its held-out numbers. **Not for clinical use.** The 0.5 threshold was not chosen for clinical sensitivity, and the SVM missed 20 of 68 malignant patients in the held-out cohort. The model needs a lesion mask; it has not been evaluated with an automatic segmenter or on any other dataset.

**Compatibility.** Created with Python 3.14.3 and the versions in `calibration_summary.json` (scikit-learn 1.9.1, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, joblib 1.6.0). Other versions may refuse to load or give different results. Only unpickle files you trust.

**About the two JSON files.** `final_evaluation_protocol.json` and `calibration_summary.json` are the copies regenerated in the 2026-11-20 rerun. Local folder paths were replaced with `<local-path-redacted>`; hashes, settings and library versions are unchanged. Because of that edit, the hash cross-references between these two files no longer verify byte for byte, which does not matter for loading the model: the notebook checks only the model digest and the recorded library versions. Do not use them with `src.heldout_evaluation` (that evaluation must not be run again).
