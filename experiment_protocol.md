# Experiment protocol

This file summarises the protocol the study followed. The authoritative machine-readable versions are `models/final_evaluation_protocol.json` (final protocol) and the design objects in `src/stable_selection.py` (`locked_protocol`) and `src/patient_calibration.py` (`calibration_design`). Deviations from the original proposal are tabulated in Section 3.7 of the report.

## Question
Do nonlinear classifiers improve breast-lesion classification while becoming more sensitive to imperfect lesion boundaries, and can stable-feature selection reduce that sensitivity without sacrificing clean-data performance? The task is benign versus malignant classification of an already outlined lesion; it is not whole-image detection.

## Data and splits
BUS-BRA v1.0 (1,875 images, 1,064 patients, 722 benign and 342 malignant patients). Patient identifiers, histology, BI-RADS and scanner metadata are not model inputs. Seed 42. An 80/20 patient-stratified split gives 851 development patients (1,514 images) and 213 test patients (361 images). Development patients are divided into five locked outer folds. All splitting is by patient. The split files in `splits/` are locked and verified by `src.splits`.

## Features
23 fixed handcrafted descriptors per image: 6 shape, 7 intensity, 10 masked texture (gray-level co-occurrence at distances 1 and 3, 32 levels, four directions averaged, pairs counted only inside the lesion). Images are converted to grayscale and resized to a longest side of 256 pixels; masks use nearest-neighbour interpolation.

## Models
Level 1: L2 logistic regression, C in {0.01, 0.1, 1, 10, 100}, class balancing off or on (10 candidates). Level 2: RBF SVM, C in {0.1, 1, 10}, gamma in {scale, 0.01, 0.1}, class balancing off or on (18 candidates). Identical inputs. Each patient contributes total training weight one across their views; patient scores average image scores. Scaling and fitting use training folds only.

## Validation
Nested patient-grouped cross-validation: five locked outer folds, three inner patient-stratified folds. The original inner objective is patient balanced accuracy, with AUC and the simpler setting as tie-breakers. Reported metrics: balanced accuracy, sensitivity, specificity, ROC AUC, Brier score, log loss, false-negative counts, confidence reliability.

## Boundary-sensitivity experiment
Trained models and thresholds stay fixed; only the mask changes. Eight predefined development conditions: mild and moderate erosion and dilation (target IoU about 0.95 and 0.90) and four shifts (about 0.90). Original construction: boundary pixels removed or added in order of distance to the contour with a pseudo-random tie-break. After unexpectedly large mild-condition losses, a post-hoc coherent construction (single connected patch near a contour anchor, nested mild and moderate edits) was added; both are reported. The held-out evaluation uses the four coherent erosion and dilation conditions.

## Stability feature selection
Ranked inside each training partition only. Score = maximum over the four coherent conditions of the mean absolute feature change (averaged within patient, then across patients) divided by the clean training weighted standard deviation; lower is better. Subset sizes 5, 10 and 15 versus a matched all-23 comparator with identical grids and objective. Inner objective: mean over three inner folds of the equal mean of five patient balanced accuracies (clean plus four coherent), so clean performance has weight 0.2. Tie-breaks are fixed in the protocol. No global feature list is used.

## Calibration and final models
Patient-level sigmoid calibration (StandardScaler then LogisticRegression, C=1) in a three-member ensemble per classifier. Each member reserves one of three patient folds for calibration, tunes on the other two with three inner folds, and fits on clean base images only. Scores are the logit of the mean LR image probability or the mean SVM margin. Prediction averages the three calibrated patient probabilities. The decision threshold is fixed at 0.5 and is not tuned. Calibration split seeds are 2020 plus a scope code and base-tuning seeds are 3000 plus 10 times the scope code plus the member index.

## Held-out evaluation
One evaluation of the frozen development-trained models on the 213 test patients, with clean and four coherent conditions. Uncertainty: 10,000 paired patient bootstrap replicates within diagnosis, seed 42, pointwise 95% percentile intervals, conditional on the fixed fitted models. The protocol requires the frozen model, protocol and source hashes to match and creates an exclusive marker before any test image is opened, so the evaluation cannot be repeated or altered. No threshold, feature or model changes were permitted after test results.

## Status
All planned experiments are complete. Development stages were independently regenerated on 2026-11-20 and reproduced the published development numbers and the final model file exactly (`RUN_ORDER.md`, report Section 6). Future work needs a separately documented evaluation plan and must not reuse the test set for tuning.
