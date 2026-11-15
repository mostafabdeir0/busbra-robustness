"""Step 14: held-out-outer-fold robustness against synthetic lesion-mask changes.

From repository root:
    python -m src.robustness_eval

Reuses LOCKED inner-selected hyperparameters recorded by Steps 8 and 9.
Refits each outer training fold using *clean* data only, reproduces the
saved clean OOF scores, then applies those fitted models to unchanged-image,
perturbed-mask features of the SAME outer validation patients.

For a perturbation condition, a patient is eligible ONLY when every one of
that patient's images has a valid perturbed mask. Ineligible patients are
retained in the prediction output with no perturbed score; paired metrics are
computed on the matching eligible patients. No held-out test data are read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from .features import FEATURE_COLUMNS
from .models import ROOT, NUM_OUTER_FOLDS, fit_model, load_development, patient_predictions
from .perturbed_features import INFO_COLUMNS, VARIANT_NAMES
from .svm_model import fit_svm, patient_svm_predictions

ID_COLUMNS = ("ID", "Case", "Pathology", "split", "dev_fold")
MODEL_NAMES = ("logreg", "svm")


def parse_boolean_column(series: pd.Series) -> pd.Series:
    """Parse CSV bools explicitly: astype(bool) would make 'False' truthy."""
    values = series.astype("string").str.strip().str.lower()
    if not values.isin(["true", "false"]).all():
        raise ValueError("Expected only True/False in perturbed-feature 'valid' column")
    return values.eq("true").astype(bool)


def load_perturbations(path: Path, clean: pd.DataFrame) -> pd.DataFrame:
    """Require one valid/unavailable row for EVERY development ID and condition."""
    pert = pd.read_csv(path, dtype={"ID": str}, low_memory=False)
    expected = list(INFO_COLUMNS) + list(FEATURE_COLUMNS)
    if list(pert.columns) != expected:
        raise ValueError("Perturbed feature table schema/order differs from Step 13")
    if len(pert) != len(clean) * len(VARIANT_NAMES):
        raise ValueError("Missing/extra perturbation condition records")
    if pert[list(ID_COLUMNS) + ["condition", "valid"]].isna().any().any():
        raise ValueError("Missing perturbation ID, label, condition, or availability")
    if pert.duplicated(["ID", "condition"]).any():
        raise ValueError("Duplicate ID-condition pair in perturbed features")
    if set(pert["condition"]) != set(VARIANT_NAMES):
        raise ValueError("Unexpected/missing perturbation condition")
    if not pert.groupby("ID")["condition"].nunique().eq(len(VARIANT_NAMES)).all():
        raise ValueError("An image does not have all eight perturbations")
    if not pert["split"].eq("development").all():
        raise ValueError("Non-development rows in perturbed feature table")
    if set(pert["ID"]) != set(clean["ID"]):
        raise ValueError("Perturbation images differ from clean development images")

    # Metadata must agree with the frozen split, not just be plausible.
    metadata = clean[list(ID_COLUMNS)].set_index("ID")
    compare = pert[list(ID_COLUMNS)].set_index("ID")
    for column in ID_COLUMNS[1:]:
        expected_values = metadata.loc[compare.index, column].to_numpy()
        if not np.array_equal(compare[column].to_numpy(), expected_values):
            raise ValueError(f"Perturbation metadata differs from clean manifest: {column}")

    pert["valid"] = parse_boolean_column(pert["valid"])
    cols = list(FEATURE_COLUMNS)
    matrix = pert[cols].to_numpy(dtype=float)
    v = pert["valid"].to_numpy(dtype=bool)
    if not np.isfinite(matrix[v]).all():
        raise ValueError("Valid perturbation has missing/nonfinite features")
    if not np.isnan(matrix[~v]).all():
        raise ValueError("Unavailable perturbation contains fabricated numeric features")
    if pert.loc[v, "actual_iou"].isna().any() or not pert.loc[~v, "actual_iou"].isna().all():
        raise ValueError("IoU missing for valid variant or present for unavailable variant")
    if not pert.loc[v, "actual_iou"].between(0, 1).all():
        raise ValueError("Invalid perturbation IoU")
    if len(pert) != len(clean) * len(VARIANT_NAMES):
        raise AssertionError("Perturbation coverage changed unexpectedly")
    return pert


def load_fold_settings(path: Path, model_name: str) -> dict[int, dict]:
    fold_table = pd.read_csv(path)
    if ("outer_fold" not in fold_table or "C" not in fold_table
            or "balance" not in fold_table):
        raise ValueError(f"Missing recorded {model_name} hyperparameters")
    if fold_table["outer_fold"].duplicated().any() or set(fold_table.outer_fold) != set(range(NUM_OUTER_FOLDS)):
        raise ValueError(f"Expected one {model_name} setting per outer fold")
    if model_name == "svm" and "gamma" not in fold_table:
        raise ValueError("Recorded SVM settings missing gamma")
    parsed = {}
    for r in fold_table.to_dict("records"):
        fold = int(r["outer_fold"])
        if not np.isfinite(float(r["C"])) or float(r["C"]) <= 0:
            raise ValueError("Invalid regularization C")
        b = str(r["balance"]).strip().lower()
        if b not in ("true", "false"):
            raise ValueError("Invalid class-balance setting")
        setting = {"C": float(r["C"]), "balance": b == "true"}
        if model_name == "svm":
            gamma = str(r["gamma"]).strip()
            if gamma != "scale":
                gamma = float(gamma)
                if not np.isfinite(gamma) or gamma <= 0:
                    raise ValueError("Invalid SVM gamma")
            setting["gamma"] = gamma
        parsed[fold] = setting
    return parsed


def validate_clean_reference(recomputed: pd.DataFrame, reference_path: Path,
                             model_name: str) -> None:
    """Fail closed if refitting does not reproduce saved Step 8/9 clean OOF."""
    previous = pd.read_csv(reference_path)
    if (len(previous) != len(recomputed) or previous.Case.duplicated().any()
            or set(previous.Case) != set(recomputed.Case)):
        raise ValueError(f"Saved {model_name} OOF patient coverage differs")
    left = recomputed.sort_values("Case").reset_index(drop=True)
    right = previous.sort_values("Case").reset_index(drop=True)
    for column in ("Case", "actual", "predicted", "outer_fold", "n_images"):
        if column not in right or not np.array_equal(left[column], right[column]):
            raise ValueError(f"Saved {model_name} OOF mismatch: {column}")
    score_key = "prob_malignant" if model_name == "logreg" else "decision_score"
    if score_key not in right or not np.allclose(
        left[score_key].to_numpy(float), right[score_key].to_numpy(float), rtol=1e-6, atol=1e-7
    ):
        raise ValueError(f"Saved {model_name} OOF score mismatch; do not continue with robustness")


def eligible_patient_ids(validation: pd.DataFrame, variants: pd.DataFrame) -> set:
    """Exclude the whole patient if even one of their images is unavailable."""
    if set(validation.ID) != set(variants.ID):
        raise ValueError("Validation image coverage differs between clean and perturbed")
    missing_ids = set(variants.loc[~variants.valid, "ID"])
    missing_cases = set(validation.loc[validation.ID.isin(missing_ids), "Case"])
    return set(validation.Case) - missing_cases


def metrics(df: pd.DataFrame, score_col: str, pred_col: str) -> dict:
    if df.empty or not set(df["actual"]) == {0, 1}:
        raise ValueError("Paired metric sample requires both diagnoses")
    y = df["actual"].to_numpy(dtype=int)
    p = df[pred_col].to_numpy(dtype=int)
    s = df[score_col].to_numpy(dtype=float)
    if not np.isfinite(s).all():
        raise ValueError("Nonfinite patient score")
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, p, labels=[0, 1]).ravel())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {"balanced_accuracy": float((sensitivity + specificity) / 2),
            "sensitivity": float(sensitivity), "specificity": float(specificity),
            "roc_auc": float(roc_auc_score(y, s)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _predict(model, frames: pd.DataFrame, model_name: str, fold: int) -> pd.DataFrame:
    return (patient_predictions(model, frames, fold=fold) if model_name == "logreg"
            else patient_svm_predictions(model, frames, outer_fold=fold))


def compare_one_condition(model, model_name: str, validation: pd.DataFrame,
                          pert_one: pd.DataFrame, fold: int) -> pd.DataFrame:
    """One row per held-out patient. Missing variants yield NA, never fallback."""
    baseline = _predict(model, validation, model_name, fold)
    score_key = "prob_malignant" if model_name == "logreg" else "decision_score"
    eligible_cases = eligible_patient_ids(validation, pert_one)
    out = baseline[["Case", "actual", "outer_fold", "n_images", score_key, "predicted"]].copy()
    out = out.rename(columns={score_key: "clean_score", "predicted": "clean_predicted"})
    out["eligible"] = out.Case.isin(eligible_cases)
    out["perturbed_score"] = np.nan
    out["perturbed_predicted"] = np.nan
    out["mean_iou"] = np.nan
    if eligible_cases:
        imgs = pert_one.loc[pert_one.Case.isin(eligible_cases)].copy()
        if not imgs.valid.all():
            raise AssertionError("Unavailable image would be evaluated")
        pert_pred = _predict(model, imgs, model_name, fold)
        elig_clean = validation.loc[validation.Case.isin(eligible_cases)]
        clean_counts = elig_clean.groupby("Case").size().sort_index()
        pert_counts = imgs.groupby("Case").size().sort_index()
        if not clean_counts.equals(pert_counts):
            raise AssertionError("Eligible patients must have all original image views")
        pert_pred = pert_pred.set_index("Case")
        ious = imgs.groupby("Case")["actual_iou"].mean()
        ix = out["eligible"]
        cases = out.loc[ix, "Case"]
        out.loc[ix, "perturbed_score"] = pert_pred.loc[cases, score_key].to_numpy(float)
        out.loc[ix, "perturbed_predicted"] = pert_pred.loc[cases, "predicted"].to_numpy(int)
        out.loc[ix, "mean_iou"] = ious.loc[cases].to_numpy(float)
    out["condition"] = pert_one.condition.iloc[0]
    out["model"] = model_name
    out["missing_reason"] = np.where(out["eligible"], "", "at_least_one_image_mask_unavailable")
    return out[["model", "condition", "Case", "actual", "outer_fold", "n_images", "eligible",
                "missing_reason", "mean_iou", "clean_score", "perturbed_score",
                "clean_predicted", "perturbed_predicted"]]


def summarize_pair(preds: pd.DataFrame, *, model: str, condition: str, fold=None) -> dict:
    n_total = len(preds)
    if preds.Case.duplicated().any():
        raise ValueError("Repeated patient in paired predictions")
    paired = preds.loc[preds.eligible].copy()
    if paired.empty:
        raise ValueError("No eligible validation patients")
    if paired[["clean_score", "perturbed_score", "clean_predicted", "perturbed_predicted"]].isna().any().any():
        raise ValueError("Missing score within eligible pair")
    clean_m = metrics(paired, "clean_score", "clean_predicted")
    pert_m = metrics(paired, "perturbed_score", "perturbed_predicted")
    flip = paired.clean_predicted.to_numpy(int) != paired.perturbed_predicted.to_numpy(int)
    malignant = paired.actual.eq(1).to_numpy()
    clean_pred = paired.clean_predicted.to_numpy(int)
    pert_pred = paired.perturbed_predicted.to_numpy(int)
    output = {
        "model": model, "condition": condition,
        "eligible_patients": len(paired), "excluded_patients": n_total - len(paired),
        "eligible_images": int(paired.n_images.sum()),
        "mean_patient_iou": float(paired.mean_iou.mean()),
        "clean_balanced_accuracy": clean_m["balanced_accuracy"],
        "pert_balanced_accuracy": pert_m["balanced_accuracy"],
        "delta_balanced_accuracy": pert_m["balanced_accuracy"] - clean_m["balanced_accuracy"],
        "clean_sensitivity": clean_m["sensitivity"],
        "pert_sensitivity": pert_m["sensitivity"],
        "delta_sensitivity": pert_m["sensitivity"] - clean_m["sensitivity"],
        "clean_specificity": clean_m["specificity"],
        "pert_specificity": pert_m["specificity"],
        "delta_specificity": pert_m["specificity"] - clean_m["specificity"],
        "clean_roc_auc": clean_m["roc_auc"],
        "pert_roc_auc": pert_m["roc_auc"],
        "delta_roc_auc": pert_m["roc_auc"] - clean_m["roc_auc"],
        "flip_count": int(flip.sum()),
        "flip_rate": float(flip.mean()),
        "new_false_negatives": int((malignant & (clean_pred == 1) & (pert_pred == 0)).sum()),
        "recovered_false_negatives": int((malignant & (clean_pred == 0) & (pert_pred == 1)).sum()),
        "clean_false_negatives": clean_m["fn"],
        "pert_false_negatives": pert_m["fn"],
        "mean_abs_score_change": float(np.mean(np.abs(paired.perturbed_score - paired.clean_score))),
    }
    if fold is not None:
        output["outer_fold"] = int(fold)
    return output


def run(clean_path: Path, manifest_path: Path, perturbed_path: Path,
        results_dir: Path, output_dir: Path) -> dict:
    dev = load_development(clean_path, manifest_path)
    pert = load_perturbations(perturbed_path, dev)
    print("BUS-BRA — PAIRED BOUNDARY ROBUSTNESS (DEVELOPMENT ONLY)")
    print("=" * 56)
    print(f"Development: {len(dev)} images | {dev.Case.nunique()} patients | {len(FEATURE_COLUMNS)} features")
    print(f"Perturbations: {len(pert)} records; {int(pert.valid.sum())} valid; {(~pert.valid).sum()} unavailable")
    print("Five locked outer folds; original inner-chosen hyperparameters; no retraining on perturbed masks")
    print("Unavailable variants: exclude affected patient from condition-specific paired metrics")
    print("Test patients: NOT USED")

    all_preds = []
    fold_stats = []
    reference_checks = {}
    training_seconds = {}
    start = perf_counter()
    for model_name in MODEL_NAMES:
        selected = load_fold_settings(results_dir / f"{model_name}_nested_cv_folds.csv", model_name)
        clean_parts = []
        model_out = []
        model_fit_seconds = 0.0
        for fold in range(NUM_OUTER_FOLDS):
            train = dev.loc[dev.dev_fold.ne(fold)]
            validation = dev.loc[dev.dev_fold.eq(fold)]
            if set(train.Case) & set(validation.Case):
                raise AssertionError("Leakage: training/validation share patient")
            setup = selected[fold]
            t = perf_counter()
            if model_name == "logreg":
                model = fit_model(train, C=setup["C"], balance=setup["balance"])
            else:
                model = fit_svm(train, C=setup["C"], gamma=setup["gamma"], balance=setup["balance"])
            model_fit_seconds += perf_counter() - t
            baseline = _predict(model, validation, model_name, fold)
            clean_parts.append(baseline)
            for condition in VARIANT_NAMES:
                variants = pert.loc[pert.condition.eq(condition) & pert.dev_fold.eq(fold)]
                rec = compare_one_condition(model, model_name, validation, variants, fold)
                fold_stats.append(summarize_pair(rec, model=model_name, condition=condition, fold=fold))
                model_out.append(rec)
            print(f"{model_name.upper()} outer fold {fold}: {len(baseline)} held-out patients; 8 conditions", flush=True)
        recomputed = pd.concat(clean_parts, ignore_index=True)
        if recomputed.Case.duplicated().any() or len(recomputed) != dev.Case.nunique():
            raise RuntimeError("Clean OOF missing or repeated patient")
        validate_clean_reference(recomputed, results_dir / f"{model_name}_nested_cv_predictions.csv", model_name)
        reference_checks[model_name] = "MATCHED"
        training_seconds[model_name] = round(model_fit_seconds, 3)
        all_preds.extend(model_out)
        print(f"{model_name.upper()} saved clean OOF predictions: MATCHED", flush=True)
    predictions = pd.concat(all_preds, ignore_index=True)
    expected = dev.Case.nunique() * len(VARIANT_NAMES) * len(MODEL_NAMES)
    if len(predictions) != expected or predictions.duplicated(["model", "condition", "Case"]).any():
        raise AssertionError("Missing/duplicate model-condition-patient records")
    summary_rows = []
    for model_name in MODEL_NAMES:
        for condition in VARIANT_NAMES:
            subset = predictions.loc[predictions.model.eq(model_name) & predictions.condition.eq(condition)]
            summary_rows.append(summarize_pair(subset, model=model_name, condition=condition))
    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_stats)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.sort_values(["model", "condition", "Case"])
    predictions.to_csv(output_dir / "robustness_paired_patient_predictions.csv", index=False, float_format="%.12g")
    summary.to_csv(output_dir / "robustness_paired_condition_summary.csv", index=False, float_format="%.12g")
    folds.to_csv(output_dir / "robustness_outer_fold_metrics.csv", index=False, float_format="%.12g")
    result = {
        "protocol": "Refit fold-specific clean models with recorded inner-CV parameters; compare on matched held-out patients",
        "development_patients": int(dev.Case.nunique()), "development_images": int(len(dev)),
        "conditions": list(VARIANT_NAMES), "model_names": list(MODEL_NAMES),
        "perturbation_records": int(len(pert)), "valid_variations": int(pert.valid.sum()),
        "unavailable_variations": int((~pert.valid).sum()),
        "patient_rows_saved": len(predictions), "test_patients_seen": 0,
        "missing_condition_policy": "Exclude entire affected patient from that condition's paired metrics; retain NA record",
        "clean_reference_oof": reference_checks,
        "fold_training_seconds_not_comparable_benchmark": training_seconds,
        "elapsed_total_seconds": round(perf_counter() - start, 3),
        "result_rows": summary.to_dict("records"),
    }
    (output_dir / "robustness_paired_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nMODEL | CONDITION              | ELIGIBLE | DELTA BA | FLIP % | NEW FN")
    print("-" * 75)
    for r in summary.itertuples():
        print(f"{r.model:6} | {r.condition:22} | {r.eligible_patients:8} | {r.delta_balanced_accuracy:+.4f} | "
              f"{100*r.flip_rate:5.2f}% | {r.new_false_negatives:6}")
    print(f"Saved: {output_dir}")
    print("Both clean reference OOF tables: MATCHED | Held-out test: NOT USED")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Patient-matched held-out-fold boundary robustness evaluation")
    p.add_argument("--clean", type=Path, default=ROOT / "data/processed/development_features.csv")
    p.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    p.add_argument("--perturbed", type=Path, default=ROOT / "data/processed/development_perturbed_features.csv")
    p.add_argument("--modeling", type=Path, default=ROOT / "results/modeling")
    p.add_argument("--output", type=Path, default=ROOT / "results/robustness")
    args = p.parse_args()
    for path in (args.clean, args.manifest, args.perturbed,
                 args.modeling / "logreg_nested_cv_folds.csv",
                 args.modeling / "logreg_nested_cv_predictions.csv",
                 args.modeling / "svm_nested_cv_folds.csv",
                 args.modeling / "svm_nested_cv_predictions.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    run(args.clean, args.manifest, args.perturbed, args.modeling, args.output)


if __name__ == "__main__":
    main()
