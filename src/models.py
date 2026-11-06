"""Leakage-aware baseline for BUS-BRA breast-lesion classification.

Place this file at src/models.py and run from repository root:
    .\\.venv\\Scripts\\python.exe -m src.models

Uses only data/processed/development_features.csv and the LOCKED image split.
Evaluates a tuned Logistic Regression with 5 outer patient folds and 3 inner
patient folds. Predictions are averaged to one per patient before scoring.
No held-out test labels or features are read, and no test model is evaluated.

The 0.50 threshold is a *provisional baseline* operating point. Calibration,
clinical operating-threshold selection and SVM comparison come later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CS = (0.01, 0.1, 1.0, 10.0, 100.0)
BALANCE_OPTIONS = (False, True)
NUM_OUTER_FOLDS = 5
NUM_INNER_FOLDS = 3


def load_development(feature_csv: Path, manifest_csv: Path) -> pd.DataFrame:
    """Verify feature columns and that every row belongs to locked development."""
    df = pd.read_csv(feature_csv)
    manifest = pd.read_csv(manifest_csv)
    id_cols = ["ID", "Case", "Pathology", "split", "dev_fold"]
    required = id_cols + list(FEATURE_COLUMNS)
    if list(df.columns) != required:
        raise ValueError("Feature table schema/order differs from validated 23-feature extractor")
    if df[id_cols].isna().any().any() or df["ID"].duplicated().any():
        raise ValueError("Missing identifiers or duplicate image IDs")
    if not df["split"].eq("development").all():
        raise ValueError("Feature table contains non-development patients")
    if not df["Pathology"].isin(("benign", "malignant")).all():
        raise ValueError("Unexpected pathology class")
    if not np.isfinite(df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite feature values")
    if (df.groupby("Case")["Pathology"].nunique() != 1).any():
        raise ValueError("A patient has multiple pathology labels")
    if (df.groupby("Case")["dev_fold"].nunique() != 1).any():
        raise ValueError("A patient occurs in multiple development folds")
    folds = set(df["dev_fold"].astype(int).unique())
    if folds != set(range(NUM_OUTER_FOLDS)):
        raise ValueError(f"Expected development fold IDs 0..4, found {folds}")

    manifest = manifest.loc[manifest["split"] == "development", id_cols]
    if manifest["ID"].duplicated().any():
        raise ValueError("Manifest has duplicate development image IDs")
    left = df[id_cols].sort_values("ID").reset_index(drop=True)
    right = manifest.sort_values("ID").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Feature table differs from locked development image manifest") from exc
    return df


def patient_image_weights(df: pd.DataFrame, balance: bool = False) -> np.ndarray:
    """Make each patient contribute total weight 1 (then optional case balancing)."""
    n_per_case = df.groupby("Case")["ID"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / n_per_case
    if balance:
        patients = df.groupby("Case")["Pathology"].first()
        class_counts = patients.value_counts()
        if len(class_counts) != 2:
            raise ValueError("Class weighting requires two classes")
        n_cases = len(patients)
        case_factors = {label: n_cases / (2.0 * int(count)) for label, count in class_counts.items()}
        weights *= df["Pathology"].map(case_factors).to_numpy(dtype=float)
    return weights


def make_model(C: float) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=float(C), max_iter=2000, random_state=SEED)),
    ])


def fit_model(df: pd.DataFrame, C: float, balance: bool) -> Pipeline:
    if df["Pathology"].nunique() != 2:
        raise ValueError("Training fold must contain both classes")
    w = patient_image_weights(df, balance=balance)
    model = make_model(C)
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = df["Pathology"].eq("malignant").astype(int).to_numpy()
    # Training-only weighted scaling and training-only weighted classifier.
    model.fit(X, y, scaler__sample_weight=w, classifier__sample_weight=w)
    return model


def patient_predictions(model: Pipeline, df: pd.Data, fold: int | None = None) -> pd.DataFrame:
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    probs = model.predict_proba(X)[:, 1]
    image_frame = pd.DataFrame({
        "Case": df["Case"].to_numpy(),
        "actual": df["Pathology"].eq("malignant").astype(int).to_numpy(),
        "prob_malignant": probs,
    })
    if (image_frame.groupby("Case")["actual"].nunique() != 1).any():
        raise ValueError("Inconsistent patient labels in prediction set")
    patient = image_frame.groupby("Case", as_index=False).agg(
        actual=("actual", "first"),
        prob_malignant=("prob_malignant", "mean"),
        n_images=("prob_malignant", "size"),
    )
    patient["predicted"] = (patient["prob_malignant"] >= 0.50).astype(int)
    if fold is not None:
        patient["outer_fold"] = int(fold)
    return patient


def score_patients(patients: pd.DataFrame) -> dict[str, float | int]:
    y = patients["actual"].to_numpy(dtype=int)
    yhat = patients["predicted"].to_numpy(dtype=int)
    probs = patients["prob_malignant"].to_numpy(dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation patients must contain both diagnosis classes")
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y, yhat, labels=[0, 1]).ravel())
    return {
        "patients": len(patients),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def inner_patient_folds(train: pd.DataFrame, seed: int, n_splits: int = NUM_INNER_FOLDS):
    """Construct stratified folds from UNIQUE PATIENTS, never image rows."""
    cases = train.groupby("Case", as_index=False)["Pathology"].first().sort_values("Case")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(cases[["Case"]], cases["Pathology"]):
        train_cases = set(cases.iloc[train_idx]["Case"])
        val_cases = set(cases.iloc[val_idx]["Case"])
        assert train_cases.isdisjoint(val_cases)
        fit_rows = train[train["Case"].isin(train_cases)]
        val_rows = train[train["Case"].isin(val_cases)]
        yield fit_rows, val_rows


def choose_settings(train: pd.DataFrame, *, seed: int = SEED,
                    cs=CS, balances=BALANCE_OPTIONS,
                    n_splits: int = NUM_INNER_FOLDS) -> dict:
    """Choose hyperparameters by inner-fold patient BA (not outer test fold)."""
    inner = list(inner_patient_folds(train, seed=seed, n_splits=n_splits))
    ranking = []
    for C in cs:
        for balance in balances:
            scores = []
            aucs = []
            for fitting, validation in inner:
                model = fit_model(fitting, C=C, balance=balance)
                scored = score_patients(patient_predictions(model, validation))
                scores.append(scored["balanced_accuracy"])
                aucs.append(scored["roc_auc"])
            ranking.append({
                "C": float(C), "balance": bool(balance),
                "mean_inner_ba": float(np.mean(scores)),
                "mean_inner_auc": float(np.mean(aucs)),
            })
    # Transparent deterministic tie-break: AUC, then simpler regularized model.
    # On equal BA/AUC, prefer smaller C and no class balancing.
    return max(ranking, key=lambda x: (
        x["mean_inner_ba"], x["mean_inner_auc"],
        -x["C"], -int(x["balance"]),
    ))


def nested_logreg_cv(development: pd.DataFrame, *, cs=CS,
                     balances=BALANCE_OPTIONS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predictions = []
    fold_stats = []
    begin = perf_counter()
    for outer in range(NUM_OUTER_FOLDS):
        train = development[development["dev_fold"].ne(outer)]
        validation = development[development["dev_fold"].eq(outer)]
        assert set(train["Case"]).isdisjoint(set(validation["Case"]))
        selected = choose_settings(train, seed=SEED + outer,
                                   cs=cs, balances=balances)
        model = fit_model(train, C=selected["C"], balance=selected["balance"])
        predicted = patient_predictions(model, validation, fold=outer)
        metrics = score_patients(predicted)
        predictions.append(predicted)
        fold_stats.append({"outer_fold": outer, **selected, **metrics})
        print(f"Outer fold {outer}: {metrics['patients']} patients | "
              f"C={selected['C']:g}, balanced={selected['balance']} | "
              f"BA={metrics['balanced_accuracy']:.3f}, "
              f"AUC={metrics['roc_auc']:.3f}, FN={metrics['fn']}", flush=True)
    oof = pd.concat(predictions, ignore_index=True).sort_values("Case").reset_index(drop=True)
    if oof["Case"].duplicated().any() or len(oof) != development["Case"].nunique():
        raise RuntimeError("OOF predictions must contain each development patient exactly once")
    overall = score_patients(oof)
    overall.update({
        "method": "nested grouped CV (5 locked outer folds, 3 inner patient-stratified folds)",
        "threshold": 0.50,
        "threshold_status": "provisional (not clinically calibrated)",
        "features": len(FEATURE_COLUMNS),
        "test_patients_used": 0,
        "elapsed_cv_seconds": round(perf_counter() - begin, 3),
    })
    return oof, pd.DataFrame(fold_stats), overall


def run(feature_csv: Path, manifest_csv: Path, output_dir: Path, model_dir: Path) -> dict:
    development = load_development(feature_csv, manifest_csv)
    print("BUS-BRA — LOGISTIC REGRESSION BASELINE")
    print("=" * 45)
    print(f"Development: {len(development)} images, {development['Case'].nunique()} patients, "
          f"{len(FEATURE_COLUMNS)} features")
    print("Protocol: 5 outer folds, 3 inner folds; patient-level metrics")
    print("Held-out test features and labels: NOT USED")
    oof, folds, result = nested_logreg_cv(development)
    print(f"Overall OOF BA={result['balanced_accuracy']:.4f}, "
          f"sensitivity={result['sensitivity']:.4f}, specificity={result['specificity']:.4f}, "
          f"AUC={result['roc_auc']:.4f}, FN={result['fn']}")

    # After honest OOF evaluation, select the settings for a development-only
    # reference model with 5-fold patient CV; do not evaluate on held-out test.
    settings = choose_settings(development, seed=SEED, n_splits=NUM_OUTER_FOLDS)
    reference = fit_model(development, C=settings["C"], balance=settings["balance"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output_dir / "logreg_nested_cv_predictions.csv", index=False)
    folds.to_csv(output_dir / "logreg_nested_cv_folds.csv", index=False)
    result["development_reference_model_settings"] = settings
    (output_dir / "logreg_nested_cv_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    joblib.dump({"pipeline": reference, "features": list(FEATURE_COLUMNS),
                 "settings": settings, "status": "development_only_provisional"},
                model_dir / "logreg_development_candidate.joblib")
    print(f"Saved evaluation results: {output_dir}")
    print(f"Saved provisional development-only model: {model_dir / 'logreg_development_candidate.joblib'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Patient-level nested CV Logistic Regression baseline")
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/development_features.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    run(args.features, args.manifest, args.output, args.models)


if __name__ == "__main__":
    main()
