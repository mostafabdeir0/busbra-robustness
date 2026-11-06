"""BUS-BRA RBF SVM with patient-level nested cross-validation.

Run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.svm_model

Uses the SAME 23 development-only features, locked outer patient folds and
patient-level evaluation as src.models (Logistic Regression). Tunable parameters
are chosen within three inner patient folds. The held-out test is never read.

IMPORTANT: SVC(probability=False) deliberately avoids libsvm's internal,
image-level probability cross-validation. Raw SVM decision values are NOT
calibrated malignancy probabilities. At this stage, 0 is a provisional decision
threshold; patient predictions average decision values across available views.
Probability calibration and threshold tuning are reserved for a later,
patient-group-safe evaluation protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FEATURE_COLUMNS
from .models import (
    ROOT, SEED, NUM_OUTER_FOLDS, NUM_INNER_FOLDS,
    load_development, patient_image_weights, inner_patient_folds,
)

CS = (0.1, 1.0, 10.0)
GAMMAS = ("scale", 0.01, 0.1)
BALANCE_OPTIONS = (False, True)


def build_svm(C: float, gamma: str | float) -> Pipeline:
    """Training-fold weighted scaling followed by RBF SVM, no hidden CV."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=float(C), gamma=gamma)),
    ])


def fit_svm(train: pd.DataFrame, *, C: float, gamma: str | float,
            balance: bool) -> Pipeline:
    if train["Pathology"].nunique() != 2:
        raise ValueError("Training fold must contain both classes")
    weights = patient_image_weights(train, balance=balance)
    X = train[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = train["Pathology"].eq("malignant").astype(int).to_numpy()
    model = build_svm(C, gamma)
    model.fit(X, y, scaler__sample_weight=weights,
              classifier__sample_weight=weights)
    return model


def patient_svm_predictions(model: Pipeline, rows: pd.DataFrame,
                            outer_fold: int | None = None) -> pd.DataFrame:
    """One patient-level score: mean SVM decision margin over their images."""
    if len(rows) == 0:
        raise ValueError("Cannot predict an empty partition")
    scores = model.decision_function(rows[list(FEATURE_COLUMNS)].to_numpy(dtype=float))
    frame = pd.DataFrame({
        "Case": rows["Case"].to_numpy(),
        "actual": rows["Pathology"].eq("malignant").astype(int).to_numpy(),
        "decision_score": np.asarray(scores, dtype=float),
    })
    if (frame.groupby("Case")["actual"].nunique() != 1).any():
        raise ValueError("Inconsistent patient labels in prediction set")
    patients = frame.groupby("Case", as_index=False).agg(
        actual=("actual", "first"),
        decision_score=("decision_score", "mean"),
        n_images=("decision_score", "size"),
    )
    patients["predicted"] = (patients["decision_score"] >= 0.0).astype(int)
    if outer_fold is not None:
        patients["outer_fold"] = int(outer_fold)
    return patients


def score_svm_patients(patients: pd.DataFrame) -> dict[str, float | int]:
    y = patients["actual"].to_numpy(dtype=int)
    yhat = patients["predicted"].to_numpy(dtype=int)
    score = patients["decision_score"].to_numpy(dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Validation patients must contain both classes")
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite SVM decision scores")
    tn, fp, fn, tp = map(int, confusion_matrix(y, yhat, labels=[0, 1]).ravel())
    return {
        "patients": len(patients),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(y, score)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def choose_svm_settings(train: pd.DataFrame, *, seed: int = SEED,
                        cs=CS, gammas=GAMMAS,
                        balances=BALANCE_OPTIONS,
                        n_splits: int = NUM_INNER_FOLDS) -> dict:
    """Rank configurations on independent, inner-validation patients only."""
    inner = list(inner_patient_folds(train, seed=seed, n_splits=n_splits))
    ranking = []
    for C in cs:
        for gamma in gammas:
            for balance in balances:
                bas, aucs = [], []
                for fitting, validation in inner:
                    model = fit_svm(fitting, C=C, gamma=gamma, balance=balance)
                    metric = score_svm_patients(patient_svm_predictions(model, validation))
                    bas.append(metric["balanced_accuracy"])
                    aucs.append(metric["roc_auc"])
                ranking.append({
                    "C": float(C),
                    "gamma": gamma,
                    "balance": bool(balance),
                    "mean_inner_ba": float(np.mean(bas)),
                    "mean_inner_auc": float(np.mean(aucs)),
                })
    # Deterministic tie-break: BA, AUC, smaller C, 'scale' gamma, no balancing.
    return max(ranking, key=lambda r: (
        r["mean_inner_ba"], r["mean_inner_auc"],
        -r["C"], int(r["gamma"] == "scale"), -int(r["balance"]),
    ))


def nested_svm_cv(development: pd.DataFrame, *, cs=CS, gammas=GAMMAS,
                  balances=BALANCE_OPTIONS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    predicted_folds, summaries = [], []
    begin = perf_counter()
    for outer in range(NUM_OUTER_FOLDS):
        train = development[development["dev_fold"].ne(outer)]
        validation = development[development["dev_fold"].eq(outer)]
        if not set(train["Case"]).isdisjoint(set(validation["Case"])):
            raise RuntimeError("Patient leakage across outer folds")
        selected = choose_svm_settings(
            train, seed=SEED + outer, cs=cs,
            gammas=gammas, balances=balances,
        )
        model = fit_svm(train, C=selected["C"], gamma=selected["gamma"],
                        balance=selected["balance"])
        predicted = patient_svm_predictions(model, validation, outer_fold=outer)
        stats = score_svm_patients(predicted)
        predicted_folds.append(predicted)
        summaries.append({"outer_fold": outer, **selected, **stats})
        print(f"Outer fold {outer}: {stats['patients']} patients | "
              f"C={selected['C']:g}, gamma={selected['gamma']}, "
              f"balanced={selected['balance']} | "
              f"BA={stats['balanced_accuracy']:.3f}, "
              f"AUC={stats['roc_auc']:.3f}, FN={stats['fn']}", flush=True)
    oof = pd.concat(predicted_folds, ignore_index=True).sort_values("Case").reset_index(drop=True)
    if oof["Case"].duplicated().any() or len(oof) != development["Case"].nunique():
        raise RuntimeError("OOF predictions must contain every development patient once")
    overall = score_svm_patients(oof)
    overall.update({
        "method": "nested patient-level CV (5 locked outer folds, 3 inner folds)",
        "features": len(FEATURE_COLUMNS),
        "threshold": 0.0,
        "threshold_status": "provisional raw margin (not calibrated probability)",
        "score_type": "SVC decision_function margin, patient-mean",
        "probability_calibration": "not performed; must use grouped CV later",
        "test_patients_used": 0,
        "elapsed_cv_seconds": round(perf_counter() - begin, 3),
    })
    return oof, pd.DataFrame(summaries), overall


def run(feature_csv: Path, manifest_csv: Path, output_dir: Path,
        model_dir: Path) -> dict:
    df = load_development(feature_csv, manifest_csv)
    print("BUS-BRA — RBF SVM CLASSIFIER")
    print("=" * 45)
    print(f"Development: {len(df)} images, {df['Case'].nunique()} patients, "
          f"{len(FEATURE_COLUMNS)} features")
    print("Protocol: 5 locked outer patient folds, 3 inner patient folds")
    print("SVM scores: raw decision margins; threshold 0 is provisional")
    print("Held-out test features and labels: NOT USED")
    oof, folds, result = nested_svm_cv(df)
    print(f"Overall OOF BA={result['balanced_accuracy']:.4f}, "
          f"sensitivity={result['sensitivity']:.4f}, "
          f"specificity={result['specificity']:.4f}, "
          f"AUC={result['roc_auc']:.4f}, FN={result['fn']}")

    # Select a provisional full-development reference model after honest OOF eval.
    chosen = choose_svm_settings(df, seed=SEED, n_splits=NUM_OUTER_FOLDS)
    reference = fit_svm(df, C=chosen["C"], gamma=chosen["gamma"],
                        balance=chosen["balance"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output_dir / "svm_nested_cv_predictions.csv", index=False)
    folds.to_csv(output_dir / "svm_nested_cv_folds.csv", index=False)
    result["development_reference_model_settings"] = chosen
    (output_dir / "svm_nested_cv_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    joblib.dump({
        "pipeline": reference,
        "features": list(FEATURE_COLUMNS),
        "settings": chosen,
        "status": "development_only_provisional_raw_decision_scores",
    }, model_dir / "svm_development_candidate.joblib")
    print(f"Saved evaluation results: {output_dir}")
    print(f"Saved provisional development-only model: "
          f"{model_dir / 'svm_development_candidate.joblib'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Patient-level nested CV RBF SVM")
    parser.add_argument("--features", type=Path,
                        default=ROOT / "data/processed/development_features.csv")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/modeling")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    run(args.features, args.manifest, args.output, args.models)


if __name__ == "__main__":
    main()
