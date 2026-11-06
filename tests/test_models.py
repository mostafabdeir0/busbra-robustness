"""Tests for leakage-safe patient-level Logistic Regression."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.features import FEATURE_COLUMNS
from src.models import (
    load_development,
    patient_image_weights,
    patient_predictions,
    score_patients,
    inner_patient_folds,
    nested_logreg_cv,
)


def fake_data(n_cases=50):
    rng = np.random.default_rng(7)
    patients = pd.DataFrame({"Case": np.arange(n_cases),
                             "Pathology": ["benign", "malignant"] * (n_cases // 2)})
    patients["dev_fold"] = -1
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, val) in enumerate(skf.split(patients[["Case"]], patients["Pathology"])):
        patients.loc[val, "dev_fold"] = fold
    rows = []
    for p in patients.itertuples():
        label = int(p.Pathology == "malignant")
        # Deliberately unequal images per patient.
        for im in range(1 + p.Case % 3):
            feats = {f: rng.normal(loc=label * (0.6 if j < 5 else 0.0), scale=1.0)
                     for j, f in enumerate(FEATURE_COLUMNS)}
            rows.append({"ID": f"fake_{p.Case}_{im}", "Case": p.Case,
                         "Pathology": p.Pathology, "split": "development",
                         "dev_fold": int(p.dev_fold), **feats})
    return pd.DataFrame(rows)


def test_patient_weights_equalize_image_counts():
    d = fake_data()
    w = patient_image_weights(d, balance=False)
    sums = pd.DataFrame({"Case": d.Case, "w": w}).groupby("Case")["w"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_balanced_weights_balance_patient_classes():
    d = fake_data()
    w = patient_image_weights(d, balance=True)
    totals = pd.DataFrame({"Pathology": d.Pathology, "w": w}).groupby("Pathology")["w"].sum()
    assert np.isclose(totals.loc["benign"], totals.loc["malignant"])


def test_load_development_matches_manifest(tmp_path):
    d = fake_data()
    feature_path, manifest_path = tmp_path / "features.csv", tmp_path / "manifest.csv"
    d.to_csv(feature_path, index=False)
    d[list(("ID", "Case", "Pathology", "split", "dev_fold"))].to_csv(manifest_path, index=False)
    loaded = load_development(feature_path, manifest_path)
    assert len(loaded) == len(d)
    d.loc[0, "split"] = "test"
    d.to_csv(feature_path, index=False)
    with pytest.raises(ValueError, match="non-development"):
        load_development(feature_path, manifest_path)


def test_patient_folds_do_not_share_patients():
    d = fake_data()
    for tr, val in inner_patient_folds(d, seed=42, n_splits=3):
        assert set(tr.Case).isdisjoint(set(val.Case))


def test_patient_aggregation_uses_mean_probability():
    d = fake_data()
    class PredictDummy:
        def predict_proba(self, X):
            p = np.linspace(.2, .8, len(X))
            return np.column_stack([1 - p, p])
    p = patient_predictions(PredictDummy(), d)
    assert len(p) == d.Case.nunique()
    assert p.Case.is_unique
    assert p.n_images.sum() == len(d)


def test_nested_cv_every_patient_is_predicted_once():
    d = fake_data()
    p, folds, summary = nested_logreg_cv(d, cs=(0.1, 1), balances=(False,))
    assert len(p) == d.Case.nunique()
    assert p.Case.is_unique
    assert set(p.outer_fold) == set(range(5))
    assert len(folds) == 5
    assert summary["test_patients_used"] == 0
    assert 0 <= summary["balanced_accuracy"] <= 1
    assert sum(summary[k] for k in ("tn", "fp", "fn", "tp")) == len(p)
