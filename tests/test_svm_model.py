"""Unit and synthetic-data integration tests for RBF SVM patient-grouped CV."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.features import FEATURE_COLUMNS
from src.models import patient_image_weights
from src.svm_model import (
    build_svm, fit_svm, patient_svm_predictions, score_svm_patients,
    choose_svm_settings, nested_svm_cv,
)


def fake_data(n_cases=50):
    rng = np.random.default_rng(9)
    patients = pd.DataFrame({
        "Case": np.arange(n_cases),
        "Pathology": ["benign", "malignant"] * (n_cases // 2),
    })
    patients["dev_fold"] = -1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, val) in enumerate(cv.split(patients[["Case"]], patients["Pathology"])):
        patients.loc[val, "dev_fold"] = fold
    records = []
    for p in patients.itertuples():
        for im in range(1 + p.Case % 3):
            label = int(p.Pathology == "malignant")
            feats = {col: rng.normal(loc=label * 0.7 if j < 4 else 0, scale=1)
                     for j, col in enumerate(FEATURE_COLUMNS)}
            records.append({"ID": f"fake_{p.Case}_{im}", "Case": p.Case,
                            "Pathology": p.Pathology, "split": "development",
                            "dev_fold": int(p.dev_fold), **feats})
    return pd.DataFrame(records)


def test_svm_probability_disabled_to_avoid_hidden_image_cv():
    m = build_svm(C=1.0, gamma="scale")
    classifier = m.named_steps["classifier"]

    assert classifier.kernel == "rbf"
    assert not hasattr(classifier, "predict_proba")
    assert hasattr(classifier, "decision_function")


def test_svm_training_and_patient_weights():
    df = fake_data()
    w = patient_image_weights(df)
    assert np.allclose(pd.DataFrame({"case": df.Case, "w": w}).groupby("case").w.sum(), 1)
    model = fit_svm(df, C=1, gamma="scale", balance=True)
    pred = patient_svm_predictions(model, df)
    assert pred.Case.is_unique and len(pred) == df.Case.nunique()
    assert pred.n_images.sum() == len(df)


def test_patient_margin_is_mean_of_image_margins():
    df = fake_data()
    m = fit_svm(df, C=1, gamma="scale", balance=False)
    pred = patient_svm_predictions(m, df)
    raw = m.decision_function(df[list(FEATURE_COLUMNS)].to_numpy())
    expected = pd.DataFrame({"Case": df.Case, "margin": raw}).groupby("Case").margin.mean()
    assert np.allclose(pred.set_index("Case").decision_score.sort_index(), expected.sort_index())


def test_svm_patient_metrics_counts_are_consistent():
    df = fake_data()
    m = fit_svm(df, C=1, gamma="scale", balance=False)
    scored = score_svm_patients(patient_svm_predictions(m, df))
    assert scored["tn"] + scored["fp"] + scored["fn"] + scored["tp"] == df.Case.nunique()
    assert 0 <= scored["roc_auc"] <= 1
    assert "brier_score" not in scored  # margins are not probabilities


def test_inner_svm_tuning_does_not_touch_outer_holdout():
    df = fake_data()
    outer_train = df[df.dev_fold != 2]
    tuned = choose_svm_settings(outer_train, cs=(0.1, 1), gammas=("scale",),
                                balances=(False,), n_splits=3)
    assert tuned["C"] in (0.1, 1)
    assert tuned["gamma"] == "scale"
    assert 0 <= tuned["mean_inner_ba"] <= 1


def test_nested_svm_predicts_every_patient_exactly_once():
    df = fake_data()
    oof, folds, metrics = nested_svm_cv(df, cs=(0.1, 1),
                                        gammas=("scale",), balances=(False,))
    assert len(oof) == df.Case.nunique() and oof.Case.is_unique
    assert set(oof.outer_fold) == set(range(5))
    assert len(folds) == 5
    assert metrics["test_patients_used"] == 0
    assert sum(metrics[k] for k in ("tn", "fp", "fn", "tp")) == len(oof)
