"""Seven checks for Step 14 patient-paired robustness and locked-fold reuse."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.models import fit_model, patient_predictions
from src.perturbed_features import INFO_COLUMNS, VARIANT_NAMES
from src.robustness_eval import (
    parse_boolean_column, load_perturbations, load_fold_settings,
    eligible_patient_ids, validate_clean_reference, compare_one_condition,
    summarize_pair, run,
)
from src.svm_model import fit_svm, patient_svm_predictions


def synthetic_data(n_cases=30):
    rng = np.random.default_rng(14)
    dev_rows = []
    for case in range(n_cases):
        label = "malignant" if (case // 5) % 2 else "benign"
        # Each fold has six patients, three in each class.
        fold = case % 5
        for view in range(1 + case % 3):
            rec = {"ID": f"bus_{case:04d}-{view}", "Case": case,
                   "Pathology": label, "split": "development", "dev_fold": fold}
            rec.update({f: float(rng.normal(loc=0.7 if label == "malignant" and j < 5 else 0))
                        for j, f in enumerate(FEATURE_COLUMNS)})
            dev_rows.append(rec)
    clean = pd.DataFrame(dev_rows)
    variants = []
    for row in clean.to_dict("records"):
        for cond in VARIANT_NAMES:
            rec = {k: row[k] for k in ("ID", "Case", "Pathology", "split", "dev_fold")}
            rec.update({"condition": cond, "valid": True, "failure_reason": "",
                        "actual_iou": 0.9, "changed_pixels": 20, "area_ratio": 1,
                        "amount_pixels": 20})
            rec.update({f: row[f] for f in FEATURE_COLUMNS})
            variants.append(rec)
    pert = pd.DataFrame(variants, columns=list(INFO_COLUMNS) + list(FEATURE_COLUMNS))
    return clean, pert


def test_bool_parser_never_converts_false_string_to_true():
    got = parse_boolean_column(pd.Series(["True", "False", "TRUE", "false"]))
    assert got.tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        parse_boolean_column(pd.Series(["unknown"]))


def test_perturbed_table_has_locked_schema_and_all_conditions(tmp_path):
    clean, pert = synthetic_data()
    path = tmp_path / "features.csv"
    pert.to_csv(path, index=False)
    result = load_perturbations(path, clean)
    assert result.valid.all()
    pert = pert.iloc[:-1]
    pert.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing/extra"):
        load_perturbations(path, clean)


def test_unavailable_variation_has_no_fabricated_features(tmp_path):
    clean, pert = synthetic_data()
    pert.loc[0, "valid"] = False
    p = tmp_path / "pert.csv"
    pert.to_csv(p, index=False)
    with pytest.raises(ValueError, match="fabricated"):
        load_perturbations(p, clean)


def test_forged_patient_metadata_is_rejected(tmp_path):
    clean, pert = synthetic_data()
    pert.loc[0, "Case"] = 999
    p = tmp_path / "pert.csv"
    pert.to_csv(p, index=False)
    with pytest.raises(ValueError, match="metadata"):
        load_perturbations(p, clean)


def test_whole_patient_is_excluded_when_one_image_is_unavailable():
    clean, pert = synthetic_data()
    subset = pert[pert.condition.eq(VARIANT_NAMES[0])].copy()
    case = 2  # Has three ultrasound views.
    img_id = clean.loc[clean.Case.eq(case), "ID"].iloc[0]
    subset.loc[subset.ID.eq(img_id), "valid"] = False
    retained = eligible_patient_ids(clean, subset)
    assert case not in retained
    assert len(retained) == clean.Case.nunique() - 1


def test_stale_reference_prediction_is_caught(tmp_path):
    clean, _ = synthetic_data()
    model = fit_model(clean.loc[clean.dev_fold.ne(0)], C=1, balance=False)
    prediction = patient_predictions(model, clean.loc[clean.dev_fold.eq(0)], fold=0)
    p = tmp_path / "oof.csv"
    prediction.to_csv(p, index=False)
    validate_clean_reference(prediction, p, "logreg")
    corrupt = prediction.copy()
    corrupt.loc[0, "prob_malignant"] += 0.05
    corrupt.to_csv(p, index=False)
    with pytest.raises(ValueError, match="score mismatch"):
        validate_clean_reference(prediction, p, "logreg")


def test_full_synthetic_run_preserves_unavailable_rows_and_reuses_folds(tmp_path):
    clean, pert = synthetic_data()
    # One lesion view in one image is invalid under a specific condition.
    bad_id = clean.loc[clean.Case.eq(2), "ID"].iloc[0]
    bad_mask = pert.ID.eq(bad_id) & pert.condition.eq("shift_right_moderate")
    pert.loc[bad_mask, "valid"] = False
    pert.loc[bad_mask, list(FEATURE_COLUMNS)] = np.nan
    pert.loc[bad_mask, ["actual_iou", "changed_pixels", "area_ratio", "amount_pixels"]] = np.nan
    pert.loc[bad_mask, "failure_reason"] = "translation_would_clip_lesion"
    p_clean = tmp_path / "clean.csv"
    p_manifest = tmp_path / "manifest.csv"
    p_pert = tmp_path / "perturbed.csv"
    mdir, out = tmp_path / "modeling", tmp_path / "robustness"
    mdir.mkdir()
    clean.to_csv(p_clean, index=False)
    clean[["ID", "Case", "Pathology", "split", "dev_fold"]].to_csv(p_manifest, index=False)
    pert.to_csv(p_pert, index=False)
    for name in ("logreg", "svm"):
        fold_settings = []
        parts = []
        for fold in range(5):
            train = clean.loc[clean.dev_fold.ne(fold)]
            val = clean.loc[clean.dev_fold.eq(fold)]
            if name == "logreg":
                model = fit_model(train, C=1, balance=False)
                pred = patient_predictions(model, val, fold=fold)
                setting = {"outer_fold": fold, "C": 1, "balance": False}
            else:
                model = fit_svm(train, C=1, gamma="scale", balance=False)
                pred = patient_svm_predictions(model, val, outer_fold=fold)
                setting = {"outer_fold": fold, "C": 1, "gamma": "scale", "balance": False}
            parts.append(pred)
            fold_settings.append(setting)
        pd.DataFrame(fold_settings).to_csv(mdir / f"{name}_nested_cv_folds.csv", index=False)
        pd.concat(parts, ignore_index=True).to_csv(mdir / f"{name}_nested_cv_predictions.csv", index=False)
    result = run(p_clean, p_manifest, p_pert, mdir, out)
    assert result["clean_reference_oof"] == {"logreg": "MATCHED", "svm": "MATCHED"}
    assert result["unavailable_variations"] == 1
    p = pd.read_csv(out / "robustness_paired_patient_predictions.csv")
    assert len(p) == clean.Case.nunique() * 8 * 2
    missing = p.loc[(p.Case == 2) & p.condition.eq("shift_right_moderate")]
    assert len(missing) == 2
    assert missing.perturbed_score.isna().all()
    assert (~missing.eligible).all()
    summary = pd.read_csv(out / "robustness_paired_condition_summary.csv")
    affected = summary.loc[summary.condition.eq("shift_right_moderate")]
    assert affected.eligible_patients.eq(clean.Case.nunique() - 1).all()
    # All valid variants were copied unchanged from clean features in this fixture.
    assert np.allclose(summary.delta_balanced_accuracy, 0)
    assert summary.flip_count.eq(0).all()
