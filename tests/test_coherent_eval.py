"""Step 16 integration tests: isolation, frozen settings, pairing, fail-closed output."""
import json

import numpy as np
import pandas as pd
import pytest

from src import coherent_eval as ce
from src.features import FEATURE_COLUMNS
from src.models import fit_model, patient_predictions, patient_image_weights
from src.robustness_eval import run as original_run
from src.svm_model import fit_svm, patient_svm_predictions
from test_robustness_eval import synthetic_data


@pytest.fixture
def checkpoint(tmp_path):
    clean, pert = synthetic_data()
    clean_path, manifest_path = tmp_path / "clean.csv", tmp_path / "manifest.csv"
    clean.to_csv(clean_path, index=False)
    clean = pd.read_csv(clean_path)
    clean[list(ce.ID_COLUMNS)].to_csv(manifest_path, index=False)
    mdir, original = tmp_path / "modeling", tmp_path / "original"
    mdir.mkdir()
    for name in ce.MODEL_NAMES:
        settings, predictions = [], []
        for fold in range(5):
            setting = {"outer_fold": fold, "C": float(fold + 1), "balance": fold % 2 == 0}
            train, val = clean.loc[clean.dev_fold.ne(fold)], clean.loc[clean.dev_fold.eq(fold)]
            if name == "logreg":
                model = fit_model(train, C=setting["C"], balance=setting["balance"])
                pred = patient_predictions(model, val, fold=fold)
            else:
                setting["gamma"] = "scale" if fold % 2 else "0.01"
                model = fit_svm(train, C=setting["C"], balance=setting["balance"],
                                gamma="scale" if fold % 2 else 0.01)
                pred = patient_svm_predictions(model, val, outer_fold=fold)
            settings.append(setting)
            predictions.append(pred)
        pd.DataFrame(settings).to_csv(mdir / f"{name}_nested_cv_folds.csv", index=False)
        pd.concat(predictions).to_csv(mdir / f"{name}_nested_cv_predictions.csv", index=False)
    pert_path = tmp_path / "original_features.csv"
    pert.to_csv(pert_path, index=False)
    original_run(clean_path, manifest_path, pert_path, mdir, original)
    coherent = pert.loc[pert.condition.isin(ce.COHERENT_CONDITIONS)].copy()
    coherent_path, geometry_path = tmp_path / "coherent.csv", tmp_path / "geometry.csv"
    coherent.to_csv(coherent_path, index=False)
    geometry = coherent[["ID", "condition", "valid", "actual_iou", "changed_pixels"]].copy()
    geometry["circularity_clean"] = coherent.shape_circularity
    geometry["circularity_coherent_protocol"] = coherent.shape_circularity
    geometry.to_csv(geometry_path, index=False)
    return (clean_path, manifest_path, coherent_path, geometry_path, mdir, original, tmp_path / "step16")


def test_end_to_end_reuses_exact_clean_training_settings_and_preserves_original(checkpoint, monkeypatch):
    clean_path, manifest, coherent_path, geometry, mdir, original, out = checkpoint
    clean = pd.read_csv(clean_path)
    before = {p: ce.sha256(p) for p in original.iterdir()}
    coherent = pd.read_csv(coherent_path)
    # Different evaluation vectors make a training-on-perturbations bug detectable.
    coherent[FEATURE_COLUMNS[0]] += 100
    coherent.to_csv(coherent_path, index=False)
    fits = []
    for name, real in (("logreg", ce.fit_model), ("svm", ce.fit_svm)):
        def spy(train, _name=name, _real=real, **kwargs):
            absent = set(clean.dev_fold) - set(train.dev_fold)
            assert len(absent) == 1
            fold = absent.pop()
            pd.testing.assert_frame_equal(train.reset_index(drop=True),
                                          clean.loc[clean.dev_fold.ne(fold)].reset_index(drop=True))
            locked = ce.load_fold_settings(mdir / f"{_name}_nested_cv_folds.csv", _name)
            assert kwargs == locked[fold]
            fits.append((_name, fold))
            model = _real(train, **kwargs)
            expected = np.average(train[list(FEATURE_COLUMNS)].to_numpy(float), axis=0,
                                  weights=patient_image_weights(train, kwargs["balance"]))
            assert np.allclose(model.named_steps["scaler"].mean_, expected)
            return model
        monkeypatch.setattr(ce, "fit_model" if name == "logreg" else "fit_svm", spy)
    # Only these tabular inputs may be read; there is no raw/test feature dependency.
    read_csv = pd.read_csv
    allowed = {p.resolve() for p in (clean_path, manifest, coherent_path, geometry,
               original / "robustness_paired_patient_predictions.csv",
               *mdir.glob("*.csv"))}
    def guarded_read(path, *args, **kwargs):
        assert path.resolve() in allowed
        return read_csv(path, *args, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(pd, "read_csv", guarded_read)
        result = ce.run(*checkpoint)
    assert fits == [(name, fold) for name in ce.MODEL_NAMES for fold in range(5)]
    assert result["clean_reference_oof"] == {"logreg": "MATCHED", "svm": "MATCHED"}
    assert result["step14_clean_predictions"] == "MATCHED"
    assert result["held_out_features_images_predictions_used"] is False
    assert {p: ce.sha256(p) for p in before} == before
    preds = pd.read_csv(out / ce.PREDICTION_FILE)
    assert len(preds) == clean.Case.nunique() * 8
    assert not preds.duplicated(["model", "condition", "Case"]).any()
    assert preds.eligible.all() and preds.protocol.eq(ce.PROTOCOL).all()
    assert set(preds.outer_fold) == set(range(5))
    assert len(pd.read_csv(out / "coherent_outer_fold_metrics.csv")) == 40
    comparison = pd.read_csv(out / "step14_vs_step16_common_patient_summary.csv")
    assert set(comparison.protocol) == {ce.PROTOCOL, ce.ORIGINAL_PROTOCOL}
    assert len(comparison) == 16
    assert len(pd.read_csv(out / "coherent_model_comparison.csv")) == 4
    assert json.loads((out / "coherent_summary.json").read_text())["inputs"]
    with pytest.raises(FileExistsError):
        ce.run(*checkpoint)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "test", "fold", "nonfinite", "unavailable", "condition"])
def test_corrupt_coherent_inputs_fail_before_output(checkpoint, corruption):
    path = checkpoint[2]
    frame = pd.read_csv(path)
    if corruption == "missing":
        frame = frame.iloc[:-1]
    elif corruption == "duplicate":
        frame.iloc[0] = frame.iloc[1]
    elif corruption == "test":
        frame.loc[0, "split"] = "test"
    elif corruption == "fold":
        frame.loc[0, "dev_fold"] = 4
    elif corruption == "nonfinite":
        frame.loc[0, FEATURE_COLUMNS[0]] = np.inf
    elif corruption == "unavailable":
        frame.loc[0, "valid"] = False
    else:
        frame.loc[0, "condition"] = "shift_left_moderate"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        ce.run(*checkpoint)
    assert not checkpoint[-1].exists()


@pytest.mark.parametrize("name", ce.MODEL_NAMES)
def test_stale_saved_oof_aborts_without_results(checkpoint, name):
    path = checkpoint[4] / f"{name}_nested_cv_predictions.csv"
    frame = pd.read_csv(path)
    col = "prob_malignant" if name == "logreg" else "decision_score"
    frame.loc[0, col] += 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="OOF score mismatch"):
        ce.run(*checkpoint)
    assert not checkpoint[-1].exists()


def test_geometry_drift_aborts(checkpoint):
    frame = pd.read_csv(checkpoint[3])
    frame.loc[0, "changed_pixels"] += 1
    frame.to_csv(checkpoint[3], index=False)
    with pytest.raises(ValueError, match="geometry mismatch"):
        ce.run(*checkpoint)
    assert not checkpoint[-1].exists()


def test_held_out_patient_overlap_aborts(checkpoint):
    path = checkpoint[1]
    frame = pd.read_csv(path)
    extra = frame.iloc[[0]].copy()
    extra["ID"], extra["split"], extra["dev_fold"] = "heldout_view", "test", np.nan
    pd.concat([frame, extra]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Leakage"):
        ce.run(*checkpoint)


def test_fractional_folds_rejected(checkpoint):
    for path in checkpoint[:2]:
        frame = pd.read_csv(path)
        frame["dev_fold"] = frame.dev_fold.astype(float)
        frame.loc[frame.dev_fold.eq(0), "dev_fold"] = 0.5
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="integers"):
        ce.run(*checkpoint)


def test_pairing_uses_common_patients_for_both_protocols(checkpoint):
    ce.run(*checkpoint)
    dev = pd.read_csv(checkpoint[0])
    old = ce.load_original(checkpoint[5] / "robustness_paired_patient_predictions.csv", dev)
    new = pd.read_csv(checkpoint[-1] / ce.PREDICTION_FILE)
    old.loc[old.Case.eq(2), "eligible"] = False
    old.loc[old.Case.eq(2), "perturbed_score"] = np.nan
    table = ce.protocol_comparison(old, new)
    assert table.common_eligible_patients.eq(dev.Case.nunique() - 1).all()
    assert table.eligible_patients.eq(dev.Case.nunique() - 1).all()
    assert np.allclose(table.delta_balanced_accuracy, 0)
    assert table.flip_count.eq(0).all()
    models = ce.model_comparison(new)
    assert np.allclose(models.svm_minus_logreg_delta_balanced_accuracy, 0)


@pytest.mark.parametrize("col", ["clean_score", "clean_predicted", "outer_fold", "actual"])
def test_stale_step14_reference_aborts(checkpoint, col):
    path = checkpoint[5] / "robustness_paired_patient_predictions.csv"
    frame = pd.read_csv(path)
    ix = frame.condition.eq("erosion_mild") & frame.model.eq("svm")
    index = frame.index[ix][0]
    frame.loc[index, col] += 0.1 if col == "clean_score" else 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        ce.run(*checkpoint)
    assert not checkpoint[-1].exists()


def test_original_output_directory_is_forbidden(checkpoint):
    args = list(checkpoint)
    args[-1] = args[-2]
    with pytest.raises(ValueError, match="separate"):
        ce.run(*args)


def test_unrelated_held_out_identifiers_do_not_enter_predictions(checkpoint):
    path = checkpoint[1]
    frame = pd.read_csv(path)
    extra = frame.iloc[[0]].copy()
    extra["ID"], extra["Case"], extra["split"], extra["dev_fold"] = "test_only", 99999, "test", np.nan
    pd.concat([frame, extra]).to_csv(path, index=False)
    result = ce.run(*checkpoint)
    preds = pd.read_csv(checkpoint[-1] / ce.PREDICTION_FILE)
    assert 99999 not in set(preds.Case)
    assert result["development_patients"] == 30
