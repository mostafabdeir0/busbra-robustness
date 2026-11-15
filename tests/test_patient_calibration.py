"""Step 20: disjoint patient calibration, frozen choices, serializable final models."""
import copy
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import patient_calibration as pc
from src import stable_evaluation as se
from src import stable_selection as ss
from test_stable_evaluation import small_protocol
from test_stable_selection import feature_tables


@pytest.fixture(scope="module")
def reference_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("calibration_reference")
    clean, coherent = feature_tables()
    clean_path, manifest, coherent_path, geometry = [root / n for n in ("clean.csv", "manifest.csv", "coherent.csv", "geometry.csv")]
    clean.to_csv(clean_path, index=False)
    clean = pd.read_csv(clean_path)
    rows = clean[["ID", "Case", "Pathology", "split", "dev_fold"]].copy()
    heldout = rows.iloc[:2].copy()
    heldout["ID"], heldout["Case"], heldout["split"], heldout["dev_fold"] = ["test_a", "test_b"], [9000, 9001], "test", np.nan
    pd.concat([rows, heldout]).to_csv(manifest, index=False)
    coherent.to_csv(coherent_path, index=False)
    geometry_rows = coherent[["ID", "condition", "valid", "actual_iou", "changed_pixels"]].copy()
    geometry_rows["circularity_clean"] = clean.set_index("ID").loc[coherent.ID, "shape_circularity"].to_numpy(float)
    geometry_rows["circularity_coherent_protocol"] = coherent.shape_circularity
    geometry_rows.to_csv(geometry, index=False)
    design, prior = root / "design", root / "step19"
    ss.run(clean_path, manifest, coherent_path, geometry, design)
    result = se.nested_evaluate(clean, coherent, small_protocol())
    prior.mkdir()
    for key in ("clean_oof_predictions.csv", "coherent_paired_predictions.csv"):
        result["tables"][key].to_csv(prior / key, index=False)
    paths = {"clean_features": clean_path, "manifest": manifest, "coherent_features": coherent_path,
             "geometry": geometry, "protocol": design / "selection_protocol.json", "design_summary": design / "selection_design_summary.json"}
    summary = {"status": "exploratory_nested_development_evaluation", "held_out_evaluation_used": False,
               "original_clean_oof": {n: "MATCHED" for n in pc.MODEL_NAMES}, "source_sha256": pc.sha256(Path(se.__file__)),
               "development_patients": 30, "development_images": len(clean),
               "inputs": {key: {"sha256": pc.sha256(path)} for key, path in paths.items()},
               "clean_summary": result["tables"]["clean_summary.csv"].to_dict("records")}
    (prior / "evaluation_summary.json").write_text(ss.canonical_json(summary))
    return root


@pytest.fixture
def checkpoint(reference_checkpoint, tmp_path):
    root = tmp_path / "inputs"
    shutil.copytree(reference_checkpoint, root)
    return (root / "clean.csv", root / "manifest.csv", root / "coherent.csv", root / "geometry.csv",
            root / "design", root / "step19", tmp_path / "step20")


def test_calibration_members_keep_base_calibration_and_excluded_patients_disjoint(monkeypatch):
    clean, coherent = feature_tables()
    pool = clean.loc[clean.dev_fold.ne(0)]
    excluded = set(clean.loc[clean.dev_fold.eq(0), "Case"])
    real_tune, real_sigmoid, real_fit = se.tune_arm, pc.fit_sigmoid, se.fit_subset
    tuned_scopes, calibration_cases, fits = [], [], []
    def tune(contexts, name, arm, protocol):
        base = set.union(*(set(c["train"].Case) for c in contexts))
        assert base.isdisjoint(excluded)
        tuned_scopes.append(base)
        for ctx in contexts:
            val_cases = set(ctx["pack"]["Case"])
            assert set(ctx["train"].Case).isdisjoint(val_cases)
            assert set(ctx["train"].Case) | val_cases == base
        return real_tune(contexts, name, arm, protocol)
    def sigmoid(scores):
        cases = set(scores.Case)
        assert cases.isdisjoint(tuned_scopes[-1] | excluded)
        assert len(scores) == len(cases)
        calibration_cases.append(cases)
        return real_sigmoid(scores)
    def fit(train, name, columns, settings):
        assert set(train.Case).issubset(tuned_scopes[-1])
        assert set(train.Case).isdisjoint(excluded)
        assert columns == pc.FEATURE_COLUMNS
        fits.append(set(train.Case))
        return real_fit(train, name, columns, settings)
    monkeypatch.setattr(se, "tune_arm", tune)
    monkeypatch.setattr(pc, "fit_sigmoid", sigmoid)
    monkeypatch.setattr(se, "fit_subset", fit)
    ensemble, audit, _ = pc.fit_ensemble(pool, coherent, "svm", excluded_cases=excluded, scope_code=0, protocol=small_protocol())
    assert len(ensemble["members"]) == len(audit) == len(calibration_cases) == 3
    assert set.union(*calibration_cases) == set(pool.Case)
    assert sum(len(c) for c in calibration_cases) == pool.Case.nunique()
    assert len(fits) == 12  # Three tuning fits plus one refit per member.


def test_excluded_values_cannot_change_ensemble_models_or_calibration():
    clean, coherent = feature_tables()
    pool = clean.loc[clean.dev_fold.ne(0)]
    excluded = set(clean.loc[clean.dev_fold.eq(0), "Case"])
    ensemble, audit, _ = pc.fit_ensemble(pool, coherent, "logreg", excluded_cases=excluded, scope_code=0, protocol=small_protocol())
    changed = coherent.copy()
    changed.loc[changed.Case.isin(excluded), list(pc.FEATURE_COLUMNS)] = np.inf
    other, checks, _ = pc.fit_ensemble(pool, changed, "logreg", excluded_cases=excluded, scope_code=0, protocol=small_protocol())
    assert audit == checks
    a, b = pc.predict_ensemble(ensemble, clean), pc.predict_ensemble(other, clean)
    pd.testing.assert_frame_equal(a, b)


def test_lr_calibration_input_is_logit_of_patient_mean_probability():
    clean, _ = feature_tables()
    model = se.fit_subset(clean, "logreg", pc.FEATURE_COLUMNS, {"C": 1.0, "balance": False})
    result = pc.patient_raw(model, clean, "logreg")
    expected = pc.logit(np.clip(result.raw_score, pc.PROBABILITY_CLIP, 1 - pc.PROBABILITY_CLIP))
    assert np.allclose(result.calibration_input, expected)
    assert len(result) == clean.Case.nunique()


def test_calibrator_weights_patients_equally_regardless_of_image_counts():
    sample = pd.DataFrame({"Case": [1, 2, 3, 4], "actual": [0, 0, 1, 1],
                           "calibration_input": [-2., -.5, .7, 2.], "n_images": [1, 100, 1, 100]})
    first = pc.fit_sigmoid(sample)
    sample["n_images"] = [1000, 1, 1000, 1]
    second = pc.fit_sigmoid(sample)
    assert np.array_equal(first.named_steps["sigmoid"].coef_, second.named_steps["sigmoid"].coef_)
    assert np.array_equal(first.named_steps["sigmoid"].intercept_, second.named_steps["sigmoid"].intercept_)


def test_ensemble_averages_calibrated_patient_probabilities_after_view_aggregation():
    clean, coherent = feature_tables()
    ensemble, _, _ = pc.fit_ensemble(clean, coherent, "logreg", excluded_cases={9000}, scope_code=5, protocol=small_protocol())
    result = pc.predict_ensemble(ensemble, clean)
    individual = []
    for member in ensemble["members"]:
        raw = pc.patient_raw(member["base_model"], clean, "logreg").sort_values("Case")
        individual.append(member["calibrator"].predict_proba(raw[["calibration_input"]].to_numpy(float))[:, 1])
    assert np.allclose(result.prob_malignant, np.mean(individual, axis=0))
    assert result.predicted.equals(result.prob_malignant.ge(.5).astype(int))
    assert result.prob_malignant.between(0, 1).all()


def test_nested_outer_coverage_and_clean_probability_metrics():
    clean, coherent = feature_tables()
    oof, pairs, scores, _, bins, audit, _ = pc.nested_calibration(clean, coherent, small_protocol())
    assert len(oof) == 60 and len(pairs) == 240
    assert not oof.duplicated(["model", "Case"]).any()
    assert not pairs.duplicated(["model", "condition", "Case"]).any()
    assert len(audit) == 30 and len(bins) == 20
    for member in audit:
        base, cal, excluded = [set(member[k]) for k in ("base_cases", "calibration_cases", "excluded_cases")]
        assert base.isdisjoint(cal | excluded) and cal.isdisjoint(excluded)
        assert excluded == set(clean.loc[clean.dev_fold.eq(member["scope_code"]), "Case"])
    for score in scores:
        assert 0 <= score["brier_score"] <= 1 and score["log_loss"] >= 0


def test_full_freeze_is_relocatable_serializable_read_only_and_has_no_test_features(checkpoint, monkeypatch):
    inputs = [*checkpoint[:4], *checkpoint[4].iterdir(), *checkpoint[5].iterdir()]
    before = {p: pc.sha256(p) for p in inputs}
    allowed = {p.resolve() for p in inputs}
    read_csv = pd.read_csv
    def read_guard(path, *args, **kwargs):
        assert Path(path).resolve() in allowed
        return read_csv(path, *args, **kwargs)
    real_ensemble = pc.fit_ensemble
    def reduced(pool, coherent, name, **kwargs):
        assert kwargs["protocol"] == ss.locked_protocol()
        kwargs["protocol"] = small_protocol()
        assert set(pool.Case).isdisjoint({9000, 9001})
        return real_ensemble(pool, coherent, name, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(pd, "read_csv", read_guard)
        context.setattr(pc, "fit_ensemble", reduced)
        summary = pc.run(*checkpoint)
    assert summary["held_out_evaluation_used"] is False
    assert summary["base_refits"] == summary["sigmoid_fits"] == 36
    assert before == {p: pc.sha256(p) for p in before}
    out = checkpoint[-1]
    protocol = json.loads((out / "final_evaluation_protocol.json").read_text())
    assert protocol["primary_model"] == "svm" and protocol["threshold"] == .5
    assert protocol["held_out_patients"] == protocol["held_out_images"] == 2
    assert summary["final_protocol_sha256"] == pc.sha256(out / "final_evaluation_protocol.json")
    assert protocol["model_artifact_sha256"] == pc.sha256(out / "frozen_calibrated_models.joblib")
    bundle = joblib.load(out / "frozen_calibrated_models.joblib")
    assert len(bundle["models"]) == 2 and all(len(e["members"]) == 3 for e in bundle["models"].values())
    clean = pd.read_csv(checkpoint[0])
    for name in pc.MODEL_NAMES:
        first = pc.predict_ensemble(bundle["models"][name], clean)
        loaded = joblib.load(out / "frozen_calibrated_models.joblib")
        pd.testing.assert_frame_equal(first, pc.predict_ensemble(loaded["models"][name], clean))
    with pytest.raises(FileExistsError):
        pc.run(*checkpoint)


@pytest.mark.parametrize("corruption", ["source", "clean_metric", "composite_metric", "input_hash", "heldout_flag"])
def test_step19_corruption_aborts_before_any_calibration(checkpoint, corruption, monkeypatch):
    path = checkpoint[5] / "evaluation_summary.json"
    content = json.loads(path.read_text())
    if corruption == "source":
        content["source_sha256"] = "invalid"
    elif corruption == "clean_metric":
        content["clean_summary"][0]["balanced_accuracy"] += .1
    elif corruption == "composite_metric":
        content["clean_summary"][0]["mean_five_condition_ba"] += .1
    elif corruption == "input_hash":
        content["inputs"]["clean_features"]["sha256"] = "invalid"
    else:
        content["held_out_evaluation_used"] = True
    path.write_text(json.dumps(content))
    def no_fit(*args, **kwargs):
        raise AssertionError("Corruption must fail before model/calibration fitting")
    monkeypatch.setattr(pc, "fit_ensemble", no_fit)
    with pytest.raises(ValueError, match="Step 19"):
        pc.run(*checkpoint)
    assert not checkpoint[-1].exists()


def test_overlapping_pool_and_excluded_patients_rejected():
    clean, coherent = feature_tables()
    with pytest.raises(ValueError, match="Leakage"):
        pc.fit_ensemble(clean, coherent, "svm", excluded_cases={0}, scope_code=0, protocol=small_protocol())


@pytest.mark.parametrize("corruption", ["duplicate", "one_class", "nonfinite"])
def test_bad_calibration_patient_sample_rejected(corruption):
    sample = pd.DataFrame({"Case": [1, 2, 3, 4], "actual": [0, 0, 1, 1], "calibration_input": [-1., -.5, .5, 1.]})
    if corruption == "duplicate":
        sample.loc[0, "Case"] = 2
    elif corruption == "one_class":
        sample["actual"] = 0
    else:
        sample.loc[0, "calibration_input"] = np.inf
    with pytest.raises(ValueError):
        pc.fit_sigmoid(sample)


def test_reliability_bins_include_probability_one_and_do_not_drop_empty_bins():
    sample = pd.DataFrame({"prob_malignant": [0., .1, 1.], "actual": [0, 0, 1]})
    bins = pc.calibration_bins(sample, "svm")
    assert len(bins) == 10 and sum(b["patients"] for b in bins) == 3
    assert bins[9]["patients"] == 1 and bins[9]["mean_probability"] == 1
    assert bins[8]["mean_probability"] is None


def test_invalid_probability_is_not_scored_as_valid_calibration():
    sample = pd.DataFrame({"actual": [0, 1], "prob_malignant": [-.1, 1.1], "predicted": [0, 1]})
    with pytest.raises(ValueError, match="probability"):
        pc.probability_metrics(sample)


def test_primary_model_method_and_threshold_fixed_before_results():
    design = pc.calibration_design()
    assert design["primary_model"] == "svm" and design["comparator"] == "logreg"
    assert design["threshold"] == .5 and design["features"] == list(pc.FEATURE_COLUMNS)
    assert design["calibration_folds"] == 3
    assert "forbidden" in design["method_selection_after_outer_results"]


def test_corrupt_step19_patient_metadata_fails(checkpoint):
    path = checkpoint[5] / "coherent_paired_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "n_images"] += 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="metadata"):
        pc.run(*checkpoint)
    assert not checkpoint[-1].exists()
