"""Synthetic-only tests; real held-out images are never read by the test suite."""
import io
import json
import shutil
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

from src import heldout_evaluation as he
from src import patient_calibration as pc
from src import stable_selection as ss
from test_stable_evaluation import small_protocol
from test_stable_selection import feature_tables


@pytest.fixture(scope="module")
def reference_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_heldout")
    calibration = root / "calibration"
    calibration.mkdir()
    clean, coherent = feature_tables()
    train_path = root / "training.csv"
    clean.to_csv(train_path, index=False)
    test = pd.DataFrame([{"ID": f"bus_fake_{case}_{view}", "Case": case,
                          "Pathology": "malignant" if case % 2 else "benign", "split": "test", "dev_fold": -1}
                         for case in range(100, 108) for view in range(1 + case % 2)])
    manifest_path = root / "manifest.csv"
    pd.concat([clean[["ID", "Case", "Pathology", "split", "dev_fold"]], test]).to_csv(manifest_path, index=False)
    archive_path = root / "fake.zip"
    yy, xx = np.ogrid[:80, :88]
    mask = ((yy - 40) / 18) ** 2 + ((xx - 44) / 22) ** 2 <= 1
    with ZipFile(archive_path, "w") as archive:
        for index, row in enumerate(test.itertuples(index=False)):
            image = np.tile(np.arange(88, dtype=np.uint8), (80, 1)) + index
            for path, array in ((f"BUSBRA/Images/{row.ID}.png", image),
                                (f"BUSBRA/Masks/mask_{row.ID.removeprefix('bus_')}.png", mask.astype(np.uint8) * 255)):
                stream = io.BytesIO()
                Image.fromarray(array).save(stream, format="PNG")
                archive.writestr(path, stream.getvalue())
    audit_path = root / "audit.json"
    he.write_json(audit_path, {"archive_sha256": pc.sha256(archive_path)})
    models = {}
    for name in he.MODEL_NAMES:
        ensemble, _, _ = pc.fit_ensemble(clean, coherent, name, excluded_cases=set(test.Case), scope_code=5, protocol=small_protocol())
        models[name] = ensemble
    bundle = {"design": pc.calibration_design(), "features": list(he.FEATURE_COLUMNS), "models": models,
              "development_case_sha256": ss.case_digest(set(clean.Case))}
    model_path = calibration / "frozen_calibrated_models.joblib"
    joblib.dump(bundle, model_path, compress=3)
    modules = [pc, pc.se, ss, pc.feature_module, pc.coherent_feature_module, pc.coherent_geometry_module,
               pc.original_geometry_module, pc.model_module, pc.svm_module, pc.robustness_module]
    protocol = {"status": "frozen_before_held_out_evaluation", "design": pc.calibration_design(),
                "primary_model": "svm", "comparator": "logreg", "threshold": .5,
                "primary_endpoint": "SVM clean patient balanced accuracy at fixed probability threshold 0.5",
                "held_out_conditions": list(he.CONDITIONS), "held_out_images": len(test), "held_out_patients": test.Case.nunique(),
                "held_out_case_sha256": ss.case_digest(set(test.Case)), "held_out_image_ids_sha256": he.image_id_digest(test.ID),
                "source_sha256": {Path(m.__file__).name: pc.sha256(Path(m.__file__)) for m in modules},
                "model_artifact_sha256": pc.sha256(model_path), "uncertainty_plan": he.UNCERTAINTY_PLAN,
                "inputs": {"manifest": {"path": str(manifest_path), "sha256": pc.sha256(manifest_path)},
                           "clean_features": {"path": str(train_path), "sha256": pc.sha256(train_path)}}}
    protocol_path = calibration / "final_evaluation_protocol.json"
    he.write_json(protocol_path, protocol)
    he.write_json(calibration / "calibration_summary.json", {
        "status": "calibration_complete_final_protocol_frozen", "held_out_evaluation_used": False,
        "final_protocol_sha256": pc.sha256(protocol_path), "model_artifact_sha256": pc.sha256(model_path),
        "versions": {p: he.version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "pillow", "scikit-image")}})
    return root


@pytest.fixture
def checkpoint(reference_checkpoint, tmp_path):
    root = tmp_path / "inputs"
    shutil.copytree(reference_checkpoint, root)
    # Relocate the synthetic training inputs, then re-fingerprint the synthetic freeze.
    ppath, spath = root / "calibration/final_evaluation_protocol.json", root / "calibration/calibration_summary.json"
    protocol, summary = he.read_json(ppath), he.read_json(spath)
    protocol["inputs"]["manifest"]["path"] = str(root / "manifest.csv")
    protocol["inputs"]["clean_features"]["path"] = str(root / "training.csv")
    he.write_json(ppath, protocol)
    summary["final_protocol_sha256"] = pc.sha256(ppath)
    he.write_json(spath, summary)
    return root / "calibration", root / "manifest.csv", root / "fake.zip", root / "audit.json", tmp_path / "heldout"


def test_preflight_never_decodes_images_predicts_or_creates_marker(checkpoint, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only preflight must not open images or predict")
    monkeypatch.setattr(he, "ZipFile", forbidden)
    monkeypatch.setattr(pc, "predict_ensemble", forbidden)
    _, _, test, _ = he.verify_frozen(*checkpoint[:4])
    assert test.Case.nunique() == 8 and len(test) == 12
    assert not (checkpoint[0] / he.MARKER).exists() and not checkpoint[-1].exists()


def test_feature_extraction_reads_exactly_the_test_ids_and_labels_them_test(checkpoint, monkeypatch):
    _, _, test, _ = he.verify_frozen(*checkpoint[:4])
    real = he.read_pair
    ids = []
    def read(archive, image_id):
        assert image_id in set(test.ID)
        ids.append(image_id)
        return real(archive, image_id)
    monkeypatch.setattr(he, "read_pair", read)
    clean, coherent = he.extract_test_features(checkpoint[2], test)
    assert set(ids) == set(test.ID) and len(ids) == len(test)
    assert len(clean) == 12 and len(coherent) == 48
    assert clean.split.eq("test").all() and coherent.split.eq("test").all()
    assert set(coherent.condition) == set(he.COHERENT_CONDITIONS)
    assert not coherent.duplicated(["ID", "condition"]).any()


def test_full_synthetic_run_never_fits_and_guards_second_run_across_output_paths(checkpoint, monkeypatch):
    import sklearn.linear_model
    import sklearn.svm
    original = {p: pc.sha256(p) for p in checkpoint[0].iterdir()}
    def no_fit(*args, **kwargs):
        raise AssertionError("Held-out evaluation must never fit or tune")
    monkeypatch.setattr(sklearn.linear_model.LogisticRegression, "fit", no_fit)
    monkeypatch.setattr(sklearn.svm.SVC, "fit", no_fit)
    monkeypatch.setattr(pc, "fit_ensemble", no_fit)
    result = he.run(*checkpoint)
    assert result["model_fits"] == result["calibrator_fits"] == 0
    assert result["selection_or_retuning"] is False
    assert result["held_out_patients"] == 8 and result["held_out_images"] == 12
    assert result["patient_prediction_rows"] == 80 and result["paired_patient_rows"] == 64
    assert original == {p: pc.sha256(p) for p in original}
    marker = he.read_json(checkpoint[0] / he.MARKER)
    assert marker["status"] == "complete" and marker["summary_sha256"] == pc.sha256(checkpoint[-1] / "held_out_summary.json")
    intervals = pd.read_csv(checkpoint[-1] / "held_out_intervals.csv")
    assert len(intervals) == 181 and intervals.primary.sum() == 1
    assert intervals.loc[intervals.primary, "model"].iloc[0] == "svm"
    weights = np.load(checkpoint[-1] / "paired_bootstrap_weights.npz")
    assert weights["counts"].shape == (10000, 8)
    for filename, fingerprint in marker["published_files_sha256"].items():
        assert pc.sha256(checkpoint[-1] / filename) == fingerprint
    second = list(checkpoint)
    second[-1] = checkpoint[-1].with_name("another_output")
    with pytest.raises(FileExistsError, match="already started"):
        he.run(*second)


def test_shared_bootstrap_preserves_class_counts_and_is_deterministic():
    y = np.array([0, 1, 0, 1, 0, 1])
    a = he.bootstrap_weights(y, resamples=200, seed=42)
    assert np.array_equal(a, he.bootstrap_weights(y, resamples=200, seed=42))
    assert (a[:, y == 0].sum(axis=1) == 3).all() and (a[:, y == 1].sum(axis=1) == 3).all()
    assert not np.array_equal(a[0], a[1])


def test_vectorized_auc_handles_ties_and_matches_explicit_patient_resampling():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([.2, .2, .5, .5, .8, .8])
    counts = he.bootstrap_weights(y, resamples=200)
    result = he.auc_from_counts(y, p, counts)
    for index in range(20):
        repeated = np.repeat(np.arange(len(y)), counts[index])
        assert result[index] == pytest.approx(roc_auc_score(y[repeated], p[repeated]))


def test_all_vectorized_metrics_match_existing_probability_metric_functions():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0., .2, .6, .4, .8, 1.])
    counts = he.bootstrap_weights(y, resamples=200)
    draws = he.metric_draws(y, p, counts)
    for index in range(10):
        repeat = np.repeat(np.arange(len(y)), counts[index])
        frame = pd.DataFrame({"actual": y[repeat], "prob_malignant": p[repeat], "predicted": (p[repeat] >= .5).astype(int)})
        expected = pc.probability_metrics(frame)
        for metric in he.METRICS:
            assert draws[metric][index] == pytest.approx(expected[metric])


def fake_predictions():
    rows = []
    for name in he.MODEL_NAMES:
        for condition in he.CONDITIONS:
            for case in range(8):
                p = .2 if case % 2 == 0 else .8
                rows.append({"model": name, "condition": condition, "Case": case,
                             "actual": case % 2, "n_images": 1, "prob_malignant": p, "predicted": int(p >= .5)})
    return pd.DataFrame(rows)


def test_identical_models_conditions_give_exact_zero_paired_contrasts():
    intervals, _, _ = he.paired_intervals(fake_predictions(), resamples=200)
    differences = intervals.loc[intervals.comparison.isin(["change_from_clean", "svm_minus_logreg", "svm_minus_logreg_change", "svm_minus_logreg_five_condition_mean"])]
    assert np.allclose(differences[["estimate", "ci_low", "ci_high"]], 0)
    assert len(intervals) == 181 and intervals.primary.sum() == 1


def test_nonpaired_patient_metadata_rejected_before_bootstrap():
    frame = fake_predictions()
    frame.loc[0, "n_images"] = 2
    with pytest.raises(ValueError, match="patient-paired"):
        he.paired_intervals(frame, resamples=200)


@pytest.mark.parametrize("corruption", ["protocol", "model", "source", "version", "archive", "manifest"])
def test_integrity_corruption_stops_before_image_access_and_marker(checkpoint, corruption, monkeypatch):
    cal, manifest, archive, audit, _ = checkpoint
    if corruption == "model":
        with (cal / "frozen_calibrated_models.joblib").open("ab") as handle:
            handle.write(b"corrupt")
    elif corruption == "archive":
        with archive.open("ab") as handle:
            handle.write(b"corrupt")
    elif corruption == "manifest":
        with manifest.open("a") as handle:
            handle.write("\n")
    else:
        path = cal / ("calibration_summary.json" if corruption == "version" else "final_evaluation_protocol.json")
        value = he.read_json(path)
        if corruption == "protocol":
            value["threshold"] = .4
        elif corruption == "source":
            value["source_sha256"]["features.py"] = "invalid"
        else:
            value["versions"]["numpy"] = "0.0"
        he.write_json(path, value)
        if corruption == "source":
            summary = he.read_json(cal / "calibration_summary.json")
            summary["final_protocol_sha256"] = pc.sha256(path)
            he.write_json(cal / "calibration_summary.json", summary)
    def forbidden(*args, **kwargs):
        raise AssertionError("Corrupt checkpoint must not access held-out images")
    monkeypatch.setattr(he, "extract_test_features", forbidden)
    with pytest.raises(ValueError):
        he.run(*checkpoint)
    assert not (cal / he.MARKER).exists()


def test_missing_coherent_condition_fails_without_publishing_predictions(checkpoint, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("Unavailable coherent condition")
    def no_predictions(*args, **kwargs):
        raise AssertionError("Never predict an incomplete cohort")
    monkeypatch.setattr(he, "coherent_boundary_change", unavailable)
    monkeypatch.setattr(he, "predict_all", no_predictions)
    with pytest.raises(ValueError, match="Unavailable"):
        he.run(*checkpoint)
    marker = he.read_json(checkpoint[0] / he.MARKER)
    assert marker["status"] == "failed" and marker["phase"] == "feature_extraction"
    assert not checkpoint[-1].exists()


def test_marker_from_an_earlier_failure_blocks_silent_restart(checkpoint):
    he.write_json(checkpoint[0] / he.MARKER, {"status": "failed", "phase": "feature_extraction"})
    with pytest.raises(FileExistsError, match="already started"):
        he.run(*checkpoint)


def test_changed_heldout_cases_fail_even_if_manifest_fingerprint_is_refreshed(checkpoint):
    calibration, manifest_path = checkpoint[:2]
    manifest = pd.read_csv(manifest_path)
    manifest.loc[manifest.Case.eq(100), "Case"] = 9999
    manifest.to_csv(manifest_path, index=False)
    protocol_path = calibration / "final_evaluation_protocol.json"
    protocol = he.read_json(protocol_path)
    protocol["inputs"]["manifest"]["sha256"] = pc.sha256(manifest_path)
    he.write_json(protocol_path, protocol)
    summary_path = calibration / "calibration_summary.json"
    summary = he.read_json(summary_path)
    summary["final_protocol_sha256"] = pc.sha256(protocol_path)
    he.write_json(summary_path, summary)
    with pytest.raises(ValueError, match="frozen identifiers"):
        he.run(*checkpoint)
    assert not (calibration / he.MARKER).exists()


def test_existing_output_is_never_overwritten(checkpoint):
    checkpoint[-1].mkdir()
    with pytest.raises(FileExistsError, match="output exists"):
        he.run(*checkpoint)


def test_bootstrap_requires_both_diagnoses():
    with pytest.raises(ValueError, match="both diagnoses"):
        he.bootstrap_weights(np.zeros(8, dtype=int), resamples=200)


def test_invalid_probability_cannot_enter_bootstrap():
    with pytest.raises(ValueError, match="probability"):
        he.metric_draws(np.array([0, 1]), np.array([-.1, 1.1]), np.array([[1, 1]]))
