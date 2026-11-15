"""Step 22 tests protect reporting integrity and prohibit new model inference."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import final_report as fr
from src import heldout_evaluation as he
from src import patient_calibration as pc
from src import stable_selection as ss
from test_heldout_evaluation import reference_checkpoint as heldout_reference


@pytest.fixture(scope="module")
def completed_study(heldout_reference, tmp_path_factory):
    source = heldout_reference
    raw_output = source / "completed_fake_evaluation"
    he.run(source / "calibration", source / "manifest.csv", source / "fake.zip", source / "audit.json", raw_output)
    root = tmp_path_factory.mktemp("synthetic_report_study")
    robust = root / "results/robustness"
    robust.mkdir(parents=True)
    (root / "splits").mkdir()
    shutil.copyfile(source / "manifest.csv", root / "splits/image_splits.csv")
    (root / "splits/split_summary.json").write_text("{}")
    shutil.copytree(source / "calibration", robust / "calibration_step20")
    shutil.copytree(raw_output, robust / "heldout_step21")
    pair_table = pd.read_csv(raw_output / "held_out_paired_summary.csv")
    pair_table.to_csv(robust / "robustness_paired_condition_summary.csv", index=False)
    for directory in ("selection_step19", "coherent_step16_run2", "uncertainty_step17"):
        (robust / directory).mkdir()
    pair_table.to_csv(robust / "coherent_step16_run2/coherent_condition_summary.csv", index=False)
    pd.DataFrame({"estimate": [0.]}).to_csv(robust / "uncertainty_step17/paired_bootstrap_intervals.csv", index=False)
    selection = []
    selections = []
    for name in fr.MODEL_NAMES:
        for arm, value in (("all_23_matched_tuning", .85), ("stability_topk", .82)):
            selection.append({"model": name, "arm": arm, "balanced_accuracy": value, "mean_five_condition_ba": value - .01})
            selections.extend({"model": name, "arm": arm, "outer_fold": fold, "k": 23 if arm == "all_23_matched_tuning" else 15} for fold in range(5))
    pd.DataFrame(selection).to_csv(robust / "selection_step19/clean_summary.csv", index=False)
    (robust / "selection_step19/outer_selections.json").write_text(json.dumps(selections))
    (robust / "selection_step19/evaluation_summary.json").write_text(json.dumps({"fit_counts": {"inner": 1680, "outer_refits": 20, "baseline_reproducibility": 10}, "elapsed_nested_seconds": 123.4}))
    cal = robust / "calibration_step20"
    clean_metrics = pd.read_csv(raw_output / "held_out_condition_metrics.csv").query("condition == 'clean'")
    clean_metrics.to_csv(cal / "calibrated_clean_summary.csv", index=False)
    summary = fr.read_json(cal / "calibration_summary.json")
    summary.update(base_inner_fits=1512, base_refits=36, sigmoid_fits=36)
    (cal / "calibration_summary.json").write_text(json.dumps(summary))
    modeling = root / "results/modeling"
    modeling.mkdir()
    for name in fr.MODEL_NAMES:
        (modeling / f"{name}_nested_cv_summary.json").write_text(json.dumps({"balanced_accuracy": .85, "roc_auc": .91}))
    return root


@pytest.fixture
def checkpoint(completed_study, tmp_path):
    root = tmp_path / "study"
    shutil.copytree(completed_study, root)
    return root, tmp_path / "report"


def refresh_published(paths, filename):
    marker_path = paths["calibration"] / "held_out_evaluation_started.json"
    marker = fr.read_json(marker_path)
    marker["published_files_sha256"][filename] = pc.sha256(paths["heldout"] / filename)
    marker_path.write_text(ss.canonical_json(marker))


@pytest.mark.parametrize("status", ["started", "failed"])
def test_incomplete_heldout_is_never_reported_as_complete(checkpoint, status):
    paths = fr.source_paths(checkpoint[0])
    marker = fr.read_json(paths["calibration"] / "held_out_evaluation_started.json")
    marker["status"] = status
    (paths["calibration"] / "held_out_evaluation_started.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="not complete"):
        fr.validate_saved_evaluation(paths)


@pytest.mark.parametrize("filename", ["held_out_condition_metrics.csv", "held_out_intervals.csv", "held_out_patient_predictions.csv", "held_out_paired_predictions.csv"])
def test_changed_published_results_rejected_by_checksum(checkpoint, filename):
    paths = fr.source_paths(checkpoint[0])
    with (paths["heldout"] / filename).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        fr.validate_saved_evaluation(paths)


@pytest.mark.parametrize("filename", ["frozen_calibrated_models.joblib", "final_evaluation_protocol.json"])
def test_frozen_models_or_protocol_changes_rejected(checkpoint, filename):
    paths = fr.source_paths(checkpoint[0])
    with (paths["calibration"] / filename).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        fr.validate_saved_evaluation(paths)


@pytest.mark.parametrize("metric", ["balanced_accuracy", "brier_score", "fn"])
def test_metric_inconsistency_rejected_even_with_refreshed_file_hash(checkpoint, metric):
    paths = fr.source_paths(checkpoint[0])
    file = paths["heldout"] / "held_out_condition_metrics.csv"
    table = pd.read_csv(file)
    table.loc[0, metric] += 1 if metric == "fn" else .1
    table.to_csv(file, index=False)
    refresh_published(paths, file.name)
    with pytest.raises(ValueError, match="metric mismatch"):
        fr.validate_saved_evaluation(paths)


@pytest.mark.parametrize("column", ["clean_score", "perturbed_score"])
def test_broken_patient_pairing_rejected_even_with_refreshed_hash(checkpoint, column):
    paths = fr.source_paths(checkpoint[0])
    file = paths["heldout"] / "held_out_paired_predictions.csv"
    table = pd.read_csv(file)
    table.loc[0, column] += .1
    table.to_csv(file, index=False)
    refresh_published(paths, file.name)
    with pytest.raises(ValueError, match="pairing differs"):
        fr.validate_saved_evaluation(paths)


def test_changed_primary_endpoint_is_not_silently_reinterpreted(checkpoint):
    paths = fr.source_paths(checkpoint[0])
    file = paths["heldout"] / "held_out_intervals.csv"
    table = pd.read_csv(file)
    table.loc[table.primary, "model"] = "logreg"
    table.to_csv(file, index=False)
    refresh_published(paths, file.name)
    with pytest.raises(ValueError, match="Primary endpoint"):
        fr.validate_saved_evaluation(paths)


def test_changed_threshold_predictions_are_rejected(checkpoint):
    paths = fr.source_paths(checkpoint[0])
    file = paths["heldout"] / "held_out_patient_predictions.csv"
    table = pd.read_csv(file)
    table.loc[0, "predicted"] = 1 - table.loc[0, "predicted"]
    table.to_csv(file, index=False)
    refresh_published(paths, file.name)
    with pytest.raises(ValueError, match="threshold"):
        fr.validate_saved_evaluation(paths)


def test_failure_audit_preserves_every_patient_and_error_overlap(checkpoint):
    context = fr.validate_saved_evaluation(fr.source_paths(checkpoint[0]))
    cases, overlap, examples = fr.failure_analysis(context["predictions"], context["manifest"])
    assert len(cases) == context["summary"]["held_out_patients"]
    assert not cases.Case.duplicated().any()
    assert sum(overlap[k] for k in ("both_wrong", "lr_only_wrong", "svm_only_wrong", "both_correct")) == len(cases)
    assert overlap["malignant_patients"] == cases.actual.sum()
    assert cases.image_ids.str.len().gt(0).all()
    assert len(examples) <= 6


def test_report_generation_never_fits_predicts_or_reruns_evaluation(checkpoint, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    before = {p: pc.sha256(p) for p in checkpoint[0].rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Reporting must not fit, predict or rerun held-out evaluation")
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(SVC, "fit", forbidden)
    monkeypatch.setattr(pc, "predict_ensemble", forbidden)
    monkeypatch.setattr(he, "run", forbidden)
    result = fr.run(*checkpoint)
    assert result["model_fits"] == result["new_predictions"] == result["planned_experiments_remaining"] == 0
    assert result["heldout_evaluation_rerun"] is False
    assert before == {p: pc.sha256(p) for p in before}
    assert (checkpoint[1] / "BUS_BRA_Final_Report.md").is_file()
    html = (checkpoint[1] / "BUS_BRA_Final_Report.html").read_text(encoding="utf-8")
    assert html.count("data:image/png;base64,") == 7
    assert result["primary_result"] == fr.read_json(fr.source_paths(checkpoint[0])["heldout"] / "held_out_summary.json")["primary_result"]


def test_model_size_inspection_only_reads_fitted_structure(checkpoint, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Model structure inspection needs no inference")
    monkeypatch.setattr(pc, "predict_ensemble", forbidden)
    sizes = fr.model_sizes(fr.source_paths(checkpoint[0])["calibration"])
    assert len(sizes) == 6 and sizes.features.eq(23).all()
    assert sizes.loc[sizes.model.eq("logreg"), "linear_coefficients"].sum() == 69
    assert sizes.loc[sizes.model.eq("svm"), "support_vectors"].gt(0).all()


def test_patient_metadata_cannot_change_under_refreshed_result_hash(checkpoint):
    paths = fr.source_paths(checkpoint[0])
    file = paths["heldout"] / "held_out_patient_predictions.csv"
    table = pd.read_csv(file)
    table.loc[0, "n_images"] += 1
    table.to_csv(file, index=False)
    refresh_published(paths, file.name)
    with pytest.raises(ValueError, match="metadata"):
        fr.validate_saved_evaluation(paths)


def test_existing_report_is_never_overwritten(checkpoint):
    checkpoint[1].mkdir()
    with pytest.raises(FileExistsError):
        fr.run(*checkpoint)
