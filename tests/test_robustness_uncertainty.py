"""Step 17 tests: patient pairing, stratification, reference integrity and no fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import robustness_uncertainty as ru
from src.robustness_eval import summarize_pair


@pytest.fixture
def checkpoint(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    mdir, cdir = tmp_path / "modeling", tmp_path / "step16"
    mdir.mkdir()
    cdir.mkdir()
    manifest = pd.DataFrame([{"ID": f"image_{case}_{view}", "Case": case,
                             "Pathology": "malignant" if (case // 5) % 2 else "benign",
                             "split": "development", "dev_fold": case % 5}
                            for case in range(30) for view in range(2)])
    manifest.to_csv(manifest_path, index=False)
    rows = []
    for protocol in (ru.ORIGINAL_PROTOCOL, ru.PROTOCOL):
        for name in ru.MODEL_NAMES:
            for condition in ru.COHERENT_CONDITIONS:
                for case in range(30):
                    actual = (case // 5) % 2
                    clean_pred = actual if case % (9 if name == "logreg" else 8) else 1 - actual
                    flip = case % (4 if protocol == ru.ORIGINAL_PROTOCOL else 11) == 0
                    pert_pred = 1 - clean_pred if flip else clean_pred
                    score = lambda p: (0.8 if p else 0.2) if name == "logreg" else (1.0 if p else -1.0)
                    rows.append({"protocol": protocol, "model": name, "condition": condition,
                                 "Case": case, "actual": actual, "outer_fold": case % 5,
                                 "n_images": 2, "eligible": True, "mean_iou": 0.95,
                                 "clean_score": score(clean_pred), "perturbed_score": score(pert_pred),
                                 "clean_predicted": clean_pred, "perturbed_predicted": pert_pred})
    frame = pd.DataFrame(rows)
    original = tmp_path / "original.csv"
    frame.loc[frame.protocol.eq(ru.ORIGINAL_PROTOCOL)].drop(columns="protocol").to_csv(original, index=False)
    coherent = frame.loc[frame.protocol.eq(ru.PROTOCOL)].copy()
    coherent.to_csv(cdir / "coherent_paired_patient_predictions.csv", index=False)
    for name in ru.MODEL_NAMES:
        part = coherent.loc[coherent.model.eq(name) & coherent.condition.eq(ru.COHERENT_CONDITIONS[0])]
        baseline = part[["Case", "actual", "outer_fold", "n_images", "clean_score", "clean_predicted"]].rename(
            columns={"clean_score": "prob_malignant" if name == "logreg" else "decision_score", "clean_predicted": "predicted"})
        baseline.to_csv(mdir / f"{name}_nested_cv_predictions.csv", index=False)
    paths = {"manifest": manifest_path, "step14_predictions": original,
             **{f"{name}_predictions": mdir / f"{name}_nested_cv_predictions.csv" for name in ru.MODEL_NAMES}}
    summary = {"protocol": ru.PROTOCOL, "development_patients": 30, "development_images": 60,
               "clean_reference_oof": {name: "MATCHED" for name in ru.MODEL_NAMES},
               "step14_clean_predictions": "MATCHED", "held_out_features_images_predictions_used": False,
               "inputs": {key: {"sha256": ru.sha256(path)} for key, path in paths.items()},
               "results": [summarize_pair(part, model=name, condition=cond)
                           for (name, cond), part in coherent.groupby(["model", "condition"])]}
    (cdir / "coherent_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return manifest_path, original, cdir, mdir, tmp_path / "step17"


def test_patient_contributions_reproduce_existing_paired_metrics(checkpoint):
    patients = ru.load_patients(checkpoint[0])
    coherent = ru.load_predictions(checkpoint[2] / "coherent_paired_patient_predictions.csv", patients, coherent=True)
    part = coherent.loc[coherent.model.eq("logreg") & coherent.condition.eq("erosion_mild")]
    known = summarize_pair(part, model="logreg", condition="erosion_mild")
    values = ru.patient_values(part).mean(axis=0)
    assert values[:4] == pytest.approx([known[key] for key in ru.METRICS[:4]])
    assert values[4] == pytest.approx(known["new_false_negatives"] / part.actual.sum())


def test_paired_resampling_cancels_identical_models_and_is_deterministic():
    labels = np.array([label for fold in range(5) for label in (0, 1) for _ in range(3)])
    folds = np.repeat(np.arange(5), 6)
    v = np.arange(len(labels), dtype=float)
    values = np.column_stack((v, v, labels, folds))
    a = ru.paired_draws(values, labels, folds, n_resamples=300, seed=17)
    b = ru.paired_draws(values, labels, folds, n_resamples=300, seed=17)
    assert np.array_equal(a, b)
    assert np.array_equal(a[:, 0], a[:, 1])
    # Every replicate preserves observed class/fold composition exactly.
    assert np.allclose(a[:, 2], labels.mean())
    assert np.allclose(a[:, 3], folds.mean())
    assert np.std(a[:, 0]) > 0


def test_full_checkpoint_reads_only_saved_tables_and_preserves_inputs(checkpoint, monkeypatch):
    before = {p: ru.sha256(p) for root in (checkpoint[2], checkpoint[3]) for p in root.iterdir()}
    before.update({p: ru.sha256(p) for p in checkpoint[:2]})
    allowed = {p.resolve() for p in before if p.suffix == ".csv"}
    read_csv = pd.read_csv
    def guard(path, *args, **kwargs):
        assert Path(path).resolve() in allowed
        return read_csv(path, *args, **kwargs)
    import src.models as models
    import src.svm_model as svm
    def no_fit(*args, **kwargs):
        raise AssertionError("Step 17 must not train models")
    with monkeypatch.context() as ctx:
        ctx.setattr(pd, "read_csv", guard)
        ctx.setattr(models, "fit_model", no_fit)
        ctx.setattr(svm, "fit_svm", no_fit)
        result = ru.run(*checkpoint, n_resamples=200)
    assert result["refitting_performed"] is False
    assert result["held_out_evaluation_used"] is False
    assert result["interval_rows"] == 140
    assert before == {p: ru.sha256(p) for p in before}
    table = pd.read_csv(checkpoint[-1] / "paired_bootstrap_intervals.csv")
    assert len(table) == 140 and table.common_patients.eq(30).all()
    assert table.ci_low.le(table.ci_high).all()
    assert not table.duplicated(["comparison", "model", "protocol", "condition", "metric"]).any()
    assert len(pd.read_csv(checkpoint[-1] / "common_patient_coverage.csv")) == 40
    repeat = list(checkpoint)
    repeat[-1] = checkpoint[-1].with_name("repeat")
    ru.run(*repeat, n_resamples=200)
    assert (checkpoint[-1] / "paired_bootstrap_intervals.csv").read_bytes() == (repeat[-1] / "paired_bootstrap_intervals.csv").read_bytes()
    with pytest.raises(FileExistsError):
        ru.run(*checkpoint, n_resamples=200)


def test_common_cohort_excludes_entire_patient_across_all_models_and_protocols(checkpoint):
    patients = ru.load_patients(checkpoint[0])
    old = ru.load_predictions(checkpoint[1], patients, coherent=False)
    new = ru.load_predictions(checkpoint[2] / "coherent_paired_patient_predictions.csv", patients, coherent=True)
    ix = old.Case.eq(2) & old.model.eq("logreg") & old.condition.eq("erosion_mild")
    old.loc[ix, "eligible"] = False
    old.loc[ix, ["perturbed_score", "perturbed_predicted"]] = np.nan
    rows, coverage = ru.analyze_condition(old, new, "erosion_mild", n_resamples=200, seed=42)
    assert {row["common_patients"] for row in rows} == {29}
    assert {row["excluded_from_common"] for row in rows} == {1}
    assert sum(row["common_patients"] for row in coverage) == 29


def test_identical_protocols_have_exact_zero_contrasts(checkpoint):
    patients = ru.load_patients(checkpoint[0])
    new = ru.load_predictions(checkpoint[2] / "coherent_paired_patient_predictions.csv", patients, coherent=True)
    rows, _ = ru.analyze_condition(new, new, "erosion_mild", n_resamples=200, seed=42)
    differences = [row for row in rows if "coherent_minus_original" in row["comparison"]]
    assert len(differences) == 15
    for row in differences:
        assert row["estimate"] == pytest.approx(0, abs=1e-14)
        assert row["ci_low"] == pytest.approx(0, abs=1e-14)
        assert row["ci_high"] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "heldout", "fold", "threshold", "nonfinite", "unavailable", "protocol"])
def test_invalid_coherent_predictions_aborts_without_outputs(checkpoint, corruption):
    path = checkpoint[2] / "coherent_paired_patient_predictions.csv"
    frame = pd.read_csv(path)
    if corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif corruption == "missing":
        frame = frame.iloc[:-1]
    elif corruption == "heldout":
        frame.loc[0, "Case"] = 9999
    elif corruption == "fold":
        frame.loc[0, "outer_fold"] = 4
    elif corruption == "threshold":
        frame.loc[0, "perturbed_predicted"] = 1 - frame.loc[0, "perturbed_predicted"]
    elif corruption == "nonfinite":
        frame.loc[0, "perturbed_score"] = np.inf
    elif corruption == "unavailable":
        frame.loc[0, "eligible"] = False
    else:
        frame.loc[0, "protocol"] = "unverified"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        ru.run(*checkpoint, n_resamples=200)
    assert not checkpoint[-1].exists()


@pytest.mark.parametrize("name", ru.MODEL_NAMES)
def test_stale_clean_reference_hash_aborts(checkpoint, name):
    path = checkpoint[3] / f"{name}_nested_cv_predictions.csv"
    frame = pd.read_csv(path)
    col = "prob_malignant" if name == "logreg" else "decision_score"
    frame.loc[0, col] += 0.05
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        ru.run(*checkpoint, n_resamples=200)


def test_modified_step16_summary_aborts(checkpoint):
    path = checkpoint[2] / "coherent_summary.json"
    summary = json.loads(path.read_text())
    summary["results"][0]["delta_balanced_accuracy"] += 0.1
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="summary mismatch"):
        ru.run(*checkpoint, n_resamples=200)
    assert not checkpoint[-1].exists()


def test_manifest_cross_split_patient_rejected(checkpoint):
    path = checkpoint[0]
    frame = pd.read_csv(path)
    extra = frame.iloc[[0]].copy()
    extra["ID"], extra["split"], extra["dev_fold"] = "heldout_view", "test", np.nan
    pd.concat([frame, extra]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Leakage"):
        ru.run(*checkpoint, n_resamples=200)


def test_bootstrap_requires_every_fold_diagnosis_stratum():
    y, folds = np.array([0, 1]), np.array([0, 0])
    with pytest.raises(ValueError, match="every locked fold"):
        ru.paired_draws(np.ones((2, 1)), y, folds, n_resamples=200, seed=42)


@pytest.mark.parametrize("resamples,seed", [(1, 42), (200, -1)])
def test_invalid_run_controls_rejected(checkpoint, resamples, seed):
    with pytest.raises(ValueError):
        ru.run(*checkpoint, n_resamples=resamples, seed=seed)
