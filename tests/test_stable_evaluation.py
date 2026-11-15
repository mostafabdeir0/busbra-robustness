"""Step 19: nested leakage controls, fair tuning, score parity and locked design."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import stable_evaluation as se
from src import stable_selection as ss
from src.models import fit_model, patient_image_weights
from src.svm_model import fit_svm
from test_stable_selection import feature_tables


def small_protocol():
    protocol = ss.locked_protocol()
    protocol["logreg"] = {"C": [1.0], "balance": [False]}
    protocol["svm"] = {"C": [1.0], "balance": [False], "gamma": ["scale"]}
    return protocol


@pytest.fixture
def checkpoint(tmp_path):
    clean, coherent = feature_tables()
    clean_path, manifest, coherent_path, geometry = [tmp_path / name for name in ("clean.csv", "manifest.csv", "coherent.csv", "geometry.csv")]
    clean.to_csv(clean_path, index=False)
    clean = pd.read_csv(clean_path)
    clean[["ID", "Case", "Pathology", "split", "dev_fold"]].to_csv(manifest, index=False)
    coherent.to_csv(coherent_path, index=False)
    audit = coherent[["ID", "condition", "valid", "actual_iou", "changed_pixels"]].copy()
    audit["circularity_clean"] = clean.set_index("ID").loc[coherent.ID, "shape_circularity"].to_numpy(float)
    audit["circularity_coherent_protocol"] = coherent.shape_circularity
    audit.to_csv(geometry, index=False)
    design, modeling, out = tmp_path / "design", tmp_path / "modeling", tmp_path / "step19"
    ss.run(clean_path, manifest, coherent_path, geometry, design)
    modeling.mkdir()
    for name in se.MODEL_NAMES:
        parts, settings = [], []
        for fold in range(5):
            setup = {"C": 1.0, "balance": False}
            if name == "svm":
                setup["gamma"] = "scale"
            model = se.fit_subset(clean.loc[clean.dev_fold.ne(fold)], name, se.FEATURE_COLUMNS, setup)
            parts.append(se._predict(model, clean.loc[clean.dev_fold.eq(fold)], name, fold))
            settings.append({"outer_fold": fold, **setup})
        pd.concat(parts).to_csv(modeling / f"{name}_nested_cv_predictions.csv", index=False)
        pd.DataFrame(settings).to_csv(modeling / f"{name}_nested_cv_folds.csv", index=False)
    return clean_path, manifest, coherent_path, geometry, design, modeling, out


def test_nested_engine_reranks_each_scope_and_fits_clean_weighted_partitions(monkeypatch):
    clean, coherent = feature_tables()
    real_fit, real_rank = se.fit_subset, ss.rank_training_features
    fit_cases, rank_scopes = [], []
    lookup = clean.set_index("ID")
    def spy_fit(train, name, columns, settings):
        pd.testing.assert_frame_equal(train.set_index("ID"), lookup.loc[train.ID], check_names=False)
        model = real_fit(train, name, columns, settings)
        weights = patient_image_weights(train, balance=settings["balance"])
        expected = np.average(train[list(columns)].to_numpy(float), weights=weights, axis=0)
        assert np.allclose(model.pipeline.named_steps["scaler"].mean_, expected)
        fit_cases.append(set(train.Case))
        return model
    def spy_rank(dev, pert, *, training_cases, forbidden_cases):
        assert training_cases.isdisjoint(forbidden_cases)
        assert training_cases | forbidden_cases == set(clean.Case)
        rank_scopes.append((set(training_cases), set(forbidden_cases)))
        return real_rank(dev, pert, training_cases=training_cases, forbidden_cases=forbidden_cases)
    monkeypatch.setattr(se, "fit_subset", spy_fit)
    monkeypatch.setattr(ss, "rank_training_features", spy_rank)
    result = se.nested_evaluate(clean, coherent, small_protocol())
    assert len(rank_scopes) == 20
    assert result["fit_counts"] == {"inner": 120, "outer_refits": 20}
    assert len(fit_cases) == 140
    assert len(result["selections"]) == 20
    tables = result["tables"]
    assert len(tables["clean_oof_predictions.csv"]) == 30 * 4
    assert len(tables["coherent_paired_predictions.csv"]) == 30 * 16
    assert len(tables["recomputed_training_rankings.csv"]) == 460
    assert len(tables["inner_candidate_summary.csv"]) == 40
    assert len(tables["coherent_summary.csv"]) == 16
    assert len(tables["coherent_outer_fold_metrics.csv"]) == 80
    assert len(tables["arm_comparison.csv"]) == 8
    for selection in result["selections"]:
        assert len(selection["features"]) == selection["k"]
        scope = next(s for s in result["scopes"] if s["outer_fold"] == selection["outer_fold"] and s["inner_fold"] == -1)
        assert selection["training_case_sha256"] == scope["training_case_sha256"]
    freq = tables["feature_selection_frequency.csv"]
    assert freq.loc[freq.arm.eq(se.ARMS[0]), "selected_outer_folds"].eq(5).all()


@pytest.mark.parametrize("name", se.MODEL_NAMES)
def test_all23_adapter_reproduces_existing_model_and_patient_predictions(name):
    clean, _ = feature_tables()
    train, val = clean.loc[clean.dev_fold.ne(0)], clean.loc[clean.dev_fold.eq(0)]
    setup = {"C": 1.0, "balance": True}
    if name == "svm":
        setup["gamma"] = "scale"
    reference = fit_model(train, **setup) if name == "logreg" else fit_svm(train, **setup)
    adapted = se.fit_subset(train, name, se.FEATURE_COLUMNS, setup)
    pd.testing.assert_frame_equal(se._predict(adapted, val, name, 0), se._predict(reference, val, name, 0))


@pytest.mark.parametrize("name", se.MODEL_NAMES)
def test_stacked_inner_scores_match_existing_patient_aggregation(name):
    clean, coherent = feature_tables()
    train, val = clean.loc[clean.dev_fold.ne(0)], clean.loc[clean.dev_fold.eq(0)]
    pert = coherent.loc[coherent.dev_fold.eq(0)]
    setup = {"C": 1.0, "balance": False, **({"gamma": "scale"} if name == "svm" else {})}
    model = se.fit_subset(train, name, se.FEATURE_COLUMNS[:5], setup)
    score = se.evaluate_inner(model, name, se.validation_pack(val, pert))
    individual = []
    for frame in [val] + [pert.loc[pert.condition.eq(c)] for c in se.COHERENT_CONDITIONS]:
        p = se._predict(model, frame, name, 0)
        key = "prob_malignant" if name == "logreg" else "decision_score"
        individual.append(se.metrics(p, key, "predicted"))
    assert score["objective"] == pytest.approx(np.mean([m["balanced_accuracy"] for m in individual]))
    assert score["clean_ba"] == individual[0]["balanced_accuracy"]
    assert score["clean_auc"] == individual[0]["roc_auc"]


def test_outer_validation_values_do_not_change_inner_rankings_or_candidate_choice():
    clean, coherent = feature_tables()
    contexts, _, _ = se.build_contexts(clean, coherent, 0)
    chosen, candidates, _ = se.tune_arm(contexts, "logreg", se.ARMS[1], small_protocol())
    clean.loc[clean.dev_fold.eq(0), list(se.FEATURE_COLUMNS)] = np.nan
    coherent.loc[coherent.dev_fold.eq(0), list(se.FEATURE_COLUMNS)] = np.inf
    clean.loc[clean.dev_fold.eq(0), "Pathology"] = "excluded"
    coherent.loc[coherent.dev_fold.eq(0), "Pathology"] = "excluded"
    altered, _, _ = se.build_contexts(clean, coherent, 0)
    other, records, _ = se.tune_arm(altered, "logreg", se.ARMS[1], small_protocol())
    assert other == chosen and records == candidates
    for a, b in zip(contexts, altered):
        pd.testing.assert_frame_equal(a["ranking"], b["ranking"])


def test_tie_break_respects_every_frozen_priority():
    base = {"mean_objective": .8, "mean_clean_ba": .8, "mean_clean_auc": .9,
            "k": 10, "settings": {"C": 1.0, "balance": False}, "gamma_index": 0}
    better = []
    for key, value in (("mean_objective", .81), ("mean_clean_ba", .81), ("mean_clean_auc", .91), ("k", 5)):
        row = copy.deepcopy(base)
        row[key] = value
        better.append(row)
        assert se.candidate_key(row) > se.candidate_key(base)
    row = copy.deepcopy(base)
    row["settings"]["C"] = .1
    assert se.candidate_key(row) > se.candidate_key(base)
    row = copy.deepcopy(base)
    row["settings"]["balance"] = True
    assert se.candidate_key(row) < se.candidate_key(base)
    row = copy.deepcopy(base)
    row["gamma_index"] = 1
    assert se.candidate_key(row) < se.candidate_key(base)
    assert max(better, key=se.candidate_key)["mean_objective"] == .81


def test_full_run_checks_locked_design_and_publishes_only_new_outputs(checkpoint, monkeypatch):
    inputs = [*checkpoint[:4], *checkpoint[4].iterdir(), *checkpoint[5].iterdir()]
    before = {p: se.sha256(p) for p in inputs}
    read_csv = pd.read_csv
    def guard(path, *args, **kwargs):
        assert Path(path).resolve() in {p.resolve() for p in inputs}
        return read_csv(path, *args, **kwargs)
    real = se.nested_evaluate
    def reduced(dev, coherent, protocol):
        assert protocol == ss.locked_protocol()
        return real(dev, coherent, small_protocol())
    with monkeypatch.context() as ctx:
        ctx.setattr(pd, "read_csv", guard)
        ctx.setattr(se, "nested_evaluate", reduced)
        summary = se.run(*checkpoint)
    assert summary["original_clean_oof"] == {"logreg": "MATCHED", "svm": "MATCHED"}
    assert summary["held_out_evaluation_used"] is False
    assert before == {p: se.sha256(p) for p in inputs}
    assert (checkpoint[-1] / "outer_selections.json").is_file()
    assert len(json.loads((checkpoint[-1] / "training_scopes.json").read_text())) == 20
    with pytest.raises(FileExistsError):
        se.run(*checkpoint)


@pytest.mark.parametrize("corruption", ["protocol", "summary", "source", "clean_hash", "coherent_hash"])
def test_frozen_checkpoint_corruption_aborts_before_fitting(checkpoint, corruption, monkeypatch):
    directory = checkpoint[4]
    path = directory / ("selection_protocol.json" if corruption == "protocol" else "selection_design_summary.json")
    content = json.loads(path.read_text())
    if corruption == "protocol":
        content["top_k_candidates"] = [6, 12]
    elif corruption == "summary":
        content["all_scopes_patient_disjoint"] = False
    elif corruption == "source":
        content["source_sha256"] = "wrong"
    else:
        content["inputs"]["clean_features" if corruption == "clean_hash" else "coherent_features"]["sha256"] = "wrong"
    path.write_text(json.dumps(content))
    def no_fit(*args, **kwargs):
        raise AssertionError("Corruption must fail before fitting")
    monkeypatch.setattr(se, "fit_subset", no_fit)
    with pytest.raises(ValueError, match="Step 18"):
        se.run(*checkpoint)
    assert not checkpoint[-1].exists()


@pytest.mark.parametrize("name", se.MODEL_NAMES)
def test_stale_original_reference_aborts_before_candidate_search(checkpoint, name, monkeypatch):
    path = checkpoint[5] / f"{name}_nested_cv_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "prob_malignant" if name == "logreg" else "decision_score"] += .1
    frame.to_csv(path, index=False)
    def no_search(*args, **kwargs):
        raise AssertionError("Stale baseline must abort before candidate search")
    monkeypatch.setattr(se, "nested_evaluate", no_search)
    with pytest.raises(ValueError, match="score mismatch"):
        se.run(*checkpoint)
    assert not checkpoint[-1].exists()


def test_missing_inner_variant_fails_closed():
    clean, coherent = feature_tables()
    with pytest.raises(ValueError):
        se.validation_pack(clean, coherent.iloc[:-1])


def test_no_subset_candidate_never_silently_uses_all23():
    clean, coherent = feature_tables()
    contexts, _, _ = se.build_contexts(clean, coherent, 0)
    for context in contexts:
        context["ranking"]["eligible"] = False
    with pytest.raises(ValueError, match="No valid"):
        se.tune_arm(contexts, "logreg", se.ARMS[1], small_protocol())


def test_arm_comparison_rejects_nonpaired_metadata():
    left = pd.DataFrame({"Case": [1], "actual": [0], "outer_fold": [0], "n_images": [2]})
    right = left.copy()
    right["n_images"] = 1
    with pytest.raises(ValueError, match="metadata"):
        se.require_pair_metadata(left, right)


def test_subset_adapter_requires_canonical_input_width():
    clean, _ = feature_tables()
    model = se.fit_subset(clean, "logreg", se.FEATURE_COLUMNS[:5], {"C": 1.0, "balance": False})
    with pytest.raises(ValueError, match="23-column"):
        model.predict_proba(clean[list(se.FEATURE_COLUMNS[:5])].to_numpy(float))


def test_noncanonical_or_non_development_training_rejected():
    clean, _ = feature_tables()
    settings = {"C": 1.0, "balance": False}
    with pytest.raises(ValueError):
        se.fit_subset(clean, "logreg", tuple(reversed(se.FEATURE_COLUMNS[:5])), settings)
    clean.loc[0, "split"] = "test"
    with pytest.raises(ValueError):
        se.fit_subset(clean, "logreg", se.FEATURE_COLUMNS[:5], settings)
