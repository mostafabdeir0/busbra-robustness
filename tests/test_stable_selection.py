"""Step 18: scope isolation, patient weighting, deterministic ranks and frozen rules."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import stable_selection as ss
from src.features import FEATURE_COLUMNS
from test_robustness_eval import synthetic_data


def feature_tables():
    clean, pert = synthetic_data()
    pert = pert.loc[pert.condition.isin(ss.COHERENT_CONDITIONS)].copy()
    matrix = clean[list(FEATURE_COLUMNS)].to_numpy(float)
    w = (1 / clean.groupby("Case").ID.transform("size")).to_numpy(float)
    mean = np.average(matrix, axis=0, weights=w)
    sd = np.sqrt(np.average((matrix - mean) ** 2, axis=0, weights=w))
    lookup = clean.set_index("ID")
    for i, condition in enumerate(ss.COHERENT_CONDITIONS):
        ix = pert.condition.eq(condition)
        pert.loc[ix, list(FEATURE_COLUMNS)] = (
            lookup.loc[pert.loc[ix, "ID"], list(FEATURE_COLUMNS)].to_numpy(float)
            + (i + 1) * np.arange(1, 24) * 0.01 * sd)
    return clean, pert


@pytest.fixture
def checkpoint(tmp_path):
    clean, pert = feature_tables()
    clean_path, manifest, coherent, geometry = [tmp_path / name for name in ("clean.csv", "manifest.csv", "coherent.csv", "geometry.csv")]
    clean.to_csv(clean_path, index=False)
    clean[["ID", "Case", "Pathology", "split", "dev_fold"]].to_csv(manifest, index=False)
    pert.to_csv(coherent, index=False)
    audit = pert[["ID", "condition", "valid", "actual_iou", "changed_pixels"]].copy()
    audit["circularity_clean"] = clean.set_index("ID").loc[pert.ID, "shape_circularity"].to_numpy(float)
    audit["circularity_coherent_protocol"] = pert.shape_circularity
    audit.to_csv(geometry, index=False)
    return clean_path, manifest, coherent, geometry, tmp_path / "step18"


def rank(clean, pert, training=None, forbidden=None):
    return ss.rank_training_features(clean, pert, training_cases=set(clean.Case) if training is None else training,
                                     forbidden_cases=set() if forbidden is None else forbidden)


def test_known_normalized_drifts_and_worst_condition_rule():
    clean, pert = feature_tables()
    result = rank(clean, pert).set_index("feature").loc[list(FEATURE_COLUMNS)]
    assert result.stability_score.to_numpy() == pytest.approx(0.04 * np.arange(1, 24))
    assert result["rank"].tolist() == list(range(1, 24))
    assert ss.selected_columns(result.reset_index(), 5) == FEATURE_COLUMNS[:5]


def test_excluded_feature_values_and_labels_cannot_affect_rankings():
    clean, pert = feature_tables()
    train, forbidden = set(clean.loc[clean.dev_fold.ne(0), "Case"]), set(clean.loc[clean.dev_fold.eq(0), "Case"])
    expected = rank(clean, pert, train, forbidden)
    clean.loc[clean.Case.isin(forbidden), list(FEATURE_COLUMNS)] = np.nan
    pert.loc[pert.Case.isin(forbidden), list(FEATURE_COLUMNS)] = np.inf
    clean.loc[clean.Case.isin(forbidden), "Pathology"] = "not_a_training_label"
    pert.loc[pert.Case.isin(forbidden), "Pathology"] = "not_a_training_label"
    pd.testing.assert_frame_equal(rank(clean, pert, train, forbidden), expected)


def test_ranking_uses_no_diagnosis_labels_for_relevance_or_class_weighting():
    clean, pert = feature_tables()
    expected = rank(clean, pert)
    for frame in (clean, pert):
        frame["Pathology"] = frame.Pathology.map({"benign": "malignant", "malignant": "benign"})
    pd.testing.assert_frame_equal(rank(clean, pert), expected)


def test_reordering_rows_does_not_change_rankings():
    clean, pert = feature_tables()
    pd.testing.assert_frame_equal(rank(clean.sample(frac=1, random_state=4), pert.sample(frac=1, random_state=5)), rank(clean, pert))


def test_replicating_one_patients_views_does_not_increase_their_weight():
    clean, pert = feature_tables()
    before = rank(clean, pert)
    c, p = clean.loc[clean.Case.eq(2)].copy(), pert.loc[pert.Case.eq(2)].copy()
    c["ID"] = c.ID + "_copy"
    p["ID"] = p.ID + "_copy"
    after = rank(pd.concat([clean, c]), pd.concat([pert, p]))
    assert after.feature.tolist() == before.feature.tolist()
    assert np.allclose(after.clean_weighted_sd, before.clean_weighted_sd)
    assert np.allclose(after.stability_score, before.stability_score)


def test_feature_unit_changes_leave_normalized_scores_invariant():
    clean, pert = feature_tables()
    before = rank(clean, pert)
    feature = FEATURE_COLUMNS[4]
    for frame in (clean, pert):
        frame[feature] = frame[feature] * 10 + 20
    after = rank(clean, pert)
    assert after.feature.tolist() == before.feature.tolist()
    assert np.allclose(after.stability_score, before.stability_score)


def test_constant_features_excluded_and_not_selected():
    clean, pert = feature_tables()
    feature = FEATURE_COLUMNS[0]
    clean[feature], pert[feature] = 7.0, 7.0
    result = rank(clean, pert)
    row = result.loc[result.feature.eq(feature)].iloc[0]
    assert not row.eligible and pd.isna(row["rank"]) and pd.isna(row.stability_score)
    assert feature not in ss.selected_columns(result, 5)


def test_exact_score_ties_follow_canonical_feature_order():
    clean, pert = feature_tables()
    lookup = clean.set_index("ID")
    pert[list(FEATURE_COLUMNS)] = lookup.loc[pert.ID, list(FEATURE_COLUMNS)].to_numpy(float)
    result = rank(clean, pert)
    assert result.feature.tolist() == list(FEATURE_COLUMNS)
    assert result.stability_score.eq(0).all()


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "metadata", "nonfinite", "unavailable", "test"])
def test_bad_training_pairs_rejected(corruption):
    clean, pert = feature_tables()
    if corruption == "missing":
        pert = pert.iloc[:-1]
    elif corruption == "duplicate":
        pert = pd.concat([pert, pert.iloc[[0]]])
    elif corruption == "metadata":
        pert.loc[pert.index[0], "Pathology"] = "malignant"
    elif corruption == "nonfinite":
        pert.loc[pert.index[0], FEATURE_COLUMNS[0]] = np.nan
    elif corruption == "unavailable":
        pert.loc[pert.index[0], "valid"] = False
    else:
        clean.loc[0, "split"] = "test"
    with pytest.raises(ValueError):
        rank(clean, pert)


def test_overlapping_or_unknown_training_cases_rejected():
    clean, pert = feature_tables()
    with pytest.raises(ValueError, match="overlap"):
        rank(clean, pert, set(clean.Case), {0})
    with pytest.raises(ValueError, match="Missing"):
        rank(clean, pert, set(clean.Case) | {9999}, set())


def test_unsupported_k_or_insufficient_eligible_features_rejected():
    clean, pert = feature_tables()
    result = rank(clean, pert)
    with pytest.raises(ValueError, match="frozen"):
        ss.selected_columns(result, 7)
    result["eligible"] = False
    with pytest.raises(ValueError, match="Not enough"):
        ss.selected_columns(result, 5)


def test_full_audit_has_twenty_disjoint_scopes_no_fits_and_no_input_changes(checkpoint, monkeypatch):
    import src.models as models
    import src.svm_model as svm
    fingerprints = {p: ss.sha256(p) for p in checkpoint[:-1]}
    read_csv = pd.read_csv
    def guard(path, *args, **kwargs):
        assert Path(path).resolve() in {p.resolve() for p in fingerprints}
        return read_csv(path, *args, **kwargs)
    def no_fit(*args, **kwargs):
        raise AssertionError("Step 18 must not fit a classifier")
    with monkeypatch.context() as context:
        context.setattr(pd, "read_csv", guard)
        context.setattr(models, "fit_model", no_fit)
        context.setattr(svm, "fit_svm", no_fit)
        result = ss.run(*checkpoint)
    assert result["training_scopes"] == 20 and result["ranking_rows"] == 460
    assert result["classifier_fits"] == 0 and result["performance_evaluated"] is False
    assert result["held_out_evaluation_used"] is False and result["global_ranking_produced"] is False
    scopes = json.loads((checkpoint[-1] / "training_scopes.json").read_text())
    dev = pd.read_csv(checkpoint[0])
    for scope in scopes:
        train, excluded = set(scope["training_cases"]), set(scope["excluded_cases"])
        assert train.isdisjoint(excluded) and train | excluded == set(dev.Case)
        outer_val = set(dev.loc[dev.dev_fold.eq(scope["outer_fold"]), "Case"])
        assert outer_val.issubset(excluded)
        if scope["context"] == "inner_training":
            assert len(excluded) > len(outer_val)
        else:
            assert excluded == outer_val
    assert fingerprints == {p: ss.sha256(p) for p in fingerprints}
    assert ss.sha256(checkpoint[-1] / "selection_protocol.json") == result["protocol_sha256"]
    with pytest.raises(FileExistsError):
        ss.run(*checkpoint)


def test_protocol_grid_matches_existing_grids_and_freezes_fair_selection():
    from src import models, svm_model
    protocol = ss.locked_protocol()
    assert protocol["logreg"]["C"] == list(models.CS)
    assert protocol["svm"]["C"] == list(svm_model.CS)
    assert protocol["svm"]["gamma"] == list(svm_model.GAMMAS)
    assert protocol["top_k_candidates"] == [5, 10, 15]
    assert protocol["arms"] == ["all_23_matched_tuning", "stability_topk"]
    assert "three inner folds" in protocol["inner_selection_objective"]


def test_audit_skips_large_k_without_padding_and_fails_if_all_candidates_invalid():
    clean, pert = feature_tables()
    for feature in FEATURE_COLUMNS[6:]:
        clean[feature], pert[feature] = 1.0, 1.0
    _, _, candidates = ss.audit_scopes(clean, pert)
    assert all(c["valid"] == (c["k"] == 5) for c in candidates)
    for feature in FEATURE_COLUMNS[4:6]:
        clean[feature], pert[feature] = 1.0, 1.0
    with pytest.raises(ValueError, match="No frozen"):
        ss.audit_scopes(clean, pert)


def test_manifest_patient_overlap_fails_before_writing(checkpoint):
    path = checkpoint[1]
    frame = pd.read_csv(path)
    extra = frame.iloc[[0]].copy()
    extra["ID"], extra["split"], extra["dev_fold"] = "heldout_image", "test", np.nan
    pd.concat([frame, extra]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Leakage"):
        ss.run(*checkpoint)
    assert not checkpoint[-1].exists()
