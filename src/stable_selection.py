"""Step 18: freeze and audit training-only stability feature selection.

No classifier fitting, candidate choice or performance evaluation in this step.
Step 19 must recompute rankings within each inner TRAINING partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coherent_eval import load_coherent, sha256, validate_manifest_isolation
from .coherent_features import COHERENT_CONDITIONS, verify_coherent_geometry
from .features import FEATURE_COLUMNS
from .models import ROOT, inner_patient_folds, load_development

TOP_K = (5, 10, 15)
SCALE_EPS = 1e-12


def locked_protocol() -> dict:
    """Design choices fixed before Step 19, not a retrospective preregistration."""
    return {
        "step": 18, "version": "stability_selection_v1",
        "status": "posthoc_experiment3_design_frozen_before_new_performance_evaluation",
        "feature_columns": list(FEATURE_COLUMNS), "conditions": list(COHERENT_CONDITIONS),
        "arms": ["all_23_matched_tuning", "stability_topk"], "top_k_candidates": list(TOP_K),
        "outer_folds": "same five frozen patient folds", "inner_folds": 3,
        "inner_seed": "42 + outer_fold", "global_feature_list_for_CV": "forbidden",
        "ranking_scale": "clean training-only patient-weighted population SD, no diagnosis weights",
        "image_weight": "1 / number of training images for that patient",
        "drift_aggregation": "mean absolute image delta within patient, then equal mean across training patients",
        "stability_score": "maximum of four condition-specific mean drifts divided by clean weighted SD; lower is better",
        "constant_policy": "exclude from subset arms if SD <= 1e-12 * max(1, abs(weighted mean)); retain in all-23 comparator",
        "ranking_tie_break": "canonical FEATURE_COLUMNS order", "scale_epsilon": SCALE_EPS,
        "candidate_shortage": "skip k above eligible feature count; fail subset arm if no k available; never pad or substitute",
        "classifier_training": "clean fitting-partition images only, selected columns, patient-weighted scaling/classifier",
        "logreg": {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "balance": [False, True]},
        "svm": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1], "balance": [False, True]},
        "thresholds": {"logreg_mean_probability": 0.5, "svm_mean_margin": 0.0},
        "inner_selection_objective": "mean over three inner folds of the equal mean of five patient BAs: clean plus four coherent conditions",
        "selection_tie_break": ["higher objective", "higher mean inner clean BA", "higher mean inner clean AUC",
                                "fewer features", "smaller C", "balance=False", "earlier gamma in fixed list"],
        "fair_comparator": "retune all-23 arm with identical inner folds, hyperparameter grids and objective; retain Steps 14/16 baselines separately",
        "outer_reports": ["clean patient metrics", "four coherent paired robustness metrics", "feature selections per outer fold", "fit time"],
        "inner_validation_perturbations": "allowed only to score inner candidates; never for ranking, scaling or classifier fit",
        "outer_validation": "excluded from ranking, scaling, classifier fitting and all parameter/candidate choices",
        "test": "held-out features/images/predictions forbidden until final frozen evaluation",
        "original_protocol": "preserved; not used for ranking or candidate tuning",
        "prohibited": ["dropping a named feature after viewing development results", "selecting k on outer OOF results",
                       "using Step 18 outer rankings inside inner CV", "global ranking across development patients"],
        "interpretation": "exploratory because earlier development results have been inspected; no guarantee of improvement or clinical realism",
    }


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def case_digest(cases) -> str:
    return hashlib.sha256(json.dumps(sorted(int(c) for c in cases)).encode()).hexdigest()


def rank_training_features(clean: pd.DataFrame, coherent: pd.DataFrame, *,
                           training_cases: set, forbidden_cases: set) -> pd.DataFrame:
    """Slice training patients FIRST: excluded feature values cannot affect fit."""
    training_cases, forbidden_cases = set(training_cases), set(forbidden_cases)
    if not training_cases or training_cases & forbidden_cases:
        raise ValueError("Empty training scope or overlap with forbidden patients")
    train = clean.loc[clean.Case.isin(training_cases)].copy()
    pert = coherent.loc[coherent.Case.isin(training_cases)].copy()
    if set(train.Case) != training_cases or set(pert.Case) != training_cases:
        raise ValueError("Missing training patients")
    if (train.ID.duplicated().any() or not train.split.eq("development").all()
            or not pert.split.eq("development").all()
            or pert.duplicated(["ID", "condition"]).any()
            or set(pert.condition) != set(COHERENT_CONDITIONS)
            or len(pert) != len(train) * 4 or not pert.valid.eq(True).all()):
        raise ValueError("Invalid training feature coverage/availability")
    train = train.sort_values("ID").reset_index(drop=True)
    matrix = train[list(FEATURE_COLUMNS)].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite clean training features")
    weights = (1.0 / train.groupby("Case").ID.transform("size")).to_numpy(float)
    mean = np.average(matrix, axis=0, weights=weights)
    sd = np.sqrt(np.average((matrix - mean) ** 2, axis=0, weights=weights))
    eligible = sd > SCALE_EPS * np.maximum(1.0, np.abs(mean))
    result = pd.DataFrame({"feature": FEATURE_COLUMNS, "canonical_order": range(len(FEATURE_COLUMNS)),
                           "clean_weighted_mean": mean, "clean_weighted_sd": sd, "eligible": eligible})
    condition_scores = []
    for condition in COHERENT_CONDITIONS:
        part = pert.loc[pert.condition.eq(condition)]
        if len(part) != len(train) or set(part.ID) != set(train.ID):
            raise ValueError("Missing training image-condition pairing")
        part = part.set_index("ID").loc[train.ID]
        for col in ("Case", "Pathology", "split", "dev_fold"):
            if not np.array_equal(part[col], train[col]):
                raise ValueError(f"Training perturbation metadata mismatch: {col}")
        alternative = part[list(FEATURE_COLUMNS)].to_numpy(float)
        if not np.isfinite(alternative).all():
            raise ValueError("Nonfinite coherent training features")
        delta = pd.DataFrame(np.abs(alternative - matrix), columns=FEATURE_COLUMNS)
        delta["Case"] = train.Case.to_numpy()
        patient_mean = delta.groupby("Case")[list(FEATURE_COLUMNS)].mean().mean().to_numpy(float)
        normalized = np.full(len(FEATURE_COLUMNS), np.nan)
        np.divide(patient_mean, sd, out=normalized, where=eligible)
        result[f"drift_{condition}"] = normalized
        condition_scores.append(normalized)
    result["stability_score"] = np.max(np.array(condition_scores), axis=0)
    ordered = result.loc[result.eligible].sort_values(["stability_score", "canonical_order"], kind="stable")
    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[ordered.index, "rank"] = np.arange(1, len(ordered) + 1)
    result["exclusion_reason"] = np.where(eligible, "", "constant_or_near_constant_clean_training_feature")
    return result.sort_values(["rank", "canonical_order"], na_position="last").reset_index(drop=True)


def selected_columns(ranking: pd.DataFrame, k: int) -> tuple[str, ...]:
    if k not in TOP_K:
        raise ValueError("k outside frozen candidate grid")
    eligible = ranking.loc[ranking.eligible].sort_values("rank")
    if len(eligible) < k:
        raise ValueError("Not enough eligible training features for k")
    selected = set(eligible.head(k).feature)
    # Model column order remains canonical, independent of score ordering.
    return tuple(feature for feature in FEATURE_COLUMNS if feature in selected)


def audit_scopes(dev: pd.DataFrame, coherent: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    ranks, scopes, candidates = [], [], []
    all_cases = set(dev.Case)
    for outer in range(5):
        outer_train = dev.loc[dev.dev_fold.ne(outer)]
        outer_val_cases = set(dev.loc[dev.dev_fold.eq(outer), "Case"])
        contexts = [("outer_refit_only", -1, outer_train, outer_val_cases)]
        for inner, (train, val) in enumerate(inner_patient_folds(outer_train, seed=42 + outer)):
            contexts.append(("inner_training", inner, train, outer_val_cases | set(val.Case)))
        for context, inner, train, forbidden in contexts:
            training_cases = set(train.Case)
            if training_cases | forbidden != all_cases:
                raise ValueError("Training and excluded scopes must partition development patients")
            ranking = rank_training_features(dev, coherent, training_cases=training_cases, forbidden_cases=forbidden)
            allowed = [k for k in TOP_K if int(ranking.eligible.sum()) >= k]
            if not allowed:
                raise ValueError("No frozen top-k candidate has sufficient eligible features")
            ranking.insert(0, "inner_fold", inner)
            ranking.insert(0, "outer_fold", outer)
            ranking.insert(0, "context", context)
            ranks.append(ranking)
            scopes.append({"context": context, "outer_fold": outer, "inner_fold": inner,
                           "training_patients": len(training_cases), "excluded_patients": len(forbidden),
                           "training_case_sha256": case_digest(training_cases), "excluded_case_sha256": case_digest(forbidden),
                           "training_cases": sorted(int(c) for c in training_cases),
                           "excluded_cases": sorted(int(c) for c in forbidden), "overlap": 0})
            for k in TOP_K:
                valid = k in allowed
                candidates.append({"context": context, "outer_fold": outer, "inner_fold": inner, "k": k,
                                   "valid": valid, "features": list(selected_columns(ranking, k)) if valid else [],
                                   "reason": "" if valid else "fewer_eligible_features_than_k"})
        print(f"Outer fold {outer}: outer-training scope and three inner-training scopes audited", flush=True)
    return pd.concat(ranks, ignore_index=True), scopes, candidates


def run(clean_path: Path, manifest: Path, coherent_path: Path, geometry_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Step 18 output exists; choose a fresh --output directory")
    paths = {"clean_features": clean_path, "manifest": manifest, "coherent_features": coherent_path,
             "geometry": geometry_path}
    fingerprints = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    dev = load_development(clean_path, manifest)
    validate_manifest_isolation(manifest, dev)
    coherent = load_coherent(coherent_path, dev)
    verify_coherent_geometry(coherent, pd.read_csv(geometry_path, dtype={"ID": str}), dev)
    protocol = locked_protocol()
    frozen_text = canonical_json(protocol)
    print("STEP 18 — TRAINING-ONLY STABILITY SELECTION DESIGN\n"
          f"Development: {len(dev)} images | {dev.Case.nunique()} patients | {len(FEATURE_COLUMNS)} features", flush=True)
    rankings, scopes, candidates = audit_scopes(dev, coherent)
    if len(rankings) != 20 * len(FEATURE_COLUMNS) or len(scopes) != 20 or len(candidates) != 60:
        raise RuntimeError("Incomplete training scope audit")
    if any(sha256(path) != fingerprints[key]["sha256"] for key, path in paths.items()):
        raise RuntimeError("Locked inputs changed during audit")
    result = {"step": 18, "status": "design_frozen_and_training_scopes_audited",
              "development_images": len(dev), "development_patients": int(dev.Case.nunique()),
              "training_scopes": len(scopes), "ranking_rows": len(rankings), "candidate_rows": len(candidates),
              "classifier_fits": 0, "performance_evaluated": False, "held_out_evaluation_used": False,
              "global_ranking_produced": False, "all_scopes_patient_disjoint": True,
              "protocol_sha256": hashlib.sha256(frozen_text.encode()).hexdigest(),
              "source_sha256": sha256(Path(__file__)), "inputs": fingerprints,
              "next_step": "19: nested training-only reranking, inner candidate selection, outer paired evaluation"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "selection_protocol.json").write_bytes(frozen_text.encode("utf-8"))
    rankings.to_csv(output / "training_scope_rankings.csv", index=False, float_format="%.12g")
    for name, content in (("training_scopes.json", scopes), ("candidate_subsets.json", candidates), ("selection_design_summary.json", result)):
        (output / name).write_text(canonical_json(content), encoding="utf-8")
    print(f"Frozen top-k candidates: {TOP_K}; matched all-23 comparator\n"
          f"Patient-disjoint scopes: {len(scopes)} | Classifier fits: 0 | Held-out evaluation: NOT USED\nSaved: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=ROOT / "data/processed/development_features.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--coherent", type=Path, default=ROOT / "data/processed/development_coherent_features.csv")
    parser.add_argument("--geometry", type=Path, default=ROOT / "results/robustness/coherent_geometry_development.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness/selection_step18")
    args = parser.parse_args()
    run(args.clean, args.manifest, args.coherent, args.geometry, args.output)


if __name__ == "__main__":
    main()
