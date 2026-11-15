"""Step 19: nested patient CV for the frozen Step 18 stability-selection design.

Refits clean models only. Coherent INNER-validation predictions score candidates;
OUTER-validation patients never affect ranking, scaling, fitting or selection.
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from . import stable_selection as ss
from .coherent_eval import load_coherent, sha256, validate_manifest_isolation
from .coherent_features import COHERENT_CONDITIONS, verify_coherent_geometry
from .features import FEATURE_COLUMNS
from .models import ROOT, inner_patient_folds, load_development, make_model, patient_image_weights
from .robustness_eval import (MODEL_NAMES, _predict, compare_one_condition, load_fold_settings,
                             metrics, summarize_pair, validate_clean_reference)
from .svm_model import build_svm

ARMS = ("all_23_matched_tuning", "stability_topk")


@dataclass
class SubsetModel:
    pipeline: object
    columns: tuple

    def matrix(self, full_matrix):
        matrix = np.asarray(full_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError("Subset adapter expects the full canonical 23-column input")
        return matrix[:, [FEATURE_COLUMNS.index(c) for c in self.columns]]

    def predict_proba(self, full_matrix):
        return self.pipeline.predict_proba(self.matrix(full_matrix))

    def decision_function(self, full_matrix):
        return self.pipeline.decision_function(self.matrix(full_matrix))


def fit_subset(train, name, columns, settings):
    columns = tuple(columns)
    if (name not in MODEL_NAMES or not columns or len(set(columns)) != len(columns)
            or columns != tuple(c for c in FEATURE_COLUMNS if c in columns)):
        raise ValueError("Invalid classifier or noncanonical subset columns")
    if (not train.split.eq("development").all() or train.ID.duplicated().any()
            or set(train.Pathology) != {"benign", "malignant"}):
        raise ValueError("Invalid clean fitting partition")
    matrix = train[list(columns)].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite clean training matrix")
    weights = patient_image_weights(train, balance=settings["balance"])
    pipeline = (make_model(settings["C"]) if name == "logreg"
                else build_svm(settings["C"], settings["gamma"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(matrix, train.Pathology.eq("malignant").to_numpy(int),
                     scaler__sample_weight=weights, classifier__sample_weight=weights)
    return SubsetModel(pipeline, columns)


def validation_pack(clean, coherent):
    """Stack five conditions for one model call while preserving patient means."""
    frames = [clean]
    for condition in COHERENT_CONDITIONS:
        part = coherent.loc[coherent.condition.eq(condition)]
        if (len(part) != len(clean) or set(part.ID) != set(clean.ID)
                or not part.valid.all() or part.ID.duplicated().any()):
            raise ValueError("Incomplete inner-validation pairing")
        part = part.set_index("ID").loc[clean.ID].reset_index()
        for col in ("Case", "Pathology", "dev_fold"):
            if not np.array_equal(part[col], clean[col]):
                raise ValueError("Inner-validation metadata mismatch")
        frames.append(part)
    return {
        "X": np.concatenate([f[list(FEATURE_COLUMNS)].to_numpy(float) for f in frames]),
        "Case": np.tile(clean.Case.to_numpy(), 5),
        "actual": np.tile(clean.Pathology.eq("malignant").to_numpy(int), 5),
        "condition_index": np.repeat(np.arange(5), len(clean)),
    }


def evaluate_inner(model, name, pack):
    scores = (model.predict_proba(pack["X"])[:, 1] if name == "logreg"
              else model.decision_function(pack["X"]))
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite inner-validation score")
    frame = pd.DataFrame({k: pack[k] for k in ("Case", "actual", "condition_index")})
    frame["score"] = scores
    if frame.groupby("Case").actual.nunique().ne(1).any():
        raise ValueError("Inconsistent validation patient diagnoses")
    patient = frame.groupby(["condition_index", "Case"], as_index=False).agg(
        actual=("actual", "first"), score=("score", "mean"))
    patient["predicted"] = patient.score.ge(0.5 if name == "logreg" else 0.0).astype(int)
    bas = []
    for condition in range(5):
        p = patient.loc[patient.condition_index.eq(condition)]
        y, pred = p.actual.to_numpy(int), p.predicted.to_numpy(int)
        if set(y) != {0, 1}:
            raise ValueError("Inner validation needs both patient diagnoses")
        bas.append(float(0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())))
    clean = patient.loc[patient.condition_index.eq(0)]
    return {"objective": float(np.mean(bas)), "clean_ba": bas[0],
            "clean_auc": float(roc_auc_score(clean.actual, clean.score)),
            **{f"ba_{condition}": bas[i + 1] for i, condition in enumerate(COHERENT_CONDITIONS)}}


def build_contexts(dev, coherent, outer):
    outer_train = dev.loc[dev.dev_fold.ne(outer)]
    outer_val = set(dev.loc[dev.dev_fold.eq(outer), "Case"])
    contexts, audit, rankings = [], [], []
    for inner, (train, val) in enumerate(inner_patient_folds(outer_train, seed=42 + outer)):
        forbidden = outer_val | set(val.Case)
        cases = set(train.Case)
        if cases & forbidden or cases | forbidden != set(dev.Case):
            raise ValueError("Leakage or missing patient in inner scope")
        ranking = ss.rank_training_features(dev, coherent, training_cases=cases, forbidden_cases=forbidden)
        scope = {"outer_fold": outer, "inner_fold": inner, "context": "inner_training",
                 "training_case_sha256": ss.case_digest(cases), "excluded_case_sha256": ss.case_digest(forbidden),
                 "training_patients": len(cases), "excluded_patients": len(forbidden)}
        contexts.append({"train": train, "pack": validation_pack(val, coherent.loc[coherent.Case.isin(set(val.Case))]),
                         "ranking": ranking, "scope": scope})
        audit.append(scope)
        rankings.append(ranking.assign(outer_fold=outer, inner_fold=inner, context="inner_training"))
    return contexts, audit, rankings


def setting_grid(name, protocol):
    grid = protocol[name]
    for C, balance in itertools.product(grid["C"], grid["balance"]):
        for gamma_index, gamma in enumerate(grid["gamma"] if name == "svm" else [None]):
            setting = {"C": C, "balance": balance}
            if name == "svm":
                setting["gamma"] = gamma
            yield setting, gamma_index


def candidate_key(row):
    return (row["mean_objective"], row["mean_clean_ba"], row["mean_clean_auc"],
            -row["k"], -row["settings"]["C"], -int(row["settings"]["balance"]), -row["gamma_index"])


def tune_arm(contexts, name, arm, protocol):
    ks = [23] if arm == ARMS[0] else protocol["top_k_candidates"]
    candidates, details = [], []
    for k in ks:
        if arm == ARMS[1] and any(int(ctx["ranking"].eligible.sum()) < k for ctx in contexts):
            continue
        columns = [FEATURE_COLUMNS if arm == ARMS[0] else ss.selected_columns(ctx["ranking"], k) for ctx in contexts]
        for settings, gamma_index in setting_grid(name, protocol):
            scores = []
            for inner, ctx in enumerate(contexts):
                start = perf_counter()
                model = fit_subset(ctx["train"], name, columns[inner], settings)
                fit_seconds = perf_counter() - start
                score = evaluate_inner(model, name, ctx["pack"])
                scores.append(score)
                details.append({**ctx["scope"], "model": name, "arm": arm, "k": k,
                                "settings": json.dumps(settings, sort_keys=True), "features": json.dumps(columns[inner]),
                                **score, "fit_seconds": fit_seconds})
            candidates.append({"model": name, "arm": arm, "k": k, "settings": settings,
                               "gamma_index": gamma_index,
                               **{f"mean_{key}": float(np.mean([s[key] for s in scores]))
                                  for key in ("objective", "clean_ba", "clean_auc")}})
    if not candidates:
        raise ValueError("No valid frozen candidate in all three inner-training folds")
    return max(candidates, key=candidate_key), candidates, details


def require_pair_metadata(left, right):
    a, b = left.sort_values("Case"), right.sort_values("Case")
    if a.Case.duplicated().any() or b.Case.duplicated().any():
        raise ValueError("Duplicate patient in arm comparison")
    if not np.array_equal(a[["Case", "actual", "outer_fold", "n_images"]], b[["Case", "actual", "outer_fold", "n_images"]]):
        raise ValueError("Patient metadata differs across comparison arms")


def arm_comparisons(predictions):
    rows = []
    for name in MODEL_NAMES:
        for condition in COHERENT_CONDITIONS:
            groups = [predictions.loc[predictions.model.eq(name) & predictions.arm.eq(arm)
                                     & predictions.condition.eq(condition)] for arm in ARMS]
            require_pair_metadata(*groups)
            a, b = [summarize_pair(p, model=name, condition=condition) for p in groups]
            rows.append({"model": name, "condition": condition, "patients": b["eligible_patients"],
                         **{f"{key}_subset_minus_all23": b[key] - a[key] for key in
                            ("clean_balanced_accuracy", "pert_balanced_accuracy", "delta_balanced_accuracy",
                             "pert_sensitivity", "pert_specificity", "flip_rate")},
                         "new_false_negatives_all23": a["new_false_negatives"],
                         "new_false_negatives_subset": b["new_false_negatives"]})
    return pd.DataFrame(rows)


def nested_evaluate(dev, coherent, protocol):
    clean_parts, paired_parts, selections, scopes, rankings = [], [], [], [], []
    candidate_rows, inner_rows, fold_clean, fold_pairs = [], [], [], []
    begin = perf_counter()
    for outer in range(5):
        contexts, audit, ranks = build_contexts(dev, coherent, outer)
        scopes.extend(audit)
        rankings.extend(ranks)
        train, val = dev.loc[dev.dev_fold.ne(outer)], dev.loc[dev.dev_fold.eq(outer)]
        ranking = ss.rank_training_features(dev, coherent, training_cases=set(train.Case), forbidden_cases=set(val.Case))
        rankings.append(ranking.assign(outer_fold=outer, inner_fold=-1, context="outer_refit_only"))
        scopes.append({"outer_fold": outer, "inner_fold": -1, "context": "outer_refit_only",
                       "training_case_sha256": ss.case_digest(set(train.Case)), "excluded_case_sha256": ss.case_digest(set(val.Case)),
                       "training_patients": train.Case.nunique(), "excluded_patients": val.Case.nunique()})
        for name in MODEL_NAMES:
            for arm in ARMS:
                start = perf_counter()
                chosen, candidates, details = tune_arm(contexts, name, arm, protocol)
                tuning_seconds = perf_counter() - start
                columns = FEATURE_COLUMNS if arm == ARMS[0] else ss.selected_columns(ranking, chosen["k"])
                start = perf_counter()
                model = fit_subset(train, name, columns, chosen["settings"])
                refit_seconds = perf_counter() - start
                selection = {"outer_fold": outer, **chosen, "features": list(columns),
                             "training_case_sha256": ss.case_digest(set(train.Case)),
                             "tuning_seconds": tuning_seconds, "outer_refit_seconds": refit_seconds}
                selections.append(selection)
                for record in candidates:
                    candidate_rows.append({"outer_fold": outer, **record, "selected": record == chosen})
                inner_rows.extend(details)
                baseline = _predict(model, val, name, outer).rename(columns={
                    "prob_malignant" if name == "logreg" else "decision_score": "score"})
                baseline["model"], baseline["arm"] = name, arm
                clean_parts.append(baseline)
                fold_clean.append({"outer_fold": outer, "model": name, "arm": arm,
                                   **metrics(baseline, "score", "predicted")})
                for condition in COHERENT_CONDITIONS:
                    variants = coherent.loc[coherent.dev_fold.eq(outer) & coherent.condition.eq(condition)]
                    paired = compare_one_condition(model, name, val, variants, outer)
                    paired["arm"] = arm
                    paired_parts.append(paired)
                    fold_pairs.append({"outer_fold": outer, "arm": arm,
                                       **summarize_pair(paired, model=name, condition=condition)})
                print(f"Outer {outer} | {name} | {arm}: k={chosen['k']} | "
                      f"inner objective={chosen['mean_objective']:.4f} | clean training only", flush=True)
    clean, pairs = pd.concat(clean_parts, ignore_index=True), pd.concat(paired_parts, ignore_index=True)
    n = dev.Case.nunique()
    if (len(clean) != n * 4 or len(pairs) != n * 16
            or clean.duplicated(["model", "arm", "Case"]).any()
            or pairs.duplicated(["model", "arm", "condition", "Case"]).any() or not pairs.eligible.all()):
        raise RuntimeError("Incomplete/duplicate OOF patient predictions")
    pooled_pairs = pd.DataFrame([{"arm": arm, **summarize_pair(part, model=name, condition=condition)}
                                for (name, arm, condition), part in pairs.groupby(["model", "arm", "condition"])])
    pooled_clean = []
    for (name, arm), part in clean.groupby(["model", "arm"]):
        clean_metrics = metrics(part, "score", "predicted")
        changed = pooled_pairs.loc[pooled_pairs.model.eq(name) & pooled_pairs.arm.eq(arm)]
        pooled_clean.append({"model": name, "arm": arm, "patients": len(part), **clean_metrics,
                             "mean_five_condition_ba": float((clean_metrics["balanced_accuracy"] + changed.pert_balanced_accuracy.sum()) / 5)})
    frequency = [{"model": name, "arm": arm, "feature": feature,
                  "selected_outer_folds": sum(s["model"] == name and s["arm"] == arm and feature in s["features"] for s in selections)}
                 for name in MODEL_NAMES for arm in ARMS for feature in FEATURE_COLUMNS]
    flat_candidates = [{**r, "settings": json.dumps(r["settings"], sort_keys=True)} for r in candidate_rows]
    return {"tables": {"clean_oof_predictions.csv": clean, "coherent_paired_predictions.csv": pairs,
                       "clean_summary.csv": pd.DataFrame(pooled_clean), "coherent_summary.csv": pooled_pairs,
                       "clean_outer_fold_metrics.csv": pd.DataFrame(fold_clean),
                       "coherent_outer_fold_metrics.csv": pd.DataFrame(fold_pairs),
                       "arm_comparison.csv": arm_comparisons(pairs),
                       "inner_candidate_summary.csv": pd.DataFrame(flat_candidates),
                       "inner_fold_scores.csv": pd.DataFrame(inner_rows),
                       "recomputed_training_rankings.csv": pd.concat(rankings, ignore_index=True),
                       "feature_selection_frequency.csv": pd.DataFrame(frequency)},
            "selections": selections, "scopes": scopes,
            "fit_counts": {"inner": len(inner_rows), "outer_refits": len(selections)},
            "elapsed_seconds": perf_counter() - begin}


def load_frozen_design(directory, fingerprints):
    protocol_path, summary_path = directory / "selection_protocol.json", directory / "selection_design_summary.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("status") != "design_frozen_and_training_scopes_audited"
            or summary.get("protocol_sha256") != sha256(protocol_path)
            or summary.get("source_sha256") != sha256(Path(ss.__file__))
            or summary.get("held_out_evaluation_used") is not False
            or summary.get("all_scopes_patient_disjoint") is not True or summary.get("training_scopes") != 20
            or protocol != ss.locked_protocol()):
        raise ValueError("Step 18 frozen protocol/source verification failed")
    for key in ("clean_features", "manifest", "coherent_features", "geometry"):
        if summary.get("inputs", {}).get(key, {}).get("sha256") != fingerprints[key]["sha256"]:
            raise ValueError(f"Step 18 input hash mismatch: {key}")
    return protocol


def check_original_oof(dev, modeling):
    for name in MODEL_NAMES:
        settings = load_fold_settings(modeling / f"{name}_nested_cv_folds.csv", name)
        pieces = []
        for fold in range(5):
            train, val = dev.loc[dev.dev_fold.ne(fold)], dev.loc[dev.dev_fold.eq(fold)]
            model = fit_subset(train, name, FEATURE_COLUMNS, settings[fold])
            pieces.append(_predict(model, val, name, fold))
        validate_clean_reference(pd.concat(pieces, ignore_index=True), modeling / f"{name}_nested_cv_predictions.csv", name)
        print(f"Original {name.upper()} clean OOF: MATCHED", flush=True)


def run(clean_path, manifest, coherent_path, geometry, design, modeling, output):
    if output.exists():
        raise FileExistsError("Step 19 output exists; choose a fresh --output directory")
    paths = {"clean_features": clean_path, "manifest": manifest, "coherent_features": coherent_path, "geometry": geometry,
             "protocol": design / "selection_protocol.json", "design_summary": design / "selection_design_summary.json",
             **{f"{name}_{kind}": modeling / f"{name}_nested_cv_{kind}.csv"
                for name in MODEL_NAMES for kind in ("folds", "predictions")}}
    fingerprints = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    protocol = load_frozen_design(design, fingerprints)
    dev = load_development(clean_path, manifest)
    validate_manifest_isolation(manifest, dev)
    coherent = load_coherent(coherent_path, dev)
    verify_coherent_geometry(coherent, pd.read_csv(geometry, dtype={"ID": str}), dev)
    print(f"STEP 19 — NESTED STABILITY-SELECTION EVALUATION\nDevelopment: {len(dev)} images | {dev.Case.nunique()} patients\n"
          "Frozen Step 18 design: VERIFIED | Held-out evaluation: NOT USED", flush=True)
    check_original_oof(dev, modeling)
    result = nested_evaluate(dev, coherent, protocol)
    if any(sha256(path) != fingerprints[key]["sha256"] for key, path in paths.items()):
        raise RuntimeError("Locked inputs changed during Step 19")
    summary = {"step": 19, "status": "exploratory_nested_development_evaluation",
               "development_images": len(dev), "development_patients": int(dev.Case.nunique()),
               "arms": list(ARMS), "model_names": list(MODEL_NAMES), "conditions": list(COHERENT_CONDITIONS),
               "protocol_sha256": fingerprints["protocol"]["sha256"], "inputs": fingerprints,
               "original_clean_oof": {name: "MATCHED" for name in MODEL_NAMES},
               "held_out_evaluation_used": False, "training": "clean partition only; inner training reranking; inner validation candidate scoring",
               "fit_counts": {**result["fit_counts"], "baseline_reproducibility": 10},
               "elapsed_nested_seconds": result["elapsed_seconds"],
               "clean_summary": result["tables"]["clean_summary.csv"].to_dict("records"),
               "source_sha256": sha256(Path(__file__)),
               "versions": {pkg: version(pkg) for pkg in ("numpy", "pandas", "scikit-learn", "scipy")},
               "limitations": "Post-hoc exploratory development analysis; no significance claim, clinical validation or final global model/feature selection."}
    output.mkdir(parents=True, exist_ok=False)
    for filename, table in result["tables"].items():
        table.to_csv(output / filename, index=False, float_format="%.12g")
    for filename, content in (("outer_selections.json", result["selections"]), ("training_scopes.json", result["scopes"]),
                              ("evaluation_summary.json", summary)):
        (output / filename).write_text(ss.canonical_json(content), encoding="utf-8")
    print(result["tables"]["clean_summary.csv"][["model", "arm", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "mean_five_condition_ba"]].to_string(index=False))
    print(f"Original clean OOF: MATCHED | Held-out evaluation: NOT USED\nSaved: {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=ROOT / "data/processed/development_features.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--coherent", type=Path, default=ROOT / "data/processed/development_coherent_features.csv")
    parser.add_argument("--geometry", type=Path, default=ROOT / "results/robustness/coherent_geometry_development.csv")
    parser.add_argument("--design", type=Path, default=ROOT / "results/robustness/selection_step18")
    parser.add_argument("--modeling", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness/selection_step19")
    args = parser.parse_args()
    run(args.clean, args.manifest, args.coherent, args.geometry, args.design, args.modeling, args.output)


if __name__ == "__main__":
    main()
