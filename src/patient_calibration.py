"""Step 20: nested, patient-level sigmoid calibration and final protocol freeze.

Base tuning/training and calibration use disjoint patients. All 23 features;
three-member calibrated ensemble per classifier; fixed probability threshold .5.
No held-out feature/image access or evaluation, and no final test results here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import stable_evaluation as se
from . import stable_selection as ss
from . import features as feature_module
from . import coherent_features as coherent_feature_module
from . import coherent_perturbations as coherent_geometry_module
from . import perturbations as original_geometry_module
from . import models as model_module
from . import svm_model as svm_module
from . import robustness_eval as robustness_module
from .coherent_eval import load_coherent, sha256, validate_manifest_isolation
from .coherent_features import COHERENT_CONDITIONS, verify_coherent_geometry
from .features import FEATURE_COLUMNS
from .models import ROOT, inner_patient_folds, load_development
from .robustness_eval import MODEL_NAMES, _predict, metrics, summarize_pair

PROBABILITY_CLIP = 1e-6


def calibration_design():
    return {"step": 20, "version": "patient_sigmoid_ensemble_v1", "features": list(FEATURE_COLUMNS),
            "arm": "all_23_matched_tuning", "primary_model": "svm", "comparator": "logreg",
            "model_choice_basis": "posthoc Step 19 development five-condition BA; not a significance claim",
            "calibration": "three-member split ensemble; each member tuned on base patients and sigmoid-fit on separate clean patients",
            "calibration_folds": 3, "calibration_split_seed": "2020 + scope_code (outer=0..4, final=5)",
            "base_tuning_folds": 3, "base_tuning_seed": "3000 + 10 * scope_code + member_index",
            "base_tuning_objective_and_grids": "unchanged Step 18 all-23 robust objective and grids",
            "patient_score": {"logreg": "logit of mean image malignancy probability, clipped to [1e-6,1-1e-6]",
                              "svm": "mean image decision margin"},
            "calibrator": "training-calibration-patient StandardScaler then LogisticRegression(C=1,max_iter=2000,random_state=42); no class or image weights",
            "calibration_data": "one row per clean calibration patient; no coherent images used for calibration fit",
            "ensemble_prediction": "mean member calibrated patient probabilities, after patient aggregation within each member",
            "threshold": 0.5, "threshold_status": "fixed research threshold; not clinically optimized or validated",
            "method_selection_after_outer_results": "forbidden; no isotonic/threshold/model switching after calibration results",
            "test_access": "no held-out features/images/predictions before Step 21",
            "interpretation": "posthoc development analysis; calibration quality must be measured, not assumed"}


def patient_raw(model, rows, name):
    result = _predict(model, rows, name, 0)
    key = "prob_malignant" if name == "logreg" else "decision_score"
    result = result.rename(columns={key: "raw_score"}).drop(columns="outer_fold")
    result["calibration_input"] = (logit(np.clip(result.raw_score.to_numpy(float), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
                                   if name == "logreg" else result.raw_score.to_numpy(float))
    return result


def fit_sigmoid(patient_scores):
    if (patient_scores.Case.duplicated().any() or set(patient_scores.actual) != {0, 1}
            or not np.isfinite(patient_scores.calibration_input).all()):
        raise ValueError("Invalid one-row-per-patient calibration sample")
    calibrator = Pipeline([("scaler", StandardScaler()),
                           ("sigmoid", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        calibrator.fit(patient_scores[["calibration_input"]].to_numpy(float), patient_scores.actual.to_numpy(int))
    return calibrator


def fit_ensemble(pool, coherent, name, *, excluded_cases, scope_code, protocol):
    """No pool patient can also belong to the outer validation/test scope."""
    pool_cases, excluded_cases = set(pool.Case), set(excluded_cases)
    if pool_cases & excluded_cases:
        raise ValueError("Leakage: ensemble pool intersects excluded patients")
    members, audit, candidates = [], [], []
    for member_index, (base, calibration) in enumerate(inner_patient_folds(pool, seed=2020 + scope_code, n_splits=3)):
        base_cases, cal_cases = set(base.Case), set(calibration.Case)
        if base_cases & cal_cases or base_cases | cal_cases != pool_cases:
            raise ValueError("Leakage or missing patient in base/calibration partition")
        contexts = []
        for inner, (train, val) in enumerate(inner_patient_folds(base, seed=3000 + 10 * scope_code + member_index, n_splits=3)):
            train_cases, val_cases = set(train.Case), set(val.Case)
            if (train_cases & val_cases or (train_cases | val_cases) != base_cases
                    or (train_cases | val_cases) & (cal_cases | excluded_cases)):
                raise ValueError("Leakage in nested base-model tuning")
            scope = {"scope_code": scope_code, "member": member_index, "inner_fold": inner,
                     "training_case_sha256": ss.case_digest(train_cases), "validation_case_sha256": ss.case_digest(val_cases)}
            contexts.append({"train": train,
                             "pack": se.validation_pack(val, coherent.loc[coherent.Case.isin(val_cases)]),
                             "ranking": None, "scope": scope})
        chosen, candidate_rows, details = se.tune_arm(contexts, name, se.ARMS[0], protocol)
        # Model selection has never observed a calibration or excluded patient's features/labels.
        model = se.fit_subset(base, name, FEATURE_COLUMNS, chosen["settings"])
        scores = patient_raw(model, calibration, name)
        calibrator = fit_sigmoid(scores)
        members.append({"base_model": model, "calibrator": calibrator, "settings": chosen["settings"]})
        audit.append({"scope_code": scope_code, "model": name, "member": member_index,
                      "base_cases": sorted(int(c) for c in base_cases), "calibration_cases": sorted(int(c) for c in cal_cases),
                      "excluded_cases": sorted(int(c) for c in excluded_cases), "chosen": chosen,
                      "inner_fits": len(details), "patient_overlap": 0})
        candidates.extend({"scope_code": scope_code, "member": member_index, "model": name, **row} for row in candidate_rows)
    return {"model": name, "members": members}, audit, candidates


def predict_ensemble(ensemble, rows):
    aligned, probabilities, raw = None, [], []
    for member in ensemble["members"]:
        score = patient_raw(member["base_model"], rows, ensemble["model"]).sort_values("Case").reset_index(drop=True)
        if aligned is None:
            aligned = score[["Case", "actual", "n_images"]].copy()
        elif not np.array_equal(aligned, score[["Case", "actual", "n_images"]]):
            raise ValueError("Ensemble patient aggregation differs across members")
        probabilities.append(member["calibrator"].predict_proba(score[["calibration_input"]].to_numpy(float))[:, 1])
        raw.append(score.raw_score.to_numpy(float))
    aligned["prob_malignant"] = np.mean(probabilities, axis=0)
    aligned["raw_score"] = np.mean(raw, axis=0)
    aligned["predicted"] = aligned.prob_malignant.ge(.5).astype(int)
    aligned["raw_predicted"] = aligned.raw_score.ge(.5 if ensemble["model"] == "logreg" else 0.0).astype(int)
    return aligned


def probability_metrics(patients):
    result = metrics(patients, "prob_malignant", "predicted")
    p, y = patients.prob_malignant.to_numpy(float), patients.actual.to_numpy(int)
    if not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Calibrated probability outside [0,1]")
    return {**result, "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1]))}


def calibration_bins(patients, name):
    result = []
    # Fixed-width bins, including the endpoint probability one.
    bins = np.minimum((patients.prob_malignant.to_numpy(float) * 10).astype(int), 9)
    for bin_index in range(10):
        part = patients.loc[bins == bin_index]
        result.append({"model": name, "bin": bin_index, "lower": bin_index / 10, "upper": (bin_index + 1) / 10,
                       "patients": len(part), "mean_probability": float(part.prob_malignant.mean()) if len(part) else None,
                       "malignant_fraction": float(part.actual.mean()) if len(part) else None})
    return result


def calibrated_pairs(ensemble, clean_predictions, changed, fold):
    pert = predict_ensemble(ensemble, changed).set_index("Case")
    out = clean_predictions[["Case", "actual", "n_images", "prob_malignant", "predicted"]].copy()
    if set(out.Case) != set(pert.index):
        raise ValueError("Incomplete calibrated condition patient pairing")
    if not np.array_equal(out.n_images.to_numpy(), pert.loc[out.Case, "n_images"].to_numpy()):
        raise ValueError("Calibrated condition missing patient image views")
    if not np.array_equal(out.actual.to_numpy(), pert.loc[out.Case, "actual"].to_numpy()):
        raise ValueError("Calibrated condition patient diagnoses differ")
    out = out.rename(columns={"prob_malignant": "clean_score", "predicted": "clean_predicted"})
    out["perturbed_score"] = pert.loc[out.Case, "prob_malignant"].to_numpy(float)
    out["perturbed_predicted"] = pert.loc[out.Case, "predicted"].to_numpy(int)
    out["outer_fold"], out["eligible"], out["model"] = fold, True, ensemble["model"]
    out["condition"] = changed.condition.iloc[0]
    out["mean_iou"] = changed.groupby("Case").actual_iou.mean().loc[out.Case].to_numpy(float)
    return out


def nested_calibration(dev, coherent, protocol):
    clean_parts, paired_parts, audit, candidates = [], [], [], []
    for fold in range(5):
        pool, val = dev.loc[dev.dev_fold.ne(fold)], dev.loc[dev.dev_fold.eq(fold)]
        for name in MODEL_NAMES:
            ensemble, checks, search = fit_ensemble(pool, coherent, name, excluded_cases=set(val.Case), scope_code=fold, protocol=protocol)
            clean = predict_ensemble(ensemble, val)
            clean["outer_fold"], clean["model"] = fold, name
            clean_parts.append(clean)
            audit.extend(checks)
            candidates.extend(search)
            for condition in COHERENT_CONDITIONS:
                changed = coherent.loc[coherent.dev_fold.eq(fold) & coherent.condition.eq(condition)]
                paired_parts.append(calibrated_pairs(ensemble, clean, changed, fold))
            print(f"Outer {fold} | {name.upper()}: three patient-disjoint calibration members complete", flush=True)
    clean, pairs = pd.concat(clean_parts, ignore_index=True), pd.concat(paired_parts, ignore_index=True)
    if (len(clean) != dev.Case.nunique() * 2 or clean.duplicated(["model", "Case"]).any()
            or len(pairs) != dev.Case.nunique() * 8 or pairs.duplicated(["model", "condition", "Case"]).any()):
        raise RuntimeError("Incomplete calibrated outer patient coverage")
    summaries = [{"model": name, "patients": len(part), **probability_metrics(part)} for name, part in clean.groupby("model")]
    pair_summary = [{**summarize_pair(part, model=name, condition=condition)}
                    for (name, condition), part in pairs.groupby(["model", "condition"])]
    bins = [row for name, part in clean.groupby("model") for row in calibration_bins(part, name)]
    return clean, pairs, summaries, pair_summary, bins, audit, candidates


def verify_step19(directory, fingerprints, dev):
    summary = json.loads((directory / "evaluation_summary.json").read_text(encoding="utf-8"))
    if (summary.get("status") != "exploratory_nested_development_evaluation"
            or summary.get("held_out_evaluation_used") is not False
            or summary.get("original_clean_oof") != {n: "MATCHED" for n in MODEL_NAMES}
            or summary.get("source_sha256") != sha256(Path(se.__file__))):
        raise ValueError("Step 19 checkpoint/source not verified")
    for key in ("clean_features", "manifest", "coherent_features", "geometry", "protocol", "design_summary"):
        if summary.get("inputs", {}).get(key, {}).get("sha256") != fingerprints[key]["sha256"]:
            raise ValueError(f"Step 19 input hash mismatch: {key}")
    clean = pd.read_csv(directory / "clean_oof_predictions.csv")
    pairs = pd.read_csv(directory / "coherent_paired_predictions.csv")
    if (clean.duplicated(["model", "arm", "Case"]).any()
            or pairs.duplicated(["model", "arm", "condition", "Case"]).any()):
        raise ValueError("Duplicate Step 19 OOF predictions")
    records = summary.get("clean_summary", [])
    if len(records) != 4 or {(r["model"], r["arm"]) for r in records} != set(itertools.product(MODEL_NAMES, se.ARMS)):
        raise ValueError("Step 19 summary requires four arms/models")
    patient_metadata = dev.groupby("Case").agg(outer_fold=("dev_fold", "first"), n_images=("ID", "size"),
                                               Pathology=("Pathology", "first"))
    patient_metadata["actual"] = patient_metadata.Pathology.eq("malignant").astype(int)
    if len(clean) != len(dev.groupby("Case")) * 4 or len(pairs) != len(dev.groupby("Case")) * 16:
        raise ValueError("Step 19 prediction table coverage mismatch")
    for row in records:
        part = clean.loc[clean.model.eq(row["model"]) & clean.arm.eq(row["arm"])]
        if len(part) != summary["development_patients"]:
            raise ValueError("Step 19 cohort coverage mismatch")
        if set(part.Case) != set(patient_metadata.index):
            raise ValueError("Step 19 patient IDs differ from development")
        for key in ("actual", "outer_fold", "n_images"):
            if not np.array_equal(part[key], patient_metadata.loc[part.Case, key]):
                raise ValueError("Step 19 locked patient metadata mismatch")
        for key, value in metrics(part, "score", "predicted").items():
            if not np.isclose(row[key], value, rtol=1e-8, atol=1e-9):
                raise ValueError("Step 19 clean summary mismatch")
        bas = []
        for condition in COHERENT_CONDITIONS:
            p = pairs.loc[pairs.model.eq(row["model"]) & pairs.arm.eq(row["arm"]) & pairs.condition.eq(condition)]
            se.require_pair_metadata(part, p)
            ordered = p.set_index("Case").loc[part.Case]
            if (not np.allclose(part.score.to_numpy(float), ordered.clean_score.to_numpy(float), rtol=1e-6, atol=1e-7)
                    or not np.array_equal(part.predicted, ordered.clean_predicted)):
                raise ValueError("Step 19 clean/paired prediction mismatch")
            if not p.eligible.astype(str).str.lower().eq("true").all():
                raise ValueError("Step 19 coherent patient unavailable")
            bas.append(summarize_pair(p.assign(eligible=True), model=row["model"], condition=condition)["pert_balanced_accuracy"])
        if not np.isclose(row["mean_five_condition_ba"], (row["balanced_accuracy"] + sum(bas)) / 5, atol=1e-9):
            raise ValueError("Step 19 five-condition summary mismatch")
    return summary


def run(clean_path, manifest, coherent_path, geometry, design, step19, output):
    if output.exists():
        raise FileExistsError("Step 20 output exists; choose a fresh --output directory")
    paths = {"clean_features": clean_path, "manifest": manifest, "coherent_features": coherent_path, "geometry": geometry,
             "protocol": design / "selection_protocol.json", "design_summary": design / "selection_design_summary.json",
             **{f"step19_{name}": step19 / filename for name, filename in
                (("summary", "evaluation_summary.json"), ("clean", "clean_oof_predictions.csv"), ("paired", "coherent_paired_predictions.csv"))}}
    hashes = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    protocol = se.load_frozen_design(design, hashes)
    dev = load_development(clean_path, manifest)
    validate_manifest_isolation(manifest, dev)
    prior = verify_step19(step19, hashes, dev)
    coherent = load_coherent(coherent_path, dev)
    verify_coherent_geometry(coherent, pd.read_csv(geometry, dtype={"ID": str}), dev)
    if prior["development_patients"] != dev.Case.nunique() or prior["development_images"] != len(dev):
        raise ValueError("Step 19 development cohort changed")
    frozen_design = calibration_design()  # Fixed before calibration CV results.
    print(f"STEP 20 — PATIENT CALIBRATION AND FINAL PROTOCOL FREEZE\n"
          f"Development: {len(dev)} images | {dev.Case.nunique()} patients\n"
          "Primary: SVM | Comparator: LR | All 23 features | Fixed probability threshold: 0.5", flush=True)
    clean, pairs, scores, pair_scores, bins, audit, candidates = nested_calibration(dev, coherent, protocol)
    final_models = {}
    manifest_ids = pd.read_csv(manifest, usecols=["ID", "Case", "split"])
    test_ids = manifest_ids.loc[manifest_ids.split.eq("test")]
    if test_ids.empty:
        raise ValueError("Locked manifest has no held-out test identifiers")
    for name in MODEL_NAMES:
        ensemble, checks, search = fit_ensemble(dev, coherent, name, excluded_cases=set(test_ids.Case), scope_code=5, protocol=protocol)
        final_models[name] = ensemble
        audit.extend(checks)
        candidates.extend(search)
        print(f"Final {name.upper()}: three development-only calibrated members fitted", flush=True)
    if any(sha256(path) != hashes[key]["sha256"] for key, path in paths.items()):
        raise RuntimeError("Locked inputs changed during Step 20")
    source_hashes = {Path(module.__file__).name: sha256(Path(module.__file__)) for module in
                     (se, ss, feature_module, coherent_feature_module, coherent_geometry_module,
                      original_geometry_module, model_module, svm_module)}
    source_hashes[Path(robustness_module.__file__).name] = sha256(Path(robustness_module.__file__))
    source_hashes[Path(__file__).name] = sha256(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    bundle = {"version": frozen_design["version"], "design": frozen_design, "features": list(FEATURE_COLUMNS),
              "models": final_models, "development_case_sha256": ss.case_digest(set(dev.Case))}
    model_path = output / "frozen_calibrated_models.joblib"
    joblib.dump(bundle, model_path, compress=3)
    final_protocol = {"step": 20, "status": "frozen_before_held_out_evaluation", "design": frozen_design,
                      "primary_model": "svm", "comparator": "logreg", "threshold": .5,
                      "primary_endpoint": "SVM clean patient balanced accuracy at fixed probability threshold 0.5",
                      "secondary_endpoints": ["LR clean metrics and paired LR/SVM clean difference", "four coherent-condition paired robustness metrics",
                                              "clean and perturbed probability Brier/log loss", "five-condition mean BA"],
                      "held_out_conditions": ["clean", *COHERENT_CONDITIONS],
                      "held_out_patients": int(test_ids.Case.nunique()), "held_out_images": len(test_ids),
                      "held_out_case_sha256": ss.case_digest(set(test_ids.Case)),
                      "held_out_image_ids_sha256": hashlib.sha256(json.dumps(sorted(test_ids.ID.astype(str))).encode()).hexdigest(),
                      "test_once_policy": "evaluate both fixed ensembles once; no retraining, model switching, threshold tuning or feature changes after seeing test results",
                      "uncertainty_plan": "10000 paired patient bootstrap replicates within test diagnosis, seed 42; pointwise percentile 95%; fixed fitted models",
                      "missing_coherent_policy": "fail checkpoint if any test image lacks a valid coherent condition; do not silently drop or replace views",
                      "original_perturbations": "preserved development experiment; not regenerated for primary held-out evaluation",
                      "model_artifact_sha256": sha256(model_path), "inputs": hashes, "source_sha256": source_hashes,
                      "limitations": "Posthoc development-based model choice; fixed threshold is research-only, not a clinical decision rule."}
    protocol_path = output / "final_evaluation_protocol.json"
    protocol_path.write_bytes(ss.canonical_json(final_protocol).encode())
    summary = {"step": 20, "status": "calibration_complete_final_protocol_frozen", "held_out_evaluation_used": False,
               "development_images": len(dev), "development_patients": int(dev.Case.nunique()),
               "calibration_design": frozen_design, "clean_cv_metrics": scores,
               "final_protocol_sha256": sha256(protocol_path), "model_artifact_sha256": sha256(model_path),
               "base_inner_fits": sum(a["inner_fits"] for a in audit), "base_refits": len(audit), "sigmoid_fits": len(audit),
               "inputs": hashes, "versions": {pkg: version(pkg) for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "pillow", "scikit-image")},
               "limitations": "Outer CV evaluates this calibrated ensemble, a different estimator from Step 19 single refits; no assumption that calibration improves scores."}
    tables = {"calibrated_clean_oof.csv": clean, "calibrated_coherent_pairs.csv": pairs,
              "calibrated_clean_summary.csv": pd.DataFrame(scores), "calibrated_coherent_summary.csv": pd.DataFrame(pair_scores),
              "clean_reliability_bins.csv": pd.DataFrame(bins)}
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False, float_format="%.12g")
    for filename, value in (("calibration_scope_audit.json", audit), ("base_candidate_search.json", candidates),
                            ("calibration_summary.json", summary), ("calibration_design.json", frozen_design)):
        (output / filename).write_text(ss.canonical_json(value), encoding="utf-8")
    print(pd.DataFrame(scores)[["model", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "brier_score", "log_loss"]].to_string(index=False))
    print(f"Final protocol: FROZEN | Held-out evaluation: NOT USED\nSaved: {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=ROOT / "data/processed/development_features.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--coherent", type=Path, default=ROOT / "data/processed/development_coherent_features.csv")
    parser.add_argument("--geometry", type=Path, default=ROOT / "results/robustness/coherent_geometry_development.csv")
    parser.add_argument("--design", type=Path, default=ROOT / "results/robustness/selection_step18")
    parser.add_argument("--step19", type=Path, default=ROOT / "results/robustness/selection_step19")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness/calibration_step20")
    args = parser.parse_args()
    run(args.clean, args.manifest, args.coherent, args.geometry, args.design, args.step19, args.output)


if __name__ == "__main__":
    main()
