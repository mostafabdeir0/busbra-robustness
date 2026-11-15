"""Step 16: separately labelled, post-hoc coherent-boundary sensitivity analysis.

No search, calibration, threshold selection, raw images or test feature access.
All preprocessing/model fits use clean outer-training data via Step 14 helpers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd

from .coherent_features import COHERENT_CONDITIONS, verify_coherent_geometry
from .features import FEATURE_COLUMNS
from .models import ROOT, NUM_OUTER_FOLDS, fit_model, load_development
from .perturbed_features import INFO_COLUMNS
from .robustness_eval import (
    MODEL_NAMES, ID_COLUMNS, _predict, compare_one_condition, load_fold_settings,
    parse_boolean_column, summarize_pair, validate_clean_reference,
)
from .svm_model import fit_svm

PROTOCOL = "step16_posthoc_spatially_coherent"
ORIGINAL_PROTOCOL = "step14_original_boundary"
PREDICTION_FILE = "coherent_paired_patient_predictions.csv"


def load_coherent(path: Path, clean: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str}, low_memory=False)
    if list(frame.columns) != list(INFO_COLUMNS) + list(FEATURE_COLUMNS):
        raise ValueError("Coherent table schema/order differs from Step 15")
    keys = list(ID_COLUMNS) + ["condition", "valid"]
    if frame[keys].isna().any().any() or frame.duplicated(["ID", "condition"]).any():
        raise ValueError("Missing metadata or duplicate coherent ID-condition")
    if (len(frame) != len(clean) * len(COHERENT_CONDITIONS)
            or set(frame.condition) != set(COHERENT_CONDITIONS)
            or set(frame.ID) != set(clean.ID)
            or not frame.groupby("ID").condition.nunique().eq(4).all()):
        raise ValueError("Coherent table must cover every development image and four conditions")
    metadata = clean.set_index("ID")
    for col in ID_COLUMNS[1:]:
        if not np.array_equal(frame[col], metadata.loc[frame.ID, col]):
            raise ValueError(f"Coherent metadata differs from locked clean manifest: {col}")
    frame["valid"] = parse_boolean_column(frame.valid)
    # Step 15 completed all four conditions; missing vectors are a failed checkpoint.
    if not frame.valid.all():
        raise ValueError("Step 15 coherent checkpoint contains unavailable variants")
    if not np.isfinite(frame[list(FEATURE_COLUMNS)].to_numpy(float)).all():
        raise ValueError("Nonfinite coherent features")
    if not frame.actual_iou.between(0, 1, inclusive="neither").all():
        raise ValueError("Invalid coherent IoU")
    return frame


def validate_manifest_isolation(manifest_path: Path, dev: pd.DataFrame) -> None:
    # Inspect split identifiers only; no held-out labels/images/features are loaded.
    manifest = pd.read_csv(manifest_path, usecols=["ID", "Case", "split", "dev_fold"])
    if manifest[["ID", "Case", "split"]].isna().any().any() or manifest.ID.duplicated().any():
        raise ValueError("Invalid locked manifest identifiers")
    if set(dev.Case) & set(manifest.loc[manifest.split.ne("development"), "Case"]):
        raise ValueError("Leakage: patient shared by development and held-out split")
    if not dev.dev_fold.isin(range(NUM_OUTER_FOLDS)).all():
        raise ValueError("Outer folds must be integers 0..4")


def load_original(path: Path, dev: pd.DataFrame) -> pd.DataFrame:
    """Validate Step 14's patient reference before any original/coherent comparison."""
    frame = pd.read_csv(path)
    required = {"model", "condition", "Case", "actual", "outer_fold", "n_images",
                "eligible", "mean_iou", "clean_score", "perturbed_score",
                "clean_predicted", "perturbed_predicted"}
    if not required.issubset(frame):
        raise ValueError("Step 14 patient reference schema missing columns")
    frame = frame.loc[frame.condition.isin(COHERENT_CONDITIONS)].copy()
    if set(frame.model) != set(MODEL_NAMES) or frame.duplicated(["model", "condition", "Case"]).any():
        raise ValueError("Invalid Step 14 model/patient coverage")
    frame["eligible"] = parse_boolean_column(frame.eligible)
    patients = dev.groupby("Case").agg(Pathology=("Pathology", "first"),
                                      outer_fold=("dev_fold", "first"), n_images=("ID", "size"))
    for model in MODEL_NAMES:
        for condition in COHERENT_CONDITIONS:
            part = frame.loc[frame.model.eq(model) & frame.condition.eq(condition)]
            if len(part) != len(patients) or set(part.Case) != set(patients.index):
                raise ValueError("Step 14 patient coverage differs from development")
            ref = patients.loc[part.Case]
            for col in ("outer_fold", "n_images"):
                if not np.array_equal(part[col], ref[col]):
                    raise ValueError(f"Step 14 metadata mismatch: {col}")
            if not np.array_equal(part.actual, ref.Pathology.eq("malignant").astype(int)):
                raise ValueError("Step 14 patient label mismatch")
            if not np.isfinite(part.clean_score).all() or not part.clean_predicted.isin([0, 1]).all():
                raise ValueError("Invalid Step 14 clean predictions")
            eligible = part.loc[part.eligible]
            if (not np.isfinite(eligible[["perturbed_score", "mean_iou"]]).all().all()
                    or not eligible.perturbed_predicted.isin([0, 1]).all()
                    or not eligible.mean_iou.between(0, 1).all()):
                raise ValueError("Invalid Step 14 eligible predictions")
    return frame


def require_same_clean(left: pd.DataFrame, right: pd.DataFrame) -> None:
    a = left.sort_values("Case").reset_index(drop=True)
    b = right.sort_values("Case").reset_index(drop=True)
    for col in ("Case", "actual", "outer_fold", "n_images", "clean_predicted"):
        if not np.array_equal(a[col], b[col]):
            raise ValueError(f"Step 14 paired clean mismatch: {col}")
    if not np.allclose(a.clean_score, b.clean_score, rtol=1e-6, atol=1e-7):
        raise ValueError("Step 14 paired clean score mismatch")


def protocol_comparison(original: pd.DataFrame, coherent: pd.DataFrame) -> pd.DataFrame:
    """Both methods use the intersection of eligible patients, within each model/condition."""
    rows = []
    for model in MODEL_NAMES:
        for condition in COHERENT_CONDITIONS:
            old = original.loc[original.model.eq(model) & original.condition.eq(condition)]
            new = coherent.loc[coherent.model.eq(model) & coherent.condition.eq(condition)]
            require_same_clean(old, new)
            common = set(old.loc[old.eligible, "Case"]) & set(new.loc[new.eligible, "Case"])
            for label, part in ((ORIGINAL_PROTOCOL, old), (PROTOCOL, new)):
                row = summarize_pair(part.loc[part.Case.isin(common)], model=model, condition=condition)
                row.update(protocol=label, common_eligible_patients=len(common),
                           patients_outside_common=len(old) - len(common))
                rows.append(row)
    return pd.DataFrame(rows)


def model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in COHERENT_CONDITIONS:
        lr = predictions.loc[predictions.model.eq("logreg") & predictions.condition.eq(condition)]
        svm = predictions.loc[predictions.model.eq("svm") & predictions.condition.eq(condition)]
        common = set(lr.loc[lr.eligible, "Case"]) & set(svm.loc[svm.eligible, "Case"])
        a = summarize_pair(lr.loc[lr.Case.isin(common)], model="logreg", condition=condition)
        b = summarize_pair(svm.loc[svm.Case.isin(common)], model="svm", condition=condition)
        rows.append({"protocol": PROTOCOL, "condition": condition,
                     "common_eligible_patients": len(common),
                     "logreg_delta_balanced_accuracy": a["delta_balanced_accuracy"],
                     "svm_delta_balanced_accuracy": b["delta_balanced_accuracy"],
                     "svm_minus_logreg_delta_balanced_accuracy": b["delta_balanced_accuracy"] - a["delta_balanced_accuracy"],
                     "logreg_flip_rate": a["flip_rate"], "svm_flip_rate": b["flip_rate"],
                     "logreg_new_false_negatives": a["new_false_negatives"],
                     "svm_new_false_negatives": b["new_false_negatives"]})
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(clean_path: Path, manifest_path: Path, coherent_path: Path,
        geometry_path: Path, modeling_dir: Path, original_dir: Path, output_dir: Path) -> dict:
    paths = {"clean": clean_path, "manifest": manifest_path, "coherent_features": coherent_path,
             "coherent_geometry": geometry_path,
             "step14_predictions": original_dir / "robustness_paired_patient_predictions.csv"}
    for name in MODEL_NAMES:
        for kind in ("folds", "predictions"):
            paths[f"{name}_{kind}"] = modeling_dir / f"{name}_nested_cv_{kind}.csv"
    filenames = [PREDICTION_FILE, "coherent_condition_summary.csv", "coherent_outer_fold_metrics.csv",
                 "step14_vs_step16_common_patient_summary.csv", "coherent_model_comparison.csv",
                 "coherent_summary.json"]
    if output_dir.resolve() in (original_dir.resolve(), modeling_dir.resolve()):
        raise ValueError("Step 16 requires a separate output directory")
    targets = [output_dir / name for name in filenames]
    if any(p.exists() for p in targets):
        raise FileExistsError("Step 16 output exists; choose a new --output directory")
    if {p.resolve() for p in paths.values()} & {p.resolve() for p in targets}:
        raise ValueError("Output would overwrite a locked input")
    fingerprints = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    dev = load_development(clean_path, manifest_path)
    validate_manifest_isolation(manifest_path, dev)
    coherent = load_coherent(coherent_path, dev)
    verify_coherent_geometry(coherent, pd.read_csv(geometry_path, dtype={"ID": str}), dev)
    original = load_original(paths["step14_predictions"], dev)
    settings = {name: load_fold_settings(paths[f"{name}_folds"], name) for name in MODEL_NAMES}
    print(f"STEP 16 — POST-HOC COHERENT BOUNDARY SENSITIVITY\n"
          f"Development: {len(dev)} images | {dev.Case.nunique()} patients | {len(coherent)} vectors", flush=True)
    predictions, folds = [], []
    for name in MODEL_NAMES:
        baselines, model_parts = [], []
        for fold in range(NUM_OUTER_FOLDS):
            train, val = dev.loc[dev.dev_fold.ne(fold)], dev.loc[dev.dev_fold.eq(fold)]
            if set(train.Case) & set(val.Case):
                raise ValueError("Leakage: outer training/validation patients overlap")
            model = (fit_model(train, **settings[name][fold]) if name == "logreg"
                     else fit_svm(train, **settings[name][fold]))
            baselines.append(_predict(model, val, name, fold))
            for condition in COHERENT_CONDITIONS:
                variant = coherent.loc[coherent.dev_fold.eq(fold) & coherent.condition.eq(condition)]
                rec = compare_one_condition(model, name, val, variant, fold)
                model_parts.append(rec)
                folds.append(summarize_pair(rec, model=name, condition=condition, fold=fold))
            print(f"{name.upper()} locked outer fold {fold}: clean training only", flush=True)
        validate_clean_reference(pd.concat(baselines, ignore_index=True), paths[f"{name}_predictions"], name)
        predictions.extend(model_parts)
        print(f"{name.upper()} saved clean OOF: MATCHED", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    if (len(predictions) != dev.Case.nunique() * 4 * 2
            or predictions.duplicated(["model", "condition", "Case"]).any()):
        raise ValueError("Missing/duplicate coherent patient predictions")
    comparison = protocol_comparison(original, predictions)
    summary = pd.DataFrame([summarize_pair(part, model=name, condition=condition)
                            for (name, condition), part in predictions.groupby(["model", "condition"])])
    predictions.insert(0, "protocol", PROTOCOL)
    summary.insert(0, "protocol", PROTOCOL)
    fold_table = pd.DataFrame(folds)
    fold_table.insert(0, "protocol", PROTOCOL)
    models = model_comparison(predictions)
    # Refuse publication if inputs changed during fitting.
    if any(sha256(path) != fingerprints[key]["sha256"] for key, path in paths.items()):
        raise RuntimeError("Locked inputs changed during evaluation")
    result = {"protocol": PROTOCOL, "analysis_status": "descriptive post-hoc sensitivity analysis",
              "development_images": len(dev), "development_patients": int(dev.Case.nunique()),
              "coherent_feature_vectors": len(coherent), "patient_rows_saved": len(predictions),
              "conditions": list(COHERENT_CONDITIONS), "model_names": list(MODEL_NAMES),
              "clean_reference_oof": {name: "MATCHED" for name in MODEL_NAMES},
              "step14_clean_predictions": "MATCHED", "coherent_geometry": "MATCHED",
              "held_out_features_images_predictions_used": False,
              "held_out_manifest_identifiers_checked_for_isolation": True,
              "training": "clean outer-training only; recorded inner settings; no retuning",
              "thresholds": {"logreg_mean_probability": 0.5, "svm_mean_decision_margin": 0.0},
              "protocol_comparison_population": "condition-specific intersection of eligible patients",
              "original_step14_outputs": "read-only; separately labelled step14_original_boundary",
              "translations": "unchanged Step 14 definitions; not reevaluated here",
              "inputs": fingerprints, "locked_settings": settings,
              "environment": {"python": platform.python_version(), **{pkg: version(pkg) for pkg in
                              ("numpy", "pandas", "scikit-learn", "scipy")}},
              "results": summary.to_dict("records")}
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame, filename in zip((predictions, summary, fold_table, comparison, models), filenames[:-1]):
        frame.to_csv(output_dir / filename, index=False, float_format="%.12g")
    (output_dir / filenames[-1]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(summary[["model", "condition", "eligible_patients", "delta_balanced_accuracy", "flip_rate", "new_false_negatives"]].to_string(index=False))
    print(f"Step 14 clean predictions: MATCHED | Held-out evaluation: NOT USED\nSaved: {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=ROOT / "data/processed/development_features.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--coherent", type=Path, default=ROOT / "data/processed/development_coherent_features.csv")
    parser.add_argument("--geometry", type=Path, default=ROOT / "results/robustness/coherent_geometry_development.csv")
    parser.add_argument("--modeling", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--original", type=Path, default=ROOT / "results/robustness")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness/coherent_step16")
    args = parser.parse_args()
    run(args.clean, args.manifest, args.coherent, args.geometry, args.modeling, args.original, args.output)


if __name__ == "__main__":
    main()
