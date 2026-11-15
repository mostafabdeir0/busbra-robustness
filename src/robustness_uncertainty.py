"""Step 17: exploratory, paired patient bootstrap of saved development OOF results.

Intervals condition on the existing OOF fits and observed fold/diagnosis counts.
They do NOT estimate retraining variability or provide confirmatory CV inference.
No fitting, feature selection, threshold changes or held-out evaluation occurs.
"""
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .coherent_eval import PROTOCOL, ORIGINAL_PROTOCOL, require_same_clean, sha256
from .coherent_features import COHERENT_CONDITIONS
from .models import ROOT
from .robustness_eval import MODEL_NAMES, parse_boolean_column, summarize_pair, validate_clean_reference

METRICS = ("delta_balanced_accuracy", "delta_sensitivity", "delta_specificity",
           "flip_rate", "new_false_negative_rate_among_malignant")
LIMITATION = ("Exploratory pointwise percentile intervals conditional on saved OOF models and "
              "observed fold/diagnosis counts. No retraining variance, simultaneous coverage, "
              "multiplicity correction or confirmatory significance claim.")


def load_patients(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, usecols=["ID", "Case", "Pathology", "split", "dev_fold"])
    if manifest[["ID", "Case", "split"]].isna().any().any() or manifest.ID.duplicated().any():
        raise ValueError("Invalid manifest identifiers")
    dev = manifest.loc[manifest.split.eq("development")].copy()
    if dev.empty or dev[["Pathology", "dev_fold"]].isna().any().any():
        raise ValueError("Missing development manifest metadata")
    if set(dev.Case) & set(manifest.loc[manifest.split.ne("development"), "Case"]):
        raise ValueError("Leakage: patient shared with held-out split")
    if (not dev.Pathology.isin(["benign", "malignant"]).all()
            or not dev.dev_fold.isin(range(5)).all()
            or set(dev.dev_fold) != set(range(5))):
        raise ValueError("Invalid development diagnosis/fold")
    if (dev.groupby("Case")[["Pathology", "dev_fold"]].nunique() != 1).any().any():
        raise ValueError("Patient crosses folds or diagnoses")
    patients = dev.groupby("Case", as_index=False).agg(
        Pathology=("Pathology", "first"), outer_fold=("dev_fold", "first"), n_images=("ID", "size"))
    patients["actual"] = patients.Pathology.eq("malignant").astype(int)
    return patients.sort_values("Case").reset_index(drop=True)


def load_predictions(path: Path, patients: pd.DataFrame, *, coherent: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model", "condition", "Case", "actual", "outer_fold", "n_images", "eligible",
                "clean_score", "perturbed_score", "clean_predicted", "perturbed_predicted", "mean_iou"}
    if not required.issubset(frame):
        raise ValueError("Missing prediction columns")
    if coherent:
        if ("protocol" not in frame or not frame.protocol.eq(PROTOCOL).all()
                or set(frame.condition) != set(COHERENT_CONDITIONS)):
            raise ValueError("Unexpected coherent protocol/conditions")
    frame = frame.loc[frame.condition.isin(COHERENT_CONDITIONS)].copy()
    if (frame[list(required - {"perturbed_score", "perturbed_predicted", "mean_iou"})].isna().any().any()
            or set(frame.model) != set(MODEL_NAMES)
            or frame.duplicated(["model", "condition", "Case"]).any()):
        raise ValueError("Missing metadata or duplicate model-condition-patient")
    frame["eligible"] = parse_boolean_column(frame.eligible)
    if coherent and not frame.eligible.all():
        raise ValueError("Completed Step 16 must have all coherent patients eligible")
    indexed = patients.set_index("Case")
    for name in MODEL_NAMES:
        for condition in COHERENT_CONDITIONS:
            part = frame.loc[frame.model.eq(name) & frame.condition.eq(condition)]
            if len(part) != len(patients) or set(part.Case) != set(patients.Case):
                raise ValueError("Prediction patient coverage differs from locked development")
            for col in ("actual", "outer_fold", "n_images"):
                if not np.array_equal(part[col], indexed.loc[part.Case, col]):
                    raise ValueError(f"Prediction metadata mismatch: {col}")
            if not np.isfinite(part.clean_score).all() or not part.clean_predicted.isin([0, 1]).all():
                raise ValueError("Invalid clean prediction")
            eligible = part.loc[part.eligible]
            if (not np.isfinite(eligible[["perturbed_score", "mean_iou"]]).all().all()
                    or not eligible.perturbed_predicted.isin([0, 1]).all()
                    or not eligible.mean_iou.between(0, 1).all()):
                raise ValueError("Invalid eligible prediction")
            if part.loc[~part.eligible, ["perturbed_score", "perturbed_predicted"]].notna().any().any():
                raise ValueError("Ineligible patient has fabricated perturbed prediction")
            for prefix in ("clean", "perturbed"):
                sample = part if prefix == "clean" else eligible
                score = sample[f"{prefix}_score"].to_numpy(float)
                if name == "logreg" and not ((score >= 0) & (score <= 1)).all():
                    raise ValueError("LR score outside probability range")
                expected = (score >= (0.5 if name == "logreg" else 0.0)).astype(int)
                if not np.array_equal(sample[f"{prefix}_predicted"], expected):
                    raise ValueError("Prediction disagrees with locked threshold")
    return frame


def validate_references(original: pd.DataFrame, coherent: pd.DataFrame, modeling: Path,
                        step16_summary: dict, hashes: dict, patients: pd.DataFrame) -> None:
    if (step16_summary.get("protocol") != PROTOCOL
            or step16_summary.get("clean_reference_oof") != {n: "MATCHED" for n in MODEL_NAMES}
            or step16_summary.get("step14_clean_predictions") != "MATCHED"
            or step16_summary.get("held_out_features_images_predictions_used") is not False
            or step16_summary.get("development_patients") != len(patients)
            or step16_summary.get("development_images") != int(patients.n_images.sum())):
        raise ValueError("Step 16 checkpoint not verified or cohort changed")
    for key in ("manifest", "step14_predictions", "logreg_predictions", "svm_predictions"):
        if step16_summary.get("inputs", {}).get(key, {}).get("sha256") != hashes[key]["sha256"]:
            raise ValueError(f"Step 16 recorded input hash mismatch: {key}")
    rows = step16_summary.get("results", [])
    if len(rows) != 8:
        raise ValueError("Step 16 summary must contain eight model-condition rows")
    saved = pd.DataFrame(rows)
    if saved.duplicated(["model", "condition"]).any():
        raise ValueError("Duplicate Step 16 summary row")
    for name in MODEL_NAMES:
        for condition in COHERENT_CONDITIONS:
            old = original.loc[original.model.eq(name) & original.condition.eq(condition)]
            new = coherent.loc[coherent.model.eq(name) & coherent.condition.eq(condition)]
            require_same_clean(old, new)
            baseline = new[["Case", "actual", "outer_fold", "n_images", "clean_predicted", "clean_score"]].rename(
                columns={"clean_predicted": "predicted", "clean_score": "prob_malignant" if name == "logreg" else "decision_score"})
            validate_clean_reference(baseline, modeling / f"{name}_nested_cv_predictions.csv", name)
            recalculated = summarize_pair(new, model=name, condition=condition)
            match = saved.loc[saved.model.eq(name) & saved.condition.eq(condition)]
            if len(match) != 1:
                raise ValueError("Missing Step 16 summary row")
            for key, value in recalculated.items():
                if isinstance(value, (int, float)):
                    if key not in match or not np.isclose(float(match.iloc[0][key]), value, rtol=1e-8, atol=1e-9):
                        raise ValueError(f"Step 16 saved summary mismatch: {key}")


def patient_values(part: pd.DataFrame) -> np.ndarray:
    """Per-patient contributions whose means equal the five target metrics."""
    y = part.actual.to_numpy(int)
    clean = part.clean_predicted.to_numpy(int)
    pert = part.perturbed_predicted.to_numpy(int)
    if set(y) != {0, 1}:
        raise ValueError("Common patient population requires both diagnoses")
    n = len(y)
    malignant, benign = y.sum(), (1 - y).sum()
    delta_correct = (pert == y).astype(float) - (clean == y).astype(float)
    return np.column_stack((delta_correct * n / (2 * np.where(y == 1, malignant, benign)),
                            y * (pert - clean) * n / malignant,
                            (1 - y) * (clean - pert) * n / benign,
                            (pert != clean).astype(float),
                            y * (clean == 1) * (pert == 0) * n / malignant))


def paired_draws(values: np.ndarray, labels: np.ndarray, folds: np.ndarray,
                 *, n_resamples: int, seed: int) -> np.ndarray:
    """Same patient resampling indices for every model/protocol/metric column."""
    if (values.ndim != 2 or len(values) != len(labels) or len(labels) != len(folds)
            or not np.isfinite(values).all() or set(labels) != {0, 1}
            or not np.isin(folds, range(5)).all() or n_resamples < 2 or seed < 0):
        raise ValueError("Invalid paired bootstrap inputs")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero((folds == fold) & (labels == label)) for fold in range(5) for label in (0, 1)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Common patient cohort needs both diagnoses in every locked fold")
    draws = np.empty((n_resamples, values.shape[1]), dtype=float)
    for start in range(0, n_resamples, 256):
        stop = min(start + 256, n_resamples)
        batch = np.zeros((stop - start, values.shape[1]))
        for group in groups:
            ix = group[rng.integers(0, len(group), size=(stop - start, len(group)))]
            batch += values[ix].sum(axis=1)
        draws[start:stop] = batch / len(values)
    return draws


def analyze_condition(original: pd.DataFrame, coherent: pd.DataFrame, condition: str,
                      *, n_resamples: int, seed: int) -> tuple[list[dict], list[dict]]:
    parts = {(protocol, name): frame.loc[frame.condition.eq(condition) & frame.model.eq(name)].copy()
             for protocol, frame in ((ORIGINAL_PROTOCOL, original), (PROTOCOL, coherent)) for name in MODEL_NAMES}
    common = set.intersection(*(set(part.loc[part.eligible, "Case"]) for part in parts.values()))
    if not common:
        raise ValueError("No common eligible patients")
    aligned = {key: part.loc[part.Case.isin(common)].sort_values("Case").reset_index(drop=True)
               for key, part in parts.items()}
    first = next(iter(aligned.values()))
    for part in aligned.values():
        if not np.array_equal(part[["Case", "actual", "outer_fold", "n_images"]],
                              first[["Case", "actual", "outer_fold", "n_images"]]):
            raise ValueError("Unpaired patient metadata")
    keys = list(aligned)
    values = np.column_stack([patient_values(aligned[key]) for key in keys])
    draws = paired_draws(values, first.actual.to_numpy(int), first.outer_fold.to_numpy(int),
                         n_resamples=n_resamples, seed=seed)
    point = values.mean(axis=0)
    def vector(protocol, name):
        weights = np.zeros(len(keys))
        weights[keys.index((protocol, name))] = 1
        return weights
    contrasts = []
    for name in MODEL_NAMES:
        contrasts.append(("coherent_change", name, PROTOCOL, vector(PROTOCOL, name)))
        contrasts.append(("protocol_difference_coherent_minus_original", name, "both",
                          vector(PROTOCOL, name) - vector(ORIGINAL_PROTOCOL, name)))
    for protocol in (ORIGINAL_PROTOCOL, PROTOCOL):
        contrasts.append(("model_difference_svm_minus_logreg", "both", protocol,
                          vector(protocol, "svm") - vector(protocol, "logreg")))
    contrasts.append(("change_in_svm_advantage_coherent_minus_original", "both", "both",
                      vector(PROTOCOL, "svm") - vector(PROTOCOL, "logreg")
                      - vector(ORIGINAL_PROTOCOL, "svm") + vector(ORIGINAL_PROTOCOL, "logreg")))
    rows = []
    for kind, name, protocol, weights in contrasts:
        for metric_ix, metric in enumerate(METRICS):
            distribution = draws[:, metric_ix::len(METRICS)] @ weights
            low, high = np.quantile(distribution, [0.025, 0.975], method="linear")
            rows.append({"comparison": kind, "model": name, "protocol": protocol,
                         "condition": condition, "metric": metric,
                         "estimate": float(point[metric_ix::len(METRICS)] @ weights),
                         "ci_low": float(low), "ci_high": float(high),
                         "common_patients": len(common), "malignant_patients": int(first.actual.sum()),
                         "excluded_from_common": len(next(iter(parts.values()))) - len(common),
                         "n_resamples": n_resamples, "condition_seed": seed,
                         "interval_scope": "exploratory_pointwise_conditional_on_saved_OOF"})
    coverage = [{"condition": condition, "outer_fold": fold, "actual": label,
                 "common_patients": int(((first.outer_fold == fold) & (first.actual == label)).sum())}
                for fold in range(5) for label in (0, 1)]
    return rows, coverage


def run(manifest: Path, original_path: Path, coherent_dir: Path, modeling: Path,
        output: Path, *, n_resamples: int = 10000, seed: int = 42) -> dict:
    if n_resamples < 200 or seed < 0:
        raise ValueError("Use at least 200 resamples and a nonnegative seed")
    # New directory only; no pre-existing checkpoint can be overwritten.
    if output.exists():
        raise FileExistsError("Step 17 output already exists; choose a new --output directory")
    paths = {"manifest": manifest, "step14_predictions": original_path,
             "step16_predictions": coherent_dir / "coherent_paired_patient_predictions.csv",
             "step16_summary": coherent_dir / "coherent_summary.json",
             **{f"{name}_predictions": modeling / f"{name}_nested_cv_predictions.csv" for name in MODEL_NAMES}}
    hashes = {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in paths.items()}
    patients = load_patients(manifest)
    original = load_predictions(original_path, patients, coherent=False)
    coherent = load_predictions(paths["step16_predictions"], patients, coherent=True)
    step16 = json.loads(paths["step16_summary"].read_text(encoding="utf-8"))
    validate_references(original, coherent, modeling, step16, hashes, patients)
    rows, coverage = [], []
    print(f"STEP 17 — PAIRED PATIENT UNCERTAINTY\nDevelopment patients: {len(patients)} | Resamples: {n_resamples}", flush=True)
    for offset, condition in enumerate(COHERENT_CONDITIONS):
        rec, counts = analyze_condition(original, coherent, condition, n_resamples=n_resamples, seed=seed + offset)
        rows.extend(rec)
        coverage.extend(counts)
        print(f"{condition}: {rec[0]['common_patients']} common patients; paired draws complete", flush=True)
    intervals = pd.DataFrame(rows)
    if any(sha256(path) != hashes[key]["sha256"] for key, path in paths.items()):
        raise RuntimeError("Saved inputs changed during Step 17")
    result = {"step": 17, "status": "exploratory_posthoc_uncertainty",
              "method": "paired patient bootstrap stratified by locked outer fold and diagnosis; percentile 95%",
              "n_resamples": n_resamples, "seed": seed,
              "condition_seeds": {c: seed + i for i, c in enumerate(COHERENT_CONDITIONS)},
              "development_patients": len(patients), "interval_rows": len(intervals),
              "population": "per-condition intersection across both models and original/coherent protocols",
              "limitations": LIMITATION, "refitting_performed": False, "held_out_evaluation_used": False,
              "clean_references": "MATCHED to nested OOF and Step 14",
              "step16_summary_recomputed": "MATCHED", "inputs": hashes,
              "source_sha256": sha256(Path(__file__)),
              "versions": {pkg: version(pkg) for pkg in ("numpy", "pandas", "scikit-learn")}}
    output.mkdir(parents=True, exist_ok=False)
    intervals.to_csv(output / "paired_bootstrap_intervals.csv", index=False, float_format="%.12g")
    pd.DataFrame(coverage).to_csv(output / "common_patient_coverage.csv", index=False)
    (output / "uncertainty_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    main = intervals.loc[intervals.comparison.eq("model_difference_svm_minus_logreg")
                         & intervals.protocol.eq(PROTOCOL) & intervals.metric.eq("delta_balanced_accuracy")]
    print(main[["condition", "estimate", "ci_low", "ci_high"]].to_string(index=False))
    print(f"{LIMITATION}\nClean references: MATCHED | Held-out evaluation: NOT USED\nSaved: {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--original", type=Path, default=ROOT / "results/robustness/robustness_paired_patient_predictions.csv")
    parser.add_argument("--coherent", type=Path, default=ROOT / "results/robustness/coherent_step16_run2")
    parser.add_argument("--modeling", type=Path, default=ROOT / "results/modeling")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness/uncertainty_step17")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.manifest, args.original, args.coherent, args.modeling, args.output,
        n_resamples=args.resamples, seed=args.seed)


if __name__ == "__main__":
    main()
