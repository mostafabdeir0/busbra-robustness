"""Step 21: one frozen held-out evaluation; no model fitting or selection.

--preflight verifies files/versions/models without reading test images or predicting.
The real run creates an exclusive marker in the calibration checkpoint BEFORE
opening test images. A changed output path cannot bypass that marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from . import patient_calibration as pc
from . import stable_selection as ss
from .coherent_features import COHERENT_CONDITIONS
from .coherent_perturbations import coherent_boundary_change
from .features import FEATURE_COLUMNS, extract_features, preprocess_pair, read_pair
from .models import ROOT
from .perturbations import TARGET_IOUS, mask_iou
from .robustness_eval import MODEL_NAMES, summarize_pair

CONDITIONS = ("clean", *COHERENT_CONDITIONS)
METRICS = ("balanced_accuracy", "sensitivity", "specificity", "roc_auc", "brier_score", "log_loss")
MARKER = "held_out_evaluation_started.json"
UNCERTAINTY_PLAN = "10000 paired patient bootstrap replicates within test diagnosis, seed 42; pointwise percentile 95%; fixed fitted models"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_bytes(ss.canonical_json(value).encode("utf-8"))


def image_id_digest(ids):
    return hashlib.sha256(json.dumps(sorted(str(i) for i in ids)).encode()).hexdigest()


def verify_frozen(calibration, manifest_path, archive_path, audit_path):
    protocol_path, summary_path = calibration / "final_evaluation_protocol.json", calibration / "calibration_summary.json"
    protocol, summary = read_json(protocol_path), read_json(summary_path)
    model_path = calibration / "frozen_calibrated_models.joblib"
    if (summary.get("status") != "calibration_complete_final_protocol_frozen"
            or summary.get("held_out_evaluation_used") is not False
            or summary.get("final_protocol_sha256") != pc.sha256(protocol_path)
            or protocol.get("status") != "frozen_before_held_out_evaluation"
            or protocol.get("design") != pc.calibration_design()
            or protocol.get("primary_model") != "svm" or protocol.get("comparator") != "logreg"
            or protocol.get("threshold") != .5 or protocol.get("held_out_conditions") != list(CONDITIONS)
            or protocol.get("uncertainty_plan") != UNCERTAINTY_PLAN):
        raise ValueError("Frozen protocol/summary verification failed")
    if pc.sha256(model_path) != protocol.get("model_artifact_sha256") or summary.get("model_artifact_sha256") != protocol.get("model_artifact_sha256"):
        raise ValueError("Frozen model artifact hash mismatch")
    source_root = Path(pc.__file__).resolve().parent
    for filename, fingerprint in protocol["source_sha256"].items():
        if Path(filename).name != filename or pc.sha256(source_root / filename) != fingerprint:
            raise ValueError(f"Frozen source hash mismatch: {filename}")
    for package, expected in summary["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Frozen package version mismatch: {package}")
    input_checks = {}
    for key, item in protocol["inputs"].items():
        path = manifest_path if key == "manifest" else Path(item["path"])
        if pc.sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen training input hash mismatch: {key}")
        input_checks[key] = {"path": str(path.resolve()), "sha256": item["sha256"]}
    manifest = pd.read_csv(manifest_path, dtype={"ID": str})
    required = ["ID", "Case", "Pathology", "split", "dev_fold"]
    if (manifest[required].isna().any().any() or manifest.ID.duplicated().any()
            or not manifest.split.isin(["development", "test"]).all()
            or not manifest.Pathology.isin(["benign", "malignant"]).all()
            or manifest.groupby("Case").Pathology.nunique().ne(1).any()
            or manifest.groupby("Case").split.nunique().ne(1).any()):
        raise ValueError("Invalid locked patient manifest")
    test = manifest.loc[manifest.split.eq("test")].sort_values("ID").reset_index(drop=True)
    dev = manifest.loc[manifest.split.eq("development")]
    if (len(test) != protocol["held_out_images"] or test.Case.nunique() != protocol["held_out_patients"]
            or ss.case_digest(set(test.Case)) != protocol["held_out_case_sha256"]
            or image_id_digest(test.ID) != protocol["held_out_image_ids_sha256"]
            or set(test.Case) & set(dev.Case) or not test.dev_fold.eq(-1).all()
            or set(test.Pathology) != {"benign", "malignant"}):
        raise ValueError("Held-out cohort differs from frozen identifiers")
    archive_hash = pc.sha256(archive_path)  # Hash bytes only: no image decoding.
    if read_json(audit_path).get("archive_sha256") != archive_hash:
        raise ValueError("Dataset archive differs from the original dataset audit")
    bundle = joblib.load(model_path)  # Load only after artifact hash and version checks.
    if (bundle.get("design") != protocol["design"] or bundle.get("features") != list(FEATURE_COLUMNS)
            or bundle.get("development_case_sha256") != ss.case_digest(set(dev.Case))
            or set(bundle.get("models", {})) != set(MODEL_NAMES)):
        raise ValueError("Frozen model bundle metadata differs")
    for name, ensemble in bundle["models"].items():
        if (ensemble.get("model") != name or len(ensemble.get("members", [])) != 3
                or any(tuple(m["base_model"].columns) != FEATURE_COLUMNS for m in ensemble["members"])):
            raise ValueError("Frozen ensemble structure differs")
    checks = {"protocol_sha256": pc.sha256(protocol_path), "summary_sha256": pc.sha256(summary_path),
              "model_sha256": pc.sha256(model_path), "manifest_sha256": pc.sha256(manifest_path),
              "archive_sha256": archive_hash, "audit_sha256": pc.sha256(audit_path),
              "inputs": input_checks, "step21_source_sha256": pc.sha256(Path(__file__))}
    return protocol, bundle, test, checks


def extract_test_features(archive_path, test):
    clean_rows, changed_rows = [], []
    with ZipFile(archive_path) as archive:
        for index, row in enumerate(test.itertuples(index=False), 1):
            image, mask = preprocess_pair(*read_pair(archive, row.ID))
            initial_image, initial_mask = image.copy(), mask.copy()
            metadata = {"ID": row.ID, "Case": int(row.Case), "Pathology": row.Pathology,
                        "split": "test", "dev_fold": -1}
            clean_rows.append({**metadata, **extract_features(image, mask)})
            for condition in COHERENT_CONDITIONS:
                kind, severity = condition.split("_")
                changed = coherent_boundary_change(mask, TARGET_IOUS[severity], kind)
                if np.array_equal(changed, mask):
                    raise ValueError(f"Unavailable coherent condition: {row.ID}/{condition}")
                iou = mask_iou(mask, changed)
                if not 0 < iou < 1:
                    raise ValueError("Invalid held-out coherent IoU")
                changed_rows.append({**metadata, "condition": condition, "valid": True, "actual_iou": iou,
                                     "changed_pixels": int(np.logical_xor(mask, changed).sum()),
                                     **extract_features(image, changed)})
            if not np.array_equal(image, initial_image) or not np.array_equal(mask, initial_mask):
                raise ValueError("Original test image/mask was mutated")
            if index % 50 == 0:
                print(f"Held-out features: {index}/{len(test)} images", flush=True)
    clean, coherent = pd.DataFrame(clean_rows), pd.DataFrame(changed_rows)
    if (len(clean) != len(test) or clean.ID.duplicated().any() or set(clean.ID) != set(test.ID)
            or len(coherent) != len(test) * 4 or coherent.duplicated(["ID", "condition"]).any()
            or not np.isfinite(clean[list(FEATURE_COLUMNS)].to_numpy(float)).all()
            or not np.isfinite(coherent[list(FEATURE_COLUMNS)].to_numpy(float)).all()):
        raise ValueError("Incomplete or nonfinite held-out feature coverage")
    return clean, coherent


def predict_all(bundle, clean, coherent):
    predictions, pairs, summaries, reliability = [], [], [], []
    for name in MODEL_NAMES:
        ensemble = bundle["models"][name]
        baseline = pc.predict_ensemble(ensemble, clean)
        for condition in CONDITIONS:
            rows = baseline.copy() if condition == "clean" else pc.predict_ensemble(ensemble, coherent.loc[coherent.condition.eq(condition)])
            rows["model"], rows["condition"], rows["outer_fold"] = name, condition, -1
            predictions.append(rows)
            summaries.append({"model": name, "condition": condition, "patients": len(rows), **pc.probability_metrics(rows)})
            reliability.extend({**b, "condition": condition} for b in pc.calibration_bins(rows, name))
            if condition != "clean":
                changed = coherent.loc[coherent.condition.eq(condition)]
                p = rows.set_index("Case")
                pair = baseline[["Case", "actual", "n_images", "prob_malignant", "predicted"]].copy().rename(
                    columns={"prob_malignant": "clean_score", "predicted": "clean_predicted"})
                if set(p.index) != set(pair.Case) or not np.array_equal(pair.n_images, p.loc[pair.Case, "n_images"]):
                    raise ValueError("Missing held-out patient views")
                pair["perturbed_score"] = p.loc[pair.Case, "prob_malignant"].to_numpy(float)
                pair["perturbed_predicted"] = p.loc[pair.Case, "predicted"].to_numpy(int)
                pair["mean_iou"] = changed.groupby("Case").actual_iou.mean().loc[pair.Case].to_numpy(float)
                pair["outer_fold"], pair["eligible"], pair["model"], pair["condition"] = -1, True, name, condition
                pairs.append(pair)
    predictions, pairs = pd.concat(predictions, ignore_index=True), pd.concat(pairs, ignore_index=True)
    n = clean.Case.nunique()
    if (len(predictions) != n * 10 or predictions.duplicated(["model", "condition", "Case"]).any()
            or len(pairs) != n * 8 or pairs.duplicated(["model", "condition", "Case"]).any()):
        raise ValueError("Incomplete held-out prediction coverage")
    pair_summary = [summarize_pair(p, model=name, condition=condition)
                    for (name, condition), p in pairs.groupby(["model", "condition"])]
    return predictions, pairs, pd.DataFrame(summaries), pd.DataFrame(pair_summary), pd.DataFrame(reliability)


def bootstrap_weights(y, *, resamples=10000, seed=42):
    if set(y) != {0, 1} or resamples < 2:
        raise ValueError("Patient bootstrap requires both diagnoses")
    rng = np.random.default_rng(seed)
    counts = np.zeros((resamples, len(y)), dtype=np.int32)
    for label in (0, 1):
        ix = np.flatnonzero(y == label)
        counts[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=resamples)
    return counts


def auc_from_counts(y, scores, counts):
    order = np.argsort(scores, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(scores[order]) != 0) + 1]
    positive = np.add.reduceat(counts[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(counts[:, order] * (1 - y[order]), starts, axis=1)
    numerator = np.sum(positive * (np.cumsum(negative, axis=1) - .5 * negative), axis=1)
    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    if (denominator == 0).any():
        raise ValueError("Bootstrap AUC sample lost a diagnosis")
    return numerator / denominator


def metric_draws(y, probability, counts):
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid bootstrap probability")
    pred = (p >= .5).astype(int)
    malignant, benign = counts @ y, counts @ (1 - y)
    sensitivity = (counts @ (y * pred)) / malignant
    specificity = (counts @ ((1 - y) * (1 - pred))) / benign
    n = counts.sum(axis=1)
    eps = np.finfo(float).eps
    loss = -(y * np.log(np.clip(p, eps, 1 - eps)) + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps)))
    return {"balanced_accuracy": .5 * (sensitivity + specificity), "sensitivity": sensitivity,
            "specificity": specificity, "roc_auc": auc_from_counts(y, p, counts),
            "brier_score": (counts @ ((p - y) ** 2)) / n, "log_loss": (counts @ loss) / n}


def paired_intervals(predictions, *, resamples=10000, seed=42):
    groups = {(name, condition): predictions.loc[predictions.model.eq(name) & predictions.condition.eq(condition)]
              .sort_values("Case").reset_index(drop=True) for name in MODEL_NAMES for condition in CONDITIONS}
    first = groups[("svm", "clean")]
    if first.Case.duplicated().any():
        raise ValueError("Duplicate bootstrap patient")
    for frame in groups.values():
        if (frame.Case.duplicated().any() or not np.array_equal(frame[["Case", "actual", "n_images"]], first[["Case", "actual", "n_images"]])):
            raise ValueError("Bootstrap comparisons are not patient-paired")
    y = first.actual.to_numpy(int)
    counts = bootstrap_weights(y, resamples=resamples, seed=seed)
    draws, points = {}, {}
    for key, frame in groups.items():
        draws[key] = metric_draws(y, frame.prob_malignant.to_numpy(float), counts)
        points[key] = pc.probability_metrics(frame)
    rows = []
    def emit(kind, model, condition, metric, point, distribution):
        low, high = np.quantile(distribution, [.025, .975], method="linear")
        rows.append({"comparison": kind, "model": model, "condition": condition, "metric": metric,
                     "estimate": float(point), "ci_low": float(low), "ci_high": float(high),
                     "patients": len(y), "resamples": resamples, "seed": seed,
                     "primary": kind == "absolute" and model == "svm" and condition == "clean" and metric == "balanced_accuracy",
                     "higher_is_better": metric not in ("brier_score", "log_loss", "flip_rate", "new_false_negative_rate"),
                     "interval_scope": "pointwise_percentile_conditional_on_frozen_models"})
    for name in MODEL_NAMES:
        for condition in CONDITIONS:
            key, base = (name, condition), (name, "clean")
            for metric in METRICS:
                emit("absolute", name, condition, metric, points[key][metric], draws[key][metric])
                if condition != "clean":
                    emit("change_from_clean", name, condition, metric,
                         points[key][metric] - points[base][metric], draws[key][metric] - draws[base][metric])
            if condition != "clean":
                clean_pred, changed_pred = groups[base].predicted.to_numpy(int), groups[key].predicted.to_numpy(int)
                flip = (clean_pred != changed_pred).astype(int)
                new_fn = y * (clean_pred == 1) * (changed_pred == 0)
                emit("paired_event", name, condition, "flip_rate", flip.mean(), (counts @ flip) / len(y))
                emit("paired_event", name, condition, "new_false_negative_rate", new_fn.sum() / y.sum(), (counts @ new_fn) / y.sum())
        composite_point = np.mean([points[(name, c)]["balanced_accuracy"] for c in CONDITIONS])
        composite_draw = np.mean([draws[(name, c)]["balanced_accuracy"] for c in CONDITIONS], axis=0)
        emit("five_condition_mean", name, "five_condition_mean", "balanced_accuracy", composite_point, composite_draw)
    for condition in CONDITIONS:
        for metric in METRICS:
            difference = points[("svm", condition)][metric] - points[("logreg", condition)][metric]
            distribution = draws[("svm", condition)][metric] - draws[("logreg", condition)][metric]
            emit("svm_minus_logreg", "both", condition, metric, difference, distribution)
            if condition != "clean":
                base_diff = points[("svm", "clean")][metric] - points[("logreg", "clean")][metric]
                base_draw = draws[("svm", "clean")][metric] - draws[("logreg", "clean")][metric]
                emit("svm_minus_logreg_change", "both", condition, metric, difference - base_diff, distribution - base_draw)
    emit("svm_minus_logreg_five_condition_mean", "both", "five_condition_mean", "balanced_accuracy",
         np.mean([points[("svm", c)]["balanced_accuracy"] - points[("logreg", c)]["balanced_accuracy"] for c in CONDITIONS]),
         np.mean([draws[("svm", c)]["balanced_accuracy"] - draws[("logreg", c)]["balanced_accuracy"] for c in CONDITIONS], axis=0))
    return pd.DataFrame(rows), counts, first[["Case", "actual"]]


def update_marker(path, record):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, record)
    temporary.replace(path)


def run(calibration, manifest, archive, audit, output):
    if output.exists():
        raise FileExistsError("Held-out output exists; preserve it and do not rerun")
    marker = calibration / MARKER
    if marker.exists():
        raise FileExistsError("This frozen checkpoint already started its held-out evaluation; inspect the existing marker/results")
    protocol, bundle, test, checks = verify_frozen(calibration, manifest, archive, audit)
    record = {"status": "started", "output": str(output.resolve()), "phase": "feature_extraction", **checks}
    # Exclusive creation protects against concurrent runs and alternate output paths.
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(ss.canonical_json(record))
    try:
        clean, coherent = extract_test_features(archive, test)
        record["phase"] = "prediction"
        update_marker(marker, record)
        predictions, pairs, summaries, pair_summary, reliability = predict_all(bundle, clean, coherent)
        record["phase"] = "bootstrap_and_publication"
        update_marker(marker, record)
        intervals, counts, patients = paired_intervals(predictions)  # Exactly frozen 10,000/42.
        if pc.sha256(archive) != checks["archive_sha256"] or pc.sha256(manifest) != checks["manifest_sha256"]:
            raise RuntimeError("Dataset or manifest changed during held-out evaluation")
        if pc.sha256(calibration / "frozen_calibrated_models.joblib") != checks["model_sha256"] or pc.sha256(calibration / "final_evaluation_protocol.json") != checks["protocol_sha256"]:
            raise RuntimeError("Frozen models or protocol changed during evaluation")
        primary = intervals.loc[intervals.primary].to_dict("records")
        if len(primary) != 1 or len(intervals) != 181:
            raise RuntimeError("Primary endpoint or interval coverage differs")
        result = {"step": 21, "status": "locked_held_out_evaluation_complete", "held_out_patients": int(test.Case.nunique()),
                  "held_out_images": len(test), "clean_feature_vectors": len(clean), "coherent_feature_vectors": len(coherent),
                  "patient_prediction_rows": len(predictions), "paired_patient_rows": len(pairs),
                  "primary_endpoint": protocol["primary_endpoint"], "primary_result": primary[0],
                  "threshold": .5, "model_fits": 0, "calibrator_fits": 0, "selection_or_retuning": False,
                  "bootstrap": {"resamples": 10000, "seed": 42, "method": "paired diagnosis-stratified multinomial patient counts; percentile 95%"},
                  "provenance": checks, "conditions": list(CONDITIONS), "clean_metrics": summaries.loc[summaries.condition.eq("clean")].to_dict("records"),
                  "limitations": "Single dataset research evaluation; posthoc development model choice; fixed research threshold; pointwise intervals conditional on fitted models; no clinical or multiplicity-adjusted claim."}
        output.mkdir(parents=True, exist_ok=False)
        for filename, table in (("held_out_clean_features.csv", clean), ("held_out_coherent_features.csv", coherent),
                                ("held_out_patient_predictions.csv", predictions), ("held_out_paired_predictions.csv", pairs),
                                ("held_out_condition_metrics.csv", summaries), ("held_out_paired_summary.csv", pair_summary),
                                ("held_out_reliability_bins.csv", reliability), ("held_out_intervals.csv", intervals)):
            table.to_csv(output / filename, index=False, float_format="%.12g")
        np.savez_compressed(output / "paired_bootstrap_weights.npz", counts=counts,
                            Case=patients.Case.to_numpy(), actual=patients.actual.to_numpy())
        write_json(output / "held_out_summary.json", result)
        write_json(output / "frozen_protocol_copy.json", protocol)
        result_hash = pc.sha256(output / "held_out_summary.json")
        record.update(status="complete", phase="complete", summary_sha256=result_hash,
                      published_files_sha256={p.name: pc.sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
        update_marker(marker, record)
        print(summaries.loc[summaries.condition.eq("clean"), ["model", "balanced_accuracy", "sensitivity", "specificity", "roc_auc", "brier_score", "log_loss"]].to_string(index=False))
        p = primary[0]
        print(f"Primary SVM clean BA: {100*p['estimate']:.2f}% | pointwise 95% CI [{100*p['ci_low']:.2f}%, {100*p['ci_high']:.2f}%]\n"
              f"Frozen models only | Fits: 0 | Retuning: NONE\nSaved: {output}", flush=True)
        return result
    except Exception as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        update_marker(marker, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=ROOT / "results/robustness/calibration_step20")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/image_splits.csv")
    parser.add_argument("--zip", type=Path, default=ROOT / "data/raw/BUSBRA.zip")
    parser.add_argument("--audit", type=Path, default=ROOT / "results/audit/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/robustness/heldout_step21")
    parser.add_argument("--preflight", action="store_true", help="Integrity check only; no image decoding, predictions or marker creation")
    args = parser.parse_args()
    if args.preflight:
        _, _, test, checks = verify_frozen(args.calibration, args.manifest, args.zip, args.audit)
        print(f"STEP 21 PREFLIGHT: PASSED | {test.Case.nunique()} held-out patients | {len(test)} images\n"
              f"Protocol/models/sources/versions/archive: MATCHED\nImages decoded: 0 | Predictions: 0 | Marker created: NO\n"
              f"Protocol SHA256: {checks['protocol_sha256']}")
    else:
        run(args.calibration, args.manifest, args.zip, args.audit, args.output)


if __name__ == "__main__":
    main()
