"""Step 22: read-only synthesis of saved experiments; no fitting or inference.

Writes an editable Markdown report, self-contained HTML, publication figures,
patient-level error audit, tables and checksums. The Step 21 marker must be complete.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import patient_calibration as pc
from . import stable_selection as ss
from .coherent_features import COHERENT_CONDITIONS
from .models import ROOT
from .robustness_eval import MODEL_NAMES, parse_boolean_column, summarize_pair

COLORS = {"logreg": "#276B8E", "svm": "#B56A25"}
LABELS = {"logreg": "Logistic Regression", "svm": "RBF SVM"}
COND_LABELS = {"erosion_mild": "Mild erosion", "dilation_mild": "Mild dilation",
               "erosion_moderate": "Moderate erosion", "dilation_moderate": "Moderate dilation", "clean": "Clean"}
FIGURES = ("01_clean_metrics", "02_clean_roc", "03_reliability", "04_confusion",
           "05_heldout_boundary_changes", "06_original_and_coherent", "07_feature_selection")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_paths(root):
    robust = root / "results/robustness"
    return {"manifest": root / "splits/image_splits.csv", "split_summary": root / "splits/split_summary.json",
            "heldout": robust / "heldout_step21", "calibration": robust / "calibration_step20",
            "step19": robust / "selection_step19", "step16": robust / "coherent_step16_run2",
            "step17": robust / "uncertainty_step17", "original": robust,
            "modeling": root / "results/modeling"}


def validate_saved_evaluation(paths):
    heldout, calibration = paths["heldout"], paths["calibration"]
    marker = read_json(calibration / "held_out_evaluation_started.json")
    summary = read_json(heldout / "held_out_summary.json")
    protocol = read_json(calibration / "final_evaluation_protocol.json")
    if marker.get("status") != "complete" or summary.get("status") != "locked_held_out_evaluation_complete":
        raise ValueError("Held-out evaluation is not complete; reporting cannot run it")
    for filename, fingerprint in marker["published_files_sha256"].items():
        if Path(filename).name != filename or pc.sha256(heldout / filename) != fingerprint:
            raise ValueError(f"Published held-out result hash mismatch: {filename}")
    if (marker["summary_sha256"] != pc.sha256(heldout / "held_out_summary.json")
            or marker["protocol_sha256"] != pc.sha256(calibration / "final_evaluation_protocol.json")
            or marker["model_sha256"] != pc.sha256(calibration / "frozen_calibrated_models.joblib")
            or summary["provenance"]["protocol_sha256"] != marker["protocol_sha256"]
            or pc.sha256(heldout / "frozen_protocol_copy.json") != marker["protocol_sha256"]
            or pc.sha256(paths["manifest"]) != marker["manifest_sha256"]
            or summary["model_fits"] != 0 or summary["calibrator_fits"] != 0 or summary["selection_or_retuning"] is not False):
        raise ValueError("Frozen held-out provenance or zero-fit checks failed")
    predictions = pd.read_csv(heldout / "held_out_patient_predictions.csv")
    pairs = pd.read_csv(heldout / "held_out_paired_predictions.csv")
    metrics = pd.read_csv(heldout / "held_out_condition_metrics.csv")
    pair_metrics = pd.read_csv(heldout / "held_out_paired_summary.csv")
    intervals = pd.read_csv(heldout / "held_out_intervals.csv")
    intervals["primary"] = parse_boolean_column(intervals.primary)
    manifest = pd.read_csv(paths["manifest"], dtype={"ID": str})
    test = manifest.loc[manifest.split.eq("test")]
    metadata = test.groupby("Case").agg(actual=("Pathology", lambda y: int(y.iloc[0] == "malignant")), n_images=("ID", "size"))
    n = len(metadata)
    if (n != summary["held_out_patients"] or len(test) != summary["held_out_images"]
            or len(predictions) != n * 10 or predictions.duplicated(["model", "condition", "Case"]).any()
            or len(pairs) != n * 8 or pairs.duplicated(["model", "condition", "Case"]).any()
            or len(metrics) != 10 or metrics.duplicated(["model", "condition"]).any()
            or len(pair_metrics) != 8 or pair_metrics.duplicated(["model", "condition"]).any()
            or len(intervals) != 181 or intervals.primary.sum() != 1):
        raise ValueError("Saved held-out coverage differs")
    for name in MODEL_NAMES:
        for condition in ("clean", *COHERENT_CONDITIONS):
            p = predictions.loc[predictions.model.eq(name) & predictions.condition.eq(condition)].sort_values("Case")
            if set(p.Case) != set(metadata.index) or not np.array_equal(p[["actual", "n_images"]], metadata.loc[p.Case, ["actual", "n_images"]]):
                raise ValueError("Saved patient metadata differs from the locked manifest")
            if not np.array_equal(p.predicted, p.prob_malignant.ge(.5).astype(int)):
                raise ValueError("Saved predictions disagree with the frozen threshold")
            recalculated = pc.probability_metrics(p)
            row = metrics.loc[metrics.model.eq(name) & metrics.condition.eq(condition)]
            if len(row) != 1:
                raise ValueError("Missing saved metric row")
            for key, value in recalculated.items():
                if not np.isclose(row.iloc[0][key], value, rtol=1e-8, atol=1e-9):
                    raise ValueError(f"Saved metric mismatch: {key}")
            if condition != "clean":
                pair = pairs.loc[pairs.model.eq(name) & pairs.condition.eq(condition)].sort_values("Case")
                baseline = predictions.loc[predictions.model.eq(name) & predictions.condition.eq("clean")].sort_values("Case")
                if (not np.array_equal(pair.Case, p.Case)
                        or not np.allclose(pair.clean_score, baseline.prob_malignant, rtol=1e-8, atol=1e-9)
                        or not np.allclose(pair.perturbed_score, p.prob_malignant, rtol=1e-8, atol=1e-9)
                        or not np.array_equal(pair.clean_predicted, baseline.predicted)
                        or not np.array_equal(pair.perturbed_predicted, p.predicted)):
                    raise ValueError("Saved clean/perturbed patient pairing differs")
                recalculated_pair = summarize_pair(pair.assign(eligible=True), model=name, condition=condition)
                saved_pair = pair_metrics.loc[pair_metrics.model.eq(name) & pair_metrics.condition.eq(condition)]
                if len(saved_pair) != 1:
                    raise ValueError("Missing saved pair summary")
                for key, value in recalculated_pair.items():
                    if isinstance(value, (int, float)) and not np.isclose(float(saved_pair.iloc[0][key]), value, rtol=1e-8, atol=1e-9):
                        raise ValueError(f"Saved pair metric mismatch: {key}")
    primary = intervals.loc[intervals.primary].iloc[0]
    if (primary["model"] != "svm" or primary["condition"] != "clean" or primary["metric"] != "balanced_accuracy"
            or not np.isclose(primary.estimate, summary["primary_result"]["estimate"])
            or not np.isclose(primary.ci_low, summary["primary_result"]["ci_low"])
            or not np.isclose(primary.ci_high, summary["primary_result"]["ci_high"])):
        raise ValueError("Primary endpoint differs from the saved analysis")
    return {"summary": summary, "protocol": protocol, "marker": marker, "predictions": predictions,
            "pairs": pairs, "metrics": metrics, "pair_metrics": pair_metrics,
            "intervals": intervals, "manifest": manifest, "primary": primary}


def failure_analysis(predictions, manifest):
    clean = predictions.loc[predictions.condition.eq("clean")]
    lr, svm = [clean.loc[clean.model.eq(name)].set_index("Case").sort_index() for name in MODEL_NAMES]
    if not np.array_equal(lr.index, svm.index):
        raise ValueError("Clean model patients are not paired")
    cases = pd.DataFrame({"Case": lr.index, "actual": lr.actual, "n_images": lr.n_images,
                          "lr_probability": lr.prob_malignant, "svm_probability": svm.prob_malignant,
                          "lr_predicted": lr.predicted, "svm_predicted": svm.predicted}).reset_index(drop=True)
    for short in ("lr", "svm"):
        cases[f"{short}_error"] = cases[f"{short}_predicted"].ne(cases.actual)
        cases[f"{short}_false_negative"] = cases.actual.eq(1) & cases[f"{short}_predicted"].eq(0)
        cases[f"{short}_false_positive"] = cases.actual.eq(0) & cases[f"{short}_predicted"].eq(1)
    image_ids = manifest.loc[manifest.split.eq("test")].groupby("Case").ID.agg(lambda ids: ";".join(sorted(ids)))
    cases["image_ids"] = image_ids.loc[cases.Case].to_numpy()
    overlap = {"both_wrong": int((cases.lr_error & cases.svm_error).sum()),
               "lr_only_wrong": int((cases.lr_error & ~cases.svm_error).sum()),
               "svm_only_wrong": int((~cases.lr_error & cases.svm_error).sum()),
               "both_correct": int((~cases.lr_error & ~cases.svm_error).sum()),
               "both_false_negative": int((cases.lr_false_negative & cases.svm_false_negative).sum()),
               "malignant_patients": int(cases.actual.sum()), "benign_patients": int(cases.actual.eq(0).sum())}
    examples = []
    for label, subset, column, ascending in (
            ("SVM false negative", cases.loc[cases.svm_false_negative], "svm_probability", True),
            ("SVM false positive", cases.loc[cases.svm_false_positive], "svm_probability", False)):
        for row in subset.sort_values([column, "Case"], ascending=[ascending, True]).head(3).itertuples():
            examples.append({"category": label, "Case": row.Case, "actual": row.actual,
                             "lr_probability": row.lr_probability, "svm_probability": row.svm_probability,
                             "image_ids": row.image_ids})
    return cases, overlap, pd.DataFrame(examples)


def load_history(paths):
    original = pd.read_csv(paths["original"] / "robustness_paired_condition_summary.csv")
    coherent = pd.read_csv(paths["step16"] / "coherent_condition_summary.csv")
    selection = pd.read_csv(paths["step19"] / "clean_summary.csv")
    selections = read_json(paths["step19"] / "outer_selections.json")
    cal = pd.read_csv(paths["calibration"] / "calibrated_clean_summary.csv")
    selection_summary = read_json(paths["step19"] / "evaluation_summary.json")
    cal_summary = read_json(paths["calibration"] / "calibration_summary.json")
    uncertainty = pd.read_csv(paths["step17"] / "paired_bootstrap_intervals.csv")
    costs = [{"stage": "19 nested feature selection", "base_inner_fits": selection_summary["fit_counts"]["inner"],
              "base_refits": selection_summary["fit_counts"]["outer_refits"], "calibrator_fits": 0,
              "baseline_check_fits": selection_summary["fit_counts"]["baseline_reproducibility"],
              "recorded_seconds": selection_summary["elapsed_nested_seconds"]},
             {"stage": "20 nested calibration and final fit", "base_inner_fits": cal_summary["base_inner_fits"],
              "base_refits": cal_summary["base_refits"], "calibrator_fits": cal_summary["sigmoid_fits"],
              "baseline_check_fits": 0, "recorded_seconds": None},
             {"stage": "21 held-out evaluation", "base_inner_fits": 0, "base_refits": 0, "calibrator_fits": 0,
              "baseline_check_fits": 0, "recorded_seconds": None}]
    return {"original": original, "coherent": coherent, "selection": selection, "selections": selections,
            "calibration": cal, "uncertainty": uncertainty, "costs": pd.DataFrame(costs),
            "baseline": {name: read_json(paths["modeling"] / f"{name}_nested_cv_summary.json") for name in MODEL_NAMES}}


def model_sizes(calibration):
    # Inspect fitted state only; never call predict/fit while reporting.
    bundle = joblib.load(calibration / "frozen_calibrated_models.joblib")
    rows = []
    for name, ensemble in bundle["models"].items():
        for index, member in enumerate(ensemble["members"]):
            classifier = member["base_model"].pipeline.named_steps["classifier"]
            rows.append({"model": name, "member": index, "features": len(member["base_model"].columns),
                         "support_vectors": int(classifier.support_vectors_.shape[0]) if name == "svm" else None,
                         "linear_coefficients": int(classifier.coef_.size) if name == "logreg" else None})
    return pd.DataFrame(rows)


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=210, bbox_inches="tight", facecolor="white")
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(context, history, directory):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False, "svg.fonttype": "none"})
    ci = context["intervals"]
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 3.3))
    for ax, metric, title in zip(axes, ["balanced_accuracy", "sensitivity", "specificity", "roc_auc"], ["Balanced accuracy", "Sensitivity", "Specificity", "ROC AUC"]):
        for index, name in enumerate(MODEL_NAMES):
            row = ci.loc[ci.comparison.eq("absolute") & ci.model.eq(name) & ci.condition.eq("clean") & ci.metric.eq(metric)].iloc[0]
            ax.plot([index, index], [row.ci_low, row.ci_high], color=COLORS[name], lw=2)
            ax.scatter(index, row.estimate, color=COLORS[name], s=45, zorder=3)
        ax.set(xticks=[0, 1], xticklabels=["LR", "SVM"], ylim=(.45, 1.01), title=title)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("Held-out clean performance with pointwise 95% intervals", fontsize=13)
    fig.tight_layout()
    save_figure(fig, directory, FIGURES[0])
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in MODEL_NAMES:
        p = context["predictions"].loc[context["predictions"].model.eq(name) & context["predictions"].condition.eq("clean")]
        fpr, tpr, _ = roc_curve(p.actual, p.prob_malignant)
        auc = context["metrics"].loc[context["metrics"].model.eq(name) & context["metrics"].condition.eq("clean"), "roc_auc"].iloc[0]
        ax.plot(fpr, tpr, label=f"{LABELS[name]}  AUC {auc:.4f}", color=COLORS[name], lw=2)
    ax.plot([0, 1], [0, 1], "--", color=".65", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out clean ROC curves", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=.15)
    save_figure(fig, directory, FIGURES[1])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, name in zip(axes, MODEL_NAMES):
        p = context["predictions"].loc[context["predictions"].model.eq(name) & context["predictions"].condition.eq("clean")]
        bins = pd.DataFrame(pc.calibration_bins(p, name)).dropna()
        ax.plot([0, 1], [0, 1], "--", color=".65")
        ax.plot(bins.mean_probability, bins.malignant_fraction, color=COLORS[name], lw=1.5)
        ax.scatter(bins.mean_probability, bins.malignant_fraction, s=15 + 2*bins.patients, color=COLORS[name], alpha=.8)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed malignant fraction", title=LABELS[name], xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.15)
    fig.suptitle("Held-out reliability by fixed bins  |  marker size reflects patient count", fontsize=12)
    fig.tight_layout()
    save_figure(fig, directory, FIGURES[2])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, name in zip(axes, MODEL_NAMES):
        row = context["metrics"].loc[context["metrics"].model.eq(name) & context["metrics"].condition.eq("clean")].iloc[0]
        matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=20,
                        color="white" if matrix[i, j] > matrix.max()*.55 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"],
               xlabel="Predicted diagnosis", ylabel="Actual diagnosis", title=LABELS[name])
    fig.suptitle("Held-out clean confusion matrices at probability threshold 0.5")
    fig.tight_layout()
    save_figure(fig, directory, FIGURES[3])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for index, name in enumerate(MODEL_NAMES):
        for pos, condition in enumerate(COHERENT_CONDITIONS):
            row = ci.loc[ci.comparison.eq("change_from_clean") & ci.model.eq(name) & ci.condition.eq(condition) & ci.metric.eq("balanced_accuracy")].iloc[0]
            y = pos + (-.12 if index == 0 else .12)
            ax.plot([100*row.ci_low, 100*row.ci_high], [y, y], color=COLORS[name], lw=2)
            ax.scatter(100*row.estimate, y, color=COLORS[name], s=35, label=LABELS[name] if pos == 0 else None, zorder=3)
    ax.axvline(0, color=".5", ls="--", lw=1)
    ax.set(yticks=range(4), yticklabels=[COND_LABELS[c] for c in COHERENT_CONDITIONS],
           xlabel="Perturbed minus clean balanced accuracy  (percentage points)", title="Held-out coherent boundary sensitivity")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    save_figure(fig, directory, FIGURES[4])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        for offset, key, label, color in ((-.18, "original", "Original method", "#89959E"), (.18, "coherent", "Coherent method", COLORS[name])):
            table = history[key].loc[history[key].model.eq(name)].set_index("condition")
            values = [100*float(table.loc[c, "delta_balanced_accuracy"]) for c in COHERENT_CONDITIONS]
            ax.bar(np.arange(4)+offset, values, .34, label=label, color=color)
        ax.axhline(0, color=".4", lw=.7)
        ax.set(xticks=range(4), xticklabels=["Mild\nerosion", "Mild\ndilation", "Moderate\nerosion", "Moderate\ndilation"], title=LABELS[name])
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("BA change  (percentage points)")
    fig.suptitle("Development sensitivity to perturbation construction  |  original clean models")
    fig.tight_layout()
    save_figure(fig, directory, FIGURES[5])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    for ax, name in zip(axes, MODEL_NAMES):
        table = history["selection"].loc[history["selection"].model.eq(name)].set_index("arm")
        for offset, arm, label, color in ((-.18, "all_23_matched_tuning", "All 23", "#89959E"), (.18, "stability_topk", "Stability subset", COLORS[name])):
            row = table.loc[arm]
            ax.plot(np.arange(2)+offset, [100*row.balanced_accuracy, 100*row.mean_five_condition_ba],
                    marker="o", linestyle="none", ms=8, label=label, color=color)
        ax.set(xticks=[0, 1], xticklabels=["Clean BA", "Five-condition BA"], title=LABELS[name], ylim=(75, 91))
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Balanced accuracy  (%)")
    fig.suptitle("Nested development evaluation of the frozen stability-selection procedure")
    fig.tight_layout()
    save_figure(fig, directory, FIGURES[6])


def interval_text(context, name, metric, condition="clean", percent=True):
    row = context["intervals"].loc[context["intervals"].comparison.eq("absolute") & context["intervals"].model.eq(name)
                                    & context["intervals"].condition.eq(condition) & context["intervals"].metric.eq(metric)].iloc[0]
    factor, decimals = (100, 2) if percent else (1, 4)
    return f"{factor*row.estimate:.{decimals}f} [{factor*row.ci_low:.{decimals}f}, {factor*row.ci_high:.{decimals}f}]"


def build_sections(context, history, overlap, examples, sizes):
    clean = context["metrics"].loc[context["metrics"].condition.eq("clean")].set_index("model")
    summary, primary, manifest = context["summary"], context["primary"], context["manifest"]
    lr, svm = clean.loc["logreg"], clean.loc["svm"]
    diff = context["intervals"].loc[context["intervals"].comparison.eq("svm_minus_logreg") & context["intervals"].condition.eq("clean")
                                     & context["intervals"].metric.eq("balanced_accuracy")].iloc[0]
    total = manifest.Case.nunique()
    cohorts = []
    for split, label in ((None, "All patients"), ("development", "Development"), ("test", "Held-out test")):
        part = manifest if split is None else manifest.loc[manifest.split.eq(split)]
        patients = part.groupby("Case").Pathology.first()
        cohorts.append([label, str(len(part)), str(len(patients)), str(int(patients.eq("benign").sum())), str(int(patients.eq("malignant").sum()))])
    selection_table = [[LABELS[r.model], "All 23" if r.arm == "all_23_matched_tuning" else "Stability subset",
                        f"{100*r.balanced_accuracy:.2f}", f"{100*r.mean_five_condition_ba:.2f}"] for r in history["selection"].itertuples()]
    clean_table = [[label, interval_text(context, "logreg", metric, percent=pct), interval_text(context, "svm", metric, percent=pct)]
                   for metric, label, pct in (("balanced_accuracy", "Balanced accuracy (%)", True), ("sensitivity", "Sensitivity (%)", True),
                                             ("specificity", "Specificity (%)", True), ("roc_auc", "ROC AUC", False),
                                             ("brier_score", "Brier score", False), ("log_loss", "Log loss", False))]
    pair_table = [["LR" if r.model == "logreg" else "SVM", COND_LABELS[r.condition], f"{100*r.delta_balanced_accuracy:+.2f}",
                   f"{100*r.flip_rate:.2f}", str(int(r.new_false_negatives)), str(int(r.recovered_false_negatives))]
                  for r in context["pair_metrics"].itertuples()]
    baseline_table = [[LABELS[n], f"{100*history['baseline'][n]['balanced_accuracy']:.2f}",
                       f"{history['baseline'][n]['roc_auc']:.4f}"] for n in MODEL_NAMES]
    calibration_table = [[LABELS[r.model], f"{100*r.balanced_accuracy:.2f}", f"{100*r.sensitivity:.2f}",
                          f"{100*r.specificity:.2f}", f"{r.brier_score:.4f}"] for r in history["calibration"].itertuples()]
    costs = [[r.stage, str(int(r.base_inner_fits)), str(int(r.base_refits)), str(int(r.calibrator_fits)),
              "Not recorded" if pd.isna(r.recorded_seconds) else f"{r.recorded_seconds:.1f}"] for r in history["costs"].itertuples()]
    def section(title, paragraphs=(), table=None, figures=()):
        return {"title": title, "paragraphs": list(paragraphs), "table": table, "figures": list(figures)}
    return [
        section("Study purpose and main findings", [
            f"We studied how lesion-boundary changes affect breast-ultrasound classification using Logistic Regression (LR) and RBF SVM with 23 handcrafted features. The study includes {total:,} patients and separates development from a locked test cohort. All model selection and calibration decisions were completed before test predictions.",
            f"On {summary['held_out_patients']} held-out patients, the primary SVM clean balanced accuracy was {100*primary.estimate:.2f}% (pointwise 95% interval {100*primary.ci_low:.2f}-{100*primary.ci_high:.2f}%). LR achieved {100*lr.balanced_accuracy:.2f}%. SVM had slightly higher sensitivity and specificity, while LR had slightly better AUC, Brier score and log loss. This is not a uniform superiority result.",
            "The original mild erosion/dilation losses depended strongly on the perturbation mechanism. A separately labelled coherent alternative greatly reduced those losses on development data. The frozen stability-selection procedure did not improve the overall accuracy-robustness objective. Final held-out changes were small and sometimes raised balanced accuracy while worsening ranking or probability losses.",
            f"At the fixed 0.5 research threshold, SVM missed {int(svm.fn)} of {overlap['malignant_patients']} malignant test patients. These results support a reproducible methodological study, not clinical deployment."],
            figures=[{"file": FIGURES[0], "caption": "Figure 1. Clean held-out operating-point metrics and AUC. Bars show pointwise 95% patient-bootstrap intervals; all scores use a 0-1 axis."}]),
        section("Dataset and evaluation design", [
            "BUS-BRA version 1.0 is the public dataset associated with Gomez-Flores, Gregorio-Calas and Pereira [1,2]. We classify benign versus malignant pathology using the provided lesion delineations. BI-RADS labels are not input features in this experiment.",
            "The random seed is 42. Patients, rather than images, define the development/test separation and five locked outer folds. The original classifiers use three inner patient-stratified folds for tuning. Each patient contributes total training weight one across their views, with optional training-only class balancing. Patient scores average image scores.",
            "Images are converted to grayscale and resized with preserved aspect ratio to longest side 256. Masks use nearest-neighbor interpolation. The 23 features comprise six shape, seven intensity and ten masked texture features. Masked GLCM pairs are counted only when both pixels lie inside the lesion, using distances one and three.",
            "Shape features are area fraction, aspect ratio, eccentricity, solidity, extent and circularity. Intensity features are mean, SD, median, 10th and 90th percentiles, skewness and entropy. Texture features are contrast, dissimilarity, homogeneity, angular second moment and correlation at each of the two GLCM distances. The existing extractor fixes intensity normalization, quantization and direction averaging; its source fingerprint is preserved.",
            "LR tunes C over 0.01, 0.1, 1, 10 and 100; RBF SVM tunes C over 0.1, 1 and 10 and gamma over scale, 0.01 and 0.1. Both consider class balancing off/on. The original inner objective is patient BA, with AUC and simpler settings as tie-breakers. Later robustness-oriented searches use the equal average of clean and four coherent BAs within each inner fold, then average over the three folds. The five-condition objective gives clean performance 20% of the total weight.",
            "Clean scaling and classifier fitting never use perturbed training masks. Coherent inner-validation conditions may score the explicitly robustness-oriented candidate objective in later experiments. Final calibration uses additional patient-disjoint base/calibration splits. The held-out test is used once with frozen models, and reporting reuses its saved predictions."],
            table={"headers": ["Cohort", "Images", "Patients", "Benign", "Malignant"], "rows": cohorts}),
        section("Original and coherent development robustness", [
            "The first experiment established clean nested patient-level baselines. The original boundary experiment then assessed eight mask conditions: mild/moderate erosion and dilation, plus four translations. Unavailable translated views were retained as unavailable; their patients were excluded only from the affected paired condition metrics.",
            "Unexpectedly large mild-condition losses motivated a development diagnostic. Pseudo-random selection among boundary pixels could create scattered contour irregularities and change perimeter-dependent features. An independent coherent alternative grows localized, connected boundary edits; mild and moderate edits are nested. Their IoU targets are approximately 0.95 and 0.90.",
            "The original results were preserved. Coherent masks were audited and their features extracted separately. Clean out-of-fold predictions reproduced for both classifiers before robustness scoring. The large original mild-condition BA losses shrank with coherent edits. This supports sensitivity to perturbation construction; it does not validate either mechanism as a realistic clinical annotation-error model.",
            "The development patient-bootstrap intervals were exploratory and conditional on saved OOF fits and observed fold/diagnosis counts. They omitted retraining variability and did not establish a uniform SVM advantage. CV uncertainty requires particular caution because training sets overlap [4]."],
            table={"headers": ["Original clean baseline", "BA (%)", "AUC"], "rows": baseline_table},
            figures=[{"file": FIGURES[5], "caption": "Figure 2. Original and coherent development BA changes for the same original clean-model procedure. This is a perturbation-construction comparison, not the final calibrated test model."}]),
        section("Frozen stability feature selection experiment", [
            "Feature stability was ranked separately inside every inner training partition. Each condition's absolute image-feature shifts were averaged within patient and then equally across patients, normalized by clean training weighted SD. The score was the largest of the four condition means. Constant or near-constant features were excluded from subset candidates.",
            "The frozen subset sizes were 5, 10 and 15. Both the subset arm and the all-23 comparator tuned the same model grids and the mean of clean plus four coherent patient BAs, averaged across three inner folds. Selected subsets were reranked on outer training patients before the outer refit. No global feature list or k was selected from outer results.",
            "Both classifiers selected 15 features in each outer fold. The subset procedure did not improve the pooled five-condition objective. Its lower clean baseline can make relative BA-change comparisons look favorable even when absolute perturbed accuracy remains lower. This negative result is retained rather than altering the grid after inspecting outer outcomes."],
            table={"headers": ["Model", "Arm", "Clean BA (%)", "Five-condition BA (%)"], "rows": selection_table},
            figures=[{"file": FIGURES[6], "caption": "Figure 3. Matched nested development comparison. These are point estimates for the complete selection procedures; the plot does not imply statistical significance."}]),
        section("Patient calibration and final protocol", [
            "The final procedure keeps all 23 features. SVM was designated primary from the Step 19 development five-condition objective; LR remains the comparator. This is a post-hoc development-based decision, not a significance claim. The sigmoid method, three-member ensemble and probability threshold 0.5 were fixed before calibration evaluation.",
            "Within each outer training pool, each member reserves one of three patient folds for calibration. Its base model is independently tuned using three inner patient folds within the other two folds, then fitted only on clean base images. Calibration receives one clean score per reserved patient: logit of mean LR image probability, or mean SVM margin. A fixed regularized sigmoid with equal patient weights is fitted to those scores. This keeps base tuning, base fitting and calibration patients disjoint [3].",
            "The patient calibrator uses StandardScaler followed by LogisticRegression with C=1, max_iter=2000 and random_state=42, without class or image-count weights. LR probabilities are clipped to [1e-6, 1-1e-6] before the logit. Calibration split seeds are 2020 plus scope code; base-tuning seeds are 3000 plus ten times scope code plus member index. Scope codes are 0-4 for outer evaluation and 5 for final fitting.",
            "For prediction, image scores are aggregated within each member, calibrated at patient level, then the three probabilities are averaged. The same procedure fits the final development-only ensembles. Outer-validation patients influence none of its tuning, scaling, fitting or calibration operations.",
            "Calibration quality is measured rather than assumed. At 0.5, outer development sensitivity fell and specificity rose relative to the earlier single-refit estimators. The ensemble has different member training sizes and settings, so cross-step differences cannot be attributed solely to the sigmoid. The threshold was not changed after observing this trade-off."],
            table={"headers": ["Calibrated outer CV", "BA (%)", "Sensitivity (%)", "Specificity (%)", "Brier"], "rows": calibration_table}),
        section("Locked held out primary performance", [
            f"The primary result is SVM clean BA {100*primary.estimate:.2f}% [{100*primary.ci_low:.2f}, {100*primary.ci_high:.2f}]. The test cohort contains {overlap['benign_patients']} benign and {overlap['malignant_patients']} malignant patients. Table intervals are pointwise percentiles from 10,000 shared patient resamples within diagnosis, seed 42, conditional on the frozen models.",
            f"The clean SVM-minus-LR BA difference is {100*diff.estimate:+.2f} percentage points [{100*diff.ci_low:.2f}, {100*diff.ci_high:.2f}]. Its interval includes zero at the boundary. LR has the higher clean AUC and lower probability losses. The evidence does not support a universal claim that SVM is better on every measure.",
            "The test is an internal patient hold-out from this dataset, not an independent external clinical cohort. Secondary intervals are not adjusted for multiple comparisons. No threshold, feature or model changes follow the test results."],
            table={"headers": ["Metric", "LR estimate [95% interval]", "SVM estimate [95% interval]"], "rows": clean_table},
            figures=[{"file": FIGURES[1], "caption": "Figure 4. Clean test ROC curves from saved probabilities. AUC measures ranking over thresholds and differs from BA at the fixed operating point."}]),
        section("Probability reliability and diagnostic errors", [
            f"Clean Brier scores were {lr.brier_score:.4f} for LR and {svm.brier_score:.4f} for SVM; log losses were {lr.log_loss:.4f} and {svm.log_loss:.4f}. Lower is better for these losses. Reliability bins summarize the same held-out probabilities, with sparse bins interpreted cautiously.",
            f"At 0.5, LR had {int(lr.fn)} false negatives and {int(lr.fp)} false positives. SVM had {int(svm.fn)} false negatives and {int(svm.fp)} false positives. SVM sensitivity was {100*svm.sensitivity:.2f}% and specificity {100*svm.specificity:.2f}%. AUC near 0.91 does not remove the missed-malignancy burden at this threshold."],
            figures=[{"file": FIGURES[2], "caption": "Figure 5. Ten fixed-width reliability bins. Marker size reflects patient count; connecting lines do not establish reliable behavior in sparsely populated regions."},
                     {"file": FIGURES[3], "caption": "Figure 6. Frozen-threshold confusion matrices. Counts are patients, not individual image views."}]),
        section("Held out coherent boundary sensitivity", [
            "Every test image had all four valid coherent feature vectors. All paired comparisons retain the same complete patients and image views. Positive signed BA change means a higher perturbed BA relative to that model's own clean baseline; it does not imply that segmentation errors are beneficial.",
            "Some edits raise BA by moving a small number of scores across 0.5. At the same time, ranking AUC or probability losses can worsen. In particular, dilation reduces AUC for both classifiers in this test cohort. Report sensitivity, specificity, flips and probability quality alongside BA rather than interpreting one increase as global robustness.",
            "New FN counts identify malignant patients correctly classified clean and missed after perturbation; recovered FN counts identify the reverse transition. Balanced accuracy can be unchanged despite individual errors moving between patients."],
            table={"headers": ["Model", "Condition", "Delta BA (pp)", "Flip (%)", "New FN", "Recovered FN"], "rows": pair_table},
            figures=[{"file": FIGURES[4], "caption": "Figure 7. Held-out perturbed-minus-clean BA changes with paired pointwise 95% intervals. Zero denotes no aggregate BA change, not zero individual prediction changes."}]),
        section("Patient level failure analysis", [
            f"There were {overlap['both_wrong']} patients misclassified by both models, {overlap['lr_only_wrong']} misclassified by LR only, and {overlap['svm_only_wrong']} misclassified by SVM only; {overlap['both_correct']} were correct for both. The shared clean false-negative count was {overlap['both_false_negative']}. These overlaps use the same saved patient predictions.",
            "The example cases below are selected deterministically: up to three SVM false negatives with the lowest malignant probability, and up to three false positives with the highest probability. This is a descriptive confidence-error audit, not a blinded clinical review. The full CSV preserves every patient, image ID and per-model error flag.",
            "No image-quality, lesion-subtype or scanner-specific cause is inferred from a probability alone. Such explanations require a separate qualified image review and appropriate metadata. Confidence in an incorrect prediction is a reliability concern, not proof of a particular biological mechanism.",
            "The report preserves all failures and does not remove hard cases, optimize a new threshold, or revise the final model based on these test examples."],
            table={"headers": ["Example category", "Dataset case", "Actual", "LR probability", "SVM probability"],
                   "rows": [[r.category, str(r.Case), "Malignant" if r.actual else "Benign", f"{r.lr_probability:.4f}", f"{r.svm_probability:.4f}"] for r in examples.itertuples()]}),
        section("Computational cost and reproducibility", [
            "Fit counts document the cost of the complete procedures, including nested searches. Step 19 also used ten original-baseline reproducibility fits, separate from its twenty outer refits. The subset arm has an additional k search, so matched hyperparameter grids do not imply equal search budgets.",
            "The final held-out evaluation performed zero model fits, zero calibrator fits and no retuning. The report likewise performs no model prediction or fitting: it recomputes descriptive metrics from saved predictions, verifies the completed evaluation marker and its published-file checksums, and inspects fitted model sizes without executing inference.",
            f"The final SVM ensemble contains {int(sizes.support_vectors.fillna(0).sum())} support vectors across three members; the LR ensemble contains {int(sizes.linear_coefficients.fillna(0).sum())} base linear coefficients. The compressed combined model artifact is {context['model_bytes']/1024/1024:.2f} MiB. These are structural/size counts, not a controlled inference-speed benchmark.",
            "Only recorded runtimes are reported. Calibration and test runtimes were not retained; hardware-normalized speed, memory and energy comparisons cannot be reconstructed. Full input/output hashes, code fingerprints and provenance are supplied separately. The saved primary protocol and one-time evaluation marker remain unchanged."],
            table={"headers": ["Stage", "Inner base fits", "Base refits", "Sigmoid fits", "Recorded seconds"], "rows": costs}),
        section("Limitations and conclusions", [
            "The data come from one public dataset. Patient separation prevents view leakage but does not establish scanner-, institution- or population-independent generalization. The study uses provided lesion masks; an end-to-end system with an automatic segmenter was not evaluated.",
            "Boundary edits are synthetic and deterministic. Their IoU targets describe mask overlap, not guaranteed clinical plausibility. The large original mild-condition effect was mechanism-dependent, motivating a clearly labelled post-hoc coherent sensitivity analysis. Development-informed choices and diagnostics are not retrospective preregistration.",
            "Stability selection was evaluated only for the frozen score, candidate sizes and objectives. Selecting 15 features in every fold does not justify expanding the grid after seeing these results. The negative finding does not prove that all stability-aware methods fail.",
            "Calibration does not establish clinical risk validity. The fixed 0.5 threshold was not selected for clinical sensitivity, costs or workflow. The observed missed malignant patients preclude treating this report as a deployment recommendation. Uncertainty intervals are pointwise and conditional on the fixed fits and observed diagnosis composition; they do not include retraining or external-distribution variability.",
            "The completed project supports three conclusions: perturbation construction materially affects measured boundary robustness; the frozen stability-subset procedure did not improve the overall development objective; and the frozen calibrated SVM attained the reported held-out BA with a substantial sensitivity limitation. LR retained small advantages on AUC and probability losses. These findings should be reported together, without claiming universal SVM superiority or changing the locked evaluation after inspection."]),
        section("References and project handoff", [
            "[1] Gomez-Flores W, Gregorio-Calas MJ, Pereira WCA. BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems. Medical Physics. 2024;51:3110-3123. DOI: 10.1002/mp.16812.",
            "[2] BUS-BRA version 1.0 dataset. Zenodo. https://zenodo.org/records/8231412. Dataset DOI: 10.5281/zenodo.8231412.",
            "[3] Scikit-learn documentation. Probability calibration. https://scikit-learn.org/stable/modules/calibration.html. Also: Nested versus non-nested cross-validation, https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html.",
            "[4] Bates S, Hastie T, Tibshirani R. Cross-validation: what does it estimate and how well does it do it? arXiv:2104.00673. https://arxiv.org/abs/2104.00673.",
            "Scientific evaluation is complete. No further planned experiment remains. The remaining handoff is author review, any course-specific formatting or submission requirements, and preservation of the source code, frozen model/protocol, evaluation marker, saved results and report checksums. Future work requires a separately documented evaluation plan; it must not reuse this test set for tuning."])
    ]


def report_markdown(document):
    lines = [f"# {document['title']}", "", document["authors"], "", f"Final research report | {document['date']}", ""]
    for section in document["sections"]:
        lines.extend([f"## {section['title']}", ""])
        for paragraph in section["paragraphs"]:
            lines.extend([paragraph, ""])
        if section["table"]:
            table = section["table"]
            lines.extend(["| " + " | ".join(table["headers"]) + " |", "| " + " | ".join(["---"]*len(table["headers"])) + " |"])
            lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in table["rows"])
            lines.append("")
        for figure in section["figures"]:
            lines.extend([f"![{figure['caption']}](figures/{figure['file']}.png)", "", figure["caption"], ""])
    return "\n".join(lines)


def report_html(document, output):
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             f"<title>{html.escape(document['title'])}</title><style>body{{font:16px/1.6 Georgia,serif;color:#17232d;max-width:1020px;margin:48px auto;padding:0 26px}}h1,h2{{font-family:Arial,sans-serif;color:#000}}h1{{font-size:32px;line-height:1.2}}h2{{margin-top:2.3em;font-size:23px}}p{{max-width:90ch}}table{{border-collapse:collapse;width:100%;font:14px/1.4 Arial,sans-serif;margin:22px 0}}th,td{{border:1px solid #d9d9d9;padding:9px 11px;text-align:left}}th{{background:#eaf0f4;color:#000}}tr:nth-child(even){{background:#f7f9fa}}figure{{margin:24px 0}}img{{width:100%;height:auto}}figcaption{{font:13px/1.5 Arial,sans-serif;color:#475461;margin-top:9px}}.byline{{font:14px Arial,sans-serif;color:#475461}}@media print{{body{{font-size:10pt;margin:0;max-width:none}}figure,table{{break-inside:avoid}}h2{{break-after:avoid}}@page{{margin:18mm}}}}</style>",
             f"<h1>{html.escape(document['title'])}</h1><p class='byline'>{html.escape(document['authors'])}<br>Final research report | {document['date']}</p>"]
    for section in document["sections"]:
        parts.append(f"<section><h2>{html.escape(section['title'])}</h2>")
        parts.extend(f"<p>{html.escape(p)}</p>" for p in section["paragraphs"])
        if section["table"]:
            t = section["table"]
            parts.append("<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in t["headers"]) + "</tr></thead><tbody>")
            parts.extend("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in t["rows"])
            parts.append("</tbody></table>")
        for figure in section["figures"]:
            encoded = base64.b64encode((output / "figures" / f"{figure['file']}.png").read_bytes()).decode()
            parts.append(f"<figure><img alt='{html.escape(figure['caption'], quote=True)}' src='data:image/png;base64,{encoded}'><figcaption>{html.escape(figure['caption'])}</figcaption></figure>")
        parts.append("</section>")
    return "\n".join(parts) + "</html>"


def run(project, output):
    if output.exists():
        raise FileExistsError("Report output exists; choose a new --output directory")
    paths = source_paths(project)
    input_files = [p for key, path in paths.items() for p in ([path] if path.is_file() else path.glob("*.json"))]
    input_files += [p for key in ("heldout", "step19", "step16", "step17", "original", "calibration") for p in paths[key].glob("*.csv")]
    input_files += [paths["calibration"] / "frozen_calibrated_models.joblib"]
    before = {str(p.resolve()): pc.sha256(p) for p in set(input_files)}
    context = validate_saved_evaluation(paths)
    history = load_history(paths)
    cases, overlap, examples = failure_analysis(context["predictions"], context["manifest"])
    sizes = model_sizes(paths["calibration"])
    context["model_bytes"] = (paths["calibration"] / "frozen_calibrated_models.joblib").stat().st_size
    document = {"title": "Breast ultrasound classification under lesion boundary uncertainty",
                "authors": "Mostafa Bdeir and Hadi Fardan", "date": "2026-11-15",
                "sections": build_sections(context, history, overlap, examples, sizes)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    make_figures(context, history, output / "figures")
    (output / "BUS_BRA_Final_Report.md").write_text(report_markdown(document), encoding="utf-8")
    (output / "BUS_BRA_Final_Report.html").write_text(report_html(document, output), encoding="utf-8")
    (output / "report_content.json").write_text(ss.canonical_json(document), encoding="utf-8")
    for filename, table in (("patient_failure_audit.csv", cases), ("illustrative_failure_cases.csv", examples),
                            ("computational_cost.csv", history["costs"]), ("final_model_sizes.csv", sizes),
                            ("heldout_clean_comparison.csv", context["metrics"].loc[context["metrics"].condition.eq("clean")]),
                            ("heldout_robustness_table.csv", context["pair_metrics"])):
        table.to_csv(output / filename, index=False, float_format="%.12g")
    if any(pc.sha256(Path(p)) != digest for p, digest in before.items()):
        raise RuntimeError("An existing study result changed during reporting")
    summary = {"step": 22, "status": "final_scientific_report_generated", "model_fits": 0, "new_predictions": 0,
               "heldout_evaluation_rerun": False, "prior_files_unchanged": True,
               "planned_experiments_remaining": 0, "figure_count": len(FIGURES), "failure_overlap": overlap,
               "primary_result": context["summary"]["primary_result"], "inputs_sha256": before,
               "source_sha256": pc.sha256(Path(__file__)),
               "outputs_sha256": {str(p.relative_to(output)): pc.sha256(p) for p in output.rglob("*") if p.is_file()},
               "remaining_handoff": "author review, course-specific formatting/submission and preservation; no further model selection"}
    (output / "report_manifest.json").write_text(ss.canonical_json(summary), encoding="utf-8")
    print(f"STEP 22 — FINAL SCIENTIFIC REPORT\nVerified complete held-out checkpoint; prior files unchanged\n"
          f"Figures: {len(FIGURES)} | Model fits: 0 | New predictions: 0\nPlanned experiments remaining: 0\nSaved: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final_step22")
    args = parser.parse_args()
    run(args.project, args.output)


if __name__ == "__main__":
    main()
