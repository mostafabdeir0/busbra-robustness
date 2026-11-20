"""Exploratory, development-only operating-threshold analysis (mitigation for missed cancers).

    python analysis/threshold_analysis.py --oof reports/submission_v2/tables/calibrated_clean_oof.csv \
        --tables reports/submission_v2/tables --figures reports/submission_v2/figures

Uses ONLY the calibrated out-of-fold development predictions (851 patients per model).
The held-out test set is not read: its threshold (0.5) was frozen before it was scored,
and choosing a new one after seeing test errors would contaminate that evaluation.

Outputs
  table_threshold_fixed.{csv,md}   sensitivity/specificity at fixed thresholds
  table_threshold_policy.{csv,md}  cross-fitted "target sensitivity" policy (threshold chosen on the
                                   other four outer folds, applied to the fifth)
  threshold_numbers.json           numbers quoted in the report
  08_threshold_curves.{png,svg}    sensitivity and specificity versus threshold
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {"logreg": "Logistic Regression", "svm": "RBF SVM"}
SHORT = {"logreg": "LR", "svm": "SVM"}
FIXED = (0.5, 0.4, 0.3, 0.2)
TARGETS = (0.85, 0.90, 0.95)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def rates(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum()); fp = int((pred & (y == 0)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def threshold_for_target(y, p, target):
    """Largest threshold whose sensitivity is at least `target` (maximises specificity under the constraint)."""
    pos = np.sort(p[y == 1])[::-1]
    k = math.ceil(target * len(pos))
    return float(pos[k - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    a = ap.parse_args()
    oof = pd.read_csv(a.oof)
    required = {"model", "Case", "actual", "prob_malignant", "outer_fold"}
    if not required.issubset(oof.columns) or oof.duplicated(["model", "Case"]).any():
        raise SystemExit("Expected one calibrated out-of-fold prediction per patient and model")
    nums = {"patients": int(oof.Case.nunique()), "malignant": int(oof.loc[oof.model.eq("svm"), "actual"].sum())}

    fixed_rows, policy_rows = [], []
    for m in ("logreg", "svm"):
        g = oof[oof.model.eq(m)]
        y, p = g.actual.to_numpy(int), g.prob_malignant.to_numpy(float)
        for t in FIXED:
            r = rates(y, p, t)
            fixed_rows.append({"Model": SHORT[m], "Threshold": f"{t:.1f}", "Sensitivity (%)": f"{100*r['sensitivity']:.1f}",
                               "Specificity (%)": f"{100*r['specificity']:.1f}", "BA (%)": f"{100*r['balanced_accuracy']:.1f}",
                               "Missed cancers (FN)": r["fn"], "False alarms (FP)": r["fp"]})
            nums[f"{m}|fixed|{t}"] = {k: (round(100 * v, 2) if isinstance(v, float) else v) for k, v in r.items()}
        for target in TARGETS:
            preds, chosen = [], []
            for k in sorted(g.outer_fold.unique()):
                rest, held = g[g.outer_fold.ne(k)], g[g.outer_fold.eq(k)]
                t = threshold_for_target(rest.actual.to_numpy(int), rest.prob_malignant.to_numpy(float), target)
                chosen.append(t)
                preds.append(pd.DataFrame({"y": held.actual.to_numpy(int), "pred": held.prob_malignant.to_numpy(float) >= t}))
            P = pd.concat(preds)
            r = rates(P.y.to_numpy(), P.pred.to_numpy().astype(float), 0.5)
            full_t = threshold_for_target(y, p, target)
            policy_rows.append({"Model": SHORT[m], "Target sensitivity": f"{int(target*100)}%",
                                "Thresholds chosen per fold": f"{min(chosen):.3f}-{max(chosen):.3f}",
                                "Sensitivity (%)": f"{100*r['sensitivity']:.1f}", "Specificity (%)": f"{100*r['specificity']:.1f}",
                                "BA (%)": f"{100*r['balanced_accuracy']:.1f}", "Missed cancers (FN)": r["fn"],
                                "False alarms (FP)": r["fp"], "Threshold from all development": f"{full_t:.3f}"})
            nums[f"{m}|policy|{target}"] = {**{k: (round(100 * v, 2) if isinstance(v, float) else v) for k, v in r.items()},
                                           "fold_thresholds": [round(c, 4) for c in chosen], "full_development_threshold": round(full_t, 4)}

    a.tables.mkdir(parents=True, exist_ok=True)
    tf, tp_ = pd.DataFrame(fixed_rows), pd.DataFrame(policy_rows)
    tf.to_csv(a.tables / "table_threshold_fixed.csv", index=False)
    (a.tables / "table_threshold_fixed.md").write_text(md(tf), encoding="utf-8")
    tp_.to_csv(a.tables / "table_threshold_policy.csv", index=False)
    (a.tables / "table_threshold_policy.md").write_text(md(tp_), encoding="utf-8")
    (a.tables / "threshold_numbers.json").write_text(json.dumps(nums, indent=2), encoding="utf-8")

    # Figure: sensitivity and specificity against threshold, per model
    grid = np.linspace(0.01, 0.99, 197)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, ("logreg", "svm")):
        g = oof[oof.model.eq(m)]
        y, p = g.actual.to_numpy(int), g.prob_malignant.to_numpy(float)
        sens = [rates(y, p, t)["sensitivity"] for t in grid]
        spec = [rates(y, p, t)["specificity"] for t in grid]
        ax.plot(grid, sens, color="#E4572E", lw=2, label="Sensitivity")
        ax.plot(grid, spec, color="#1F6F78", lw=2, ls="--", label="Specificity")
        t90 = nums[f"{m}|policy|0.9"]["full_development_threshold"]
        ax.axvline(0.5, color="#5B6B70", lw=1, ls=":")
        ax.axvline(t90, color="#12343B", lw=1)
        ax.text(0.505, 0.03, "fixed 0.5", fontsize=8, color="#5B6B70")
        ax.text(t90 + 0.005, 0.10, f"90% target\n{t90:.2f}", fontsize=8, color="#12343B")
        ax.set_title(NAMES[m], fontsize=11)
        ax.set_xlabel("Probability threshold (patient is called malignant at or above it)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Rate (development, calibrated out-of-fold)")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False, bbox_to_anchor=(0.52, 0.12))
    fig.suptitle("Development-only operating-threshold analysis (not applied to the test set)", fontsize=11)
    fig.tight_layout()
    a.figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figures / "08_threshold_curves.png", dpi=150)
    fig.savefig(a.figures / "08_threshold_curves.svg")
    print(tf.to_string(index=False)); print(); print(tp_.to_string(index=False))


if __name__ == "__main__":
    main()
