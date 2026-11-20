"""Build the development-only report tables from saved rebuild outputs.

Run from the project root of a development rebuild (after src.patient_calibration):
    python analysis/build_report_tables.py --root . --out reports/submission_v2/tables

Reads only development outputs. It never touches held-out files. Writes CSV tables plus a JSON of
the numbers quoted in the report, so every figure in the text has one source.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def load_dev_script(root: Path):
    path = Path(__file__).with_name("development_analysis.py")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("development_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "reports/submission_v2/tables").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rb = root / "results/robustness"
    nums = {}
    names = {"logreg": "LR", "svm": "SVM"}
    cond_names = {"erosion_mild": "Mild erosion", "dilation_mild": "Mild dilation",
                  "erosion_moderate": "Moderate erosion", "dilation_moderate": "Moderate dilation",
                  "shift_left_moderate": "Shift left", "shift_right_moderate": "Shift right",
                  "shift_up_moderate": "Shift up", "shift_down_moderate": "Shift down"}
    order = list(cond_names)

    # 1. Original eight-condition development stress test
    o = pd.read_csv(rb / "robustness_paired_condition_summary.csv")
    o["order"] = o.condition.map(order.index)
    o = o.sort_values(["model", "order"])
    t = pd.DataFrame({
        "Model": o.model.map(names), "Condition": o.condition.map(cond_names),
        "Patients": o.eligible_patients,
        "Delta BA (pp)": (o.delta_balanced_accuracy * 100).map("{:+.2f}".format),
        "Flip (%)": (o.flip_rate * 100).map("{:.1f}".format),
        "New FN": o.new_false_negatives, "Recovered FN": o.recovered_false_negatives})
    t.to_csv(out / "table_original_eight_conditions.csv", index=False)
    (out / "table_original_eight_conditions.md").write_text(md(t), encoding="utf-8")
    nums["original"] = {f"{r.model}|{r.condition}": {"delta_pp": round(r.delta_balanced_accuracy * 100, 2),
                                                      "flip_pct": round(r.flip_rate * 100, 1)}
                        for r in o.itertuples()}

    # 2. Coherent development comparison
    c = pd.read_csv(rb / "coherent_step16/coherent_condition_summary.csv")
    c["order"] = c.condition.map(order.index)
    c = c.sort_values(["model", "order"])
    tc = pd.DataFrame({
        "Model": c.model.map(names), "Condition": c.condition.map(cond_names),
        "Delta BA (pp)": (c.delta_balanced_accuracy * 100).map("{:+.2f}".format),
        "Flip (%)": (c.flip_rate * 100).map("{:.1f}".format),
        "New FN": c.new_false_negatives, "Recovered FN": c.recovered_false_negatives})
    tc.to_csv(out / "table_coherent_development.csv", index=False)
    (out / "table_coherent_development.md").write_text(md(tc), encoding="utf-8")
    nums["coherent"] = {f"{r.model}|{r.condition}": round(r.delta_balanced_accuracy * 100, 2) for r in c.itertuples()}

    # 3. Stability subset selection frequencies (union of the two classifiers' identical rankings)
    f = pd.read_csv(rb / "selection_step19/feature_selection_frequency.csv")
    s = f[f.arm == "stability_topk"].pivot(index="feature", columns="model", values="selected_outer_folds")
    s = s.sort_values(["logreg", "svm"], ascending=False).reset_index()
    ts = s.rename(columns={"feature": "Feature", "logreg": "LR folds selected (of 5)", "svm": "SVM folds selected (of 5)"})
    ts.to_csv(out / "table_stability_selection_frequency.csv", index=False)
    (out / "table_stability_selection_frequency.md").write_text(md(ts), encoding="utf-8")
    nums["never_selected"] = s.loc[(s.logreg == 0) & (s.svm == 0), "feature"].tolist()

    # 4. Chosen hyperparameters of the original baselines (per outer fold)
    rows = []
    for m in ("logreg", "svm"):
        d = pd.read_csv(root / f"results/modeling/{m}_nested_cv_folds.csv")
        for r in d.itertuples():
            rows.append({"Model": names[m], "Outer fold": r.outer_fold, "C": f"{r.C:g}",
                         "gamma": getattr(r, "gamma", "-") if m == "svm" else "-",
                         "Class balancing": "on" if r.balance else "off",
                         "Outer-fold BA (%)": f"{r.balanced_accuracy * 100:.1f}"})
    th = pd.DataFrame(rows)
    th.to_csv(out / "table_chosen_hyperparameters.csv", index=False)
    (out / "table_chosen_hyperparameters.md").write_text(md(th), encoding="utf-8")

    # 5. Fold variation and failure breakdown on calibrated development out-of-fold predictions
    dev_mod = load_dev_script(root)
    development = dev_mod.load_development(root / "data/processed/development_features.csv",
                                           root / "splits/image_splits.csv")
    oof = pd.read_csv(rb / "calibration_step20/calibrated_clean_oof.csv")
    from sklearn.metrics import balanced_accuracy_score
    fold = pd.DataFrame([{"model": m, "fold": k, "BA": balanced_accuracy_score(g.actual, g.predicted)}
                         for (m, k), g in oof.groupby(["model", "outer_fold"])])
    fv = fold.groupby("model").BA.agg(["mean", "std", "min", "max"]).reset_index()
    tf = pd.DataFrame({"Model": fv.model.map(names), "Mean BA (%)": (fv["mean"] * 100).map("{:.2f}".format),
                       "SD across folds (pp)": (fv["std"] * 100).map("{:.2f}".format),
                       "Min (%)": (fv["min"] * 100).map("{:.1f}".format),
                       "Max (%)": (fv["max"] * 100).map("{:.1f}".format)})
    tf.to_csv(out / "table_fold_variation.csv", index=False)
    (out / "table_fold_variation.md").write_text(md(tf), encoding="utf-8")
    nums["fold"] = {r.model: {"mean": round(r.mean * 100, 2), "sd": round(r.std * 100, 2)} for r in fv.itertuples()}

    meta = development.groupby("Case").agg(area=("shape_area_fraction", "mean"), views=("ID", "size"),
                                           pathology=("Pathology", "first"))
    meta["Lesion size tertile"] = pd.qcut(meta.area, 3, labels=["small", "middle", "large"])
    frame = oof.merge(meta.reset_index(), on="Case", validate="many_to_one")
    assert len(frame) == len(oof) and (frame.actual == frame.pathology.eq("malignant")).all()
    frame["error"] = frame.actual.ne(frame.predicted)
    parts = []
    for group, label in (("Lesion size tertile", "Lesion size"), ("views", "Views per patient")):
        g = frame.groupby(["model", "pathology", group], observed=True).error.agg(["size", "sum", "mean"]).reset_index()
        g.columns = ["model", "pathology", "level", "patients", "errors", "rate"]
        g["Factor"] = label
        parts.append(g)
    fb = pd.concat(parts)
    tfb = pd.DataFrame({"Model": fb.model.map(names), "Class": fb.pathology.str.capitalize(),
                        "Factor": fb.Factor, "Group": fb.level.astype(str), "Patients": fb.patients,
                        "Errors": fb.errors, "Error rate (%)": (fb.rate * 100).map("{:.1f}".format)})
    tfb = tfb.sort_values(["Factor", "Model", "Class", "Group"], ascending=[True, True, True, True])
    tfb.to_csv(out / "table_failure_breakdown.csv", index=False)
    (out / "table_failure_breakdown.md").write_text(md(tfb), encoding="utf-8")

    # 6. Complexity benchmark (development fold 0, selected settings, no tuning)
    bench = dev_mod.benchmark(root, development, args.repeats, 0)
    bench.to_csv(out / "table_complexity_raw.csv", index=False)
    rows = []
    for m in ("logreg", "svm"):
        b = bench[bench.model == m]
        fit = b[b.stage.str.startswith("fit")].iloc[0]
        pred = b[b.stage.str.startswith("serial")].iloc[0]
        rows.append({"Model": names[m], "Fit, no tuning (ms, median [IQR])": f"{fit.median_ms:.1f} [{fit.q1_ms:.1f}-{fit.q3_ms:.1f}]",
                     "Predict one patient (ms)": f"{pred.median_ms:.2f} [{pred.q1_ms:.2f}-{pred.q3_ms:.2f}]",
                     "Compressed model (KB)": f"{fit.compressed_model_bytes / 1024:.1f}",
                     "Tuning candidates": 10 if m == "logreg" else 18})
    tcx = pd.DataFrame(rows)
    tcx.to_csv(out / "table_complexity.csv", index=False)
    (out / "table_complexity.md").write_text(md(tcx), encoding="utf-8")
    nums["complexity"] = {r["Model"]: r for r in rows}
    nums["benchmark_setup"] = {"fit_images": int(bench.fit_images.iloc[0]), "fit_patients": int(bench.fit_patients.iloc[0]),
                               "repeats": int(bench.repeats.iloc[0])}

    # 7. Feature extraction time per image (development, excludes ZIP reads)
    from time import perf_counter
    from zipfile import ZipFile
    from src.features import extract_features, preprocess_pair, read_pair
    ids = development.sort_values("ID").ID.head(40).tolist()
    with ZipFile(root / "data/raw/BUSBRA.zip") as z:
        pairs = [read_pair(z, i) for i in ids]
    samples = []
    for _ in range(args.repeats + 1):
        t0 = perf_counter()
        for im, mk in pairs:
            extract_features(*preprocess_pair(im, mk))
        samples.append((perf_counter() - t0) / len(pairs) * 1000)
    nums["extract_ms_per_image"] = round(float(np.median(samples[1:])), 2)

    (out / "report_numbers.json").write_text(json.dumps(nums, indent=2, default=str), encoding="utf-8")
    print("tables written to", out)


if __name__ == "__main__":
    main()
