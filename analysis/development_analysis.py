"""Optional development analyses for the BUS-BRA submission."""

import argparse
import io
import platform
import statistics
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.features import extract_features, preprocess_pair, read_pair
from src.models import fit_model, load_development, patient_predictions
from src.svm_model import fit_svm, patient_svm_predictions


def timing(operation, repeats):
    operation()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        samples.append((perf_counter() - start) * 1000)
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return statistics.median(samples), quartiles[0], quartiles[2]


def selected_settings(path, model, fold):
    row = pd.read_csv(path).set_index("outer_fold").loc[fold]
    settings = {"C": float(row.C), "balance": str(row.balance).lower() == "true"}
    if model == "svm":
        settings["gamma"] = row.gamma if row.gamma == "scale" else float(row.gamma)
    return settings


def benchmark(root, development, repeats, fold):
    training = development.loc[development.dev_fold.ne(fold)]
    validation = development.loc[development.dev_fold.eq(fold)]
    assert set(training.Case).isdisjoint(validation.Case)
    samples = [rows for _, rows in validation.groupby("Case")][:20]
    records = []
    for name, fit, predict in (("logreg", fit_model, patient_predictions),
                               ("svm", fit_svm, patient_svm_predictions)):
        source = root / f"results/modeling/{name}_nested_cv_folds.csv"
        settings = selected_settings(source, name, fold)
        model = fit(training, **settings)
        blob = io.BytesIO()
        joblib.dump(model, blob, compress=3)
        for stage, operation in (
            ("fit including scaling, excluding tuning", lambda: fit(training, **settings)),
            ("serial patient prediction including aggregation",
             lambda: [predict(model, patient) for patient in samples]),
        ):
            divisor = len(samples) if stage.startswith("serial") else 1
            median, q1, q3 = timing(operation, repeats)
            records.append({"model": name, "stage": stage, "median_ms": median / divisor,
                            "q1_ms": q1 / divisor, "q3_ms": q3 / divisor,
                            "fit_images": len(training), "fit_patients": training.Case.nunique(),
                            "timed_patients": len(samples), "repeats": repeats,
                            "compressed_model_bytes": len(blob.getvalue())})
        if name == "logreg":
            from src.features import FEATURE_COLUMNS
            coefficients = pd.Series(model.named_steps["classifier"].coef_.ravel(),
                                     index=FEATURE_COLUMNS, name="standardized_coefficient")
            print("\nLR coefficients for this development fold and its selected settings")
            print(coefficients.loc[coefficients.abs().nlargest(8).index].to_string())
    archive = root / "data/raw/BUSBRA.zip"
    if archive.is_file():
        ids = validation.sort_values("ID").ID.head(20).tolist()
        with ZipFile(archive) as dataset:
            pairs = [read_pair(dataset, image_id) for image_id in ids]
        def extract_batch():
            for image, mask in pairs:
                extract_features(*preprocess_pair(image, mask))
        median, q1, q3 = timing(extract_batch, repeats)
        print("\nPreprocess and extract, excluding ZIP reads and image decoding (ms/image)")
        print({"median": median / len(ids), "q1": q1 / len(ids), "q3": q3 / len(ids)})
    return pd.DataFrame(records)


def failure_breakdown(development, predictions):
    metadata = development.groupby("Case").agg(
        area=("shape_area_fraction", "mean"), views=("ID", "size"),
        pathology=("Pathology", "first"))
    metadata["area_tertile"] = pd.qcut(metadata.area, 3, labels=["small", "middle", "large"])
    frame = predictions.merge(metadata.reset_index(), on="Case", validate="many_to_one")
    if len(frame) != len(predictions) or not (frame.actual == frame.pathology.eq("malignant")).all():
        raise ValueError("OOF patient coverage or diagnoses disagree with development features")
    if frame.duplicated(["model", "Case"]).any():
        raise ValueError("Expected one OOF prediction per patient and model")
    frame["error"] = frame.actual.ne(frame.predicted)
    for group in ("area_tertile", "views"):
        print(f"\nDevelopment OOF errors by {group} and pathology")
        print(frame.groupby(["model", "pathology", group], observed=True).error
              .agg(patients="size", errors="sum", error_rate="mean").to_string())
    fold_metrics = [{"model": model, "fold": fold,
                     "BA": balanced_accuracy_score(rows.actual, rows.predicted)}
                    for (model, fold), rows in frame.groupby(["model", "outer_fold"])]
    print("\nClean OOF balanced accuracy across outer folds")
    print(pd.DataFrame(fold_metrics).groupby("model").BA.agg(["mean", "std", "count"]).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error("--repeats must be at least 10")
    root = args.root.resolve()
    development = load_development(root / "data/processed/development_features.csv",
                                   root / "splits/image_splits.csv")
    print(f"{platform.platform()} | CPU: {platform.processor()}")
    print(benchmark(root, development, args.repeats, args.fold).to_string(index=False))
    path = root / "results/robustness/calibration_step20/calibrated_clean_oof.csv"
    if path.is_file():
        failure_breakdown(development, pd.read_csv(path))


if __name__ == "__main__":
    main()
