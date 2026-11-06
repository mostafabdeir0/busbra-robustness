"""Audit BUS-BRA development features without fitting a model.

Place in src/feature_audit.py and run from the repository root:
    .\\.venv\\Scripts\\python.exe -m src.feature_audit

Read-only for source data. All outputs go to results/feature_audit/.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "development_features.csv"
MANIFEST = ROOT / "splits" / "image_splits.csv"
OUT = ROOT / "results" / "feature_audit"
ID_COLS = ["ID", "Case", "Pathology", "split", "dev_fold"]


def fail(message: str) -> None:
    raise AssertionError(message)


def validate_bounds(f: pd.DataFrame) -> list[str]:
    errors = []
    checks = {
        "shape_area_fraction": (0, 1),
        "shape_eccentricity": (0, 1),
        "shape_solidity": (0, 1),
        "shape_extent": (0, 1),
        "int_mean": (0, 1),
        "int_std": (0, 1),
        "int_median": (0, 1),
        "int_p10": (0, 1),
        "int_p90": (0, 1),
        "glcm_homogeneity_d1": (0, 1),
        "glcm_homogeneity_d3": (0, 1),
        "glcm_asm_d1": (0, 1),
        "glcm_asm_d3": (0, 1),
        "glcm_correlation_d1": (-1, 1),
        "glcm_correlation_d3": (-1, 1),
    }
    for name, (lo, hi) in checks.items():
        if name not in f.columns:
            errors.append(f"Missing required bounded feature: {name}")
        elif not f[name].between(lo - 1e-8, hi + 1e-8).all():
            errors.append(f"Invalid range: {name} outside [{lo}, {hi}]")
    if "shape_axis_ratio" in f and (f["shape_axis_ratio"] < 1 - 1e-8).any():
        errors.append("shape_axis_ratio < 1")
    for name in f.columns:
        if name.startswith(("glcm_contrast_", "glcm_dissimilarity_")) or name == "int_entropy":
            if (f[name] < -1e-8).any():
                errors.append(f"Negative value in {name}")
    if {"int_p10", "int_median", "int_p90"}.issubset(f.columns):
        if (f["int_p10"] > f["int_median"] + 1e-8).any() or (f["int_median"] > f["int_p90"] + 1e-8).any():
            errors.append("Intensity percentile order is inconsistent")
    return errors


def main() -> None:
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing features: {FEATURES}")
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    d = pd.read_csv(FEATURES, dtype={"ID": str})
    m = pd.read_csv(MANIFEST, dtype={"ID": str})
    if d.columns.duplicated().any():
        fail("Feature table has duplicated column names")
    if not set(ID_COLS).issubset(d.columns):
        fail(f"Missing ID/split columns: {set(ID_COLS) - set(d.columns)}")
    feature_names = [c for c in d.columns if c not in ID_COLS]
    if len(feature_names) != 23:
        fail(f"Expected exactly 23 features, found {len(feature_names)}")
    if not all(pd.api.types.is_numeric_dtype(d[c]) for c in feature_names):
        fail("One or more features have nonnumeric dtype")
    if d.ID.duplicated().any() or m.ID.duplicated().any():
        fail("Duplicate IDs in features or split manifest")
    dev = m.loc[m["split"].eq("development"), ID_COLS]
    if set(d.ID) != set(dev.ID):
        missing = sorted(set(dev.ID) - set(d.ID))
        extra = sorted(set(d.ID) - set(dev.ID))
        fail(f"Image coverage differs from locked development split; missing={missing[:5]}, extra={extra[:5]}")
    merged = d[ID_COLS].merge(dev, on="ID", how="inner", suffixes=("_features", "_manifest"), validate="one_to_one")
    for field in ("Case", "Pathology", "split", "dev_fold"):
        if not (merged[f"{field}_features"].astype(str) == merged[f"{field}_manifest"].astype(str)).all():
            fail(f"Feature table differs from locked split manifest in {field}")
    if not d["split"].eq("development").all() or d.Case.nunique() != 851 or len(d) != 1514:
        fail("Unexpected development partition counts")
    for col in ("Pathology", "dev_fold"):
        if d.groupby("Case")[col].nunique().max() != 1:
            fail(f"Inconsistent {col} across images of a patient")
    if set(d["dev_fold"].unique()) != set(range(5)):
        fail("Expected folds 0–4")
    x = d[feature_names]
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        fail("Missing or nonfinite features")
    errors = validate_bounds(x)
    if errors:
        fail("Feature range checks failed:\n" + "\n".join(errors))
    zero_var = x.columns[x.nunique(dropna=False) <= 1].tolist()
    if zero_var:
        fail(f"Constant features found: {zero_var}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary = x.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary.to_csv(OUT / "feature_distribution_summary.csv", float_format="%.8g")
    corr = x.corr(method="pearson").abs()
    pairs = []
    for i, a in enumerate(feature_names):
        for b in feature_names[i+1:]:
            val = float(corr.loc[a, b])
            if np.isfinite(val) and val >= 0.95:
                pairs.append({"feature_a": a, "feature_b": b, "abs_pearson_r": val})
    cp = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "abs_pearson_r"])
    cp = cp.sort_values("abs_pearson_r", ascending=False)
    cp.to_csv(OUT / "high_correlations.csv", index=False, float_format="%.6g")

    # Flag potential extremes for human review, but do NOT drop them.
    med = x.median()
    mad = (x - med).abs().median()
    flagged = []
    for name in feature_names:
        if mad[name] <= 1e-12:
            continue
        score = 0.6745 * (x[name] - med[name]).abs() / mad[name]
        for i in np.where(score.to_numpy() > 6)[0]:
            flagged.append({"ID": d.iloc[i]["ID"], "Case": d.iloc[i]["Case"],
                            "feature": name, "value": float(x.iloc[i][name]),
                            "robust_z": float(score.iloc[i])})
    flags = pd.DataFrame(flagged, columns=["ID", "Case", "feature", "value", "robust_z"])
    flags = flags.sort_values("robust_z", ascending=False)
    flags.to_csv(OUT / "potential_outliers.csv", index=False, float_format="%.6g")

    report = {
        "images": int(len(d)), "patients": int(d.Case.nunique()),
        "features": len(feature_names), "test_images_seen": 0,
        "missing_values": int(x.isna().sum().sum()),
        "duplicate_ids": int(d.ID.duplicated().sum()),
        "duplicate_feature_vectors": int(x.duplicated().sum()),
        "constant_features": zero_var,
        "high_correlation_pairs_abs_r_at_least_0_95": len(cp),
        "potential_outlier_values_robust_z_over_6": len(flags),
        "class_patients": {str(k): int(v) for k, v in d.groupby("Pathology").Case.nunique().items()},
        "fold_validation_patients": {str(k): int(v) for k, v in d.groupby("dev_fold").Case.nunique().items()},
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBUS-BRA FEATURE QUALITY AUDIT")
    print("=" * 45)
    print(f"Images: {report['images']} | Patients: {report['patients']} | Features: {report['features']}")
    print("Locked patient/image split: MATCHED")
    print("Feature ranges and quantile ordering: PASSED")
    print(f"Missing: {report['missing_values']} | Constant features: {len(zero_var)} | Duplicate feature vectors: {report['duplicate_feature_vectors']}")
    print(f"Highly correlated feature pairs (|r| >= 0.95): {len(cp)}")
    if len(cp):
        print(cp.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Potential extreme feature values for review: {len(flags)} (NOT removed)")
    print(f"Reports saved in: {OUT}")
    print("SUCCESS: Structural feature validation passed; review correlations and outliers next.")


if __name__ == "__main__":
    main()
