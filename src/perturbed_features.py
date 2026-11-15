"""Step 13: extract lesion-mask-perturbed BUS-BRA development features.

Commands from project root:
  python -m src.perturbed_features --limit 12
  python -m src.perturbed_features --all-development

Uses the EXACT preprocessed grayscale image for all masks of each image.
Preserves all eight intended rows per development image, including genuinely
unavailable translations (with valid=False and missing numerical features).
Verifies geometry against the frozen Step 12 audit; test split stays untouched.
No classifier is fit or evaluated here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, PROJECT_ROOT, extract_features, preprocess_pair, read_pair
from .perturbations import make_variants

INFO_COLUMNS = (
    "ID", "Case", "Pathology", "split", "dev_fold", "condition",
    "valid", "failure_reason", "actual_iou", "changed_pixels", "area_ratio", "amount_pixels"
)
VARIANT_NAMES = (
    "erosion_mild", "dilation_mild", "erosion_moderate", "dilation_moderate",
    "shift_left_moderate", "shift_right_moderate",
    "shift_up_moderate", "shift_down_moderate"
)


def image_rows(image_id: str, case: int, pathology: str, dev_fold: int,
               image: np.ndarray, original_mask: np.ndarray) -> list[dict]:
    """Generate eight rows without changing the image or original mask."""
    if image.shape != original_mask.shape:
        raise ValueError("Image/mask dimensions differ")
    base_image = image.copy()
    base_mask = original_mask.copy()
    variants = make_variants(original_mask)
    if tuple(variants) != VARIANT_NAMES:
        raise AssertionError("Perturbation condition order/schema changed")
    rows = []
    for name, var in variants.items():
        row = {
            "ID": str(image_id), "Case": int(case), "Pathology": str(pathology),
            "split": "development", "dev_fold": int(dev_fold), "condition": name,
            "valid": var is not None, "failure_reason": "" if var is not None else "translation_would_clip_lesion",
            "actual_iou": float(var.iou) if var is not None else np.nan,
            "changed_pixels": int(var.changed_pixels) if var is not None else np.nan,
            "area_ratio": float(var.area_ratio) if var is not None else np.nan,
            "amount_pixels": int(var.amount_pixels) if var is not None else np.nan,
        }
        if var is None:
            row.update({f: np.nan for f in FEATURE_COLUMNS})
        else:
            if np.array_equal(var.mask, original_mask) or not np.array_equal(image, base_image):
                raise AssertionError("Mask did not change or ultrasound image was altered")
            try:
                row.update(extract_features(image, var.mask))
            except (ValueError, IndexError, ArithmeticError) as exc:
                raise RuntimeError(f"Feature extraction failed for {image_id} / {name}") from exc
        rows.append(row)
    if not np.array_equal(image, base_image) or not np.array_equal(original_mask, base_mask):
        raise AssertionError("Function mutated original ultrasound image or mask")
    return rows


def check_geometry_match(generated: pd.DataFrame, audited: pd.DataFrame) -> None:
    """Fail closed when recomputed eligibility/IoU diverges from Step 12 audit."""
    keys = ["ID", "condition"]
    columns = ["ID", "condition", "valid", "actual_iou", "changed_pixels", "area_ratio", "amount_pixels"]
    if not set(columns).issubset(audited.columns):
        raise ValueError(f"Geometry audit missing columns: {set(columns)-set(audited.columns)}")
    if generated.duplicated(keys).any() or audited.duplicated(keys).any():
        raise ValueError("Duplicate ID-condition pair")
    m = generated[columns].merge(audited[columns], on=keys, how="outer", suffixes=("_new", "_audit"), indicator=True)
    if not m._merge.eq("both").all() or len(m) != len(generated):
        raise ValueError("Geometry audit and feature rows have different IDs or conditions")
    if not (m.valid_new.astype(bool) == m.valid_audit.astype(bool)).all():
        raise ValueError("Perturbation availability disagrees with Step 12 geometry audit")
    for col in ("actual_iou", "changed_pixels", "area_ratio", "amount_pixels"):
        a = pd.to_numeric(m[f"{col}_new"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(m[f"{col}_audit"], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, rtol=1e-8, atol=1e-8, equal_nan=True):
            raise ValueError(f"Step 12 geometry audit mismatch: {col}")


def extract_perturbed_development(zip_path: Path, manifest_path: Path, clean_path: Path,
                                  geometry_path: Path, output_path: Path,
                                  summary_path: Path, limit: int | None = 12) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"ID": str})
    required = {"ID", "Case", "Pathology", "split", "dev_fold"}
    if not required.issubset(manifest):
        raise ValueError(f"Split manifest missing: {required - set(manifest)}")
    if manifest.ID.duplicated().any():
        raise ValueError("Duplicated image IDs")
    dev = manifest.loc[manifest.split.eq("development")].sort_values("ID")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        dev = dev.head(limit)
    if len(dev) == 0 or not dev.dev_fold.between(0, 4).all():
        raise ValueError("Invalid development-only subset")
    clean = pd.read_csv(clean_path, dtype={"ID": str})
    if clean["ID"].duplicated().any():
        raise ValueError("Duplicate image IDs in clean feature table")
    clean = clean.set_index("ID")
    if not set(dev.ID).issubset(clean.index):
        raise ValueError("Missing original clean features for selected development images")
    for rec in dev.itertuples(index=False):
        prev = clean.loc[rec.ID]
        if (int(prev.Case) != int(rec.Case) or str(prev.Pathology) != str(rec.Pathology)
                or str(prev.split) != "development" or int(prev.dev_fold) != int(rec.dev_fold)):
            raise ValueError(f"Clean features conflict with manifest for {rec.ID}")
    audited = pd.read_csv(geometry_path, dtype={"ID": str})
    audited = audited.loc[audited.ID.isin(set(dev.ID))].copy()
    if len(audited) != 8 * len(dev):
        raise ValueError("Geometry audit does not contain eight rows for every selected development image")
    rows = []
    with ZipFile(zip_path) as archive:
        for n, rec in enumerate(dev.itertuples(index=False), 1):
            image, mask = preprocess_pair(*read_pair(archive, str(rec.ID)))
            # Ensure identical preprocessing/extractor to previous clean-mask experiment.
            recomputed_clean = extract_features(image, mask)
            cached = clean.loc[rec.ID, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
            if not np.allclose(cached, np.array(list(recomputed_clean.values())), rtol=1e-8, atol=1e-8):
                raise ValueError(f"Original clean features no longer reproduce for {rec.ID}")
            rows.extend(image_rows(rec.ID, rec.Case, rec.Pathology, rec.dev_fold, image, mask))
            if n % 100 == 0:
                print(f"Extracted perturbed features for {n}/{len(dev)} development images...", flush=True)
    result = pd.DataFrame(rows, columns=list(INFO_COLUMNS) + list(FEATURE_COLUMNS))
    if len(result) != 8*len(dev) or result.ID.nunique() != len(dev):
        raise AssertionError("Incomplete perturbation output")
    valid = result.valid.astype(bool)
    if not result.split.eq("development").all():
        raise AssertionError("Test image leaked into perturbation extraction")
    if not np.isfinite(result.loc[valid, list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise AssertionError("Nonfinite valid feature measurement")
    if not result.loc[~valid, list(FEATURE_COLUMNS)].isna().all().all():
        raise AssertionError("Unavailable masks must not be represented by invented features")
    check_geometry_match(result, audited)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.12g")
    summary = {
        "development_images": int(len(dev)), "unique_patients": int(dev.Case.nunique()),
        "conditions": len(VARIANT_NAMES), "intended_rows": int(len(result)),
        "valid_rows": int(valid.sum()), "unavailable_rows": int((~valid).sum()),
        "numerical_features": len(FEATURE_COLUMNS), "test_images_seen": 0,
        "geometry_audit_matched": True,
        "by_condition": {name: {"valid": int(g.valid.sum()), "unavailable": int((~g.valid).sum()),
                                "iou_min": (None if g.actual_iou.isna().all() else float(g.actual_iou.min())),
                                "iou_median": (None if g.actual_iou.isna().all() else float(g.actual_iou.median()))}
                         for name, g in result.groupby("condition", sort=False)}
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("BUS-BRA PERTURBED FEATURE EXTRACTION")
    print("="*44)
    print(f"Development images: {len(dev)} | Patients: {dev.Case.nunique()} | Features: {len(FEATURE_COLUMNS)}")
    print(f"Available features: {int(valid.sum())}/{len(result)} variants")
    print(f"Unavailable variants: {int((~valid).sum())} (explicitly flagged; no substitute clean masks)")
    print("Original clean features: MATCHED")
    print("Step 12 geometry report: MATCHED")
    print("Test set: NOT USED")
    print(f"Saved feature table: {output_path}")
    print(f"Saved summary: {summary_path}")
    return result


def main():
    p = argparse.ArgumentParser(description="Extract unchanged-image, changed-mask BUS-BRA features")
    p.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data/raw/BUSBRA.zip")
    p.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/image_splits.csv")
    p.add_argument("--clean", type=Path, default=PROJECT_ROOT / "data/processed/development_features.csv")
    p.add_argument("--geometry", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--all-development", action="store_true")
    args = p.parse_args()
    if args.all_development and args.limit != 12:
        p.error("Do not combine --all-development with --limit")
    limit = None if args.all_development else args.limit
    base = PROJECT_ROOT / "results/robustness"
    geometry = args.geometry or base / ("perturbation_geometry_development.csv" if limit is None else f"perturbation_geometry_smoke_{limit}.csv")
    output = args.output or PROJECT_ROOT / "data/processed" / ("development_perturbed_features.csv" if limit is None else f"perturbed_features_smoke_{limit}.csv")
    summary = args.summary or base / ("perturbed_feature_summary.json" if limit is None else f"perturbed_feature_smoke_{limit}_summary.json")
    for path in (args.zip, args.manifest, args.clean, geometry):
        if not path.is_file():
            raise FileNotFoundError(path)
    extract_perturbed_development(args.zip, args.manifest, args.clean, geometry, output, summary, limit)


if __name__ == "__main__":
    main()
