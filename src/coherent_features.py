"""Step 15: post-hoc coherent-boundary feature extraction, development-only.

Run from project root:
    python -m src.coherent_features --limit 12
    python -m src.coherent_features --all-development

Alternative protocol: four coherent erosion/dilation conditions only.
Original Step 13 feature tables, original Step 14 results and translation
conditions are neither read as modified data nor overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .coherent_perturbations import coherent_boundary_change
from .features import FEATURE_COLUMNS, PROJECT_ROOT, extract_features, preprocess_pair, read_pair
from .perturbations import TARGET_IOUS, mask_iou
from .perturbed_features import INFO_COLUMNS

COHERENT_CONDITIONS = (
    "erosion_mild", "dilation_mild", "erosion_moderate", "dilation_moderate"
)


def coherent_image_rows(image_id: str, case: int, pathology: str, dev_fold: int,
                        image: np.ndarray, original_mask: np.ndarray) -> list[dict]:
    """Generate four alternative records from one UNCHANGED image/mask pair."""
    if image.shape != original_mask.shape or image.ndim != 2:
        raise ValueError("Image and original mask must have matching 2D shape")
    original_image_copy = image.copy()
    original_mask_copy = original_mask.copy()
    rows = []
    for condition in COHERENT_CONDITIONS:
        kind, severity = condition.split("_")
        changed = coherent_boundary_change(original_mask, TARGET_IOUS[severity], kind)
        if np.array_equal(changed, original_mask):
            raise AssertionError("A coherent perturbation left the mask unchanged")
        changed_pixels = int(np.logical_xor(original_mask, changed).sum())
        row = {
            "ID": str(image_id), "Case": int(case), "Pathology": str(pathology),
            "split": "development", "dev_fold": int(dev_fold), "condition": condition,
            "valid": True, "failure_reason": "",
            "actual_iou": mask_iou(original_mask, changed),
            "changed_pixels": changed_pixels,
            "area_ratio": float(changed.sum() / original_mask.sum()),
            "amount_pixels": changed_pixels,
        }
        row.update(extract_features(image, changed))
        rows.append(row)
    if not np.array_equal(image, original_image_copy):
        raise AssertionError("Original ultrasound image was mutated")
    if not np.array_equal(original_mask, original_mask_copy):
        raise AssertionError("Original lesion mask was mutated")
    return rows


def verify_coherent_geometry(generated: pd.DataFrame, geometry: pd.DataFrame,
                             clean: pd.DataFrame) -> None:
    """Reject drift between Step 14.3 geometry and newly extracted features."""
    keys = ["ID", "condition"]
    required = set(keys + ["valid", "actual_iou", "changed_pixels",
                           "circularity_clean", "circularity_coherent_protocol"])
    if not required.issubset(geometry):
        raise ValueError(f"Coherent geometry report lacks: {required - set(geometry)}")
    if generated.duplicated(keys).any() or geometry.duplicated(keys).any():
        raise ValueError("Duplicate ID/condition in alternative geometry audit")
    if geometry["valid"].astype(str).str.lower().ne("true").any():
        raise ValueError("Step 14.3 reported an invalid coherent mask")
    merged = generated.merge(geometry[list(required)], on=keys, how="outer",
                             suffixes=("_new", "_audit"), indicator=True,
                             validate="one_to_one")
    if len(merged) != len(generated) or not merged["_merge"].eq("both").all():
        raise ValueError("Geometry audit and extracted features have different coverage")
    for a, b in (("actual_iou_new", "actual_iou_audit"),
                 ("changed_pixels_new", "changed_pixels_audit"),
                 ("shape_circularity", "circularity_coherent_protocol")):
        if not np.allclose(merged[a].to_numpy(float), merged[b].to_numpy(float),
                           rtol=1e-8, atol=1e-8):
            raise ValueError(f"Coherent geometry mismatch: {a} versus {b}")
    reference = clean.set_index("ID")["shape_circularity"]
    old_circ = reference.loc[merged["ID"]].to_numpy(float)
    if not np.allclose(old_circ, merged["circularity_clean"].to_numpy(float),
                       rtol=1e-8, atol=1e-8):
        raise ValueError("Coherent audit's original circularity differs from clean features")


def extract_coherent_development(zip_path: Path, manifest_path: Path, clean_path: Path,
                                 geometry_path: Path, output_path: Path,
                                 summary_path: Path, limit: int | None = 12) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"ID": str})
    needed = {"ID", "Case", "Pathology", "split", "dev_fold"}
    if not needed.issubset(manifest) or manifest.ID.duplicated().any():
        raise ValueError("Invalid frozen image split manifest")
    dev = manifest.loc[manifest.split.eq("development")].sort_values("ID")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        dev = dev.head(limit)
    if dev.empty or dev.dev_fold.isna().any() or not dev.dev_fold.between(0, 4).all():
        raise ValueError("Invalid development subset")

    clean = pd.read_csv(clean_path, dtype={"ID": str})
    if clean.ID.duplicated().any() or not set(dev.ID).issubset(set(clean.ID)):
        raise ValueError("Missing or duplicate original clean features")
    indexed_clean = clean.set_index("ID")
    audit = pd.read_csv(geometry_path, dtype={"ID": str})
    audit = audit.loc[audit.ID.isin(set(dev.ID))].copy()
    if len(audit) != len(dev) * len(COHERENT_CONDITIONS):
        raise ValueError("Geometry audit lacks four conditions per selected image")

    rows = []
    with ZipFile(zip_path) as archive:
        for idx, item in enumerate(dev.itertuples(index=False), 1):
            cached_row = indexed_clean.loc[item.ID]
            if (int(cached_row.Case) != int(item.Case)
                    or str(cached_row.Pathology) != str(item.Pathology)
                    or str(cached_row.split) != "development"
                    or int(cached_row.dev_fold) != int(item.dev_fold)):
                raise ValueError(f"Clean features/manifest disagree for {item.ID}")
            image, mask = preprocess_pair(*read_pair(archive, item.ID))
            recomputed_clean = extract_features(image, mask)
            cached_features = cached_row[list(FEATURE_COLUMNS)].to_numpy(float)
            if not np.allclose(cached_features, list(recomputed_clean.values()),
                               rtol=1e-8, atol=1e-8):
                raise ValueError(f"Original features differ for {item.ID}")
            rows.extend(coherent_image_rows(item.ID, item.Case, item.Pathology,
                                            item.dev_fold, image, mask))
            if idx % 100 == 0:
                print(f"Extracted coherent features for {idx}/{len(dev)} images...", flush=True)
    result = pd.DataFrame(rows, columns=list(INFO_COLUMNS) + list(FEATURE_COLUMNS))
    if len(result) != len(dev) * 4 or result.duplicated(["ID", "condition"]).any():
        raise AssertionError("Missing/duplicate alternative perturbation rows")
    if not result.split.eq("development").all() or not result.valid.all():
        raise AssertionError("Unexpected split or invalid alternative perturbation")
    if not np.isfinite(result[list(FEATURE_COLUMNS)].to_numpy(float)).all():
        raise AssertionError("Nonfinite alternative feature")
    verify_coherent_geometry(result, audit, clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.12g")
    summary = {
        "status": "post-hoc coherent boundary sensitivity features; original protocol preserved",
        "development_images": int(len(dev)), "development_patients": int(dev.Case.nunique()),
        "coherent_conditions": list(COHERENT_CONDITIONS),
        "feature_vectors": int(len(result)), "valid_feature_vectors": int(result.valid.sum()),
        "numeric_features": len(FEATURE_COLUMNS), "test_images_seen": 0,
        "clean_features_reproduced": True, "coherent_geometry_audit_matched": True,
        "original_step14_outputs_untouched": True,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("BUS-BRA — POST-HOC COHERENT FEATURE EXTRACTION")
    print("=" * 51)
    print(f"Development images: {len(dev)} | Patients: {dev.Case.nunique()} | Features: {len(FEATURE_COLUMNS)}")
    print(f"Valid alternative feature vectors: {len(result)}/{len(dev) * 4}")
    print("Cached clean features and Step 14.3 geometry: MATCHED")
    print("Original Step 14 robustness results: UNCHANGED")
    print("Held-out test: NOT USED")
    print(f"Saved: {output_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Generate alternative coherent-boundary feature table")
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
        p.error("Do not combine --all-development with a custom --limit")
    limit = None if args.all_development else args.limit
    root = PROJECT_ROOT / "results/robustness"
    geom = args.geometry or root / ("coherent_geometry_development.csv" if limit is None
                                   else f"coherent_geometry_smoke_{limit}.csv")
    out = args.output or PROJECT_ROOT / "data/processed" / (
        "development_coherent_features.csv" if limit is None else f"coherent_features_smoke_{limit}.csv")
    summary = args.summary or root / ("coherent_feature_summary.json" if limit is None
                                    else f"coherent_feature_smoke_{limit}_summary.json")
    for path in (args.zip, args.manifest, args.clean, geom):
        if not path.is_file():
            raise FileNotFoundError(path)
    extract_coherent_development(args.zip, args.manifest, args.clean, geom, out, summary, limit)


if __name__ == "__main__":
    main()
