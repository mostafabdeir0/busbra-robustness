"""BUS-BRA v1.0 dataset audit. Run from the busbra-robustness project root.

Reads data/raw/BUSBRA.zip directly (no extraction needed).
Checks image/mask pairing, patient label consistency, image dimensions,
mask validity, and official cross-validation fold grouping.
Outputs are written to results/audit/ (excluded from Git by default).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def scan_dataset(archive_path: Path, output_dir: Path) -> dict:
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Dataset archive not found: {archive_path}\n"
            "Download BUSBRA.zip from https://zenodo.org/records/8231412 "
            "and place it at data/raw/BUSBRA.zip."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict] = []
    dimension_mismatches: list[dict] = []
    mask_area_fractions: list[float] = []

    with archive_path.open("rb") as archive_file:
        sha256 = hashlib.file_digest(archive_file, "sha256").hexdigest()

    try:
        with ZipFile(archive_path, "r") as zf:
            paths = set(zf.namelist())
            csv_path = "BUSBRA/bus_data.csv"
            if csv_path not in paths:
                raise ValueError(f"Missing {csv_path} inside archive")
            metadata = pd.read_csv(io.BytesIO(zf.read(csv_path)))
            required = {"ID", "Case", "Pathology", "Width", "Height"}
            missing_columns = required - set(metadata.columns)
            if missing_columns:
                raise ValueError(f"Missing metadata columns: {sorted(missing_columns)}")

            image_paths = {
                p for p in paths if p.startswith("BUSBRA/Images/") and p.endswith(".png")
            }
            mask_paths = {
                p for p in paths if p.startswith("BUSBRA/Masks/") and p.endswith(".png")
            }

            duplicate_ids = int(metadata["ID"].duplicated().sum())
            label_values = metadata["Pathology"].astype(str).str.strip().str.lower()
            if not label_values.isin(["benign", "malignant"]).all():
                issues.append({"type": "unexpected_pathology_value", "detail": str(sorted(label_values.unique()))})
            case_label_count = (
                metadata.assign(_label=label_values).groupby("Case") ["_label"].nunique()
            )
            inconsistent_cases = case_label_count[case_label_count != 1].index.tolist()
            if inconsistent_cases:
                issues.append({"type": "inconsistent_patient_labels", "detail": str(inconsistent_cases[:20])})
            if duplicate_ids:
                issues.append({"type": "duplicate_image_ID", "detail": str(duplicate_ids)})

            expected_images = {f"BUSBRA/Images/{x}.png" for x in metadata["ID"].astype(str)}
            expected_masks = {
                f"BUSBRA/Masks/mask_{x.removeprefix('bus_')}.png"
                for x in metadata["ID"].astype(str)
            }
            for category, actual, expected in (
                ("images", image_paths, expected_images),
                ("masks", mask_paths, expected_masks),
            ):
                for p in sorted(expected - actual):
                    issues.append({"type": f"missing_{category}", "detail": p})
                for p in sorted(actual - expected):
                    issues.append({"type": f"unlisted_{category}", "detail": p})

            bad_image_mask_dimensions = 0
            empty_masks = 0
            nonbinary_masks = 0
            unreadable_files = 0
            verified_pairs = 0
            for row in metadata.itertuples(index=False):
                identifier = str(row.ID)
                img_path = f"BUSBRA/Images/{identifier}.png"
                mask_path = f"BUSBRA/Masks/mask_{identifier.removeprefix('bus_')}.png"
                if img_path not in paths or mask_path not in paths:
                    continue
                try:
                    with Image.open(io.BytesIO(zf.read(img_path))) as image:
                        image.load()
                        image_size = image.size
                    with Image.open(io.BytesIO(zf.read(mask_path))) as mask:
                        mask.load()
                        mask_size = mask.size
                        mask_array = np.asarray(mask.convert("L"))
                except (OSError, ValueError, UnidentifiedImageError) as exc:
                    unreadable_files += 1
                    issues.append({"type": "unreadable_image_or_mask", "detail": f"{identifier}: {exc}"})
                    continue

                if image_size != mask_size:
                    bad_image_mask_dimensions += 1
                    issues.append({"type": "image_mask_size_mismatch", "detail": identifier})
                if image_size != (int(row.Width), int(row.Height)):
                    dimension_mismatches.append({
                        "ID": identifier,
                        "csv_width": int(row.Width),
                        "csv_height": int(row.Height),
                        "actual_width": image_size[0],
                        "actual_height": image_size[1],
                    })
                if not np.any(mask_array):
                    empty_masks += 1
                    issues.append({"type": "empty_mask", "detail": identifier})
                unique_vals = np.unique(mask_array)
                if not set(unique_vals.tolist()).issubset({0, 1, 255}):
                    nonbinary_masks += 1
                    issues.append({"type": "nonbinary_mask", "detail": f"{identifier}: values={unique_vals[:8]}"})
                mask_area_fractions.append(float(np.count_nonzero(mask_array) / mask_array.size))
                verified_pairs += 1

            official_folds: dict[str, dict] = {}
            for fold_filename in ("5-fold-cv.csv", "10-fold-cv.csv"):
                fold_path = f"BUSBRA/{fold_filename}"
                if fold_path not in paths:
                    issues.append({"type": "missing_official_fold_file", "detail": fold_filename})
                    continue
                folds = pd.read_csv(io.BytesIO(zf.read(fold_path)))
                if not {"ID", "kFold"}.issubset(folds.columns):
                    issues.append({"type": "invalid_fold_columns", "detail": fold_filename})
                    continue
                combined = metadata[["ID", "Case"]].merge(
                    folds[["ID", "kFold"]], on="ID", how="left", validate="one_to_one"
                )
                missing_folds = int(combined["kFold"].isna().sum())
                split_cases = int((combined.groupby("Case")["kFold"].nunique() > 1).sum())
                official_folds[fold_filename] = {
                    "fold_counts_by_image": {
                        str(k): int(v) for k, v in folds["kFold"].value_counts().sort_index().items()
                    },
                    "images_missing_fold": missing_folds,
                    "patients_assigned_to_multiple_folds": split_cases,
                }
                if missing_folds or split_cases:
                    issues.append({
                        "type": "fold_integrity_problem", "detail": f"{fold_filename}: missing={missing_folds}, split_cases={split_cases}"
                    })

            patient_labels = (
                metadata.assign(_label=label_values)
                .drop_duplicates(subset=["Case"])["_label"]
                .value_counts().to_dict()
            )
            image_labels = label_values.value_counts().to_dict()
            summary = {
                "archive_name": archive_path.name,
                "archive_sha256": sha256,
                "image_png_files": len(image_paths),
                "mask_png_files": len(mask_paths),
                "metadata_rows": len(metadata),
                "patients": int(metadata["Case"].nunique()),
                "patient_labels": {k: int(v) for k, v in patient_labels.items()},
                "image_labels": {k: int(v) for k, v in image_labels.items()},
                "images_per_patient": {
                    str(k): int(v) for k, v in metadata.groupby("Case").size().value_counts().sort_index().items()
                },
                "duplicate_image_ids": duplicate_ids,
                "inconsistent_patient_labels": len(inconsistent_cases),
                "verified_image_mask_pairs": verified_pairs,
                "image_mask_dimension_mismatches": bad_image_mask_dimensions,
                "metadata_dimension_mismatches": len(dimension_mismatches),
                "empty_masks": empty_masks,
                "nonbinary_masks": nonbinary_masks,
                "unreadable_pairs": unreadable_files,
                "mask_area_fraction_min": min(mask_area_fractions) if mask_area_fractions else None,
                "mask_area_fraction_max": max(mask_area_fractions) if mask_area_fractions else None,
                "official_folds": official_folds,
                "blocking_issue_count": len(issues),
            }
    except BadZipFile as exc:
        raise ValueError("The BUSBRA.zip archive is corrupted or not a ZIP file") from exc

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(dimension_mismatches, columns=[
        "ID", "csv_width", "csv_height", "actual_width", "actual_height"
    ]).to_csv(output_dir / "metadata_dimension_mismatches.csv", index=False)
    pd.DataFrame(issues, columns=["type", "detail"]).to_csv(output_dir / "issues.csv", index=False)

    print("\nBUS-BRA DATA AUDIT")
    print("=" * 46)
    print(f"Image files / mask files: {summary['image_png_files']} / {summary['mask_png_files']}")
    print(f"Metadata rows: {summary['metadata_rows']}; unique patients: {summary['patients']}")
    print(f"Patient labels: {summary['patient_labels']}")
    print(f"Verified image-mask pairs: {verified_pairs}")
    print(f"Image-mask size mismatches: {bad_image_mask_dimensions}")
    print(f"CSV-vs-image dimension discrepancies: {len(dimension_mismatches)} (nonblocking; investigate)")
    print(f"Empty masks: {empty_masks}; nonbinary masks: {nonbinary_masks}")
    for name, checks in official_folds.items():
        print(f"{name}: split patients={checks['patients_assigned_to_multiple_folds']}, missing IDs={checks['images_missing_fold']}")
    print(f"Blocking issues: {len(issues)}")
    print(f"Report: {output_dir / 'summary.json'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BUS-BRA image/mask/label integrity")
    parser.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data" / "raw" / "BUSBRA.zip")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "audit")
    args = parser.parse_args()
    summary = scan_dataset(args.zip, args.output)
    if summary["blocking_issue_count"]:
        raise SystemExit("Audit found blocking issues; read results/audit/issues.csv")


if __name__ == "__main__":
    main()
