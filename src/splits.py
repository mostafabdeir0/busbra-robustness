"""Create leakage-safe, deterministic patient-level BUS-BRA splits.

Place at src/splits.py and run from the project root:
    .\\.venv\\Scripts\\python.exe -m src.splits

Reads data/raw/BUSBRA.zip directly; writes CSV/JSON under splits/.
The split files are small and SHOULD be committed to Git for reproducibility.
This script does not extract or modify BUS-BRA images or masks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TEST_FRACTION = 0.20
CV_FOLDS = 5


def safe_write_csv(data: pd.DataFrame, path: Path) -> None:
    """Prevent accidental changes to previously locked split assignments."""
    if path.exists():
        existing = pd.read_csv(path)
        try:
            pd.testing.assert_frame_equal(
                existing.reset_index(drop=True),
                data.reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as exc:
            raise RuntimeError(
                f"Locked split file differs from today's output: {path}. "
                "Do not overwrite it without investigating the cause."
            ) from exc
        print(f"Verified existing locked file: {path}")
    else:
        data.to_csv(path, index=False)
        print(f"Created: {path}")


def create_splits(zip_path: Path, output_dir: Path) -> dict:
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {zip_path}")

    with ZipFile(zip_path) as zf:
        csv_bytes = zf.read("BUSBRA/bus_data.csv")
    metadata = pd.read_csv(io.BytesIO(csv_bytes))
    required = {"Case", "ID", "Pathology"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(metadata.columns))}")
    if metadata[list(required)].isna().any().any():
        raise ValueError("Metadata contains missing Case, ID or Pathology values")
    if metadata["ID"].duplicated().any():
        raise ValueError("Duplicate image IDs; investigate before splitting")

    metadata = metadata.copy()
    metadata["Pathology"] = metadata["Pathology"].str.strip().str.lower()
    if not metadata["Pathology"].isin({"benign", "malignant"}).all():
        raise ValueError("Unexpected pathology value")
    if (metadata.groupby("Case")["Pathology"].nunique() != 1).any():
        raise ValueError("One patient has inconsistent pathology labels")

    # Each patient appears exactly once in this table.
    patients = (
        metadata.groupby("Case", as_index=False)
        .agg(Pathology=("Pathology", "first"), n_images=("ID", "size"))
        .sort_values("Case")
        .reset_index(drop=True)
    )
    if len(patients) != 1064 or len(metadata) != 1875:
        raise ValueError("Unexpected BUS-BRA sample counts; verify dataset version")

    development, test = train_test_split(
        patients,
        test_size=TEST_FRACTION,
        stratify=patients["Pathology"],
        shuffle=True,
        random_state=SEED,
    )
    patients["split"] = "test"
    patients["dev_fold"] = -1
    patients.loc[development.index, "split"] = "development"

    # Fold assignment is performed on patients, never on image rows.
    dev_for_cv = patients.loc[patients["split"] == "development"].copy()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_train_idx, valid_idx) in enumerate(
        cv.split(dev_for_cv[["Case"]], dev_for_cv["Pathology"])
    ):
        patient_row_indices = dev_for_cv.iloc[valid_idx].index
        patients.loc[patient_row_indices, "dev_fold"] = fold

    assert len(patients) == patients["Case"].nunique()
    assert set(patients.loc[patients["split"] == "development", "Case"]).isdisjoint(
        set(patients.loc[patients["split"] == "test", "Case"])
    )
    assert (patients.loc[patients["split"] == "test", "dev_fold"] == -1).all()
    assert set(patients.loc[patients["split"] == "development", "dev_fold"]) == set(range(CV_FOLDS))

    # Image manifest is for convenience; it inherits patient assignments exactly.
    images = metadata[["ID", "Case", "Pathology"]].merge(
        patients[["Case", "split", "dev_fold"]],
        on="Case", how="left", validate="many_to_one"
    ).sort_values(["Case", "ID"]).reset_index(drop=True)
    assert len(images) == len(metadata)
    assert (images.groupby("Case")["split"].nunique() == 1).all()
    assert (images.groupby("Case")["dev_fold"].nunique() == 1).all()
    assert not images[["split", "dev_fold"]].isna().any().any()

    output_dir.mkdir(parents=True, exist_ok=True)
    patients = patients[["Case", "Pathology", "n_images", "split", "dev_fold"]]
    safe_write_csv(patients, output_dir / "patient_splits.csv")
    safe_write_csv(images, output_dir / "image_splits.csv")

    def counts(df: pd.DataFrame) -> dict:
        return {
            "patients": int(len(df)),
            "images": int(df["n_images"].sum()),
            "benign_patients": int((df["Pathology"] == "benign").sum()),
            "malignant_patients": int((df["Pathology"] == "malignant").sum()),
        }

    summary = {
        "random_seed": SEED,
        "test_fraction_requested": TEST_FRACTION,
        "cv_folds": CV_FOLDS,
        "dataset_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "all": counts(patients),
        "development": counts(patients[patients["split"] == "development"]),
        "test": counts(patients[patients["split"] == "test"]),
        "development_validation_folds": {
            str(f): counts(patients[(patients["split"] == "development") & (patients["dev_fold"] == f)])
            for f in range(CV_FOLDS)
        },
    }
    summary_path = output_dir / "split_summary.json"
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Existing split summary differs: {summary_path}")
    summary_path.write_text(serialized, encoding="utf-8")

    print("\nBUS-BRA PATIENT SPLIT VERIFICATION")
    print("=" * 45)
    for label in ("all", "development", "test"):
        c = summary[label]
        print(f"{label.title():12} {c['patients']:4} patients | {c['images']:4} images | "
              f"benign {c['benign_patients']:3} | malignant {c['malignant_patients']:3}")
    for fold, c in summary["development_validation_folds"].items():
        print(f"Dev fold {fold}: {c['patients']} validation patients "
              f"(benign {c['benign_patients']}, malignant {c['malignant_patients']})")
    print("Patient leakage: 0")
    print("Fold overlap: 0")
    print(f"Split files: {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible patient-level BUS-BRA splits")
    parser.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data/raw/BUSBRA.zip")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "splits")
    args = parser.parse_args()
    create_splits(args.zip, args.output)


if __name__ == "__main__":
    main()
