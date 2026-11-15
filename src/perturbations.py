"""BUS-BRA: deterministic, label-blind lesion mask perturbations.

Usage from repository root:
    .\\.venv\\Scripts\\python.exe -m src.perturbations --limit 12

This is the GEOMETRY AUDIT / smoke test only. No models, test-set data, or
labels are used to *select* perturbations. The image remains unchanged.
Eight prescribed cases: partial boundary erosion/dilation at ~.95/.90
original-mask IoU, and translations left/right/up/down at ~.90 IoU.

Partial erosion/dilation removes/adds boundary pixels in order of distance
to the existing contour, with deterministic tie-breaking within each distance
band. This avoids identical mild and moderate variants on small lesions.
These are synthetic approximations, not actual radiologist annotations.

IoU targets are approximate because masks are pixel grids. Record actual
IoU, and invalid/unavailable variants; do not silently substitute identity.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .features import PROJECT_ROOT, preprocess_pair, read_pair

# These are protocol hypotheses, not empirical severity estimates.
TARGET_IOUS = {"mild": 0.95, "moderate": 0.90}
MAX_TRANSLATION = 24
MIN_PIXELS = 15


@dataclass(frozen=True)
class Variant:
    condition: str
    mask: np.ndarray
    iou: float
    changed_pixels: int
    area_ratio: float
    amount_pixels: int


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Masks must have same 2-D shape")
    union = np.logical_or(a, b).sum()
    if not union:
        raise ValueError("Both masks empty")
    return float(np.logical_and(a, b).sum() / union)


def partial_boundary_change(mask: np.ndarray, target_iou: float,
                            kind: str) -> np.ndarray:
    """Modify approximately 5% or 10% of area near the lesion contour.

    Unlike integer-radius disk morphology, this produces distinct mild/moderate
    alterations for small masks. Candidate pixel order uses distance-to-contour
    with deterministic tie breaks based on original mask bytes. The stricter
    perturbation is a superset of the gentler perturbation's changed pixels.
    """
    original = np.asarray(mask, dtype=bool)
    if kind not in {"erosion", "dilation"}:
        raise ValueError("Expected erosion or dilation")
    if not (0 < target_iou < 1):
        raise ValueError("target_iou must lie strictly between 0 and 1")
    area = int(original.sum())
    if area < MIN_PIXELS:
        raise ValueError("Original mask is too small")
    # IoU(A\B,A) = (area - removed)/area for removal;
    # IoU(A,A∪B) = area/(area + added) for addition.
    n_change = (round(area*(1-target_iou)) if kind == "erosion"
                else round(area*(1/target_iou-1)))
    n_change = max(1, int(n_change))
    candidates = original if kind == "erosion" else ~original
    locations = np.flatnonzero(candidates)
    if kind == "erosion" and area-n_change < MIN_PIXELS:
        raise ValueError("Erosion would make mask too small")
    if n_change > len(locations):
        raise ValueError("Insufficient surrounding pixels for dilation")
    distances = (ndi.distance_transform_edt(original) if kind == "erosion"
                 else ndi.distance_transform_edt(~original))
    seed = int.from_bytes(hashlib.blake2b(original.tobytes(), digest_size=8).digest(),
                          byteorder="little")
    noise = np.random.default_rng(seed).random(len(locations))
    # Proximity to contour is the primary sort key; reproducible pseudo-random
    # ordering breaks ties within a one-pixel boundary ring.
    sorted_locations = locations[np.lexsort((noise, distances.flat[locations]))]
    chosen = sorted_locations[:n_change]
    updated = original.copy()
    updated.flat[chosen] = (kind == "dilation")
    return updated


def translate_no_wrap(mask: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, int]:
    """Zero-padded shift, never wraps around; returns number of clipped pixels."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected 2-D mask")
    h, w = mask.shape
    result = np.zeros_like(mask)
    if abs(dy) >= h or abs(dx) >= w:
        return result, int(mask.sum())
    sy0, sy1 = max(0, -dy), min(h, h-dy)
    sx0, sx1 = max(0, -dx), min(w, w-dx)
    if sy0 < sy1 and sx0 < sx1:
        result[sy0+dy:sy1+dy, sx0+dx:sx1+dx] = mask[sy0:sy1, sx0:sx1]
    return result, int(mask.sum() - result.sum())


def _closest(mask: np.ndarray, name: str, target: float, candidates) -> Variant | None:
    chosen = None
    for amount, candidate in candidates:
        if candidate.sum() < MIN_PIXELS:
            continue
        overlap = mask_iou(mask, candidate)
        # No identity fallback; all accepted changes must modify the boundary.
        if overlap >= 1.0:
            continue
        comparison = (abs(overlap-target), amount)
        if chosen is None or comparison < chosen[0]:
            chosen = (comparison, amount, candidate.copy(), overlap)
    if chosen is None:
        return None
    _, amount, altered, overlap = chosen
    return Variant(name, altered, overlap,
                   int(np.logical_xor(mask, altered).sum()),
                   float(altered.sum()/mask.sum()), int(amount))


def make_variants(mask: np.ndarray) -> dict[str, Variant | None]:
    """Return all eight named variants; None means impossible without clipping/empty mask.

    Approximate target IoU is optimized purely from each mask's geometry.
    No outcome labels, model scores, or held-out samples influence parameters.
    """
    original = np.asarray(mask, dtype=bool)
    if original.ndim != 2 or original.sum() < MIN_PIXELS:
        raise ValueError("Invalid or too-small original lesion mask")
    result: dict[str, Variant | None] = {}
    for name, target in TARGET_IOUS.items():
        for kind in ("erosion", "dilation"):
            condition = f"{kind}_{name}"
            altered = partial_boundary_change(original, target, kind)
            result[condition] = Variant(
                condition, altered, mask_iou(original, altered),
                int(np.logical_xor(original, altered).sum()),
                float(altered.sum()/original.sum()),
                int(np.logical_xor(original, altered).sum()),
            )
    for direction, dyunit, dxunit in (
        ("left", 0, -1), ("right", 0, 1),
        ("up", -1, 0), ("down", 1, 0)
    ):
        candidates = []
        for distance in range(1, MAX_TRANSLATION+1):
            shifted, clipped = translate_no_wrap(original, dyunit*distance, dxunit*distance)
            if clipped == 0:
                candidates.append((distance, shifted))
        result[f"shift_{direction}_moderate"] = _closest(
            original, f"shift_{direction}_moderate", TARGET_IOUS["moderate"], candidates)
    assert len(result) == 8
    return result


def audit_variants(zip_path: Path, manifest_path: Path, output_path: Path,
                   limit: int | None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    necessary = {"ID", "Case", "Pathology", "split", "dev_fold"}
    if not necessary.issubset(manifest.columns):
        raise ValueError("Manifest missing expected split columns")
    subset = manifest.loc[manifest["split"].eq("development")].sort_values("ID")
    if limit is not None:
        if limit <= 0:
            raise ValueError("Positive --limit required")
        subset = subset.head(limit)
    if subset.empty:
        raise ValueError("No development images found")
    all_rows = []
    with ZipFile(zip_path) as zf:
        for n, record in enumerate(subset.itertuples(index=False), 1):
            image, mask = preprocess_pair(*read_pair(zf, str(record.ID)))
            # The same 'image' is intentionally reused by all variants.
            assert image.shape == mask.shape
            variants = make_variants(mask)
            for condition, variation in variants.items():
                row = {"ID": str(record.ID), "Case": record.Case,
                       "condition": condition, "valid": variation is not None}
                if variation is not None:
                    row.update({"actual_iou": variation.iou,
                                "changed_pixels": variation.changed_pixels,
                                "area_ratio": variation.area_ratio,
                                "amount_pixels": variation.amount_pixels})
                all_rows.append(row)
            if n % 100 == 0:
                print(f"Geometry-checked {n}/{len(subset)} development images...", flush=True)
    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("BUS-BRA PERTURBATION GEOMETRY AUDIT")
    print("="*44)
    print(f"Development images: {subset.shape[0]} | Prespecified variants: 8")
    print(f"Valid variants: {int(df.valid.sum())}/{len(df)}")
    print(df.groupby("condition", sort=False).agg(
        valid=("valid", "sum"),
        median_iou=("actual_iou", "median"),
        min_iou=("actual_iou", "min"),
        max_iou=("actual_iou", "max")
    ).round(3).to_string())
    print(f"Saved: {output_path}")
    print("These are geometry checks, not classifier robustness results.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="BUS-BRA eight-mask geometry audit")
    parser.add_argument("--zip", type=Path, default=PROJECT_ROOT/"data/raw/BUSBRA.zip")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT/"splits/image_splits.csv")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--all-development", action="store_true")
    args = parser.parse_args()
    limit = None if args.all_development else args.limit
    output = PROJECT_ROOT/"results/robustness"/(
        "perturbation_geometry_development.csv" if limit is None else
        f"perturbation_geometry_smoke_{limit}.csv")
    audit_variants(args.zip, args.manifest, output, limit)


if __name__ == "__main__":
    main()
