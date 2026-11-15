"""Exploratory Step 14.2: spatially localized boundary-perturbation sensitivity audit.

IMPORTANT: This does not replace the frozen Step 11--14 protocol or its saved
results. It creates an alternative, post-hoc perturbation mechanism solely to
assess dependence on the original pseudo-random boundary-pixel tie-breaking.

Algorithm: At a label-blind, deterministic boundary anchor, grow one connected
4-neighbour patch into (dilation) or from (erosion) the lesion. Its priority
favours the original contour and proximity to the anchor; the 10%-area change
includes the 5%-area change exactly. Ultrasound pixels never change.

Commands:
  python -m src.coherent_perturbations --limit 12
  python -m src.coherent_perturbations --all-development
"""
from __future__ import annotations

import argparse
import json
import heapq
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import perimeter_crofton

from .features import PROJECT_ROOT, preprocess_pair, read_pair
from .perturbations import (MIN_PIXELS, TARGET_IOUS, Variant,
                            make_variants, mask_iou, partial_boundary_change)

NEIGHBORS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def choose_anchor(mask: np.ndarray) -> tuple[int, int]:
    """Choose an interior contour pixel with maximum image-frame clearance."""
    boundary = mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(2, 1))
    yy, xx = np.nonzero(boundary)
    if len(yy) == 0:
        raise ValueError("Lesion has no usable boundary")
    h, w = mask.shape
    clearance = np.minimum.reduce([yy, xx, h - yy - 1, w - xx - 1])
    # np.nonzero is row-major; the first maximum is a deterministic tie-break.
    i = int(np.argmax(clearance))
    return int(yy[i]), int(xx[i])


def _grow_patch(mask: np.ndarray, kind: str, n: int, anchor: tuple[int, int]) -> np.ndarray:
    """Grow n changed pixels in a single 4-connected region, near one anchor."""
    h, w = mask.shape
    if kind == "erosion":
        allowed = mask
        origin = anchor
        depth = ndi.distance_transform_edt(mask)
    elif kind == "dilation":
        allowed = ~mask
        adjacent = [(anchor[0] + dy, anchor[1] + dx) for dy, dx in NEIGHBORS]
        adjacent = [(y, x) for y, x in adjacent
                    if 0 <= y < h and 0 <= x < w and allowed[y, x]]
        if not adjacent:
            raise ValueError("No exterior pixel adjacent to boundary anchor")
        # Pick the nearest admissible exterior pixel, using a fixed neighbor order.
        origin = adjacent[0]
        depth = ndi.distance_transform_edt(~mask)
    else:
        raise ValueError("kind must be erosion or dilation")
    if n <= 0 or n > int(allowed.sum()):
        raise ValueError("Number of changed pixels is invalid")

    sigma = max(3.0, 0.60 * np.sqrt(float(mask.sum()) / np.pi))
    seen = np.zeros((h, w), dtype=bool)
    chosen = np.zeros((h, w), dtype=bool)
    q: list[tuple[float, int, int]] = []

    def push(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and allowed[y, x] and not seen[y, x]:
            seen[y, x] = True
            radial = ((y - origin[0]) ** 2 + (x - origin[1]) ** 2) / (2.0 * sigma ** 2)
            # Penalize depth; remove/add along existing contour first.
            heapq.heappush(q, (float(radial + 0.45 * depth[y, x]), y, x))

    push(*origin)
    changed = 0
    while changed < n:
        if not q:
            raise ValueError("Could not grow coherent patch to desired size")
        _, y, x = heapq.heappop(q)
        chosen[y, x] = True
        changed += 1
        for dy, dx in NEIGHBORS:
            push(y + dy, x + dx)
    return chosen


def coherent_boundary_change(mask: np.ndarray, target_iou: float, kind: str) -> np.ndarray:
    original = np.asarray(mask, dtype=bool)
    if original.ndim != 2 or int(original.sum()) < MIN_PIXELS:
        raise ValueError("Invalid or too-small lesion mask")
    if kind not in ("erosion", "dilation") or not 0.0 < target_iou < 1.0:
        raise ValueError("Invalid perturbation type or target")
    area = int(original.sum())
    n = max(1, round(area * (1 - target_iou)) if kind == "erosion"
            else round(area * (1 / target_iou - 1)))
    if kind == "erosion" and area - n < MIN_PIXELS:
        raise ValueError("Erosion would leave too few lesion pixels")
    patch = _grow_patch(original, kind, n, choose_anchor(original))
    changed = original.copy()
    changed[patch] = (kind == "dilation")
    return changed


def make_coherent_variants(mask: np.ndarray) -> dict[str, Variant | None]:
    """Same eight condition names; only erosion/dilation algorithm differs.

    Translation variants use the unmodified original implementation, so the
    comparison changes ONE aspect of the existing experimental protocol.
    """
    original = np.asarray(mask, dtype=bool)
    # Reuse *exactly* the Step 11 translation variants and clipping policy.
    original_variants = make_variants(original)
    out: dict[str, Variant | None] = {}
    for severity, target in TARGET_IOUS.items():
        for kind in ("erosion", "dilation"):
            name = f"{kind}_{severity}"
            changed = coherent_boundary_change(original, target, kind)
            count = int(np.logical_xor(original, changed).sum())
            out[name] = Variant(name, changed, mask_iou(original, changed), count,
                                float(changed.sum() / original.sum()), count)
    for key in ("shift_left_moderate", "shift_right_moderate",
                "shift_up_moderate", "shift_down_moderate"):
        out[key] = original_variants[key]
    return out


def _circ(mask: np.ndarray) -> float:
    perimeter = float(perimeter_crofton(mask, directions=4))
    return float(4 * np.pi * mask.sum() / (perimeter ** 2))


def audit(zip_path: Path, manifest_path: Path, out_path: Path, *, limit: int | None = 12) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"ID": str})
    necessary = {"ID", "Case", "Pathology", "split", "dev_fold"}
    if not necessary.issubset(manifest.columns) or manifest.ID.duplicated().any():
        raise ValueError("Invalid locked image manifest")
    dev = manifest.loc[manifest.split.eq("development")].sort_values("ID")
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        dev = dev.head(limit)
    if dev.empty:
        raise ValueError("No development records")
    rows = []
    with ZipFile(zip_path) as archive:
        for idx, item in enumerate(dev.itertuples(index=False), 1):
            img, original = preprocess_pair(*read_pair(archive, str(item.ID)))
            frozen = img.copy()
            old_circ = _circ(original)
            # Four *changed* conditions only. The four translations are exactly
            # the original algorithm and therefore need no repeated audit.
            # Avoid expensive duplicate construction of 96 translation masks
            # per image (24 shifts x four directions) for this geometry audit.
            for severity, target in TARGET_IOUS.items():
                for kind in ("erosion", "dilation"):
                    condition = f"{kind}_{severity}"
                    old_mask = partial_boundary_change(original, target, kind)
                    new_mask = coherent_boundary_change(original, target, kind)
                    rows.append({
                        "ID": str(item.ID), "Case": int(item.Case),
                        "condition": condition, "valid": True,
                        "actual_iou": mask_iou(original, new_mask),
                        "changed_pixels": int(np.logical_xor(original, new_mask).sum()),
                        "circularity_clean": old_circ,
                        "circularity_original_protocol": _circ(old_mask),
                        "circularity_coherent_protocol": _circ(new_mask),
                    })
            if not np.array_equal(img, frozen):
                raise AssertionError("Ultrasound image mutated")
            if idx % 100 == 0:
                print(f"Audited {idx}/{len(dev)} development images...", flush=True)
    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False, float_format="%.12g")
    old_d = (result.circularity_original_protocol - result.circularity_clean).abs()
    new_d = (result.circularity_coherent_protocol - result.circularity_clean).abs()
    result = result.assign(abs_circ_change_original=old_d, abs_circ_change_coherent=new_d)
    s = result.groupby("condition", sort=False).agg(
        valid=("valid", "sum"), median_iou=("actual_iou", "median"),
        median_abs_circ_original=("abs_circ_change_original", "median"),
        median_abs_circ_coherent=("abs_circ_change_coherent", "median")
    ).round(4)
    summary = {
        "purpose": "post-hoc geometry-only algorithm sensitivity check; NOT replacement of Step 14",
        "development_images": int(len(dev)), "changed_conditions": 4,
        "unchanged_translation_conditions": 4, "test_patients_used": 0,
        "model_predictions_used": 0, "original_step_14_results_untouched": True,
        "valid_variants": int(result.valid.sum()), "intended_variants": int(len(result))
    }
    (out_path.parent / (out_path.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("BUS-BRA STEP 14.2 — COHERENT-BOUNDARY SENSITIVITY AUDIT")
    print("=" * 65)
    print(f"Development images: {len(dev)} | Changed-condition rows: {len(result)} | Valid: {result.valid.sum()}")
    print("Four original translation conditions are unchanged and are not regenerated here.")
    print(s.to_string())
    print("Geometry-only; NO model training/predictions; original Step 14 results preserved.")
    print(f"Saved: {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc alternative coherent boundary masks")
    parser.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data/raw/BUSBRA.zip")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/image_splits.csv")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--all-development", action="store_true")
    args = parser.parse_args()
    if args.all_development and args.limit != 12:
        parser.error("Do not combine --all-development with --limit")
    limit = None if args.all_development else args.limit
    dest = PROJECT_ROOT / "results/robustness" / (
        "coherent_geometry_development.csv" if limit is None else f"coherent_geometry_smoke_{limit}.csv"
    )
    audit(args.zip, args.manifest, dest, limit=limit)


if __name__ == "__main__":
    main()
