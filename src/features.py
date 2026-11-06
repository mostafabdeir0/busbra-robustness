"""Reproducible image + lesion-mask features for BUS-BRA.

Install as src/features.py. From the project root, use:
    .\\.venv\\Scripts\\python.exe -m src.features --limit 12

This smoke test extracts 23 handcrafted features from DEVELOPMENT images only.
The test split is deliberately excluded; no scaling or model fitting happens here.

Preprocessing: grayscale image, mask nearest-neighbor; both resized with
aspect ratio preserved to longest side = 256 pixels. Masked GLCMs count ONLY
pixel pairs with BOTH pixels inside the lesion. No blacked-out background.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import perimeter_crofton, regionprops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_SIDE = 256
GLCM_LEVELS = 32
GLCM_DISTANCES = (1, 3)
GLCM_DIRECTIONS = ((0, 1), (1, 1), (1, 0), (-1, 1))
SHAPE = (
    "shape_area_fraction", "shape_axis_ratio", "shape_eccentricity",
    "shape_solidity", "shape_extent", "shape_circularity",
)
INTENSITY = (
    "int_mean", "int_std", "int_median", "int_p10", "int_p90",
    "int_skewness", "int_entropy",
)
TEXTURE_PROPERTIES = ("contrast", "dissimilarity", "homogeneity", "asm", "correlation")
TEXTURE = tuple(f"glcm_{p}_d{d}" for d in GLCM_DISTANCES for p in TEXTURE_PROPERTIES)
FEATURE_COLUMNS = SHAPE + INTENSITY + TEXTURE
assert len(FEATURE_COLUMNS) == 23


def preprocess_pair(image: np.ndarray, mask: np.ndarray, max_side: int = MAX_SIDE):
    """Convert arrays to uint8 grayscale and bool mask; resize together.

    Scaling factor is computed from the ACTUAL image dimensions (not CSV).
    """
    if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
        raise ValueError("Expected same-shaped two-dimensional image and mask")
    if max_side < 32:
        raise ValueError("max_side must be >= 32")
    if not np.any(mask):
        raise ValueError("Empty lesion mask")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("Grayscale image must be numeric")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf")
    arr = np.clip(image, 0, 255).astype(np.uint8)
    binary = (mask != 0).astype(np.uint8) * 255
    height, width = arr.shape
    scale = max_side / max(height, width)
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    img_out = np.asarray(Image.fromarray(arr).resize(output_size, Image.Resampling.BILINEAR))
    mask_out = np.asarray(Image.fromarray(binary).resize(output_size, Image.Resampling.NEAREST)) > 0
    if not mask_out.any():
        raise ValueError("Mask became empty after resize")
    return img_out, mask_out


def _glcm_properties(gray: np.ndarray, lesion: np.ndarray) -> dict[str, float]:
    """Calculate masked, symmetric, normalized GLCMs at two distances.

    For each direction/distance, only lesion-interior pixel pairs count.
    Each direction contributes equally to the final per-distance means.
    """
    quantized = np.minimum(gray.astype(np.uint16) * GLCM_LEVELS // 256, GLCM_LEVELS - 1)
    h, w = lesion.shape
    grid = np.arange(GLCM_LEVELS, dtype=np.float64)
    ii = grid[:, None]
    jj = grid[None, :]
    dif = ii - jj
    results: dict[str, float] = {}

    for distance in GLCM_DISTANCES:
        directional: list[dict[str, float]] = []
        for step_y, step_x in GLCM_DIRECTIONS:
            dy, dx = distance * step_y, distance * step_x
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            if sy0 >= sy1 or sx0 >= sx1:
                continue
            target_y0, target_y1 = sy0 + dy, sy1 + dy
            target_x0, target_x1 = sx0 + dx, sx1 + dx
            inside = lesion[sy0:sy1, sx0:sx1] & lesion[target_y0:target_y1, target_x0:target_x1]
            if not inside.any():
                continue
            a = quantized[sy0:sy1, sx0:sx1][inside]
            b = quantized[target_y0:target_y1, target_x0:target_x1][inside]
            count = np.bincount(a * GLCM_LEVELS + b, minlength=GLCM_LEVELS**2)
            cm = count.reshape((GLCM_LEVELS, GLCM_LEVELS)).astype(np.float64)
            cm += cm.T
            cm /= cm.sum()
            p_i, p_j = cm.sum(axis=1), cm.sum(axis=0)
            mu_i, mu_j = np.dot(grid, p_i), np.dot(grid, p_j)
            var_i = np.dot((grid - mu_i)**2, p_i)
            var_j = np.dot((grid - mu_j)**2, p_j)
            if var_i < 1e-12 or var_j < 1e-12:
                corr = 1.0
            else:
                corr = float(np.sum(cm * (ii - mu_i) * (jj - mu_j)) / math.sqrt(var_i * var_j))
            directional.append({
                "contrast": float(np.sum((dif**2) * cm)),
                "dissimilarity": float(np.sum(np.abs(dif) * cm)),
                "homogeneity": float(np.sum(cm / (1.0 + dif**2))),
                "asm": float(np.sum(cm**2)),
                "correlation": corr,
            })
        if not directional:
            raise ValueError(f"Mask too small for masked GLCM at distance {distance}")
        for prop in TEXTURE_PROPERTIES:
            results[f"glcm_{prop}_d{distance}"] = float(np.mean([d[prop] for d in directional]))
    return results


def extract_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Extract 23 features from a PREPROCESSED grayscale image and binary mask.

    For perturbation experiments, change *mask* only and call this function
    again on the SAME preprocessed image. Do NOT resize perturbed masks anew.
    """
    if image.ndim != 2 or mask.ndim != 2 or image.shape != mask.shape:
        raise ValueError("image and mask must be 2D arrays of matching size")
    binary = mask.astype(bool)
    if binary.sum() < 15:
        raise ValueError("Lesion mask contains fewer than 15 pixels")
    if not np.isfinite(image).all():
        raise ValueError("Image contains nonfinite values")
    if np.min(image) < 0 or np.max(image) > 255:
        raise ValueError("Expected grayscale intensities from 0 to 255")

    props = regionprops(binary.astype(np.uint8))[0]
    area = float(binary.sum())
    perim = float(perimeter_crofton(binary, directions=4))
    minor = float(props.axis_minor_length)
    major = float(props.axis_major_length)
    if perim <= 0 or minor <= 0:
        raise ValueError("Degenerate lesion shape")

    pixels = np.asarray(image[binary], dtype=np.float64)
    center = float(pixels.mean())
    sd = float(pixels.std(ddof=0))
    hist = np.histogram(pixels, bins=32, range=(0, 256))[0].astype(np.float64)
    probabilities = hist[hist > 0] / len(pixels)
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    skewness = 0.0 if sd < 1e-12 else float(np.mean(((pixels - center) / sd) ** 3))
    feats = {
        "shape_area_fraction": float(area / binary.size),
        "shape_axis_ratio": float(major / minor),
        "shape_eccentricity": float(props.eccentricity),
        "shape_solidity": float(props.solidity),
        "shape_extent": float(props.extent),
        "shape_circularity": float(4.0 * math.pi * area / (perim**2)),
        "int_mean": center / 255.0,
        "int_std": sd / 255.0,
        "int_median": float(np.median(pixels)) / 255.0,
        "int_p10": float(np.percentile(pixels, 10)) / 255.0,
        "int_p90": float(np.percentile(pixels, 90)) / 255.0,
        "int_skewness": skewness,
        "int_entropy": entropy,
    }
    feats.update(_glcm_properties(np.asarray(image, dtype=np.uint8), binary))
    assert tuple(feats) == FEATURE_COLUMNS, "Internal feature schema changed unexpectedly"
    if not np.isfinite(list(feats.values())).all():
        raise ValueError("Extracted nonfinite feature value")
    return feats


def read_pair(zf: ZipFile, image_id: str):
    """Read one BUS-BRA image and matching binary lesion mask from ZIP."""
    image_path = f"BUSBRA/Images/{image_id}.png"
    mask_path = f"BUSBRA/Masks/mask_{image_id.removeprefix('bus_')}.png"
    with Image.open(io.BytesIO(zf.read(image_path))) as im:
        image = np.asarray(im.convert("L"))
    with Image.open(io.BytesIO(zf.read(mask_path))) as m:
        mask = np.asarray(m.convert("L")) > 0
    return image, mask


def extract_development(zip_path: Path, manifest_path: Path, output_path: Path,
                        limit: int | None = 12) -> pd.DataFrame:
    """Extract features only from images preassigned to development.

    `limit=None` processes all 1,514 development images. A small fixed limit
    is useful for initial correctness checks without a long run.
    """
    manifest = pd.read_csv(manifest_path, dtype={"ID": str})
    expected = {"ID", "Case", "Pathology", "split", "dev_fold"}
    if not expected.issubset(manifest.columns):
        raise ValueError(f"Missing manifest fields: {expected - set(manifest.columns)}")
    if manifest.ID.duplicated().any():
        raise ValueError("Duplicated image ID in split manifest")
    subset = manifest.loc[manifest.split.eq("development")].sort_values("ID")
    if subset.empty:
        raise ValueError("No development rows in the split manifest")
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        subset = subset.head(limit)
    all_rows: list[dict] = []
    with ZipFile(zip_path) as zf:
        for n, rec in enumerate(subset.itertuples(index=False), start=1):
            image, mask = read_pair(zf, rec.ID)
            image, mask = preprocess_pair(image, mask)
            feat = extract_features(image, mask)
            all_rows.append({
                "ID": str(rec.ID), "Case": rec.Case, "Pathology": rec.Pathology,
                "split": rec.split, "dev_fold": int(rec.dev_fold), **feat,
            })
            if n % 100 == 0:
                print(f"Extracted {n}/{len(subset)} development images...")
    result = pd.DataFrame(all_rows)
    if len(result) != len(subset) or not result["split"].eq("development").all():
        raise RuntimeError("Output violates development-only constraint")
    if result[list(FEATURE_COLUMNS)].isna().any().any():
        raise RuntimeError("Feature output contains missing values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.12g")
    print(f"Extracted {len(result)} development images")
    print(f"Computed {len(FEATURE_COLUMNS)} features per image")
    print(f"Saved: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract BUS-BRA features (development patients only)")
    parser.add_argument("--zip", type=Path, default=PROJECT_ROOT / "data/raw/BUSBRA.zip")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "splits/image_splits.csv")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=12, help="Sample size for smoke test; default=12")
    parser.add_argument("--all-development", action="store_true", help="Process all development images")
    args = parser.parse_args()
    if args.all_development and args.limit != 12:
        parser.error("Do not combine --all-development with --limit")
    limit = None if args.all_development else args.limit
    filename = "development_features.csv" if limit is None else f"development_features_smoke_{limit}.csv"
    output = args.output or PROJECT_ROOT / "data/processed" / filename
    if not args.zip.is_file():
        raise FileNotFoundError(args.zip)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    extract_development(args.zip, args.manifest, output, limit=limit)


if __name__ == "__main__":
    main()
