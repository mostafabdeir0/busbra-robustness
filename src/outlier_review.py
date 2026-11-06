"""Review the most extreme BUS-BRA development feature values, without modifying data.

Copy to src/outlier_review.py and run from repository root:
    .\\.venv\\Scripts\\python.exe -m src.outlier_review

Reads only DEVELOPMENT cases from the saved feature table. Recomputes the 23
features from source images/masks, checks agreement with cached features,
and creates a contact sheet of the highest-ranked unusual observations.
"""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .features import FEATURE_COLUMNS, preprocess_pair, extract_features, read_pair

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / 'data' / 'raw' / 'BUSBRA.zip'
FEATURES_PATH = ROOT / 'data' / 'processed' / 'development_features.csv'
FLAGS_PATH = ROOT / 'results' / 'feature_audit' / 'potential_outliers.csv'
OUT_DIR = ROOT / 'results' / 'feature_audit'
CONTACT_SHEET = OUT_DIR / 'outlier_review.png'
REVIEW_CSV = OUT_DIR / 'outlier_review.csv'
TOP_N = 12
COLUMNS = 3
CELL_W, CELL_H = 370, 322
DISPLAY_W, DISPLAY_H = 345, 241


def render_tile(image: np.ndarray, mask: np.ndarray, info: dict) -> Image.Image:
    im = Image.fromarray(image.astype(np.uint8), mode='L')
    ma = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
    view = ImageOps.contain(im, (DISPLAY_W, DISPLAY_H), method=Image.Resampling.BILINEAR)
    ma = ma.resize(view.size, Image.Resampling.NEAREST)
    inside = np.asarray(ma) > 0
    gray = np.asarray(view)
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    rgb[inside] = 0.76 * rgb[inside] + 0.24 * np.array([36, 222, 85], dtype=np.float32)
    border = inside.copy()
    border[1:-1, 1:-1] &= ~(
        inside[:-2, 1:-1] & inside[2:, 1:-1]
        & inside[1:-1, :-2] & inside[1:-1, 2:]
    )
    rgb[border] = np.array([245, 55, 48], dtype=np.float32)
    preview = Image.fromarray(rgb.clip(0, 255).astype(np.uint8), 'RGB')
    tile = Image.new('RGB', (CELL_W, CELL_H), '#FFFFFF')
    tile.paste(preview, ((CELL_W - view.width) // 2, 4))
    d = ImageDraw.Draw(tile)
    y = DISPLAY_H + 12
    d.text((8, y), f"{info['ID']}  |  {info['Pathology']}", fill='#1A1A1A')
    d.text((8, y + 18), f"Top flag: {info['top_feature']}  (z={info['max_robust_z']:.1f})", fill='#333333')
    d.text((8, y + 36), f"ASM d3: {info['asm_d3']:.3f}  |  Entropy: {info['int_entropy']:.2f}", fill='#333333')
    d.text((8, y + 54), f"Dominant gray bin: {info['top_bin_fraction']:.0%}  |  Mask area: {info['mask_area_pixels']} px", fill='#555555')
    return tile


def main() -> None:
    for p in (ZIP_PATH, FEATURES_PATH, FLAGS_PATH):
        if not p.is_file():
            raise FileNotFoundError(f'Missing expected file: {p}')
    feats = pd.read_csv(FEATURES_PATH, dtype={'ID': str})
    flags = pd.read_csv(FLAGS_PATH, dtype={'ID': str})
    if not feats['split'].eq('development').all():
        raise ValueError('Feature table contains nondevelopment images')
    if feats.ID.duplicated().any():
        raise ValueError('Duplicate IDs in feature table')
    needed = {'ID', 'feature', 'robust_z'}
    if not needed.issubset(flags.columns):
        raise ValueError(f'Outlier report missing columns: {needed - set(flags.columns)}')
    # Cover the largest outliers AND a variety of feature families, rather than
    # reviewing only ASM examples because they dominate the highest z-scores.
    ranked = flags.sort_values('robust_z', ascending=False).drop_duplicates('ID')
    selected_ids = list(ranked.head(4)['ID'])
    for _, candidate in flags.sort_values('robust_z', ascending=False).drop_duplicates('feature').iterrows():
        identifier = str(candidate['ID'])
        if identifier not in selected_ids:
            selected_ids.append(identifier)
        if len(selected_ids) == TOP_N:
            break
    for identifier in ranked['ID']:
        if len(selected_ids) == TOP_N:
            break
        if identifier not in selected_ids:
            selected_ids.append(identifier)
    chosen = flags.sort_values('robust_z', ascending=False).drop_duplicates('ID').set_index('ID').loc[selected_ids].reset_index()
    joined = chosen.merge(feats, on='ID', how='left', validate='one_to_one', suffixes=('', '_cached'))
    if joined.Pathology.isna().any():
        raise ValueError('Flag contains image ID missing from locked development data')

    rows: list[dict] = []
    tiles: list[Image.Image] = []
    with ZipFile(ZIP_PATH) as zf:
        for _, r in joined.iterrows():
            image, lesion = read_pair(zf, str(r['ID']))
            image, lesion = preprocess_pair(image, lesion)
            recalculated = extract_features(image, lesion)
            stored = np.array([float(r[name]) for name in FEATURE_COLUMNS])
            fresh = np.array([recalculated[name] for name in FEATURE_COLUMNS])
            delta = float(np.max(np.abs(stored - fresh)))
            if not np.allclose(stored, fresh, rtol=1e-6, atol=1e-8):
                raise AssertionError(f"Cached features do not reproduce for {r['ID']}: max abs error {delta}")
            gray32 = np.minimum(image[lesion].astype(np.uint16) * 32 // 256, 31)
            bins = np.bincount(gray32, minlength=32)
            info = {
                'ID': str(r['ID']),
                'Case': int(r['Case']),
                'Pathology': r['Pathology'],
                'top_feature': r['feature'],
                'max_robust_z': float(r['robust_z']),
                'asm_d3': float(recalculated['glcm_asm_d3']),
                'int_entropy': float(recalculated['int_entropy']),
                'top_bin_fraction': float(bins.max() / bins.sum()),
                'mask_area_pixels': int(lesion.sum()),
                'max_recomputed_feature_difference': delta,
            }
            rows.append(info)
            tiles.append(render_tile(image, lesion, info))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REVIEW_CSV, index=False, float_format='%.9g')
    nrows = math.ceil(len(tiles) / COLUMNS)
    canvas = Image.new('RGB', (COLUMNS * CELL_W, 42 + nrows * CELL_H), '#F1F3F5')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 13), 'BUS-BRA: extreme-value review (green mask, red boundary; DEV only)', fill='#111111')
    for j, tile in enumerate(tiles):
        canvas.paste(tile, ((j % COLUMNS) * CELL_W, 42 + (j // COLUMNS) * CELL_H))
    canvas.save(CONTACT_SHEET)

    print('\nBUS-BRA OUTLIER VISUAL REVIEW')
    print('=' * 44)
    print(f'Reviewed unique development images: {len(rows)}')
    print('Cached feature recomputation: MATCHED for all reviewed images')
    print('Original ZIP and extracted feature table: UNCHANGED')
    print('Highest-flagged examples:')
    print(pd.DataFrame(rows)[['ID', 'top_feature', 'max_robust_z', 'asm_d3', 'int_entropy', 'top_bin_fraction']]
          .to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'Contact sheet: {CONTACT_SHEET}')
    print(f'Review CSV:    {REVIEW_CSV}')
    print('Check each mask alignment and lesion appearance manually before closing audit.')


if __name__ == '__main__':
    main()
