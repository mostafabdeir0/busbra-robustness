"""Generate a reproducible visual review sheet of BUS-BRA images and lesion masks.

Save as src/visual_audit.py and run from the project root:
    .\\.venv\\Scripts\\python.exe -m src.visual_audit

Reads data/raw/BUSBRA.zip directly and writes a contact sheet to
results/figures/busbra_visual_audit.png. Does not modify the source data.
"""
from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / 'data' / 'raw' / 'BUSBRA.zip'
OUTPUT = ROOT / 'results' / 'figures' / 'busbra_visual_audit.png'

THUMB_W, THUMB_H = 288, 210
CELL_W, CELL_H = 310, 282
COLUMNS = 4


def choose_examples(metadata: pd.DataFrame) -> list[tuple[pd.Series, str]]:
    """Choose one image per case: 6 benign, 6 malignant, 4 metadata anomalies."""
    chosen: list[tuple[pd.Series, str]] = []
    seen_cases: set[str] = set()
    ordered = metadata.sort_values(['Case', 'ID'])
    for category in ('benign', 'malignant'):
        subset = ordered[ordered['Pathology'].astype(str).str.lower() == category]
        one_per_case = subset.drop_duplicates('Case')
        indices = np.linspace(0, len(one_per_case) - 1, 6, dtype=int)
        for row_idx in indices:
            row = one_per_case.iloc[int(row_idx)]
            case = str(row['Case'])
            if case not in seen_cases:
                chosen.append((row, 'routine sample'))
                seen_cases.add(case)
    with ZipFile(ZIP_PATH) as zf:
        for _, row in ordered.iterrows():
            image_path = f"BUSBRA/Images/{row['ID']}.png"
            with Image.open(io.BytesIO(zf.read(image_path))) as img:
                actual_w, actual_h = img.size
            if (actual_w, actual_h) != (int(row['Width']), int(row['Height'])):
                chosen.append((row, 'CSV dimension discrepancy'))
    return chosen


def make_tile(zf: ZipFile, row: pd.Series, reason: str) -> Image.Image:
    identifier = str(row['ID'])
    image_path = f'BUSBRA/Images/{identifier}.png'
    mask_path = f"BUSBRA/Masks/mask_{identifier.removeprefix('bus_')}.png"
    with Image.open(io.BytesIO(zf.read(image_path))) as f:
        img = f.convert('L')
    with Image.open(io.BytesIO(zf.read(mask_path))) as f:
        mask = f.convert('L')
    if img.size != mask.size:
        raise ValueError(f'Image/mask size mismatch for {identifier}')

    image_view = ImageOps.contain(img, (THUMB_W, THUMB_H), Image.Resampling.BILINEAR)
    mask_view = mask.resize(image_view.size, Image.Resampling.NEAREST)
    inside = np.asarray(mask_view) > 0
    a = np.asarray(image_view).copy()
    rgb = np.stack([a, a, a], axis=-1).astype(np.float32)
    # Transparent green fill inside the lesion; display-only, no changes to input data.
    rgb[inside] = 0.76 * rgb[inside] + 0.24 * np.array([40, 225, 80], dtype=np.float32)
    # Thin red outline of the lesion boundary.
    core = inside.copy()
    core[1:-1, 1:-1] = (
        inside[1:-1, 1:-1] & inside[:-2, 1:-1] & inside[2:, 1:-1]
        & inside[1:-1, :-2] & inside[1:-1, 2:]
    )
    border = inside & ~core
    rgb[border] = np.array([255, 55, 50], dtype=np.float32)
    overlay = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), 'RGB')

    tile = Image.new('RGB', (CELL_W, CELL_H), 'white')
    dx = (CELL_W - overlay.width) // 2
    tile.paste(overlay, (dx, 6))
    draw = ImageDraw.Draw(tile)
    y = THUMB_H + 12
    draw.text((10, y), f"{identifier} | Case {row['Case']} | {row['Pathology']}", fill='black')
    draw.text((10, y + 17), f"Mask area: {inside.mean():.1%}", fill='black')
    if reason == 'CSV dimension discrepancy':
        draw.text((10, y + 34), 'Metadata dimension discrepancy', fill=(155, 55, 15))
    else:
        draw.text((10, y + 34), 'Representative patient sample', fill=(80, 80, 80))
    return tile


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f'Missing archive: {ZIP_PATH}')
    with ZipFile(ZIP_PATH) as zf:
        metadata = pd.read_csv(io.BytesIO(zf.read('BUSBRA/bus_data.csv')))
    selected = choose_examples(metadata)
    rows = (len(selected) + COLUMNS - 1) // COLUMNS
    sheet = Image.new('RGB', (CELL_W * COLUMNS, CELL_H * rows + 44), '#f0f1f3')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 14), 'BUS-BRA visual audit: green = mask interior; red = mask boundary', fill='black')
    with ZipFile(ZIP_PATH) as zf:
        for i, (row, reason) in enumerate(selected):
            tile = make_tile(zf, row, reason)
            sheet.paste(tile, ((i % COLUMNS) * CELL_W, 44 + (i // COLUMNS) * CELL_H))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)
    print(f'Generated visual audit for {len(selected)} image-mask pairs:')
    print(f'  {OUTPUT}')
    print('Review every tile: boundary placement, visible lesion, mask validity, possible quality issues.')
    print('Do not change labels or masks based on visual impression alone.')


if __name__ == '__main__':
    main()
