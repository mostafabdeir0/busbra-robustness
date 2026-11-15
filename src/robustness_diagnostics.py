"""Post-hoc development-only diagnostic for the surprising Step 14 severity pattern.

Run from the project root:
    python -m src.robustness_diagnostics

No retraining and NO test images. Reads development feature tables and saved
patient predictions. Saves feature-shift, patient flip-pattern and contour
inspection artifacts; it does not modify the perturbation protocol or models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, PROJECT_ROOT, preprocess_pair, read_pair, extract_features
from .perturbations import make_variants
from .perturbed_features import VARIANT_NAMES

COMPARE = (("erosion_mild", "erosion_moderate"), ("dilation_mild", "dilation_moderate"))
REVIEW = ("clean", "erosion_mild", "erosion_moderate", "dilation_mild", "dilation_moderate")


def parse_flags(column: pd.Series) -> pd.Series:
    """Strict CSV bool parser: bool('False') incorrectly evaluates to True."""
    values = column.astype('string').str.strip().str.lower()
    if not values.isin(['true', 'false']).all():
        raise ValueError('Expected only True/False availability values')
    return values.eq('true').astype(bool)


def check_tables(clean: pd.DataFrame, perturbed: pd.DataFrame) -> pd.DataFrame:
    """Join valid perturbed rows to original, development-only features."""
    fields = {'ID', 'Case', 'split', 'dev_fold'} | set(FEATURE_COLUMNS)
    if not fields.issubset(clean.columns) or not (fields | {'condition', 'valid'}).issubset(perturbed.columns):
        raise ValueError('Missing required columns in feature table')
    if clean.ID.duplicated().any() or perturbed.duplicated(['ID', 'condition']).any():
        raise ValueError('Duplicate image or image-condition key')
    if not clean.split.eq('development').all() or not perturbed.split.eq('development').all():
        raise ValueError('This analysis must use development data only')
    if not set(perturbed.ID).issubset(set(clean.ID)):
        raise ValueError('Perturbed image not present in clean development table')
    valid = parse_flags(perturbed.valid)
    part = perturbed.loc[valid, ['ID', 'Case', 'dev_fold', 'condition', *FEATURE_COLUMNS]].copy()
    orig = clean[['ID', 'Case', 'dev_fold', *FEATURE_COLUMNS]]
    merged = part.merge(orig, on='ID', how='inner', suffixes=('', '_clean'), validate='many_to_one')
    if len(merged) != len(part) or not (merged['Case'] == merged['Case_clean']).all() or not (merged['dev_fold'] == merged['dev_fold_clean']).all():
        raise ValueError('Missing images or patient/fold metadata mismatch')
    return merged


def feature_drift(clean: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Descriptive drift standardized by the ORIGINAL development feature std.

    This is NOT preprocessing for any classifier, nor a fitted model.
    """
    std = clean[list(FEATURE_COLUMNS)].std(ddof=0).replace(0, np.nan)
    if std.isna().any():
        raise ValueError('Constant feature prevents drift standardization')
    records = []
    for condition, rows in paired.groupby('condition', sort=False):
        for feature in FEATURE_COLUMNS:
            delta = rows[feature].to_numpy(float) - rows[feature+'_clean'].to_numpy(float)
            if not np.isfinite(delta).all():
                raise ValueError('Non-finite feature drift')
            records.append({
                'condition': condition, 'feature': feature, 'valid_images': len(rows),
                'median_signed_delta': float(np.median(delta)),
                'median_abs_delta': float(np.median(np.abs(delta))),
                'median_abs_delta_dev_sd': float(np.median(np.abs(delta)) / std[feature]),
                'mean_abs_delta_dev_sd': float(np.mean(np.abs(delta)) / std[feature]),
                'p90_abs_delta_dev_sd': float(np.percentile(np.abs(delta), 90) / std[feature]),
            })
    return pd.DataFrame(records)


def flip_discordance(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fields = {'model', 'condition', 'Case', 'actual', 'eligible', 'clean_predicted', 'perturbed_predicted'}
    if not fields.issubset(preds.columns):
        raise ValueError('Missing model prediction columns')
    if preds.duplicated(['model', 'condition', 'Case']).any():
        raise ValueError('Duplicate model-condition-patient record')
    preds = preds.copy()
    preds['eligible'] = parse_flags(preds.eligible)
    summaries, patient_rows = [], []
    for model, data in preds.groupby('model', sort=True):
        for mild, moderate in COMPARE:
            a = data.loc[data.condition.eq(mild)].copy()
            b = data.loc[data.condition.eq(moderate)].copy()
            p = a.merge(b, on='Case', how='inner', validate='one_to_one', suffixes=('_mild', '_moderate'))
            if len(p) != len(a) or len(p) != len(b):
                raise ValueError('Mild/moderate patient coverage differs')
            if not (p.actual_mild.eq(p.actual_moderate) & p.clean_predicted_mild.eq(p.clean_predicted_moderate)).all():
                raise ValueError('Mild/moderate rows have inconsistent clean reference')
            p = p.loc[p.eligible_mild & p.eligible_moderate].copy()
            if not len(p):
                raise ValueError('No patients eligible for both severities')
            mild_flip = p.perturbed_predicted_mild.ne(p.clean_predicted_mild)
            moderate_flip = p.perturbed_predicted_moderate.ne(p.clean_predicted_moderate)
            mild_only = mild_flip & ~moderate_flip
            moderate_only = moderate_flip & ~mild_flip
            summaries.append({
                'model': model, 'family': mild.removesuffix('_mild'),
                'eligible_both': int(len(p)),
                'mild_flip_count': int(mild_flip.sum()),
                'moderate_flip_count': int(moderate_flip.sum()),
                'mild_only_flips': int(mild_only.sum()),
                'moderate_only_flips': int(moderate_only.sum()),
                'both_flip': int((mild_flip & moderate_flip).sum()),
                'neither_flip': int((~mild_flip & ~moderate_flip).sum()),
                'mild_only_malignant': int((mild_only & p.actual_mild.eq(1)).sum()),
            })
            for ix, row in p.loc[mild_only | moderate_only].iterrows():
                patient_rows.append({
                    'model': model, 'family': mild.removesuffix('_mild'),
                    'Case': int(row.Case), 'actual': int(row.actual_mild),
                    'mild_only_flip': bool(mild_only.loc[ix]),
                    'moderate_only_flip': bool(moderate_only.loc[ix]),
                    'clean_predicted': int(row.clean_predicted_mild),
                    'mild_predicted': int(row.perturbed_predicted_mild),
                    'moderate_predicted': int(row.perturbed_predicted_moderate),
                })
    return pd.DataFrame(summaries), pd.DataFrame(patient_rows)


def create_gallery(zip_path: Path, paired: pd.DataFrame, output: Path) -> list[str]:
    """Compare identical grayscale images with five original/changed masks."""
    small = paired.loc[paired.condition.isin(['erosion_mild', 'erosion_moderate', 'dilation_mild', 'dilation_moderate']),
                       ['ID', 'condition', 'shape_circularity', 'shape_circularity_clean']].copy()
    small['shift'] = (small.shape_circularity - small.shape_circularity_clean).abs()
    tab = small.pivot_table(index='ID', columns='condition', values='shift', aggfunc='first')
    required = ['erosion_mild', 'erosion_moderate', 'dilation_mild', 'dilation_moderate']
    tab = tab.dropna(subset=required)
    # Larger positive scores mean mild changes alter circularity more than moderate changes.
    score = ((tab.erosion_mild - tab.erosion_moderate).clip(lower=0)
             + (tab.dilation_mild - tab.dilation_moderate).clip(lower=0))
    chosen = list(score.sort_values(ascending=False).head(4).index)
    if not chosen:
        raise ValueError('No complete erosion/dilation examples for contour gallery')
    fig, axes = plt.subplots(len(chosen), len(REVIEW), figsize=(14, 2.8 * len(chosen)), squeeze=False)
    with ZipFile(zip_path) as archive:
        for i, image_id in enumerate(chosen):
            image, mask = preprocess_pair(*read_pair(archive, image_id))
            variants = make_variants(mask)
            for j, name in enumerate(REVIEW):
                ax = axes[i, j]
                selected = mask if name == 'clean' else variants[name].mask
                ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                ax.contour(selected.astype(float), levels=[0.5], colors=['#f97316'], linewidths=1.0)
                if j:
                    ax.contour(mask.astype(float), levels=[0.5], colors=['#34d399'], linewidths=.65, linestyles='dotted')
                circ = extract_features(image, selected)['shape_circularity']
                iou = '1.000' if name == 'clean' else f'{variants[name].iou:.3f}'
                ax.set_title(f'{image_id}\n{name}\nIoU={iou}, circ={circ:.3f}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Orange = evaluated mask; green dotted = original mask (development only)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165, bbox_inches='tight')
    plt.close(fig)
    return chosen


def run(project_root: Path, *, show_gallery=True, pert_path: Path | None = None,
        predictions_path: Path | None = None) -> dict:
    root = Path(project_root)
    clean_path = root/'data/processed/development_features.csv'
    pert_path = pert_path or root/'data/processed/development_perturbed_features.csv'
    predictions_path = predictions_path or root/'results/robustness/robustness_paired_patient_predictions.csv'
    output_dir = root/'results/robustness'
    for fp in (clean_path, pert_path, predictions_path):
        if not fp.is_file():
            raise FileNotFoundError(fp)
    clean = pd.read_csv(clean_path, dtype={'ID': str})
    pert = pd.read_csv(pert_path, dtype={'ID': str}, low_memory=False)
    preds = pd.read_csv(predictions_path)
    if (len(pert) != len(clean) * len(VARIANT_NAMES)
            or pert.ID.nunique() != clean.ID.nunique()
            or set(pert.condition) != set(VARIANT_NAMES)
            or not pert.groupby('ID').condition.nunique().eq(len(VARIANT_NAMES)).all()):
        raise ValueError('Expected all eight perturbation records for every development image')
    paired = check_tables(clean, pert)
    drift = feature_drift(clean, paired)
    discord, patients = flip_discordance(preds)
    output_dir.mkdir(parents=True, exist_ok=True)
    drift.to_csv(output_dir/'diagnostics_feature_shift_by_condition.csv', index=False, float_format='%.10g')
    discord.to_csv(output_dir/'diagnostics_mild_moderate_flip_patterns.csv', index=False)
    patients.to_csv(output_dir/'diagnostics_severity_discordant_patients.csv', index=False)
    gallery_ids=[]
    if show_gallery:
        gallery_ids=create_gallery(root/'data/raw/BUSBRA.zip', paired, output_dir/'diagnostics_boundary_gallery.png')
    summary={
        'purpose':'Post-hoc diagnosis of non-monotonic mild/moderate mask robustness results',
        'development_images':int(clean.ID.nunique()),
        'perturbation_rows':int(len(pert)),
        'valid_perturbation_rows':int(len(paired)),
        'test_images_used':0, 'model_retraining':False,
        'gallery_image_ids':gallery_ids,
        'comparison':[x for x in discord.to_dict(orient='records')],
    }
    (output_dir/'diagnostics_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('BUS-BRA STEP 14.1 — SEVERITY / FEATURE DRIFT AUDIT')
    print('='*56)
    print(f"Development: {len(clean)} images | Valid perturbation rows: {len(paired)} | Test patients used: 0")
    print('\nCIRCULARITY feature shift (original feature units):')
    circ=drift.loc[drift.feature.eq('shape_circularity') & drift.condition.isin(
        ['erosion_mild','erosion_moderate','dilation_mild','dilation_moderate'])]
    print(circ[['condition','median_signed_delta','median_abs_delta','median_abs_delta_dev_sd']].to_string(index=False,float_format=lambda n:f'{n:.4f}'))
    print('\nPREDICTION FLIP OVERLAP (same eligible patients for both severities):')
    print(discord[['model','family','eligible_both','mild_flip_count','moderate_flip_count','mild_only_flips','moderate_only_flips','mild_only_malignant']].to_string(index=False))
    print('\nSaved diagnostic CSVs and summary in:',output_dir)
    if show_gallery:
        print('Mask comparison gallery:',output_dir/'diagnostics_boundary_gallery.png')
    print('Important: descriptive post-hoc diagnostic; do not change experimental protocol based on this alone.')
    return summary


def main():
    p=argparse.ArgumentParser(description='Audit severity paradox in development-only robustness results')
    p.add_argument('--project-root',type=Path,default=PROJECT_ROOT)
    p.add_argument('--no-gallery',action='store_true')
    args=p.parse_args()
    run(args.project_root,show_gallery=not args.no_gallery)


if __name__=='__main__':
    main()
